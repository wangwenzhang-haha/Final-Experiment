"""LLM (or fallback) explanation generator."""
from __future__ import annotations

import json
import os
from typing import Any, Dict

from src.explain.schema import Explanation


class ExplanationGenerator:
    """Generate explanations via OpenAI API when available, otherwise fallback."""

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                import openai  # type: ignore

                self.client = openai.OpenAI(api_key=api_key)
            except Exception:
                self.client = None

    def _fallback(self, user_id: str, item_id: str, evidence: Dict[str, Any]) -> Explanation:
        """Deterministic explanation builder used when no API key is set."""
        interactions = [f"{e['item_id']}" for e in evidence.get("interactions", [])]
        meta_texts = [f"{m['attr']}={m['value']}" for m in evidence.get("metadata", [])]
        path_texts = [" -> ".join(p.get("nodes", [])) for p in evidence.get("kg_paths", [])]
        reasoning = [
            f"User {user_id} previously interacted with {', '.join(interactions)}" if interactions else "Limited interaction history available.",
            f"Candidate item shares attributes: {', '.join(meta_texts)}" if meta_texts else "Metadata similarity not available.",
            f"Graph paths considered: {', '.join(path_texts)}" if path_texts else "No graph paths found.",
        ]
        short = (
            f"Because you liked {', '.join(interactions[:2])}, item {item_id} matches similar attributes."
            if interactions
            else f"Item {item_id} is suggested based on popularity."
        )
        detailed = " ".join(reasoning)
        return Explanation(short=short, detailed=detailed, reasoning_steps=reasoning)

    def generate(self, user_id: str, item_id: str, prompt: str, evidence: Dict[str, Any]) -> Explanation:
        """Call the LLM using a structured prompt; fallback on errors."""
        if self.client is None:
            return self._fallback(user_id, item_id, evidence)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )
            text = response.choices[0].message["content"]  # type: ignore
            parsed = json.loads(text)
            payload = parsed.get("explanation", parsed)
            return Explanation(
                short=payload.get("short", ""),
                detailed=payload.get("detailed", ""),
                reasoning_steps=payload.get("reasoning_steps", []),
            )
        except Exception:
            return self._fallback(user_id, item_id, evidence)


__all__ = ["ExplanationGenerator"]
