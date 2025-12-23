"""LLM（或回退规则）解释生成器。"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from src.explain.schema import Explanation


class ExplanationGenerator:
    """优先使用 OpenAI API 生成解释，缺少 Key 时退回规则逻辑。"""

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
        """无 API Key 时的确定性解释组装，确保可复现。"""
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

    def _sanitize_payload(self, payload: Dict[str, Any]) -> Explanation:
        """校验并修正 LLM 返回的字段，确保与 schema 对齐。"""

        short = str(payload.get("short", "")).strip()
        detailed = str(payload.get("detailed", "")).strip()
        reasoning_raw = payload.get("reasoning_steps", [])
        if isinstance(reasoning_raw, list):
            reasoning: List[str] = [str(s) for s in reasoning_raw if str(s).strip()]
        else:
            reasoning = [str(reasoning_raw)] if reasoning_raw else []

        if not detailed:
            detailed = short or "未能生成详细解释。"
        if not short:
            short = detailed[:80]
        if not reasoning:
            reasoning = ["模型未返回推理步骤，已填充占位说明。"]

        return Explanation(short=short, detailed=detailed, reasoning_steps=reasoning)

    def generate(self, user_id: str, item_id: str, prompt: str, evidence: Dict[str, Any]) -> Explanation:
        """使用结构化提示调用 LLM；失败则自动回退。"""
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
            if not isinstance(parsed, dict):
                raise ValueError("LLM 返回的不是 JSON 对象")
            payload = parsed.get("explanation", parsed)
            if not isinstance(payload, dict):
                raise ValueError("explanation 字段必须是对象")
            return self._sanitize_payload(payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 解释生成失败，使用回退逻辑：{exc}")
            return self._fallback(user_id, item_id, evidence)


__all__ = ["ExplanationGenerator"]
