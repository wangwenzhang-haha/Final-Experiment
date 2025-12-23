"""Prompt helpers for explanation generation."""
from __future__ import annotations

from typing import List

from src.explain.schema import EXPLANATION_JSON_GUIDE


def build_explanation_prompt(
    user_id: str,
    item_id: str,
    user_profile: str,
    item_desc: str,
    path_descriptions: List[str],
) -> str:
    """Assemble a concise prompt describing the user, item, and evidence paths."""
    path_text = "\n".join([f"- {p}" for p in path_descriptions]) or "- none -"
    schema_hint = str(EXPLANATION_JSON_GUIDE)
    return (
        "You are an explainable recommender. Use only the provided evidence.\n"
        "Return JSON with keys: user_id, item_id, recommendations, evidence, explanation.\n"
        "explanation must contain short, detailed, reasoning_steps (list).\n"
        "Do not invent new entities; use nodes from evidence paths or metadata.\n"
        f"Schema hint: {schema_hint}\n"
        f"User: {user_id}\n"
        f"Candidate item: {item_id}\n"
        f"User profile: {user_profile}\n"
        f"Item description: {item_desc}\n"
        f"Evidence paths:\n{path_text}\n"
        "Ensure reasoning_steps references the evidence above and stays within 2-4 hops."
    )


__all__ = ["build_explanation_prompt"]
