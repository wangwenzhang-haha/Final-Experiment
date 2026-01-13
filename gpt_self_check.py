"""LLM self-check utility for hallucination detection in recommendation explanations."""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"


def _build_prompt(explanation_text: str, allowed_entities: List[str]) -> str:
    entities_text = ", ".join([f"\"{entity}\"" for entity in allowed_entities]) or "无"
    return (
        "你是一个推荐系统专家。以下是某个推荐解释系统生成的理由：\n"
        f"【推荐解释】：\"{explanation_text}\"\n\n"
        "该用户路径中仅涉及以下实体和关系：\n"
        f"【路径包含实体】：{entities_text}\n\n"
        "请判断这个推荐解释是否使用了路径中未出现的实体或概念？"
        "请回答 JSON 格式：{\"hallucinated\": true/false, \"reason\": \"...\"}。"
    )


def _parse_response(content: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(content)
        hallucinated = bool(parsed.get("hallucinated"))
        reason = str(parsed.get("reason", ""))
        return {"hallucinated": hallucinated, "reason": reason}
    except json.JSONDecodeError:
        cleaned = content.strip()
        hallucinated = cleaned.startswith("是") or "是" in cleaned
        return {"hallucinated": hallucinated, "reason": cleaned}


def gpt_self_check(explanation_text: str, allowed_entities: List[str]) -> Dict[str, Any]:
    """Check if an explanation introduces entities outside the allowed list."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    prompt = _build_prompt(explanation_text, allowed_entities)
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": "You are a strict JSON generator."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    return _parse_response(content)


def gpt_self_check_batch(
    explanations: Sequence[str],
    allowed_entities: List[str],
    output_path: Optional[str] = None,
    output_format: str = "json",
) -> List[Dict[str, Any]]:
    """Run self-check on multiple explanations and optionally save to disk."""
    results = [gpt_self_check(explanation, allowed_entities) for explanation in explanations]
    if output_path:
        if output_format.lower() == "json":
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(results, handle, ensure_ascii=False, indent=2)
        elif output_format.lower() == "csv":
            with open(output_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["hallucinated", "reason"])
                writer.writeheader()
                writer.writerows(results)
        else:
            raise ValueError("output_format must be 'json' or 'csv'.")
    return results


__all__ = ["gpt_self_check", "gpt_self_check_batch"]
