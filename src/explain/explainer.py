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
            f"用户 {user_id} 曾经交互过：{', '.join(interactions)}" if interactions else "交互记录较少。",
            f"候选物品的相关属性：{', '.join(meta_texts)}" if meta_texts else "缺少可用的元数据相似度。",
            f"参考的图路径：{', '.join(path_texts)}" if path_texts else "未找到可用的图路径。",
        ]
        short = (
            f"因为你喜欢 {', '.join(interactions[:2])}，物品 {item_id} 的相关属性相似。"
            if interactions
            else f"物品 {item_id} 基于流行度被推荐。"
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
            message = response.choices[0].message
            if hasattr(message, "content"):
                text = message.content
            else:
                text = message["content"]  # type: ignore[index]
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
