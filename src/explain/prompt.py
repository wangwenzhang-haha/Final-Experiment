"""Prompt helpers for explanation generation."""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

from src.explain.schema import EXPLANATION_JSON_GUIDE


def format_path(path: Sequence[str] | str) -> str:
    """Translate a multi-hop path into a single Chinese explanation sentence."""
    if isinstance(path, str):
        cleaned = path.strip()
        cleaned = cleaned.lstrip("[").rstrip("]")
        nodes = [node.strip() for node in cleaned.split("→") if node.strip()]
    else:
        nodes = [str(node).strip() for node in path if str(node).strip()]

    if not nodes:
        return "路径为空，无法生成解释。"

    label_map = {
        "u": "用户",
        "user": "用户",
        "i": "物品",
        "item": "物品",
        "brand": "品牌",
        "category": "类别",
        "cate": "类别",
    }
    translated = [label_map.get(node, node) for node in nodes]
    path_text = " → ".join(translated)

    if len(translated) == 1:
        return f"路径为：{path_text}，表示该实体自身的属性或信息。"
    if len(translated) == 2:
        return f"路径为：{path_text}，表示{translated[0]}与{translated[1]}存在直接关联。"
    return (
        f"路径为：{path_text}，表示{translated[0]}与{translated[1]}相关联，"
        f"并进一步关联到{translated[-1]}。"
    )


class PromptGenerator:
    """Generate structured prompts with slot-based templates."""

    def __init__(self, templates: Mapping[str, str] | None = None) -> None:
        self.templates: Dict[str, str] = dict(templates) if templates else self._default_templates()

    def _default_templates(self) -> Dict[str, str]:
        return {
            "concise": (
                "你是推荐解释生成器，请根据以下信息生成简洁说明。\n"
                "用户画像：{user_profile}\n"
                "物品画像：{item_profile}\n"
                "证据：{evidence}\n"
                "要求：用1-2句话说明推荐原因，并明确关联证据。"
            ),
            "detailed": (
                "你是推荐解释生成器，请输出结构化解释。\n"
                "用户画像：{user_profile}\n"
                "物品画像：{item_profile}\n"
                "证据：{evidence}\n"
                "要求：包含推荐理由、证据引用和2-3步推理过程。"
            ),
            "knowledge": (
                "你是具备知识推理能力的推荐解释生成器。\n"
                "用户画像：{user_profile}\n"
                "物品画像：{item_profile}\n"
                "证据：{evidence}\n"
                "要求：结合知识关系进行解释，输出结构化要点。"
            ),
        }

    def generate(
        self,
        user_profile: str,
        item_profile: str,
        evidence: str,
        style: str = "concise",
    ) -> str:
        template = self.templates.get(style)
        if template is None:
            raise ValueError(f"Unknown style: {style}")
        return template.format(
            user_profile=user_profile,
            item_profile=item_profile,
            evidence=evidence,
        )


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


__all__ = ["PromptGenerator", "build_explanation_prompt", "format_path"]
