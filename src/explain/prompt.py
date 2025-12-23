"""构造解释生成提示词的工具函数。"""
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
    """拼装简洁提示词，包含用户、候选物品和证据路径，约束输出为 JSON。"""
    path_text = "\n".join([f"- {p}" for p in path_descriptions]) or "- none -"
    schema_hint = str(EXPLANATION_JSON_GUIDE)
    return (
        "你是一个可解释推荐助手，只能使用给定证据。\n"
        "输出 JSON，包含 user_id、item_id、recommendations、evidence、explanation。\n"
        "explanation 字段下必须有 short、detailed、reasoning_steps（列表）。\n"
        "不要虚构实体，只能引用证据路径或元数据中的节点。\n"
        f"Schema 提示: {schema_hint}\n"
        f"用户: {user_id}\n"
        f"候选物品: {item_id}\n"
        f"用户画像: {user_profile}\n"
        f"物品描述: {item_desc}\n"
        f"证据路径:\n{path_text}\n"
        "确保 reasoning_steps 引用上述证据，并保持 2-4 跳。"
    )


__all__ = ["build_explanation_prompt"]
