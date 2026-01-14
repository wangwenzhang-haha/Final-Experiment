"""Prompt generator for explanation tasks."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import yaml


PromptTemplates = Mapping[str, str]


def _read_templates_from_path(path: Union[str, Path]) -> PromptTemplates:
    template_path = Path(path)
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    suffix = template_path.suffix.lower()
    if suffix == ".json":
        return json.loads(template_path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(template_path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported template file type: {suffix}")


def format_path(path: Sequence[str]) -> str:
    """Convert a path like [u1 -> i2 -> BrandX] into a natural language sentence."""
    if isinstance(path, str):
        return f"相关实体是 {path}。"
    if not path:
        return "没有可用的证据路径。"

    nodes = list(path)
    if len(nodes) == 1:
        return f"相关实体是 {nodes[0]}。"

    user_node = nodes[0]
    first_item = nodes[1]
    sentences = [f"你（{user_node}）浏览或互动过的物品是 {first_item}。"]

    remaining = nodes[2:]
    if not remaining:
        return "".join(sentences)

    brand_nodes = [n for n in remaining if "brand" in str(n).lower() or "品牌" in str(n)]
    other_nodes = [n for n in remaining if n not in brand_nodes]

    if brand_nodes:
        sentences.append(f"相关品牌是 {', '.join(brand_nodes)}。")
    if other_nodes:
        sentences.append(f"关联实体包括 {', '.join(other_nodes)}。")

    return "".join(sentences)


@dataclass
class PromptGenerator:
    """Generate prompts with style and target control tokens."""

    templates: Optional[PromptTemplates] = None
    template_path: Optional[Union[str, Path]] = None
    control_tokens: Sequence[str] = field(default_factory=list)
    output_format: str = "text"

    def __post_init__(self) -> None:
        if self.templates is None and self.template_path is None:
            self.templates = {
                "concise": (
                    "你是一名推荐系统助手。\n"
                    "用户画像：{user_profile}\n"
                    "候选物品：{item_name}\n"
                    "证据：{evidence}\n"
                    "请用一句话给出简洁理由。"
                ),
                "detailed": (
                    "你是一名解释型推荐系统助手。\n"
                    "用户画像：{user_profile}\n"
                    "候选物品：{item_name}\n"
                    "证据路径：\n{evidence}\n"
                    "请给出详细的推荐理由，需覆盖路径中的关键实体。"
                ),
                "knowledge": (
                    "你是一名知识型推荐系统助手。\n"
                    "用户画像：{user_profile}\n"
                    "候选物品：{item_name}\n"
                    "知识证据：\n{evidence}\n"
                    "请用知识型语气说明推荐原因，并保持事实一致。"
                ),
            }
        elif self.templates is None and self.template_path is not None:
            self.templates = _read_templates_from_path(self.template_path)

        if self.templates is None:
            raise ValueError("Templates must be provided or loaded from a file.")

    def _build_control_prefix(self, style: str, target: Optional[str]) -> str:
        tokens = [f"<STYLE={style}>"]
        if target:
            tokens.append(f"<TARGET={target}>")
        tokens.extend([f"<{token}>" for token in self.control_tokens])
        return " ".join(tokens)

    def _normalize_paths(self, paths: Iterable[Sequence[str]]) -> str:
        formatted = [f"- {format_path(path)}" for path in paths]
        return "\n".join(formatted) if formatted else "- 无"

    def generate(
        self,
        user: str,
        item: str,
        paths: Iterable[Sequence[str]],
        *,
        style: str = "concise",
        target: Optional[str] = None,
        user_profile: Optional[str] = None,
        item_name: Optional[str] = None,
        extra_slots: Optional[Dict[str, Any]] = None,
        output_format: Optional[str] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Generate a prompt with slot-filling and optional JSON output."""
        if self.templates is None:
            raise ValueError("Templates not initialized.")
        if style not in self.templates:
            raise KeyError(f"Unknown style: {style}")

        evidence = self._normalize_paths(paths)
        slots: Dict[str, Any] = {
            "user_profile": user_profile or user,
            "item_name": item_name or item,
            "evidence": evidence,
            "user": user,
            "item": item,
        }
        if extra_slots:
            slots.update(extra_slots)

        prompt_body = self.templates[style].format(**slots)
        control_prefix = self._build_control_prefix(style, target)
        prompt_text = f"{control_prefix}\n{prompt_body}".strip()

        output_kind = output_format or self.output_format
        if output_kind == "json":
            return {
                "user": user,
                "item": item,
                "style": style,
                "target": target,
                "prompt_text": prompt_text,
                "slots": slots,
            }
        if output_kind == "text":
            return prompt_text
        raise ValueError(f"Unsupported output_format: {output_kind}")


__all__ = ["PromptGenerator", "format_path"]
