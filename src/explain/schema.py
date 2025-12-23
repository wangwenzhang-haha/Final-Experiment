"""解释输出用到的数据类定义。

每个字段都可以直接序列化成 JSON，用于端到端示例和后续评测。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class Recommendation:
    """推荐结果（物品 + 分数）。"""
    item_id: str
    score: float


@dataclass
class Evidence:
    """可追溯证据，包含交互、metadata 和 KG 路径。"""
    interactions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    kg_paths: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Explanation:
    """解释文本及推理步骤。"""
    short: str
    detailed: str
    reasoning_steps: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineOutput:
    """总输出：用户 + item + 推荐列表 + 证据 + 解释。"""
    user_id: str
    item_id: str
    user_profile: Dict[str, Any]
    recommendations: List[Recommendation]
    evidence: Evidence
    explanation: Explanation

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["recommendations"] = [asdict(r) for r in self.recommendations]
        data["evidence"] = self.evidence.to_dict()
        data["explanation"] = self.explanation.to_dict()
        return data


EXPLANATION_JSON_GUIDE = {
    "user_id": "string",
    "item_id": "string",
    "recommendations": [{"item_id": "string", "score": 0.0}],
    "evidence": {
        "interactions": [{"user_id": "string", "item_id": "string"}],
        "metadata": [{"entity": "string", "attr": "string", "value": "string"}],
        "kg_paths": [
            {"nodes": ["..."], "edges": ["..."], "confidence": 0.0}
        ],
    },
    "explanation": {
        "short": "string",
        "detailed": "string",
        "reasoning_steps": ["..."]
    },
}

__all__ = [
    "Recommendation",
    "Evidence",
    "Explanation",
    "PipelineOutput",
    "EXPLANATION_JSON_GUIDE",
]
