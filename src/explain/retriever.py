"""证据检索组件：向量相似度 + 简易图路径。"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from src.pipeline.simple_graph import SimpleGraph


def _tokenize(text: str) -> List[str]:
    """简单的空格分词，并统一小写，便于稀疏向量化。"""
    return [tok.lower() for tok in text.replace("/", " ").replace(":", " ").split() if tok]


class VectorEvidenceRetriever:
    """基于物品元数据的余弦相似检索，无需额外依赖。"""

    def __init__(self, items: List[Dict[str, str]]):
        self.items = items
        self.item_vectors: Dict[str, Counter] = {}
        for item in items:
            item_id = item.get("item_id", "")
            text = self._item_text(item)
            self.item_vectors[item_id] = self._vectorize(text)

    def _vectorize(self, text: str) -> Counter:
        """将文本转为词袋计数，兼容稀疏计算。"""
        return Counter(_tokenize(text))

    def _item_text(self, item: Dict[str, str]) -> str:
        """把标题/品类/品牌拼接成描述字符串。"""
        parts = [item.get("title", ""), item.get("category", ""), item.get("brand", "")]
        return " ".join([p for p in parts if p])

    def describe_item(self, item_id: str) -> str:
        """根据 item_id 返回可读描述，用于提示词。"""
        for item in self.items:
            if item.get("item_id") == item_id:
                return self._item_text(item)
        return ""

    def user_profile_vector(self, interacted_items: Sequence[str]) -> Counter:
        """累加历史交互物品的向量，粗略表示用户画像。"""
        total = Counter()
        for item_id in interacted_items:
            total += self.item_vectors.get(item_id, Counter())
        return total

    def _cosine(self, a: Counter, b: Counter) -> float:
        """计算两个稀疏计数向量的余弦相似度。"""
        if not a or not b:
            return 0.0
        dot = sum(a[t] * b.get(t, 0) for t in a)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def most_similar_items(
        self, vector: Counter, topn: int = 3, exclude: Iterable[str] | None = None, min_score: float = 0.01
    ) -> List[Tuple[str, float]]:
        """返回与用户画像最相似的前 N 个物品及分数，并过滤弱相关或已看过的物品。"""

        if not vector:
            return []

        scores: List[Tuple[str, float]] = []
        blocked = set(exclude or [])
        for item_id, vec in self.item_vectors.items():
            if item_id in blocked:
                continue
            score = self._cosine(vector, vec)
            if score <= min_score:
                continue
            scores.append((item_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topn]


class GraphEvidenceFinder:
    """检索用户节点到物品节点的多跳路径。"""

    def __init__(self, graph: SimpleGraph):
        self.graph = graph

    def find_paths(self, user_id: str, item_id: str, max_hops: int = 4, limit: int = 2):
        """返回跳数不超过阈值的简单路径，数量受 limit 限制。"""
        user_node = f"user:{user_id}"
        item_node = f"item:{item_id}"
        if not (self.graph.has_node(user_node) and self.graph.has_node(item_node)):
            return []
        paths = self.graph.all_simple_paths(user_node, item_node, cutoff=max_hops)
        results = []
        for path in paths:
            edges: List[str] = []
            nodes: List[str] = []
            for idx in range(len(path) - 1):
                a, b = path[idx], path[idx + 1]
                attrs = self.graph.get_edge_attrs(a, b)
                relation = attrs.get("relation") or attrs.get("type") or "related_to"
                edges.append(relation)
                nodes.append(a)
            nodes.append(path[-1])
            results.append(
                {
                    "nodes": nodes,
                    "edges": edges,
                    "confidence": 1.0 / (len(nodes) or 1),
                }
            )
            if len(results) >= limit:
                break
        # 按路径长度（越短越靠前）和置信度排序，避免顺序不稳定。
        results.sort(key=lambda p: (len(p.get("nodes", [])), -p.get("confidence", 0.0)))
        return results


__all__ = ["VectorEvidenceRetriever", "GraphEvidenceFinder"]
