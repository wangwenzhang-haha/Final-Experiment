"""Evidence retrieval components (vector similarity + simple graph paths)."""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from src.pipeline.simple_graph import SimpleGraph


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer that lowercases tokens."""
    return [tok.lower() for tok in text.replace("/", " ").replace(":", " ").split() if tok]


class VectorEvidenceRetriever:
    """Cosine-similarity retrieval over item metadata without external deps."""

    def __init__(self, items: List[Dict[str, str]]):
        self.items = items
        self.item_vectors: Dict[str, Counter] = {}
        for item in items:
            item_id = item.get("item_id", "")
            text = self._item_text(item)
            self.item_vectors[item_id] = self._vectorize(text)

    def _vectorize(self, text: str) -> Counter:
        """Convert text to a bag-of-words counter."""
        return Counter(_tokenize(text))

    def _item_text(self, item: Dict[str, str]) -> str:
        """Combine title/category/brand into a descriptive string."""
        parts = [item.get("title", ""), item.get("category", ""), item.get("brand", "")]
        return " ".join([p for p in parts if p])

    def describe_item(self, item_id: str) -> str:
        """Return readable description for an item id."""
        for item in self.items:
            if item.get("item_id") == item_id:
                return self._item_text(item)
        return ""

    def user_profile_vector(self, interacted_items: Sequence[str]) -> Counter:
        """Aggregate vectors of interacted items to approximate a profile."""
        total = Counter()
        for item_id in interacted_items:
            total += self.item_vectors.get(item_id, Counter())
        return total

    def _cosine(self, a: Counter, b: Counter) -> float:
        """Compute cosine similarity between two sparse counters."""
        if not a or not b:
            return 0.0
        dot = sum(a[t] * b.get(t, 0) for t in a)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def most_similar_items(self, vector: Counter, topn: int = 3) -> List[Tuple[str, float]]:
        """Return top-n most similar items to the given profile vector."""
        scores: List[Tuple[str, float]] = []
        for item_id, vec in self.item_vectors.items():
            score = self._cosine(vector, vec)
            scores.append((item_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topn]


class GraphEvidenceFinder:
    """Retrieve multi-hop paths between a user and an item."""

    def __init__(self, graph: SimpleGraph):
        self.graph = graph

    def find_paths(self, user_id: str, item_id: str, max_hops: int = 4, limit: int = 2):
        """Return up to `limit` simple paths within the hop cutoff."""
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
        return results


__all__ = ["VectorEvidenceRetriever", "GraphEvidenceFinder"]
