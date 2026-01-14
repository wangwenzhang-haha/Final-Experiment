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

    @staticmethod
    def is_valid_edge(src: str, dst: str, relation: str) -> bool:
        """Filter out low-quality relations during traversal."""
        blocked_relations = {"related_to", "unknown", "noise"}
        return relation not in blocked_relations

    @staticmethod
    def score_path(path: Dict[str, List[Dict[str, str]]]) -> float:
        """Score a path using relation weights and path length."""
        edge_scores: List[float] = []
        for edge in path.get("edges", []):
            rel = edge.get("relation", "")
            rel_weight = {
                "interacts_with": 1.0,
                "has_brand": 1.2,
                "has_category": 1.0,
                "similar_to": 0.7,
                "related_to": 0.3,
                "cs_related": 0.4,
            }.get(rel, 0.5)
            edge_scores.append(rel_weight)
        if not edge_scores:
            return 0.0
        length_score = 1.0 / max(len(path.get("nodes", [])), 1)
        return length_score * (sum(edge_scores) / len(edge_scores))

    def find_paths(self, user_id: str, item_id: str, max_hops: int = 4, limit: int = 2):
        """Return up to `limit` simple paths within the hop cutoff."""
        user_node = f"user:{user_id}"
        item_node = f"item:{item_id}"
        if not (self.graph.has_node(user_node) and self.graph.has_node(item_node)):
            return []
        max_paths = 50
        collected: List[Dict[str, List[Dict[str, str]]]] = []

        def dfs(current: str, depth: int, visited: List[str], edges: List[Dict[str, str]]):
            if depth > max_hops or len(collected) >= max_paths:
                return
            if current == item_node:
                collected.append({"nodes": list(visited), "edges": list(edges)})
                return
            for edge in self.graph.neighbors(current):
                nxt = edge.target
                if nxt in visited:
                    continue
                attrs = edge.attrs or {}
                relation = attrs.get("relation") or attrs.get("type") or "related_to"
                if not self.is_valid_edge(current, nxt, relation):
                    continue
                visited.append(nxt)
                edges.append({"src": current, "dst": nxt, "relation": relation})
                dfs(nxt, depth + 1, visited, edges)
                edges.pop()
                visited.pop()

        dfs(user_node, 0, [user_node], [])

        scored = []
        for path in collected:
            scored.append({**path, "score": self.score_path(path)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]


__all__ = ["VectorEvidenceRetriever", "GraphEvidenceFinder"]
