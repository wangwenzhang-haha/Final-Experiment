"""A minimal undirected graph utility for the toy pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class Edge:
    target: str
    attrs: Dict[str, Any]


class SimpleGraph:
    """Tiny undirected graph helper tailored for the toy demo."""

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.adj: Dict[str, List[Edge]] = {}

    def add_node(self, node: str, **attrs):
        """Insert or update a node with optional attributes."""
        self.nodes.setdefault(node, {}).update(attrs)
        self.adj.setdefault(node, [])

    def add_edge(self, src: str, dst: str, **attrs):
        """Store an undirected edge between two nodes."""
        self.add_node(src)
        self.add_node(dst)
        self.adj[src].append(Edge(target=dst, attrs=attrs))
        self.adj[dst].append(Edge(target=src, attrs=attrs))

    def has_node(self, node: str) -> bool:
        return node in self.nodes

    def neighbors(self, node: str) -> Iterable[Edge]:
        return self.adj.get(node, [])

    def get_edge_attrs(self, src: str, dst: str) -> Dict[str, Any]:
        """Retrieve edge attributes if the edge exists."""
        for edge in self.adj.get(src, []):
            if edge.target == dst:
                return edge.attrs
        return {}

    def all_simple_paths(self, source: str, target: str, cutoff: int) -> List[List[str]]:
        """Depth-first search for simple paths up to a hop cutoff."""
        paths: List[List[str]] = []

        def dfs(current: str, target: str, depth: int, visited: List[str]):
            if depth > cutoff:
                return
            if current == target:
                paths.append(list(visited))
                return
            for edge in self.neighbors(current):
                nxt = edge.target
                if nxt in visited:
                    continue
                visited.append(nxt)
                dfs(nxt, target, depth + 1, visited)
                visited.pop()

        dfs(source, target, 0, [source])
        return paths


__all__ = ["SimpleGraph", "Edge"]
