"""玩具流水线使用的极简无向图工具。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class Edge:
    target: str
    attrs: Dict[str, Any]


class SimpleGraph:
    """为演示定制的轻量无向图工具类。"""

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.adj: Dict[str, List[Edge]] = {}

    def add_node(self, node: str, **attrs):
        """插入或更新节点，可附带属性。"""
        self.nodes.setdefault(node, {}).update(attrs)
        self.adj.setdefault(node, [])

    def add_edge(self, src: str, dst: str, **attrs):
        """创建两个节点间的无向边，自动补齐缺失节点。"""
        self.add_node(src)
        self.add_node(dst)
        self.adj[src].append(Edge(target=dst, attrs=attrs))
        self.adj[dst].append(Edge(target=src, attrs=attrs))

    def has_node(self, node: str) -> bool:
        return node in self.nodes

    def neighbors(self, node: str) -> Iterable[Edge]:
        return self.adj.get(node, [])

    def get_edge_attrs(self, src: str, dst: str) -> Dict[str, Any]:
        """读取指定边的属性，找不到则返回空字典。"""
        for edge in self.adj.get(src, []):
            if edge.target == dst:
                return edge.attrs
        return {}

    def all_simple_paths(self, source: str, target: str, cutoff: int) -> List[List[str]]:
        """使用 DFS 找到跳数不超过 cutoff 的简单路径。"""
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
