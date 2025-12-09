# 文件: path_extractor.py
# 功能: 提取多跳路径并转为解释候选文本

import networkx as nx

class PathExtractor:
    def __init__(self, G: nx.Graph):
        self.G = G

    def extract_paths(self, user_id: str, item_id: str, max_len: int = 3):
        src = f"u_{user_id}"
        tgt = f"i_{item_id}"
        if not (self.G.has_node(src) and self.G.has_node(tgt)):
            return []
        try:
            paths = list(nx.all_simple_paths(self.G, source=src, target=tgt, cutoff=max_len))
            return paths[:5]  # 限制最多返回5条
        except:
            return []

    def path_to_text(self, path):
        edges = []
        for i in range(len(path) - 1):
            a, b = path[i], path[i+1]
            relation = self.G.edges[a, b].get('type', 'related to')
            edges.append(f"[{a}] --{relation}--> [{b}]")
        return " -> ".join(edges)
