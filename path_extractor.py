"""从图中提取多跳路径并格式化的辅助工具。"""

import networkx as nx


class PathExtractor:
    """查找用户到物品的路径，并转换成可读短语。"""

    def __init__(self, G: nx.Graph):
        # 保存图引用，便于多次查询复用。
        self.G = G

    def extract_paths(self, user_id: str, item_id: str, max_len: int = 3):
        """返回用户节点到物品节点的简单路径，跳数不超过 max_len。"""
        src = f"u_{user_id}"
        tgt = f"i_{item_id}"
        if not (self.G.has_node(src) and self.G.has_node(tgt)):
            return []
        try:
            paths = list(nx.all_simple_paths(self.G, source=src, target=tgt, cutoff=max_len))
            return paths[:5]  # 仅保留前 5 条，避免输出过长。
        except Exception:
            # 图为空等情况会触发异常，吞掉以保证演示不中断。
            return []

    def path_to_text(self, path):
        """将路径节点序列转成箭头串联的可读字符串。"""
        edges = []
        for i in range(len(path) - 1):
            a, b = path[i], path[i+1]
            relation = self.G.edges[a, b].get('type', 'related to')
            edges.append(f"[{a}] --{relation}--> [{b}]")
        return " -> ".join(edges)
