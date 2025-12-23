"""基于 NetworkX 构建用户-物品-属性融合图的工具。"""

import networkx as nx
import pandas as pd


class GraphBuilder:
    """逐步拼装无向图，用于离线分析或演示检索。"""

    def __init__(self):
        # 共用一个图实例，便于在多方法间复用。
        self.G = nx.Graph()

    def add_user_item_edges(self, interactions: pd.DataFrame):
        """添加用户到交互物品的边，节点带类型前缀避免冲突。"""
        for _, row in interactions.iterrows():
            # Prefix nodes with type tags to avoid ID collisions.
            self.G.add_node(f"u_{row['user_id']}", type='user')
            self.G.add_node(f"i_{row['item_id']}", type='item')
            self.G.add_edge(f"u_{row['user_id']}", f"i_{row['item_id']}", type='interact')

    def add_item_attribute_edges(self, item_meta: pd.DataFrame):
        """将物品与品牌/品类等属性连接，缺失值跳过以保持稀疏。"""
        for _, row in item_meta.iterrows():
            item_node = f"i_{row['item_id']}"
            for attr_name in ['brand', 'category']:
                # Skip missing attribute values to keep the graph sparse.
                if pd.notna(row[attr_name]):
                    attr_node = f"{attr_name}_{row[attr_name]}"
                    self.G.add_node(attr_node, type=attr_name)
                    self.G.add_edge(item_node, attr_node, type=attr_name)

    def get_graph(self):
        """返回内部 NetworkX 图供下游使用。"""
        return self.G
