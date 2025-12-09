# 文件: graph_builder.py
# 功能: 构建用户-物品-属性融合图（NetworkX）

import networkx as nx
import pandas as pd

class GraphBuilder:
    def __init__(self):
        self.G = nx.Graph()

    def add_user_item_edges(self, interactions: pd.DataFrame):
        for _, row in interactions.iterrows():
            self.G.add_node(f"u_{row['user_id']}", type='user')
            self.G.add_node(f"i_{row['item_id']}", type='item')
            self.G.add_edge(f"u_{row['user_id']}", f"i_{row['item_id']}", type='interact')

    def add_item_attribute_edges(self, item_meta: pd.DataFrame):
        for _, row in item_meta.iterrows():
            item_node = f"i_{row['item_id']}"
            for attr_name in ['brand', 'category']:
                if pd.notna(row[attr_name]):
                    attr_node = f"{attr_name}_{row[attr_name]}"
                    self.G.add_node(attr_node, type=attr_name)
                    self.G.add_edge(item_node, attr_node, type=attr_name)

    def get_graph(self):
        return self.G
