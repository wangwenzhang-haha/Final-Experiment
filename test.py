"""用于构图和路径提取的冒烟测试脚本。"""

from graph_builder import GraphBuilder
from path_extractor import PathExtractor
import pandas as pd
import transformers  # imported to ensure dependency availability during CI
import torch  # same for torch-related sanity checks
import networkx
import pandas
import bitsandbytes

# 模拟数据：构造最小的用户-物品交互与元数据
interactions = pd.DataFrame([
    {'user_id': 'u1', 'item_id': 'i1'},
    {'user_id': 'u1', 'item_id': 'i2'}
])
item_meta = pd.DataFrame([
    {'item_id': 'i1', 'brand': 'Nike', 'category': 'Shoes'},
    {'item_id': 'i2', 'brand': 'Adidas', 'category': 'Shoes'}
])

# 构图：添加交互边和属性边
builder = GraphBuilder()
builder.add_user_item_edges(interactions)
builder.add_item_attribute_edges(item_meta)
G = builder.get_graph()

# 提取路径：从用户 u1 到物品 i2 的路径示例
extractor = PathExtractor(G)
paths = extractor.extract_paths('u1', 'i2')
for path in paths:
    print(extractor.path_to_text(path))
