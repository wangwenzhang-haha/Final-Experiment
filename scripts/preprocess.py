#!/usr/bin/env python3
"""
预处理脚本：读取 data/raw/* 并构建一个简单的融合图，保存为 pickle。
依赖最小化：使用纯 Python 数据结构（SimpleGraph），避免额外安装。
"""
import csv
import os
import sys
import argparse
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.simple_graph import SimpleGraph


def build_fused_graph(raw_dir: str, out_path: str):
    """读取交互 / metadata / 常识三元组并构建一个简易融合图。"""
    graph = SimpleGraph()
    # 读取 items -> metadata 节点
    items_path = os.path.join(raw_dir, "items.csv")
    if os.path.exists(items_path):
        with open(items_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = f"item:{row['item_id']}"
                graph.add_node(item, node_type="item", title=row.get("title"), category=row.get("category"), brand=row.get("brand"))
                if row.get("category"):
                    cat = f"category:{row.get('category')}"
                    graph.add_node(cat, node_type="category")
                    graph.add_edge(item, cat, relation="has_category")
                if row.get("brand"):
                    brand = f"brand:{row.get('brand')}"
                    graph.add_node(brand, node_type="brand")
                    graph.add_edge(item, brand, relation="has_brand")
    # 读取 interactions
    inter_path = os.path.join(raw_dir, "interactions.csv")
    if os.path.exists(inter_path):
        with open(inter_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                user = f"user:{row['user_id']}"
                item = f"item:{row['item_id']}"
                graph.add_node(user, node_type="user")
                graph.add_edge(user, item, relation=row.get("action", "interact"))
    # 读取 commonsense
    commons_path = os.path.join(raw_dir, "commonsense_edges.csv")
    if os.path.exists(commons_path):
        with open(commons_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                head = f"cs:{row['head']}"
                tail = f"cs:{row['tail']}"
                rel = row.get("relation", "related_to")
                graph.add_node(head, node_type="cs_entity")
                graph.add_node(tail, node_type="cs_entity")
                graph.add_edge(head, tail, relation=rel, confidence=float(row.get("confidence", 1.0)))
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"Saved fused graph to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out", default="data/processed/fused_graph.pkl")
    args = parser.parse_args()
    build_fused_graph(args.raw_dir, args.out)
