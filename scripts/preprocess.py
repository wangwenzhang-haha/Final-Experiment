#!/usr/bin/env python3
"""
预处理脚本 stub：读取 data/raw/* 并构建一个简单的融合图（NetworkX），保存为 pickle。
实际工程应替换为更完整的图构建与对齐逻辑。
"""
import os
import csv
import argparse
import networkx as nx
import pickle
from pathlib import Path

def build_fused_graph(raw_dir, out_path):
    G = nx.MultiDiGraph()
    # 读取 items -> metadata 节点
    items_path = os.path.join(raw_dir, "items.csv")
    if os.path.exists(items_path):
        with open(items_path, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                item = f"item:{r['item_id']}"
                G.add_node(item, node_type="item", title=r.get("title"), category=r.get("category"), brand=r.get("brand"))
                # add metadata nodes
                if r.get("category"):
                    cat = f"category:{r.get('category')}"
                    G.add_node(cat, node_type="category")
                    G.add_edge(item, cat, relation="has_category")
                if r.get("brand"):
                    brand = f"brand:{r.get('brand')}"
                    G.add_node(brand, node_type="brand")
                    G.add_edge(item, brand, relation="has_brand")
    # 读取 interactions
    inter_path = os.path.join(raw_dir, "interactions.csv")
    if os.path.exists(inter_path):
        with open(inter_path, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                user = f"user:{r['user_id']}"
                item = f"item:{r['item_id']}"
                G.add_node(user, node_type="user")
                G.add_edge(user, item, relation=r.get("action", "interact"))
    # 读取 commonsense
    commons_path = os.path.join(raw_dir, "commonsense_edges.csv")
    if os.path.exists(commons_path):
        with open(commons_path, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                head = r['head']
                tail = r['tail']
                rel = r['relation']
                head_n = f"cs:{head}"
                tail_n = f"cs:{tail}"
                G.add_node(head_n, node_type="cs_entity")
                G.add_node(tail_n, node_type="cs_entity")
                G.add_edge(head_n, tail_n, relation=rel, confidence=float(r.get("confidence", 1.0)))
    # save
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved fused graph to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out", default="data/processed/fused_graph.pkl")
    args = parser.parse_args()
    build_fused_graph(args.raw_dir, args.out)