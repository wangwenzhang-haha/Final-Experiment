#!/usr/bin/env python3
"""
生成 toy 数据：interactions.csv, items.csv, commonsense_edges.csv
"""
import csv
import os

OUT_DIR = "data/raw"
os.makedirs(OUT_DIR, exist_ok=True)

# interactions: user_id,item_id,timestamp,action
interactions = [
    ("u1", "i1", "2025-01-01T00:00:00", "click"),
    ("u1", "i2", "2025-01-02T00:00:00", "buy"),
    ("u2", "i3", "2025-01-03T00:00:00", "click"),
]

with open(os.path.join(OUT_DIR, "interactions.csv"), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["user_id", "item_id", "timestamp", "action"])
    writer.writerows(interactions)

# items: item_id,title,category,brand,meta_json
items = [
    ("i1", "Fantastic Widget", "Gadgets", "BrandX", '{"price": 19.9}'),
    ("i2", "Super Gadget", "Gadgets", "BrandY", '{"price": 29.9}'),
    ("i3", "Cool Thing", "Accessories", "BrandX", '{"price": 9.9}'),
]

with open(os.path.join(OUT_DIR, "items.csv"), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["item_id", "title", "category", "brand", "meta_json"])
    writer.writerows(items)

# commonsense_edges: head,relation,tail,confidence
commonsense = [
    ("BrandX", "similar_to", "BrandY", "0.7"),
    ("Gadgets", "related_to", "Accessories", "0.6"),
]

with open(os.path.join(OUT_DIR, "commonsense_edges.csv"), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["head", "relation", "tail", "confidence"])
    writer.writerows(commonsense)

print(f"Toy data generated in {OUT_DIR}")