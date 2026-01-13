"""Build a fused graph from CSV sources with optional ID alignment."""
from __future__ import annotations

import csv
import os
import pickle
from pathlib import Path
from typing import Dict, Set, Tuple

from src.pipeline.simple_graph import SimpleGraph

MappingKey = Tuple[str, str]
MappingValue = Tuple[str, str]


def load_id_mapping(path: str) -> Dict[str, Dict[MappingKey, MappingValue]]:
    """Load ID alignment CSV into forward/reverse lookup tables."""
    if not os.path.exists(path):
        return {"forward": {}, "reverse": {}}

    forward: Dict[MappingKey, MappingValue] = {}
    reverse: Dict[MappingKey, MappingValue] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_type = (row.get("source_type") or "").strip()
            source_id = (row.get("source_id") or "").strip()
            target_type = (row.get("target_type") or "").strip()
            target_id = (row.get("target_id") or "").strip()
            if not source_type or not source_id or not target_type or not target_id:
                continue
            source = (source_type, source_id)
            target = (target_type, target_id)
            forward[source] = target
            reverse[target] = source
    return {"forward": forward, "reverse": reverse}


def _node_id(node_type: str, raw_id: str) -> str:
    return f"{node_type}:{raw_id}"


def _resolve_commonsense_entity(
    entity_id: str,
    mapping: Dict[str, Dict[MappingKey, MappingValue]],
    seen_items: Set[str],
    seen_brands: Set[str],
    seen_categories: Set[str],
) -> Tuple[str, str]:
    """Resolve commonsense entity IDs to aligned node IDs/types."""
    reverse = mapping.get("reverse", {})
    mapped = reverse.get(("kg_entity", entity_id))
    if mapped:
        node_type, raw_id = mapped
        return _node_id(node_type, raw_id), node_type

    if entity_id in seen_items:
        return _node_id("item", entity_id), "item"
    if entity_id in seen_brands:
        return _node_id("brand", entity_id), "brand"
    if entity_id in seen_categories:
        return _node_id("category", entity_id), "category"

    return _node_id("cs", entity_id), "cs_entity"


def _add_node_with_source(graph: SimpleGraph, node: str, source: str, **attrs) -> None:
    graph.add_node(node, **attrs)
    node_data = graph.nodes.get(node, {})
    sources = set(node_data.get("sources", []))
    sources.add(source)
    node_data["sources"] = sorted(sources)


def build_fused_graph(raw_dir: str, out_path: str) -> None:
    """读取交互 / metadata / 常识三元组并构建一个简易融合图。"""
    graph = SimpleGraph()

    id_map_path = os.path.join(raw_dir, "id_map.csv")
    mapping = load_id_mapping(id_map_path)

    seen_items: Set[str] = set()
    seen_brands: Set[str] = set()
    seen_categories: Set[str] = set()

    items_path = os.path.join(raw_dir, "items.csv")
    if os.path.exists(items_path):
        with open(items_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item_id = row.get("item_id", "").strip()
                if not item_id:
                    continue
                seen_items.add(item_id)
                item = _node_id("item", item_id)
                _add_node_with_source(
                    graph,
                    item,
                    "metadata",
                    node_type="item",
                    title=row.get("title"),
                    category=row.get("category"),
                    brand=row.get("brand"),
                )
                category = (row.get("category") or "").strip()
                if category:
                    seen_categories.add(category)
                    cat = _node_id("category", category)
                    _add_node_with_source(graph, cat, "metadata", node_type="category")
                    graph.add_edge(item, cat, relation="has_category")
                brand = (row.get("brand") or "").strip()
                if brand:
                    seen_brands.add(brand)
                    brand_node = _node_id("brand", brand)
                    _add_node_with_source(graph, brand_node, "metadata", node_type="brand")
                    graph.add_edge(item, brand_node, relation="has_brand")

    inter_path = os.path.join(raw_dir, "interactions.csv")
    if os.path.exists(inter_path):
        with open(inter_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_id = row.get("user_id", "").strip()
                item_id = row.get("item_id", "").strip()
                if not user_id or not item_id:
                    continue
                user = _node_id("user", user_id)
                item = _node_id("item", item_id)
                _add_node_with_source(graph, user, "interactions", node_type="user")
                _add_node_with_source(graph, item, "interactions", node_type="item")
                graph.add_edge(user, item, relation=row.get("action", "interact"))

    commons_path = os.path.join(raw_dir, "commonsense_edges.csv")
    if os.path.exists(commons_path):
        with open(commons_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                head_raw = (row.get("head") or "").strip()
                tail_raw = (row.get("tail") or "").strip()
                if not head_raw or not tail_raw:
                    continue
                head, head_type = _resolve_commonsense_entity(
                    head_raw, mapping, seen_items, seen_brands, seen_categories
                )
                tail, tail_type = _resolve_commonsense_entity(
                    tail_raw, mapping, seen_items, seen_brands, seen_categories
                )
                _add_node_with_source(graph, head, "commonsense", node_type=head_type)
                _add_node_with_source(graph, tail, "commonsense", node_type=tail_type)
                rel = row.get("relation", "related_to")
                graph.add_edge(head, tail, relation=rel, confidence=float(row.get("confidence", 1.0)))

    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"Saved fused graph to {out_path}")


__all__ = ["build_fused_graph", "load_id_mapping"]
