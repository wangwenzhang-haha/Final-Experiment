"""Explainable recommender demo pipeline without heavy dependencies.

The pipeline follows the minimal module breakdown described in the project goal:
- Data loading (interactions, metadata, commonsense triples)
- Recommender (popularity baseline)
- Retrieval (vector similarity + simple graph paths)
- Goal/Preference inference (lightweight heuristics)
- LLM explanation generation with a JSON-constrained prompt
"""
from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from src.explain import (
    Evidence,
    ExplanationGenerator,
    GraphEvidenceFinder,
    PipelineOutput,
    Recommendation,
    VectorEvidenceRetriever,
    build_explanation_prompt,
)
from src.pipeline.simple_graph import SimpleGraph
from src.graph.build_fused_graph import build_fused_graph


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class DatasetLoader:
    """Load toy interaction, item, and commonsense data."""

    def __init__(self, raw_dir: str = "data/raw", fused_path: str = "data/processed/fused_graph.pkl"):
        self.raw_dir = raw_dir
        self.fused_path = fused_path

    def load_interactions(self) -> List[Dict[str, str]]:
        path = Path(self.raw_dir) / "interactions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Interactions file not found at {path}. Run scripts/generate_toy_data.py first.")
        return _read_csv(path)

    def load_items(self) -> List[Dict[str, str]]:
        path = Path(self.raw_dir) / "items.csv"
        if not path.exists():
            raise FileNotFoundError(f"Items file not found at {path}. Run scripts/generate_toy_data.py first.")
        return _read_csv(path)

    def load_graph(self) -> SimpleGraph:
        fused_path = Path(self.fused_path)
        if not fused_path.exists():
            print(f"[info] fused graph not found at {fused_path}, building automatically...")
            build_fused_graph(self.raw_dir, str(fused_path))
        with fused_path.open("rb") as f:
            import pickle

            return pickle.load(f)


class PopularityRecommender:
    """A minimal popularity baseline recommender."""

    def __init__(self, interactions: List[Dict[str, str]]):
        self.item_counts = Counter(row["item_id"] for row in interactions)

    def recommend(self, seen_items: Iterable[str], k: int = 3) -> List[Recommendation]:
        """Return the most popular unseen items."""
        seen = set(seen_items)
        recs: List[Recommendation] = []
        for item_id, count in self.item_counts.most_common():
            if item_id in seen:
                continue
            recs.append(Recommendation(item_id=item_id, score=float(count)))
            if len(recs) >= k:
                break
        return recs


class UserGoalInferer:
    """Infer a simple user goal from interaction history and metadata."""

    def __init__(self, items: List[Dict[str, str]]):
        self.item_lookup = {item.get("item_id"): item for item in items}

    def summarize(self, history_items: Sequence[str]) -> Dict[str, object]:
        """Return top categories/brands and a short text summary."""

        categories: Counter[str] = Counter()
        brands: Counter[str] = Counter()
        for item_id in history_items:
            item = self.item_lookup.get(item_id, {})
            if item.get("category"):
                categories[item["category"]] += 1
            if item.get("brand"):
                brands[item["brand"]] += 1
        top_categories = [cat for cat, _ in categories.most_common(2)]
        top_brands = [br for br, _ in brands.most_common(2)]
        summary_bits = []
        if top_categories:
            summary_bits.append(f"prefers categories: {', '.join(top_categories)}")
        if top_brands:
            summary_bits.append(f"often chooses brands: {', '.join(top_brands)}")
        summary_text = "; ".join(summary_bits) if summary_bits else "sparse history"
        return {
            "top_categories": top_categories,
            "top_brands": top_brands,
            "summary": summary_text,
        }


class ExplainableRecommenderPipeline:
    """Tie together recommender, retrieval, and explanation steps."""

    def __init__(
        self,
        raw_dir: str = "data/raw",
        fused_graph: str = "data/processed/fused_graph.pkl",
        similar_topn: int = 3,
        llm_model: str = "gpt-3.5-turbo",
        llm_temperature: float = 0.2,
    ):
        self.loader = DatasetLoader(raw_dir=raw_dir, fused_path=fused_graph)
        self.interactions = self.loader.load_interactions()
        self.items = self.loader.load_items()
        self.graph = self.loader.load_graph()
        # Lightweight modules that keep the demo dependency-free.
        self.recommender = PopularityRecommender(self.interactions)
        self.goal_inferer = UserGoalInferer(self.items)
        self.vector_retriever = VectorEvidenceRetriever(self.items)
        self.graph_finder = GraphEvidenceFinder(self.graph)
        self.explainer = ExplanationGenerator(model=llm_model, temperature=llm_temperature)
        self.similar_topn = similar_topn

    def _user_profile_text(self, user_id: str) -> Tuple[str, List[str]]:
        """Concatenate item descriptions from the user's history."""
        history_items = [row["item_id"] for row in self.interactions if row.get("user_id") == user_id]
        texts = [self.vector_retriever.describe_item(i) for i in history_items]
        return " | ".join([t for t in texts if t]), history_items

    def _metadata_evidence(self, item_id: str) -> List[Dict[str, str]]:
        """Collect attribute/value pairs for the target item."""
        evidence = []
        for item in self.items:
            if item.get("item_id") == item_id:
                for attr in ["title", "category", "brand"]:
                    value = item.get(attr)
                    if value:
                        evidence.append({"entity": item_id, "attr": attr, "value": str(value)})
                break
        return evidence

    def run(self, top_k: int = 3, max_hops: int = 4, explain: bool = True) -> List[PipelineOutput]:
        """Execute the full recommend -> retrieve -> explain loop for all users."""
        results: List[PipelineOutput] = []
        users = sorted({row.get("user_id") for row in self.interactions})
        for user_id in users:
            # 1) Profile + goal extraction from historical interactions.
            profile_text, history_items = self._user_profile_text(user_id)
            goal = self.goal_inferer.summarize(history_items)
            # 2) Generate candidate recommendations using the popularity baseline.
            recs = self.recommender.recommend(history_items, k=top_k)
            # 3) Prepare retrieval hints for vector similarity.
            user_vec = self.vector_retriever.user_profile_vector(history_items)
            similar_items = self.vector_retriever.most_similar_items(user_vec, topn=self.similar_topn)
            for rec in recs:
                # 4) Mine short graph paths for transparency.
                paths = self.graph_finder.find_paths(user_id, rec.item_id, max_hops=max_hops, limit=2)
                path_texts = [" -> ".join(p["nodes"]) for p in paths]
                # 5) Craft a constrained prompt that references the evidence only.
                prompt = build_explanation_prompt(
                    user_id=user_id,
                    item_id=rec.item_id,
                    user_profile=goal.get("summary") or profile_text or "no profile available",
                    item_desc=self.vector_retriever.describe_item(rec.item_id),
                    path_descriptions=path_texts,
                )
                # 6) Assemble evidence fields, mixing metadata and vector neighbors.
                vector_clues = [
                    {"entity": rec.item_id, "attr": "similar_to", "value": f"{sid} ({score:.3f})"}
                    for sid, score in similar_items
                ]
                evidence = Evidence(
                    interactions=[{"user_id": user_id, "item_id": i} for i in history_items],
                    metadata=self._metadata_evidence(rec.item_id) + vector_clues,
                    kg_paths=paths,
                )
                # 7) Choose LLM-backed or deterministic fallback explanation.
                if explain:
                    explanation = self.explainer.generate(user_id, rec.item_id, prompt, evidence.to_dict())
                else:
                    explanation = self.explainer._fallback(user_id, rec.item_id, evidence.to_dict())
                # 8) Emit a structured record that can be serialized to JSONL.
                result = PipelineOutput(
                    user_id=user_id,
                    item_id=rec.item_id,
                    user_profile=goal,
                    recommendations=recs,
                    evidence=evidence,
                    explanation=explanation,
                )
                results.append(result)
        return results


def save_results(results: List[PipelineOutput], out_path: str):
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} records to {out_path}")


__all__ = ["ExplainableRecommenderPipeline", "save_results"]
