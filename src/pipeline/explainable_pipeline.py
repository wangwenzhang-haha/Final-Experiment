"""无重依赖的可解释推荐演示流水线。

流程对应项目拆解：
- 数据加载（交互、元数据、常识三元组）
- 推荐器（简单流行度基线）
- 检索（向量相似 + 简易图路径）
- 用户目标/偏好推断（轻量启发式）
- LLM 生成 JSON 结构化解释
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
from scripts.preprocess import build_fused_graph


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class DatasetLoader:
    """加载玩具数据集：交互、物品、融合图。"""

    def __init__(self, raw_dir: str = "data/raw", fused_path: str = "data/processed/fused_graph.pkl"):
        self.raw_dir = raw_dir
        self.fused_path = fused_path

    def load_interactions(self) -> List[Dict[str, str]]:
        path = Path(self.raw_dir) / "interactions.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Interactions file not found at {path}. Run scripts/generate_toy_data.py first."
            )
        return _read_csv(path)

    def load_items(self) -> List[Dict[str, str]]:
        path = Path(self.raw_dir) / "items.csv"
        if not path.exists():
            raise FileNotFoundError(f"Items file not found at {path}. Run scripts/generate_toy_data.py first.")
        return _read_csv(path)

    def load_graph(self) -> SimpleGraph:
        """加载融合图；缺失时调用预处理构建，方便图检索模块使用。"""
        fused_path = Path(self.fused_path)
        if not fused_path.exists():
            print(f"[info] fused graph not found at {fused_path}, building automatically...")
            build_fused_graph(self.raw_dir, str(fused_path))
        with fused_path.open("rb") as f:
            import pickle

            return pickle.load(f)


class PopularityRecommender:
    """最小化的流行度基线推荐器。"""

    def __init__(self, interactions: List[Dict[str, str]]):
        self.item_counts = Counter(row["item_id"] for row in interactions)

    def recommend(self, seen_items: Iterable[str], k: int = 3) -> List[Recommendation]:
        """返回未出现过的最热门物品，避免重复曝光。"""
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
    """基于交互历史和元数据推断用户兴趣关键词。"""

    def __init__(self, items: List[Dict[str, str]]):
        self.item_lookup = {item.get("item_id"): item for item in items}

    def summarize(self, history_items: Sequence[str]) -> Dict[str, object]:
        """提取常出现的品类与品牌，并生成一句摘要。"""

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
            summary_bits.append(f"偏好品类：{', '.join(top_categories)}")
        if top_brands:
            summary_bits.append(f"常选品牌：{', '.join(top_brands)}")
        summary_text = "；".join(summary_bits) if summary_bits else "历史行为较少"
        return {
            "top_categories": top_categories,
            "top_brands": top_brands,
            "summary": summary_text,
        }


class ExplainableRecommenderPipeline:
    """串联推荐、检索、解释三步，实现端到端可追溯输出。"""

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
        if not self.interactions:
            raise ValueError("交互数据为空，请先生成或提供最小的 interactions.csv")
        if not self.items:
            raise ValueError("物品元数据为空，请检查 items.csv 是否正确生成")
        self.graph = self.loader.load_graph()
        # 轻量实现，尽量避免额外依赖，方便快速跑通。
        self.recommender = PopularityRecommender(self.interactions)
        self.goal_inferer = UserGoalInferer(self.items)
        self.vector_retriever = VectorEvidenceRetriever(self.items)
        self.graph_finder = GraphEvidenceFinder(self.graph)
        self.explainer = ExplanationGenerator(model=llm_model, temperature=llm_temperature)
        self.similar_topn = similar_topn

    def _user_profile_text(self, user_id: str) -> Tuple[str, List[str]]:
        """拼接用户历史物品的描述，返回文本摘要和物品列表。"""
        history_items = []
        seen = set()
        for row in self.interactions:
            if row.get("user_id") == user_id:
                iid = row.get("item_id")
                if iid and iid not in seen:
                    history_items.append(iid)
                    seen.add(iid)
        texts = [self.vector_retriever.describe_item(i) for i in history_items]
        return " | ".join([t for t in texts if t]), history_items

    def _metadata_evidence(self, item_id: str) -> List[Dict[str, str]]:
        """收集目标物品的属性/取值对，后续可作为解释证据。"""
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
        """对全部用户执行推荐→检索→解释流程，返回结构化结果。"""
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        results: List[PipelineOutput] = []
        users = sorted({row.get("user_id") for row in self.interactions})
        for user_id in users:
            # 1) 从历史交互中提炼画像与目标。
            profile_text, history_items = self._user_profile_text(user_id)
            goal = self.goal_inferer.summarize(history_items)
            # 2) 用流行度基线生成候选推荐列表。
            recs = self.recommender.recommend(history_items, k=top_k)
            # 3) 构造用户向量，便于向量相似度检索。
            user_vec = self.vector_retriever.user_profile_vector(history_items)
            similar_items = self.vector_retriever.most_similar_items(
                user_vec, topn=self.similar_topn, exclude=history_items
            )
            for rec in recs:
                # 4) 在融合图中挖掘短路径，提升解释可追溯性。
                paths = self.graph_finder.find_paths(user_id, rec.item_id, max_hops=max_hops, limit=2)
                path_texts = [" -> ".join(p["nodes"]) for p in paths]
                # 5) 构造仅引用证据的提示词，约束生成内容。
                prompt = build_explanation_prompt(
                    user_id=user_id,
                    item_id=rec.item_id,
                    user_profile=goal.get("summary") or profile_text or "no profile available",
                    item_desc=self.vector_retriever.describe_item(rec.item_id),
                    path_descriptions=path_texts,
                )
                # 6) 整合元数据、相似物品等证据，便于解释输出引用。
                vector_clues = [
                    {"entity": rec.item_id, "attr": "similar_to", "value": f"{sid} ({score:.3f})"}
                    for sid, score in similar_items
                ]
                evidence = Evidence(
                    interactions=[{"user_id": user_id, "item_id": i} for i in history_items],
                    metadata=self._metadata_evidence(rec.item_id) + vector_clues,
                    kg_paths=paths,
                )
                # 7) 优先调用 LLM，否则退回到可复现的规则解释。
                if explain:
                    explanation = self.explainer.generate(user_id, rec.item_id, prompt, evidence.to_dict())
                else:
                    explanation = self.explainer._fallback(user_id, rec.item_id, evidence.to_dict())
                # 8) 组装可序列化的结果，方便保存与评测。
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
