#!/usr/bin/env python3
"""解释型推荐系统的演示入口脚本。

流程：数据加载 → 基线推荐 → 证据检索 → LLM 解释 → 保存 JSON 结果。
逻辑刻意保持简洁，方便把最小可运行原型的执行步骤写清楚，便于阅读和复用。
"""
import argparse
from importlib import import_module, util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

yaml_spec = util.find_spec("yaml")
yaml = import_module("yaml") if yaml_spec else None

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preprocess import build_fused_graph
from src.pipeline.explainable_pipeline import ExplainableRecommenderPipeline, save_results


def load_config(path: Path) -> Dict[str, Any]:
    """读取 YAML（优先）或 JSON 配置文件，失败立即抛错以便快速定位问题。"""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        if yaml is not None:
            return yaml.safe_load(f)
        return json.load(f)


def validate_config(config: Dict[str, Any]):
    """对关键字段做基本类型与数值校验，避免静默使用非法配置。"""

    if not isinstance(config, dict):
        raise ValueError("配置文件格式错误，应为字典结构")

    dataset_conf = config.get("dataset", {})
    pipeline_conf = config.get("pipeline", {})
    retrieval_conf = config.get("retrieval", {})
    llm_conf = config.get("llm", {})

    if not isinstance(dataset_conf.get("raw_dir", ""), str):
        raise ValueError("dataset.raw_dir 必须是字符串路径")
    if not isinstance(dataset_conf.get("fused_graph", ""), str):
        raise ValueError("dataset.fused_graph 必须是字符串路径")

    topk = pipeline_conf.get("topk", 10)
    max_hops = pipeline_conf.get("max_hops", 4)
    similar_topn = retrieval_conf.get("similar_topn", 3)
    if not isinstance(topk, int) or topk <= 0:
        raise ValueError("pipeline.topk 必须为正整数")
    if not isinstance(max_hops, int) or max_hops <= 0:
        raise ValueError("pipeline.max_hops 必须为正整数")
    if not isinstance(similar_topn, int) or similar_topn <= 0:
        raise ValueError("retrieval.similar_topn 必须为正整数")

    if llm_conf and not isinstance(llm_conf.get("model", ""), str):
        raise ValueError("llm.model 必须为字符串")


def ensure_toy_data(raw_dir: Path):
    """检查并生成玩具数据集；缺失时自动调用生成脚本。"""
    interactions_path = raw_dir / "interactions.csv"
    if interactions_path.exists():
        return
    print("[info] toy data not found; generating via scripts/generate_toy_data.py ...")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_toy_data.py")], check=True)


def ensure_fused_graph(raw_dir: Path, fused_path: Path):
    """检查并生成融合图，用于图检索；不存在时执行预处理。"""
    if fused_path.exists():
        return
    print(f"[info] fused graph missing at {fused_path}; building now ...")
    build_fused_graph(str(raw_dir), str(fused_path))


def main():
    parser = argparse.ArgumentParser(description="Run explainable recommender demo")
    parser.add_argument("--dataset", default="toy", help="Dataset name (for logging)")
    parser.add_argument("--k", type=int, default=None, help="Top-K recommendations override")
    parser.add_argument("--config", default="configs/demo.yaml", help="Path to demo YAML config")
    parser.add_argument("--explain", action="store_true", help="Enable LLM explanation generation")
    args = parser.parse_args()

    # 读取配置，优先使用参数指定文件，否则使用默认 demo 配置。
    config = load_config(Path(args.config))
    validate_config(config)

    dataset_conf = config.get("dataset", {})
    pipeline_conf = config.get("pipeline", {})
    retrieval_conf = config.get("retrieval", {})
    llm_conf = config.get("llm", {})

    raw_dir = Path(dataset_conf.get("raw_dir", "data/raw"))
    fused_graph = Path(dataset_conf.get("fused_graph", "data/processed/fused_graph.pkl"))
    out_path = dataset_conf.get("out_path", "outputs/demo_results.jsonl")
    raw_dir.mkdir(parents=True, exist_ok=True)
    fused_graph.parent.mkdir(parents=True, exist_ok=True)

    # 确保最基本的输入文件存在，缺失时自动补齐。
    ensure_toy_data(raw_dir)
    ensure_fused_graph(raw_dir, fused_graph)

    topk = args.k if args.k is not None else pipeline_conf.get("topk", 10)
    if topk is not None and topk <= 0:
        raise ValueError("参数 --k 必须为正整数")
    max_hops = pipeline_conf.get("max_hops", 4)

    # 依据配置构建流水线组件，复用前置模块以减少依赖。
    pipeline = ExplainableRecommenderPipeline(
        raw_dir=str(raw_dir),
        fused_graph=str(fused_graph),
        similar_topn=retrieval_conf.get("similar_topn", 3),
        llm_model=llm_conf.get("model", "gpt-3.5-turbo"),
        llm_temperature=llm_conf.get("temperature", 0.2),
    )
    print(
        f"[info] running dataset={args.dataset} topk={topk} max_hops={max_hops} explain={args.explain} output={out_path}"
    )
    results = pipeline.run(top_k=topk, max_hops=max_hops, explain=args.explain)
    save_results(results, out_path)
    if results:
        print(f"Example record:\n{results[0].to_dict()}")


if __name__ == "__main__":
    main()
