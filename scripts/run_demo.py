#!/usr/bin/env python3
"""Entry point for the explainable recommender demo.

Steps: data load -> baseline recommend -> evidence retrieval -> LLM explanation -> save JSON output.
Each helper is intentionally lightweight so the script can act as a documented recipe
for running the minimum viable prototype end to end.
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
    """Load YAML (preferred) or JSON config from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        if yaml is not None:
            return yaml.safe_load(f)
        return json.load(f)


def ensure_toy_data(raw_dir: Path):
    """Generate the toy dataset if it does not exist yet."""
    interactions_path = raw_dir / "interactions.csv"
    if interactions_path.exists():
        return
    print("[info] toy data not found; generating via scripts/generate_toy_data.py ...")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_toy_data.py")], check=True)


def ensure_fused_graph(raw_dir: Path, fused_path: Path):
    """Trigger preprocessing to create the fused graph for graph retrieval."""
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

    # Load user-specified or default config to control paths and parameters.
    config = load_config(Path(args.config))
    dataset_conf = config.get("dataset", {})
    pipeline_conf = config.get("pipeline", {})
    retrieval_conf = config.get("retrieval", {})
    llm_conf = config.get("llm", {})

    raw_dir = Path(dataset_conf.get("raw_dir", "data/raw"))
    fused_graph = Path(dataset_conf.get("fused_graph", "data/processed/fused_graph.pkl"))
    out_path = dataset_conf.get("out_path", "outputs/demo_results.jsonl")

    # Make sure minimum inputs exist before running the pipeline.
    ensure_toy_data(raw_dir)
    ensure_fused_graph(raw_dir, fused_graph)

    topk = args.k if args.k is not None else pipeline_conf.get("topk", 10)
    max_hops = pipeline_conf.get("max_hops", 4)

    # Build the pipeline with the selected hyperparameters.
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
