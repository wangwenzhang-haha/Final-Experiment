#!/usr/bin/env python3
"""Run the toy end-to-end explainable recommender pipeline."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.explainable_pipeline import ExplainableRecommenderPipeline, save_results


def main():
    parser = argparse.ArgumentParser(description="Run explainable recommender demo")
    parser.add_argument("--raw_dir", default="data/raw", help="Directory containing raw CSV files")
    parser.add_argument(
        "--fused_graph", default="data/processed/fused_graph.pkl", help="Path to fused graph pickle"
    )
    parser.add_argument("--topk", type=int, default=3, help="Number of recommendations per user")
    parser.add_argument("--max_hops", type=int, default=4, help="Maximum hops for graph evidence")
    parser.add_argument(
        "--out", default="outputs/demo_results.jsonl", help="Where to store the JSONL output"
    )
    args = parser.parse_args()

    pipeline = ExplainableRecommenderPipeline(raw_dir=args.raw_dir, fused_graph=args.fused_graph)
    results = pipeline.run(top_k=args.topk, max_hops=args.max_hops)
    save_results(results, args.out)
    print(f"Example record:\n{results[0].to_dict() if results else 'no results generated'}")


if __name__ == "__main__":
    main()
