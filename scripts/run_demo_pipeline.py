#!/usr/bin/env python3
"""运行玩具版端到端可解释推荐流水线的脚本。"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.explainable_pipeline import ExplainableRecommenderPipeline, save_results


def main():
    parser = argparse.ArgumentParser(description="运行可解释推荐演示")
    parser.add_argument("--raw_dir", default="data/raw", help="原始 CSV 所在目录")
    parser.add_argument(
        "--fused_graph", default="data/processed/fused_graph.pkl", help="融合图 pickle 文件路径"
    )
    parser.add_argument("--topk", type=int, default=3, help="每个用户的推荐数量")
    parser.add_argument("--max_hops", type=int, default=4, help="图证据允许的最大跳数")
    parser.add_argument(
        "--out", default="outputs/demo_results.jsonl", help="JSONL 输出保存位置"
    )
    args = parser.parse_args()

    pipeline = ExplainableRecommenderPipeline(raw_dir=args.raw_dir, fused_graph=args.fused_graph)
    results = pipeline.run(top_k=args.topk, max_hops=args.max_hops)
    save_results(results, args.out)
    print(f"示例记录:\n{results[0].to_dict() if results else 'no results generated'}")


if __name__ == "__main__":
    main()
