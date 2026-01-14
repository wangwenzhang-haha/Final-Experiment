#!/usr/bin/env python3
"""
预处理脚本：读取 data/raw/* 并构建一个简单的融合图，保存为 pickle。
依赖最小化：使用纯 Python 数据结构（SimpleGraph），避免额外安装。
"""
import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph.build_fused_graph import build_fused_graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out", default="data/processed/fused_graph.pkl")
    args = parser.parse_args()
    build_fused_graph(args.raw_dir, args.out)
