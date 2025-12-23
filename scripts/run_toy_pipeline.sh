#!/bin/bash
set -euo pipefail

# 最小玩具流水线：生成数据 → 预处理 → 运行可解释推荐 Demo
python3 scripts/run_demo.py --dataset toy --k 2 --explain --config configs/demo.yaml

cat <<'EOF'
后续步骤（占位）：
 - 训练推荐：python scripts/train_reco.py --config config/default.yaml
 - 学习边权/掩码：python scripts/learn_edge_mask.py --config config/default.yaml
 - 提取路径：python scripts/extract_paths.py --config config/default.yaml --topk 3
 - 构建提示词：python scripts/build_prompts.py --paths outputs/paths.jsonl
 - 生成解释：python scripts/generate_explanations.py --prompts outputs/prompts.jsonl
EOF
