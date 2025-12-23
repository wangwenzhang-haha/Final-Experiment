#!/bin/bash
set -euo pipefail

# Minimal toy pipeline: generate data -> preprocess -> run explainable recommender demo
python3 scripts/run_demo.py --dataset toy --k 2 --explain --config configs/demo.yaml

cat <<'EOF'
Next steps (placeholders):
 - train reco: python scripts/train_reco.py --config config/default.yaml
 - learn edge mask: python scripts/learn_edge_mask.py --config config/default.yaml
 - extract paths: python scripts/extract_paths.py --config config/default.yaml --topk 3
 - build prompts: python scripts/build_prompts.py --paths outputs/paths.jsonl
 - generate explanations: python scripts/generate_explanations.py --prompts outputs/prompts.jsonl
EOF
