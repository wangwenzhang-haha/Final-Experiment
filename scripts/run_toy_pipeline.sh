#!/bin/bash
set -e
# Minimal toy pipeline: generate -> preprocess -> (placeholders for later stages)
python3 scripts/generate_toy_data.py
python3 scripts/preprocess.py --raw_dir data/raw --out data/processed/fused_graph.pkl

echo "Next steps (placeholders):"
echo " - train reco: python scripts/train_reco.py --config config/default.yaml"
echo " - learn edge mask: python scripts/learn_edge_mask.py --config config/default.yaml"
echo " - extract paths: python scripts/extract_paths.py --config config/default.yaml --topk 3"
echo " - build prompts: python scripts/build_prompts.py --paths outputs/paths.jsonl"
echo " - generate explanations: python scripts/generate_explanations.py --prompts outputs/prompts.jsonl"