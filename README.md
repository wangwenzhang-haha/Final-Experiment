```markdown
# MultiDim-MultiHop Explainable Recommender (KG + LLM)

一句话概述
- 三图融合（交互 / 元数据 / 常识），通过 m-core 剪枝 + edge-mask 学习提取多跳路径，用动态 Prompt + LoRA 生成可追溯、事实约束的推荐解释。

当前目标
- 生成多维（interaction / metadata / commonsense）且多跳（hop >= 2）的可追溯解释；
- 保证解释基于抽取出的路径证据，不允许 LLM 发明未在路径中出现的实体或关系；
- 在不损失推荐效果的前提下，提供高质量的自然语言解释并可评估其准确性与稳定性。

主要特性
- 三源异构信息融合图构建（交互 / 元数据 / 常识）
- 支持 LightGCN / GCN / GraphSAGE 作为推荐骨干
- m-core 剪枝 + edge mask learning（学习边重要性）
- Top-K 加权路径抽取（Dijkstra / beam search 等）
- 动态 Prompt：原子模板库 + 填槽 + 控制标记
- LoRA 轻量微调（可选）与标准推理
- 双评估：推荐（Recall@K、Coverage 等）与解释（BERTScore/BLEURT/GPTScore、多样性/稳定性等）
- 可追溯输出：paths.jsonl、prompts.jsonl、explanations.jsonl、eval_report.json

仓库建议结构
- src/
  - config/           # yaml/json 配置
  - data/             # 原始数据、预处理、toy 数据生成
  - graph/            # 图构建、对齐、m-core、edge-mask
  - model/            # 推荐模型训练与接口（LightGCN/GCN/GraphSAGE）
  - path/             # 路径抽取、路径评分与合法性判定
  - prompt/           # 原子模板库、Prompt Builder、用户/物品画像生成
  - llm/              # LLM 接口、LoRA 微调、post-checker
  - eval/             # 推荐与解释评估脚本
  - utils/            # IO / logging / seed / metrics
- scripts/
  - preprocess.py
  - train_reco.py
  - learn_edge_mask.py
  - extract_paths.py
  - build_prompts.py
  - finetune_lora.py
  - generate_explanations.py
  - evaluate.py
- config/default.yaml
- requirements.txt
- README.md
- docs/
  - architecture.md
- outputs/ (运行时生成)
  - checkpoints/, logs/, paths.jsonl, prompts.jsonl, explanations.jsonl, eval_report.json

快速开始（Minimal reproducible example）
1) 环境
   - Python 3.8+
   - 附带 GPU 时推荐 CUDA 支持（用于训练/LoRA 微调）
   - pip install -r requirements.txt

2) 准备数据
   - data/raw/interactions.csv: user_id,item_id,timestamp,action
   - data/raw/items.csv: item_id,title,category,brand,meta_json
   - data/raw/commonsense_edges.csv: head,relation,tail,confidence
   - 仓库提供 toy 数据脚本：scripts/generate_toy_data.py（生成小规模示例）

3) 预处理并构建融合图
   - python scripts/preprocess.py --config config/default.yaml
   - 输出：src/data/processed/fused_graph.pkl、id_map.json

4) 训练推荐模型（示例）
   - python scripts/train_reco.py --config config/default.yaml --save_dir outputs/checkpoints/reco/

5) 学习 edge-mask 与剪枝（可选）
   - python scripts/learn_edge_mask.py --config config/default.yaml --graph outputs/fused_graph.pkl

6) 抽取 Top-K 多跳路径
   - python scripts/extract_paths.py --config config/default.yaml --topk 3 --max_hops 4 --out outputs/paths.jsonl

7) 构建 Prompt & 生成解释
   - python scripts/build_prompts.py --paths outputs/paths.jsonl --out outputs/prompts.jsonl
   - python scripts/generate_explanations.py --prompts outputs/prompts.jsonl --llm_provider openai --out outputs/explanations.jsonl

8) 评估
   - python scripts/evaluate.py --pred outputs/explanations.jsonl --gold data/gold_explanations.jsonl --out outputs/eval_report.json

配置示例（config/default.yaml - 片段）
```yaml
graph:
  relation_types: ["interaction","metadata","commonsense"]
  align_strategy: "id_map"
  m_core: 5
  edge_mask:
    enabled: true
    lr: 1e-3
model:
  backbone: "lightgcn"        # lightgcn | gcn | graphsage
  embedding_dim: 64
  lr: 0.001
  epochs: 20
path:
  topk: 3
  max_hops: 4
  scoring: "weighted_shortest"  # weighted_shortest | beam_score
prompt:
  atomic_library: "src/prompt/atomic_templates.yaml"
  control_tokens: ["CONCISE","FACTUAL"]
llm:
  provider: "openai"         # openai | local
  api_key_env: "OPENAI_API_KEY"
  lora:
    enabled: true
    rank: 8
    alpha: 16
eval:
  reco_metrics: ["Recall@20","Coverage"]
  explain_metrics: ["BERTScore","BLEURT"]
```

I/O 规范（关键输出）
- outputs/paths.jsonl
  - 每行 JSON 示例：
    {"user": "u123", "item": "i456", "paths": [{"nodes":["u123","i789","brand_X","i456"], "rels":["interact","has_brand","same_brand"], "score":0.85, "hops":3}], "meta": {...}}
- outputs/prompts.jsonl
  - 每行 JSON 示例：
    {"user":"u123","item":"i456","prompt_struct": {...}, "prompt_text":"..."}
- outputs/explanations.jsonl
  - 每行 JSON 示例：
    {"user":"u123","item":"i456","explanation":"...","prompt_id":"p001","paths_ref":["p1","p2"], "llm_stats": {...}}
- outputs/eval_report.json
  - 推荐与解释指标、运行配置（config hash）、seed、checkpoint

设计与工程约束（必须遵守）
- 三原则优先：事实约束、可追溯路径、多维多跳。LLM 输出必须以 paths.jsonl 中的实体/关系为事实来源。
- 所有 LLM 生成后必须走 post-checker：解析输出中引用的实体/关系，检查是否都在 paths 中；若不符，触发纠错策略（重写 prompt / 降低温度 / 打上警告）。
- 保存完整运行快照：config、seed、git commit、checkpoints、paths、prompts、explanations、eval_report。
- 模块化接口：每个模块应有清晰的函数签名与输入/输出（便于单测）。
- 路径合法性：hop >= 2 且覆盖 >= 2 个信息源（interaction/metadata/commonsense）。

评估要点
- 推荐：Recall@K、Coverage、NDCG（可选）
- 解释：BERTScore-F1、BLEURT、GPTScore（调用时需考虑 API 成本）
- 多样性：Diversity@K（不同路径/解释的多样性）
- 稳定性：对同一 prompt 多次生成结果的方差（测量随机性/可靠性）

开发建议（实践层面）
- Prompt 中强制加入 Evidence 段（列出 Path#1..Path#K），并要求 LLM 在解释中引用 Path 索引。
- 原子模板库（JSON/YAML）定义模板 id、slot、控制标签与 verbosity_hint，便于动态拼接。
- 在路径抽取阶段产生结构化证据（节点/关系/分数/来源），并将其作为 LLM 的唯一事实来源。
- 推荐实现轻量 post-checker：通过简单正则/NER 匹配解析 LLM 输出实体并校验。
- 提供 toy demo notebook，展示从数据 -> 路径 -> prompt -> 解释 -> 评估 的最小流程。