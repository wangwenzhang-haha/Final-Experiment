```markdown
# MultiDim-MultiHop Explainable Recommender (KG + LLM)

一句话概述
- 三图融合（交互 / 元数据 / 常识），通过 m-core 剪枝 + edge-mask 学习提取多跳路径，用动态 Prompt 与可选 LoRA 生成可追溯、事实约束的推荐解释。

项目目标
- 为推荐结果提供多维（interaction / metadata / commonsense）、多跳（hop ≥ 2）且可追溯的路径证据，并基于这些证据生成事实约束的自然语言解释。
- 输出包含：推荐结果、Top-K 路径证据与基于证据的自然语言解释；所有中间产物需可复现并可校验。

主要特性
- 三源异构信息融合：交互日志、物品元数据、常识 KG
- 推荐骨干（可选）：LightGCN / GCN / GraphSAGE（仓库含最小 LightGCN 示例）
- 图剪枝与边重要性：m-core 剪枝、edge-mask 学习（可扩展）
- 多跳路径抽取：基于加权图的 Dijkstra / beam search（支持 hop ≥ 2 且跨维度）
- 动态 Prompt：原子模板库 + 填槽 + 控制标记
- LLM 支持：OpenAI 或本地模型；可选 LoRA 轻量微调（需量化与 offload 支持）
- 评估：推荐（Recall@K、Coverage）、解释（BERTScore、BLEURT、GPTScore）、多样性与稳定性指标
- 可追溯输出：paths.jsonl、prompts.jsonl、explanations.jsonl、eval_report.json（含 config/seed/checkpoint）

仓库结构（示例）
- src/
  - data/      # 数据处理
  - graph/     # 图构建、剪枝、edge-mask
  - model/     # 推荐模型（LightGCN/GCN/GraphSAGE）
  - path/      # 路径抽取与评分
  - prompt/    # 原子模板库、Prompt Builder
  - llm/       # LLM 接口、LoRA 训练/加载、post-checker
  - eval/      # 推荐与解释评估工具
  - utils/     # 工具函数（IO、日志、seed）
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
- docs/architecture.md
- outputs/    # 运行时产物（checkpoints/, paths.jsonl, prompts.jsonl, ...）

快速开始（最小复现实验）
1) 安装依赖
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt

2) 运行玩具级端到端 Demo（包含数据生成 -> 图构建 -> 推荐 -> 证据检索 -> 解释输出）
   bash scripts/run_toy_pipeline.sh
   # 等价：python scripts/run_demo.py --dataset toy --k 10 --explain --config configs/demo.yaml
   # 产物：outputs/demo_results.jsonl，单行 JSON 满足输出 schema，包含推荐列表、用户画像摘要、图路径、元数据与解释

3) 准备或生成示例数据（交互、元数据、常识 KG）
   - data/raw/interactions.csv
   - data/raw/items.csv
   - data/raw/commonsense_edges.csv
   或使用仓库中的 toy 数据脚本（若有）

4) 预处理并构建融合图
   python scripts/preprocess.py --raw_dir data/raw --out data/processed/fused_graph.pkl

5) 训练推荐模型（示例）
   python scripts/train_reco.py --config config/default.yaml --save_dir outputs/checkpoints/reco/

6) （可选）学习 edge-mask 与剪枝，抽取 Top-K 多跳路径，构建 prompts，生成解释并评估
   - python scripts/learn_edge_mask.py ...
   - python scripts/extract_paths.py ...
   - python scripts/build_prompts.py ...
   - python scripts/generate_explanations.py ...
   - python scripts/evaluate.py ...

配置示例（config/default.yaml - 关键片段）
```yaml
graph:
  relation_types: ["interaction", "metadata", "commonsense"]
  align_strategy: "id_map"
  m_core: 5

model:
  backbone: "lightgcn"
  embedding_dim: 64
  lr: 0.001
  epochs: 20
  train_batch_size: 256

path:
  topk: 3
  max_hops: 4
  min_hops: 2
  min_dims: 2

prompt:
  atomic_library: "src/prompt/atomic_templates.yaml"
  control_tokens: ["CONCISE", "FACTUAL"]

llm:
  provider: "openai"
  api_key_env: "OPENAI_API_KEY"
  lora:
    enabled: false
    quantize:
      enabled: true
      bits: 4
```
```

I/O 规范（关键输出）
- outputs/paths.jsonl
  - 每行 JSON 示例：
    {"user":"u123","item":"i456","paths":[{"nodes":["u123","i789","brand_X","i456"],"rels":["interact","has_brand","same_brand"],"score":0.85,"hops":3}],"meta":{}}
- outputs/prompts.jsonl
  - 每行 JSON 示例：
    {"user":"u123","item":"i456","prompt_struct":{...},"prompt_text":"..."}
- outputs/explanations.jsonl
  - 每行 JSON 示例：
    {"user":"u123","item":"i456","explanation":"...","prompt_id":"p001","paths_ref":["p1","p2"],"llm_stats":{}}
- outputs/eval_report.json
  - 汇总推荐与解释指标、config hash、seed、checkpoint 信息

工程原则（必须遵守）
- 事实约束：LLM 生成必须基于 paths.jsonl 中的实体/关系；禁止引入未列明的实体/属性。
- 可追溯性：保存并版本化所有中间产物（paths/prompts/explanations/checkpoints）。
- 多维多跳：解释路径需满足 hop ≥ 2 且覆盖 ≥ 2 类信息源。
- 模块化与可测试性：每个模块须定义清晰的 I/O（便于单元测试与 CI）。

联系方式与引用
- 作者：wangwenzhang-haha
- 若用于研究请在发表时引用相关论文/技术报告（在 docs/ 中补充引用信息）

License
- MIT（默认，可按需更改）
```