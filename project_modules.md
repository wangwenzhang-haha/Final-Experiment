# 项目初始模块分类（MVP 分层）与目录建议

下面给出项目的模块化划分、每个模块的职责、输入输出定义、以及首个迭代（MVP）应优先完成的子任务和验收条件。目标是把复杂系统拆成易迭代、可并行开发的模块。

## 1 总体目录结构（建议）
- /data/                   # 原始与处理后数据
- /src/
  - /ingest/               # 数据采集与 ETL
  - /kg/                   # 知识图谱导入、实体链接、图检索
  - /augment/              # LLM 常识生成、数据增强脚本
  - /embed/                # embedding 与向量索引（FAISS/Chroma）
  - /retrieval/            # 组合检索器：向量 + KG path
  - /reasoner/             # 解释生成（prompt 模板、adapter 加载）
  - /trainer/              # LoRA/QLoRA 训练脚本与 utils
  - /eval/                 # 自动评估脚本与人工标注工具
  - /serve/                # 在线推理 API / CLI / docker
  - /ops/                  # deploy 脚本、cloud 启动脚本、monitoring
- /notebooks/              # 试验性 notebook（探索性分析）
- /tests/                  # 单元/集成测试
- requirements.txt
- README.md

---

## 2 模块详细说明（输入/输出 & 优先级）

### 2.1 ingest（优先级：高）
- 职责：
  - 导入原始日志、物品元数据、外部 KG 数据（RDF/CSV/JSON）。
  - 统一 schema，保存为 jsonl。
- 输入：原始日志、原始 KG 文件
- 输出：data/raw.jsonl、data/kg_nodes.jsonl / edges.jsonl
- MVP 任务：
  - 简单 ETL 脚本，把示例数据转换为训练/索引所需格式
- 验收：
  - 能输出 10–100 条标准化 jsonl 样本

### 2.2 kg（知识图谱）（优先级：中高）
- 职责：
  - 实体识别与链接（NER+EL）、KG 导入（Neo4j / NetworkX / DGL 表示）、路径检索接口
- 输入：标准化的实体/元数据
- 输出：KG 查询 API（get_neighbors(node, hops), get_paths(src, tgt, max_hops)）
- MVP 任务：
  - 提供一个轻量 KG 存储（NetworkX 本地内存）并能返回 1~2 hop 的邻居与路径文本化
- 验收：
  - 能对 10 条样例返回 KG 路径并文本化为 human-readable 字符串

### 2.3 augment（优先级：高）
- 职责：
  - 使用大模型（cloud 或本地）生成 commonsense 补充与解释示例
  - 输出增强后的记录（包含 commonsense 字段与 evidence id 列表）
- 输入：data/raw.jsonl
- 输出：data/augmented.jsonl
- MVP 任务：
  - 用 OpenAI 或 HF API 批量生成 1–3 条 commonsense 补充并保存
- 验收：
  - 生成的增强字段 > 90% 非空（人工 spot check）

### 2.4 embed（优先级：高）
- 职责：
  - 将文本（instruction + commonsense）做 embedding 并建立 FAISS 索引
- 输入：data/augmented.jsonl
- 输出：data/faiss_index/ (faiss.index + metas.jsonl)
- MVP 任务：
  - 使用 sentence-transformers 建表并保存 index
- 验收：
  - 能检索到与 query 语义相关的 Top-K 文本

### 2.5 retrieval（优先级：高）
- 职责：
  - 组合检索：先做向量检索再做 KG 路径检索（或并行），返回合并证据集合
- 输入：query、faiss index、KG 接口
- 输出：evidence_list（text id、kg path、score）
- MVP 任务：
  - 实现 retrieve(query) 返回 Top-3 文本 + Top-2 KG 路径
- 验收：
  - 返回结果包含来源 id 与简短摘要

### 2.6 reasoner（推理/解释生成）（优先级：高）
- 职责：
  - 接受 evidence_list，构建 prompt 并调用 LLM（本地量化模型或云）
  - 输出 structured explanation（short, long, chain）
- 输入：evidence_list、prompt template
- 输出：explanation JSON
- MVP 任务：
  - 基于 prompt 工程实现生成并输出 JSON（不做 LoRA）
- 验收：
  - 对 50 个 query，生成解释并至少 80% 符合格式（JSON valid）且能被人工识别为合理

### 2.7 trainer（LoRA 微调）（优先级：中）
- 职责：
  - 训练 LoRA adapter（QLoRA），管理模型版本、adapter 导出/导入
- 输入：train.jsonl（instruction, evidence, chain, target）
- 输出：adapter/ 目录（可加载）
- MVP 任务：
  - 提供 cloud launch script（一次训练能跑通）
- 验收：
  - adapter 能在本地加载并用于推理（格式正确）

### 2.8 eval（优先级：中）
- 职责：
  - 自动评估脚本、人工打标工具、对解释的事实校验器
- 输入：generated explanations, gold annotations
- 输出：metrics report（accuracy, consistency, coverage）
- MVP 任务：
  - 实现简单的 fact-check against KG（断言是否在 KG 中存在）
- 验收：
  - 能输出每条解释的“KG 命中率”

### 2.9 serve / ops（优先级：中）
- 职责：
  - 构建简单的 API（Flask/FastAPI）暴露检索 + 解释接口；docker 化部署脚本
- 输入：请求 query
- 输出：JSON 响应（explanation + evidence）
- MVP 任务：
  - 一个本地可跑的 CLI / HTTP 服务
- 验收：
  - 本地能并发处理 5 QPS（非优化状态下可接受 1–5 QPS）

---

## 3 初始迭代（Sprint 0）任务列表（2–5 天目标）
1. 环境与脚手架
   - 建立 repo 目录，requirements.txt，conda env 说明
2. 数据准备
   - ingest: 准备 100 条样例并标准化
3. KG 快速实现
   - networkx 实现并暴露 get_neighbors/get_paths
4. 增强与索引
   - augment: 用 cloud LLM 生成 commonsense（10–50 条）
   - embed: 建 FAISS 索引并检索 demo
5. 基本推理 pipeline
   - retrieval + reasoner (prompt) 本地 demo，输出 JSON
6. 验收
   - 完成 end-to-end demo：给 query，返回 explanation + evidence

---

## 4 API / 接口规范（简要）
- /api/explain (POST)
  - body: { "query": "...", "topk": 3, "mode": "fast|verbose" }
  - response: { "short": "...", "long": "...", "evidence_chain":[{ "step":"", "source":{"type":"text|kg","id":""},"confidence":0.8, "verified":true }], "meta": {...} }

---

## 5 开发与验收建议
- 每个模块都应包含单元测试（至少输入输出契约测试）。
- 使用小批量真实样本做人工评审回合（每次迭代 50 条）。
- 优先完成可复现的 end-to-end demo（哪怕是远程调用大模型），再逐步把逻辑迁移到本地量化模型 + LoRA adapter。

---

如果你同意此模块划分，我可以：
- 基于上面的目录给出每个模块的 skeleton 代码文件（包含 README 与单元测试 stub）。
- 或者把 Sprint 0 的具体任务拆为 GitHub issue 列表并给出实现优先级与时间估计。你想先要哪个结果？