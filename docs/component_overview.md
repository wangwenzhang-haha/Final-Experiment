# 组件功能速览（持续更新）

> 该文件用于快速了解各部分代码的职责，后续如有新增模块请同步补充。

## 脚本层（scripts/）
- `scripts/run_demo.py`：一键执行数据加载→推荐→证据检索→解释生成→保存 JSONL 的入口。
- `scripts/preprocess.py`：将原始交互、metadata、常识三元组融合成简单图 `fused_graph.pkl`。
- `scripts/run_toy_pipeline.sh`：最小演示脚本，调用 `run_demo.py` 跑通全流程。

## 解释模块（src/explain/）
- `schema.py`：定义推荐结果、证据、解释、整体输出的数据类与 JSON 模板。
- `retriever.py`：向量检索（基于词袋余弦）与图路径检索（多跳简单路径）。
- `prompt.py`：根据用户、物品、证据拼装受控的 LLM 提示词。
- `explainer.py`：调用 OpenAI API 生成解释，若无 API Key 则使用可追溯的本地 fallback。

## 流水线模块（src/pipeline/）
- `simple_graph.py`：轻量无向图实现，支持添加节点/边和检索简单路径。
- `explainable_pipeline.py`：将数据加载、受欢迎度推荐、证据检索、用户目标推断、解释生成串成完整流程。

## 数据与输出
- `data/raw/`：默认原始 CSV 目录（interactions/items/commonsense_edges）。
- `data/processed/fused_graph.pkl`：预处理后的融合图，供图检索使用。
- `outputs/demo_results.jsonl`：`run_demo.py` 的默认输出，每行一个 `PipelineOutput` 记录。

## 其它
- `configs/demo.yaml`：演示用配置，指定路径、Top-K、检索和 LLM 参数。
- `README.md` & `docs/architecture.md`：项目背景、结构与快速开始说明。
