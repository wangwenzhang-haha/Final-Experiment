# 项目运行流程与参与文件

> 本文档说明从数据准备到解释生成的全流程，并列出每一步涉及的主要脚本与模块文件。

## 1. 环境准备

**目的**：安装运行所需依赖与（可选）GPU 支持。

**相关文件**
- `requirements.txt`（Python 依赖列表）
- `README.md`（快速开始说明）

## 2. 数据准备与输入规范

**目的**：准备交互、元数据、常识 KG 三类数据。

**相关文件**
- `scripts/preprocess.py`（数据清洗与融合入口）
- `src/data/`（数据读取、映射、格式化）
- 约定输入（示例）：
  - `data/raw/interactions.csv`
  - `data/raw/items.csv`
  - `data/raw/commonsense_edges.csv`

## 3. 预处理与融合图构建

**目的**：将多源数据融合为可训练图结构。

**相关文件**
- `scripts/preprocess.py`
- `src/graph/`（图构建、对齐、剪枝）
- 输出示例：`data/processed/fused_graph.pkl`

## 4. 推荐模型训练

**目的**：训练基础推荐模型（如 LightGCN/GCN/GraphSAGE）。

**相关文件**
- `scripts/train_reco.py`
- `src/model/`（推荐模型定义）
- `config/default.yaml`（训练配置）
- 输出示例：`outputs/checkpoints/reco/`

## 5.（可选）Edge-mask 学习与剪枝

**目的**：学习边重要性并进行图剪枝以提取可靠证据。

**相关文件**
- `scripts/learn_edge_mask.py`
- `src/graph/`（edge-mask、m-core 剪枝逻辑）
- 输出示例：`outputs/edge_masks/`

## 6. 多跳路径抽取

**目的**：在融合图上抽取多维多跳路径证据。

**相关文件**
- `scripts/extract_paths.py`
- `src/path/`（路径搜索、评分）
- 输出示例：`outputs/paths.jsonl`

## 7. Prompt 构建

**目的**：将路径证据转换为可控 Prompt。

**相关文件**
- `scripts/build_prompts.py`
- `src/prompt/`（原子模板库与 Prompt Builder）
- 输出示例：`outputs/prompts.jsonl`

## 8. 解释生成（LLM）

**目的**：使用 LLM 基于路径证据生成解释。

**相关文件**
- `scripts/generate_explanations.py`
- `src/llm/`（LLM 接口与后处理）
- 输出示例：`outputs/explanations.jsonl`

## 9. 评估

**目的**：评估推荐效果与解释质量。

**相关文件**
- `scripts/evaluate.py`
- `src/eval/`（指标计算）
- 输出示例：`outputs/eval_report.json`

## 10. 端到端玩具示例

**目的**：快速验证全流程是否可运行。

**相关文件**
- `scripts/run_toy_pipeline.sh`（端到端脚本）
- `scripts/run_demo.py`（等价入口）
- `configs/demo.yaml`
- 输出示例：`outputs/demo_results.jsonl`
