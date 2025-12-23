# 数据流向与工作流程（持续更新）

本节按执行顺序说明 demo 的关键步骤、数据形态以及涉及的文件。

1. **生成/准备原始数据**  
   - 入口：`scripts/run_demo.py` 调用 `ensure_toy_data`。  
   - 依赖：`scripts/generate_toy_data.py`（自动生成交互/物品/commonsense CSV）。

2. **构建融合图**  
   - 入口：`scripts/run_demo.py` 内的 `ensure_fused_graph`。  
   - 实现：`scripts/preprocess.py` 的 `build_fused_graph` 使用 `src/pipeline/simple_graph.py`。  
   - 输入：`data/raw/interactions.csv`, `data/raw/items.csv`, `data/raw/commonsense_edges.csv`。  
   - 输出：`data/processed/fused_graph.pkl`（用于图检索）。

3. **加载数据**  
   - 模块：`src/pipeline/explainable_pipeline.py` 的 `DatasetLoader`。  
   - 读取：交互/物品 CSV + 上一步的融合图。

4. **推荐生成（Baseline）**  
   - 模块：`PopularityRecommender`（`explainable_pipeline.py`）。  
   - 输入：训练交互，过滤用户已看，输出 Top-K `Recommendation` 列表。

5. **用户目标/画像提取**  
   - 模块：`UserGoalInferer`（`explainable_pipeline.py`）。  
   - 输入：用户历史 item_id，匹配 `items.csv` 分类/品牌，形成摘要。

6. **证据检索**  
   - 向量检索：`VectorEvidenceRetriever`（`src/explain/retriever.py`）基于物品文本词袋计算相似 item。  
   - 图路径：`GraphEvidenceFinder`（`src/explain/retriever.py`）在 `SimpleGraph` 上寻找用户→物品多跳路径。  
   - 输出：metadata/相似度线索 + KG 路径，封装为 `Evidence`。

7. **提示词与解释生成**  
   - 提示词：`build_explanation_prompt`（`src/explain/prompt.py`）组合用户画像、目标物品描述、路径文本。  
   - 解释：`ExplanationGenerator`（`src/explain/explainer.py`）调用 LLM 或 fallback，产出 `Explanation`。

8. **结果组装与持久化**  
   - 结构：`PipelineOutput`（`src/explain/schema.py`）整合用户、推荐列表、证据、解释。  
   - 序列化：`save_results`（`src/pipeline/explainable_pipeline.py`）写入 `outputs/demo_results.jsonl`。

运行命令示例：
```bash
python scripts/run_demo.py --dataset toy --k 10 --explain --config configs/demo.yaml
```
