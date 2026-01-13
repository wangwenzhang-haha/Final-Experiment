"""CLI for automatic evaluation of explanation candidates stored in JSONL."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _require_module(module_name: str, purpose: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(
            f"Missing optional dependency '{module_name}' required for {purpose}. "
            "Install it first."
        )


def _parse_metrics(metrics_arg: str) -> List[str]:
    metrics = [metric.strip().lower() for metric in metrics_arg.split(",") if metric.strip()]
    return metrics


@dataclass
class ExplanationSample:
    data: Dict[str, Any]

    @property
    def reference(self) -> str:
        return str(self.data.get("reference", ""))

    @property
    def candidate(self) -> str:
        return str(self.data.get("candidate", ""))

    @property
    def gold_entities(self) -> List[str]:
        entities = self.data.get("gold_entities", [])
        if isinstance(entities, list):
            return [str(entity) for entity in entities]
        return []


class ExplanationEvaluator:
    def __init__(self, metrics: Sequence[str]) -> None:
        self.metrics = set(metrics)

    def compute_bertscore(self, samples: Sequence[ExplanationSample]) -> List[float]:
        _require_module("bert_score", "BERTScore evaluation")
        from bert_score import BERTScorer

        references = [sample.reference for sample in samples]
        candidates = [sample.candidate for sample in samples]
        scorer = BERTScorer(lang="zh", rescale_with_baseline=True)
        _precision, _recall, f1 = scorer.score(candidates, references)
        return f1.tolist()

    def compute_bleurt(self, samples: Sequence[ExplanationSample]) -> List[float]:
        _require_module("evaluate", "BLEURT evaluation")
        import evaluate

        references = [sample.reference for sample in samples]
        candidates = [sample.candidate for sample in samples]
        metric = evaluate.load("bleurt")
        results = metric.compute(predictions=candidates, references=references)
        return list(results["scores"])

    @staticmethod
    def compute_entity_match(samples: Sequence[ExplanationSample]) -> List[float]:
        scores = []
        for sample in samples:
            gold_entities = sample.gold_entities
            if not gold_entities:
                scores.append(0.0)
                continue
            matched = sum(1 for entity in gold_entities if entity in sample.candidate)
            scores.append(matched / len(gold_entities))
        return scores

    def compute_gpt_judge(self, samples: Sequence[ExplanationSample], model: str) -> List[Dict[str, Any]]:
        _require_module("openai", "GPT evaluation")
        import openai

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY must be set to use GPT evaluation.")

        openai.api_key = os.environ["OPENAI_API_KEY"]
        responses: List[Dict[str, Any]] = []
        for sample in samples:
            prompt = (
                "请根据清晰度、连贯性和事实正确性评价以下推荐解释。\n"
                f"参考解释：{sample.reference}\n"
                f"候选解释：{sample.candidate}\n"
                "请给出1-5分评分，并简要说明理由。"
            )
            if hasattr(openai, "OpenAI"):
                client = openai.OpenAI()
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = completion.choices[0].message.content
            else:
                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = completion["choices"][0]["message"]["content"]
            responses.append({"gpt_judge": content})
        return responses

    def evaluate(self, samples: Sequence[ExplanationSample], gpt_model: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = [dict(sample.data) for sample in samples]
        if "bertscore" in self.metrics:
            scores = self.compute_bertscore(samples)
            for result, score in zip(results, scores):
                result["bertscore"] = float(score)
        if "bleurt" in self.metrics:
            scores = self.compute_bleurt(samples)
            for result, score in zip(results, scores):
                result["bleurt"] = float(score)
        if "entity" in self.metrics:
            scores = self.compute_entity_match(samples)
            for result, score in zip(results, scores):
                result["entity_match_score"] = float(score)
        if "gpt" in self.metrics:
            judgments = self.compute_gpt_judge(samples, gpt_model)
            for result, judge in zip(results, judgments):
                result.update(judge)
        return results


def _load_samples(path: str) -> List[ExplanationSample]:
    samples: List[ExplanationSample] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            samples.append(ExplanationSample(data=data))
    return samples


def _write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _write_summary(path: str, records: Sequence[Dict[str, Any]], group_by: Sequence[str]) -> None:
    _require_module("pandas", "summary CSV report generation")
    import pandas as pd

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No records to summarize.")
    metric_columns = [
        column
        for column in ["bertscore", "bleurt", "entity_match_score"]
        if column in df.columns
    ]
    if not metric_columns:
        raise RuntimeError("No metric columns found for summary output.")
    summary = df.groupby(list(group_by), dropna=False)[metric_columns].mean().reset_index()
    summary.to_csv(path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate explanation candidates stored in JSONL.")
    parser.add_argument("--input", required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 文件路径")
    parser.add_argument(
        "--metrics",
        default="bertscore,bleurt,entity",
        help="用逗号分隔的评估指标列表 (bertscore, bleurt, entity, gpt)",
    )
    parser.add_argument(
        "--summary-output",
        help="可选：输出汇总 CSV 报告路径",
    )
    parser.add_argument(
        "--group-by",
        default="model,prompt_type",
        help="汇总 CSV 的分组字段 (逗号分隔)",
    )
    parser.add_argument(
        "--gpt-model",
        default="gpt-4o-mini",
        help="启用 GPT 评测时使用的模型名称",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    metrics = _parse_metrics(args.metrics)
    if not metrics:
        raise ValueError("至少指定一个评估指标。")

    samples = _load_samples(args.input)
    evaluator = ExplanationEvaluator(metrics=metrics)
    results = evaluator.evaluate(samples, gpt_model=args.gpt_model)
    _write_jsonl(args.output, results)

    if args.summary_output:
        group_by = [value.strip() for value in args.group_by.split(",") if value.strip()]
        if not group_by:
            raise ValueError("group-by 字段不能为空。")
        _write_summary(args.summary_output, results, group_by)


if __name__ == "__main__":
    main()
