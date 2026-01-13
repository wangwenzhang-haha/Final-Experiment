"""Utilities for checking explanation consistency against path entities."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ASCII_WORD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
ALNUM_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}")


def _is_ascii_word(entity: str) -> bool:
    return bool(ASCII_WORD_RE.match(entity))


def _compile_entity_patterns(entities: Iterable[str]) -> list[tuple[str, str, re.Pattern | str]]:
    compiled: list[tuple[str, str, re.Pattern | str]] = []
    for entity in entities:
        if not entity:
            continue
        if _is_ascii_word(entity):
            pattern = re.compile(rf"\b{re.escape(entity.casefold())}\b")
            compiled.append((entity, "ascii", pattern))
        else:
            compiled.append((entity, "text", entity))
    return compiled


def _entity_in_text(entity: str, explanation_text: str, lower_text: str) -> bool:
    if not entity:
        return False
    if _is_ascii_word(entity):
        pattern = re.compile(rf"\b{re.escape(entity.casefold())}\b")
        return bool(pattern.search(lower_text))
    return entity in explanation_text


def extract_entities_from_explanation(
    text: str,
    candidate_entities: Iterable[str] | None = None,
) -> set[str]:
    """Extract entities from an explanation.

    If candidate_entities is provided, return the subset that appears in text.
    Otherwise, extract simple Chinese or alphanumeric tokens as candidates.
    """

    if candidate_entities is not None:
        lower_text = text.casefold()
        return {
            entity
            for entity in candidate_entities
            if _entity_in_text(entity, text, lower_text)
        }

    entities = set(CHINESE_RE.findall(text))
    entities.update(ALNUM_TOKEN_RE.findall(text))
    return entities


def get_missing_or_extra_entities(
    pred_entities: Iterable[str],
    gold_entities: Iterable[str],
) -> tuple[set[str], set[str]]:
    """Return missing and extra entities compared to gold_entities."""

    gold_set = {entity for entity in gold_entities if entity}
    pred_set = {entity for entity in pred_entities if entity}
    missing = gold_set - pred_set
    extra = pred_set - gold_set
    return missing, extra


def check_explanation_consistency(
    gold_entities: Iterable[str],
    explanation_text: str,
    *,
    return_score: bool = False,
    consider_extra: bool = False,
    candidate_entities: Iterable[str] | None = None,
) -> bool | float:
    """Check whether an explanation is consistent with gold path entities.

    Args:
        gold_entities: Entities from the structured path.
        explanation_text: Generated explanation text.
        return_score: If True, return a float score (matched / total).
        consider_extra: If True, also flag extra entities not in gold_entities.
        candidate_entities: Optional entity pool used for extra-entity detection.

    Returns:
        bool or float score depending on return_score.
    """

    gold_list = [entity for entity in gold_entities if entity]
    if not gold_list:
        return 1.0 if return_score else True

    lower_text = explanation_text.casefold()
    matched = {
        entity
        for entity in gold_list
        if _entity_in_text(entity, explanation_text, lower_text)
    }
    score = len(matched) / len(gold_list)

    consistent = score == 1.0
    if consider_extra:
        pred_entities = extract_entities_from_explanation(
            explanation_text, candidate_entities=candidate_entities
        )
        _, extra = get_missing_or_extra_entities(pred_entities, gold_list)
        consistent = consistent and not extra

    return score if return_score else consistent


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_json(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_explanations_jsonl(
    jsonl_path: Path,
    output_path: Path,
    *,
    output_format: str = "json",
    consider_extra: bool = False,
    candidate_entities: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    """Evaluate explanation consistency in a JSONL file.

    Each JSONL entry should have keys: output (explanation text) and gold_entities.
    """

    records = _load_jsonl(jsonl_path)
    results: list[dict[str, object]] = []
    for record in records:
        explanation_text = str(record.get("output", ""))
        gold_entities = record.get("gold_entities", [])
        matched_entities = extract_entities_from_explanation(
            explanation_text, candidate_entities=gold_entities
        )
        missing, extra = get_missing_or_extra_entities(matched_entities, gold_entities)
        score = check_explanation_consistency(
            gold_entities,
            explanation_text,
            return_score=True,
            consider_extra=consider_extra,
            candidate_entities=candidate_entities,
        )
        is_consistent = check_explanation_consistency(
            gold_entities,
            explanation_text,
            return_score=False,
            consider_extra=consider_extra,
            candidate_entities=candidate_entities,
        )
        results.append(
            {
                "input": record.get("input"),
                "output": explanation_text,
                "gold_entities": list(gold_entities),
                "matched_entities": sorted(matched_entities),
                "missing_entities": sorted(missing),
                "extra_entities": sorted(extra),
                "score": score,
                "is_consistent": is_consistent,
            }
        )

    output_format = output_format.lower()
    if output_format == "csv":
        _write_csv(output_path, results)
    else:
        _write_json(output_path, results)
    return results


def _parse_candidate_entities(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check explanation consistency.")
    parser.add_argument("--input", required=True, help="Path to JSONL file.")
    parser.add_argument("--output", required=True, help="Path to output file.")
    parser.add_argument(
        "--output-format",
        choices=["json", "csv"],
        default="json",
        help="Output format.",
    )
    parser.add_argument(
        "--consider-extra",
        action="store_true",
        help="Flag explanations that mention entities outside gold list.",
    )
    parser.add_argument(
        "--candidate-entities",
        help="Comma-separated entity pool for extra-entity detection.",
    )
    args = parser.parse_args()

    jsonl_path = Path(args.input)
    output_path = Path(args.output)
    candidate_entities = _parse_candidate_entities(args.candidate_entities)

    evaluate_explanations_jsonl(
        jsonl_path,
        output_path,
        output_format=args.output_format,
        consider_extra=args.consider_extra,
        candidate_entities=candidate_entities,
    )


if __name__ == "__main__":
    main()
