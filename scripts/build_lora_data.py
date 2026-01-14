#!/usr/bin/env python3
"""
Build LoRA fine-tuning data from structured recommendation graph artifacts.

Input sources:
- CSV: header with fields like user_profile,item_profile,path_texts,gold_explanation
- JSON: list of objects
- JSONL: one object per line

Output:
- JSONL with schema: {"input": "<USER>: ...\n<ITEM>: ...\n<EVIDENCE>: ...", "output": "..."}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Dict, Any, List


DEFAULT_USER_FIELD = "user_profile"
DEFAULT_ITEM_FIELD = "item_profile"
DEFAULT_PATH_FIELD = "path_texts"
DEFAULT_EXPLANATION_FIELD = "gold_explanation"

def _normalize_text(value: Any, joiner: str) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return joiner.join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _infer_format(path: Path, override: str | None) -> str:
    if override:
        return override.lower()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"
    raise ValueError(f"Unsupported input format for {path}. Use --format to specify.")


def _read_csv(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def _read_json(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("JSON input must be a list of objects.")
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("Each JSON entry must be an object.")
        yield row


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_num}.") from exc
            if not isinstance(row, dict):
                raise ValueError("Each JSONL entry must be an object.")
            yield row


def _build_sample(
    record: Dict[str, Any],
    user_field: str,
    item_field: str,
    path_field: str,
    explanation_field: str,
    joiner: str,
) -> Dict[str, str]:
    user_profile = _normalize_text(record.get(user_field), joiner)
    item_profile = _normalize_text(record.get(item_field), joiner)
    path_texts = _normalize_text(record.get(path_field), joiner)
    explanation = _normalize_text(record.get(explanation_field), joiner)

    prompt = "\n".join(
        [
            f"<USER>: {user_profile}",
            f"<ITEM>: {item_profile}",
            f"<EVIDENCE>: {path_texts}",
        ]
    )

    return {"input": prompt.strip(), "output": explanation.strip()}


def _load_records(path: Path, fmt: str) -> Iterable[Dict[str, Any]]:
    if fmt == "csv":
        return _read_csv(path)
    if fmt == "json":
        return _read_json(path)
    if fmt == "jsonl":
        return _read_jsonl(path)
    raise ValueError(f"Unsupported format: {fmt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build LoRA fine-tuning JSONL data from structured recommendation inputs."
    )
    parser.add_argument("--input", required=True, help="Path to CSV/JSON/JSONL input.")
    parser.add_argument("--output", required=True, help="Path to output JSONL file.")
    parser.add_argument(
        "--format",
        choices=["csv", "json", "jsonl"],
        help="Input format (auto-infer from suffix if omitted).",
    )
    parser.add_argument("--user-field", default=DEFAULT_USER_FIELD)
    parser.add_argument("--item-field", default=DEFAULT_ITEM_FIELD)
    parser.add_argument("--path-field", default=DEFAULT_PATH_FIELD)
    parser.add_argument("--explanation-field", default=DEFAULT_EXPLANATION_FIELD)
    parser.add_argument(
        "--joiner",
        default=" ",
        help="Joiner used when path_texts/user/item is a list.",
    )
    parser.add_argument(
        "--drop-missing",
        action="store_true",
        help="Skip records with empty explanation or prompt fields.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    fmt = _infer_format(input_path, args.format)
    records = _load_records(input_path, fmt)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            sample = _build_sample(
                record,
                user_field=args.user_field,
                item_field=args.item_field,
                path_field=args.path_field,
                explanation_field=args.explanation_field,
                joiner=args.joiner,
            )
            if args.drop_missing and (not sample["input"] or not sample["output"]):
                skipped += 1
                continue
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    sys.stderr.write(
        f"Saved {written} samples to {output_path} (skipped {skipped}).\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
