#!/usr/bin/env python3
"""Merge shard outputs from official controller evals and score them officially."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hallmark_mlx.eval.upstream_hallmark import (
    _ensure_upstream_path,
    load_entries,
    result_to_row,
)
from hallmark_mlx.utils.io import read_json, write_json
from hallmark_mlx.utils.jsonl import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", default="/tmp/hallmark-upstream")
    parser.add_argument("--split", default="dev_public")
    parser.add_argument("--shards-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", default="hallmark_mlx_bibtex_first_fallback")
    parser.add_argument(
        "--description",
        default="hallmark-mlx controller with deterministic finalizer",
    )
    return parser


def _load_shard_dirs(shards_root: Path) -> list[Path]:
    return sorted(path for path in shards_root.iterdir() if path.is_dir())


def main() -> None:
    args = build_parser().parse_args()
    upstream_root = Path(args.upstream_root).resolve()
    shards_root = Path(args.shards_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = _load_shard_dirs(shards_root)
    trace_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    shard_progress: list[dict[str, object]] = []
    for shard_dir in shard_dirs:
        progress_path = shard_dir / "progress.json"
        if progress_path.exists():
            shard_progress.append(read_json(progress_path))
        trace_rows.extend(read_jsonl(shard_dir / "traces.jsonl"))
        prediction_rows.extend(read_jsonl(shard_dir / "predictions.jsonl"))

    write_jsonl(output_dir / "traces.jsonl", trace_rows)
    write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    write_json(output_dir / "merge_manifest.json", {"shards": shard_progress})

    _ensure_upstream_path(upstream_root)
    from hallmark.dataset.schema import Prediction
    from hallmark.evaluation.metrics import evaluate

    entries = list(load_entries(upstream_root, split=args.split))
    predictions = [Prediction.from_dict(row) for row in prediction_rows]
    result = evaluate(
        entries=entries,
        predictions=predictions,
        tool_name=args.method_name,
        split_name=args.split,
    )
    write_json(output_dir / "result.json", result.to_dict())
    row = result_to_row(
        result,
        source="executed_here",
        description=args.description,
        available_here=True,
        status_message="Executed here with official evaluator from merged shard outputs.",
        notes=f"Merged {len(shard_dirs)} shard directories from {shards_root}.",
    )
    write_json(output_dir / "row.json", row)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
