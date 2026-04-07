"""Merge sharded official evaluation outputs and score them officially."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hallmark_mlx.eval.upstream_hallmark import (
    _ensure_upstream_path,
    load_entries,
    result_to_row,
)
from hallmark_mlx.utils.io import read_json, write_json
from hallmark_mlx.utils.jsonl import read_jsonl, write_jsonl


def load_shard_dirs(shards_root: Path) -> list[Path]:
    """Return sorted shard directories under one shard root."""

    return sorted(path for path in shards_root.iterdir() if path.is_dir())


def merge_official_eval_shards(
    *,
    upstream_root: Path,
    split: str,
    shards_root: Path,
    output_dir: Path,
    method_name: str,
    description: str,
) -> dict[str, Any]:
    """Merge shard outputs and rescore them with the upstream official evaluator."""

    upstream_root = upstream_root.resolve()
    shards_root = shards_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = load_shard_dirs(shards_root)
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

    entries = list(load_entries(upstream_root, split=split))
    predictions = [Prediction.from_dict(row) for row in prediction_rows]
    result = evaluate(
        entries=entries,
        predictions=predictions,
        tool_name=method_name,
        split_name=split,
    )
    write_json(output_dir / "result.json", result.to_dict())
    row = result_to_row(
        result,
        source="executed_here",
        description=description,
        available_here=True,
        status_message="Executed here with official evaluator from merged shard outputs.",
        notes=f"Merged {len(shard_dirs)} shard directories from {shards_root}.",
    )
    write_json(output_dir / "row.json", row)
    return row
