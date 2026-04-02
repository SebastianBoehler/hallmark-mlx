"""Evaluation entrypoints."""

from __future__ import annotations

from pathlib import Path

from hallmark_mlx.eval.metrics import compute_metrics
from hallmark_mlx.utils.io import write_json
from hallmark_mlx.utils.jsonl import read_jsonl


def run_local_eval(
    predictions_path: str | Path,
    gold_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, float]:
    """Evaluate local predictions against gold JSONL."""

    prediction_rows = read_jsonl(predictions_path)
    gold_rows = read_jsonl(gold_path)
    metrics = compute_metrics(gold_rows, prediction_rows)
    if output_path is not None:
        write_json(output_path, metrics)
    return metrics
