"""Helpers for compare32 evaluation of hallmark-mlx methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hallmark_mlx.config import load_config
from hallmark_mlx.eval.policy_modes import build_policy_runner
from hallmark_mlx.eval.policy_rollout import evaluate_policy_rollout
from hallmark_mlx.utils.io import read_json, write_json

JSONDict = dict[str, Any]


def _tracked_row(name: str, description: str, metrics: dict[str, Any]) -> JSONDict:
    return {
        "name": name,
        "group": "hallmark_mlx",
        "source": "executed_here",
        "description": description,
        "available_here": True,
        "status_message": "Executed here on tracked compare32 split.",
        "split": "compare32",
        "num_entries": metrics["num_examples"],
        "coverage": 1.0,
        "partial": False,
        "notes": "",
        "detection_rate": metrics["detection_rate"],
        "false_positive_rate": metrics["false_positive_rate"],
        "f1_hallucination": metrics["f1_hallucinated"],
        "tier_weighted_f1": None,
        "mcc": None,
        "ece": None,
        "tier3_f1": None,
        "coverage_adjusted_f1": metrics["f1_hallucinated"],
        "mean_api_calls": metrics["avg_tool_calls"],
        "num_uncertain": metrics["num_uncertain_predictions"],
        "union_recall_at_k": {},
        "completion_rate": metrics["completion_rate"],
        "tool_use_rate": metrics["tool_use_rate"],
        "first_turn_tool_call_rate": metrics["first_turn_tool_call_rate"],
        "label_accuracy": metrics["label_accuracy"],
        "budget_2_f1_hallucinated": metrics["budget_2_f1_hallucinated"],
        "budget_4_f1_hallucinated": metrics["budget_4_f1_hallucinated"],
    }


def run_compare32_row(
    *,
    config_path: Path,
    mode_name: str,
    compare32_gold_path: Path,
    method_name: str,
    description: str,
    output_dir: Path,
    rerun: bool,
    adapter_path: Path | None = None,
) -> JSONDict:
    """Run one hallmark-mlx method on the tracked compare32 split."""

    row_path = output_dir / "row.json"
    if row_path.exists() and not rerun:
        return read_json(row_path)
    config = load_config(config_path)
    if adapter_path is not None:
        config.model.adapter_path = adapter_path.resolve()
    runner = build_policy_runner(config, mode_name)
    metrics = evaluate_policy_rollout(
        compare32_gold_path,
        runner,
        output_metrics_path=output_dir / "metrics.json",
        output_predictions_path=output_dir / "predictions.jsonl",
        output_traces_path=output_dir / "traces.jsonl",
        tool_budgets=config.eval.tool_call_budgets,
    )
    row = _tracked_row(method_name, description, metrics)
    write_json(row_path, row)
    return row
