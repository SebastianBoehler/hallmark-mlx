#!/usr/bin/env python3
"""Run one repo-native Weco trial and print frontier metrics."""

from __future__ import annotations

import argparse

from hallmark_mlx.eval.frontier import collect_frontier_metrics
from hallmark_mlx.eval.policy_rollout import evaluate_policy_rollout
from hallmark_mlx.utils.io import write_json
from hallmark_mlx.weco_support import (
    build_weco_runner,
    format_metric_lines,
    load_trial_spec,
    materialize_trial_config,
    trial_output_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one Weco-edited hallmark-mlx trial source.",
    )
    parser.add_argument("--source", required=True, help="Editable Weco trial source file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trial_spec = load_trial_spec(args.source)
    if not trial_spec.eval_input_path.exists():
        raise SystemExit(f"Tracked eval input does not exist: {trial_spec.eval_input_path}")
    materialized_root = trial_spec.source_path.parent.parent / "artifacts" / "weco" / "materialized"
    config, materialized_path = materialize_trial_config(
        trial_spec,
        materialized_root,
    )
    runner = build_weco_runner(config, trial_spec.policy_mode)
    output_dir = trial_output_dir(config, trial_spec)
    metrics = evaluate_policy_rollout(
        trial_spec.eval_input_path,
        runner,
        output_metrics_path=output_dir / "metrics.json",
        output_predictions_path=output_dir / "predictions.jsonl",
        output_traces_path=output_dir / "traces.jsonl",
        tool_budgets=config.eval.tool_call_budgets,
    )
    frontier_metrics = collect_frontier_metrics(metrics)
    write_json(
        output_dir / "summary.json",
        {
            "trial_name": trial_spec.trial_name,
            "policy_mode": trial_spec.policy_mode,
            "source_path": str(trial_spec.source_path),
            "materialized_config_path": str(materialized_path),
            "eval_input_path": str(trial_spec.eval_input_path),
            "metrics": frontier_metrics,
        },
    )
    for line in format_metric_lines(frontier_metrics):
        print(line)
    print(f"materialized_config_path: {materialized_path}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
