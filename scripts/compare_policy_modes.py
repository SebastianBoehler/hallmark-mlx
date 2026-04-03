#!/usr/bin/env python3
"""Compare tool-only, policy+deterministic, and policy+generative modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hallmark_mlx.config import load_config
from hallmark_mlx.eval.policy_modes import build_policy_runner
from hallmark_mlx.eval.policy_rollout import evaluate_policy_rollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_qwen_1_5b.yaml")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison: dict[str, dict[str, float | None]] = {}
    for mode_name in ("tool_only", "policy_deterministic", "policy_generative"):
        runner = build_policy_runner(config, mode_name)
        mode_dir = output_dir / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)
        metrics = evaluate_policy_rollout(
            input_path,
            runner,
            output_metrics_path=mode_dir / "metrics.json",
            output_predictions_path=mode_dir / "predictions.jsonl",
            output_traces_path=mode_dir / "traces.jsonl",
            limit=args.limit,
            tool_budgets=config.eval.tool_call_budgets,
        )
        comparison[mode_name] = metrics

    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
