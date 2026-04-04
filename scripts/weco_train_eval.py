#!/usr/bin/env python3
"""Train one Weco-edited Qwen policy trial and score its rollout frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hallmark_mlx.eval.frontier import (
    collect_frontier_metrics,
    collect_guarded_frontier_metrics,
)
from hallmark_mlx.eval.metrics import DEFAULT_TOOL_BUDGETS
from hallmark_mlx.eval.policy_rollout import evaluate_policy_rollout
from hallmark_mlx.training.mlx_lora import plan_training_run, run_training
from hallmark_mlx.types import FinalizationMode
from hallmark_mlx.utils.io import ensure_dir, write_json
from hallmark_mlx.weco_support import (
    build_weco_runner,
    format_metric_lines,
    load_trial_spec,
    materialize_trial_config,
    trial_run_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARE32_PATH = ROOT / "data" / "weco" / "hallmark_dev_compare32_gold_traces.jsonl"
BASELINE_COMPARE32_METRICS_PATH = (
    ROOT
    / "artifacts"
    / "weco"
    / "hallmark-qwen-train-frontier-iterative"
    / "61460529b25c"
    / "hallmark_dev_compare32_gold_traces"
    / "metrics.json"
)
FIXED_TRIAL_NAME = "hallmark-qwen-train-frontier-guarded"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate one Weco-edited hallmark-mlx Qwen policy trial.",
    )
    parser.add_argument("--source", required=True, help="Editable Weco trial source file.")
    parser.add_argument(
        "--eval-input-path",
        help="Optional override for the tracked evaluation trace JSONL.",
    )
    return parser


def _adapter_ready(adapter_path: Path) -> bool:
    return (adapter_path / "adapters.safetensors").exists()


def _cache_summary(summary_path: Path) -> dict[str, object] | None:
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_train_trial(config_path: Path, trial_spec, config) -> None:
    if trial_spec.policy_mode != "policy_deterministic":
        raise SystemExit("Qwen Weco train trials must keep POLICY_MODE=policy_deterministic.")
    if trial_spec.trial_name != FIXED_TRIAL_NAME:
        raise SystemExit(f"Qwen Weco train trials must keep TRIAL_NAME={FIXED_TRIAL_NAME}.")
    if config.model.temperature != 0.0:
        raise SystemExit("Qwen Weco train trials must keep model.temperature=0.0.")
    if config.model.finalization_mode != FinalizationMode.DETERMINISTIC:
        raise SystemExit("Qwen Weco train trials must keep deterministic finalization.")
    if not config.model.force_bibtex_updater_first:
        raise SystemExit("Qwen Weco train trials must keep BibTeX-first routing enabled.")
    if not 384 <= config.model.max_tokens <= 768:
        raise SystemExit("model.max_tokens must stay within [384, 768].")
    if not 3 <= config.model.max_rollout_rounds <= 5:
        raise SystemExit("model.max_rollout_rounds must stay within [3, 5].")

    train = config.training
    if not 10 <= train.num_layers <= 14:
        raise SystemExit("training.num_layers must stay within [10, 14].")
    if not 7e-5 <= train.learning_rate <= 1.5e-4:
        raise SystemExit("training.learning_rate must stay within [7e-5, 1.5e-4].")
    if not 80 <= train.num_iterations <= 160:
        raise SystemExit("training.num_iterations must stay within [80, 160].")
    if not 4096 <= train.max_seq_length <= 6144:
        raise SystemExit("training.max_seq_length must stay within [4096, 6144].")
    if not 4 <= train.grad_accumulation_steps <= 8:
        raise SystemExit("training.grad_accumulation_steps must stay within [4, 8].")
    if train.seed != 7:
        raise SystemExit("training.seed must stay fixed at 7 for guarded Weco trials.")

    tool_names = (
        "crossref",
        "openalex",
        "dblp",
        "acl_anthology",
        "arxiv",
        "semantic_scholar",
    )
    for name in tool_names:
        service = getattr(config.tools, name)
        if not service.enabled:
            raise SystemExit(f"Tool {name} must remain enabled in guarded Weco trials.")
        if not 1 <= service.rows <= 5:
            raise SystemExit(f"Tool {name} rows must stay within [1, 5].")

    if config_path.name != "hallmark-qwen-train-frontier-guarded.yaml":
        raise SystemExit(
            "Guarded Weco trials must materialize to "
            "hallmark-qwen-train-frontier-guarded.yaml."
        )


def _evaluate_split(
    eval_input_path: Path,
    eval_dir: Path,
    runner,
) -> dict[str, float]:
    metrics_path = eval_dir / "metrics.json"
    metrics = _load_json(metrics_path) if metrics_path.exists() else None
    if metrics is None:
        metrics = evaluate_policy_rollout(
            eval_input_path,
            runner,
            output_metrics_path=metrics_path,
            output_predictions_path=eval_dir / "predictions.jsonl",
            output_traces_path=eval_dir / "traces.jsonl",
            tool_budgets=DEFAULT_TOOL_BUDGETS,
        )
    split_summary_path = eval_dir / "summary.json"
    write_json(
        split_summary_path,
        {
            "eval_input_path": str(eval_input_path),
            "metrics": collect_frontier_metrics(metrics),
        },
    )
    return {key: float(value) for key, value in metrics.items()}


def main() -> None:
    args = build_parser().parse_args()
    trial_spec = load_trial_spec(args.source)
    eval_input_path = (
        Path(args.eval_input_path).resolve()
        if args.eval_input_path is not None
        else trial_spec.eval_input_path
    )
    if not eval_input_path.exists():
        raise SystemExit(f"Tracked eval input does not exist: {eval_input_path}")
    compare32_path = DEFAULT_COMPARE32_PATH.resolve()
    if args.eval_input_path is None and not compare32_path.exists():
        raise SystemExit(f"Tracked compare32 input does not exist: {compare32_path}")
    if args.eval_input_path is None and not BASELINE_COMPARE32_METRICS_PATH.exists():
        raise SystemExit(
            f"Guarded compare32 baseline metrics missing: {BASELINE_COMPARE32_METRICS_PATH}"
        )

    materialized_root = trial_spec.source_path.parent.parent / "artifacts" / "weco" / "materialized"
    base_config, materialized_path = materialize_trial_config(trial_spec, materialized_root)
    _validate_train_trial(materialized_path, trial_spec, base_config)
    run_id = trial_run_fingerprint(materialized_path)
    run_root = ensure_dir(base_config.weco.experiment_dir / trial_spec.trial_name / run_id)
    eval_dir = ensure_dir(run_root / eval_input_path.stem)
    summary_path = (
        run_root / "summary.json"
        if args.eval_input_path is None
        else eval_dir / "summary.json"
    )
    cached = _cache_summary(summary_path)
    if cached is not None:
        metrics = cached.get("metrics", {})
        if isinstance(metrics, dict):
            for line in format_metric_lines({k: float(v) for k, v in metrics.items()}):
                print(line)
        print(f"materialized_config_path: {materialized_path}")
        print(f"output_dir: {eval_dir}")
        return

    config = base_config.model_copy(
        update={
            "model": base_config.model.model_copy(update={"adapter_path": run_root / "adapter"}),
            "training": base_config.training.model_copy(update={"output_dir": run_root / "train"}),
            "weco": base_config.weco.model_copy(update={"enabled": False}),
        }
    )
    plan = plan_training_run(config)
    if not _adapter_ready(config.model.adapter_path):
        run_training(config)

    runner = build_weco_runner(config, trial_spec.policy_mode)
    metrics = _evaluate_split(eval_input_path, eval_dir, runner)
    frontier_metrics = collect_frontier_metrics(metrics)
    if args.eval_input_path is None:
        compare_dir = ensure_dir(run_root / compare32_path.stem)
        compare_metrics = _evaluate_split(compare32_path, compare_dir, runner)
        frontier_metrics = collect_guarded_frontier_metrics(
            metrics,
            compare_metrics,
            _load_json(BASELINE_COMPARE32_METRICS_PATH),
        )
    write_json(
        summary_path,
        {
            "trial_name": trial_spec.trial_name,
            "policy_mode": trial_spec.policy_mode,
            "source_path": str(trial_spec.source_path),
            "materialized_config_path": str(materialized_path),
            "eval_input_path": str(eval_input_path),
            "compare32_input_path": str(compare32_path) if args.eval_input_path is None else None,
            "run_id": run_id,
            "train_manifest_path": str(plan.manifest_path),
            "adapter_path": str(config.model.adapter_path),
            "metrics": frontier_metrics,
        },
    )
    for line in format_metric_lines(frontier_metrics):
        print(line)
    print(f"materialized_config_path: {materialized_path}")
    print(f"output_dir: {eval_dir}")


if __name__ == "__main__":
    main()
