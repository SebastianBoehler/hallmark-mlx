#!/usr/bin/env python3
"""Run HALLMARK baseline comparisons against hallmark-mlx methods."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from hallmark_mlx.config import load_config
from hallmark_mlx.eval.compare_rows import (
    failed_execution_row,
    merge_registry_rows,
)
from hallmark_mlx.eval.official_compare import evaluate_runner_on_entries
from hallmark_mlx.eval.policy_modes import build_policy_runner
from hallmark_mlx.eval.timeouts import run_with_timeout
from hallmark_mlx.eval.tracked_compare import run_compare32_row
from hallmark_mlx.eval.upstream_hallmark import (
    list_registry_rows,
    load_entries,
    load_history_rows,
    load_published_rows,
    result_to_row,
    run_baseline,
)
from hallmark_mlx.utils.io import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM_ROOT = Path("/tmp/hallmark-upstream")
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "hallmark_baseline_compare"
COMPARE32_GOLD_PATH = ROOT / "data" / "weco" / "hallmark_dev_compare32_gold_traces.jsonl"
FAST_LOCAL_BASELINES = (
    "doi_presence_heuristic",
    "title_oracle",
    "random",
    "always_hallucinated",
    "always_valid",
    "venue_oracle",
)
EXPENSIVE_LOCAL_BASELINES = (
    "doi_only_no_prescreening",
    "bibtexupdater_no_prescreening",
    "harc_no_prescreening",
)
FINETUNED_MODELS = (
    (
        "hallmark_mlx_qwen_round7_compare32",
        "Qwen 1.5B LoRA round 7",
        ROOT / "artifacts" / "adapters" / "qwen25_1_5b_round7",
    ),
    (
        "hallmark_mlx_qwen_round8_compare32",
        "Qwen 1.5B LoRA round 8",
        ROOT / "artifacts" / "adapters" / "qwen25_1_5b_round8",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", default=str(DEFAULT_UPSTREAM_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--include-expensive", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser


def _load_executed_baseline_rows(
    upstream_root: Path,
    output_dir: Path,
    registry_map: dict[str, dict[str, Any]],
    rerun: bool,
    baseline_names: tuple[str, ...],
    timeout_seconds: int,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for baseline_name in baseline_names:
        row_path = output_dir / baseline_name / "row.json"
        if row_path.exists() and not rerun:
            rows[baseline_name] = read_json(row_path)
            continue
        info = registry_map.get(baseline_name, {})
        try:
            predictions, result = run_with_timeout(
                timeout_seconds,
                lambda baseline_name=baseline_name: run_baseline(
                    upstream_root,
                    baseline_name=baseline_name,
                    split="dev_public",
                ),
            )
            row = result_to_row(
                result,
                source="executed_here",
                description=str(info.get("description", baseline_name)),
                available_here=bool(info.get("available_here", True)),
                status_message="Executed here as a local baseline.",
                notes=f"Generated {len(predictions)} predictions locally.",
            )
        except Exception as exc:
            row = failed_execution_row(
                info or {"name": baseline_name, "description": baseline_name},
                notes=str(exc),
            )
        row["group"] = "upstream_baseline"
        write_json(row_path, row)
        rows[baseline_name] = row
    return rows


def _run_dev_public_controller_row(
    upstream_root: Path,
    output_dir: Path,
    rerun: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    row_path = output_dir / "row.json"
    if row_path.exists() and not rerun:
        return read_json(row_path)
    try:
        row = run_with_timeout(
            timeout_seconds,
            lambda: _evaluate_controller_row(upstream_root, output_dir),
        )
    except Exception as exc:
        row = failed_execution_row(
            {
                "name": "hallmark_mlx_bibtex_first_fallback",
                "description": "hallmark-mlx controller with deterministic finalizer",
                "available_here": True,
            },
            notes=str(exc),
        )
    row["group"] = "hallmark_mlx"
    write_json(row_path, row)
    return row


def _evaluate_controller_row(upstream_root: Path, output_dir: Path) -> dict[str, Any]:
    config = load_config(ROOT / "configs" / "base.yaml")
    runner = build_policy_runner(config, "bibtex_first_fallback")
    entries = list(load_entries(upstream_root, split="dev_public"))
    return evaluate_runner_on_entries(
        upstream_root=upstream_root,
        entries=entries,
        split_name="dev_public",
        runner=runner,
        method_name="hallmark_mlx_bibtex_first_fallback",
        output_dir=output_dir,
        description="hallmark-mlx controller with deterministic finalizer",
        source="executed_here",
        resume=True,
        entry_timeout_seconds=60,
        progress_every=10,
    )


def main() -> None:
    args = build_parser().parse_args()
    upstream_root = Path(args.upstream_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    registry_rows = list_registry_rows(upstream_root)
    registry_map = {row["name"]: row for row in registry_rows}
    baseline_names = FAST_LOCAL_BASELINES + (
        EXPENSIVE_LOCAL_BASELINES if args.include_expensive else ()
    )
    executed_rows = _load_executed_baseline_rows(
        upstream_root,
        output_dir / "local_baselines",
        registry_map,
        args.rerun,
        baseline_names,
        args.timeout_seconds,
    )
    published_rows = load_published_rows(upstream_root)
    history_rows = load_history_rows(upstream_root)

    dev_public_rows = merge_registry_rows(
        registry_rows,
        executed_rows,
        history_rows | published_rows,
    )
    dev_public_rows.append(
        _run_dev_public_controller_row(
            upstream_root,
            output_dir / "hallmark_mlx_bibtex_first_fallback_dev_public",
            args.rerun,
            args.timeout_seconds,
        )
    )

    compare32_rows = [
        run_compare32_row(
            config_path=ROOT / "configs" / "base.yaml",
            mode_name="bibtex_first_fallback",
            compare32_gold_path=COMPARE32_GOLD_PATH,
            method_name="hallmark_mlx_bibtex_first_fallback_compare32",
            description="hallmark-mlx controller with deterministic finalizer",
            output_dir=output_dir / "hallmark_mlx_bibtex_first_fallback_compare32",
            rerun=args.rerun,
        )
    ]
    for adapter_name, description, adapter_path in FINETUNED_MODELS:
        compare32_rows.append(
            run_compare32_row(
                config_path=ROOT / "configs" / "train_qwen_1_5b.yaml",
                mode_name="policy_deterministic",
                compare32_gold_path=COMPARE32_GOLD_PATH,
                method_name=adapter_name,
                description=description,
                adapter_path=adapter_path,
                output_dir=output_dir / adapter_name,
                rerun=args.rerun,
            )
        )

    summary = {
        "upstream_root": str(upstream_root),
        "registry_rows": registry_rows,
        "dev_public_rows": dev_public_rows,
        "compare32_rows": compare32_rows,
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
