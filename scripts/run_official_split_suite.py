#!/usr/bin/env python3
"""Run official-split comparisons for hallmark-mlx and core HALLMARK baselines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from hallmark_mlx.config import load_config
from hallmark_mlx.eval.compare_rows import failed_execution_row
from hallmark_mlx.eval.official_compare import evaluate_runner_on_entries
from hallmark_mlx.eval.policy_modes import build_policy_runner
from hallmark_mlx.eval.timeouts import run_with_timeout
from hallmark_mlx.eval.upstream_hallmark import (
    load_entries,
    load_published_rows,
    result_to_row,
    run_baseline,
)
from hallmark_mlx.utils.io import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM_ROOT = Path("/tmp/hallmark-upstream")
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "official_split_suite"
OFFICIAL_SPLITS = ("dev_public", "test_public", "stress_test")
BASELINES = ("bibtexupdater", "harc")
CONTROLLER_ROW_PATHS = {
    split: (
        ROOT
        / "artifacts"
        / "official_eval_sharded_fast5"
        / f"{split}_bibtex_first_fallback"
        / "merged"
        / "row.json"
    )
    for split in OFFICIAL_SPLITS
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", default=str(DEFAULT_UPSTREAM_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--entry-timeout-seconds", type=int, default=5)
    parser.add_argument("--baseline-timeout-seconds", type=int, default=1200)
    parser.add_argument(
        "--splits",
        default=",".join(OFFICIAL_SPLITS),
        help="Comma-separated official splits to run.",
    )
    parser.add_argument("--controller-only", action="store_true")
    parser.add_argument(
        "--run-live-baselines",
        action="store_true",
        help=(
            "Execute bibtexupdater/harc on non-dev official splits instead of "
            "reporting controller rows only."
        ),
    )
    return parser


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row.get("f1_hallucination") or -1.0), reverse=True)


def _cached_controller_row(split: str) -> dict[str, Any] | None:
    row_path = CONTROLLER_ROW_PATHS.get(split)
    if row_path is None or not row_path.exists():
        return None
    return read_json(row_path)


def _controller_row(
    upstream_root: Path,
    output_dir: Path,
    split: str,
    rerun: bool,
    entry_timeout_seconds: int,
) -> dict[str, Any]:
    row_path = output_dir / "row.json"
    if row_path.exists() and not rerun:
        return read_json(row_path)
    cached_row = _cached_controller_row(split)
    if cached_row is not None and not rerun:
        row = cached_row
        write_json(row_path, row)
        return row

    config = load_config(ROOT / "configs" / "base.yaml")
    runner = build_policy_runner(config, "bibtex_first_fallback")
    entries = list(load_entries(upstream_root, split=split))
    row = evaluate_runner_on_entries(
        upstream_root=upstream_root,
        entries=entries,
        split_name=split,
        runner=runner,
        method_name="hallmark_mlx_bibtex_first_fallback",
        output_dir=output_dir,
        description="hallmark-mlx controller with deterministic finalizer",
        source="executed_here",
        resume=True,
        entry_timeout_seconds=entry_timeout_seconds,
        progress_every=10,
    )
    write_json(row_path, row)
    return row


def _baseline_row(
    upstream_root: Path,
    output_dir: Path,
    split: str,
    baseline_name: str,
    rerun: bool,
    baseline_timeout_seconds: int,
    published_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row_path = output_dir / "row.json"
    if row_path.exists() and not rerun:
        return read_json(row_path)
    if baseline_name in published_rows and split == "dev_public" and not rerun:
        row = published_rows[baseline_name]
        write_json(row_path, row)
        return row
    try:
        predictions, result = run_with_timeout(
            baseline_timeout_seconds,
            lambda: run_baseline(upstream_root, baseline_name=baseline_name, split=split),
        )
        row = result_to_row(
            result,
            source="executed_here",
            description=baseline_name,
            available_here=True,
            status_message="Executed here with official evaluator.",
            notes=f"Generated {len(predictions)} predictions locally.",
        )
    except Exception as exc:
        row = failed_execution_row(
            {"name": baseline_name, "description": baseline_name, "available_here": True},
            notes=str(exc),
        )
    write_json(row_path, row)
    return row


def main() -> None:
    args = build_parser().parse_args()
    upstream_root = Path(args.upstream_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_splits = tuple(split.strip() for split in args.splits.split(",") if split.strip())

    summary: dict[str, Any] = {
        "upstream_root": str(upstream_root),
        "hidden_test_available": (upstream_root / "data" / "v1.0" / "test_hidden.jsonl").exists(),
        "splits": {},
    }

    for split in selected_splits:
        print(f"[suite] split={split}", flush=True)
        split_dir = output_dir / split
        published_rows = load_published_rows(upstream_root, split=split)
        print(f"[suite] split={split} method=hallmark_mlx_bibtex_first_fallback", flush=True)
        rows = [
            _controller_row(
                upstream_root,
                split_dir / "hallmark_mlx_bibtex_first_fallback",
                split,
                args.rerun,
                args.entry_timeout_seconds,
            )
        ]
        if args.controller_only:
            summary["splits"][split] = _sorted_rows(rows)
            continue
        if split != "dev_public" and not args.run_live_baselines:
            summary["splits"][split] = _sorted_rows(rows)
            continue
        for baseline_name in BASELINES:
            print(f"[suite] split={split} baseline={baseline_name}", flush=True)
            rows.append(
                _baseline_row(
                    upstream_root,
                    split_dir / baseline_name,
                    split,
                    baseline_name,
                    args.rerun,
                    args.baseline_timeout_seconds,
                    published_rows,
                )
            )
        summary["splits"][split] = _sorted_rows(rows)

    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
