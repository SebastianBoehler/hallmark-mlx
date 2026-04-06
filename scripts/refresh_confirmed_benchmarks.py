#!/usr/bin/env python3
"""Sync confirmed official rows into summary artifacts and regenerate reports."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIRM_ROOT = ROOT / "artifacts" / "official_eval_sharded_fast5_confirm"
OFFICIAL_SUMMARY_PATH = ROOT / "artifacts" / "official_split_suite" / "summary.json"
BASELINE_SUMMARY_PATH = ROOT / "artifacts" / "hallmark_baseline_compare" / "summary.json"
SPLITS = ("dev_public", "test_public", "stress_test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-export", action="store_true")
    return parser


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _confirmed_row(split: str) -> dict[str, object]:
    return _read_json(CONFIRM_ROOT / f"{split}_bibtex_first_fallback" / "merged" / "row.json")


def _sync_official_summary() -> None:
    summary = _read_json(OFFICIAL_SUMMARY_PATH)
    splits = summary.get("splits", {})
    for split in SPLITS:
        rows = list(splits.get(split, []))
        row = _confirmed_row(split)
        replaced = False
        for idx, existing in enumerate(rows):
            if existing.get("name") == "hallmark_mlx_bibtex_first_fallback":
                rows[idx] = row
                replaced = True
                break
        if not replaced:
            rows.insert(0, row)
        splits[split] = rows
    summary["splits"] = splits
    _write_json(OFFICIAL_SUMMARY_PATH, summary)


def _sync_baseline_summary() -> None:
    summary = _read_json(BASELINE_SUMMARY_PATH)
    rows = list(summary.get("dev_public_rows", []))
    row = _confirmed_row("dev_public")
    for idx, existing in enumerate(rows):
        if existing.get("name") == "hallmark_mlx_bibtex_first_fallback":
            row["group"] = existing.get("group", "hallmark_mlx")
            rows[idx] = row
            break
    else:
        row["group"] = "hallmark_mlx"
        rows.append(row)
    summary["dev_public_rows"] = rows
    _write_json(BASELINE_SUMMARY_PATH, summary)


def _run_export(script_name: str) -> None:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT / "src") if existing_pythonpath is None else (
        f"{ROOT / 'src'}:{existing_pythonpath}"
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script_name)],
        cwd=ROOT,
        env=env,
        check=True,
    )


def main() -> None:
    args = build_parser().parse_args()
    _sync_official_summary()
    _sync_baseline_summary()

    if not args.skip_export:
        _run_export("export_official_split_report.py")
        _run_export("export_submission_readiness.py")
        _run_export("export_submission_packet.py")
        _run_export("plot_benchmark_comparison.py")
        _run_export("export_benchmark_table.py")

    print(OFFICIAL_SUMMARY_PATH)
    print(BASELINE_SUMMARY_PATH)


if __name__ == "__main__":
    main()
