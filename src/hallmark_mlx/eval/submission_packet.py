"""Helpers for packaging confirmed official HALLMARK submission artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

from hallmark_mlx.utils.io import read_json

ROOT = Path(__file__).resolve().parents[3]
CONFIRM_ROOT = ROOT / "artifacts" / "official_eval_sharded_fast5_confirm"
SUBMISSION_ROOT = ROOT / "artifacts" / "submission" / "hallmark"
OFFICIAL_SPLITS = ("dev_public", "test_public", "stress_test")
SUBMISSION_SKILL = "paper-references"
DEFAULT_UPSTREAM_ROOT = "/tmp/hallmark-upstream"
EXPECTED_NUM_ENTRIES = {
    "dev_public": 1119,
    "test_public": 831,
    "stress_test": 121,
}


@dataclass(frozen=True)
class ConfirmedSplitPaths:
    """Resolved paths for one confirmed official split run."""

    split: str
    root: Path
    row_path: Path
    result_path: Path
    predictions_path: Path
    traces_path: Path


def confirmed_split_paths(split: str) -> ConfirmedSplitPaths:
    """Return the canonical confirmed artifact paths for one official split."""

    root = CONFIRM_ROOT / f"{split}_bibtex_first_fallback" / "merged"
    return ConfirmedSplitPaths(
        split=split,
        root=root,
        row_path=root / "row.json",
        result_path=root / "result.json",
        predictions_path=root / "predictions.jsonl",
        traces_path=root / "traces.jsonl",
    )


def load_confirmed_outputs(split: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the confirmed official row/result pair for one split."""

    paths = confirmed_split_paths(split)
    return read_json(paths.row_path), read_json(paths.result_path)


def validate_confirmed_row(row: dict[str, Any]) -> list[str]:
    """Validate the expected invariants for one confirmed official row."""

    split = str(row["split"])
    checks = [
        f"split == {split}: {row.get('split') == split}",
        f"num_entries == {EXPECTED_NUM_ENTRIES[split]}: "
        f"{row.get('num_entries') == EXPECTED_NUM_ENTRIES[split]}",
        f"partial == False: {row.get('partial') is False}",
        f"coverage == 1.0: {float(row.get('coverage') or 0.0) == 1.0}",
        f"source == executed_here: {row.get('source') == 'executed_here'}",
    ]
    if "official evaluator" not in str(row.get("status_message", "")).lower():
        raise ValueError("Row status_message does not confirm the official evaluator.")
    if any(check.endswith("False") for check in checks):
        raise ValueError("Official row validation failed:\n" + "\n".join(checks))
    return checks


def build_metrics(row: dict[str, Any]) -> dict[str, float | None]:
    """Map a confirmed row into the public benchmark metric schema."""

    return {
        "detection_rate": float(row["detection_rate"]),
        "f1_hallucination": float(row["f1_hallucination"]),
        "tier_weighted_f1": float(row["tier_weighted_f1"]),
        "fpr": None
        if row.get("false_positive_rate") is None
        else float(row["false_positive_rate"]),
        "ece": None if row.get("ece") is None else float(row["ece"]),
    }


def build_tier_breakdown(result: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Convert official per-tier metrics into the public JSONL schema."""

    breakdown: dict[str, dict[str, float]] = {}
    for tier, metrics in dict(result.get("per_tier_metrics", {})).items():
        tier_key = f"tier-{tier}"
        breakdown[tier_key] = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
    return breakdown


def build_category_breakdown(result: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Convert official per-type metrics into the public JSONL schema."""

    breakdown: dict[str, dict[str, float]] = {}
    for category, metrics in dict(result.get("per_type_metrics", {})).items():
        if category == "valid":
            continue
        breakdown[category] = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
    return breakdown


def shard_plan(num_entries: int, *, shards: int = 8) -> list[tuple[int, int, int]]:
    """Split one official run into fixed shards of near-equal size."""

    chunk = ceil(num_entries / shards)
    plan: list[tuple[int, int, int]] = []
    for shard in range(shards):
        offset = shard * chunk
        if offset >= num_entries:
            break
        limit = min(chunk, num_entries - offset)
        plan.append((shard, offset, limit))
    return plan


def wrapper_reproduction_command(split: str, *, target: str = "controller") -> str:
    """Return the simplest public rerun command for one official split."""

    return (
        "uv run python scripts/run_submission_eval.py "
        f"--upstream-root {DEFAULT_UPSTREAM_ROOT} "
        f"--split {split} "
        f"--target {target} "
        f"--output-dir artifacts/submission_eval/{split}_{target}"
    )


def summary_line(split: str, row: dict[str, object]) -> str:
    """Format one compact split summary line for reports."""

    return (
        f"- `{split}`: DR {float(row['detection_rate']):.3f}, "
        f"F1-H {float(row['f1_hallucination']):.3f}, "
        f"TW-F1 {float(row['tier_weighted_f1']):.3f}, "
        f"FPR "
        + (
            "—"
            if row.get("false_positive_rate") is None
            else f"{float(row['false_positive_rate']):.3f}"
        )
        + f", ECE {float(row['ece']):.3f}"
    )


def build_submission_entry(
    *,
    split: str,
    row: dict[str, Any],
    result: dict[str, Any],
    version: str,
    date: str,
    reproduction: str,
) -> dict[str, Any]:
    """Build one Research Agora-compatible JSONL result entry."""

    metrics = build_metrics(row)
    entry = {
        "benchmark-id": "hallmark",
        "skill": SUBMISSION_SKILL,
        "model": "hybrid-controller",
        "version": version,
        "date": date,
        "split": split,
        "num_entries": int(row["num_entries"]),
        "metrics": metrics,
        "tier-breakdown": build_tier_breakdown(result),
        "category-breakdown": build_category_breakdown(result),
        "notes": (
            "BibTeX-first deterministic controller with scholarly tool fallback. "
            "Official evaluator run with 5s per-entry timeout and 8-way sharding."
        ),
        "reproduction": reproduction,
    }
    return entry
