"""Helpers for running and reading official HALLMARK baselines."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from hallmark_mlx.utils.io import read_json

JSONDict = dict[str, Any]


def _ensure_upstream_path(upstream_root: Path) -> None:
    root = str(upstream_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_registry(upstream_root: str | Path) -> dict[str, Any]:
    """Load the upstream HALLMARK baseline registry."""

    root = Path(upstream_root).resolve()
    _ensure_upstream_path(root)
    from hallmark.baselines.registry import get_registry

    return get_registry()


def list_registry_rows(upstream_root: str | Path) -> list[JSONDict]:
    """Return serializable baseline registry metadata."""

    root = Path(upstream_root).resolve()
    _ensure_upstream_path(root)
    from hallmark.baselines.registry import check_available

    rows: list[JSONDict] = []
    for name, info in load_registry(root).items():
        available, message = check_available(name)
        rows.append(
            {
                "name": name,
                "description": info.description,
                "confidence_type": info.confidence_type,
                "is_free": info.is_free,
                "requires_api_key": info.requires_api_key,
                "env_var": info.env_var,
                "pip_packages": list(info.pip_packages),
                "cli_commands": list(info.cli_commands),
                "available_here": available,
                "availability_message": message,
            }
        )
    return rows


def load_entries(
    upstream_root: str | Path,
    *,
    split: str = "dev_public",
    version: str = "v1.0",
) -> list[Any]:
    """Load official HALLMARK entries."""

    root = Path(upstream_root).resolve()
    _ensure_upstream_path(root)
    from hallmark.dataset.loader import load_split

    return load_split(split=split, version=version, data_dir=root / "data")


def evaluate_predictions(
    upstream_root: str | Path,
    *,
    split: str,
    predictions: list[Any],
    tool_name: str,
    version: str = "v1.0",
) -> Any:
    """Evaluate Prediction objects with the upstream HALLMARK evaluator."""

    root = Path(upstream_root).resolve()
    _ensure_upstream_path(root)
    from hallmark.evaluation.metrics import evaluate

    entries = load_entries(root, split=split, version=version)
    return evaluate(
        entries=entries,
        predictions=predictions,
        tool_name=tool_name,
        split_name=split,
    )


def load_prediction_file(
    upstream_root: str | Path,
    prediction_path: str | Path,
) -> list[Any]:
    """Load Prediction objects from a JSONL file."""

    root = Path(upstream_root).resolve()
    _ensure_upstream_path(root)
    from hallmark.dataset.schema import Prediction

    path = Path(prediction_path)
    return [
        Prediction.from_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_baseline(
    upstream_root: str | Path,
    *,
    baseline_name: str,
    split: str = "dev_public",
    version: str = "v1.0",
) -> tuple[list[Any], Any]:
    """Run one upstream baseline and evaluate it with official metrics."""

    root = Path(upstream_root).resolve()
    _ensure_upstream_path(root)
    from hallmark.baselines.registry import run_baseline as upstream_run_baseline

    entries = load_entries(root, split=split, version=version)
    predictions = upstream_run_baseline(baseline_name, entries)
    result = evaluate_predictions(
        root,
        split=split,
        predictions=predictions,
        tool_name=baseline_name,
        version=version,
    )
    return predictions, result


def result_to_row(
    result: Any,
    *,
    source: str,
    description: str,
    available_here: bool,
    status_message: str = "OK",
    partial: bool = False,
    notes: str = "",
) -> JSONDict:
    """Normalize an upstream EvaluationResult into one comparison row."""

    summary = result.summary()
    return {
        "name": result.tool_name,
        "source": source,
        "description": description,
        "available_here": available_here,
        "status_message": status_message,
        "split": result.split_name,
        "num_entries": result.num_entries,
        "coverage": result.coverage,
        "partial": partial or result.coverage < 0.999,
        "notes": notes,
        "detection_rate": result.detection_rate,
        "false_positive_rate": result.false_positive_rate,
        "f1_hallucination": result.f1_hallucination,
        "tier_weighted_f1": result.tier_weighted_f1,
        "mcc": result.mcc,
        "ece": result.ece,
        "tier3_f1": result.tier3_f1,
        "coverage_adjusted_f1": result.coverage_adjusted_f1,
        "mean_api_calls": result.mean_api_calls,
        "num_uncertain": result.num_uncertain,
        "union_recall_at_k": dict(result.union_recall_at_k),
    } | summary


def load_published_rows(
    upstream_root: str | Path,
    *,
    split: str = "dev_public",
) -> dict[str, JSONDict]:
    """Load official published baseline results shipped by the upstream repo."""

    root = Path(upstream_root).resolve()
    _ensure_upstream_path(root)
    from hallmark.dataset.schema import EvaluationResult

    manifest_path = root / "data" / "v1.0" / "baseline_results" / "manifest.json"
    results_dir = manifest_path.parent
    manifest = read_json(manifest_path)
    rows: dict[str, JSONDict] = {}

    for filename, meta in dict(manifest.get("files", {})).items():
        result = EvaluationResult.from_dict(read_json(results_dir / filename))
        baseline_name = str(meta["baseline"])
        rows[baseline_name] = result_to_row(
            result,
            source="upstream_published_result",
            description=baseline_name,
            available_here=False,
            status_message="Loaded from upstream published result.",
            partial=int(meta.get("num_entries", result.num_entries)) < 1119,
            notes=f"Published by upstream repo in {filename}.",
        )

    llm_predictions_path = results_dir / "llm_tool_augmented_dev_public.jsonl"
    if split == "dev_public" and llm_predictions_path.exists():
        predictions = load_prediction_file(root, llm_predictions_path)
        result = evaluate_predictions(
            root,
            split=split,
            predictions=predictions,
            tool_name="llm_tool_augmented",
        )
        rows["llm_tool_augmented"] = result_to_row(
            result,
            source="upstream_published_predictions",
            description="GPT-5.1 augmented with bibtex-updater evidence",
            available_here=False,
            status_message="Recomputed from upstream shipped predictions.",
            notes="Predictions JSONL shipped upstream; metrics recomputed locally.",
        )
    return rows


def load_history_rows(
    upstream_root: str | Path,
    *,
    split: str = "dev_public",
) -> dict[str, JSONDict]:
    """Load the best available upstream historical partial results per baseline."""

    history_path = Path(upstream_root).resolve() / "results" / "history.jsonl"
    if not history_path.exists():
        return {}
    selected: dict[str, tuple[int, int, JSONDict]] = {}
    for index, line in enumerate(history_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split_name") != split:
            continue
        baseline_name = str(row["tool_name"])
        score = (int(row.get("num_entries", 0)), index)
        if baseline_name not in selected or score >= selected[baseline_name][:2]:
            selected[baseline_name] = (score[0], score[1], row)

    rows: dict[str, JSONDict] = {}
    for baseline_name, (_, _, row) in selected.items():
        rows[baseline_name] = {
            "name": baseline_name,
            "source": "upstream_history_partial",
            "description": baseline_name,
            "available_here": False,
            "status_message": "Loaded from upstream historical result.",
            "split": str(row["split_name"]),
            "num_entries": int(row.get("num_entries", 0)),
            "coverage": None,
            "partial": True,
            "notes": "Historical upstream run; not a current full dev_public published result.",
            "detection_rate": row.get("detection_rate"),
            "false_positive_rate": row.get("false_positive_rate"),
            "f1_hallucination": row.get("f1_hallucination"),
            "tier_weighted_f1": row.get("tier_weighted_f1"),
            "mcc": None,
            "ece": None,
            "tier3_f1": None,
            "coverage_adjusted_f1": None,
            "mean_api_calls": None,
            "num_uncertain": None,
            "union_recall_at_k": {},
            "timestamp": row.get("timestamp"),
        }
    return rows
