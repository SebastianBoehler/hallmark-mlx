"""Row-shaping helpers for benchmark comparison reports."""

from __future__ import annotations

from typing import Any


def unavailable_row(registry_row: dict[str, Any]) -> dict[str, Any]:
    is_available = bool(registry_row["available_here"])
    return {
        "name": registry_row["name"],
        "source": "not_run" if is_available else "unavailable",
        "description": registry_row["description"],
        "available_here": is_available,
        "status_message": (
            "Available here but not executed in this run."
            if is_available
            else registry_row["availability_message"]
        ),
        "split": "dev_public",
        "num_entries": 0,
        "coverage": 0.0,
        "partial": False,
        "notes": "",
        "detection_rate": None,
        "false_positive_rate": None,
        "f1_hallucination": None,
        "tier_weighted_f1": None,
        "mcc": None,
        "ece": None,
        "tier3_f1": None,
        "coverage_adjusted_f1": None,
        "mean_api_calls": None,
        "num_uncertain": None,
        "union_recall_at_k": {},
    }


def failed_execution_row(
    registry_row: dict[str, Any],
    *,
    notes: str,
) -> dict[str, Any]:
    return {
        "name": registry_row["name"],
        "source": "execution_failed",
        "description": registry_row["description"],
        "available_here": bool(registry_row.get("available_here", True)),
        "status_message": "Baseline execution failed during this run.",
        "split": "dev_public",
        "num_entries": 0,
        "coverage": 0.0,
        "partial": False,
        "notes": notes,
        "detection_rate": None,
        "false_positive_rate": None,
        "f1_hallucination": None,
        "tier_weighted_f1": None,
        "mcc": None,
        "ece": None,
        "tier3_f1": None,
        "coverage_adjusted_f1": None,
        "mean_api_calls": None,
        "num_uncertain": None,
        "union_recall_at_k": {},
    }


def merge_registry_rows(
    registry_rows: list[dict[str, Any]],
    executed_rows: dict[str, dict[str, Any]],
    published_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for registry_row in registry_rows:
        name = registry_row["name"]
        if name in executed_rows:
            row = executed_rows[name]
        elif name in published_rows:
            row = dict(published_rows[name])
            row["available_here"] = registry_row["available_here"]
            row["status_message"] = (
                f"{row['status_message']} Availability here: "
                f"{registry_row['availability_message']}"
            )
            row["description"] = registry_row["description"]
        else:
            row = unavailable_row(registry_row)
        row["group"] = "upstream_baseline"
        rows.append(row)
    return rows
