"""Frontier-style metrics for budget-aware policy optimization."""

from __future__ import annotations

from collections.abc import Mapping

DEFAULT_FRONTIER_METRIC_ORDER = (
    "frontier_score",
    "budget_1_f1_hallucinated",
    "budget_2_f1_hallucinated",
    "budget_4_f1_hallucinated",
    "budget_8_f1_hallucinated",
    "label_accuracy",
    "budget_1_resolved_rate",
    "avg_tool_calls",
    "first_turn_tool_call_rate",
    "completion_rate",
)


def _value(metrics: Mapping[str, float | None], key: str) -> float:
    value = metrics.get(key)
    return float(value) if value is not None else 0.0


def compute_frontier_score(metrics: Mapping[str, float | None]) -> float:
    """Scalarize quality across tool-call budgets to guide Weco search."""

    return (
        0.50 * _value(metrics, "budget_1_f1_hallucinated")
        + 0.25 * _value(metrics, "budget_2_f1_hallucinated")
        + 0.15 * _value(metrics, "budget_4_f1_hallucinated")
        + 0.05 * _value(metrics, "budget_8_f1_hallucinated")
        + 0.05 * _value(metrics, "label_accuracy")
    )


def collect_frontier_metrics(metrics: Mapping[str, float | None]) -> dict[str, float]:
    """Return the metric subset that matters for Weco optimization."""

    collected = {
        key: _value(metrics, key)
        for key in DEFAULT_FRONTIER_METRIC_ORDER
        if key != "frontier_score"
    }
    collected["frontier_score"] = compute_frontier_score(metrics)
    return collected
