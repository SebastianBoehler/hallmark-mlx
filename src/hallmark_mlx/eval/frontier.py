"""Frontier-style metrics for budget-aware policy optimization."""

from __future__ import annotations

from collections.abc import Mapping

DEFAULT_FRONTIER_METRIC_ORDER = (
    "frontier_score",
    "search64_frontier_score",
    "compare32_frontier_score",
    "compare32_frontier_delta",
    "compare32_false_positive_rate",
    "compare32_false_positive_delta",
    "compare32_label_accuracy",
    "compare32_label_accuracy_delta",
    "compare32_guard_penalty",
    "guard_constraint_triggered",
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


def compute_guarded_frontier_score(
    search_metrics: Mapping[str, float | None],
    compare_metrics: Mapping[str, float | None],
    baseline_compare_metrics: Mapping[str, float | None],
) -> float:
    """Scalarize search quality while penalizing held-out compare32 regressions."""

    search_score = compute_frontier_score(search_metrics)
    compare_score = compute_frontier_score(compare_metrics)
    baseline_compare_score = compute_frontier_score(baseline_compare_metrics)
    compare_delta = compare_score - baseline_compare_score
    compare_fpr_delta = _value(compare_metrics, "false_positive_rate") - _value(
        baseline_compare_metrics,
        "false_positive_rate",
    )
    compare_label_accuracy_delta = _value(compare_metrics, "label_accuracy") - _value(
        baseline_compare_metrics,
        "label_accuracy",
    )
    guard_penalty = (
        0.75 * max(0.0, -compare_delta)
        + 0.25 * max(0.0, compare_fpr_delta)
        + 0.25 * max(0.0, -compare_label_accuracy_delta)
    )
    constraint_penalty = 0.0
    if compare_delta < -0.01 or compare_fpr_delta > 0.02:
        constraint_penalty = 0.05
    return search_score - guard_penalty - constraint_penalty


def collect_frontier_metrics(metrics: Mapping[str, float | None]) -> dict[str, float]:
    """Return the metric subset that matters for Weco optimization."""

    collected = {
        key: _value(metrics, key)
        for key in DEFAULT_FRONTIER_METRIC_ORDER
        if key != "frontier_score"
    }
    collected["frontier_score"] = compute_frontier_score(metrics)
    return collected


def collect_guarded_frontier_metrics(
    search_metrics: Mapping[str, float | None],
    compare_metrics: Mapping[str, float | None],
    baseline_compare_metrics: Mapping[str, float | None],
) -> dict[str, float]:
    """Return a guarded two-split metric view for Qwen Weco training trials."""

    search_score = compute_frontier_score(search_metrics)
    compare_score = compute_frontier_score(compare_metrics)
    baseline_compare_score = compute_frontier_score(baseline_compare_metrics)
    compare_delta = compare_score - baseline_compare_score
    compare_fpr = _value(compare_metrics, "false_positive_rate")
    compare_fpr_delta = compare_fpr - _value(baseline_compare_metrics, "false_positive_rate")
    compare_label_accuracy = _value(compare_metrics, "label_accuracy")
    compare_label_accuracy_delta = compare_label_accuracy - _value(
        baseline_compare_metrics,
        "label_accuracy",
    )
    guard_penalty = (
        0.75 * max(0.0, -compare_delta)
        + 0.25 * max(0.0, compare_fpr_delta)
        + 0.25 * max(0.0, -compare_label_accuracy_delta)
    )
    constraint_triggered = float(compare_delta < -0.01 or compare_fpr_delta > 0.02)

    collected = collect_frontier_metrics(search_metrics)
    collected["frontier_score"] = compute_guarded_frontier_score(
        search_metrics,
        compare_metrics,
        baseline_compare_metrics,
    )
    collected["search64_frontier_score"] = search_score
    collected["compare32_frontier_score"] = compare_score
    collected["compare32_frontier_delta"] = compare_delta
    collected["compare32_false_positive_rate"] = compare_fpr
    collected["compare32_false_positive_delta"] = compare_fpr_delta
    collected["compare32_label_accuracy"] = compare_label_accuracy
    collected["compare32_label_accuracy_delta"] = compare_label_accuracy_delta
    collected["compare32_guard_penalty"] = guard_penalty + 0.05 * constraint_triggered
    collected["guard_constraint_triggered"] = constraint_triggered
    return collected
