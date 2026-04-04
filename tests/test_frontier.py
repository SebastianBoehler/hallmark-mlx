import pytest

from hallmark_mlx.eval.frontier import (
    collect_frontier_metrics,
    collect_guarded_frontier_metrics,
    compute_frontier_score,
    compute_guarded_frontier_score,
)


def test_compute_frontier_score_weights_early_budget_most() -> None:
    score = compute_frontier_score(
        {
            "budget_1_f1_hallucinated": 1.0,
            "budget_2_f1_hallucinated": 0.5,
            "budget_4_f1_hallucinated": 0.25,
            "budget_8_f1_hallucinated": 0.0,
            "label_accuracy": 0.5,
        }
    )

    assert score == 0.5 + 0.125 + 0.0375 + 0.0 + 0.025


def test_collect_frontier_metrics_replaces_missing_values_with_zero() -> None:
    metrics = collect_frontier_metrics({"label_accuracy": 0.75})

    assert metrics["frontier_score"] == pytest.approx(0.0375)
    assert metrics["budget_1_f1_hallucinated"] == 0.0
    assert metrics["label_accuracy"] == 0.75


def test_compute_guarded_frontier_score_penalizes_compare32_regression() -> None:
    baseline_compare = {
        "budget_1_f1_hallucinated": 0.8648648648648648,
        "budget_2_f1_hallucinated": 0.8648648648648648,
        "budget_4_f1_hallucinated": 0.8648648648648648,
        "budget_8_f1_hallucinated": 0.8648648648648648,
        "label_accuracy": 0.84375,
        "false_positive_rate": 0.3125,
    }
    baseline_search = {
        "budget_1_f1_hallucinated": 0.912621359223301,
        "budget_2_f1_hallucinated": 0.912621359223301,
        "budget_4_f1_hallucinated": 0.912621359223301,
        "budget_8_f1_hallucinated": 0.912621359223301,
        "label_accuracy": 0.859375,
        "false_positive_rate": 0.5,
    }
    regressed_compare = {
        "budget_1_f1_hallucinated": 0.8421052631578948,
        "budget_2_f1_hallucinated": 0.8421052631578948,
        "budget_4_f1_hallucinated": 0.8421052631578948,
        "budget_8_f1_hallucinated": 0.8421052631578948,
        "label_accuracy": 0.8125,
        "false_positive_rate": 0.375,
    }
    search_improved = {
        "budget_1_f1_hallucinated": 0.9215686274509804,
        "budget_2_f1_hallucinated": 0.9215686274509804,
        "budget_4_f1_hallucinated": 0.9215686274509804,
        "budget_8_f1_hallucinated": 0.9215686274509804,
        "label_accuracy": 0.859375,
        "false_positive_rate": 0.5,
    }

    guarded = compute_guarded_frontier_score(
        search_improved,
        regressed_compare,
        baseline_compare,
    )
    baseline = compute_guarded_frontier_score(
        baseline_search,
        baseline_compare,
        baseline_compare,
    )

    assert guarded < baseline


def test_collect_guarded_frontier_metrics_exposes_compare_guardrails() -> None:
    metrics = collect_guarded_frontier_metrics(
        search_metrics={"label_accuracy": 0.8},
        compare_metrics={"label_accuracy": 0.7, "false_positive_rate": 0.4},
        baseline_compare_metrics={"label_accuracy": 0.8, "false_positive_rate": 0.3},
    )

    assert metrics["search64_frontier_score"] == pytest.approx(0.04)
    assert metrics["compare32_label_accuracy_delta"] == pytest.approx(-0.1)
    assert metrics["compare32_false_positive_delta"] == pytest.approx(0.1)
    assert metrics["guard_constraint_triggered"] == 1.0
