import pytest

from hallmark_mlx.eval.frontier import collect_frontier_metrics, compute_frontier_score


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
