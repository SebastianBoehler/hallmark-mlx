from hallmark_mlx.eval.metrics import compute_metrics


def test_compute_metrics_reports_absent_positive_class() -> None:
    gold_rows = [
        {"bibtex_key": "a", "label": "VALID"},
        {"bibtex_key": "b", "label": "UNCERTAIN"},
    ]
    prediction_rows = [
        {"bibtex_key": "a", "label": "VALID"},
        {"bibtex_key": "b", "label": "UNCERTAIN"},
    ]

    metrics = compute_metrics(gold_rows, prediction_rows)

    assert metrics["num_hallucinated_gold"] == 0.0
    assert metrics["recall_hallucinated"] is None
    assert metrics["f1_hallucinated"] is None
    assert metrics["num_uncertain_predictions"] == 1.0


def test_compute_metrics_excludes_uncertain_predictions_from_confusion_matrix() -> None:
    gold_rows = [
        {"bibtex_key": "a", "label": "HALLUCINATED"},
        {"bibtex_key": "b", "label": "VALID"},
    ]
    prediction_rows = [
        {"bibtex_key": "a", "label": "UNCERTAIN"},
        {"bibtex_key": "b", "label": "HALLUCINATED"},
    ]

    metrics = compute_metrics(gold_rows, prediction_rows)

    assert metrics["tp"] == 0.0
    assert metrics["fp"] == 1.0
    assert metrics["fn"] == 0.0
    assert metrics["tn"] == 0.0
    assert metrics["num_uncertain_predictions"] == 1.0


def test_compute_metrics_adds_budgeted_scores_when_api_calls_are_present() -> None:
    gold_rows = [
        {"bibtex_key": "a", "label": "HALLUCINATED"},
        {"bibtex_key": "b", "label": "VALID"},
    ]
    prediction_rows = [
        {"bibtex_key": "a", "label": "HALLUCINATED", "api_calls": 1},
        {"bibtex_key": "b", "label": "HALLUCINATED", "api_calls": 3},
    ]

    metrics = compute_metrics(gold_rows, prediction_rows, tool_budgets=(1, 2))

    assert metrics["budget_1_within_budget_rate"] == 0.5
    assert metrics["budget_1_resolved_rate"] == 0.5
    assert metrics["budget_1_precision_hallucinated"] == 1.0
    assert metrics["budget_1_f1_hallucinated"] == 1.0
    assert metrics["budget_2_num_uncertain_predictions"] == 1.0
