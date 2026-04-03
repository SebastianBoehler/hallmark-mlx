"""Local evaluation metrics for citation hallucination tasks."""

from __future__ import annotations

from hallmark_mlx.types import HallmarkLabel

DEFAULT_TOOL_BUDGETS: tuple[int, ...] = (1, 2, 4, 8)


def _safe_divide_or_none(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def normalize_label(label: str) -> HallmarkLabel:
    normalized = label.upper()
    if normalized == HallmarkLabel.HALLUCINATED.value:
        return HallmarkLabel.HALLUCINATED
    if normalized == HallmarkLabel.UNCERTAIN.value:
        return HallmarkLabel.UNCERTAIN
    return HallmarkLabel.VALID


def _api_calls_for_row(row: dict[str, object]) -> int | None:
    value = row.get("api_calls")
    if value is None:
        return None
    return int(value)


def _compute_core_metrics(
    gold_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
) -> dict[str, float | None]:
    gold_by_key = {str(row["bibtex_key"]): normalize_label(str(row["label"])) for row in gold_rows}
    pred_by_key = {str(row["bibtex_key"]): normalize_label(str(row["label"])) for row in prediction_rows}

    tp = fp = fn = tn = 0
    num_uncertain = 0
    num_missing = 0
    label_matches = 0
    for key, gold_label in gold_by_key.items():
        pred_label = pred_by_key.get(key)
        if pred_label is None:
            num_missing += 1
            pred_for_cm = HallmarkLabel.VALID
            # Missing predictions are not counted as label matches regardless of gold label.
        elif pred_label == HallmarkLabel.UNCERTAIN:
            num_uncertain += 1
            if gold_label == HallmarkLabel.UNCERTAIN:
                label_matches += 1
            continue
        else:
            pred_for_cm = pred_label
            if pred_for_cm == gold_label:
                label_matches += 1

        if gold_label == HallmarkLabel.HALLUCINATED and pred_for_cm == HallmarkLabel.HALLUCINATED:
            tp += 1
        elif gold_label != HallmarkLabel.HALLUCINATED and pred_for_cm == HallmarkLabel.HALLUCINATED:
            fp += 1
        elif gold_label == HallmarkLabel.HALLUCINATED:
            fn += 1
        else:
            tn += 1

    num_hallucinated_gold = sum(label == HallmarkLabel.HALLUCINATED for label in gold_by_key.values())
    num_uncertain_gold = sum(label == HallmarkLabel.UNCERTAIN for label in gold_by_key.values())
    num_valid_gold = sum(label == HallmarkLabel.VALID for label in gold_by_key.values())
    precision = _safe_divide_or_none(tp, tp + fp)
    recall = _safe_divide_or_none(tp, tp + fn)
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    num_examples = float(len(gold_by_key))
    return {
        "num_examples": num_examples,
        "num_hallucinated_gold": float(num_hallucinated_gold),
        "num_valid_gold": float(num_valid_gold),
        "num_uncertain_gold": float(num_uncertain_gold),
        "num_uncertain_predictions": float(num_uncertain),
        "num_missing_predictions": float(num_missing),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "label_accuracy": _safe_divide_or_none(label_matches, len(gold_by_key)),
        "detection_rate": recall,
        "false_positive_rate": _safe_divide_or_none(fp, fp + tn),
        "precision_hallucinated": precision,
        "recall_hallucinated": recall,
        "f1_hallucinated": f1,
    }


def _apply_tool_budget(
    prediction_rows: list[dict[str, object]],
    max_api_calls: int,
) -> list[dict[str, object]]:
    budgeted_rows: list[dict[str, object]] = []
    for row in prediction_rows:
        budgeted_row = dict(row)
        api_calls = _api_calls_for_row(budgeted_row)
        if api_calls is not None and api_calls > max_api_calls:
            budgeted_row["label"] = HallmarkLabel.UNCERTAIN.value
            budgeted_row["confidence"] = 0.0
            budgeted_row["reason"] = (
                f"Prediction exceeded the {max_api_calls}-tool-call evaluation budget."
            )
        budgeted_rows.append(budgeted_row)
    return budgeted_rows


def _budget_sweep_metrics(
    gold_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    tool_budgets: tuple[int, ...],
) -> dict[str, float | None]:
    if not any(_api_calls_for_row(row) is not None for row in prediction_rows):
        return {}

    total = len(prediction_rows)
    # Pre-compute api_calls once per row to avoid repeated dict lookups.
    api_calls_per_row = [_api_calls_for_row(row) for row in prediction_rows]
    num_missing_api_calls = sum(v is None for v in api_calls_per_row)
    metrics: dict[str, float | None] = {}
    metrics["num_missing_api_calls"] = float(num_missing_api_calls)
    for budget in tool_budgets:
        budgeted_rows = _apply_tool_budget(prediction_rows, budget)
        budget_metrics = _compute_core_metrics(gold_rows, budgeted_rows)
        # Only count rows that actually reported api_calls when computing budget rates;
        # rows with missing api_calls are excluded so as not to bias the rates upward.
        rows_with_calls = total - num_missing_api_calls
        within_budget = sum(
            calls is not None and calls <= budget
            for calls in api_calls_per_row
        )
        resolved_within_budget = sum(
            calls is not None
            and calls <= budget
            and normalize_label(str(prediction_rows[i]["label"])) != HallmarkLabel.UNCERTAIN
            for i, calls in enumerate(api_calls_per_row)
        )
        metrics[f"budget_{budget}_within_budget_rate"] = (
            within_budget / rows_with_calls if rows_with_calls else None
        )
        metrics[f"budget_{budget}_resolved_rate"] = (
            resolved_within_budget / rows_with_calls if rows_with_calls else None
        )
        for key, value in budget_metrics.items():
            metrics[f"budget_{budget}_{key}"] = value
    return metrics


def compute_metrics(
    gold_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    *,
    tool_budgets: tuple[int, ...] = DEFAULT_TOOL_BUDGETS,
) -> dict[str, float | None]:
    """Compute a local metric suite, optionally swept over tool-call budgets."""

    metrics = _compute_core_metrics(gold_rows, prediction_rows)
    metrics.update(_budget_sweep_metrics(gold_rows, prediction_rows, tool_budgets))
    return metrics
