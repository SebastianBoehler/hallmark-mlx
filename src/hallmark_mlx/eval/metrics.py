"""Local evaluation metrics for citation hallucination tasks."""

from __future__ import annotations

from hallmark_mlx.types import HallmarkLabel


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def normalize_label(label: str) -> HallmarkLabel:
    normalized = label.upper()
    if normalized == HallmarkLabel.HALLUCINATED.value:
        return HallmarkLabel.HALLUCINATED
    if normalized == HallmarkLabel.UNCERTAIN.value:
        return HallmarkLabel.UNCERTAIN
    return HallmarkLabel.VALID


def compute_metrics(
    gold_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
) -> dict[str, float]:
    """Compute a small local metric suite."""

    gold_by_key = {str(row["bibtex_key"]): normalize_label(str(row["label"])) for row in gold_rows}
    pred_by_key = {str(row["bibtex_key"]): normalize_label(str(row["label"])) for row in prediction_rows}

    tp = fp = fn = tn = 0
    for key, gold_label in gold_by_key.items():
        pred_label = pred_by_key.get(key, HallmarkLabel.UNCERTAIN)
        pred_for_cm = HallmarkLabel.VALID if pred_label == HallmarkLabel.UNCERTAIN else pred_label
        if gold_label == HallmarkLabel.HALLUCINATED and pred_for_cm == HallmarkLabel.HALLUCINATED:
            tp += 1
        elif gold_label != HallmarkLabel.HALLUCINATED and pred_for_cm == HallmarkLabel.HALLUCINATED:
            fp += 1
        elif gold_label == HallmarkLabel.HALLUCINATED:
            fn += 1
        else:
            tn += 1

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return {
        "detection_rate": recall,
        "false_positive_rate": _safe_divide(fp, fp + tn),
        "precision_hallucinated": precision,
        "recall_hallucinated": recall,
        "f1_hallucinated": _safe_divide(2 * precision * recall, precision + recall),
    }
