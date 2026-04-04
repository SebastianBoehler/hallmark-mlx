from __future__ import annotations

from types import SimpleNamespace

from hallmark_mlx.eval.upstream_hallmark import result_to_row


def _result(*, split_name: str, num_entries: int, coverage: float) -> SimpleNamespace:
    return SimpleNamespace(
        tool_name="demo",
        split_name=split_name,
        num_entries=num_entries,
        coverage=coverage,
        detection_rate=0.0,
        false_positive_rate=0.0,
        f1_hallucination=0.0,
        tier_weighted_f1=0.0,
        mcc=0.0,
        ece=0.0,
        tier3_f1=0.0,
        coverage_adjusted_f1=0.0,
        mean_api_calls=0.0,
        num_uncertain=0,
        union_recall_at_k={},
        summary=lambda: {"fpr": 0.0},
    )


def test_result_to_row_full_official_split_is_not_partial() -> None:
    row = result_to_row(
        _result(split_name="stress_test", num_entries=121, coverage=1.0),
        source="executed_here",
        description="demo",
        available_here=True,
    )
    assert row["partial"] is False


def test_result_to_row_incomplete_coverage_is_partial() -> None:
    row = result_to_row(
        _result(split_name="test_public", num_entries=830, coverage=0.95),
        source="executed_here",
        description="demo",
        available_here=True,
    )
    assert row["partial"] is True
