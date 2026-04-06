from __future__ import annotations

from hallmark_mlx.eval.submission_packet import (
    build_submission_entry,
    validate_confirmed_row,
)


def test_validate_confirmed_row_accepts_full_official_test_public_row() -> None:
    checks = validate_confirmed_row(
        {
            "split": "test_public",
            "num_entries": 831,
            "coverage": 1.0,
            "partial": False,
            "source": "executed_here",
            "status_message": "Executed here with official evaluator from merged shard outputs.",
        }
    )
    assert all(check.endswith("True") for check in checks)


def test_build_submission_entry_maps_metrics_and_filters_valid_category() -> None:
    entry = build_submission_entry(
        split="test_public",
        row={
            "detection_rate": 0.9,
            "f1_hallucination": 0.8,
            "tier_weighted_f1": 0.85,
            "false_positive_rate": 0.1,
            "ece": 0.05,
            "num_entries": 10,
        },
        result={
            "per_tier_metrics": {
                "1": {"detection_rate": 1.0, "f1": 0.9, "num_hallucinated": 3},
            },
            "per_type_metrics": {
                "near_miss_title": {"count": 2, "detection_rate": 0.5, "f1": 0.6},
                "valid": {"count": 8, "false_positive_rate": 0.1},
            },
        },
        version="abc123",
        date="2026-04-06",
        reproduction="bash reproduce_test_public.sh",
    )

    assert entry["benchmark-id"] == "hallmark"
    assert entry["metrics"]["fpr"] == 0.1
    assert entry["tier-breakdown"]["tier-1"]["detection_rate"] == 1.0
    assert "valid" not in entry["category-breakdown"]
    assert entry["category-breakdown"]["near_miss_title"]["f1"] == 0.6
