"""Deterministic split assignment helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from random import Random

from hallmark_mlx.data.contamination import trace_family_id
from hallmark_mlx.data.schemas import VerificationTrace


def assign_family_splits(
    family_ids: Iterable[str],
    *,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    """Assign each citation family to train, valid, or test."""

    unique_families = sorted(set(family_ids))
    rng = Random(seed)
    rng.shuffle(unique_families)
    valid_cutoff = int(len(unique_families) * valid_ratio)
    test_cutoff = valid_cutoff + int(len(unique_families) * test_ratio)

    assignments: dict[str, str] = {}
    for index, family_id in enumerate(unique_families):
        if index < valid_cutoff:
            assignments[family_id] = "valid"
        elif index < test_cutoff:
            assignments[family_id] = "test"
        else:
            assignments[family_id] = "train"
    return assignments


def split_traces(
    traces: list[VerificationTrace],
    *,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[VerificationTrace]]:
    """Split traces while keeping citation families intact."""

    family_map = assign_family_splits(
        (trace_family_id(trace) for trace in traces if not trace.input.private_holdout),
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    buckets: dict[str, list[VerificationTrace]] = defaultdict(list)
    for trace in traces:
        if trace.input.private_holdout:
            buckets["holdout"].append(trace)
            continue
        buckets[family_map[trace_family_id(trace)]].append(trace)
    return dict(buckets)
