"""Helpers for bootstrapping seed traces from input JSONL."""

from __future__ import annotations

from pathlib import Path

from hallmark_mlx.data.schemas import VerificationInput
from hallmark_mlx.inference.policy_runner import PolicyRunner
from hallmark_mlx.utils.jsonl import iter_jsonl, write_jsonl


def bootstrap_trace_dataset(
    input_path: str | Path,
    output_path: str | Path,
    runner: PolicyRunner,
    *,
    limit: int | None = None,
) -> Path:
    """Generate verification traces from a JSONL file of `VerificationInput` records."""

    traces = []
    for index, row in enumerate(iter_jsonl(input_path)):
        if limit is not None and index >= limit:
            break
        verification_input = VerificationInput.model_validate(row)
        trace = runner.run(verification_input)
        traces.append(trace.model_dump(exclude_none=True))
    return write_jsonl(output_path, traces)
