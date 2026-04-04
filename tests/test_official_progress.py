from pathlib import Path

from hallmark_mlx.data.schemas import VerificationInput
from hallmark_mlx.eval.official_progress import (
    build_error_trace,
    load_completed_traces,
    write_progress_manifest,
    write_trace_checkpoint,
)
from hallmark_mlx.types import InputType, VerificationVerdict


def test_official_progress_roundtrip(tmp_path: Path) -> None:
    verification_input = VerificationInput(
        record_id="entry-1",
        input_type=InputType.BIBTEX_ENTRY,
        raw_input="@article{test,title={Test}}",
        benchmark_bibtex_key="test-key",
    )
    trace = build_error_trace(
        verification_input,
        reason="timed out",
        wall_clock_seconds=12.5,
    )

    write_trace_checkpoint(tmp_path, 3, trace)
    loaded = load_completed_traces(tmp_path)

    assert list(loaded) == [3]
    assert loaded[3].final_decision is not None
    assert loaded[3].final_decision.verdict == VerificationVerdict.ABSTAIN
    assert loaded[3].metadata["trace_status"] == "error"


def test_progress_manifest_contains_resume_fields(tmp_path: Path) -> None:
    path = write_progress_manifest(
        tmp_path,
        method_name="hallmark_mlx_bibtex_first_fallback",
        split_name="dev_public",
        total_entries=1119,
        completed_entries=111,
        error_entries=7,
        entry_timeout_seconds=60,
        status="running",
    )

    payload = path.read_text(encoding="utf-8")

    assert '"completed_entries": 111' in payload
    assert '"remaining_entries": 1008' in payload
    assert '"entry_timeout_seconds": 60' in payload
    assert '"status": "running"' in payload
