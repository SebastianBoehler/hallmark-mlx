"""State helpers for resumable official benchmark runs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hallmark_mlx.data.schemas import FinalDecision, VerificationInput, VerificationTrace
from hallmark_mlx.types import VerificationAction, VerificationVerdict
from hallmark_mlx.utils.io import read_json, write_json

JSONDict = dict[str, Any]


def now_utc_iso() -> str:
    """Return a stable UTC timestamp string."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def entries_dir(output_root: Path) -> Path:
    """Return the directory that stores one JSON trace per official entry."""

    target = output_root / "entries"
    target.mkdir(parents=True, exist_ok=True)
    return target


def entry_trace_path(output_root: Path, index: int) -> Path:
    """Return the checkpoint path for one benchmark entry."""

    return entries_dir(output_root) / f"{index:04d}.json"


def load_completed_traces(output_root: Path) -> dict[int, VerificationTrace]:
    """Load all completed per-entry traces from disk."""

    completed: dict[int, VerificationTrace] = {}
    for path in sorted(entries_dir(output_root).glob("*.json")):
        try:
            index = int(path.stem)
        except ValueError:
            continue
        completed[index] = VerificationTrace.model_validate(read_json(path))
    return completed


def write_trace_checkpoint(
    output_root: Path,
    index: int,
    trace: VerificationTrace,
) -> Path:
    """Persist one completed official benchmark trace."""

    return write_json(entry_trace_path(output_root, index), trace.model_dump(exclude_none=True))


def build_error_trace(
    verification_input: VerificationInput,
    *,
    reason: str,
    wall_clock_seconds: float,
) -> VerificationTrace:
    """Create an explicit abstaining trace for runtime failures."""

    return VerificationTrace(
        policy_version="runtime_error",
        input=verification_input,
        next_action=VerificationAction.ABSTAIN,
        final_decision=FinalDecision(
            verdict=VerificationVerdict.ABSTAIN,
            confidence=0.0,
            rationale=f"Controller execution failed: {reason}",
            abstain_reason="runtime_error",
        ),
        metadata={
            "trace_status": "error",
            "run_error": reason,
            "wall_clock_seconds": wall_clock_seconds,
        },
    )


def write_progress_manifest(
    output_root: Path,
    *,
    method_name: str,
    split_name: str,
    total_entries: int,
    completed_entries: int,
    error_entries: int,
    entry_timeout_seconds: int | None,
    status: str,
) -> Path:
    """Persist a compact manifest for a resumable official eval run."""

    payload: JSONDict = {
        "method_name": method_name,
        "split_name": split_name,
        "total_entries": total_entries,
        "completed_entries": completed_entries,
        "error_entries": error_entries,
        "remaining_entries": max(total_entries - completed_entries, 0),
        "completion_fraction": completed_entries / total_entries if total_entries else 0.0,
        "entry_timeout_seconds": entry_timeout_seconds,
        "status": status,
        "updated_at": now_utc_iso(),
    }
    return write_json(output_root / "progress.json", payload)
