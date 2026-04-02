"""Dataset construction for trace-based policy learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from hallmark_mlx.data.contamination import detect_family_overlap, trace_family_id
from hallmark_mlx.data.schemas import VerificationTrace
from hallmark_mlx.data.splitters import split_traces
from hallmark_mlx.utils.io import ensure_dir, write_json
from hallmark_mlx.utils.jsonl import iter_jsonl, write_jsonl


@dataclass(slots=True)
class DatasetBuildSummary:
    """High-level dataset build output."""

    total_records: int
    split_counts: dict[str, int]
    unique_families: int
    train_eval_overlap_families: list[str]


def load_traces(path: str | Path) -> list[VerificationTrace]:
    """Load verification traces from JSONL."""

    return [VerificationTrace.model_validate(row) for row in iter_jsonl(path)]


def build_trace_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> DatasetBuildSummary:
    """Build contamination-aware train, valid, test, and holdout splits."""

    traces = load_traces(input_path)
    output_root = ensure_dir(output_dir)
    split_map = split_traces(
        traces,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    for split_name, split_traces_list in split_map.items():
        rows = [trace.to_training_dict() for trace in split_traces_list]
        write_jsonl(output_root / f"{split_name}.jsonl", rows)

    train_records = split_map.get("train", [])
    eval_records = split_map.get("valid", []) + split_map.get("test", [])
    overlap = sorted(detect_family_overlap(train_records, eval_records))
    summary = DatasetBuildSummary(
        total_records=len(traces),
        split_counts={name: len(records) for name, records in split_map.items()},
        unique_families=len({trace_family_id(trace) for trace in traces}),
        train_eval_overlap_families=overlap,
    )
    write_json(output_root / "summary.json", asdict(summary))
    return summary
