"""Load trace datasets and prepare chat-format training files."""

from __future__ import annotations

from pathlib import Path

from hallmark_mlx.data.schemas import VerificationTrace
from hallmark_mlx.inference.finalizer import finalize_trace
from hallmark_mlx.training.trace_formatter import format_trace_examples_for_sft
from hallmark_mlx.types import TrainingExampleFormat
from hallmark_mlx.utils.io import ensure_dir
from hallmark_mlx.utils.jsonl import iter_jsonl, write_jsonl


def load_trace_split(path: str | Path) -> list[VerificationTrace]:
    """Load one split of verification traces."""

    return [VerificationTrace.model_validate(row) for row in iter_jsonl(path)]


def prepare_training_dataset(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    example_format: TrainingExampleFormat = TrainingExampleFormat.TRACE_JSON,
) -> dict[str, Path]:
    """Convert trace JSONL splits into MLX-friendly chat JSONL."""

    source_root = Path(dataset_dir)
    target_root = ensure_dir(output_dir)
    outputs: dict[str, Path] = {}

    for split_name in ("train", "valid"):
        split_path = source_root / f"{split_name}.jsonl"
        traces = load_trace_split(split_path)
        # Traces from the warm-start bootstrap have no final_decision, which means
        # format_tool_transcript_steps_for_sft emits zero finalization examples and the
        # model never learns when to stop.  Apply the conservative heuristic finalizer
        # so every trace carries a decision before it enters the formatter.
        traces = [finalize_trace(t) for t in traces]
        formatted = [
            example
            for trace in traces
            for example in format_trace_examples_for_sft(trace, example_format=example_format)
        ]
        outputs[split_name] = write_jsonl(target_root / f"{split_name}.jsonl", formatted)

    return outputs
