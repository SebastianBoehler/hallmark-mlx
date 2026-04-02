from pathlib import Path

from hallmark_mlx.config import ToolsConfig
from hallmark_mlx.data.schemas import VerificationInput
from hallmark_mlx.inference.bootstrap import bootstrap_trace_dataset
from hallmark_mlx.inference.policy_runner import PolicyRunner
from hallmark_mlx.inference.tool_executor import ToolExecutor
from hallmark_mlx.inference.warm_start_policy import WarmStartPolicyModel
from hallmark_mlx.types import InputType
from hallmark_mlx.utils.jsonl import read_jsonl, write_jsonl


class EmptyToolExecutor(ToolExecutor):
    def __init__(self) -> None:
        super().__init__(ToolsConfig())

    def execute_many(self, tool_calls):  # type: ignore[no-untyped-def]
        return []


def test_bootstrap_trace_dataset_writes_trace_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "traces.jsonl"
    write_jsonl(
        input_path,
        [
            VerificationInput(
                record_id="seed-1",
                input_type=InputType.RAW_CITATION_STRING,
                raw_input="Vaswani et al. Attention Is All You Need. NeurIPS 2017.",
            ).model_dump(exclude_none=True),
        ],
    )
    runner = PolicyRunner(model=WarmStartPolicyModel(), tool_executor=EmptyToolExecutor())

    bootstrap_trace_dataset(input_path, output_path, runner)

    rows = read_jsonl(output_path)
    assert len(rows) == 1
    assert rows[0]["policy_version"] == "warm-start-v0"
    assert rows[0]["input"]["record_id"] == "seed-1"
