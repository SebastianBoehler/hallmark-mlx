from hallmark_mlx.config import ToolsConfig
from hallmark_mlx.data.schemas import (
    CandidateMatch,
    ParsedBibliographicFields,
    ToolInvocation,
    ToolResultSummary,
    VerificationInput,
    VerificationTrace,
)
from hallmark_mlx.inference.policy_runner import PolicyRunner
from hallmark_mlx.inference.tool_executor import ToolExecutor
from hallmark_mlx.types import (
    EvidenceStrength,
    FinalizationMode,
    InputType,
    ToolName,
    VerificationAction,
    VerificationVerdict,
)


class FakeModel:
    def propose_trace(self, verification_input: VerificationInput) -> VerificationTrace:
        return VerificationTrace(
            input=verification_input,
            parsed_fields=ParsedBibliographicFields(title="Attention Is All You Need"),
            next_action=VerificationAction.QUERY_CROSSREF,
            tool_calls=[
                ToolInvocation(
                    tool=ToolName.CROSSREF,
                    action="search_works",
                    arguments={"query": "Attention Is All You Need Vaswani 2017"},
                ),
            ],
        )


class FakeToolExecutor(ToolExecutor):
    def __init__(self) -> None:
        super().__init__(ToolsConfig())

    def execute_many(self, tool_calls):  # type: ignore[no-untyped-def]
        return [
            ToolResultSummary(
                tool=ToolName.CROSSREF,
                action="search_works",
                ok=True,
                evidence_strength=EvidenceStrength.STRONG,
                candidate_count=1,
                candidate_summaries=[
                    CandidateMatch(
                        source="crossref",
                        title="Attention Is All You Need",
                        authors=["Ashish Vaswani"],
                        year=2017,
                        doi="10.5555/3295222.3295349",
                        score=0.95,
                    ),
                ],
            ),
        ]


class InteractiveFakeModel:
    def run_with_tools(
        self,
        verification_input: VerificationInput,
        tool_executor: ToolExecutor,
    ) -> VerificationTrace:
        result = tool_executor.execute_many(
            [
                ToolInvocation(
                    tool=ToolName.CROSSREF,
                    action="search_works",
                    arguments={"query": verification_input.raw_input},
                )
            ]
        )
        return VerificationTrace(
            input=verification_input,
            parsed_fields=ParsedBibliographicFields(title="Attention Is All You Need"),
            next_action=VerificationAction.FINALIZE,
            tool_calls=[
                ToolInvocation(
                    tool=ToolName.CROSSREF,
                    action="search_works",
                    arguments={"query": verification_input.raw_input},
                )
            ],
            tool_results=result,
        )


def test_policy_runner_finalizes_trace() -> None:
    runner = PolicyRunner(model=FakeModel(), tool_executor=FakeToolExecutor())
    trace = runner.run(
        VerificationInput(
            record_id="trace-1",
            input_type=InputType.RAW_CITATION_STRING,
            raw_input="Attention Is All You Need",
        ),
    )

    assert trace.final_decision is not None
    assert trace.final_decision.verdict == VerificationVerdict.VERIFIED


def test_policy_runner_uses_interactive_model_when_available() -> None:
    runner = PolicyRunner(
        model=InteractiveFakeModel(),
        tool_executor=FakeToolExecutor(),
        finalization_mode=FinalizationMode.GENERATIVE,
    )
    trace = runner.run(
        VerificationInput(
            record_id="trace-2",
            input_type=InputType.RAW_CITATION_STRING,
            raw_input="Attention Is All You Need",
        ),
    )

    assert len(trace.tool_results) == 1
    assert trace.final_decision is None


def test_policy_runner_can_force_deterministic_finalization() -> None:
    runner = PolicyRunner(
        model=InteractiveFakeModel(),
        tool_executor=FakeToolExecutor(),
        finalization_mode=FinalizationMode.DETERMINISTIC,
    )
    trace = runner.run(
        VerificationInput(
            record_id="trace-3",
            input_type=InputType.RAW_CITATION_STRING,
            raw_input="Attention Is All You Need",
        ),
    )

    assert trace.final_decision is not None
    assert trace.final_decision.verdict == VerificationVerdict.VERIFIED


def test_tool_executor_returns_failed_summary_for_invalid_arguments() -> None:
    executor = ToolExecutor(ToolsConfig())

    result = executor.execute(
        ToolInvocation(
            tool=ToolName.CROSSREF,
            action="search_works",
            arguments={},
        ),
    )

    assert result.ok is False
    assert "query" in (result.notes or "")
