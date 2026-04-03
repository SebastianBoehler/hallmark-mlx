from hallmark_mlx.data.schemas import (
    FinalDecision,
    ProposedQuery,
    ParsedBibliographicFields,
    ToolInvocation,
    ToolResultSummary,
    VerificationInput,
    VerificationTrace,
)
from hallmark_mlx.training.trace_formatter import (
    format_tool_transcript_steps_for_sft,
    format_tool_transcript_for_sft,
    format_trace_for_sft,
)
from hallmark_mlx.types import (
    EvidenceStrength,
    InputType,
    VerificationAction,
    VerificationVerdict,
)


def test_trace_formatter_produces_chat_example() -> None:
    trace = VerificationTrace(
        input=VerificationInput(
            record_id="trace-1",
            input_type=InputType.RAW_CITATION_STRING,
            raw_input="Attention Is All You Need",
        ),
        parsed_fields=ParsedBibliographicFields(title="Attention Is All You Need"),
        next_action=VerificationAction.QUERY_CROSSREF,
        final_decision=FinalDecision(
            verdict=VerificationVerdict.VERIFIED,
            confidence=0.9,
            rationale="Example decision.",
        ),
    )

    example = format_trace_for_sft(trace)

    assert len(example["messages"]) == 4
    assert example["messages"][-1]["role"] == "assistant"
    assert '"next_action": "query_crossref"' in example["messages"][-1]["content"]


def test_tool_transcript_formatter_unrolls_tool_sequence() -> None:
    trace = VerificationTrace(
        input=VerificationInput(
            record_id="trace-2",
            input_type=InputType.RAW_CITATION_STRING,
            raw_input="Attention Is All You Need",
        ),
        parsed_fields=ParsedBibliographicFields(title="Attention Is All You Need"),
        proposed_query=ProposedQuery(
            query="Attention Is All You Need Vaswani 2017",
            purpose="resolve_canonical_record",
        ),
        next_action=VerificationAction.QUERY_OPENALEX,
        tool_calls=[
            ToolInvocation(
                tool="crossref",
                action="search_works",
                arguments={"query": "Attention Is All You Need Vaswani 2017", "rows": 5},
            ),
            ToolInvocation(
                tool="openalex",
                action="search_works",
                arguments={"query": "Attention Is All You Need Vaswani 2017", "rows": 5},
            ),
        ],
        tool_results=[
            ToolResultSummary(
                tool="crossref",
                action="search_works",
                ok=True,
                evidence_strength=EvidenceStrength.STRONG,
                candidate_count=1,
                notes="Exact title match.",
            ),
            ToolResultSummary(
                tool="openalex",
                action="search_works",
                ok=True,
                evidence_strength=EvidenceStrength.STRONG,
                candidate_count=1,
                notes="Independent confirmation.",
            ),
        ],
        final_decision=FinalDecision(
            verdict=VerificationVerdict.VERIFIED,
            confidence=0.93,
            rationale="Two tools agree on a single record.",
        ),
    )

    example = format_tool_transcript_for_sft(trace)
    messages = example["messages"]

    assert len(messages) == 7
    assert example["metadata"]["training_format"] == "tool_transcript"
    assert "tools" in example
    assert messages[2]["role"] == "assistant"
    assert "tool_calls" in messages[2]
    assert messages[2]["tool_calls"][0]["name"] == "crossref.search_works"
    assert messages[3]["role"] == "tool"
    assert '"evidence_strength"' in messages[3]["content"]
    assert messages[-1]["role"] == "assistant"
    assert '"verdict"' in messages[-1]["content"]
    assert '"rationale"' in messages[-1]["content"]


def test_stepwise_transcript_formatter_emits_separate_tool_and_final_targets() -> None:
    trace = VerificationTrace(
        input=VerificationInput(
            record_id="trace-3",
            input_type=InputType.RAW_CITATION_STRING,
            raw_input="Sentence-BERT Reimers 2019",
        ),
        tool_calls=[
            ToolInvocation(
                tool="crossref",
                action="search_works",
                arguments={"query": "Sentence-BERT Reimers 2019", "rows": 3},
            ),
        ],
        tool_results=[
            ToolResultSummary(
                tool="crossref",
                action="search_works",
                ok=True,
                evidence_strength=EvidenceStrength.STRONG,
                candidate_count=1,
                notes="Exact title match.",
            ),
        ],
        final_decision=FinalDecision(
            verdict=VerificationVerdict.VERIFIED,
            confidence=0.97,
            rationale="The citation matches a single strong Crossref candidate.",
        ),
    )

    examples = format_tool_transcript_steps_for_sft(trace)

    assert len(examples) == 2
    assert examples[0]["metadata"]["target_type"] == "tool_call"
    assert "tool_calls" in examples[0]["messages"][-1]
    assert examples[1]["metadata"]["target_type"] == "final_decision"
    assert '"verdict"' in examples[1]["messages"][-1]["content"]
