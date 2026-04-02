from hallmark_mlx.data.schemas import (
    FinalDecision,
    ParsedBibliographicFields,
    VerificationInput,
    VerificationTrace,
)
from hallmark_mlx.training.trace_formatter import format_trace_for_sft
from hallmark_mlx.types import InputType, VerificationAction, VerificationVerdict


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
