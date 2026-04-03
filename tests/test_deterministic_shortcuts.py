from hallmark_mlx.data.schemas import (
    ParsedBibliographicFields,
    ToolInvocation,
    ToolResultSummary,
    VerificationInput,
)
from hallmark_mlx.inference.deterministic_shortcuts import maybe_finalize_deterministically
from hallmark_mlx.types import (
    EvidenceStrength,
    InputType,
    ToolName,
    VerificationAction,
    VerificationVerdict,
)


def test_maybe_finalize_deterministically_returns_bibtex_verdict() -> None:
    trace = maybe_finalize_deterministically(
        verification_input=VerificationInput(
            record_id="bert",
            input_type=InputType.BIBTEX_ENTRY,
            raw_input="@inproceedings{bert,title={BERT}}",
        ),
        policy_version="mlx-transcript-v2",
        parsed_fields=ParsedBibliographicFields(title="BERT"),
        suspected_issues=[],
        proposed_query=None,
        next_action=VerificationAction.PARSE_INPUT,
        tool_calls=[
            ToolInvocation(
                tool=ToolName.BIBTEX_UPDATER,
                action="check_bibtex",
                arguments={"bibtex": "@inproceedings{bert,title={BERT}}", "strict": True},
            )
        ],
        tool_results=[
            ToolResultSummary(
                tool=ToolName.BIBTEX_UPDATER,
                action="check_bibtex",
                ok=True,
                evidence_strength=EvidenceStrength.STRONG,
                notes="INFO:   VERIFIED: 1",
                raw_payload={"status": "VERIFIED", "status_counts": {"VERIFIED": 1}},
            )
        ],
        candidate_ranking=None,
        metadata={},
    )

    assert trace is not None
    assert trace.final_decision is not None
    assert trace.final_decision.verdict == VerificationVerdict.VERIFIED


def test_maybe_finalize_deterministically_keeps_abstain_open() -> None:
    trace = maybe_finalize_deterministically(
        verification_input=VerificationInput(
            record_id="claim",
            input_type=InputType.CLAIM_FOR_SUPPORTING_REFS,
            raw_input="Transformers always reduce hallucinations.",
        ),
        policy_version="mlx-transcript-v2",
        parsed_fields=ParsedBibliographicFields(claim_text="Transformers always reduce hallucinations."),
        suspected_issues=[],
        proposed_query=None,
        next_action=VerificationAction.PARSE_INPUT,
        tool_calls=[],
        tool_results=[],
        candidate_ranking=None,
        metadata={},
    )

    assert trace is None
