from hallmark_mlx.data.schemas import (
    FinalDecision,
    ParsedBibliographicFields,
    ToolResultSummary,
    VerificationInput,
    VerificationTrace,
)
from hallmark_mlx.inference.finalizer import finalize_trace
from hallmark_mlx.types import EvidenceStrength, InputType, ToolName, VerificationVerdict


def _trace_with_bibtex_status(status: str) -> VerificationTrace:
    return VerificationTrace(
        input=VerificationInput(
            record_id="trace",
            input_type=InputType.BIBTEX_ENTRY,
            raw_input="@inproceedings{test,title={Test},author={A},year={2024},doi={10.1/test},booktitle={ICML}}",
        ),
        parsed_fields=ParsedBibliographicFields(
            title="Test",
            authors=["A"],
            year=2024,
            venue="ICML",
            doi="10.1/test",
        ),
        tool_results=[
            ToolResultSummary(
                tool=ToolName.BIBTEX_UPDATER,
                action="check_bibtex",
                ok=True,
                evidence_strength=EvidenceStrength.STRONG,
                notes=status,
                raw_payload={"status": status},
            ),
        ],
    )


def test_finalize_trace_respects_verified_bibtex_status() -> None:
    trace = finalize_trace(_trace_with_bibtex_status("VERIFIED"), force=True)

    assert trace.final_decision is not None
    assert trace.final_decision.verdict == VerificationVerdict.VERIFIED


def test_finalize_trace_maps_venue_mismatch_to_hallucinated() -> None:
    trace = finalize_trace(_trace_with_bibtex_status("VENUE_MISMATCH"), force=True)

    assert trace.final_decision is not None
    assert trace.final_decision.verdict == VerificationVerdict.HALLUCINATED


def test_finalize_trace_can_override_existing_generated_decision() -> None:
    trace = _trace_with_bibtex_status("DOI_NOT_FOUND").model_copy(
        update={
            "final_decision": FinalDecision(
                verdict=VerificationVerdict.ABSTAIN,
                confidence=0.1,
                rationale="generated fallback",
            ),
        },
    )

    finalized = finalize_trace(trace, force=True)

    assert finalized.final_decision is not None
    assert finalized.final_decision.verdict == VerificationVerdict.HALLUCINATED
