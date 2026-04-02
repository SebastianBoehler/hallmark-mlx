from hallmark_mlx.data.schemas import FinalDecision, VerificationInput, VerificationTrace
from hallmark_mlx.eval.hallmark_adapter import prediction_from_trace
from hallmark_mlx.types import InputType, VerificationVerdict


def test_prediction_from_trace_uses_benchmark_key() -> None:
    trace = VerificationTrace(
        input=VerificationInput(
            record_id="trace-1",
            input_type=InputType.BIBTEX_ENTRY,
            raw_input="@article{test}",
            benchmark_bibtex_key="a3f9c2b1d4e76f85",
        ),
        final_decision=FinalDecision(
            verdict=VerificationVerdict.UNSUPPORTED,
            confidence=0.8,
            rationale="No candidate matched.",
            subtest_results={"doi_resolves": False},
        ),
    )

    prediction = prediction_from_trace(trace)

    assert prediction["bibtex_key"] == "a3f9c2b1d4e76f85"
    assert prediction["label"] == "HALLUCINATED"
    assert prediction["confidence"] == 0.8
