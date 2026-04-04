from hallmark_mlx.config import ModelConfig
from hallmark_mlx.data.schemas import CandidateMatch, ToolResultSummary, VerificationInput
from hallmark_mlx.inference.bibtex_first_fallback_policy import BibtexFirstFallbackPolicyModel
from hallmark_mlx.inference.policy_runner import PolicyRunner
from hallmark_mlx.types import EvidenceStrength, FinalizationMode, InputType, ToolName


class FakeToolExecutor:
    def __init__(self, results: list[ToolResultSummary]) -> None:
        self._results = results

    def execute(self, tool_call):  # noqa: ANN001
        if not self._results:
            raise AssertionError(f"Unexpected tool call: {tool_call}")
        return self._results.pop(0)


def test_bibtex_first_fallback_overrides_false_positive_bibtex_mismatch() -> None:
    model = BibtexFirstFallbackPolicyModel(
        ModelConfig(max_rollout_rounds=3, force_bibtex_updater_first=True)
    )
    tool_executor = FakeToolExecutor(
        [
            ToolResultSummary(
                tool=ToolName.BIBTEX_UPDATER,
                action="check_bibtex",
                ok=True,
                evidence_strength=EvidenceStrength.MODERATE,
                raw_payload={"status": "VENUE_MISMATCH", "status_counts": {"VENUE_MISMATCH": 1}},
            ),
            ToolResultSummary(
                tool=ToolName.CROSSREF,
                action="resolve_doi",
                ok=True,
                evidence_strength=EvidenceStrength.STRONG,
                candidate_count=1,
                candidate_summaries=[
                    CandidateMatch(
                        source="crossref",
                        title="Convex optimization with an interpolation-based projection and its application to deep learning",
                        authors=["Riad Akrour", "Asma Atamna", "Jan Peters"],
                        year=2021,
                        venue="Machine Learning",
                        doi="10.1007/S10994-021-06037-Z",
                        score=1.0,
                    )
                ],
            ),
        ]
    )
    trace = PolicyRunner(
        model=model,
        tool_executor=tool_executor,
        finalization_mode=FinalizationMode.DETERMINISTIC,
    ).run(
        VerificationInput(
            record_id="valid-venue",
            input_type=InputType.BIBTEX_ENTRY,
            raw_input=(
                "@inproceedings{valid-venue,title={Convex optimization with an interpolation-based "
                "projection and its application to deep learning},author={Riad Akrour and Asma "
                "Atamna and Jan Peters},year={2021},doi={10.1007/S10994-021-06037-Z},"
                "booktitle={Mach. Learn.}}"
            ),
        )
    )

    assert trace.final_decision is not None
    assert trace.final_decision.verdict.value == "verified"
    assert len(trace.tool_results) == 2


def test_bibtex_first_fallback_marks_future_date_as_hallucinated() -> None:
    model = BibtexFirstFallbackPolicyModel(
        ModelConfig(max_rollout_rounds=2, force_bibtex_updater_first=True)
    )
    tool_executor = FakeToolExecutor(
        [
            ToolResultSummary(
                tool=ToolName.BIBTEX_UPDATER,
                action="check_bibtex",
                ok=True,
                evidence_strength=EvidenceStrength.MODERATE,
                raw_payload={"status": "FUTURE_DATE", "status_counts": {"FUTURE_DATE": 1}},
            )
        ]
    )
    trace = PolicyRunner(
        model=model,
        tool_executor=tool_executor,
        finalization_mode=FinalizationMode.DETERMINISTIC,
    ).run(
        VerificationInput(
            record_id="future-date",
            input_type=InputType.BIBTEX_ENTRY,
            raw_input="@inproceedings{future-date,title={A Statistical Theory of Cold Posteriors in Deep Neural Networks},author={Laurence Aitchison},year={2032},booktitle={ICLR}}",
        )
    )

    assert trace.final_decision is not None
    assert trace.final_decision.verdict.value == "hallucinated"
    assert len(trace.tool_results) == 1


def test_bibtex_status_is_used_even_when_tool_returns_nonzero() -> None:
    model = BibtexFirstFallbackPolicyModel(
        ModelConfig(max_rollout_rounds=2, force_bibtex_updater_first=True)
    )
    tool_executor = FakeToolExecutor(
        [
            ToolResultSummary(
                tool=ToolName.BIBTEX_UPDATER,
                action="check_bibtex",
                ok=False,
                evidence_strength=EvidenceStrength.NONE,
                raw_payload={"status": "HALLUCINATED", "status_counts": {"HALLUCINATED": 1}},
            )
        ]
    )
    trace = PolicyRunner(
        model=model,
        tool_executor=tool_executor,
        finalization_mode=FinalizationMode.DETERMINISTIC,
    ).run(
        VerificationInput(
            record_id="strict-hallucinated",
            input_type=InputType.BIBTEX_ENTRY,
            raw_input="@inproceedings{strict-hallucinated,title={Fabricated Paper},author={Alice Example},year={2024},booktitle={NeurIPS}}",
        )
    )

    assert trace.final_decision is not None
    assert trace.final_decision.verdict.value == "hallucinated"
