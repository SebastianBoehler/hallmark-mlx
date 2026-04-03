"""BibTeX Updater status to verdict mapping."""

from __future__ import annotations

from hallmark_mlx.data.schemas import FinalDecision, ParsedBibliographicFields, VerificationTrace
from hallmark_mlx.types import ToolName, VerificationVerdict

_HALLUCINATED_BIBTEX_STATUSES = {
    "AUTHOR_MISMATCH",
    "DOI_NOT_FOUND",
    "TITLE_MISMATCH",
    "VENUE_MISMATCH",
    "YEAR_MISMATCH",
}
_CORRECTABLE_BIBTEX_STATUSES = {"PARTIAL_MATCH"}


def _bibtex_status(trace: VerificationTrace) -> str | None:
    for result in trace.tool_results:
        if result.tool != ToolName.BIBTEX_UPDATER or not result.ok:
            continue
        raw_status = result.raw_payload.get("status")
        if isinstance(raw_status, str) and raw_status:
            return raw_status
    return None


def decision_from_bibtex_status(
    trace: VerificationTrace,
    fields: ParsedBibliographicFields,
) -> FinalDecision | None:
    """Convert BibTeX Updater summary status into a deterministic final verdict."""

    status = _bibtex_status(trace)
    if status is None:
        return None
    if status == "VERIFIED":
        return FinalDecision(
            verdict=VerificationVerdict.VERIFIED,
            confidence=0.97,
            rationale="BibTeX Updater verified the citation entry.",
            subtest_results={
                "doi_resolves": bool(fields.doi),
                "metadata_alignment": True,
                "cross_db_agreement": True,
            },
        )
    if status in _HALLUCINATED_BIBTEX_STATUSES:
        subtests = {
            "doi_resolves": status != "DOI_NOT_FOUND",
            "metadata_alignment": False,
        }
        if status == "VENUE_MISMATCH":
            subtests["venue_correct"] = False
        if status == "YEAR_MISMATCH":
            subtests["year_correct"] = False
        if status == "TITLE_MISMATCH":
            subtests["title_exists"] = False
        if status == "AUTHOR_MISMATCH":
            subtests["authors_match"] = False
        return FinalDecision(
            verdict=VerificationVerdict.HALLUCINATED,
            confidence=0.92,
            rationale=(
                "BibTeX Updater found a material metadata contradiction "
                f"({status.lower()})."
            ),
            subtest_results=subtests,
        )
    if status in _CORRECTABLE_BIBTEX_STATUSES:
        return FinalDecision(
            verdict=VerificationVerdict.CORRECTED,
            confidence=0.84,
            rationale="BibTeX Updater found a real paper with partially misaligned metadata.",
            should_update_bibtex=True,
            subtest_results={"metadata_alignment": False},
        )
    return None
