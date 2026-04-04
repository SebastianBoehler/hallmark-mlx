"""Deterministic verdict engine for citation verification traces."""

from __future__ import annotations

from collections import Counter

from hallmark_mlx.data.schemas import (
    CandidateMatch,
    CandidateRanking,
    FinalDecision,
    ParsedBibliographicFields,
    VerificationTrace,
)
from hallmark_mlx.inference.bibtex_verdicts import decision_from_bibtex_status
from hallmark_mlx.inference.metadata_match import (
    field_mismatches,
    metadata_match_score,
    normalized_doi,
)
from hallmark_mlx.inference.warm_start_policy import parse_input
from hallmark_mlx.types import InputType, VerificationVerdict


def _has_parsed_signal(fields: ParsedBibliographicFields) -> bool:
    return any(
        (
            fields.title,
            fields.authors,
            fields.year,
            fields.venue,
            fields.doi,
            fields.arxiv_id,
            fields.url,
            fields.claim_text,
        )
    )


def _effective_fields(trace: VerificationTrace) -> ParsedBibliographicFields:
    if _has_parsed_signal(trace.parsed_fields):
        return trace.parsed_fields
    return parse_input(trace.input)



def finalize_trace(trace: VerificationTrace, *, force: bool = False) -> VerificationTrace:
    """Derive a deterministic final verdict from tool evidence."""

    if trace.final_decision is not None and not force:
        return trace

    fields = _effective_fields(trace)
    trace = trace.model_copy(update={"parsed_fields": fields})
    bibtex_decision = decision_from_bibtex_status(trace, fields)
    if bibtex_decision is not None:
        return trace.model_copy(update={"final_decision": bibtex_decision})

    successful_results = [result for result in trace.tool_results if result.ok]
    candidates = [
        candidate
        for result in successful_results
        for candidate in result.candidate_summaries
    ]
    if not successful_results:
        decision = FinalDecision(
            verdict=VerificationVerdict.ABSTAIN,
            confidence=0.25,
            rationale="No successful verification tools returned usable evidence.",
            abstain_reason="no_tool_evidence",
        )
        return trace.model_copy(update={"final_decision": decision})
    if not candidates:
        decision = FinalDecision(
            verdict=VerificationVerdict.ABSTAIN,
            confidence=0.35,
            rationale="Tools ran but returned no candidate records; insufficient evidence to decide.",
            abstain_reason="no_candidates_returned",
        )
        return trace.model_copy(update={"final_decision": decision})

    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: metadata_match_score(
            fields,
            candidate,
            require_complete_authors=(
                trace.input.input_type == InputType.BIBTEX_ENTRY and bool(fields.doi)
            ),
        ),
        reverse=True,
    )
    candidate_ranking = CandidateRanking(candidates=ranked_candidates, preferred_index=0)
    doi_votes = Counter(candidate.doi.lower() for candidate in ranked_candidates if candidate.doi)
    top_candidate = ranked_candidates[0]
    require_complete_authors = trace.input.input_type == InputType.BIBTEX_ENTRY and bool(fields.doi)
    mismatches = field_mismatches(
        fields,
        top_candidate,
        require_complete_authors=require_complete_authors,
    )
    top_match_score = metadata_match_score(
        fields,
        top_candidate,
        require_complete_authors=require_complete_authors,
    )

    if doi_votes:
        preferred_doi, support_count = doi_votes.most_common(1)[0]
        if support_count >= 2:
            doi_candidates = [
                candidate
                for candidate in ranked_candidates
                if candidate.doi and candidate.doi.lower() == preferred_doi
            ]
            representative = doi_candidates[0] if doi_candidates else top_candidate
            mismatches = field_mismatches(
                fields,
                representative,
                require_complete_authors=require_complete_authors,
            )
            if not mismatches:
                decision = FinalDecision(
                    verdict=VerificationVerdict.VERIFIED,
                    confidence=min(0.78 + (0.07 * support_count), 0.96),
                    rationale=f"Multiple sources agree on DOI {preferred_doi} and metadata aligns.",
                    subtest_results={
                        "doi_resolves": True,
                        "cross_db_agreement": True,
                        "metadata_alignment": True,
                    },
                )
                candidate_ranking.rationale = "Multiple tools agree on the same DOI and metadata."
                return trace.model_copy(
                    update={"candidate_ranking": candidate_ranking, "final_decision": decision},
                )
            if mismatches & {"venue", "year", "title", "authors"}:
                decision = FinalDecision(
                    verdict=VerificationVerdict.HALLUCINATED,
                    confidence=0.9,
                    rationale=(
                        f"Multiple sources agree on DOI {preferred_doi}, but the cited metadata "
                        f"mismatches the resolved record ({', '.join(sorted(mismatches))})."
                    ),
                    subtest_results={"doi_resolves": True, "metadata_alignment": False},
                )
                candidate_ranking.rationale = "Resolved DOI contradicts the cited metadata."
                return trace.model_copy(
                    update={"candidate_ranking": candidate_ranking, "final_decision": decision},
                )

    if fields.doi:
        matching_doi_candidates = [
            candidate
            for candidate in ranked_candidates
            if candidate.doi and normalized_doi(candidate.doi) == normalized_doi(fields.doi)
        ]
        if matching_doi_candidates:
            representative = matching_doi_candidates[0]
            mismatches = (
                field_mismatches(
                    fields,
                    representative,
                    require_complete_authors=require_complete_authors,
                )
                - {"doi"}
            )
            if not mismatches:
                decision = FinalDecision(
                    verdict=VerificationVerdict.VERIFIED,
                    confidence=0.9,
                    rationale="An external resolver returned the cited DOI with aligned metadata.",
                    subtest_results={"doi_resolves": True, "metadata_alignment": True},
                )
                candidate_ranking.rationale = "Resolved DOI record aligns with the cited entry."
                return trace.model_copy(
                    update={"candidate_ranking": candidate_ranking, "final_decision": decision},
                )

    if top_match_score >= 1.0 and not mismatches:
        decision = FinalDecision(
            verdict=VerificationVerdict.VERIFIED,
            confidence=min(0.7 + (0.15 * min(top_match_score, 1.5)), 0.94),
            rationale="The top candidate aligns strongly with the parsed citation metadata.",
            subtest_results={"title_exists": True, "metadata_alignment": True},
        )
        candidate_ranking.rationale = "Top candidate aligns strongly with the citation metadata."
        return trace.model_copy(
            update={"candidate_ranking": candidate_ranking, "final_decision": decision},
        )

    if len(ranked_candidates) == 1 and top_match_score >= 0.7 and not mismatches:
        decision = FinalDecision(
            verdict=VerificationVerdict.VERIFIED,
            confidence=0.81,
            rationale="A single strong candidate aligns with the cited metadata.",
            subtest_results={"title_exists": True, "metadata_alignment": True},
        )
        candidate_ranking.rationale = "Single strong candidate with no metadata contradictions."
        return trace.model_copy(
            update={"candidate_ranking": candidate_ranking, "final_decision": decision},
        )

    if top_match_score >= 0.8 and mismatches:
        verdict = (
            VerificationVerdict.HALLUCINATED
            if mismatches & {"venue", "year", "title", "authors"}
            else VerificationVerdict.CORRECTED
        )
        decision = FinalDecision(
            verdict=verdict,
            confidence=0.82,
            rationale=(
                "A strong candidate was found, but the cited metadata requires correction "
                f"({', '.join(sorted(mismatches))})."
            ),
            should_update_bibtex=verdict == VerificationVerdict.CORRECTED,
            subtest_results={"metadata_alignment": False},
        )
        candidate_ranking.rationale = "Top candidate is strong, but metadata mismatches remain."
        return trace.model_copy(
            update={"candidate_ranking": candidate_ranking, "final_decision": decision},
        )

    candidate_ranking.needs_disambiguation = True
    candidate_ranking.rationale = "Evidence exists, but the candidate set remains ambiguous."
    decision = FinalDecision(
        verdict=VerificationVerdict.ABSTAIN,
        confidence=0.46,
        rationale="Retrieved evidence is ambiguous and should not be collapsed into a hard verdict.",
        abstain_reason="ambiguous_candidate_set",
    )
    return trace.model_copy(
        update={"candidate_ranking": candidate_ranking, "final_decision": decision},
    )
