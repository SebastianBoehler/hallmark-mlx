"""Conservative finalization logic for verification traces."""

from __future__ import annotations

from collections import Counter

from hallmark_mlx.data.schemas import CandidateMatch, CandidateRanking, FinalDecision, VerificationTrace
from hallmark_mlx.types import VerificationVerdict


def _normalize_text(value: str | None) -> str:
    return "".join(character.lower() for character in (value or "") if character.isalnum() or character.isspace()).strip()


def _token_set(value: str | None) -> set[str]:
    return {token for token in _normalize_text(value).split() if token}


def _metadata_match_score(trace: VerificationTrace, candidate: CandidateMatch) -> float:
    score = candidate.score * 0.15
    parsed = trace.parsed_fields

    parsed_title = _normalize_text(parsed.title)
    candidate_title = _normalize_text(candidate.title)
    if parsed_title and candidate_title:
        if parsed_title == candidate_title:
            score += 0.6
        else:
            overlap = len(_token_set(parsed_title) & _token_set(candidate_title))
            if overlap >= 3:
                score += 0.2
            elif overlap == 0:
                score -= 0.4

    if parsed.year and candidate.year:
        if parsed.year == candidate.year:
            score += 0.25
        elif abs(parsed.year - candidate.year) == 1:
            score += 0.05
        else:
            score -= 0.35

    if parsed.venue and candidate.venue:
        parsed_venue = _normalize_text(parsed.venue)
        candidate_venue = _normalize_text(candidate.venue)
        if parsed_venue and candidate_venue and (
            parsed_venue in candidate_venue or candidate_venue in parsed_venue
        ):
            score += 0.2
        else:
            score -= 0.05

    if parsed.authors and candidate.authors:
        parsed_surname = parsed.authors[0].split()[-1].lower()
        candidate_surnames = {author.split()[-1].lower() for author in candidate.authors if author.split()}
        if parsed_surname in candidate_surnames:
            score += 0.25
        else:
            score -= 0.2

    return min(score, 1.5)


def finalize_trace(trace: VerificationTrace) -> VerificationTrace:
    """Turn tool outputs into a conservative final decision."""

    if trace.final_decision is not None:
        return trace

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
            rationale="No successful verification tools returned evidence.",
            abstain_reason="no_tool_evidence",
        )
        return trace.model_copy(update={"final_decision": decision})

    if not candidates:
        decision = FinalDecision(
            verdict=VerificationVerdict.UNSUPPORTED,
            confidence=0.72,
            rationale="Tooling ran, but no candidate record supported the citation.",
            subtest_results={"title_exists": False},
        )
        return trace.model_copy(update={"final_decision": decision})

    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: _metadata_match_score(trace, candidate),
        reverse=True,
    )
    doi_votes = Counter(candidate.doi.lower() for candidate in ranked_candidates if candidate.doi)
    candidate_ranking = CandidateRanking(candidates=ranked_candidates, preferred_index=0)

    if doi_votes:
        preferred_doi, support_count = doi_votes.most_common(1)[0]
        if support_count >= 2:
            decision = FinalDecision(
                verdict=VerificationVerdict.VERIFIED,
                confidence=min(0.7 + (0.1 * support_count), 0.95),
                rationale=f"Multiple sources agree on DOI {preferred_doi}.",
                subtest_results={
                    "doi_resolves": True,
                    "cross_db_agreement": True,
                },
            )
            candidate_ranking.rationale = "Top DOI is supported by multiple tools."
            return trace.model_copy(
                update={
                    "candidate_ranking": candidate_ranking,
                    "final_decision": decision,
                },
            )

    top_candidate = ranked_candidates[0]
    top_match_score = _metadata_match_score(trace, top_candidate)
    if top_match_score >= 1.0:
        decision = FinalDecision(
            verdict=VerificationVerdict.VERIFIED,
            confidence=min(0.7 + (0.15 * min(top_match_score, 1.5)), 0.94),
            rationale="The top candidate aligns strongly with the parsed citation metadata.",
            subtest_results={
                "title_exists": True,
                "metadata_alignment": True,
            },
        )
        candidate_ranking.rationale = "Top candidate aligns strongly with the parsed citation metadata."
        return trace.model_copy(
            update={
                "candidate_ranking": candidate_ranking,
                "final_decision": decision,
            },
        )

    if len(ranked_candidates) == 1 and top_candidate.score >= 0.85:
        decision = FinalDecision(
            verdict=VerificationVerdict.VERIFIED,
            confidence=0.82,
            rationale="A single high-confidence candidate match was retrieved.",
            subtest_results={"title_exists": True},
        )
        candidate_ranking.rationale = "Single high-confidence candidate."
        return trace.model_copy(
            update={
                "candidate_ranking": candidate_ranking,
                "final_decision": decision,
            },
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
        update={
            "candidate_ranking": candidate_ranking,
            "final_decision": decision,
        },
    )
