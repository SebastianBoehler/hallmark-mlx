"""Metadata comparison helpers for deterministic citation finalization."""

from __future__ import annotations

from html import unescape
import unicodedata

from hallmark_mlx.data.schemas import CandidateMatch, ParsedBibliographicFields


def normalize_text(value: str | None) -> str:
    """Normalize text for coarse metadata comparison."""

    unescaped = unescape(value or "")
    normalized = unicodedata.normalize("NFKD", unescaped)
    ascii_folded = "".join(character for character in normalized if not unicodedata.combining(character))
    return "".join(
        character.lower()
        for character in ascii_folded
        if character.isalnum() or character.isspace()
    ).strip()


def normalized_doi(value: str | None) -> str:
    """Normalize DOI strings for stable equality checks."""

    return (value or "").strip().lower()


def token_set(value: str | None) -> set[str]:
    """Return normalized whitespace tokens."""

    return {token for token in normalize_text(value).split() if token}


def _prefix_compatible(left: str, right: str) -> bool:
    short, long = sorted((left, right), key=len)
    return len(short) >= 1 and long.startswith(short)


def venues_compatible(left: str | None, right: str | None) -> bool:
    """Return true when venue strings plausibly refer to the same outlet."""

    normalized_left = normalize_text(left)
    normalized_right = normalize_text(right)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left in normalized_right or normalized_right in normalized_left:
        return True
    left_tokens = token_set(normalized_left)
    right_tokens = token_set(normalized_right)
    if not left_tokens or not right_tokens:
        return False
    return all(
        any(_prefix_compatible(left_token, right_token) for right_token in right_tokens)
        for left_token in left_tokens
    ) or all(
        any(_prefix_compatible(right_token, left_token) for left_token in left_tokens)
        for right_token in right_tokens
    )


def _normalized_surnames(authors: list[str]) -> set[str]:
    surnames: set[str] = set()
    for author in authors:
        parts = [part for part in normalize_text(author).split() if part]
        if parts:
            surnames.add(parts[-1])
    return surnames


def authors_compatible(
    parsed_authors: list[str],
    candidate_authors: list[str],
    *,
    require_complete_list: bool = False,
) -> bool:
    """Check whether the cited author list is covered by the candidate record."""

    if not parsed_authors or not candidate_authors:
        return True
    parsed_surnames = _normalized_surnames(parsed_authors)
    candidate_surnames = _normalized_surnames(candidate_authors)
    if require_complete_list:
        return parsed_surnames == candidate_surnames
    return parsed_surnames.issubset(candidate_surnames)


def field_mismatches(
    fields: ParsedBibliographicFields,
    candidate: CandidateMatch,
    *,
    require_complete_authors: bool = False,
) -> set[str]:
    """Return the material metadata mismatches between parsed fields and one candidate."""

    mismatches: set[str] = set()
    if fields.doi and candidate.doi and normalized_doi(fields.doi) != normalized_doi(candidate.doi):
        mismatches.add("doi")
    if fields.year and candidate.year and fields.year != candidate.year:
        mismatches.add("year")
    if fields.venue and candidate.venue and not venues_compatible(fields.venue, candidate.venue):
        mismatches.add("venue")
    if fields.title and candidate.title:
        parsed_title = normalize_text(fields.title)
        candidate_title = normalize_text(candidate.title)
        if parsed_title and candidate_title and parsed_title != candidate_title:
            overlap = len(token_set(parsed_title) & token_set(candidate_title))
            if overlap < 3:
                mismatches.add("title")
    if fields.authors and candidate.authors and not authors_compatible(
        fields.authors,
        candidate.authors,
        require_complete_list=require_complete_authors,
    ):
        mismatches.add("authors")
    return mismatches


def metadata_match_score(
    fields: ParsedBibliographicFields,
    candidate: CandidateMatch,
    *,
    require_complete_authors: bool = False,
) -> float:
    """Return a coarse confidence score for one candidate record."""

    score = candidate.score * 0.15
    parsed_title = normalize_text(fields.title)
    candidate_title = normalize_text(candidate.title)
    if parsed_title and candidate_title:
        if parsed_title == candidate_title:
            score += 0.6
        else:
            overlap = len(token_set(parsed_title) & token_set(candidate_title))
            if overlap >= 3:
                score += 0.2
            elif overlap == 0:
                score -= 0.4
    if fields.year and candidate.year:
        if fields.year == candidate.year:
            score += 0.25
        elif abs(fields.year - candidate.year) == 1:
            score += 0.05
        else:
            score -= 0.35
    if fields.venue and candidate.venue:
        score += 0.2 if venues_compatible(fields.venue, candidate.venue) else -0.1
    if fields.authors and candidate.authors:
        score += (
            0.25
            if authors_compatible(
                fields.authors,
                candidate.authors,
                require_complete_list=require_complete_authors,
            )
            else -0.2
        )
    return min(score, 1.5)
