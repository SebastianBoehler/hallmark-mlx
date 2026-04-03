"""Crossref search wrapper."""

from __future__ import annotations

from urllib.parse import quote, urlencode

from hallmark_mlx.data.schemas import CandidateMatch
from hallmark_mlx.tools.http import build_user_agent, fetch_json

_BASE_URL = "https://api.crossref.org/works"


def _extract_year(item: dict) -> int | None:
    for key in ("published-print", "published-online", "issued"):
        parts = item.get(key, {}).get("date-parts", [])
        if parts and parts[0] and parts[0][0] is not None:
            return int(parts[0][0])
    return None


def _candidate_from_item(item: dict, *, rationale: str, score: float) -> CandidateMatch:
    authors = []
    for author in item.get("author", []):
        name = " ".join(part for part in [author.get("given"), author.get("family")] if part)
        if name:
            authors.append(name)
    return CandidateMatch(
        source="crossref",
        title=(item.get("title") or [None])[0],
        authors=authors,
        year=_extract_year(item),
        venue=(item.get("container-title") or [None])[0],
        doi=item.get("DOI"),
        url=item.get("URL"),
        score=score,
        rationale=rationale,
    )


def lookup_work_by_doi(
    doi: str,
    *,
    timeout: float = 15.0,
    email: str | None = None,
) -> list[CandidateMatch]:
    """Resolve a single work by DOI."""

    user_agent = build_user_agent(email)
    if email:
        url = f"{_BASE_URL}/{quote(doi, safe='')}" + f"?{urlencode({'mailto': email})}"
    else:
        url = f"{_BASE_URL}/{quote(doi, safe='')}"
    payload = fetch_json(url, timeout=timeout, user_agent=user_agent)
    item = payload.get("message")
    if not isinstance(item, dict):
        return []
    return [_candidate_from_item(item, rationale="Crossref DOI lookup result.", score=1.0)]


def search_works(
    query: str,
    *,
    rows: int = 5,
    timeout: float = 15.0,
    email: str | None = None,
) -> list[CandidateMatch]:
    """Search Crossref for bibliographic candidates."""

    params = {"query.bibliographic": query, "rows": rows}
    if email:
        params["mailto"] = email
    url = f"{_BASE_URL}?{urlencode(params)}"
    user_agent = build_user_agent(email)
    payload = fetch_json(url, timeout=timeout, user_agent=user_agent)
    items = payload.get("message", {}).get("items", [])
    candidates: list[CandidateMatch] = []
    for item in items:
        raw_score = float(item.get("score", 0.0) or 0.0)
        candidates.append(
            _candidate_from_item(
                item,
                rationale="Crossref bibliographic search result.",
                score=min(raw_score / 100.0, 1.0),
            ),
        )
    return candidates
