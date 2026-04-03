"""OpenAlex search wrapper."""

from __future__ import annotations

from urllib.parse import urlencode

from hallmark_mlx.data.schemas import CandidateMatch
from hallmark_mlx.tools.http import build_user_agent, fetch_json

_BASE_URL = "https://api.openalex.org/works"


def _candidate_from_item(item: dict, *, rationale: str, score: float) -> CandidateMatch:
    authors = [
        author.get("author", {}).get("display_name")
        for author in item.get("authorships", [])
        if author.get("author", {}).get("display_name")
    ]
    primary_location = item.get("primary_location") or {}
    source = primary_location.get("source") or {}
    return CandidateMatch(
        source="openalex",
        title=item.get("title"),
        authors=authors,
        year=item.get("publication_year"),
        venue=source.get("display_name"),
        doi=item.get("doi"),
        url=item.get("id"),
        score=score,
        rationale=rationale,
    )


def lookup_work_by_doi(
    doi: str,
    *,
    timeout: float = 15.0,
    email: str | None = None,
) -> list[CandidateMatch]:
    """Resolve a single OpenAlex work by DOI."""

    normalized = doi.strip().lower()
    params = {"filter": f"doi:{normalized}", "per-page": 1}
    if email:
        params["mailto"] = email
    url = f"{_BASE_URL}?{urlencode(params)}"
    user_agent = build_user_agent(email)
    payload = fetch_json(url, timeout=timeout, user_agent=user_agent)
    items = payload.get("results", [])
    if not items:
        return []
    return [_candidate_from_item(items[0], rationale="OpenAlex DOI lookup result.", score=1.0)]


def search_works(
    query: str,
    *,
    per_page: int = 5,
    timeout: float = 15.0,
    email: str | None = None,
) -> list[CandidateMatch]:
    """Search OpenAlex for candidate works."""

    params = {"search": query, "per-page": per_page}
    if email:
        params["mailto"] = email
    url = f"{_BASE_URL}?{urlencode(params)}"
    user_agent = build_user_agent(email)
    payload = fetch_json(url, timeout=timeout, user_agent=user_agent)
    candidates: list[CandidateMatch] = []
    for item in payload.get("results", []):
        candidates.append(
            _candidate_from_item(
                item,
                rationale="OpenAlex work search result.",
                score=min(float(item.get("relevance_score", 0.0) or 0.0) / 100.0, 1.0),
            ),
        )
    return candidates
