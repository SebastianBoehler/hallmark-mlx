"""Build tracked Weco search and comparison splits from official HALLMARK entries."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from hallmark_mlx.data.schemas import (
    FinalDecision,
    ParsedBibliographicFields,
    VerificationInput,
    VerificationTrace,
)
from hallmark_mlx.types import InputType, VerificationAction, VerificationVerdict

JSONDict = dict[str, Any]

SEARCH_BUCKETS: tuple[tuple[str, int], ...] = (
    ("VALID_DOI", 8),
    ("VALID_NO_DOI", 8),
    ("H_TIER1", 8),
    ("H_TIER2", 24),
    ("H_TIER3", 16),
)
COMPARE_BUCKETS: tuple[tuple[str, int], ...] = (
    ("VALID_DOI", 8),
    ("VALID_NO_DOI", 8),
    ("H_TIER1", 4),
    ("H_TIER2", 6),
    ("H_TIER3", 6),
)


def _normalize_year(raw_year: str | None) -> int | None:
    if raw_year is None:
        return None
    try:
        return int(raw_year)
    except ValueError:
        return None


def _split_authors(raw_author: str | None) -> list[str]:
    if not raw_author:
        return []
    return [author.strip() for author in raw_author.split(" and ") if author.strip()]


def _bucket_name(entry: JSONDict) -> str:
    if entry["label"] == "VALID":
        return "VALID_DOI" if "doi" in entry.get("fields", {}) else "VALID_NO_DOI"
    tier = entry.get("difficulty_tier")
    return f"H_TIER{tier}"


def _diversity_key(entry: JSONDict) -> str:
    if entry["label"] == "VALID":
        fields = entry.get("fields", {})
        return str(
            fields.get("booktitle") or fields.get("journal") or fields.get("publisher") or "unknown"
        )
    return str(entry.get("hallucination_type") or entry.get("generation_method") or "unknown")


def _render_bibtex(entry: JSONDict) -> str:
    fields: JSONDict = dict(entry.get("fields", {}))
    ordered_keys = [
        "title",
        "author",
        "year",
        "doi",
        "booktitle",
        "journal",
        "volume",
        "number",
        "pages",
        "publisher",
        "url",
    ]
    remaining = sorted(key for key in fields if key not in ordered_keys)
    lines = [f"@{entry['bibtex_type']}{{{entry['bibtex_key']},"]
    for key in [*ordered_keys, *remaining]:
        if key not in fields:
            continue
        value = str(fields[key]).replace("\n", " ").strip()
        lines.append(f"  {key} = {{{value}}},")
    lines.append("}")
    return "\n".join(lines)


def _subtest_bools(entry: JSONDict) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for key, value in dict(entry.get("subtests", {})).items():
        if isinstance(value, bool):
            result[key] = value
    return result


def hallmark_entry_to_trace(entry: JSONDict, *, split_name: str) -> VerificationTrace:
    """Convert one official HALLMARK entry into a local gold trace."""

    fields: JSONDict = dict(entry.get("fields", {}))
    verdict = (
        VerificationVerdict.VERIFIED
        if entry["label"] == "VALID"
        else VerificationVerdict.HALLUCINATED
    )
    return VerificationTrace(
        policy_version="hallmark-benchmark-gold-v1",
        input=VerificationInput(
            record_id=str(entry["bibtex_key"]),
            input_type=InputType.BIBTEX_ENTRY,
            raw_input=_render_bibtex(entry),
            benchmark_bibtex_key=str(entry["bibtex_key"]),
            private_holdout=False,
            source_metadata={
                "hallmark_label": entry["label"],
                "hallmark_split": split_name,
                "difficulty_tier": entry.get("difficulty_tier"),
                "hallucination_type": entry.get("hallucination_type"),
            },
        ),
        parsed_fields=ParsedBibliographicFields(
            title=fields.get("title"),
            authors=_split_authors(fields.get("author")),
            year=_normalize_year(fields.get("year")),
            venue=fields.get("booktitle") or fields.get("journal"),
            doi=fields.get("doi"),
            url=fields.get("url"),
            bibtex_type=entry.get("bibtex_type"),
        ),
        next_action=VerificationAction.PARSE_INPUT,
        final_decision=FinalDecision(
            verdict=verdict,
            confidence=1.0,
            rationale=str(entry.get("explanation") or "Gold benchmark annotation."),
            should_update_bibtex=False,
            subtest_results=_subtest_bools(entry),
        ),
        metadata={"hallmark_entry": entry},
    )


def _select_diverse(entries: list[JSONDict], count: int) -> list[JSONDict]:
    groups: dict[str, list[JSONDict]] = defaultdict(list)
    for entry in sorted(entries, key=lambda item: str(item["bibtex_key"])):
        groups[_diversity_key(entry)].append(entry)
    ordered_groups = sorted(groups)
    selected: list[JSONDict] = []
    while len(selected) < count:
        progressed = False
        for name in ordered_groups:
            if not groups[name]:
                continue
            selected.append(groups[name].pop(0))
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError(f"Not enough entries to satisfy count={count}.")
    return selected


def _take_bucket(
    entries: list[JSONDict],
    bucket: str,
    count: int,
) -> tuple[list[JSONDict], list[JSONDict]]:
    matching = [entry for entry in entries if _bucket_name(entry) == bucket]
    if len(matching) < count:
        raise ValueError(f"Bucket {bucket} only has {len(matching)} entries, need {count}.")
    selected = _select_diverse(matching, count)
    selected_keys = {str(entry["bibtex_key"]) for entry in selected}
    remaining = [entry for entry in entries if str(entry["bibtex_key"]) not in selected_keys]
    return selected, remaining


def build_weco_splits(
    entries: list[JSONDict],
    *,
    split_name: str,
) -> tuple[list[VerificationTrace], list[VerificationTrace], JSONDict]:
    """Build disjoint Weco search and comparison splits."""

    remaining = list(entries)
    search_entries: list[JSONDict] = []
    compare_entries: list[JSONDict] = []

    for bucket, count in SEARCH_BUCKETS:
        selected, remaining = _take_bucket(remaining, bucket, count)
        search_entries.extend(selected)
    for bucket, count in COMPARE_BUCKETS:
        selected, remaining = _take_bucket(remaining, bucket, count)
        compare_entries.extend(selected)

    search_traces = [
        hallmark_entry_to_trace(entry, split_name=split_name) for entry in search_entries
    ]
    compare_traces = [
        hallmark_entry_to_trace(entry, split_name=split_name) for entry in compare_entries
    ]

    search_keys = {trace.input.benchmark_bibtex_key for trace in search_traces}
    compare_keys = {trace.input.benchmark_bibtex_key for trace in compare_traces}
    overlap = sorted(search_keys & compare_keys)
    if overlap:
        raise ValueError(f"Search and comparison splits overlap: {overlap[:3]}")

    manifest = {
        "source_split": split_name,
        "selection_strategy": {
            "search_buckets": dict(SEARCH_BUCKETS),
            "compare_buckets": dict(COMPARE_BUCKETS),
            "diversity_key": "venue for valid; hallucination_type for hallucinated",
        },
        "search_count": len(search_traces),
        "compare_count": len(compare_traces),
        "search_label_counts": Counter(
            trace.input.source_metadata["hallmark_label"] for trace in search_traces
        ),
        "compare_label_counts": Counter(
            trace.input.source_metadata["hallmark_label"] for trace in compare_traces
        ),
    }
    return search_traces, compare_traces, manifest
