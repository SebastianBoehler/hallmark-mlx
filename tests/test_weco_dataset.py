from hallmark_mlx.eval.weco_dataset import build_weco_splits, hallmark_entry_to_trace


def test_hallmark_entry_to_trace_maps_gold_fields() -> None:
    entry = {
        "bibtex_key": "abc123",
        "bibtex_type": "inproceedings",
        "fields": {
            "author": "Alice Example and Bob Example",
            "title": "A Study on Verification",
            "year": "2024",
            "doi": "10.1234/example",
            "booktitle": "ICML",
        },
        "label": "HALLUCINATED",
        "hallucination_type": "fabricated_doi",
        "difficulty_tier": 2,
        "explanation": "Fabricated DOI.",
        "subtests": {"doi_resolves": False},
    }

    trace = hallmark_entry_to_trace(entry, split_name="dev_public")

    assert trace.input.benchmark_bibtex_key == "abc123"
    assert trace.parsed_fields.authors == ["Alice Example", "Bob Example"]
    assert trace.final_decision is not None
    assert trace.final_decision.verdict.value == "hallucinated"


def test_build_weco_splits_is_disjoint_and_sized() -> None:
    entries = []
    index = 0
    for bucket, count in (
        ("VALID_DOI", 16),
        ("VALID_NO_DOI", 16),
        ("H_TIER1", 20),
        ("H_TIER2", 40),
        ("H_TIER3", 24),
    ):
        for j in range(count):
            entry = {
                "bibtex_key": f"k{index:03d}",
                "bibtex_type": "inproceedings",
                "fields": {
                    "author": f"Author {index}",
                    "title": f"Title {index}",
                    "year": "2024",
                    "booktitle": f"Venue {j % 5}",
                },
                "label": "VALID" if bucket.startswith("VALID") else "HALLUCINATED",
                "difficulty_tier": None if bucket.startswith("VALID") else int(bucket[-1]),
                "hallucination_type": None if bucket.startswith("VALID") else f"type_{j % 7}",
                "explanation": "Example",
                "subtests": {},
            }
            if bucket == "VALID_DOI":
                entry["fields"]["doi"] = f"10.1000/{index}"
            entries.append(entry)
            index += 1

    search_traces, compare_traces, manifest = build_weco_splits(entries, split_name="dev_public")

    assert len(search_traces) == 64
    assert len(compare_traces) == 32
    assert manifest["search_count"] == 64
    assert manifest["compare_count"] == 32
    search_keys = {trace.input.benchmark_bibtex_key for trace in search_traces}
    compare_keys = {trace.input.benchmark_bibtex_key for trace in compare_traces}
    assert search_keys.isdisjoint(compare_keys)
