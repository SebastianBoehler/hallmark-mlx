from hallmark_mlx.config import ToolsConfig
from hallmark_mlx.data.schemas import CandidateMatch, ToolInvocation
from hallmark_mlx.inference.tool_executor import ToolExecutor
from hallmark_mlx.tools.acl_anthology import extract_anthology_id, resolve_record
from hallmark_mlx.tools.arxiv import extract_arxiv_id, resolve_record as resolve_arxiv_record
from hallmark_mlx.tools.dblp import search_works
from hallmark_mlx.types import ToolName


def test_dblp_search_parses_single_hit_dict(monkeypatch) -> None:
    def fake_fetch_json(url: str, *, timeout: float, user_agent: str) -> dict:
        assert "dblp.org/search/publ/api" in url
        return {
            "result": {
                "hits": {
                    "hit": {
                        "info": {
                            "title": "Attention Is All You Need",
                            "authors": {"author": ["Ashish Vaswani", "Noam Shazeer"]},
                            "venue": "NeurIPS",
                            "year": "2017",
                            "doi": "10.5555/3295222.3295349",
                            "url": "https://dblp.org/rec/conf/nips/VaswaniSPUJGKP17",
                        },
                    },
                },
            },
        }

    monkeypatch.setattr("hallmark_mlx.tools.dblp.fetch_json", fake_fetch_json)

    candidates = search_works("Attention Is All You Need Vaswani 2017", rows=3)

    assert len(candidates) == 1
    assert candidates[0].title == "Attention Is All You Need"
    assert candidates[0].venue == "NeurIPS"
    assert candidates[0].doi == "10.5555/3295222.3295349"
    assert candidates[0].score > 0.8


def test_acl_anthology_resolve_record_parses_bib(monkeypatch) -> None:
    bib_text = (
        "@inproceedings{devlin-etal-2019-bert,\n"
        "  title = {BERT: Pre-training of Deep Bidirectional Transformers for Language "
        "Understanding},\n"
        "  author = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, "
        "Kristina},\n"
        "  booktitle = {Proceedings of the 2019 Conference of the North American Chapter "
        "of the Association for Computational Linguistics},\n"
        "  year = {2019},\n"
        "  doi = {10.18653/v1/N19-1423},\n"
        "  url = {https://aclanthology.org/N19-1423}\n"
        "}"
    )

    monkeypatch.setattr(
        "hallmark_mlx.tools.acl_anthology.fetch_text",
        lambda *args, **kwargs: bib_text,
    )

    candidates = resolve_record(doi="10.18653/v1/N19-1423")

    assert extract_anthology_id("10.18653/v1/N19-1423") == "N19-1423"
    assert len(candidates) == 1
    assert (
        candidates[0].title
        == "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    )
    assert candidates[0].year == 2019
    assert candidates[0].doi == "10.18653/v1/N19-1423"


def test_tool_executor_supports_dblp_and_acl(monkeypatch) -> None:
    monkeypatch.setattr(
        "hallmark_mlx.inference.tool_executor.search_dblp_works",
        lambda query, rows, timeout: [
            CandidateMatch(source="dblp", title="Attention Is All You Need", score=0.95),
        ],
    )
    monkeypatch.setattr(
        "hallmark_mlx.inference.tool_executor.resolve_acl_record",
        lambda anthology_id=None, doi=None, url=None, timeout=15.0: [
            CandidateMatch(
                source="acl_anthology",
                title="BERT",
                doi="10.18653/v1/N19-1423",
                score=1.0,
            ),
        ],
    )

    executor = ToolExecutor(ToolsConfig())
    dblp_result = executor.execute(
        ToolInvocation(
            tool=ToolName.DBLP,
            action="search_works",
            arguments={"query": "Attention Is All You Need Vaswani 2017", "rows": 3},
        ),
    )
    acl_result = executor.execute(
        ToolInvocation(
            tool=ToolName.ACL_ANTHOLOGY,
            action="resolve_record",
            arguments={"doi": "10.18653/v1/N19-1423"},
        ),
    )

    assert dblp_result.ok is True
    assert dblp_result.candidate_count == 1
    assert dblp_result.evidence_strength.value == "strong"
    assert acl_result.ok is True
    assert acl_result.matched_identifiers["anthology_id"] == "N19-1423"
    assert acl_result.matched_identifiers["doi"] == "10.18653/v1/N19-1423"


def test_arxiv_resolve_record_parses_atom(monkeypatch) -> None:
    atom = (
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<feed xmlns='http://www.w3.org/2005/Atom'>"
        "<entry>"
        "<id>http://arxiv.org/abs/2602.12124v1</id>"
        "<published>2026-02-17T00:00:00Z</published>"
        "<title>Capability-Oriented Training Induced Alignment Risk</title>"
        "<author><name>Yujun Zhou</name></author>"
        "<author><name>Yue Huang</name></author>"
        "</entry>"
        "</feed>"
    )

    monkeypatch.setattr("hallmark_mlx.tools.arxiv.fetch_text", lambda *args, **kwargs: atom)

    candidates = resolve_arxiv_record(doi="10.48550/arXiv.2602.12124v1")

    assert extract_arxiv_id("10.48550/arXiv.2602.12124v1") == "2602.12124v1"
    assert len(candidates) == 1
    assert candidates[0].title == "Capability-Oriented Training Induced Alignment Risk"
    assert candidates[0].year == 2026
    assert candidates[0].venue == "arXiv"


def test_tool_executor_supports_arxiv(monkeypatch) -> None:
    monkeypatch.setattr(
        "hallmark_mlx.inference.tool_executor.resolve_arxiv_record",
        lambda arxiv_id=None, doi=None, url=None, timeout=15.0: [
            CandidateMatch(
                source="arxiv",
                title="Capability-Oriented Training Induced Alignment Risk",
                year=2026,
                url="http://arxiv.org/abs/2602.12124v1",
                score=1.0,
            ),
        ],
    )

    executor = ToolExecutor(ToolsConfig())
    result = executor.execute(
        ToolInvocation(
            tool=ToolName.ARXIV,
            action="resolve_record",
            arguments={"doi": "10.48550/arXiv.2602.12124v1"},
        ),
    )

    assert result.ok is True
    assert result.candidate_count == 1
    assert result.evidence_strength.value == "strong"
