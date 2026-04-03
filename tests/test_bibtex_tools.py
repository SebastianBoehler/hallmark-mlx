import subprocess
from pathlib import Path

from hallmark_mlx.tools.bibtex_updater import (
    check_bibtex,
    extract_status_counts,
    primary_status,
    summarize_result,
)
from hallmark_mlx.training.prompts import build_available_tools_prompt


def test_check_bibtex_builds_expected_command(monkeypatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fake_run(command, capture_output, check, text):  # type: ignore[no-untyped-def]
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    bib_path = tmp_path / "refs.bib"
    bib_path.write_text("@article{test,title={Test}}\n", encoding="utf-8")

    result = check_bibtex(bib_path, strict=True)

    assert result.returncode == 0
    assert commands[0][-1] == "--strict"


def test_summarize_result_is_stable() -> None:
    summary = summarize_result(
        result=type(
            "Result",
            (),
            {
                "returncode": 0,
                "stdout": "ok",
                "stderr": "",
                "report": {"summary": {"issues_detected": 2, "checked_entries": 4}},
            },
        )(),
    )

    assert summary["ok"] is True
    assert summary["issues_detected"] == 2
    assert summary["candidate_count"] == 4
    assert summary["status"] is None


def test_bibtex_status_helpers_parse_summary_output() -> None:
    output = (
        "INFO: SUMMARY: 1 entries checked\n"
        "INFO: By status:\n"
        "INFO:   VERIFIED: 1\n"
        "INFO:   VENUE_MISMATCH: 2\n"
    )

    assert extract_status_counts(output) == {"VERIFIED": 1, "VENUE_MISMATCH": 2}
    assert primary_status(output) == "VENUE_MISMATCH"


def test_check_bibtex_ignores_empty_report_file(monkeypatch, tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"

    def fake_run(command, capture_output, check, text):  # type: ignore[no-untyped-def]
        report_path.write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="parse error")

    monkeypatch.setattr("subprocess.run", fake_run)

    result = check_bibtex(
        "@inproceedings{broken,title={Broken}",
        report_path=report_path,
    )

    assert result.returncode == 1
    assert result.report is None
    assert result.stderr == "parse error"


def test_available_tools_prompt_lists_required_arguments_and_verdicts() -> None:
    prompt = build_available_tools_prompt()

    assert 'bibtex_updater.check_bibtex requires {"bibtex": "..."} or {"path": "..."}' in prompt
    assert 'crossref.resolve_doi requires {"doi": "..."}' in prompt
    assert 'dblp.search_works requires {"query": "...", "rows": N}' in prompt
    assert (
        'acl_anthology.resolve_record requires one of {"anthology_id": "..."}, {"doi": "..."}, '
        'or {"url": "..."}' in prompt
    )
    assert (
        "valid final verdict values are exactly: verified, corrected, hallucinated, "
        "unsupported, abstain" in prompt
    )
