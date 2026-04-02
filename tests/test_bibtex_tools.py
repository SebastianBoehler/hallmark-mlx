import subprocess
from pathlib import Path

from hallmark_mlx.tools.bibtex_updater import check_bibtex, summarize_result


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
