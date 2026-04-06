#!/usr/bin/env python3
"""Export a maintainer-facing HALLMARK submission packet."""

from __future__ import annotations

import argparse
import subprocess
from datetime import date
from pathlib import Path

from hallmark_mlx.eval.submission_packet import (
    OFFICIAL_SPLITS,
    ROOT,
    SUBMISSION_ROOT,
    SUBMISSION_SKILL,
    build_submission_entry,
    confirmed_split_paths,
    load_confirmed_outputs,
    shard_plan,
    validate_confirmed_row,
)
from hallmark_mlx.utils.io import ensure_dir, write_json, write_text
from hallmark_mlx.utils.jsonl import write_jsonl

OUTPUT_DIR = SUBMISSION_ROOT / "current"
OUTPUT_REPORT = ROOT / "docs" / "reports" / "hallmark_submission_packet.md"
HF_MODEL_URL = "https://huggingface.co/sebastianboehler/hallmark-mlx-qwen25-1.5b-lora"
HF_DATASET_URL = (
    "https://huggingface.co/datasets/sebastianboehler/"
    "hallmark-mlx-reviewed-policy-traces"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser


def _run_git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _git_version() -> tuple[str, bool]:
    commit = _run_git("rev-parse", "HEAD")
    dirty = bool(_run_git("status", "--porcelain"))
    return (f"{commit}-dirty" if dirty else commit), dirty


def _quoted(path: Path) -> str:
    return str(path).replace(" ", "\\ ")


def _repro_script(split: str, *, output_dir: Path) -> str:
    row, _ = load_confirmed_outputs(split)
    plan = shard_plan(int(row["num_entries"]))
    split_root = (
        "artifacts/official_eval_submission_repro/" f"{split}_bibtex_first_fallback"
    )
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"',
        'UPSTREAM_ROOT="${1:-/tmp/hallmark-upstream}"',
        'export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"',
        f'SPLIT="{split}"',
        f'SPLIT_ROOT="$ROOT/{split_root}"',
        'mkdir -p "$SPLIT_ROOT/shards"',
        "",
    ]
    for shard, offset, limit in plan:
        lines.extend(
            [
                'uv run python "$ROOT/scripts/run_official_controller_eval.py" \\',
                '  --upstream-root "$UPSTREAM_ROOT" \\',
                '  --split "$SPLIT" \\',
                "  --mode bibtex_first_fallback \\",
                f'  --output-dir "$SPLIT_ROOT/shards/shard_{shard:02d}" \\',
                "  --entry-timeout-seconds 5 \\",
                f"  --offset {offset} \\",
                f"  --limit {limit}",
                "",
            ]
        )
    lines.extend(
        [
            'description=$(',
            '  printf %s \\',
            '    "hallmark-mlx controller with deterministic finalizer " \\',
            '    "(5s per-entry timeout, 8-way sharded official run)"',
            ")",
            "",
            'uv run python "$ROOT/scripts/merge_official_eval_shards.py" \\',
            '  --upstream-root "$UPSTREAM_ROOT" \\',
            '  --split "$SPLIT" \\',
            '  --shards-root "$SPLIT_ROOT/shards" \\',
            '  --output-dir "$SPLIT_ROOT/merged" \\',
            "  --method-name hallmark_mlx_bibtex_first_fallback \\",
            '  --description "$description"',
            "",
        ]
    )
    script_path = output_dir / f"reproduce_{split}.sh"
    write_text(script_path, "\n".join(lines))
    script_path.chmod(0o755)
    return f"bash {_quoted(script_path)}"


def _summary_line(split: str, row: dict[str, object]) -> str:
    return (
        f"- `{split}`: DR {float(row['detection_rate']):.3f}, "
        f"F1-H {float(row['f1_hallucination']):.3f}, "
        f"TW-F1 {float(row['tier_weighted_f1']):.3f}, "
        f"FPR "
        + (
            "—"
            if row.get("false_positive_rate") is None
            else f"{float(row['false_positive_rate']):.3f}"
        )
        + f", ECE {float(row['ece']):.3f}"
    )


def _maintainer_message(
    *,
    output_dir: Path,
    version: str,
    dirty: bool,
    checks_by_split: dict[str, list[str]],
    rows: dict[str, dict[str, object]],
) -> str:
    warning = (
        "- Version note: this packet was exported from a dirty worktree. "
        "Re-run the exporter after the final submission commit if you want a clean hash."
        if dirty
        else "- Version note: this packet is pinned to a clean git commit."
    )
    check_lines = [
        f"  - `{split}`: " + ", ".join(checks)
        for split, checks in checks_by_split.items()
    ]
    return "\n".join(
        [
            "# HALLMARK Submission Packet",
            "",
            "This packet is ready to send to the HALLMARK / Research Agora maintainers.",
            "",
            "## Primary Public Row",
            "",
            _summary_line("test_public", rows["test_public"]),
            "",
            "## Supporting Official Rows",
            "",
            _summary_line("dev_public", rows["dev_public"]),
            _summary_line("stress_test", rows["stress_test"]),
            "",
            "## System",
            "",
            f"- Registry skill: `{SUBMISSION_SKILL}`",
            "- Method: `hallmark-mlx`",
            "- Runtime: BibTeX-first deterministic controller with scholarly fallback",
            "- Benchmark mode: `bibtex_first_fallback`",
            (
                "- Evaluation protocol: upstream official evaluator, "
                "5s per-entry timeout, 8-way sharding"
            ),
            f"- Version: `{version}`",
            warning,
            f"- Model bundle: {HF_MODEL_URL}",
            f"- Dataset bundle: {HF_DATASET_URL}",
            "",
            "## Protocol Checks",
            "",
            *check_lines,
            "",
            "## Hidden-Split Request",
            "",
            (
                "Please evaluate the same code path and commit on the hidden test split. "
                "The attached public row for leaderboard comparison is `test_public`; "
                "`dev_public` and `stress_test` are included as supporting context only."
            ),
            "",
            "## Included Files",
            "",
            f"- Results JSONL: `{output_dir / 'results.jsonl'}`",
            f"- Metadata: `{output_dir / 'metadata.json'}`",
            "- Reproduction scripts:",
            "  - `reproduce_dev_public.sh`",
            "  - `reproduce_test_public.sh`",
            "  - `reproduce_stress_test.sh`",
            "",
        ]
    )


def _report(
    *,
    output_dir: Path,
    version: str,
    dirty: bool,
    rows: dict[str, dict[str, object]],
) -> str:
    state = "dirty worktree" if dirty else "clean commit"
    return "\n".join(
        [
            "# HALLMARK Submission Packet",
            "",
            f"- Packet directory: `{output_dir}`",
            f"- Version: `{version}` ({state})",
            f"- Primary public submission row: `{confirmed_split_paths('test_public').row_path}`",
            "",
            "## Official Rows",
            "",
            *[_summary_line(split, rows[split]) for split in OFFICIAL_SPLITS],
            "",
            "## Included Artifacts",
            "",
            f"- `{output_dir / 'results.jsonl'}`",
            f"- `{output_dir / 'metadata.json'}`",
            f"- `{output_dir / 'MAINTAINER_MESSAGE.md'}`",
            "",
            "## Submission Guidance",
            "",
            "- Use `test_public` as the public row in submission notes.",
            "- Keep `dev_public` and `stress_test` as supporting context only.",
            "- Ask the maintainers to evaluate the same commit on the hidden split.",
            "",
        ]
    )


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    output_jsonl = output_dir / "results.jsonl"
    output_metadata = output_dir / "metadata.json"
    output_message = output_dir / "MAINTAINER_MESSAGE.md"
    version, dirty = _git_version()
    export_date = date.today().isoformat()

    entries = []
    rows: dict[str, dict[str, object]] = {}
    checks_by_split: dict[str, list[str]] = {}
    for split in OFFICIAL_SPLITS:
        row, result = load_confirmed_outputs(split)
        checks_by_split[split] = validate_confirmed_row(row)
        rows[split] = row
        reproduction = _repro_script(split, output_dir=output_dir)
        entries.append(
            build_submission_entry(
                split=split,
                row=row,
                result=result,
                version=version,
                date=export_date,
                reproduction=reproduction,
            )
        )

    write_jsonl(output_jsonl, entries)
    write_json(
        output_metadata,
        {
            "benchmark_id": "hallmark",
            "version": version,
            "dirty": dirty,
            "date": export_date,
            "primary_split": "test_public",
            "row_paths": {
                split: str(confirmed_split_paths(split).row_path) for split in OFFICIAL_SPLITS
            },
            "result_paths": {
                split: str(confirmed_split_paths(split).result_path) for split in OFFICIAL_SPLITS
            },
            "predictions_paths": {
                split: str(confirmed_split_paths(split).predictions_path)
                for split in OFFICIAL_SPLITS
            },
            "traces_paths": {
                split: str(confirmed_split_paths(split).traces_path) for split in OFFICIAL_SPLITS
            },
            "hf_model_url": HF_MODEL_URL,
            "hf_dataset_url": HF_DATASET_URL,
        },
    )
    write_text(
        output_message,
        _maintainer_message(
            output_dir=output_dir,
            version=version,
            dirty=dirty,
            checks_by_split=checks_by_split,
            rows=rows,
        ),
    )
    write_text(
        OUTPUT_REPORT,
        _report(output_dir=output_dir, version=version, dirty=dirty, rows=rows),
    )
    print(output_jsonl)
    print(output_metadata)
    print(output_message)
    print(OUTPUT_REPORT)


if __name__ == "__main__":
    main()
