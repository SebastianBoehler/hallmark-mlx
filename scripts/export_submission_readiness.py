#!/usr/bin/env python3
"""Export a submission-readiness report for the official HALLMARK dev_public row."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles, figsizes

from hallmark_mlx.eval.submission_snapshot import (
    LEADERBOARD_GENERATED,
    LEADERBOARD_ROWS,
    LEADERBOARD_URL,
    OUR_RESULT_PATH,
    OUR_ROW_PATH,
    OUTPUT_JSON,
    OUTPUT_PDF,
    OUTPUT_PNG,
    OUTPUT_REPORT,
    REPRO_COMMAND,
)
from hallmark_mlx.utils.io import read_json, write_json, write_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-path", default=str(OUR_ROW_PATH))
    parser.add_argument("--result-path", default=str(OUR_RESULT_PATH))
    return parser


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _validate_official_row(row: dict[str, Any]) -> list[str]:
    checks = [
        f"split == dev_public: {row.get('split') == 'dev_public'}",
        f"num_entries == 1119: {row.get('num_entries') == 1119}",
        f"partial == False: {row.get('partial') is False}",
        f"coverage == 1.0: {float(row.get('coverage') or 0.0) == 1.0}",
        f"source == executed_here: {row.get('source') == 'executed_here'}",
    ]
    if "official evaluator" not in str(row.get("status_message", "")).lower():
        raise SystemExit("Row status_message does not confirm the official evaluator.")
    if any(check.endswith("False") for check in checks):
        raise SystemExit("Official row validation failed:\n" + "\n".join(checks))
    return checks


def _our_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "hallmark-mlx",
        "type": "HYBRID",
        "detection_rate": float(row["detection_rate"]),
        "f1_hallucination": float(row["f1_hallucination"]),
        "tier_weighted_f1": float(row["tier_weighted_f1"]),
        "false_positive_rate": float(row["false_positive_rate"]),
        "ece": float(row["ece"]),
        "source": "executed_here",
    }


def _ranked_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row["f1_hallucination"]), reverse=True)


def _our_rank(rows: list[dict[str, Any]]) -> int:
    return next(idx for idx, item in enumerate(rows, start=1) if item["name"] == "hallmark-mlx")


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = (
        "rank",
        "name",
        "type",
        "detection_rate",
        "f1_hallucination",
        "tier_weighted_f1",
        "false_positive_rate",
        "ece",
        "source",
    )
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for idx, row in enumerate(rows, start=1):
        values = [idx, row["name"], row["type"]]
        values.extend(row[key] for key in columns[3:])
        lines.append("| " + " | ".join(_fmt(value) for value in values) + " |")
    return "\n".join(lines)


def _plot(rows: list[dict[str, Any]]) -> None:
    plt.rcParams.update(bundles.icml2024(column="full", nrows=1, ncols=5, usetex=False))
    plt.rcParams.update(figsizes.icml2024_full(nrows=1, ncols=5))
    figure, axes = plt.subplots(1, 5, sharey=True)
    figure.set_size_inches(14.0, 6.5)
    metrics = (
        ("detection_rate", "DR ↑", False),
        ("f1_hallucination", "F1-H ↑", False),
        ("tier_weighted_f1", "TW-F1 ↑", False),
        ("false_positive_rate", "FPR ↓", True),
        ("ece", "ECE ↓", True),
    )
    y = np.arange(len(rows))
    labels = [str(row["name"]) for row in rows]
    colors = ["#2ca02c" if row["name"] == "hallmark-mlx" else "#7f7f7f" for row in rows]
    for ax, (key, title, lower_is_better) in zip(axes, metrics, strict=True):
        values = [0.0 if row[key] is None else float(row[key]) for row in rows]
        ax.barh(y, values, color=colors)
        ax.set_title(title)
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        if lower_is_better:
            ax.invert_xaxis()
        for yi, row, value in zip(y, rows, values, strict=True):
            if row[key] is None:
                continue
            ax.text(
                value,
                yi,
                f" {_fmt(row[key])}",
                va="center",
                ha="left" if not lower_is_better else "right",
                fontsize=7,
            )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    figure.suptitle("HALLMARK dev_public: Agora Snapshot vs hallmark-mlx")
    figure.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight")
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = build_parser().parse_args()
    row = read_json(Path(args.row_path).resolve())
    result = read_json(Path(args.result_path).resolve())
    checks = _validate_official_row(row)

    rows = _ranked_rows([_our_row(row), *LEADERBOARD_ROWS])
    snapshot = {
        "leaderboard_url": LEADERBOARD_URL,
        "leaderboard_generated": LEADERBOARD_GENERATED,
        "our_row_path": str(Path(args.row_path).resolve()),
        "our_result_path": str(Path(args.result_path).resolve()),
        "validation_checks": checks,
        "rows": rows,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_JSON, snapshot)
    _plot(rows)

    report = "\n".join(
        [
            "# HALLMARK Submission Readiness",
            "",
            "## Protocol Check",
            "",
            *[f"- {check}" for check in checks],
            f"- Official row file: `{Path(args.row_path).resolve()}`",
            f"- Official result file: `{Path(args.result_path).resolve()}`",
            (
                "- Runtime disclosure: 5s per-entry timeout, 8-way sharded run, "
                "merged and rescored with the upstream official evaluator."
            ),
            "",
            "## Leaderboard Snapshot",
            "",
            (
                f"Compared against the public Agora leaderboard snapshot from "
                f"{LEADERBOARD_GENERATED} at {LEADERBOARD_URL}."
            ),
            "",
            _markdown_table(rows),
            "",
            "## Outcome",
            "",
            (
                "`hallmark-mlx` ranks "
                f"#{_our_rank(rows)} "
                "by `F1-H` in this snapshot."
            ),
            (
                f"- Our row: DR {_fmt(row['detection_rate'])}, "
                f"F1-H {_fmt(row['f1_hallucination'])}, "
                f"TW-F1 {_fmt(row['tier_weighted_f1'])}, "
                f"FPR {_fmt(row['false_positive_rate'])}, "
                f"ECE {_fmt(row['ece'])}."
            ),
            (
                "- Current public top row before ours: bibtex-updater with DR 0.946, F1-H 0.908, "
                "TW-F1 0.936, FPR 0.179, ECE 0.297."
            ),
            "",
            "## Submission Caveats",
            "",
            "- This is a full `dev_public` result, not a hidden-test result.",
            "- The evaluation logic is upstream-official; the orchestration is local.",
            "- The timeout/sharding setup should be disclosed in any submission notes.",
            "",
            "## Reproduction",
            "",
            "```bash",
            REPRO_COMMAND.rstrip(),
            "```",
            "",
            "## Extra Context",
            "",
            (
                f"- Full metrics from `result.json`: AUROC {_fmt(result['auroc'])}, "
                f"AUPRC {_fmt(result['auprc'])}, MCC {_fmt(result['mcc'])}, "
                f"mean API calls {_fmt(result['mean_api_calls'])}."
            ),
            "",
        ]
    )
    write_text(OUTPUT_REPORT, report)
    print(OUTPUT_JSON)
    print(OUTPUT_REPORT)
    print(OUTPUT_PNG)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
