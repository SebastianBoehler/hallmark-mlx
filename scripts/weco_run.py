#!/usr/bin/env python3
"""Launch a repo-native Weco optimization for hallmark-mlx."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from hallmark_mlx.weco_support import WecoSupportError, build_weco_eval_command, require_weco_cli

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO_ROOT / "weco_targets" / "hallmark_policy_trial.py"
DEFAULT_INSTRUCTIONS = REPO_ROOT / "weco_targets" / "hallmark_policy_trial_instructions.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch Weco on the hallmark-mlx policy frontier target.",
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--instructions", default=str(DEFAULT_INSTRUCTIONS))
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--metric", default="frontier_score")
    parser.add_argument("--goal", default="maximize")
    parser.add_argument("--model")
    parser.add_argument("--log-dir", default=".runs")
    parser.add_argument("--save-logs", action="store_true")
    parser.add_argument("--apply-change", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args, passthrough = build_parser().parse_known_args()
    try:
        source_path = Path(args.source).resolve()
        instructions_path = Path(args.instructions).resolve()
        command = [
            require_weco_cli(),
            "run",
            "--source",
            str(source_path),
            "--eval-command",
            build_weco_eval_command(source_path, sys.executable),
            "--metric",
            args.metric,
            "--goal",
            args.goal,
            "--steps",
            str(args.steps),
            "--log-dir",
            args.log_dir,
            "--additional-instructions",
            str(instructions_path),
        ]
        if args.model:
            command.extend(["--model", args.model])
        if args.save_logs:
            command.append("--save-logs")
        if args.apply_change:
            command.append("--apply-change")
        command.extend(passthrough)
        if args.dry_run:
            print(shlex.join(command))
            return
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    except (WecoSupportError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
