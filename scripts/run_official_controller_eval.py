#!/usr/bin/env python3
"""Run one hallmark-mlx controller on an official HALLMARK split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hallmark_mlx.config import load_config
from hallmark_mlx.eval.official_compare import evaluate_runner_on_entries
from hallmark_mlx.eval.policy_modes import build_policy_runner
from hallmark_mlx.eval.upstream_hallmark import load_entries

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--upstream-root", default="/tmp/hallmark-upstream")
    parser.add_argument("--split", default="dev_public")
    parser.add_argument("--mode", default="bibtex_first_fallback")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "official_eval" / "bibtex_first_fallback_dev_public"),
    )
    parser.add_argument("--entry-timeout-seconds", type=int, default=60)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(Path(args.config))
    runner = build_policy_runner(config, args.mode)
    entries = list(load_entries(args.upstream_root, split=args.split))
    entries = entries[args.offset :]
    if args.limit is not None:
        entries = entries[: args.limit]
    row = evaluate_runner_on_entries(
        upstream_root=args.upstream_root,
        entries=entries,
        split_name=args.split,
        runner=runner,
        method_name=f"hallmark_mlx_{args.mode}",
        output_dir=args.output_dir,
        description=f"hallmark-mlx {args.mode} controller",
        source="executed_here",
        resume=not args.no_resume,
        entry_timeout_seconds=args.entry_timeout_seconds,
        progress_every=args.progress_every,
    )
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
