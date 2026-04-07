#!/usr/bin/env python3
"""Merge shard outputs from official controller evals and score them officially."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hallmark_mlx.eval.official_merge import merge_official_eval_shards

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", default="/tmp/hallmark-upstream")
    parser.add_argument("--split", default="dev_public")
    parser.add_argument("--shards-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", default="hallmark_mlx_bibtex_first_fallback")
    parser.add_argument(
        "--description",
        default="hallmark-mlx controller with deterministic finalizer",
    )
    return parser
def main() -> None:
    args = build_parser().parse_args()
    row = merge_official_eval_shards(
        upstream_root=Path(args.upstream_root),
        split=args.split,
        shards_root=Path(args.shards_root),
        output_dir=Path(args.output_dir),
        method_name=args.method_name,
        description=args.description,
    )
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
