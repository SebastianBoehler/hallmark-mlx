"""Static snapshot data for HALLMARK submission-readiness exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUR_ROW_PATH = (
    ROOT
    / "artifacts"
    / "official_eval_sharded_fast5"
    / "dev_public_bibtex_first_fallback"
    / "merged"
    / "row.json"
)
OUR_RESULT_PATH = (
    ROOT
    / "artifacts"
    / "official_eval_sharded_fast5"
    / "dev_public_bibtex_first_fallback"
    / "merged"
    / "result.json"
)
OUTPUT_JSON = ROOT / "artifacts" / "hallmark_submission_readiness" / "leaderboard_snapshot.json"
OUTPUT_REPORT = ROOT / "docs" / "reports" / "hallmark_submission_readiness.md"
OUTPUT_PNG = ROOT / "docs" / "figures" / "hallmark_submission_leaderboard.png"
OUTPUT_PDF = ROOT / "docs" / "figures" / "hallmark_submission_leaderboard.pdf"

LEADERBOARD_URL = "https://rpatrik96.github.io/research-agora/benchmarks.html"
LEADERBOARD_GENERATED = "2026-04-02"
REPRO_COMMAND = """for offset in 0 140 280 420 560 700 840 980; do
  shard=$(printf "shard_%02d" $((offset / 140)))
  uv run python scripts/run_official_controller_eval.py \\
    --upstream-root /tmp/hallmark-upstream \\
    --split dev_public \\
    --mode bibtex_first_fallback \\
    --output-dir \\
      artifacts/official_eval_sharded_fast5/dev_public_bibtex_first_fallback/shards/${shard} \\
    --entry-timeout-seconds 5 \\
    --offset ${offset} \\
    --limit 140
done

description=$(
  printf %s \
    "hallmark-mlx controller with deterministic finalizer " \
    "(5s per-entry timeout, 8-way sharded official run)"
)

uv run python scripts/merge_official_eval_shards.py \\
  --upstream-root /tmp/hallmark-upstream \\
  --split dev_public \\
  --shards-root artifacts/official_eval_sharded_fast5/dev_public_bibtex_first_fallback/shards \\
  --output-dir artifacts/official_eval_sharded_fast5/dev_public_bibtex_first_fallback/merged \\
  --method-name hallmark_mlx_bibtex_first_fallback \\
  --description "${description}"
"""

LEADERBOARD_ROWS: list[dict[str, Any]] = [
    {
        "name": "bibtex-updater",
        "type": "TOOL",
        "detection_rate": 0.946,
        "f1_hallucination": 0.908,
        "tier_weighted_f1": 0.936,
        "false_positive_rate": 0.179,
        "ece": 0.297,
        "source": "research_agora_snapshot",
    },
    {
        "name": "GPT-5.1",
        "type": "LLM",
        "detection_rate": 0.797,
        "f1_hallucination": 0.822,
        "tier_weighted_f1": 0.846,
        "false_positive_rate": 0.171,
        "ece": 0.107,
        "source": "research_agora_snapshot",
    },
    {
        "name": "Qwen 3 235B",
        "type": "LLM",
        "detection_rate": 0.832,
        "f1_hallucination": 0.737,
        "tier_weighted_f1": 0.806,
        "false_positive_rate": 0.551,
        "ece": 0.294,
        "source": "research_agora_snapshot",
    },
    {
        "name": "DeepSeek R1",
        "type": "LLM",
        "detection_rate": 0.871,
        "f1_hallucination": 0.737,
        "tier_weighted_f1": 0.814,
        "false_positive_rate": 0.640,
        "ece": 0.247,
        "source": "research_agora_snapshot",
    },
    {
        "name": "Mistral Large",
        "type": "LLM",
        "detection_rate": 0.691,
        "f1_hallucination": 0.731,
        "tier_weighted_f1": 0.743,
        "false_positive_rate": 0.258,
        "ece": 0.247,
        "source": "research_agora_snapshot",
    },
    {
        "name": "DeepSeek V3",
        "type": "LLM",
        "detection_rate": 0.880,
        "f1_hallucination": 0.721,
        "tier_weighted_f1": 0.805,
        "false_positive_rate": 0.730,
        "ece": 0.331,
        "source": "research_agora_snapshot",
    },
    {
        "name": "Gemini 2.5 Flash",
        "type": "LLM",
        "detection_rate": 0.482,
        "f1_hallucination": 0.617,
        "tier_weighted_f1": 0.608,
        "false_positive_rate": 0.101,
        "ece": 0.286,
        "source": "research_agora_snapshot",
    },
    {
        "name": "DOI-only",
        "type": "TOOL",
        "detection_rate": 0.256,
        "f1_hallucination": 0.361,
        "tier_weighted_f1": 0.314,
        "false_positive_rate": 0.195,
        "ece": 0.143,
        "source": "research_agora_snapshot",
    },
    {
        "name": "HaRC",
        "type": "TOOL",
        "detection_rate": 0.143,
        "f1_hallucination": 0.250,
        "tier_weighted_f1": 0.165,
        "false_positive_rate": 0.002,
        "ece": 0.011,
        "source": "research_agora_snapshot",
    },
    {
        "name": "verify-citations",
        "type": "TOOL",
        "detection_rate": 0.300,
        "f1_hallucination": 0.240,
        "tier_weighted_f1": 0.302,
        "false_positive_rate": 0.133,
        "ece": None,
        "source": "research_agora_snapshot",
    },
]
