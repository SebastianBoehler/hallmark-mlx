# Roadmap

## Phase 1: Minimum Viable Prototype

- ingest trace JSONL
- build contamination-aware splits
- wrap BibTeX Updater and public metadata APIs
- emit HALLMARK-style prediction JSONL
- run a first local benchmark loop on prompting plus tools

## Phase 2: Real MLX Policy Inference

- implement structured JSON decoding for an MLX-backed policy model
- add prompt templates for tool decision traces
- support adapter loading and local inference on Apple Silicon
- benchmark abstention behavior

## Phase 3: Fine-Tuning

- finalize trace formatting for SFT
- tighten the current `mlx_lm lora` launcher against the installed MLX release and benchmark hardware limits
- add run manifests and experiment logging
- compare prompted versus fine-tuned policies

## Phase 4: Benchmark Integration

- import HALLMARK splits through a clean dataset adapter
- add per-subtest reporting
- add contamination reports against benchmark entries
- produce reproducible result bundles for public sharing

## Phase 5: Tooling Depth

- improve BibTeX correction with structured updater reports
- add better preprint versus published-version policies
- add optional adapters for additional scholarly metadata sources
- formalize disagreement handling between tools
