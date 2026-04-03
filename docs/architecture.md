# Architecture

`hallmark-mlx` separates citation verification into four layers so the research question stays legible.

## 1. Policy Model

The policy model does not directly emit a final benchmark label. It emits a structured verification trace with fields for:

- parsed bibliographic metadata,
- suspected issues,
- proposed retrieval queries,
- next actions,
- tool arguments,
- candidate rankings,
- and a final decision.

This keeps the training target aligned with process quality rather than unsupported latent answers.

## 2. Tool Layer

The tool layer is intentionally externalized. It currently contains wrappers for:

- BibTeX Updater,
- Crossref,
- OpenAlex,
- DBLP,
- ACL Anthology,
- Semantic Scholar.

The policy asks for tool calls. The tool executor normalizes raw outputs into `ToolResultSummary` objects. This lets the model learn a stable interface even if individual upstream APIs change.

## 3. Finalizer

The finalizer is conservative. It turns tool outputs into a final decision only when the evidence is sufficiently consistent. When evidence is missing or ambiguous, the finalizer abstains or marks the citation as unsupported rather than fabricating confidence.

This layer is not intended to replace learned reasoning. It is intended to:

- enforce calibrated abstention,
- centralize deterministic post-processing,
- and make evaluation behavior easier to audit.

## 4. Evaluation Layer

The evaluation layer has two jobs:

1. serialize predictions into HALLMARK-style JSONL, and
2. compute small local metrics for development runs.

Real HALLMARK evaluation should use the exact benchmark-provided `bibtex_key` values. The local adapter can generate deterministic fallback keys for dry runs, but those are not benchmark-valid identifiers.

## MLX Boundary

MLX-specific logic is isolated in `training/mlx_lora.py`. That module writes a reproducibility manifest and launches `python -m mlx_lm lora ...` using the standard MLX CLI surface. Real runs still depend on the installed `mlx-lm` version and a tokenizer-compatible instruct checkpoint.

## Weco Boundary

Weco is treated as an optional experiment orchestrator. The current scaffold writes a small stub artifact when Weco is enabled so future optimization work can plug in without reshaping the rest of the codebase.
