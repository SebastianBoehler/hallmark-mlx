# Contributing

Thanks for contributing to `hallmark-mlx`.

## Scope

This repository focuses on citation verification research infrastructure:

- tool-using verification policies
- MLX LoRA training on Apple Silicon
- official HALLMARK benchmarking
- reproducible datasets, adapters, and reports

## Setup

```bash
uv sync --extra dev --extra mlx --extra weco
uv run pre-commit install
```

If you are not on Apple Silicon, skip the `mlx` extra:

```bash
uv sync --extra dev --extra weco
uv run pre-commit install
```

## Development Workflow

1. Make changes in small logical commits.
2. Keep modules short and composable.
3. Prefer explicit failures over silent fallbacks.
4. Treat official benchmark claims as reportable only after rerunning official splits.

## Code Quality

Run before opening a PR:

```bash
uv run ruff check .
uv run pytest
```

Pre-commit runs:

- `check-yaml`
- `check-toml`
- `check-added-large-files`
- `check-merge-conflict`
- `ruff check --fix`
- `ruff format`
- trailing whitespace cleanup
- end-of-file normalization

## Benchmarking Rules

- Use `dev_public`, `test_public`, and `stress_test` for public benchmark claims.
- Treat `search64` and `compare32` as internal model-selection splits only.
- If a fresh official rerun changes a headline number, refresh the benchmark artifacts before reusing the old claim.

## Experiment Notes

Record material training, Weco, and benchmark changes in:

- [.lab-book/README.md](.lab-book/README.md)

Create a dated file for each meaningful experiment batch.

## Pull Requests

Good PRs usually include:

- the goal of the change
- the config or code paths touched
- the validation commands run
- artifact paths for any new benchmark or training outputs

## Large Changes

For bigger additions, prefer to land work in this order:

1. infrastructure or interfaces
2. model or benchmark changes
3. regenerated reports and figures
