#!/usr/bin/env bash
set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required on macOS." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/." >&2
  exit 1
fi

if ! command -v pipx >/dev/null 2>&1; then
  echo "pipx is required to isolate bibtex-updater." >&2
  exit 1
fi

brew install pkg-config cmake
uv sync --extra dev --extra mlx --extra weco
pipx install bibtex-updater

echo "Mac setup complete. Verify the Qwen MLX stack, weco, and bibtex-updater before training."
