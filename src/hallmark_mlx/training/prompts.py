"""Prompt templates for trace-based policy learning."""

from __future__ import annotations

import json

from hallmark_mlx.data.schemas import VerificationInput

POLICY_SYSTEM_PROMPT = """You are a citation verification policy model.

Your job is not to guess whether a citation is true from memory.
Your job is to decide how to verify it.

Always reason in a structured way:
1. parse the citation or claim,
2. identify ambiguity and likely failure modes,
3. choose the next verification action,
4. specify tool arguments precisely,
5. summarize retrieved evidence conservatively,
6. abstain when evidence is insufficient.

Return structured JSON for a verification trace.
"""


def build_user_prompt(verification_input: VerificationInput) -> str:
    """Render a user prompt for a verification task."""

    return (
        "Construct a verification trace for the following input.\n"
        "Use tool calls only when they are justified.\n\n"
        f"{json.dumps(verification_input.model_dump(exclude_none=True), indent=2)}"
    )


def build_available_tools_prompt() -> str:
    """Describe the current tool palette."""

    return (
        "Available tools: bibtex_updater, crossref, openalex, semantic_scholar.\n"
        "Prefer tool use over unsupported latent claims."
    )
