"""Formatting helpers for training examples."""

from __future__ import annotations

import json
from typing import Any

from hallmark_mlx.data.schemas import VerificationTrace
from hallmark_mlx.training.prompts import (
    POLICY_SYSTEM_PROMPT,
    build_available_tools_prompt,
    build_user_prompt,
)


def render_trace_json(trace: VerificationTrace) -> str:
    """Render a trace as canonical JSON."""

    return json.dumps(trace.to_training_dict(), indent=2, sort_keys=True)


def format_trace_for_sft(trace: VerificationTrace) -> dict[str, Any]:
    """Convert a trace into a chat-style supervised example."""

    return {
        "messages": [
            {"role": "system", "content": POLICY_SYSTEM_PROMPT.strip()},
            {"role": "system", "content": build_available_tools_prompt()},
            {"role": "user", "content": build_user_prompt(trace.input)},
            {"role": "assistant", "content": render_trace_json(trace)},
        ],
        "metadata": {
            "record_id": trace.trace_id,
            "policy_version": trace.policy_version,
        },
    }
