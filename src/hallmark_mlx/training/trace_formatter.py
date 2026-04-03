"""Formatting helpers for training examples."""

from __future__ import annotations

import json
from typing import Any

from hallmark_mlx.data.schemas import VerificationTrace
from hallmark_mlx.protocol import (
    compact_final_decision_payload,
    compact_tool_result_payload,
)
from hallmark_mlx.training.prompts import (
    POLICY_SYSTEM_PROMPT,
    build_available_tools_prompt,
    build_tool_schemas,
    build_user_prompt,
)
from hallmark_mlx.types import TrainingExampleFormat


def render_trace_json(trace: VerificationTrace) -> str:
    """Render a trace as canonical JSON."""

    return json.dumps(trace.to_training_dict(), indent=2, sort_keys=True)


def render_tool_call_message(trace: VerificationTrace, index: int) -> dict[str, Any]:
    """Render one assistant tool-call turn using Qwen-native structured calls."""

    tool_call = trace.tool_calls[index]
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "name": f"{tool_call.tool.value}.{tool_call.action}",
                "arguments": tool_call.arguments,
            }
        ],
    }


def render_tool_observation(result_index: int, trace: VerificationTrace) -> dict[str, str]:
    """Render one tool result as a native tool message."""

    result = trace.tool_results[result_index]
    payload = compact_tool_result_payload(result)
    return {
        "role": "tool",
        "content": json.dumps(payload, indent=2, sort_keys=True),
    }


def render_final_transcript_message(trace: VerificationTrace) -> str:
    """Render the final assistant decision turn.

    The final assistant turn is a compact JSON verdict object.
    """

    if trace.final_decision is not None:
        return json.dumps(
            compact_final_decision_payload(trace.final_decision),
            indent=2,
            sort_keys=True,
        )
    return render_trace_json(trace)


def format_trace_json_for_sft(trace: VerificationTrace) -> dict[str, Any]:
    """Convert a trace into the original single-turn JSON target format."""

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
            "training_format": TrainingExampleFormat.TRACE_JSON.value,
        },
    }


def _base_transcript_messages(trace: VerificationTrace) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": POLICY_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": build_user_prompt(trace.input)},
    ]


def format_tool_transcript_for_sft(trace: VerificationTrace) -> dict[str, Any]:
    """Convert a trace into a multi-turn tool transcript example."""

    messages = _base_transcript_messages(trace)

    pair_count = min(len(trace.tool_calls), len(trace.tool_results))
    for index in range(pair_count):
        messages.append(render_tool_call_message(trace, index))
        messages.append(render_tool_observation(index, trace))

    for index in range(pair_count, len(trace.tool_calls)):
        messages.append(render_tool_call_message(trace, index))

    # NOTE: tool_results can never exceed pair_count by construction (pair_count =
    # min(calls, results)), so there is no orphan-observations case to handle here.

    messages.append({"role": "assistant", "content": render_final_transcript_message(trace)})
    return {
        "messages": messages,
        "tools": build_tool_schemas(),
        "metadata": {
            "record_id": trace.trace_id,
            "policy_version": trace.policy_version,
            "training_format": TrainingExampleFormat.TOOL_TRANSCRIPT.value,
        },
    }


def format_tool_transcript_steps_for_sft(trace: VerificationTrace) -> list[dict[str, Any]]:
    """Explode a trace into stepwise assistant targets for stronger behavior cloning."""

    messages = _base_transcript_messages(trace)
    examples: list[dict[str, Any]] = []
    pair_count = min(len(trace.tool_calls), len(trace.tool_results))

    for index in range(len(trace.tool_calls)):
        examples.append(
            {
                "messages": [*messages, render_tool_call_message(trace, index)],
                "tools": build_tool_schemas(),
                "metadata": {
                    "record_id": trace.trace_id,
                    "policy_version": trace.policy_version,
                    "training_format": TrainingExampleFormat.TOOL_TRANSCRIPT_STEPS.value,
                    "target_type": "tool_call",
                    "target_index": index,
                },
            },
        )
        if index < pair_count:
            messages.append(render_tool_call_message(trace, index))
            messages.append(render_tool_observation(index, trace))

    if trace.final_decision is not None:
        examples.append(
            {
                "messages": [*messages, {"role": "assistant", "content": render_final_transcript_message(trace)}],
                "tools": build_tool_schemas(),
                "metadata": {
                    "record_id": trace.trace_id,
                    "policy_version": trace.policy_version,
                    "training_format": TrainingExampleFormat.TOOL_TRANSCRIPT_STEPS.value,
                    "target_type": "final_decision",
                },
            },
        )
    return examples


def format_trace_for_sft(
    trace: VerificationTrace,
    *,
    example_format: TrainingExampleFormat = TrainingExampleFormat.TRACE_JSON,
) -> dict[str, Any]:
    """Convert a trace into the requested supervised example layout."""

    if example_format == TrainingExampleFormat.TOOL_TRANSCRIPT:
        return format_tool_transcript_for_sft(trace)
    return format_trace_json_for_sft(trace)


def format_trace_examples_for_sft(
    trace: VerificationTrace,
    *,
    example_format: TrainingExampleFormat = TrainingExampleFormat.TRACE_JSON,
) -> list[dict[str, Any]]:
    """Return one or more supervised examples for a trace."""

    if example_format == TrainingExampleFormat.TOOL_TRANSCRIPT_STEPS:
        return format_tool_transcript_steps_for_sft(trace)
    return [format_trace_for_sft(trace, example_format=example_format)]
