"""Pure helpers for MLX transcript rollouts."""

from __future__ import annotations

import json

from hallmark_mlx.data.schemas import ToolInvocation, ToolResultSummary, VerificationInput
from hallmark_mlx.protocol import compact_tool_result_payload
from hallmark_mlx.training.prompts import POLICY_SYSTEM_PROMPT, build_user_prompt


def initial_messages(verification_input: VerificationInput) -> list[dict[str, str]]:
    """Build the initial chat transcript for a verification task."""

    return [
        {"role": "system", "content": POLICY_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": build_user_prompt(verification_input)},
    ]


def render_tool_result_message(result: ToolResultSummary, index: int) -> str:
    """Serialize a compact tool observation for the next chat turn."""

    _ = index
    return json.dumps(compact_tool_result_payload(result), indent=2, sort_keys=True)


def render_tool_call_message(tool_call: ToolInvocation) -> str:
    """Serialize a tool call in the transcript format expected by the parser."""

    payload = {
        "name": f"{tool_call.tool.value}.{tool_call.action}",
        "arguments": tool_call.arguments,
    }
    return "<tool_call>\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n</tool_call>"


def trace_metadata(
    assistant_turns: list[str],
    parse_errors: list[str],
    premature_final_decision_rounds: list[int],
    first_response_tool_call_count: int,
    first_response_had_final_decision: bool,
) -> dict[str, object]:
    """Return rollout metadata in the stable trace shape."""

    return {
        "policy_backend": "mlx",
        "assistant_turns": assistant_turns,
        "parse_errors": parse_errors,
        "premature_final_decision_rounds": premature_final_decision_rounds,
        "first_response_tool_call_count": first_response_tool_call_count,
        "first_response_had_final_decision": first_response_had_final_decision,
    }
