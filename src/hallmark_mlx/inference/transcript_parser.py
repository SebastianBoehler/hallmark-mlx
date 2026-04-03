"""Parsing helpers for tagged transcript generations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from hallmark_mlx.data.schemas import (
    CandidateRanking,
    FinalDecision,
    ParsedBibliographicFields,
    ProposedQuery,
    SuspectedIssue,
    ToolInvocation,
)
from hallmark_mlx.types import VerificationAction


@dataclass(slots=True)
class ParsedAssistantTurn:
    """Structured content recovered from one assistant transcript turn."""

    tool_calls: list[ToolInvocation] = field(default_factory=list)
    parsed_fields: ParsedBibliographicFields | None = None
    suspected_issues: list[SuspectedIssue] = field(default_factory=list)
    proposed_query: ProposedQuery | None = None
    next_action: VerificationAction | None = None
    candidate_ranking: CandidateRanking | None = None
    final_decision: FinalDecision | None = None
    errors: list[str] = field(default_factory=list)


def _extract_blocks(text: str, tag: str) -> list[str]:
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL)
    return [match.strip() for match in pattern.findall(text)]


def _parse_json_payload(payload: str, tag: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        errors.append(f"Malformed <{tag}> block: {exc}")
        return None
    if not isinstance(parsed, dict):
        errors.append(f"<{tag}> must contain a JSON object.")
        return None
    return parsed


def _normalize_tool_call_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Support both native Qwen tool names and local tool/action payloads."""

    if "name" not in payload or ("tool" in payload and "action" in payload):
        return payload
    name = payload["name"]
    if not isinstance(name, str) or "." not in name:
        return payload
    tool_name, action = name.split(".", 1)
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}
    return {
        "tool": tool_name,
        "action": action,
        "arguments": arguments,
    }


def _parse_verification_plan(payload: dict[str, Any], turn: ParsedAssistantTurn) -> None:
    fields = payload.get("parsed_fields")
    if isinstance(fields, dict):
        turn.parsed_fields = ParsedBibliographicFields.model_validate(fields)
    issues = payload.get("suspected_issues", [])
    if isinstance(issues, list):
        turn.suspected_issues = [
            SuspectedIssue.model_validate(issue) for issue in issues if isinstance(issue, dict)
        ]
    proposed_query = payload.get("proposed_query")
    if isinstance(proposed_query, dict):
        turn.proposed_query = ProposedQuery.model_validate(proposed_query)
    next_action = payload.get("next_action")
    if isinstance(next_action, str):
        try:
            turn.next_action = VerificationAction(next_action)
        except ValueError as exc:
            turn.errors.append(f"Invalid next_action value: {exc}")


def _extract_json_object_candidates(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    index = 0
    while True:
        brace_index = text.find("{", index)
        if brace_index == -1:
            return candidates
        try:
            payload, end_index = decoder.raw_decode(text, brace_index)
        except json.JSONDecodeError:
            index = brace_index + 1
            continue
        if isinstance(payload, dict):
            candidates.append(payload)
        index = end_index


def _parse_untagged_json_turn(text: str, turn: ParsedAssistantTurn) -> None:
    for data in _extract_json_object_candidates(text):
        if "parsed_fields" in data and isinstance(data["parsed_fields"], dict):
            try:
                turn.parsed_fields = ParsedBibliographicFields.model_validate(data["parsed_fields"])
            except Exception as exc:  # noqa: BLE001
                turn.errors.append(f"Invalid untagged parsed_fields payload: {exc}")

        if "suspected_issues" in data and isinstance(data["suspected_issues"], list):
            try:
                turn.suspected_issues = [
                    SuspectedIssue.model_validate(issue)
                    for issue in data["suspected_issues"]
                    if isinstance(issue, dict)
                ]
            except Exception as exc:  # noqa: BLE001
                turn.errors.append(f"Invalid untagged suspected_issues payload: {exc}")

        if "proposed_query" in data and isinstance(data["proposed_query"], dict):
            try:
                turn.proposed_query = ProposedQuery.model_validate(data["proposed_query"])
            except Exception as exc:  # noqa: BLE001
                turn.errors.append(f"Invalid untagged proposed_query payload: {exc}")

        if "next_action" in data and isinstance(data["next_action"], str):
            try:
                turn.next_action = VerificationAction(data["next_action"])
            except ValueError as exc:
                turn.errors.append(f"Invalid untagged next_action value: {exc}")

        if "tool_calls" in data and isinstance(data["tool_calls"], list):
            for payload in data["tool_calls"]:
                if not isinstance(payload, dict):
                    continue
                try:
                    turn.tool_calls.append(ToolInvocation.model_validate(payload))
                except Exception as exc:  # noqa: BLE001
                    turn.errors.append(f"Invalid untagged tool call payload: {exc}")

        if "candidate_ranking" in data and isinstance(data["candidate_ranking"], dict):
            try:
                turn.candidate_ranking = CandidateRanking.model_validate(data["candidate_ranking"])
            except Exception as exc:  # noqa: BLE001
                turn.errors.append(f"Invalid untagged candidate_ranking payload: {exc}")

        if "final_decision" in data and isinstance(data["final_decision"], dict):
            payload = dict(data["final_decision"])
        elif {"verdict", "rationale"}.issubset(data):
            payload = dict(data)
        else:
            continue

        if "test_results" in payload and "subtest_results" not in payload:
            payload["subtest_results"] = payload.pop("test_results")
        payload.setdefault("confidence", 0.0)
        try:
            turn.final_decision = FinalDecision.model_validate(payload)
            turn.errors.append("Assistant returned an untagged JSON object.")
            return
        except Exception as exc:  # noqa: BLE001
            turn.errors.append(f"Invalid untagged final decision payload: {exc}")


def parse_assistant_turn(text: str) -> ParsedAssistantTurn:
    """Parse one generated assistant message into typed transcript blocks."""

    turn = ParsedAssistantTurn()

    for payload in _extract_blocks(text, "verification_plan"):
        data = _parse_json_payload(payload, "verification_plan", turn.errors)
        if data is not None:
            _parse_verification_plan(data, turn)
            break

    for payload in _extract_blocks(text, "follow_up"):
        data = _parse_json_payload(payload, "follow_up", turn.errors)
        if data is not None and isinstance(data.get("next_action"), str):
            try:
                turn.next_action = VerificationAction(data["next_action"])
            except ValueError as exc:
                turn.errors.append(f"Invalid next_action value: {exc}")
            break

    for payload in _extract_blocks(text, "tool_call"):
        data = _parse_json_payload(payload, "tool_call", turn.errors)
        if data is None:
            continue
        data = _normalize_tool_call_payload(data)
        try:
            turn.tool_calls.append(ToolInvocation.model_validate(data))
        except Exception as exc:  # noqa: BLE001
            turn.errors.append(f"Invalid tool call payload: {exc}")

    ranking_blocks = _extract_blocks(text, "candidate_ranking")
    if ranking_blocks:
        # Take the first valid block; consistent with verification_plan/follow_up.
        data = _parse_json_payload(ranking_blocks[0], "candidate_ranking", turn.errors)
        if data is not None:
            try:
                turn.candidate_ranking = CandidateRanking.model_validate(data)
            except Exception as exc:  # noqa: BLE001
                turn.errors.append(f"Invalid candidate ranking payload: {exc}")

    final_blocks = _extract_blocks(text, "final_decision")
    if final_blocks:
        # Take the first valid block, consistent with the untagged-JSON fallback path.
        # If the model emits a premature first block, the training signal should fix it
        # rather than silently promoting the last block.
        data = _parse_json_payload(final_blocks[0], "final_decision", turn.errors)
        if data is not None:
            try:
                turn.final_decision = FinalDecision.model_validate(data)
            except Exception as exc:  # noqa: BLE001
                turn.errors.append(f"Invalid final decision payload: {exc}")

    if not turn.tool_calls and turn.final_decision is None:
        _parse_untagged_json_turn(text, turn)

    return turn
