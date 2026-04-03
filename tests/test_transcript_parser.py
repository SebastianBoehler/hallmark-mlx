from hallmark_mlx.inference.transcript_parser import parse_assistant_turn


def test_parse_assistant_turn_extracts_tool_call_and_final_decision() -> None:
    turn = parse_assistant_turn(
        """
<verification_plan>
{
  "parsed_fields": {"title": "Sentence-BERT"},
  "suspected_issues": [],
  "proposed_query": {"query": "Sentence-BERT Reimers", "purpose": "resolve", "expected_candidates": 3},
  "next_action": "query_crossref"
}
</verification_plan>

<tool_call>
{
  "tool": "crossref",
  "action": "search_works",
  "arguments": {"query": "Sentence-BERT Reimers", "rows": 3}
}
</tool_call>
        """
    )

    assert turn.parsed_fields is not None
    assert turn.parsed_fields.title == "Sentence-BERT"
    assert turn.next_action is not None
    assert turn.next_action.value == "query_crossref"
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].tool.value == "crossref"


def test_parse_assistant_turn_collects_malformed_blocks_as_errors() -> None:
    turn = parse_assistant_turn("<tool_call>{not json}</tool_call>")

    assert turn.tool_calls == []
    assert turn.errors


def test_parse_assistant_turn_recovers_untagged_final_json() -> None:
    turn = parse_assistant_turn(
        """
{
  "rationale": "No matching record was found.",
  "verdict": "unsupported",
  "test_results": {
    "title_exists": false
  }
}
        """
    )

    assert turn.final_decision is not None
    assert turn.final_decision.verdict.value == "unsupported"
    assert turn.final_decision.subtest_results == {"title_exists": False}
    assert turn.errors


def test_parse_assistant_turn_supports_native_tool_name_payload() -> None:
    turn = parse_assistant_turn(
        """
<tool_call>
{
  "name": "crossref.search_works",
  "arguments": {
    "query": "Sentence-BERT Reimers 2019",
    "rows": 3
  }
}
</tool_call>
        """
    )

    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].tool.value == "crossref"
    assert turn.tool_calls[0].action == "search_works"


def test_parse_assistant_turn_recovers_first_final_json_from_repeated_objects() -> None:
    turn = parse_assistant_turn(
        """
{
  "confidence": 0.91,
  "rationale": "The entry is verified.",
  "verdict": "verified"
}

{
  "confidence": 0.91,
  "rationale": "The entry is verified.",
  "verdict": "verified"
}
        """
    )

    assert turn.final_decision is not None
    assert turn.final_decision.verdict.value == "verified"
