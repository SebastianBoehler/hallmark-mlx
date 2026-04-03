from hallmark_mlx.inference.mlx_policy import _strip_generation_prefix, _truncate_at_stop_markers


def test_strip_generation_prefix_removes_prompt_and_eos() -> None:
    prompt = "<s>system user assistant"
    text = "<s>system user assistant<tool_call>{}</tool_call></s>"

    cleaned = _strip_generation_prefix(text, prompt, "</s>")

    assert cleaned == "<tool_call>{}</tool_call>"


def test_strip_generation_prefix_leaves_completion_when_prompt_is_absent() -> None:
    cleaned = _strip_generation_prefix("<tool_call>{}</tool_call>", "prompt", None)

    assert cleaned == "<tool_call>{}</tool_call>"


def test_strip_generation_prefix_supports_prefilled_generation_prompt() -> None:
    prompt = "<s>assistant<tool_call>\n"
    text = "<s>assistant<tool_call>\n{\n  \"tool\": \"crossref\"\n}\n</tool_call>"

    cleaned = _strip_generation_prefix(text, prompt, None)

    assert cleaned == '{\n  "tool": "crossref"\n}\n</tool_call>'


def test_truncate_at_stop_markers_keeps_first_protocol_block_only() -> None:
    text = (
        '{\n  "tool": "crossref"\n}\n</tool_call>\n'
        '{"verdict":"verified"}'
    )

    cleaned = _truncate_at_stop_markers(text, ("</tool_call>",))

    assert cleaned == '{\n  "tool": "crossref"\n}\n</tool_call>'
