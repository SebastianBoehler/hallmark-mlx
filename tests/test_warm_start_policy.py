from hallmark_mlx.data.schemas import VerificationInput
from hallmark_mlx.inference.warm_start_policy import WarmStartPolicyModel, parse_bibtex_entry
from hallmark_mlx.types import InputType, ToolName, VerificationAction


def test_parse_bibtex_entry_extracts_core_fields() -> None:
    fields = parse_bibtex_entry(
        """@article{vaswani2017attention,
        title={Attention Is All You Need},
        author={Vaswani, Ashish and Shazeer, Noam},
        year={2017},
        doi={10.5555/3295222.3295349},
        journal={NeurIPS}
    }""",
    )

    assert fields.title == "Attention Is All You Need"
    assert fields.year == 2017
    assert fields.doi == "10.5555/3295222.3295349"
    assert fields.bibtex_type == "article"


def test_warm_start_policy_builds_tool_chain_for_raw_citation() -> None:
    trace = WarmStartPolicyModel().propose_trace(
        VerificationInput(
            record_id="trace-1",
            input_type=InputType.RAW_CITATION_STRING,
            raw_input="Vaswani et al. Attention Is All You Need. NeurIPS 2017.",
        ),
    )

    assert trace.policy_version == "warm-start-v0"
    assert trace.parsed_fields.title == "Attention Is All You Need"
    assert trace.parsed_fields.authors == ["Vaswani"]
    assert trace.parsed_fields.year == 2017
    assert trace.parsed_fields.venue == "NeurIPS"
    assert trace.next_action == VerificationAction.QUERY_CROSSREF
    assert [tool_call.tool for tool_call in trace.tool_calls] == [
        ToolName.CROSSREF,
        ToolName.OPENALEX,
        ToolName.SEMANTIC_SCHOLAR,
    ]


def test_warm_start_policy_uses_bibtex_updater_for_bibtex_entries() -> None:
    trace = WarmStartPolicyModel().propose_trace(
        VerificationInput(
            record_id="trace-2",
            input_type=InputType.BIBTEX_ENTRY,
            raw_input="""@inproceedings{test,title={Attention Is All You Need},author={Ashish Vaswani},year={2017}}""",
        ),
    )

    assert trace.tool_calls[0].tool == ToolName.BIBTEX_UPDATER
    assert trace.tool_calls[0].action == "check_bibtex"
