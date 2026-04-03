"""MLX-backed transcript rollout for tool-using citation verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from hallmark_mlx.config import ModelConfig
from hallmark_mlx.data.schemas import ParsedBibliographicFields, ToolInvocation, ToolResultSummary, VerificationInput, VerificationTrace
from hallmark_mlx.inference.deterministic_shortcuts import maybe_finalize_deterministically
from hallmark_mlx.inference.mlx_rollout_helpers import (
    initial_messages,
    render_tool_call_message,
    render_tool_result_message,
    trace_metadata,
)
from hallmark_mlx.inference.tool_executor import ToolExecutor
from hallmark_mlx.inference.transcript_parser import parse_assistant_turn
from hallmark_mlx.training.prompts import build_tool_schemas
from hallmark_mlx.types import FinalizationMode, VerificationAction
from hallmark_mlx.types import InputType, ToolName

def _strip_generation_prefix(text: str, prompt: str, eos_token: str | None) -> str:
    cleaned = text.strip()
    if eos_token:
        cleaned = cleaned.replace(eos_token, "").strip()
    if prompt and cleaned.startswith(prompt):
        cleaned = cleaned[len(prompt) :].lstrip()
    return cleaned


def _truncate_at_stop_markers(text: str, stop_markers: Iterable[str]) -> str:
    """Cut generated content at the earliest protocol stop marker."""

    earliest_index: int | None = None
    matched_marker = ""
    for marker in stop_markers:
        marker_index = text.find(marker)
        if marker_index == -1:
            continue
        if earliest_index is None or marker_index < earliest_index:
            earliest_index = marker_index
            matched_marker = marker
    if earliest_index is None:
        return text
    stop_index = earliest_index + len(matched_marker)
    return text[:stop_index].rstrip()


class MLXPolicyModel:
    """Roll out a tool-using transcript with a fine-tuned MLX adapter."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        adapter_path = Path(self.config.adapter_path)
        if not adapter_path.exists():
            msg = f"Adapter path does not exist: {adapter_path}"
            raise FileNotFoundError(msg)
        from mlx_lm import load

        self._model, self._tokenizer = load(
            self.config.base_model,
            adapter_path=str(adapter_path),
            lazy=False,
        )
        return self._model, self._tokenizer

    def _generate_turn(
        self,
        messages: list[dict[str, str]],
        *,
        prefix: str | None = None,
        stop_markers: tuple[str, ...] = ("</tool_call>",),
    ) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = self._ensure_loaded()
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=build_tool_schemas(),
            tokenize=False,
            add_generation_prompt=True,
        )
        generation_prompt = f"{prompt}{prefix}" if prefix else prompt
        try:
            text = generate(
                model,
                tokenizer,
                generation_prompt,
                verbose=False,
                max_tokens=self.config.max_tokens,
                sampler=make_sampler(self.config.temperature),
                stop_words=list(stop_markers),
            )
        except TypeError:
            # Older mlx_lm versions do not support stop_words; fall back gracefully.
            text = generate(
                model,
                tokenizer,
                generation_prompt,
                verbose=False,
                max_tokens=self.config.max_tokens,
                sampler=make_sampler(self.config.temperature),
            )
        eos_token = getattr(tokenizer, "eos_token", None)
        cleaned = _strip_generation_prefix(text, generation_prompt, eos_token)
        cleaned = _truncate_at_stop_markers(cleaned, stop_markers)
        return f"{prefix}{cleaned}" if prefix else cleaned

    def _prime_bibtex_check(
        self,
        verification_input: VerificationInput,
        tool_executor: ToolExecutor,
        messages: list[dict[str, str]],
        assistant_turns: list[str],
        tool_calls: list[ToolInvocation],
        tool_results: list[ToolResultSummary],
    ) -> tuple[int, bool]:
        if (
            not self.config.force_bibtex_updater_first
            or verification_input.input_type != InputType.BIBTEX_ENTRY
        ):
            return 0, False
        tool_call = ToolInvocation(
            tool=ToolName.BIBTEX_UPDATER,
            action="check_bibtex",
            arguments={"bibtex": verification_input.raw_input, "strict": True},
        )
        assistant_text = render_tool_call_message(tool_call)
        assistant_turns.append(assistant_text)
        messages.append({"role": "assistant", "content": assistant_text})
        tool_calls.append(tool_call)
        result = tool_executor.execute(tool_call)
        tool_results.append(result)
        messages.append(
            {
                "role": "tool",
                "content": render_tool_result_message(result, len(tool_results) - 1),
            },
        )
        return 1, False

    def run_with_tools(
        self,
        verification_input: VerificationInput,
        tool_executor: ToolExecutor,
    ) -> VerificationTrace:
        messages = initial_messages(verification_input)
        parsed_fields = ParsedBibliographicFields()
        suspected_issues = []
        proposed_query = None
        next_action = VerificationAction.PARSE_INPUT
        tool_calls = []
        tool_results = []
        candidate_ranking = None
        assistant_turns: list[str] = []
        parse_errors: list[str] = []
        premature_final_decision_rounds: list[int] = []
        first_response_tool_call_count, first_response_had_final_decision = self._prime_bibtex_check(
            verification_input,
            tool_executor,
            messages,
            assistant_turns,
            tool_calls,
            tool_results,
        )
        finished_trace = None
        if self.config.finalization_mode == FinalizationMode.DETERMINISTIC:
            finished_trace = maybe_finalize_deterministically(
                verification_input=verification_input,
                policy_version=f"mlx-transcript-{self.config.system_prompt_version}",
                parsed_fields=parsed_fields,
                suspected_issues=suspected_issues,
                proposed_query=proposed_query,
                next_action=next_action,
                tool_calls=tool_calls,
                tool_results=tool_results,
                candidate_ranking=candidate_ranking,
                metadata=trace_metadata(
                    assistant_turns,
                    parse_errors,
                    premature_final_decision_rounds,
                    first_response_tool_call_count,
                    first_response_had_final_decision,
                ),
            )
        if finished_trace is not None:
            return finished_trace

        for round_index in range(self.config.max_rollout_rounds):
            assistant_prefix = "<tool_call>\n" if round_index == 0 else None
            assistant_text = self._generate_turn(messages, prefix=assistant_prefix)
            assistant_turns.append(assistant_text)
            messages.append({"role": "assistant", "content": assistant_text})
            parsed = parse_assistant_turn(assistant_text)
            parse_errors.extend(parsed.errors)

            if round_index == 0 and first_response_tool_call_count == 0:
                first_response_tool_call_count = len(parsed.tool_calls)
                first_response_had_final_decision = parsed.final_decision is not None

            if parsed.parsed_fields is not None:
                parsed_fields = parsed.parsed_fields
                suspected_issues = parsed.suspected_issues
                proposed_query = parsed.proposed_query
            if parsed.next_action is not None:
                next_action = parsed.next_action
            if parsed.candidate_ranking is not None:
                candidate_ranking = parsed.candidate_ranking

            if parsed.tool_calls:
                if parsed.final_decision is not None:
                    premature_final_decision_rounds.append(round_index)
                for tool_call in parsed.tool_calls:
                    tool_calls.append(tool_call)
                    result = tool_executor.execute(tool_call)
                    tool_results.append(result)
                    messages.append(
                        {
                            "role": "tool",
                            "content": render_tool_result_message(
                                result,
                                len(tool_results) - 1,
                            ),
                        }
                    )
                finished_trace = None
                if self.config.finalization_mode == FinalizationMode.DETERMINISTIC:
                    finished_trace = maybe_finalize_deterministically(
                        verification_input=verification_input,
                        policy_version=f"mlx-transcript-{self.config.system_prompt_version}",
                        parsed_fields=parsed_fields,
                        suspected_issues=suspected_issues,
                        proposed_query=proposed_query,
                        next_action=next_action,
                        tool_calls=tool_calls,
                        tool_results=tool_results,
                        candidate_ranking=candidate_ranking,
                        metadata=trace_metadata(
                            assistant_turns,
                            parse_errors,
                            premature_final_decision_rounds,
                            first_response_tool_call_count,
                            first_response_had_final_decision,
                        ),
                    )
                if finished_trace is not None:
                    return finished_trace
                continue

            if parsed.final_decision is not None:
                return VerificationTrace(
                    policy_version=f"mlx-transcript-{self.config.system_prompt_version}",
                    input=verification_input,
                    parsed_fields=parsed_fields,
                    suspected_issues=suspected_issues,
                    proposed_query=proposed_query,
                    next_action=VerificationAction.FINALIZE,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    candidate_ranking=candidate_ranking,
                    final_decision=parsed.final_decision,
                    metadata=trace_metadata(
                        assistant_turns,
                        parse_errors,
                        premature_final_decision_rounds,
                        first_response_tool_call_count,
                        first_response_had_final_decision,
                    ),
                )

            if not assistant_text:
                break

        return VerificationTrace(
            policy_version=f"mlx-transcript-{self.config.system_prompt_version}",
            input=verification_input,
            parsed_fields=parsed_fields,
            suspected_issues=suspected_issues,
            proposed_query=proposed_query,
            next_action=next_action,
            tool_calls=tool_calls,
            tool_results=tool_results,
            candidate_ranking=candidate_ranking,
            metadata=trace_metadata(
                assistant_turns,
                parse_errors,
                premature_final_decision_rounds,
                first_response_tool_call_count,
                first_response_had_final_decision,
            )
            | {"rollout_error": "Model did not emit a final_decision within the rollout budget."},
        )
