"""Inference orchestration for trace-generating policy models."""

from __future__ import annotations

from typing import Protocol

from hallmark_mlx.config import ModelConfig
from hallmark_mlx.data.schemas import VerificationInput, VerificationTrace
from hallmark_mlx.inference.finalizer import finalize_trace
from hallmark_mlx.inference.tool_executor import ToolExecutor
from hallmark_mlx.inference.warm_start_policy import WarmStartPolicyModel
from hallmark_mlx.types import FinalizationMode, ModelBackend


class PolicyModel(Protocol):
    """Protocol for trace-generating policy models."""

    def propose_trace(self, verification_input: VerificationInput) -> VerificationTrace:
        """Return a trace proposal before tool execution."""


def load_policy_model(config: ModelConfig) -> PolicyModel:
    """Load a policy model backend from config."""

    if config.backend == ModelBackend.WARM_START:
        return WarmStartPolicyModel()
    if config.backend == ModelBackend.MLX:
        from hallmark_mlx.inference.mlx_policy import MLXPolicyModel

        return MLXPolicyModel(config)
    if config.backend == ModelBackend.EXTERNAL:
        msg = "External policy backends are declared but not yet implemented."
        raise NotImplementedError(msg)
    msg = f"Unsupported backend: {config.backend}"
    raise ValueError(msg)


class PolicyRunner:
    """Execute policy inference, tools, and conservative finalization."""

    def __init__(
        self,
        model: PolicyModel,
        tool_executor: ToolExecutor,
        *,
        finalization_mode: FinalizationMode = FinalizationMode.DETERMINISTIC,
    ) -> None:
        self.model = model
        self.tool_executor = tool_executor
        self.finalization_mode = finalization_mode

    def _finalize(self, trace: VerificationTrace) -> VerificationTrace:
        if bool(trace.metadata.get("finalization_locked", False)):
            return trace
        if self.finalization_mode == FinalizationMode.GENERATIVE:
            return trace
        return finalize_trace(trace, force=True)

    def run(self, verification_input: VerificationInput) -> VerificationTrace:
        interactive_run = getattr(self.model, "run_with_tools", None)
        if callable(interactive_run):
            trace = interactive_run(verification_input, self.tool_executor)
            return self._finalize(trace)
        trace = self.model.propose_trace(verification_input)
        tool_results = self.tool_executor.execute_many(trace.tool_calls)
        enriched_trace = trace.model_copy(update={"tool_results": tool_results})
        return self._finalize(enriched_trace)
