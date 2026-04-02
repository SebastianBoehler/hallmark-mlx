"""Inference orchestration for trace-generating policy models."""

from __future__ import annotations

from typing import Protocol

from hallmark_mlx.config import ModelConfig
from hallmark_mlx.data.schemas import VerificationInput, VerificationTrace
from hallmark_mlx.inference.finalizer import finalize_trace
from hallmark_mlx.inference.tool_executor import ToolExecutor
from hallmark_mlx.inference.warm_start_policy import WarmStartPolicyModel
from hallmark_mlx.types import ModelBackend


class PolicyModel(Protocol):
    """Protocol for trace-generating policy models."""

    def propose_trace(self, verification_input: VerificationInput) -> VerificationTrace:
        """Return a trace proposal before tool execution."""


class MLXPolicyModel:
    """Placeholder MLX policy model boundary."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def propose_trace(self, verification_input: VerificationInput) -> VerificationTrace:
        msg = (
            "Structured MLX inference is not wired yet. Implement a decoder that "
            "maps model output into `VerificationTrace` JSON."
        )
        raise NotImplementedError(msg)


def load_policy_model(config: ModelConfig) -> PolicyModel:
    """Load a policy model backend from config."""

    if config.backend == ModelBackend.WARM_START:
        return WarmStartPolicyModel()
    if config.backend == ModelBackend.MLX:
        return MLXPolicyModel(config)
    if config.backend == ModelBackend.EXTERNAL:
        msg = "External policy backends are declared but not yet implemented."
        raise NotImplementedError(msg)
    msg = f"Unsupported backend: {config.backend}"
    raise ValueError(msg)


class PolicyRunner:
    """Execute policy inference, tools, and conservative finalization."""

    def __init__(self, model: PolicyModel, tool_executor: ToolExecutor) -> None:
        self.model = model
        self.tool_executor = tool_executor

    def run(self, verification_input: VerificationInput) -> VerificationTrace:
        trace = self.model.propose_trace(verification_input)
        tool_results = self.tool_executor.execute_many(trace.tool_calls)
        enriched_trace = trace.model_copy(update={"tool_results": tool_results})
        return finalize_trace(enriched_trace)
