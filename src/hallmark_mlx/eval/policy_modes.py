"""Policy-mode helpers shared by evaluation and Weco scripts."""

from __future__ import annotations

from hallmark_mlx.config import AppConfig
from hallmark_mlx.inference.bibtex_first_fallback_policy import BibtexFirstFallbackPolicyModel
from hallmark_mlx.inference.policy_runner import PolicyRunner, load_policy_model
from hallmark_mlx.inference.tool_executor import ToolExecutor
from hallmark_mlx.inference.warm_start_policy import WarmStartPolicyModel
from hallmark_mlx.types import FinalizationMode


def build_policy_runner(config: AppConfig, mode_name: str) -> PolicyRunner:
    """Build a runner for a named policy mode."""

    tool_executor = ToolExecutor(config.tools)
    if mode_name == "tool_only":
        return PolicyRunner(
            model=WarmStartPolicyModel(),
            tool_executor=tool_executor,
            finalization_mode=FinalizationMode.DETERMINISTIC,
        )
    if mode_name == "policy_deterministic":
        deterministic_config = config.model.model_copy(
            update={"finalization_mode": FinalizationMode.DETERMINISTIC},
        )
        return PolicyRunner(
            model=load_policy_model(deterministic_config),
            tool_executor=tool_executor,
            finalization_mode=FinalizationMode.DETERMINISTIC,
        )
    if mode_name == "bibtex_first_fallback":
        return PolicyRunner(
            model=BibtexFirstFallbackPolicyModel(config.model),
            tool_executor=tool_executor,
            finalization_mode=FinalizationMode.DETERMINISTIC,
        )
    if mode_name == "policy_generative":
        generative_config = config.model.model_copy(
            update={"finalization_mode": FinalizationMode.GENERATIVE},
        )
        return PolicyRunner(
            model=load_policy_model(generative_config),
            tool_executor=tool_executor,
            finalization_mode=FinalizationMode.GENERATIVE,
        )
    raise ValueError(f"Unsupported policy mode: {mode_name}")
