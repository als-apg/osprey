"""Structured Output Agent for Optimization Config Generation.

This module provides an agent that generates OptimizationConfig dicts for the
tuning_scripts API. Instead of generating raw Xopt YAML, the LLM fills in
structured config fields (environment, algorithm, iterations, etc.) which are
then submitted as JSON to the ``/optimization/start`` endpoint.

This simplifies the LLM's job from "generate valid Xopt YAML" to "fill in
config fields" and reduces validation complexity.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from osprey.models.langchain import get_langchain_model
from osprey.utils.logger import get_logger

logger = get_logger("xopt_optimizer")


# =============================================================================
# STRUCTURED OUTPUT MODEL
# =============================================================================


class OptimizationConfigOutput(BaseModel):
    """Structured output model mirroring tuning_scripts' OptimizationConfig schema.

    The LLM fills in these fields based on the optimization objective and strategy.
    Fields left as None are excluded from the final config dict.
    """

    environment_name: str | None = Field(
        default=None,
        description="Badger/Xopt environment name (e.g. 'als_injector_sim')",
    )
    objective_name: str | None = Field(
        default=None,
        description="Name of the objective to optimize",
    )
    algorithm: str = Field(
        default="upper_confidence_bound",
        description="Algorithm type: upper_confidence_bound, expected_improvement, mobo, random",
    )
    n_iterations: int = Field(
        default=20,
        description="Number of optimization iterations to run",
    )
    n_initial_samples: int | None = Field(
        default=None,
        description="Number of initial random samples before Bayesian optimization",
    )
    variables: list[str] | None = Field(
        default=None,
        description="List of variable names to optimize",
    )
    variable_overrides: dict[str, Any] | None = Field(
        default=None,
        description="Per-variable overrides (bounds, types, etc.)",
    )


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

CONFIG_GENERATION_PROMPT = """You are an optimization configuration generator for accelerator tuning.

You must select appropriate settings for an optimization run based on the user's
objective and strategy.

## Algorithm Selection Guide

Based on the optimization strategy:

**Exploration** (map the parameter space):
- Use "random" or "upper_confidence_bound" (with high exploration weight)
- Higher n_initial_samples for broader coverage

**Optimization** (converge on optimal values):
- Use "expected_improvement" for single-objective
- Use "mobo" for multi-objective
- Use "upper_confidence_bound" for balanced explore/exploit

## Available Algorithms
- "upper_confidence_bound" — Bayesian optimization with UCB acquisition (default, good general choice)
- "expected_improvement" — Bayesian optimization with EI acquisition (good for exploitation)
- "mobo" — Multi-objective Bayesian optimization
- "random" — Random sampling (good for initial exploration)

## Important Notes
- Use generic/placeholder names unless the user provides specific names
- Do NOT invent specific accelerator channel names
- Set n_iterations based on the scope of the task (default 20)
- Set n_initial_samples only if you want to override the default
"""


# =============================================================================
# AGENT CLASS
# =============================================================================


class ConfigGenerationAgent:
    """Agent for generating optimization config dicts via structured LLM output.

    Uses ``model.with_structured_output(OptimizationConfigOutput)`` for a single
    LLM call that returns a validated Pydantic model, then dumps it to a dict.
    """

    def __init__(self, model_config: dict[str, Any] | None = None):
        self.model_config = model_config

    def _get_model(self):
        if self.model_config:
            return get_langchain_model(model_config=self.model_config)
        raise ValueError(
            "No model_config provided to ConfigGenerationAgent. "
            "Ensure xopt_optimizer.config_generation.model_config_name is set in config.yml "
            "or that 'orchestrator' model is configured as fallback."
        )

    async def generate_config(
        self,
        objective: str,
        strategy: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate an optimization config dict using structured LLM output.

        Args:
            objective: The optimization objective
            strategy: The selected strategy ("exploration" or "optimization")
            context: Optional additional context

        Returns:
            Config dict with None values excluded
        """
        model = self._get_model()
        structured_model = model.with_structured_output(OptimizationConfigOutput)

        user_message = (
            f"Generate an optimization configuration for:\n\n"
            f"**Objective:** {objective}\n"
            f"**Strategy:** {strategy}\n"
        )
        if context:
            user_message += f"\n**Additional Context:** {context}\n"

        logger.info("Generating optimization config via structured output...")

        result = await structured_model.ainvoke(
            [
                {"role": "system", "content": CONFIG_GENERATION_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )

        config = result.model_dump(exclude_none=True)
        logger.info(f"Config generation complete: {config}")
        return config


def create_config_generation_agent(
    model_config: dict[str, Any] | None = None,
) -> ConfigGenerationAgent:
    """Factory function to create a config generation agent.

    Args:
        model_config: Optional model configuration

    Returns:
        Configured ConfigGenerationAgent instance
    """
    return ConfigGenerationAgent(model_config=model_config)
