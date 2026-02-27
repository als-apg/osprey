"""Decision Node for XOpt Optimizer Service.

This node routes the workflow based on machine state assessment,
selecting the appropriate optimization strategy.

Supports two modes (configured via xopt_optimizer.decision.mode):
- "llm": LLM-based decision making with structured output
- "mock": Fast placeholder that always selects exploration (default)
"""

from typing import Any

from pydantic import BaseModel, Field

from osprey.utils.config import get_model_config, get_xopt_optimizer_config
from osprey.utils.logger import get_logger

from ..models import MachineState, XOptExecutionState, XOptStrategy

logger = get_logger("xopt_optimizer")


# =============================================================================
# STRUCTURED OUTPUT MODEL FOR LLM DECISION
# =============================================================================


class StrategyDecision(BaseModel):
    """Structured output for LLM strategy decision.

    This model is used with LangChain's `with_structured_output` to ensure
    the LLM returns a valid strategy selection with reasoning.
    """

    strategy: XOptStrategy = Field(
        description="The optimization strategy: 'exploration' or 'optimization'"
    )
    reasoning: str = Field(description="Brief explanation of why this strategy was selected")


# =============================================================================
# SYSTEM PROMPT FOR LLM DECISION
# =============================================================================

DECISION_SYSTEM_PROMPT = """You are an expert accelerator optimizer decision system.

Your task is to select the appropriate optimization strategy based on:
1. The **Machine State** value (ready, not_ready, or unknown) - this is the authoritative assessment
2. The user's optimization objective

## Available Strategies

- **exploration**: Use when the machine is READY and starting a new optimization campaign.
  This strategy prioritizes coverage and discovery over immediate optimization.

- **optimization**: Use when the machine is READY, the objective is clear,
  and you want to aggressively optimize toward the goal. This strategy prioritizes
  finding the best solution quickly.

- **abort**: Use ONLY when machine_state is NOT_READY or UNKNOWN.

## Decision Guidelines

IMPORTANT: Trust the Machine State value. The state assessment has already been performed
by a dedicated agent that checked all safety criteria. Your job is to select a strategy
based on that assessment.

1. If machine_state is "ready" -> Select EXPLORATION or OPTIMIZATION based on objective
2. If machine_state is "not_ready" or "unknown" -> Select ABORT
3. For new optimization objectives, prefer EXPLORATION
4. For well-defined objectives with good starting conditions, consider OPTIMIZATION

Select the most appropriate strategy and explain your reasoning briefly."""


# =============================================================================
# CONFIGURATION
# =============================================================================


def _get_decision_config() -> dict[str, Any]:
    """Get decision node configuration from osprey config.

    Reads from config structure:
        xopt_optimizer:
          decision:
            mode: "mock"  # or "llm"
            model_config_name: "xopt_decision"  # References models section

    Returns:
        Configuration dict with mode and model_config
    """
    xopt_config = get_xopt_optimizer_config()
    decision_config = xopt_config.get("decision", {})

    # Resolve model config from name reference
    # Falls back to "orchestrator" model if xopt-specific model not configured
    model_config = None
    model_config_name = decision_config.get("model_config_name", "xopt_decision")
    try:
        model_config = get_model_config(model_config_name)
        # Check if the model config is valid (has provider)
        if not model_config or not model_config.get("provider"):
            logger.debug(
                f"Model '{model_config_name}' not configured, falling back to orchestrator"
            )
            model_config = get_model_config("orchestrator")
    except Exception as e:
        logger.warning(
            f"Could not load model config '{model_config_name}': {e}, falling back to orchestrator"
        )
        model_config = get_model_config("orchestrator")

    return {
        "mode": decision_config.get("mode", "mock"),  # Default to mock for fast testing
        "model_config": model_config,
    }


# =============================================================================
# LLM-BASED DECISION
# =============================================================================


async def _make_llm_decision(
    objective: str,
    machine_state: MachineState,
    machine_state_details: dict[str, Any] | None,
    model_config: dict[str, Any],
) -> StrategyDecision:
    """Make strategy decision using LLM with structured output.

    Args:
        objective: The optimization objective
        machine_state: Current machine state assessment
        machine_state_details: Additional machine state details
        model_config: Model configuration for the LLM

    Returns:
        StrategyDecision with selected strategy and reasoning
    """
    from osprey.models.langchain import get_langchain_model

    # Get the model and configure for structured output
    model = get_langchain_model(model_config=model_config)
    structured_model = model.with_structured_output(StrategyDecision)

    # Build the user message with context
    # Only include key summary info, not raw agent response (which may contain confusing intermediate reasoning)
    details_text = ""
    if machine_state_details:
        # Extract only the relevant summary fields, excluding raw_response
        summary_fields = {}
        if "channels_checked" in machine_state_details:
            summary_fields["channels_checked"] = machine_state_details["channels_checked"]
        if "key_observations" in machine_state_details:
            summary_fields["key_observations"] = machine_state_details["key_observations"]
        if summary_fields:
            details_text = f"\n\nMachine State Details:\n{summary_fields}"

    user_message = f"""Please select the appropriate optimization strategy.

**Optimization Objective:** {objective}
**Machine State:** {machine_state.value}
{details_text}

Based on this information, which strategy should we use?"""

    # Invoke the model
    result = await structured_model.ainvoke(
        [
            {"role": "system", "content": DECISION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )

    return result


# =============================================================================
# MOCK DECISION
# =============================================================================


def _make_mock_decision(
    machine_state: MachineState,
    machine_state_details: dict[str, Any] | None,
) -> StrategyDecision:
    """Make strategy decision using mock logic (for testing).

    Always selects exploration unless machine is not ready.

    Args:
        machine_state: Current machine state assessment
        machine_state_details: Additional machine state details

    Returns:
        StrategyDecision with selected strategy and reasoning
    """
    if machine_state == MachineState.NOT_READY:
        reason = (
            machine_state_details.get("reason", "Machine not ready")
            if machine_state_details
            else "Machine not ready"
        )
        return StrategyDecision(
            strategy=XOptStrategy.ABORT,
            reasoning=reason,
        )

    if machine_state == MachineState.UNKNOWN:
        reason = (
            machine_state_details.get("reason", "Machine state unknown")
            if machine_state_details
            else "Machine state unknown"
        )
        return StrategyDecision(
            strategy=XOptStrategy.ABORT,
            reasoning=reason,
        )

    # Default to exploration for mock mode
    return StrategyDecision(
        strategy=XOptStrategy.EXPLORATION,
        reasoning="Machine ready, starting with exploration",
    )


# =============================================================================
# NODE FACTORY
# =============================================================================


def create_decision_node():
    """Create the decision node for LangGraph integration.

    This factory function creates a node that routes based on machine
    state assessment, selecting the appropriate optimization strategy.

    The decision mode is controlled via configuration:
    - xopt_optimizer.decision.mode: "mock" | "llm"

    Returns:
        Async function that takes XOptExecutionState and returns state updates
    """

    async def decision_node(state: XOptExecutionState) -> dict[str, Any]:
        """Route based on machine state assessment.

        Supports two modes:
        - "mock": Fast placeholder that always selects exploration (default)
        - "llm": LLM-based decision making with structured output
        """
        node_logger = get_logger("xopt_optimizer", state=state)
        # Get configuration
        decision_config = _get_decision_config()
        mode = decision_config.get("mode", "mock")
        is_mock = mode == "mock"
        mode_indicator = " (mock)" if is_mock else ""

        node_logger.status(f"Selecting optimization strategy{mode_indicator}...")

        machine_state = state.get("machine_state")
        machine_state_details = state.get("machine_state_details")
        request = state.get("request")
        objective = request.optimization_objective if request else "Unknown objective"

        try:
            # Make decision based on mode
            if mode == "llm":
                decision = await _make_llm_decision(
                    objective=objective,
                    machine_state=machine_state,
                    machine_state_details=machine_state_details,
                    model_config=decision_config.get("model_config"),
                )
            else:
                decision = _make_mock_decision(
                    machine_state=machine_state,
                    machine_state_details=machine_state_details,
                )

            # Handle abort strategy
            if decision.strategy == XOptStrategy.ABORT:
                node_logger.key_info(f"Strategy: ABORT{mode_indicator}")
                return {
                    "selected_strategy": XOptStrategy.ABORT,
                    "decision_reasoning": decision.reasoning,
                    "is_failed": True,
                    "failure_reason": f"Strategy decision: {decision.reasoning}",
                    "current_stage": "failed",
                }

            # Strategy selected successfully - log as key_info for visibility
            node_logger.key_info(f"Strategy: {decision.strategy.value.upper()}{mode_indicator}")

            return {
                "selected_strategy": decision.strategy,
                "decision_reasoning": decision.reasoning,
                "current_stage": "yaml_gen",
            }

        except Exception as e:
            node_logger.error(f"Strategy decision failed: {e}")

            # Fall back to mock decision on error
            node_logger.warning("Falling back to mock decision due to error")
            decision = _make_mock_decision(
                machine_state=machine_state,
                machine_state_details=machine_state_details,
            )

            if decision.strategy == XOptStrategy.ABORT:
                return {
                    "selected_strategy": XOptStrategy.ABORT,
                    "decision_reasoning": f"Fallback: {decision.reasoning}",
                    "is_failed": True,
                    "failure_reason": f"Strategy decision failed: {e}",
                    "current_stage": "failed",
                }

            return {
                "selected_strategy": decision.strategy,
                "decision_reasoning": f"Fallback: {decision.reasoning}",
                "current_stage": "yaml_gen",
            }

    return decision_node
