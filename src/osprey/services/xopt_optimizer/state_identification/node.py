"""State Identification Node for XOpt Optimizer Service.

This node assesses machine readiness for optimization.

Supports two modes (configured via xopt_optimizer.state_identification.mode):
- "react": ReAct agent with tools for reading reference files, querying channels,
           and determining machine state (default)
- "mock": Fast placeholder that always returns READY

Configuration:
    xopt_optimizer:
      state_identification:
        mode: "react"  # or "mock"
        mock_files: true  # Use mock file data (for testing without real files)
        reference_path: "path/to/docs"  # Optional path to reference files
        model_config_name: "xopt_state_identification"  # Model config reference
"""

from typing import Any

from osprey.utils.config import get_model_config, get_xopt_optimizer_config
from osprey.utils.logger import get_logger

from ..models import MachineState, XOptExecutionState

logger = get_logger("xopt_optimizer")


# =============================================================================
# CONFIGURATION
# =============================================================================


def _get_state_identification_config() -> dict[str, Any]:
    """Get state identification configuration from osprey config.

    Reads from config structure:
        xopt_optimizer:
          state_identification:
            mode: "mock"  # or "react"
            mock_files: true  # Use mock file data (default: true for testing)
            reference_path: "path/to/docs"  # Optional path to reference files
            model_config_name: "xopt_state_identification"  # References models section

    Returns:
        Configuration dict with mode, mock_files, reference_path, and model_config
    """
    xopt_config = get_xopt_optimizer_config()
    state_id_config = xopt_config.get("state_identification", {})

    # Resolve model config from name reference
    # Falls back to "orchestrator" model if xopt-specific model not configured
    model_config = None
    model_config_name = state_id_config.get("model_config_name", "xopt_state_identification")
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
        "mode": state_id_config.get("mode", "react"),  # Default to react agent
        "mock_files": state_id_config.get("mock_files", True),  # Default to mock files
        "reference_path": state_id_config.get("reference_path"),  # Optional
        "model_config": model_config,
    }


# =============================================================================
# MOCK STATE ASSESSMENT
# =============================================================================


def _assess_state_mock() -> tuple[MachineState, dict[str, Any]]:
    """Assess machine state using mock logic (for testing).

    Always returns READY with placeholder details.

    Returns:
        Tuple of (MachineState, details dict)
    """
    return MachineState.READY, {
        "assessment": "mock",
        "note": "Mock implementation - always returns READY",
    }


# =============================================================================
# REACT AGENT STATE ASSESSMENT
# =============================================================================


async def _assess_state_react(
    objective: str,
    model_config: dict[str, Any],
    mock_files: bool = True,
    reference_path: str | None = None,
) -> tuple[MachineState, dict[str, Any]]:
    """Assess machine state using ReAct agent with tools.

    The agent:
    1. Reads reference documentation about machine ready criteria
    2. Checks current channel values from the control system
    3. Determines if the machine is READY, NOT_READY, or UNKNOWN

    Args:
        objective: The optimization objective
        model_config: Model configuration for the agent
        mock_files: If True, use mock reference file data
        reference_path: Path to reference documentation (if not using mock)

    Returns:
        Tuple of (MachineState, details dict)
    """
    from .agent import create_state_identification_agent

    agent = create_state_identification_agent(
        reference_path=reference_path,
        mock_files=mock_files,
        model_config=model_config,
    )

    try:
        machine_state, details = await agent.assess_state(objective=objective)
        return machine_state, details
    except Exception as e:
        logger.warning(f"ReAct agent failed, falling back to mock: {e}")
        return _assess_state_mock()


# =============================================================================
# NODE FACTORY
# =============================================================================


def create_state_identification_node():
    """Create the state identification node for LangGraph integration.

    This factory function creates a node that assesses machine readiness
    for optimization.

    The assessment mode is controlled via configuration:
    - xopt_optimizer.state_identification.mode: "mock" | "react"

    Returns:
        Async function that takes XOptExecutionState and returns state updates
    """

    async def state_identification_node(state: XOptExecutionState) -> dict[str, Any]:
        """Assess machine readiness for optimization.

        Supports two modes:
        - "mock": Fast placeholder that always returns READY (default)
        - "react": ReAct agent with tools for reading reference files and channels
        """
        node_logger = get_logger("xopt_optimizer", state=state)

        # Get configuration
        state_id_config = _get_state_identification_config()
        mode = state_id_config.get("mode", "react")
        is_mock = mode == "mock"

        request = state.get("request")
        objective = request.optimization_objective if request else "Unknown objective"

        # Log with (mock) indicator if in mock mode
        mode_indicator = " (mock)" if is_mock else ""
        node_logger.status(f"Assessing machine state{mode_indicator}...")

        try:
            if mode == "react":
                machine_state, details = await _assess_state_react(
                    objective=objective,
                    model_config=state_id_config.get("model_config"),
                    mock_files=state_id_config.get("mock_files", True),
                    reference_path=state_id_config.get("reference_path"),
                )
            else:
                machine_state, details = _assess_state_mock()

            # Log result with (mock) indicator
            node_logger.key_info(f"Machine state: {machine_state.value.upper()}{mode_indicator}")

            return {
                "machine_state": machine_state,
                "machine_state_details": details,
                "current_stage": "decision",
            }

        except Exception as e:
            node_logger.error(f"State assessment failed: {e}")

            # Fall back to mock on error
            node_logger.warning("Falling back to mock state assessment due to error")
            machine_state, details = _assess_state_mock()

            node_logger.key_info(f"Machine state: {machine_state.value.upper()} (mock)")

            return {
                "machine_state": machine_state,
                "machine_state_details": details,
                "current_stage": "decision",
            }

    return state_identification_node
