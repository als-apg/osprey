"""YAML Generation Node for XOpt Optimizer Service.

This node generates XOpt YAML configurations and prepares approval interrupt data.
It follows the Python executor's analyzer pattern where the node that generates
content also creates the approval interrupt data.

Supports two modes (configured via osprey.xopt_optimizer.yaml_generation.mode):
- "react": ReAct agent generates YAML (default) - adapts based on file availability:
  - If example files exist: Agent reads them and learns patterns
  - If no examples: Agent generates from built-in XOpt knowledge
- "mock": Placeholder YAML for quick testing (use for fast iteration)

Example YAML files are optional. If provided, place them in:
    osprey.xopt_optimizer.yaml_generation.examples_path: "path/to/yamls"

DO NOT add accelerator-specific YAML parameters without operator input.
"""

from pathlib import Path
from typing import Any

from osprey.utils.config import get_model_config, get_xopt_optimizer_config
from osprey.utils.logger import get_logger

from ..exceptions import YamlGenerationError
from ..models import XOptError, XOptExecutionState, XOptStrategy

logger = get_logger("xopt_optimizer")

# Default path for example YAML files (relative to working directory)
DEFAULT_EXAMPLES_PATH = "_agent_data/xopt_examples/yaml_templates"


def _get_yaml_generation_config() -> dict[str, Any]:
    """Get YAML generation configuration from osprey config.

    Reads from config structure:
        xopt_optimizer:
          yaml_generation:
            mode: "react"
            examples_path: "..."
            model_config_name: "xopt_yaml_generation"  # References models section

    Returns:
        Configuration dict with mode, examples_path, and model_config
    """
    xopt_config = get_xopt_optimizer_config()
    yaml_config = xopt_config.get("yaml_generation", {})

    # Resolve model config from name reference
    # Falls back to "orchestrator" model if xopt-specific model not configured
    model_config = None
    model_config_name = yaml_config.get("model_config_name", "xopt_yaml_generation")
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
        "mode": yaml_config.get("mode", "react"),  # Default to react (agent-based)
        "examples_path": yaml_config.get("examples_path"),  # None if not specified
        "model_config": model_config,
    }


def _generate_placeholder_yaml(objective: str, strategy: XOptStrategy) -> str:
    """Generate placeholder XOpt YAML configuration.

    PLACEHOLDER: This generates a minimal valid YAML structure.
    Used when yaml_generation.mode is "mock".

    DO NOT add accelerator-specific parameters without operator input.
    """
    return f"""# XOpt Optimization Configuration
# PLACEHOLDER - Generated for: {objective}
# Strategy: {strategy.value}

# NOTE: This is a MOCK configuration for testing the workflow.
# Set yaml_generation.mode: "react" to use the ReAct agent.

generator:
  name: random  # Placeholder generator
  # Real implementation would use appropriate generator based on strategy

evaluator:
  function: placeholder_objective
  # Real implementation would define actual objective function

vocs:
  variables:
    param_1:
      type: continuous
      lower: 0.0
      upper: 10.0
    param_2:
      type: continuous
      lower: -1.0
      upper: 1.0
  objectives:
    objective_1:
      type: minimize
  constraints: {{}}
  statics: {{}}

n_initial: 5
max_evaluations: 20

# NOTE: This is a placeholder configuration.
# Actual XOpt parameters will be determined based on:
# - Historical YAML examples from the facility
# - Operator-defined parameter bounds
# - Machine-specific safety constraints
"""


async def _generate_yaml_with_react_agent(
    objective: str,
    strategy: XOptStrategy,
    examples_path: str | None,
    model_config: dict[str, Any] | None = None,
) -> str:
    """Generate YAML using the ReAct agent.

    The agent dynamically adapts:
    - If examples_path has YAML files: Agent gets file tools and reads examples
    - If no examples: Agent generates from built-in XOpt knowledge

    Args:
        objective: The optimization objective
        strategy: The selected strategy
        examples_path: Path to example YAML files (optional)
        model_config: Optional model configuration for the agent

    Returns:
        Generated YAML configuration string
    """
    from .agent import create_yaml_generation_agent

    # Check if examples path exists - if not, agent will work without file tools
    if examples_path:
        path = Path(examples_path)
        if not path.exists():
            examples_path = None

    # Create and run the agent (it adapts based on whether examples exist)
    agent = create_yaml_generation_agent(
        examples_path=examples_path,
        model_config=model_config,
    )

    try:
        yaml_config = await agent.generate_yaml(
            objective=objective,
            strategy=strategy.value,
        )
        return yaml_config
    except Exception as e:
        logger.warning(f"ReAct agent failed, falling back to mock: {e}")
        return _generate_placeholder_yaml(objective, strategy)


def _validate_yaml(yaml_config: str) -> None:
    """Validate generated YAML configuration.

    PLACEHOLDER: Basic validation only.
    Real implementation would use XOpt schema validation.
    """
    if not yaml_config or not yaml_config.strip():
        raise YamlGenerationError(
            "Generated YAML is empty",
            generated_yaml=yaml_config,
            validation_errors=["Empty YAML configuration"],
        )
    # Future: Add XOpt schema validation


def create_yaml_generation_node():
    """Create the YAML generation node for LangGraph integration.

    This factory function creates a node that generates XOpt YAML
    configurations and prepares approval interrupt data.

    The generation mode is controlled via configuration:
    - osprey.xopt_optimizer.yaml_generation.mode: "mock" | "react"

    Returns:
        Async function that takes XOptExecutionState and returns state updates
    """

    async def yaml_generation_node(state: XOptExecutionState) -> dict[str, Any]:
        """Generate XOpt YAML configuration.

        Supports two modes:
        - "mock": Fast placeholder generation for testing (default)
        - "react": ReAct agent reads examples and generates YAML

        Also prepares approval interrupt data following the Python
        executor's analyzer pattern.
        """
        node_logger = get_logger("xopt_optimizer", state=state)

        # Get configuration
        yaml_gen_config = _get_yaml_generation_config()
        mode = yaml_gen_config.get("mode", "mock")
        is_mock = mode == "mock"
        mode_indicator = " (mock)" if is_mock else ""

        node_logger.status(f"Generating XOpt configuration{mode_indicator}...")

        # Track generation attempts
        attempt = state.get("yaml_generation_attempt", 0) + 1
        request = state.get("request")
        strategy = state.get("selected_strategy", XOptStrategy.EXPLORATION)
        objective = request.optimization_objective if request else "Unknown objective"

        try:
            # Generate YAML configuration based on mode
            if mode == "react":
                yaml_config = await _generate_yaml_with_react_agent(
                    objective=objective,
                    strategy=strategy,
                    examples_path=yaml_gen_config.get("examples_path", DEFAULT_EXAMPLES_PATH),
                    model_config=yaml_gen_config.get("model_config"),
                )
            else:
                yaml_config = _generate_placeholder_yaml(objective, strategy)

            # Validate YAML
            _validate_yaml(yaml_config)

            node_logger.key_info(f"YAML configuration generated{mode_indicator}")

            # Prepare approval interrupt data (following Python executor pattern)
            requires_approval = request.require_approval if request else True

            if requires_approval:
                # Import here to avoid circular imports
                from osprey.approval.approval_system import create_xopt_approval_interrupt

                machine_state_details = state.get("machine_state_details")

                approval_interrupt_data = create_xopt_approval_interrupt(
                    yaml_config=yaml_config,
                    strategy=strategy.value,
                    objective=objective,
                    machine_state_details=machine_state_details,
                    step_objective=f"Execute XOpt optimization: {objective}",
                )

                return {
                    "generated_yaml": yaml_config,
                    "yaml_generation_attempt": attempt,
                    "yaml_generation_failed": False,
                    "requires_approval": True,
                    "approval_interrupt_data": approval_interrupt_data,
                    "current_stage": "approval",
                }
            else:
                return {
                    "generated_yaml": yaml_config,
                    "yaml_generation_attempt": attempt,
                    "yaml_generation_failed": False,
                    "requires_approval": False,
                    "current_stage": "execution",
                }

        except YamlGenerationError:
            # Re-raise YAML generation errors
            raise

        except Exception as e:
            node_logger.warning(f"YAML generation failed: {e}")

            error = XOptError(
                error_type="yaml_generation",
                error_message=str(e),
                stage="yaml_generation",
                attempt_number=attempt,
            )
            error_chain = list(state.get("error_chain", [])) + [error]

            # Check retry limit
            max_retries = request.retries if request else 3
            retry_limit_exceeded = len(error_chain) >= max_retries

            return {
                "yaml_generation_attempt": attempt,
                "yaml_generation_failed": True,
                "error_chain": error_chain,
                "is_failed": retry_limit_exceeded,
                "failure_reason": (
                    f"YAML generation failed after {max_retries} attempts"
                    if retry_limit_exceeded
                    else None
                ),
                "current_stage": "yaml_gen" if not retry_limit_exceeded else "failed",
            }

    return yaml_generation_node
