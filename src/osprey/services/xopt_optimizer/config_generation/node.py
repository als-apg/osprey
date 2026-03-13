"""Config Generation Node for XOpt Optimizer Service.

This node generates OptimizationConfig dicts and prepares approval interrupt data.
It follows the Python executor's analyzer pattern where the node that generates
content also creates the approval interrupt data.

Supports two modes (configured via osprey.xopt_optimizer.config_generation.mode):
- "structured": LLM with structured output fills in config fields
- "mock": Placeholder config for quick testing (use for fast iteration)

DO NOT add accelerator-specific parameters without operator input.
"""

from typing import Any

from osprey.utils.config import get_model_config, get_xopt_optimizer_config
from osprey.utils.logger import get_logger

from ..exceptions import ConfigGenerationError
from ..execution.api_client import TuningScriptsAPIError, TuningScriptsClient
from ..models import XOptError, XOptExecutionState, XOptStrategy

logger = get_logger("xopt_optimizer")

# Allowed algorithm values for validation
_ALLOWED_ALGORITHMS = frozenset({
    "upper_confidence_bound",
    "expected_improvement",
    "mobo",
    "random",
})


def _get_config_generation_config() -> dict[str, Any]:
    """Get config generation configuration from osprey config.

    Reads from config structure:
        xopt_optimizer:
          config_generation:
            mode: "structured"
            model_config_name: "xopt_config_generation"
            default_algorithm: "upper_confidence_bound"
            default_environment: null

    Returns:
        Configuration dict with mode, model_config, and defaults
    """
    xopt_config = get_xopt_optimizer_config()
    gen_config = xopt_config.get("config_generation", {})

    # Resolve model config from name reference
    model_config = None
    model_config_name = gen_config.get("model_config_name", "xopt_config_generation")
    try:
        model_config = get_model_config(model_config_name)
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
        "mode": gen_config.get("mode", "mock"),
        "model_config": model_config,
        "default_algorithm": gen_config.get("default_algorithm", "upper_confidence_bound"),
        "default_environment": gen_config.get("default_environment"),
    }


def _generate_placeholder_config(objective: str, strategy: XOptStrategy) -> dict[str, Any]:
    """Generate a placeholder optimization config dict.

    Used when config_generation.mode is "mock".

    DO NOT add accelerator-specific parameters without operator input.
    """
    algorithm = "random" if strategy == XOptStrategy.EXPLORATION else "upper_confidence_bound"
    return {
        "algorithm": algorithm,
        "n_iterations": 20,
        "note": (
            f"Placeholder config for: {objective} (strategy: {strategy.value}). "
            "Set config_generation.mode: 'structured' to use the LLM agent."
        ),
    }


async def _generate_config_with_agent(
    objective: str,
    strategy: XOptStrategy,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate config using the structured-output agent.

    Args:
        objective: The optimization objective
        strategy: The selected strategy
        model_config: Model configuration for the agent

    Returns:
        Config dict from the agent
    """
    from .agent import create_config_generation_agent

    agent = create_config_generation_agent(model_config=model_config)

    try:
        return await agent.generate_config(
            objective=objective,
            strategy=strategy.value,
        )
    except Exception as e:
        logger.warning(f"Structured agent failed, falling back to mock: {e}")
        return _generate_placeholder_config(objective, strategy)


async def _resolve_environment(
    config: dict[str, Any], node_logger: Any
) -> None:
    """Resolve environment_name if missing by asking the user.

    Queries the tuning_scripts API for available environments, then
    uses a LangGraph interrupt to present the options and wait for
    the user's choice.

    If only one valid environment exists it is auto-selected silently.

    Modifies ``config`` in place.
    """
    if config.get("environment_name"):
        return

    # Fetch available environments from the API
    try:
        client = TuningScriptsClient()
        environments = await client.list_environments()
    except (TuningScriptsAPIError, Exception) as e:
        raise ConfigGenerationError(
            "No environment_name configured and the tuning_scripts API is unreachable. "
            "Set xopt_optimizer.config_generation.default_environment in config.yml "
            "or ensure the tuning_scripts API is running.",
            generated_config=config,
            validation_errors=[f"Missing environment_name, API unreachable: {e}"],
        ) from e

    valid_envs = [env for env in environments if env.get("valid", False)]

    if not valid_envs:
        available = [env.get("name", "?") for env in environments]
        raise ConfigGenerationError(
            "No valid optimization environments found on the tuning_scripts API. "
            f"Available (invalid): {available}. "
            "Configure a valid environment or set default_environment in config.yml.",
            generated_config=config,
            validation_errors=["No valid environments available"],
        )

    # Single environment — auto-select
    if len(valid_envs) == 1:
        config["environment_name"] = valid_envs[0]["name"]
        node_logger.info(f"Auto-selected environment: {valid_envs[0]['name']}")
        return

    # Multiple environments — auto-select the first valid one as fallback.
    # (Environment selection is normally handled at the capability level via
    # a question interrupt; this path runs only when the capability didn't
    # resolve an environment ahead of time.)
    first = valid_envs[0]
    config["environment_name"] = first["name"]
    node_logger.info(
        f"Auto-selected first valid environment: {first['name']} "
        f"(from {len(valid_envs)} available)"
    )


def _match_environment(
    choice: str, environments: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Match a user's choice (number or name) to an environment.

    Returns the matched environment dict, or None if no match found.
    """
    # Try as a 1-based index
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(environments):
            return environments[idx]
    except ValueError:
        pass

    # Try exact name match
    for env in environments:
        if env["name"] == choice:
            return env

    # Try case-insensitive prefix match
    lower = choice.lower()
    for env in environments:
        if env["name"].lower().startswith(lower):
            return env

    return None


def _validate_config(config: dict[str, Any]) -> None:
    """Validate the generated optimization config.

    Checks that required keys are present and algorithm is in the allowed set.
    """
    if not config:
        raise ConfigGenerationError(
            "Generated config is empty",
            generated_config=config,
            validation_errors=["Empty configuration"],
        )

    algorithm = config.get("algorithm")
    if algorithm and algorithm not in _ALLOWED_ALGORITHMS:
        raise ConfigGenerationError(
            f"Invalid algorithm: {algorithm}",
            generated_config=config,
            validation_errors=[f"Algorithm '{algorithm}' not in {sorted(_ALLOWED_ALGORITHMS)}"],
        )


def create_config_generation_node():
    """Create the config generation node for LangGraph integration.

    This factory function creates a node that generates optimization config
    dicts and prepares approval interrupt data.

    The generation mode is controlled via configuration:
    - osprey.xopt_optimizer.config_generation.mode: "mock" | "structured"

    Returns:
        Async function that takes XOptExecutionState and returns state updates
    """

    async def config_generation_node(state: XOptExecutionState) -> dict[str, Any]:
        """Generate optimization config.

        Supports two modes:
        - "mock": Fast placeholder generation for testing (default)
        - "structured": LLM with structured output fills in config fields

        Also prepares approval interrupt data following the Python
        executor's analyzer pattern.
        """
        node_logger = get_logger("xopt_optimizer", state=state)

        # Get configuration
        gen_config = _get_config_generation_config()
        mode = gen_config.get("mode", "mock")
        is_mock = mode == "mock"
        mode_indicator = " (mock)" if is_mock else ""

        node_logger.status(f"Generating optimization config{mode_indicator}...")

        # Track generation attempts
        attempt = state.get("config_generation_attempt", 0) + 1
        request = state.get("request")
        strategy = state.get("selected_strategy", XOptStrategy.EXPLORATION)
        objective = request.optimization_objective if request else "Unknown objective"

        try:
            # Generate config based on mode
            if mode == "structured":
                optimization_config = await _generate_config_with_agent(
                    objective=objective,
                    strategy=strategy,
                    model_config=gen_config.get("model_config"),
                )
            else:
                optimization_config = _generate_placeholder_config(objective, strategy)

            # Honor pre-resolved environment from capability (highest priority)
            if request and request.environment_name and not optimization_config.get("environment_name"):
                optimization_config["environment_name"] = request.environment_name

            # Apply defaults from config if not already set by the generator
            default_env = gen_config.get("default_environment")
            if default_env and not optimization_config.get("environment_name"):
                optimization_config["environment_name"] = default_env

            default_algo = gen_config.get("default_algorithm")
            if default_algo and not optimization_config.get("algorithm"):
                optimization_config["algorithm"] = default_algo

            # Resolve environment_name from the API if still missing
            await _resolve_environment(optimization_config, node_logger)

            # Validate config
            _validate_config(optimization_config)

            node_logger.key_info(f"Optimization config generated{mode_indicator}")

            # Prepare approval interrupt data (following Python executor pattern)
            requires_approval = request.require_approval if request else True

            if requires_approval:
                from osprey.approval.approval_system import create_xopt_approval_interrupt

                machine_state_details = state.get("machine_state_details")

                approval_interrupt_data = create_xopt_approval_interrupt(
                    optimization_config=optimization_config,
                    strategy=strategy.value,
                    objective=objective,
                    machine_state_details=machine_state_details,
                    step_objective=f"Execute XOpt optimization: {objective}",
                )

                return {
                    "optimization_config": optimization_config,
                    "config_generation_attempt": attempt,
                    "config_generation_failed": False,
                    "requires_approval": True,
                    "approval_interrupt_data": approval_interrupt_data,
                    "current_stage": "approval",
                }
            else:
                return {
                    "optimization_config": optimization_config,
                    "config_generation_attempt": attempt,
                    "config_generation_failed": False,
                    "requires_approval": False,
                    "current_stage": "execution",
                }

        except ConfigGenerationError:
            raise

        except Exception as e:
            node_logger.warning(f"Config generation failed: {e}")

            error = XOptError(
                error_type="config_generation",
                error_message=str(e),
                stage="config_generation",
                attempt_number=attempt,
            )
            error_chain = list(state.get("error_chain", [])) + [error]

            # Check retry limit
            max_retries = request.retries if request else 3
            retry_limit_exceeded = len(error_chain) >= max_retries

            return {
                "config_generation_attempt": attempt,
                "config_generation_failed": True,
                "error_chain": error_chain,
                "is_failed": retry_limit_exceeded,
                "failure_reason": (
                    f"Config generation failed after {max_retries} attempts"
                    if retry_limit_exceeded
                    else None
                ),
                "current_stage": "config_gen" if not retry_limit_exceeded else "failed",
            }

    return config_generation_node
