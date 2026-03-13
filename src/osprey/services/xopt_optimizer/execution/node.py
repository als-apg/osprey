"""Execution Node for XOpt Optimizer Service.

This node executes XOpt optimization runs by submitting the generated optimization
config to the tuning_scripts API and polling for results.

The tuning_scripts API (FastAPI + Redis + Xopt) handles the actual optimization
execution. This node acts as an HTTP client that:
1. Health-checks the API
2. Submits the optimization config via POST /optimization/start
3. Polls until completion via GET /optimization/{job_id}
4. Returns the full result state as the run_artifact

## Badger/XOpt Environment Integration

XOpt/Badger uses an "Environment" abstraction that defines the optimization problem:
- **variables**: Tunable parameters with bounds (e.g., magnet setpoints)
- **observables**: Measurable outputs (e.g., beam position, emittance)

The Environment communicates with the control system via an "Interface" that
implements `get_values()` and `set_values()`. This maps naturally to Osprey's
ConnectorFactory:

```python
# Example: Badger Interface using Osprey's ConnectorFactory
from osprey.connectors.factory import ConnectorFactory

class OspreyInterface(Interface):
    name = "osprey"

    async def get_values(self, channel_names):
        connector = await ConnectorFactory.create_control_system_connector()
        return {name: (await connector.read_channel(name)).value
                for name in channel_names}

    async def set_values(self, channel_inputs):
        connector = await ConnectorFactory.create_control_system_connector()
        for name, value in channel_inputs.items():
            await connector.write_channel(name, value)
```

This allows XOpt to work with any control system backend (EPICS, mock, etc.)
configured in Osprey's config.yml, including generated soft IOCs.

See: https://github.com/xopt-org/Badger (Environment and Interface classes)
"""

from typing import Any

from osprey.utils.logger import get_logger

from ..models import XOptExecutionState
from .api_client import TuningScriptsAPIError, TuningScriptsClient

logger = get_logger("xopt_optimizer")


async def _run_xopt_placeholder(config: dict[str, Any]) -> dict[str, Any]:
    """Placeholder for XOpt execution (used when API is unavailable).

    Returns mock results for testing without a running tuning_scripts API.
    """
    return {
        "status": "completed",
        "evaluations": 0,
        "best_value": None,
        "best_parameters": {},
        "config_used": config,
        "note": "This is a placeholder result. The tuning_scripts API was not available.",
    }


def create_executor_node():
    """Create the execution node for LangGraph integration.

    This factory function creates a node that executes XOpt optimization
    runs by submitting YAML to the tuning_scripts API. Falls back to
    placeholder execution if the API is unreachable.

    Returns:
        Async function that takes XOptExecutionState and returns state updates
    """

    async def executor_node(state: XOptExecutionState) -> dict[str, Any]:
        """Execute XOpt optimization via the tuning_scripts API."""
        node_logger = get_logger("xopt_optimizer", state=state)
        node_logger.status("Executing XOpt optimization...")

        optimization_config = state.get("optimization_config")

        if not optimization_config:
            node_logger.error("No optimization config available for execution")
            return {
                "execution_error": "No optimization config generated",
                "execution_failed": True,
                "is_failed": True,
                "failure_reason": "Missing optimization config for execution",
                "current_stage": "failed",
            }

        client = TuningScriptsClient()

        try:
            # 1. Health check
            node_logger.info("Checking tuning_scripts API health...")
            await client.health_check()
            node_logger.info("API health check passed")

            # 2. Submit config
            node_logger.info("Submitting optimization config to tuning_scripts API...")
            job_id = await client.submit_config(optimization_config)
            node_logger.info(f"Optimization job submitted: {job_id}")

            # 3. Poll until complete
            node_logger.info(f"Polling job {job_id} for results...")
            full_state = await client.poll_until_complete(job_id)
            node_logger.info(f"Job {job_id} completed with status: {full_state.get('status')}")

            # 4. Build run_artifact
            run_artifact = {
                "job_id": job_id,
                "status": full_state.get("status", "unknown"),
                "data": full_state.get("data"),
                "environment_name": full_state.get("environment_name"),
                "objective_name": full_state.get("objective_name"),
                "variable_names": full_state.get("variable_names"),
                "results_path": full_state.get("results_path"),
                "config_used": optimization_config,
                "logs": full_state.get("logs", ""),
            }

            return {
                "run_artifact": run_artifact,
                "execution_failed": False,
                "current_stage": "analysis",
            }

        except TuningScriptsAPIError as e:
            if e.status_code is not None:
                # API returned an error response — real failure
                node_logger.error(f"Tuning scripts API error: {e}")
                return {
                    "execution_error": str(e),
                    "execution_failed": True,
                    "is_failed": True,
                    "failure_reason": f"Tuning scripts API error: {e}",
                    "current_stage": "failed",
                }

            # Connection error (API not running) — fall back to placeholder
            node_logger.warning(f"API unreachable ({e}), falling back to placeholder execution")
            run_artifact = await _run_xopt_placeholder(optimization_config)
            return {
                "run_artifact": run_artifact,
                "execution_failed": False,
                "current_stage": "analysis",
            }

        except Exception as e:
            node_logger.error(f"XOpt execution failed: {e}")
            return {
                "execution_error": str(e),
                "execution_failed": True,
                "is_failed": True,
                "failure_reason": f"XOpt execution error: {e}",
                "current_stage": "failed",
            }

    return executor_node
