import logging
from pathlib import Path
from typing import Any

from osprey_optimizer.api.job_tracker import JobLifecycle
from osprey_optimizer.config.logging import setup_logging
from osprey_optimizer.models.config import OptimizationConfig
from osprey_optimizer.services.execution import ExecutionService

logger = logging.getLogger(__name__)


def run_optimization_process(
    job_id: str, config_dict: dict[str, Any], results_dir: str | None = None
) -> None:
    """Execute the optimization process in a background job."""
    setup_logging(level=logging.INFO)
    logger.info("Starting background optimization process %s", job_id)

    with JobLifecycle(job_id):
        # config_dict is a dump of OptimizationConfig; validating it again
        # re-applies its defaults and validators.
        try:
            config = OptimizationConfig.model_validate(config_dict)
        except Exception:
            logger.exception("Invalid configuration passed to optimization process")
            raise

        logger.info("Configured job %s (Environment: %s)", job_id, config.environment_name)

        # job_id doubles as the control key so API/Web can pause/cancel this run.
        service = ExecutionService(
            control_key=job_id,
            specific_run_dir=Path(results_dir) if results_dir else None,
        )

        results_path = service.start_optimization(config)

        if results_path:
            logger.info("Optimization process %s completed. Results at %s", job_id, results_path)
        else:
            logger.warning("Optimization process %s finished without results", job_id)
