"""Create/start lifecycle for optimization jobs.

Split from the optimization router so the generic ``POST /jobs/{id}/start``
endpoint can drive an optimization start without importing a router module.
``create`` is non-reserving (no active-slot claim, status CREATED); ``start``
runs the atomic :func:`launch_job` claim+spawn.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import HTTPException, status

from osprey.services.optimizer.api.environment_ops import (
    environment_presets,
    lookup_environment,
    require_known_environment,
)
from osprey.services.optimizer.api.job_control import (
    launch_job,
    load_stored_config,
    persist_created_job,
    require_created_or_409,
    store_created_config,
)
from osprey.services.optimizer.api.job_status import read_job_status
from osprey.services.optimizer.api.job_tracker import JobTracker
from osprey.services.optimizer.api.runner import run_optimization_process
from osprey.services.optimizer.models.api import JobResponse
from osprey.services.optimizer.models.config import OptimizationConfig, machine_config
from osprey.services.optimizer.models.enums import JobType
from osprey.services.optimizer.models.exceptions import ConnectionError as OptimizerConnectionError
from osprey.services.optimizer.optimization.state import (
    get_optimization_reader,
    get_optimization_writer,
)
from osprey.services.optimizer.results import initialize_results_dir
from osprey.services.optimizer.services.draft_resolution import resolve_draft_config
from osprey.services.optimizer.services.plugin_registry import EnvironmentRegistry

logger = logging.getLogger(__name__)


def _materialize_objective_direction(
    config: OptimizationConfig, registry: EnvironmentRegistry
) -> None:
    """Record the objective's effective direction in *config* when it is unset.

    Direction is a property of the observable, not user input, so clients (the
    cockpit, the API) leave ``objective_direction`` unset. Resolve it from the
    environment here so even a CREATED draft's stored config is self-describing,
    mirroring the fallback the optimization engine applies at run time.

    Best-effort: if the objective is not a known observable, leave it unset and
    let start-time validation report the error.
    """
    if (
        config.objective_direction is not None
        or not config.objective_name
        or not config.environment_name
    ):
        return
    try:
        observable = lookup_environment(config.environment_name, registry).observables.get(
            config.objective_name
        )
    except Exception:
        # Leave direction unset on any lookup failure (unknown/invalid env, construction
        # error). Start-time validation re-resolves direction and rejects a bad env/objective,
        # so this stays a pure draft-creation step rather than a new failure mode.
        logger.warning(
            "Could not resolve objective direction for '%s' in '%s'; leaving unset",
            config.objective_name,
            config.environment_name,
            exc_info=True,
        )
        return
    if observable is not None:
        config.objective_direction = observable.direction


def _resolve_draft_or_422(config: OptimizationConfig, registry: EnvironmentRegistry) -> None:
    """Resolve the draft against live readbacks; 422 when resolution is impossible.

    The readbacks are the run's own target's: a window centred on the current
    value means the current value on the machine the run will drive, so a draft
    names its ``control_target`` from creation, not only at start.

    Runs before any persistence, so a failed create leaves no residue and a
    failed update keeps the previously stored config.
    """
    name = config.environment_name
    if name is None:  # callers validate presence; keep the type narrow here
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="environment_name is required to resolve an optimization draft.",
        )
    if config.control_target is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "control_target is required to resolve an optimization draft: its bounds are "
                "read from the machine the run will drive (va, standin or live)."
            ),
        )
    env_config = {
        **(config.environment_config or {}),
        **machine_config(config.control_target, config.execution_mode),
    }
    env = registry.load_environment(name, env_config)
    try:
        with env:
            resolve_draft_config(config, env)
    except (ValueError, OptimizerConnectionError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Cannot resolve draft against current machine state: {e}",
        ) from e


def _require_environment(
    config: OptimizationConfig, registry: EnvironmentRegistry, *, action: str
) -> None:
    """422 unless the config names a known environment; ``action`` labels the message."""
    if not config.environment_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"environment_name is required to {action} an optimization.",
        )
    require_known_environment(config.environment_name, registry)


_PRESET_FIELDS = ("variables", "algorithm", "n_iterations", "objective_name")


def _apply_preset(config: OptimizationConfig, registry: EnvironmentRegistry) -> None:
    """Fill the fields *config* leaves unset from the preset it names.

    A field the caller set, even to its default, is theirs. Overrides merge per
    variable, the caller's winning, so a draft can start from a preset and
    widen one window. 422 when the environment has no such preset.
    """
    if config.preset is None or config.environment_name is None:
        return
    presets = environment_presets(registry, config.environment_name)
    preset = presets.get(config.preset)
    if preset is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Environment '{config.environment_name}' has no preset '{config.preset}'. "
                f"Available: {sorted(presets)}"
            ),
        )
    for field in _PRESET_FIELDS:
        value = getattr(preset, field)
        if value is not None and field not in config.model_fields_set:
            setattr(config, field, value)
    if preset.variable_overrides:
        config.variable_overrides = {
            **preset.variable_overrides,
            **(config.variable_overrides or {}),
        }


def _prepare_draft(config: OptimizationConfig, registry: EnvironmentRegistry) -> None:
    """Enrich + resolve a draft config in place (shared by create and update)."""
    _materialize_objective_direction(config, registry)
    _resolve_draft_or_422(config, registry)


def create_optimization_job(
    config: OptimizationConfig,
    tracker: JobTracker,
    registry: EnvironmentRegistry,
) -> str:
    """Validate + persist an optimization config and register a CREATED job.

    Non-reserving: never claims the active slot and never 409s on a running job.
    The single-running-job guard fires later, at start.
    """
    _require_environment(config, registry, action="create")
    _apply_preset(config, registry)
    if not config.objective_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="objective_name is required to create an optimization.",
        )
    _prepare_draft(config, registry)

    job_id = str(uuid.uuid4())
    persist_created_job(
        tracker=tracker,
        job_id=job_id,
        job_type=JobType.OPTIMIZATION,
        payload_json=config.model_dump_json(),
        make_writer=get_optimization_writer,
    )

    logger.info("Created optimization job %s (not started)", job_id)
    return job_id


def update_optimization_config(
    job_id: str,
    config: OptimizationConfig,
    tracker: JobTracker,
    registry: EnvironmentRegistry,
) -> JobResponse:
    """Replace a CREATED job's stored config and refresh its TTL (live-sync).

    409 unless the job is in CREATED state — a running/finished job's config is
    immutable. Partial configs are allowed; full validation (objective present,
    objective exists in the env) happens at start, so the explicit-objective rule
    is untouched.
    """
    require_created_or_409(job_id, "update", get_optimization_reader(job_id).get_status())
    _require_environment(config, registry, action="update")
    _apply_preset(config, registry)
    _prepare_draft(config, registry)

    store_created_config(tracker, job_id, config.model_dump_json())

    logger.info("Updated config for optimization job %s", job_id)
    return read_job_status(tracker, job_id)


def read_optimization_config(job_id: str) -> OptimizationConfig:
    """Return a job's stored optimization config (read-side mirror of update).

    404 if no config is stored (job never existed, or the CREATED draft expired).
    Readable in any status — config is immutable once running, but still readable.
    """
    raw = load_stored_config(
        job_id,
        missing_status=status.HTTP_404_NOT_FOUND,
        missing_detail=f"No stored config for job {job_id} (unknown or expired).",
    )
    return OptimizationConfig.model_validate_json(raw)


def start_optimization_job(
    job_id: str,
    tracker: JobTracker,
    registry: EnvironmentRegistry,  # noqa: ARG001 - kept for the router's call shape
) -> JobResponse:
    """Start the worker for a CREATED optimization job.

    409 unless the job is in CREATED state; reads the stored config and runs the
    atomic ``launch_job`` claim+spawn (409 if another job is already running).
    """
    require_created_or_409(job_id, "start", get_optimization_reader(job_id).get_status())

    raw = load_stored_config(
        job_id,
        missing_status=status.HTTP_409_CONFLICT,
        missing_detail=f"Cannot start job {job_id}: stored config not found (expired). Re-create the job.",
    )
    config = OptimizationConfig.model_validate_json(raw)

    try:
        results_dir = initialize_results_dir(config, job_id)
        results_dir_str = str(results_dir)
    except Exception as e:
        logger.exception("Failed to initialize results storage")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize results storage: {e}",
        ) from e

    pid = launch_job(
        tracker=tracker,
        job_id=job_id,
        job_type=JobType.OPTIMIZATION,
        make_writer=get_optimization_writer,
        spawn_target=run_optimization_process,
        spawn_args=(job_id, config.model_dump(), results_dir_str),
    )

    # Register the canonical run id ("<env>/<timestamp>") now that the launch
    # succeeded, so results_path is populated from the start response onward
    # (the worker re-writes the same value in _reset_state_for_run). A failed
    # launch must leave the CREATED draft without a results_path.
    get_optimization_writer(job_id).set_last_results_path(results_dir.resolve())

    logger.info("Started optimization job %s (PID: %s)", job_id, pid)
    return read_job_status(tracker, job_id)
