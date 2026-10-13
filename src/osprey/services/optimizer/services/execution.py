"""Service for running optimization execution."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.services.optimizer.events import (
    EventBus,
    EventCallback,
    EventType,
    IterationCompletedData,
    OptimizationEvent,
    OptimizationStartedData,
)
from osprey.services.optimizer.models.config import OptimizationConfig
from osprey.services.optimizer.models.enums import JobStatus
from osprey.services.optimizer.models.exceptions import CancellationError, RestoreFailedError
from osprey.services.optimizer.models.types import BestPoint

from ..optimization.best_point import build_best_point
from ..optimization.evaluation import Evaluator
from ..optimization.state import OptimizationStateWriter, get_optimization_writer
from ..optimization.types import OptimizationResult
from ..results import ResultMetadata, save_early_results, save_optimization_results
from ..utils.cancellation import CancellationToken, get_cancellation_token
from ..utils.control import check_cancellation_token
from ..utils.log_capture import LogCaptureHandler, capture_logs
from ..utils.redis_client import ping_redis
from .orchestration import (
    OptimizationEngine,
    RunnerServices,
    apply_best_point,
    read_required_variables,
    require_objective,
    resolve_environment,
)
from .plugin_registry import EnvironmentRegistry
from .result_pipeline import ParquetResultWriter, process_results

logger = logging.getLogger(__name__)


@dataclass
class InitialStateResult:
    """Result of measuring the initial machine state."""

    initial_point: dict[str, float]
    env_metadata: dict[str, Any]
    start_values: dict[str, float]
    """The tuned variables' values before the run moved anything: what a restore writes."""


class ExecutionService:
    """Service for running optimization execution.

    This class orchestrates optimization execution, state management,
    and results handling in a headless environment (Worker).
    """

    def __init__(
        self,
        results_dir: Path | None = None,
        control_key: str | None = None,
        specific_run_dir: Path | None = None,
    ) -> None:
        """Initialize the execution service.

        Args:
            results_dir: Directory for saving results. If None, defaults to "results".
            control_key: Redis key for pause/resume/cancel coordination.
                Must be a job-scoped key (e.g. the job_id). If None or empty,
                raises ValueError.
            specific_run_dir: If provided, this specific directory is used for output instead of
                              creating a new timestamped one. This bypasses the default
                              structure creation.
        """
        base_dir = results_dir if results_dir is not None else Path("results")
        self._results_dir = base_dir.resolve()
        self._results_dir.mkdir(parents=True, exist_ok=True)

        self._specific_run_dir = specific_run_dir.resolve() if specific_run_dir else None

        if not control_key:
            raise ValueError("control_key is required for ExecutionService — pass a job-scoped key")
        self._control_state_key = control_key

        self._cancellation_token: CancellationToken | None = None
        self._poll_interval = 0.2

        self._init_state_persistence()

    def _init_state_persistence(self) -> None:
        """Initialize Redis client + optimization state writer for this job.

        Raises if the connection cannot be established — callers must not
        silently degrade to a no-Redis mode.
        """
        ping_redis()
        self._writer: OptimizationStateWriter = get_optimization_writer(self._control_state_key)
        logger.info("Connected to Redis for optimization control state")

    def _create_apply_fn(
        self,
        config: OptimizationConfig,
    ) -> Callable[[OptimizationResult], BestPoint]:
        """Create function to apply best point to machine using EnvironmentRegistry."""

        def apply_best_point_fn(result: OptimizationResult) -> BestPoint:
            """Apply the best optimization result to the environment."""
            logger.info("Applying best point to machine...")
            registry = EnvironmentRegistry()
            _, env = resolve_environment(config, registry.load_environment)

            best_point = apply_best_point(env, result)

            logger.info("Best point applied successfully")
            return best_point

        return apply_best_point_fn

    def _restore_initial_values(
        self, config: OptimizationConfig, start_values: dict[str, float], reason: str
    ) -> None:
        """Write the run's starting values back to the machine it drove.

        Only the variables this run tuned, through the same checked connector
        every other write took, and not cancellable: an operator who cancels
        asked for the run to stop, not for the machine to be left mid-search.
        A readonly run wrote nothing, so there is nothing to undo.
        """
        if config.execution_mode == "readonly":
            logger.info("Readonly run: nothing to restore after %s", reason)
            return
        if not start_values:
            logger.info("No variables were tuned; nothing to restore after %s", reason)
            return
        logger.info(
            "Restoring %d starting value(s) after %s: %s", len(start_values), reason, start_values
        )
        _, env = resolve_environment(config, EnvironmentRegistry().load_environment)
        with env:
            env.set_variables(start_values)
        logger.info("Starting values restored")

    def _restore_or_raise(
        self,
        config: OptimizationConfig,
        start_values: dict[str, float],
        reason: str,
        cause: BaseException | None,
    ) -> None:
        """Restore, turning a failed restore into the job's outcome."""
        try:
            self._restore_initial_values(config, start_values, reason)
        except Exception as restore_exc:
            logger.exception("Restoring the starting values after %s failed", reason)
            ended = "finished" if cause is None else f"ended by {reason}"
            raise RestoreFailedError(
                f"The run {ended}, and writing its starting values back failed: {restore_exc}. "
                "Some, all or none of them were written; check the machine before the next run."
            ) from (cause or restore_exc)

    def _get_cancellation_token(self) -> CancellationToken:
        """Get or create cancellation token for run control.

        Raises if Redis is unavailable — cross-worker cancel/pause is
        required for correctness (without it, user cancel/pause requests
        silently no-op). Callers must handle the failure; do not fall
        back to a no-token mode.
        """
        if self._cancellation_token is None:
            self._cancellation_token = get_cancellation_token(
                self._control_state_key,
                poll_interval=self._poll_interval,
            )
        return self._cancellation_token

    def _reset_state_for_run(self, config: OptimizationConfig) -> None:
        """Reset state for new optimization run."""
        self._writer.set_status(JobStatus.RUNNING)
        self._writer.set_error_message(None)

        if self._specific_run_dir:
            self._writer.set_last_results_path(self._specific_run_dir)
        else:
            self._writer.set_last_results_path(None)

        self._writer.set_data(None)
        self._writer.set_objective_name(config.objective_name)
        self._writer.set_environment_name(config.environment_name)
        self._writer.set_variable_names(None)  # Will be set in _measure_initial_state

    def _measure_initial_state(self, config: OptimizationConfig) -> InitialStateResult:
        """Capture the initial state of machine variables and the objective value.

        Raises:
            ValueError: If ``config.environment_name`` is missing, or the
                configured objective is not present in the environment.
            ALSConnectionError: If required variable PVs cannot be read.
            KeyError: If the environment plugin name is not registered.
            Exception: Any other failure raised by environment loading,
                ``env.get_variables``, or the evaluator.

        There is no safe way to proceed without an initial state, so callers
        must translate failures into an ``OPTIMIZATION_ERROR`` status rather
        than running with an empty baseline.
        """
        environment_registry = EnvironmentRegistry()
        environment_name, env = resolve_environment(config, environment_registry.load_environment)

        with env:
            active_variables = config.variables or list(env.variables.keys())
            self._writer.set_variable_names(active_variables)

            all_variable_names = list(env.variables.keys())
            all_initial_values = read_required_variables(env, all_variable_names)
            active_initial_values = {
                k: v for k, v in all_initial_values.items() if k in active_variables
            }

            objective_name = require_objective(env, config.objective_name, environment_name)

            objective_meta = env.observables[objective_name]
            direction = config.objective_direction or objective_meta.direction
            # Write the resolved direction back so every config.json persisted
            # from here on records the direction the run actually used.
            config.objective_direction = direction
            self._writer.set_objective_direction(direction)

            evaluator = Evaluator(env, objective_name)

            # Evaluator re-sets the variables we just read — same values, so safe.
            objective_result = evaluator(active_initial_values)

            initial_point = {**active_initial_values, **objective_result}

        self._writer.save_snapshot("initial", initial_point)

        # feeds the enriched result summary
        var_fields = {"description", "units", "step_size"}
        var_meta = {
            name: env.variables[name].model_dump(include=var_fields)
            for name in all_variable_names
            if name in env.variables
        }
        obs_fields = {"description", "units", "direction", "target_value"}
        obs_meta = {
            name: meta.model_dump(include=obs_fields) for name, meta in env.observables.items()
        }
        env_metadata: dict[str, Any] = {"variables": var_meta, "observables": obs_meta}
        self._writer.save_metadata(env_metadata)

        # only observables define targets
        target_dict = {
            name: meta.target_value
            for name, meta in env.observables.items()
            if meta.target_value is not None
        }
        if target_dict:
            self._writer.save_snapshot("target", target_dict)

        initial_objective = initial_point.get(objective_name)
        logger.info("Captured initial state with objective: %s", initial_objective)
        return InitialStateResult(
            initial_point=initial_point,
            env_metadata=env_metadata,
            start_values=active_initial_values,
        )

    def _setup_event_handling(
        self, on_event: EventCallback | None, log_handler: LogCaptureHandler
    ) -> EventBus:
        """Configure the event bus and subscribe default listeners.

        Takes the live ``log_handler`` so the listener can only ever be built —
        and subscribed — once log capture is running.
        """
        event_bus = EventBus()

        on_event_listener = self._create_event_listener(on_event, log_handler)

        event_bus.subscribe_all(on_event_listener)

        return event_bus

    def _resolve_output_dir(self, config: OptimizationConfig) -> Path:
        """Return the directory results should be written to for this run."""
        if self._specific_run_dir:
            logger.info("Using pre-initialized run directory: %s", self._specific_run_dir)
            return self._specific_run_dir

        if not config.environment_name:
            raise ValueError("config.environment_name is required.")

        output_dir = self._results_dir / config.environment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _save_partial_results(self, config: OptimizationConfig, output_dir: Path) -> Path | None:
        """Save whatever data has been accumulated so far to disk.

        Called on cancellation so that partial results are not lost.
        """
        data = self._writer.get_data()
        if data is None or data.is_empty():
            logger.info("No partial data to save")
            return None

        objective = config.objective_name
        if objective is None or objective not in data.columns:
            logger.warning(
                "Objective '%s' not in partial data columns; skipping best_point", objective
            )
            best_point: BestPoint = {}
        else:
            direction = config.objective_direction
            if direction is None:
                # _measure_initial_state resolves and writes this back before the
                # first iteration; any partial data implies it has already run.
                raise ValueError(
                    "objective_direction unresolved; cannot select a partial best point"
                )
            if config.variables:
                variable_names = config.variables
            else:
                # The active names were recorded by _measure_initial_state; never
                # infer them from data columns, which also carry bookkeeping/
                # constraint/fixed-feature columns (#106).
                recorded = self._writer.get_variable_names()
                if recorded is None:
                    raise ValueError(
                        "variable_names not recorded in state; cannot select a partial best point"
                    )
                variable_names = recorded
            best_point = build_best_point(data, objective, variable_names, direction=direction)

        run_dir = save_optimization_results(
            data,
            best_point,
            config,
            output_dir=output_dir,
            metadata=ResultMetadata(job_id=self._control_state_key),
        )
        logger.info("Saved partial results (%d rows) to %s", len(data), run_dir)
        self._writer.set_last_results_path(run_dir)
        return run_dir

    def _cleanup_after_run(self) -> None:
        """Cleanup after optimization run."""
        self._cancellation_token = None

    @staticmethod
    def _save_logs_to_file(results_dir: Path | None, log_handler: LogCaptureHandler | None) -> None:
        """Save captured logs to a file in the results directory.

        Args:
            results_dir: Directory to save logs to, or None to skip.
            log_handler: Log capture handler with buffered messages, or None to skip.
        """
        if results_dir is None or log_handler is None:
            return
        try:
            logs = log_handler.get_logs()
            if logs:
                (results_dir / "logs.txt").write_text(logs)
                logger.info("  - Logs: %s", results_dir / "logs.txt")
        except Exception as e:
            logger.warning("Failed to save logs to file: %s", e)

    def start_optimization(
        self,
        config: OptimizationConfig,
        *,
        on_event: EventCallback | None = None,
    ) -> Path | None:
        """Start and run optimization.

        Does domain work only (results path, partial-results-on-cancel, what the
        machine is left at) and lets the outcome propagate: returns the results
        dir on success, re-raises on cancel/error. The wrapping
        ``JobLifecycle.__exit__`` is the single owner of terminal status — this
        method does not write it.

        What the machine is left at is the config's: ``on_complete`` when the run
        finishes, ``on_abort`` when it is cancelled or fails after its starting
        values were measured (before that, nothing was moved).

        Returns:
            Path to the results directory on success.

        Raises:
            CancellationError: if the run was cancelled; partial results are saved
                first so the lifecycle classifies it as CANCELLED, not ERROR.
            RestoreFailedError: if writing the starting values back failed; the
                original cancel or failure is its ``__cause__``.
            Exception: any other failure, propagated verbatim. Optimizer-backend
                failures already arrive as ``OptimizerError`` from the adapter.
        """
        self._reset_state_for_run(config)

        output_dir: Path | None = None
        results_dir: Path | None = None
        log_handler: LogCaptureHandler | None = None
        start_values: dict[str, float] | None = None
        try:
            with capture_logs() as log_handler:
                token = self._get_cancellation_token()
                # A cancel can land while the worker is still spawning (~15 s of
                # imports), and POST /jobs/{id}/cancel accepts it from any
                # non-terminal state. reset() clears the flag, so honour it first
                # — otherwise the request is acknowledged, silently discarded, and
                # the run proceeds on hardware. Cancel only: wait_if_paused() would
                # block here, and pause is CAS-gated to RUNNING upstream.
                token.check_cancellation()
                token.reset()

                initial_state = self._measure_initial_state(config)
                start_values = initial_state.start_values

                if self._specific_run_dir:
                    save_early_results(
                        self._specific_run_dir,
                        initial_state.initial_point,
                        initial_state.env_metadata,
                        config=config,
                    )

                event_bus = self._setup_event_handling(on_event, log_handler)

                output_dir = self._resolve_output_dir(config)
                services = RunnerServices(cancellation_token=token, event_bus=event_bus)

                check_cancellation_token(token)

                engine = OptimizationEngine(
                    config,
                    environment_factory=EnvironmentRegistry().load_environment,
                    services=services,
                    job_id=self._control_state_key,
                    initial_point=initial_state.initial_point,
                )
                result = engine.run()

                results_dir = process_results(
                    result,
                    writer=ParquetResultWriter(),
                    output_dir=output_dir,
                    apply_fn=self._create_apply_fn(config),
                    apply_best_point=config.on_complete == "apply_best",
                    check_cancellation=token.check_cancellation,
                    wait_if_paused=token.wait_if_paused,
                )
                if config.on_complete == "restore":
                    self._restore_or_raise(config, start_values, "completion", None)

                self._writer.set_last_results_path(results_dir)

        except CancellationError as exc:
            if output_dir:
                results_dir = self._save_partial_results(config, output_dir)
            if start_values is not None and config.on_abort == "restore":
                self._restore_or_raise(config, start_values, "cancellation", exc)
            raise
        except RestoreFailedError:
            raise
        except Exception as exc:
            if start_values is not None and config.on_abort == "restore":
                self._restore_or_raise(config, start_values, f"failure ({exc})", exc)
            raise
        finally:
            self._save_logs_to_file(results_dir, log_handler)
            self._cleanup_after_run()

        return results_dir

    def _create_event_listener(
        self, on_event: EventCallback | None, log_handler: LogCaptureHandler
    ) -> Callable[[OptimizationEvent], None]:
        """Create the event listener that persists data/logs and forwards events.

        ``log_handler`` is captured in the closure rather than read off the
        service: ``EventBus.emit`` isolates subscribers by swallowing their
        exceptions, so a listener that could fail on missing state would take
        the job's live log stream down silently.
        """

        def on_event_listener(event: OptimizationEvent) -> None:
            """Handle optimization events and update state."""
            if event.type == EventType.ITERATION_COMPLETED and isinstance(
                event.data, IterationCompletedData
            ):
                self._writer.set_data(event.data.data)
                self._writer.set_total_evaluations(event.data.total)

            if (
                event.type == EventType.OPTIMIZATION_STARTED
                and isinstance(event.data, OptimizationStartedData)
                and event.data.variable_bounds is not None
            ):
                self._writer.set_variable_bounds(event.data.variable_bounds)

            self._writer.save_logs(log_handler.get_logs())
            if on_event:
                on_event(event)

        return on_event_listener
