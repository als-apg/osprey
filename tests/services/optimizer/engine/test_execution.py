"""Tests for services/execution.py — the ExecutionService.

Canonical home for ExecutionService, covering: _get_cancellation_token,
_measure_initial_state (return values, side-effect snapshots/metadata, and error
paths), state transitions (_handle_*), the CancellationError path,
_resolve_output_dir, and end-to-end start_optimization runs against the fake
environment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from osprey.services.optimizer.events import (
    EventType,
    IterationCompletedData,
    OptimizationEvent,
    OptimizationStartedData,
    PhaseCompletedData,
    PhaseStartedData,
)
from osprey.services.optimizer.models.config import OptimizationConfig
from osprey.services.optimizer.models.enums import AlgorithmType, JobStatus
from osprey.services.optimizer.models.exceptions import CancellationError, OptimizerError
from osprey.services.optimizer.models.exceptions import ConnectionError as OptimizerConnectionError
from osprey.services.optimizer.models.protocols import PVReading
from osprey.services.optimizer.models.types import Direction
from osprey.services.optimizer.optimization.types import OptimizationMetadata, OptimizationResult
from osprey.services.optimizer.services.draft_resolution import resolve_draft_config
from osprey.services.optimizer.services.execution import ExecutionService
from osprey.services.optimizer.services.plugin_registry import EnvironmentRegistry
from osprey.services.optimizer.utils.log_capture import LogCaptureHandler


def _config(**overrides: Any) -> OptimizationConfig:
    """Build an ``OptimizationConfig`` with this file's recurring defaults, overridden as needed."""
    defaults: dict[str, Any] = {
        "environment_name": "test_plugin",
        "control_target": "va",
        "objective_name": "charge_ltb",
        "algorithm": AlgorithmType.RANDOM,
    }
    defaults.update(overrides)
    return OptimizationConfig(**defaults)


def _raise(exc: BaseException):
    """Return a function that, when called with any args, raises ``exc``.

    Used to build lambdas that raise when monkeypatched into call sites.
    """

    def _f(*args: object, **kwargs: object) -> object:
        raise exc

    return _f


def _pingable_redis_mock() -> MagicMock:
    """MagicMock Redis client with ``ping()`` explicitly truthy."""
    mock = MagicMock()
    mock.ping.return_value = True
    return mock


def _seed_env_midpoints(env: object, n_active: int = 2) -> list[str]:
    """Seed every env variable to its bounds midpoint; return the first ``n_active`` names.

    Seeds *all* variables (not just the active subset) because the service
    reads the full variable set during initial-state capture — leaving any
    variable at its simulator default risks an out-of-bounds failure.
    """
    env_vars = env.variables  # type: ignore[attr-defined]
    with env:  # type: ignore[attr-defined]
        midpoints = {n: (env_vars[n].min_value + env_vars[n].max_value) / 2.0 for n in env_vars}
        env.set_variables(midpoints)  # type: ignore[attr-defined]
    return list(env_vars.keys())[:n_active]


@pytest.fixture
def exec_service(mock_redis: None, tmp_path: Path) -> ExecutionService:
    return ExecutionService(results_dir=tmp_path / "results", control_key="test-job")


# __init__ control_key requirement (regression guard for #41)


class TestControlKeyRequired:
    """ExecutionService raises ValueError when control_key is None or empty
    instead of silently using 'default_control'. (regression guard for #41)"""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"control_key": None},
            {"control_key": ""},
            {},
        ],
        ids=["none", "empty", "omitted"],
    )
    def test_missing_control_key_raises(
        self, mock_redis: None, tmp_path: Path, kwargs: dict[str, Any]
    ) -> None:
        """Passing control_key=None/'' or omitting it entirely raises ValueError."""
        with pytest.raises(ValueError, match="control_key is required"):
            ExecutionService(results_dir=tmp_path / "results", **kwargs)

    def test_explicit_key_is_used(self, mock_redis: None, tmp_path: Path) -> None:
        """Passing an explicit control_key uses it."""
        svc = ExecutionService(results_dir=tmp_path / "results", control_key="job-abc-123")
        assert svc._control_state_key == "job-abc-123"


class TestGetCancellationToken:
    def test_redis_failure_raises(
        self, exec_service: ExecutionService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Break Redis *after* init so _get_cancellation_token fails
        monkeypatch.setattr(
            "osprey.services.optimizer.utils.redis_client.get_redis_client",
            _raise(RuntimeError("no redis")),
        )
        with pytest.raises(RuntimeError, match="no redis"):
            exec_service._get_cancellation_token()

    def test_second_call_cached(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        svc_call_count = 0

        def counting_redis(binary: bool = False) -> MagicMock:
            nonlocal svc_call_count
            svc_call_count += 1
            return _pingable_redis_mock()

        monkeypatch.setattr(
            "osprey.services.optimizer.utils.redis_client.get_redis_client",
            counting_redis,
        )
        svc = ExecutionService(results_dir=tmp_path / "results", control_key="test-job")

        token = svc._get_cancellation_token()
        assert token is not None
        calls_after_first = svc_call_count

        # Several repeat invocations must all return the SAME cached token object
        # without touching Redis again — not just the second.
        for _ in range(3):
            assert svc._get_cancellation_token() is token
        assert svc_call_count == calls_after_first


class TestMeasureInitialState:
    def test_returns_vars_and_objective(
        self, exec_service: ExecutionService, mock_plugin_env
    ) -> None:
        var_names = _seed_env_midpoints(mock_plugin_env, n_active=2)
        config = _config(variables=var_names)

        result = exec_service._measure_initial_state(config)
        assert isinstance(result.initial_point, dict)
        # initial_point contains only the active (tuned) variables plus the objective
        expected_keys = set(var_names) | {"charge_ltb"}
        assert set(result.initial_point.keys()) == expected_keys
        # env_metadata records every env variable by name
        assert set(result.env_metadata["variables"]) == set(mock_plugin_env.variables)
        assert "charge_ltb" in result.env_metadata["observables"]

    def test_none_environment_raises(self, exec_service: ExecutionService) -> None:
        config = _config(environment_name=None)
        with pytest.raises(ValueError, match="environment_name"):
            exec_service._measure_initial_state(config)

    def test_failed_var_reads_raise_alsconnerror(
        self, exec_service: ExecutionService, mock_plugin_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the env returns None for any variable, _measure_initial_state aborts hard."""
        monkeypatch.setattr(
            mock_plugin_env,
            "get_variables",
            lambda names: {n: PVReading(value=None, connected=False) for n in names},
        )
        config = _config()
        with pytest.raises(
            OptimizerConnectionError, match="Cannot read variables required for optimization"
        ):
            exec_service._measure_initial_state(config)

    def test_objective_missing_raises(
        self, exec_service: ExecutionService, mock_plugin_env
    ) -> None:
        config = _config(objective_name="nonexistent_obj")
        with pytest.raises(ValueError, match="Objective 'nonexistent_obj' not found"):
            exec_service._measure_initial_state(config)

    def test_saves_initial_and_target_snapshots(
        self, exec_service: ExecutionService, mock_plugin_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_measure_initial_state saves a 'target' snapshot from observable metadata and
        an 'initial' snapshot holding only the tuned variables + objective."""
        save_snapshot_spy = MagicMock(wraps=exec_service._writer.save_snapshot)
        monkeypatch.setattr(exec_service._writer, "save_snapshot", save_snapshot_spy)

        # Use a small subset of variables the fake env initializes within bounds.
        config = _config(
            variables=["LN______MASTPH_AC00", "LN______AS2_PH_AC00"], environment_config={}
        )
        exec_service._reset_state_for_run(config)
        exec_service._measure_initial_state(config)

        snapshot_names = [c.args[0] for c in save_snapshot_spy.call_args_list]
        assert "initial" in snapshot_names
        assert "target" in snapshot_names

        target_call = next(c for c in save_snapshot_spy.call_args_list if c.args[0] == "target")
        target_data = target_call.args[1]
        assert "charge_gtl" in target_data
        assert "charge_ltb" in target_data
        assert target_data["charge_gtl"] == 5.0

        initial_call = next(c for c in save_snapshot_spy.call_args_list if c.args[0] == "initial")
        initial_data = initial_call.args[1]

        assert "LN______MASTPH_AC00" in initial_data
        assert "LN______AS2_PH_AC00" in initial_data
        assert "charge_ltb" in initial_data

        # No untuned environment variable leaks into the snapshot.
        assert config.variables is not None
        untuned = set(mock_plugin_env.variables) - set(config.variables)
        assert untuned, "fixture must expose untuned env variables for this test to be meaningful"
        assert not (untuned & set(initial_data))

    @pytest.mark.parametrize(
        ("objective_direction", "expected"),
        [
            # charge_ltb's observable metadata declares MAXIMIZE.
            pytest.param(None, "MAXIMIZE", id="unset_takes_observable_default"),
            pytest.param(Direction.MINIMIZE, "MINIMIZE", id="explicit_override_wins"),
        ],
    )
    def test_resolves_objective_direction_onto_config(
        self,
        exec_service: ExecutionService,
        mock_plugin_env,
        objective_direction: Direction | None,
        expected: str,
    ) -> None:
        """A config without an explicit direction gets the environment's default
        written back; an explicit override wins over the observable's default.

        The resolved direction must land on the config object itself so every
        config.json persisted afterwards records the direction the run actually
        used (the web's best-so-far line is omitted when it is null).
        """
        config = _config(
            objective_direction=objective_direction,
            variables=["LN______MASTPH_AC00"],
            environment_config={},
        )
        assert config.objective_direction == objective_direction

        exec_service._reset_state_for_run(config)
        exec_service._measure_initial_state(config)

        assert config.objective_direction == expected

    def test_saves_metadata(
        self, exec_service: ExecutionService, mock_plugin_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_measure_initial_state saves variable & observable metadata."""
        save_metadata_spy = MagicMock(wraps=exec_service._writer.save_metadata)
        monkeypatch.setattr(exec_service._writer, "save_metadata", save_metadata_spy)

        config = _config(variables=["LN______MASTPH_AC00"], environment_config={})
        exec_service._reset_state_for_run(config)
        exec_service._measure_initial_state(config)

        save_metadata_spy.assert_called_once()
        metadata = save_metadata_spy.call_args.args[0]
        assert "variables" in metadata
        assert "observables" in metadata
        assert "LN______MASTPH_AC00" in metadata["variables"]
        var_meta = metadata["variables"]["LN______MASTPH_AC00"]
        assert "description" in var_meta
        assert "units" in var_meta
        assert "step_size" in var_meta
        assert "charge_ltb" in metadata["observables"]
        obs_meta = metadata["observables"]["charge_ltb"]
        assert "direction" in obs_meta
        assert "target_value" in obs_meta


class TestStartOptimizationErrorPropagation:
    """start_optimization does domain work only and re-raises; JobLifecycle.__exit__
    is the single owner of terminal status (see test_job_tracker.py)."""

    def test_initial_state_failure_propagates_with_reason(
        self,
        exec_service: ExecutionService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Regression for #23: an initial-state failure must not be swallowed into a
        silent empty baseline — it propagates with the underlying reason preserved,
        so the wrapping JobLifecycle records OPTIMIZATION_ERROR with that message."""
        monkeypatch.setattr(
            "osprey.services.optimizer.services.plugin_registry.EnvironmentRegistry.load_environment",
            _raise(
                RuntimeError("Cannot read variables required for optimization: ['LN______MASTPH']")
            ),
        )
        exec_service._writer = MagicMock()

        config = _config(environment_name="broken")
        with pytest.raises(RuntimeError, match="Cannot read variables required for optimization"):
            exec_service.start_optimization(config)


class TestStartOptimizationCancellation:
    def test_cancellation_propagates(
        self,
        exec_service: ExecutionService,
        mock_plugin_env,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A direct CancellationError propagates so __exit__ records CANCELLED."""
        monkeypatch.setattr(
            "osprey.services.optimizer.services.execution.OptimizationEngine.run",
            MagicMock(side_effect=CancellationError("cancelled")),
        )
        # Seed env so _measure_initial_state succeeds and the run reaches engine.run.
        var_names = _seed_env_midpoints(mock_plugin_env, n_active=2)
        config = _config(variables=var_names, n_iterations=1)
        with pytest.raises(CancellationError):
            exec_service.start_optimization(config)

    def test_cancellation_saves_partial_results(
        self,
        exec_service: ExecutionService,
        mock_plugin_env,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A cancellation after output_dir is resolved saves partial results
        before propagating, so __exit__ records CANCELLED with data on disk."""
        monkeypatch.setattr(
            "osprey.services.optimizer.services.execution.OptimizationEngine.run",
            MagicMock(side_effect=CancellationError("Optimization was cancelled by user")),
        )
        save_mock = MagicMock(return_value=None)
        monkeypatch.setattr(exec_service, "_save_partial_results", save_mock)

        # _measure_initial_state raises on bad reads, so seed before the mocked session.run fires.
        var_names = _seed_env_midpoints(mock_plugin_env, n_active=2)

        config = _config(variables=var_names, n_iterations=1)
        with pytest.raises(CancellationError):
            exec_service.start_optimization(config)

        # Cancellation path (not the error path) saved partial results.
        save_mock.assert_called_once()
        cfg_arg, dir_arg = save_mock.call_args.args
        assert cfg_arg is config
        assert dir_arg == exec_service._results_dir / "test_plugin"

    def test_cancellation_before_output_dir_skips_partial_save(
        self,
        exec_service: ExecutionService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cancellation raised before _resolve_output_dir has set output_dir
        must not trigger _save_partial_results — there's nothing to save yet —
        and still propagates as a CancellationError."""
        monkeypatch.setattr(
            ExecutionService,
            "_measure_initial_state",
            _raise(CancellationError("early cancel")),
        )
        save_mock = MagicMock()
        monkeypatch.setattr(exec_service, "_save_partial_results", save_mock)

        config = _config(environment_name="env")
        with pytest.raises(CancellationError):
            exec_service.start_optimization(config)
        save_mock.assert_not_called()


class TestStartOptimizationOrchestration:
    """Verify ExecutionService composes OptimizationEngine and process_results correctly."""

    def test_runs_engine_then_processes_results(
        self,
        exec_service: ExecutionService,
        fake_initial_state: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The engine's result is handed to process_results, whose run directory is
        recorded as the results path (domain work) — but no terminal status: that is
        JobLifecycle.__exit__'s job (see test_job_tracker.py)."""
        results_dir = tmp_path / "run1"
        mock_result = MagicMock()
        mock_engine_cls = MagicMock()
        mock_engine_cls.return_value.run.return_value = mock_result
        monkeypatch.setattr(
            "osprey.services.optimizer.services.execution.OptimizationEngine", mock_engine_cls
        )

        mock_process = MagicMock(return_value=results_dir)
        monkeypatch.setattr(
            "osprey.services.optimizer.services.execution.process_results", mock_process
        )
        exec_service._writer = MagicMock()

        config = _config(n_iterations=1)
        result = exec_service.start_optimization(config)

        mock_engine_cls.return_value.run.assert_called_once()
        mock_process.assert_called_once()
        assert mock_process.call_args.args[0] is mock_result
        assert result == results_dir
        exec_service._writer.set_last_results_path.assert_called_with(results_dir)
        status_calls = [c.args[0] for c in exec_service._writer.set_status.call_args_list]
        assert JobStatus.COMPLETED not in status_calls

    def test_cancellation_before_engine_skips_engine_run(
        self,
        exec_service: ExecutionService,
        fake_initial_state: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_engine_cls = MagicMock()
        monkeypatch.setattr(
            "osprey.services.optimizer.services.execution.OptimizationEngine", mock_engine_cls
        )

        def fake_check(token: object) -> None:
            raise CancellationError("cancelled before engine")

        monkeypatch.setattr(
            "osprey.services.optimizer.services.execution.check_cancellation_token",
            fake_check,
        )

        config = _config(n_iterations=1)
        with pytest.raises(CancellationError):
            exec_service.start_optimization(config)

        mock_engine_cls.return_value.run.assert_not_called()


class TestResolveOutputDir:
    def test_none_env_name_raises(self, exec_service: ExecutionService) -> None:
        config = _config(environment_name=None, objective_name="obj")
        with pytest.raises(ValueError, match="environment_name"):
            exec_service._resolve_output_dir(config)


class TestSaveLogsToFile:
    def test_writes_logs_txt(self, tmp_path: Path) -> None:
        handler = LogCaptureHandler()
        handler.emit(logging.LogRecord("test", logging.INFO, "", 0, "hello world", (), None))
        ExecutionService._save_logs_to_file(tmp_path, handler)
        logs_file = tmp_path / "logs.txt"
        assert logs_file.exists()
        assert "hello world" in logs_file.read_text()

    @pytest.mark.parametrize(
        ("with_dir", "with_handler", "emit_record"),
        [
            pytest.param(False, True, True, id="results_dir_none"),
            pytest.param(True, False, False, id="handler_none"),
            pytest.param(True, True, False, id="logs_empty"),
        ],
    )
    def test_noop_writes_nothing(
        self, tmp_path: Path, with_dir: bool, with_handler: bool, emit_record: bool
    ) -> None:
        handler = None
        if with_handler:
            handler = LogCaptureHandler()
            if emit_record:
                handler.emit(logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None))

        # Must not raise in any of these shapes.
        ExecutionService._save_logs_to_file(tmp_path if with_dir else None, handler)
        assert not (tmp_path / "logs.txt").exists()

    def test_does_not_raise_on_write_failure(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        handler = LogCaptureHandler()
        handler.emit(logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None))
        nonexistent = tmp_path / "does" / "not" / "exist"
        # Log persistence is best-effort: the failure is logged as a warning, not raised.
        with caplog.at_level(
            logging.WARNING, logger="osprey.services.optimizer.services.execution"
        ):
            ExecutionService._save_logs_to_file(nonexistent, handler)
        assert any("Failed to save logs to file" in msg for msg in caplog.messages)


class TestCreateEventListener:
    """The event listener is the sole writer of optimization ``data`` to Redis.

    When the engine no longer writes data itself, this listener must reliably
    persist the DataFrame on every ITERATION_COMPLETED event.
    """

    @pytest.fixture
    def listener_writer_spies(
        self, exec_service: ExecutionService
    ) -> tuple[MagicMock, MagicMock, LogCaptureHandler]:
        """Install spies for set_data/save_logs and provide a real log handler."""
        set_data_spy = MagicMock()
        save_logs_spy = MagicMock()
        exec_service._writer.set_data = set_data_spy  # type: ignore[method-assign]
        exec_service._writer.save_logs = save_logs_spy  # type: ignore[method-assign]
        return set_data_spy, save_logs_spy, LogCaptureHandler()

    def test_iteration_completed_writes_dataframe_and_total_to_writer(
        self,
        exec_service: ExecutionService,
        listener_writer_spies: tuple[MagicMock, MagicMock, LogCaptureHandler],
    ) -> None:
        """ITERATION_COMPLETED persists the DataFrame and the resolved total
        (initial shots + iterations) so the progress bar's denominator counts
        initial samples, not just iterations.
        """
        set_data_spy, save_logs_spy, handler = listener_writer_spies
        set_total_spy = MagicMock()
        exec_service._writer.set_total_evaluations = set_total_spy  # type: ignore[method-assign]
        listener = exec_service._create_event_listener(on_event=None, log_handler=handler)

        df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        event = OptimizationEvent(
            type=EventType.ITERATION_COMPLETED,
            data=IterationCompletedData(
                phase="random", iteration=1, total=36, best_value=0.0, data=df
            ),
        )
        listener(event)

        set_data_spy.assert_called_once_with(df)
        set_total_spy.assert_called_once_with(36)
        save_logs_spy.assert_called_once()

    def test_optimization_started_persists_variable_bounds(
        self,
        exec_service: ExecutionService,
        listener_writer_spies: tuple[MagicMock, MagicMock, LogCaptureHandler],
    ) -> None:
        """The resolved bounds carried on OPTIMIZATION_STARTED are persisted so the
        live figure can normalize against the fixed search space.
        """
        *_, handler = listener_writer_spies
        set_bounds_spy = MagicMock()
        exec_service._writer.set_variable_bounds = set_bounds_spy  # type: ignore[method-assign]
        listener = exec_service._create_event_listener(on_event=None, log_handler=handler)

        bounds = {"x": [0.0, 10.0], "y": [-1.5, 1.5]}
        event = OptimizationEvent(
            type=EventType.OPTIMIZATION_STARTED,
            data=OptimizationStartedData(objective="loss", variable_bounds=bounds),
        )
        listener(event)

        set_bounds_spy.assert_called_once_with(bounds)

    @pytest.mark.parametrize(
        "event",
        [
            pytest.param(
                OptimizationEvent(
                    type=EventType.PHASE_STARTED, data=PhaseStartedData(phase="random")
                ),
                id="non_iteration_event",
            ),
        ],
    )
    def test_skips_set_data_without_dataframe_payload(
        self,
        exec_service: ExecutionService,
        listener_writer_spies: tuple[MagicMock, MagicMock, LogCaptureHandler],
        event: OptimizationEvent,
    ) -> None:
        """Only an ITERATION_COMPLETED event writes data; logs are saved on
        every event either way."""
        set_data_spy, save_logs_spy, handler = listener_writer_spies
        listener = exec_service._create_event_listener(on_event=None, log_handler=handler)

        listener(event)

        set_data_spy.assert_not_called()
        save_logs_spy.assert_called_once()

    def test_forwards_event_to_on_event_callback(
        self,
        exec_service: ExecutionService,
        listener_writer_spies: tuple[MagicMock, MagicMock, LogCaptureHandler],
    ) -> None:
        *_, handler = listener_writer_spies
        on_event_spy = MagicMock()
        listener = exec_service._create_event_listener(on_event=on_event_spy, log_handler=handler)

        event = OptimizationEvent(
            type=EventType.PHASE_STARTED,
            data=PhaseStartedData(phase="random"),
        )
        listener(event)

        on_event_spy.assert_called_once_with(event)

    def test_bus_events_keep_streaming_captured_logs(
        self,
        exec_service: ExecutionService,
        listener_writer_spies: tuple[MagicMock, MagicMock, LogCaptureHandler],
    ) -> None:
        """The live log stream survives every event emitted through the bus.

        The listener takes its handler from the closure, so it cannot fire
        without one. Previously it read a nullable attribute and raised on
        None — and ``EventBus.emit`` swallows subscriber exceptions, so the
        job's log stream went dark with only a log line to show for it.
        """
        _, save_logs_spy, handler = listener_writer_spies
        handler.emit(logging.LogRecord("run", logging.INFO, "", 0, "iteration 1 done", (), None))

        bus = exec_service._setup_event_handling(None, handler)
        for phase_event in (
            OptimizationEvent(type=EventType.PHASE_STARTED, data=PhaseStartedData(phase="random")),
            OptimizationEvent(
                type=EventType.PHASE_COMPLETED, data=PhaseCompletedData(phase="random")
            ),
        ):
            bus.emit(phase_event)

        assert save_logs_spy.call_count == 2
        assert all("iteration 1 done" in call.args[0] for call in save_logs_spy.call_args_list)


class TestCreateApplyFn:
    """`_create_apply_fn` returns a callable that writes the best point back to the env."""

    def test_apply_fn_writes_variables_to_environment(self, exec_service: ExecutionService) -> None:
        env = MagicMock()
        env.__enter__.return_value = env
        env.__exit__.return_value = None
        env.variables = {"var_a": MagicMock(), "var_b": MagicMock()}

        config = _config(environment_name="any")
        with patch.object(EnvironmentRegistry, "load_environment", return_value=env):
            apply_fn = exec_service._create_apply_fn(config)
            result = OptimizationResult(
                data=pl.DataFrame({"var_a": [1.0]}),
                best_point={"var_a": 1.5, "var_b": -0.25, "extra_key": 9.9},
                metadata=OptimizationMetadata(config=config),
            )
            applied = apply_fn(result)

        assert applied == result.best_point
        # Only variables present in env.variables should be written.
        env.set_variables.assert_called_once_with({"var_a": 1.5, "var_b": -0.25})


class TestSpecificRunDir:
    def test_resolved_as_output_dir_and_recorded_on_reset(
        self, mock_redis: None, tmp_path: Path
    ) -> None:
        """A pre-initialized run dir is the resolved output dir, and _reset_state_for_run
        records it as the results path."""
        run_dir = tmp_path / "specific"
        run_dir.mkdir()
        svc = ExecutionService(
            results_dir=tmp_path / "results", control_key="job-x", specific_run_dir=run_dir
        )
        config = _config(environment_name="e")

        assert svc._resolve_output_dir(config) == run_dir.resolve()

        svc._writer = MagicMock()
        svc._reset_state_for_run(config)
        svc._writer.set_last_results_path.assert_called_once_with(run_dir.resolve())


class TestMeasureInitialStateNoTargetValues:
    def test_no_target_values_skips_target_snapshot(
        self,
        exec_service: ExecutionService,
        mock_plugin_env,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When no observable defines a target_value, the 'target' snapshot is not saved."""
        var_names = _seed_env_midpoints(mock_plugin_env, n_active=1)

        observables_no_targets = {
            name: meta.model_copy(update={"target_value": None})
            for name, meta in mock_plugin_env.observables.items()
        }
        monkeypatch.setattr(
            type(mock_plugin_env),
            "observables",
            property(lambda self: observables_no_targets),
        )

        save_snapshot_spy = MagicMock(wraps=exec_service._writer.save_snapshot)
        monkeypatch.setattr(exec_service._writer, "save_snapshot", save_snapshot_spy)

        config = _config(variables=var_names)
        exec_service._reset_state_for_run(config)
        exec_service._measure_initial_state(config)

        snapshot_names = [c.args[0] for c in save_snapshot_spy.call_args_list]
        assert "target" not in snapshot_names


class TestSavePartialResultsObjectiveMissing:
    def test_objective_not_in_data_columns(
        self,
        exec_service: ExecutionService,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """When the objective column is absent from accumulated data, best_point is empty."""
        partial = pl.DataFrame({"var_a": [0.1, 0.2]})  # no objective column
        exec_service._writer = MagicMock()
        exec_service._writer.get_data.return_value = partial

        save_mock = MagicMock(return_value=tmp_path / "saved")
        monkeypatch.setattr(
            "osprey.services.optimizer.services.execution.save_optimization_results", save_mock
        )

        config = _config(environment_name="env")
        result = exec_service._save_partial_results(config, tmp_path)

        assert result == tmp_path / "saved"
        assert save_mock.call_args.args[1] == {}


class TestSavePartialResultsBestPoint:
    """The cancellation partial-save must honor objective direction and skip
    non-finite objective rows when picking the best point — same contract as the
    runner's ``_best_finite_row``."""

    def _run(
        self,
        exec_service: ExecutionService,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        data: pl.DataFrame,
        direction: Direction,
    ) -> dict[str, float]:
        exec_service._writer = MagicMock()
        exec_service._writer.get_data.return_value = data

        save_mock = MagicMock(return_value=tmp_path / "saved")
        monkeypatch.setattr(
            "osprey.services.optimizer.services.execution.save_optimization_results", save_mock
        )

        config = _config(environment_name="env", variables=["var_a"], objective_direction=direction)
        exec_service._save_partial_results(config, tmp_path)
        best_point: dict[str, float] = save_mock.call_args.args[1]
        return best_point

    @pytest.mark.parametrize(
        ("direction", "expected_var_a"),
        [("MINIMIZE", 0.2), ("MAXIMIZE", 0.3)],
    )
    def test_picks_best_objective_for_direction(
        self,
        exec_service: ExecutionService,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        direction: Direction,
        expected_var_a: float,
    ) -> None:
        # objectives [5.0, 1.0, 9.0]: MINIMIZE picks the 1.0 row, MAXIMIZE the 9.0 row.
        data = pl.DataFrame({"var_a": [0.1, 0.2, 0.3], "charge_ltb": [5.0, 1.0, 9.0]})
        best_point = self._run(exec_service, monkeypatch, tmp_path, data, direction)
        # The best point holds the selected variables only — never the objective column.
        assert best_point == {"var_a": expected_var_a}


def test_save_partial_results_unresolved_direction_raises(
    exec_service: ExecutionService, tmp_path: Path
) -> None:
    """objective present in data but direction None -> raise ValueError."""
    data = pl.DataFrame({"var_a": [0.1], "charge_ltb": [1.0]})
    exec_service._writer = MagicMock()
    exec_service._writer.get_data.return_value = data

    config = _config(environment_name="env", objective_direction=None)
    with pytest.raises(ValueError, match="objective_direction unresolved"):
        exec_service._save_partial_results(config, tmp_path)


def test_save_partial_results_uses_recorded_variable_names_when_config_empty(
    exec_service: ExecutionService,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """config.variables empty -> use the active names the writer recorded in
    ``_measure_initial_state``, never "every non-objective column": bookkeeping
    columns like ``timestamp`` must stay out of best_point (#39, #106)."""
    data = pl.DataFrame({"var_a": [0.1, 0.2], "charge_ltb": [5.0, 1.0], "timestamp": [1.0, 2.0]})
    exec_service._writer = MagicMock()
    exec_service._writer.get_data.return_value = data
    exec_service._writer.get_variable_names.return_value = ["var_a"]

    save_mock = MagicMock(return_value=tmp_path / "saved")
    monkeypatch.setattr(
        "osprey.services.optimizer.services.execution.save_optimization_results", save_mock
    )

    config = _config(environment_name="env", objective_direction=Direction.MINIMIZE)
    config.variables = []
    exec_service._save_partial_results(config, tmp_path)

    best_point = save_mock.call_args.args[1]
    assert best_point == {"var_a": 0.2}  # row with objective 1.0


def test_save_partial_results_unrecorded_variable_names_raises(
    exec_service: ExecutionService, tmp_path: Path
) -> None:
    """config.variables empty and no names recorded in state -> raise, never guess."""
    data = pl.DataFrame({"var_a": [0.1], "charge_ltb": [1.0]})
    exec_service._writer = MagicMock()
    exec_service._writer.get_data.return_value = data
    exec_service._writer.get_variable_names.return_value = None

    config = _config(environment_name="env", objective_direction=Direction.MINIMIZE)
    config.variables = []
    with pytest.raises(ValueError, match="variable_names not recorded"):
        exec_service._save_partial_results(config, tmp_path)


def test_start_optimization_propagates_optimizer_error_verbatim(
    exec_service: ExecutionService, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A backend failure arrives already translated (OptimizerError) and the
    service re-raises it verbatim — no re-wrapping, message intact."""
    # Holding the last point isolates propagation from the default restore.
    config = _config(environment_name="env", on_abort="hold")

    monkeypatch.setattr(exec_service, "_resolve_output_dir", lambda c: tmp_path)
    monkeypatch.setattr(exec_service, "_get_cancellation_token", lambda: MagicMock())
    monkeypatch.setattr(exec_service, "_measure_initial_state", lambda c: MagicMock())
    monkeypatch.setattr(
        exec_service, "_setup_event_handling", lambda on_event, log_handler: MagicMock()
    )
    monkeypatch.setattr(exec_service, "_create_apply_fn", lambda c: MagicMock())

    engine = MagicMock()
    engine.run.side_effect = OptimizerError("Optimizer backend failed: boom: epics down")
    monkeypatch.setattr(
        "osprey.services.optimizer.services.execution.OptimizationEngine",
        MagicMock(return_value=engine),
    )

    with pytest.raises(OptimizerError, match="epics down"):
        exec_service.start_optimization(config)


class TestStartOptimizationSpecificRunDir:
    def test_specific_run_dir_calls_save_early_results(
        self,
        mock_redis: None,
        mock_plugin_env,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "specific-run"
        run_dir.mkdir()
        svc = ExecutionService(
            results_dir=tmp_path / "results",
            control_key="job-z",
            specific_run_dir=run_dir,
        )

        save_early_mock = MagicMock()
        monkeypatch.setattr(
            "osprey.services.optimizer.services.execution.save_early_results", save_early_mock
        )
        # Short-circuit the heavy session/run path to isolate save_early_results.
        # The RuntimeError propagates — the service does not swallow it.
        monkeypatch.setattr(
            "osprey.services.optimizer.services.execution.OptimizationEngine.run",
            MagicMock(side_effect=RuntimeError("stop here")),
        )

        var_names = _seed_env_midpoints(mock_plugin_env, n_active=1)

        config = _config(variables=var_names, n_iterations=1)
        with pytest.raises(RuntimeError, match="stop here"):
            svc.start_optimization(config)
        save_early_mock.assert_called_once()
        assert save_early_mock.call_args.args[0] == run_dir.resolve()


class TestExecutionEndToEnd:
    """start_optimization exercised end-to-end against the fake env (light mocking)."""

    def test_execution_service_runs_end_to_end(
        self, exec_service: ExecutionService, mock_plugin_env
    ) -> None:
        """ExecutionService runs a minimal optimization end-to-end and writes results."""
        # Variables whose bounds include 0.0 (the fake env's setpoints default to 0.0).
        config = _config(
            n_iterations=3,
            environment_config={},
            variables=["LN______MASTPH_AC00", "LN______AS2_PH_AC00"],
        )
        with mock_plugin_env:
            resolve_draft_config(config, mock_plugin_env)

        result_path = exec_service.start_optimization(
            config.model_copy(update={"on_complete": "hold"})
        )

        assert result_path is not None
        assert result_path.exists()
        assert (result_path / "config.json").exists()
        # 1 initial sample (RANDOM default) + 3 iterations, with both tuned
        # variables and the objective recorded per row.
        data = pl.read_parquet(result_path / "results.parquet")
        assert data.height == 4
        assert {"LN______MASTPH_AC00", "LN______AS2_PH_AC00", "charge_ltb"} <= set(data.columns)
