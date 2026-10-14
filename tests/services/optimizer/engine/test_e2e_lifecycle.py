"""End-to-end lifecycle tests for optimization pause/resume/cancel.

Exercises the **real** ExecutionService → OptimizationEngine → Xopt path
against the mock simulator environment, with only ``get_redis_client``
patched to use the embedded test Redis.

Covers four lifecycle scenarios: normal completion, pause→resume→complete,
cancel while running, and cancel while paused.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import polars as pl
import pytest
from redis import Redis

from osprey.services.optimizer.api.job_tracker import JobLifecycle
from osprey.services.optimizer.events import EventType, OptimizationEvent
from osprey.services.optimizer.models.config import OptimizationConfig
from osprey.services.optimizer.models.enums import JobStatus
from osprey.services.optimizer.models.exceptions import CancellationError
from osprey.services.optimizer.optimization.state import get_optimization_reader
from osprey.services.optimizer.results import load_results
from osprey.services.optimizer.services.draft_resolution import resolve_draft_config
from osprey.services.optimizer.services.execution import ExecutionService
from osprey.services.optimizer.services.plugin_registry import EnvironmentRegistry
from osprey.services.optimizer.utils.cancellation import CancellationToken
from osprey.services.optimizer.utils.job_keys import cancellation_key

from .conftest import wait_until

EXPECTED_ARTIFACTS = {"config.json", "results.parquet", "best_point.json", "metadata.json"}

N_ITERATIONS = 3
# With RANDOM algorithm (default): 1 initial sample + N_ITERATIONS steps.
# Total evaluations = N_ITERATIONS + 1 = 4.


def _make_config(n_iterations: int = N_ITERATIONS) -> OptimizationConfig:
    """Build a config and resolve it against the fake plugin env (``mock_plugin_env``
    must already be active), mirroring what the API/CLI does before a worker sees it."""
    config = OptimizationConfig(
        n_iterations=n_iterations,
        environment_config={},
        objective_name="charge_ltb",
        environment_name="test_plugin",
        control_target="va",
        variables=["LN______MASTPH_AC00", "LN______AS2_PH_AC00"],
    )
    env = EnvironmentRegistry().load_environment("test_plugin", config.environment_config or {})
    with env:
        resolve_draft_config(config, env)
    return config


def _make_redis_factory(redis_url: str) -> Callable[..., Redis[Any]]:
    def factory(binary: bool = False) -> Redis[Any]:
        if binary:
            return Redis.from_url(redis_url, decode_responses=False)
        return Redis.from_url(redis_url, decode_responses=True)

    return factory


@pytest.fixture
def lifecycle_worker(redis_server: str, tmp_path: Path) -> Generator[Any, None, None]:
    """Redis-backed control token + threaded ``_run_service`` runner.

    Collapses the per-test plumbing (Redis client, cancellation token, result
    holder, daemon worker with traceback capture, key cleanup) shared by the
    pause/resume and cancel lifecycle tests.
    """
    redis_client = Redis.from_url(redis_server, decode_responses=True)
    keys: list[str] = []
    threads: list[threading.Thread] = []

    class _LifecycleWorker:
        def __init__(self) -> None:
            self.result_holder: dict[str, Any] = {"path": None, "error": None}

        def make_token(self, control_key: str) -> CancellationToken:
            keys.append(cancellation_key(control_key))
            return CancellationToken(
                redis_client=redis_client,
                redis_key=cancellation_key(control_key),
                poll_interval=0.05,
            )

        def start(
            self, control_key: str, on_event: Callable[[OptimizationEvent], None]
        ) -> threading.Thread:
            def worker() -> None:
                try:
                    self.result_holder["path"] = _run_service(
                        redis_server, control_key, tmp_path, on_event=on_event
                    )
                except Exception:
                    import traceback

                    self.result_holder["error"] = traceback.format_exc()

            thread = threading.Thread(target=worker, daemon=True)
            threads.append(thread)
            thread.start()
            return thread

    yield _LifecycleWorker()

    for thread in threads:
        thread.join(timeout=1.0)
    for key in keys:
        redis_client.delete(key)
    redis_client.close()


def _make_iteration_signal() -> tuple[threading.Event, Callable[[OptimizationEvent], None]]:
    """Return ``(event, callback)`` — *event* is set on the first ITERATION_COMPLETED.

    The callback is passed as ``on_event`` to ``start_optimization``.
    It fires *after* the event listener has already called ``set_data``, so
    by the time the test thread wakes up, data is persisted in Redis.
    """
    first_iteration_done = threading.Event()

    def on_event(event: OptimizationEvent) -> None:
        if event.type == EventType.ITERATION_COMPLETED and not first_iteration_done.is_set():
            first_iteration_done.set()

    return first_iteration_done, on_event


def _make_first_iteration_action(
    action: Callable[[], None],
) -> tuple[threading.Event, Callable[[OptimizationEvent], None]]:
    """Return ``(event, callback)`` that runs *action* once, on the first iteration.

    Events are dispatched synchronously in the worker thread, so running the
    action (e.g. ``token.pause`` or ``token.cancel``) from inside the callback
    guarantees the worker observes its effect at the *next* control checkpoint —
    the top of the iteration loop, before any further ``X.step()``. The run is
    therefore deterministically stopped mid-flight with only partial data
    persisted, rather than racing to completion (and, for pause, blocking at the
    post-run result-pipeline checkpoint where all rows are already saved).
    """
    first_iteration_done = threading.Event()

    def on_event(event: OptimizationEvent) -> None:
        if event.type == EventType.ITERATION_COMPLETED and not first_iteration_done.is_set():
            action()
            first_iteration_done.set()

    return first_iteration_done, on_event


def _read_redis_status(control_key: str) -> JobStatus:
    return get_optimization_reader(control_key).get_status()


def _assert_artifacts_exist(results_dir: Path, expected: set[str] = EXPECTED_ARTIFACTS) -> None:
    for filename in expected:
        filepath = results_dir / filename
        assert filepath.exists(), f"Expected artifact {filename} not found in {results_dir}"
        assert filepath.stat().st_size > 0, f"Artifact {filename} is empty"


def _assert_result_content(results_dir: Path, *, min_rows: int = 1) -> None:
    data = pl.read_parquet(results_dir / "results.parquet")
    assert len(data) >= min_rows, f"Expected >= {min_rows} rows, got {len(data)}"
    assert "charge_ltb" in data.columns

    with (results_dir / "config.json").open() as f:
        config = json.load(f)
    assert config["objective_name"] == "charge_ltb"
    assert config["environment_name"] == "test_plugin"

    with (results_dir / "best_point.json").open() as f:
        best = json.load(f)
    assert isinstance(best, dict)
    assert len(best) > 0

    with (results_dir / "metadata.json").open() as f:
        meta = json.load(f)
    assert "job_id" in meta


def _assert_cancelled_partial_run(worker: Any, thread: threading.Thread, control_key: str) -> None:
    """Assert the shared post-cancel outcome: the worker finished cleanly, wrote a
    complete artifact set holding only part of the run's rows, and left the job
    CANCELLED."""
    thread.join(timeout=15.0)
    assert not thread.is_alive(), "Worker should have completed"
    assert worker.result_holder["error"] is None, f"Worker raised: {worker.result_holder['error']}"

    result_path = worker.result_holder["path"]
    assert result_path is not None
    _assert_artifacts_exist(result_path)
    _assert_result_content(result_path, min_rows=1)

    total_expected = N_ITERATIONS + 1
    data = pl.read_parquet(result_path / "results.parquet")
    assert 1 <= len(data) < total_expected, f"Expected partial rows, got {len(data)}"

    assert _read_redis_status(control_key) == JobStatus.CANCELLED


# Service runner — only patches get_redis_client, runs real engine


def _run_service(
    redis_url: str,
    control_key: str,
    tmp_path: Path,
    on_event: Callable[[OptimizationEvent], None] | None = None,
    config: OptimizationConfig | None = None,
) -> Path | None:
    """Run start_optimization inside a JobLifecycle, end-to-end.

    Mirrors the production worker (``run_optimization_process``): the service does
    domain work and propagates the outcome, while the wrapping JobLifecycle owns
    the terminal status. A cancel surfaces as a CancellationError that JobLifecycle
    classifies as CANCELLED and that the process boundary then absorbs — replicated
    here by swallowing it. The results path is read back from Redis (set by the
    success path or by partial-results-on-cancel), not from a return value.

    Only ``get_redis_client`` is patched (to use the embedded test Redis). The real
    OptimizationEngine, Xopt loop, evaluator, and event pipeline all execute against
    the mock simulator environment.
    """
    config = config or _make_config()
    with patch(
        "osprey.services.optimizer.utils.redis_client.get_redis_client",
        side_effect=_make_redis_factory(redis_url),
    ):
        try:
            with JobLifecycle(control_key):
                service = ExecutionService(
                    results_dir=tmp_path / "results",
                    control_key=control_key,
                )
                service.start_optimization(
                    config.model_copy(update={"on_complete": "hold"}), on_event=on_event
                )
        except CancellationError:
            pass  # JobLifecycle.__exit__ already recorded CANCELLED — mirror the process boundary.

        results_path = get_optimization_reader(control_key).get_last_results_path()
        return Path(results_path) if results_path else None


class TestNormalCompletion:
    """Optimization completes all iterations without pause/cancel."""

    @pytest.mark.usefixtures("mock_plugin_env")
    def test_creates_all_artifacts_and_reloads_via_load_results(
        self, redis_server: str, tmp_path: Path
    ) -> None:
        """One real run covers both halves of the completion contract: the artifact
        set it writes, and the round-trip back through ``load_results`` (full
        variable schema + best point), which reads those same files."""
        control_key = "test_lifecycle_normal"
        result_path = _run_service(redis_server, control_key, tmp_path)

        assert result_path is not None
        _assert_artifacts_exist(result_path)
        _assert_result_content(result_path, min_rows=N_ITERATIONS)

        status = _read_redis_status(control_key)
        assert status == JobStatus.COMPLETED

        data, best_point, config = load_results(str(result_path))

        variables = ["LN______MASTPH_AC00", "LN______AS2_PH_AC00"]
        assert "charge_ltb" in data.columns
        for var in variables:
            assert var in data.columns
            assert var in best_point
        assert config["objective_name"] == "charge_ltb"
        assert config["environment_name"] == "test_plugin"


class TestPauseResumeCompletion:
    """Optimization is paused, then resumed, and completes normally."""

    @pytest.mark.usefixtures("mock_plugin_env")
    def test_pause_resume_creates_all_artifacts(self, lifecycle_worker: Any) -> None:
        control_key = "test_lifecycle_pause_resume"
        first_iter_done, on_event = _make_iteration_signal()

        control_token = lifecycle_worker.make_token(control_key)
        thread = lifecycle_worker.start(control_key, on_event)

        assert first_iter_done.wait(timeout=10.0), "First iteration should complete"
        control_token.pause()
        assert wait_until(control_token.is_waiting), "Worker should be blocked in wait_if_paused"
        control_token.resume()

        thread.join(timeout=15.0)
        assert not thread.is_alive(), "Worker should have completed"
        assert lifecycle_worker.result_holder["error"] is None, lifecycle_worker.result_holder[
            "error"
        ]

        result_path = lifecycle_worker.result_holder["path"]
        assert result_path is not None
        _assert_artifacts_exist(result_path)
        _assert_result_content(result_path, min_rows=N_ITERATIONS)

        status = _read_redis_status(control_key)
        assert status == JobStatus.COMPLETED


class TestCancelWhileRunning:
    """Optimization is cancelled after at least one iteration."""

    @pytest.mark.usefixtures("mock_plugin_env")
    def test_cancel_saves_partial_results(self, lifecycle_worker: Any) -> None:
        control_key = "test_lifecycle_cancel_running"
        control_token = lifecycle_worker.make_token(control_key)
        # Cancel from inside the first-iteration callback so the next
        # handle_control_callbacks raises CancellationError deterministically,
        # with only partial data persisted — rather than racing the main thread's
        # cancel against a run that may already have completed all iterations.
        first_iter_done, on_event = _make_first_iteration_action(control_token.cancel)

        thread = lifecycle_worker.start(control_key, on_event)

        assert first_iter_done.wait(timeout=10.0), "First iteration should complete"
        _assert_cancelled_partial_run(lifecycle_worker, thread, control_key)


class TestCancelWhilePaused:
    """Optimization is paused, then cancelled while blocked in wait_if_paused."""

    @pytest.mark.usefixtures("mock_plugin_env")
    def test_cancel_while_paused_saves_partial_results(self, lifecycle_worker: Any) -> None:
        control_key = "test_lifecycle_cancel_paused"
        control_token = lifecycle_worker.make_token(control_key)
        # Pause from inside the first-iteration callback so the worker blocks
        # deterministically at the top of the iteration loop (with only partial
        # data) rather than racing to completion and blocking at the post-run
        # result-pipeline checkpoint, where all rows would already be persisted.
        first_iter_done, on_event = _make_first_iteration_action(control_token.pause)

        thread = lifecycle_worker.start(control_key, on_event)

        assert first_iter_done.wait(timeout=10.0), "First iteration should complete"
        assert wait_until(control_token.is_waiting), "Worker should be blocked in wait_if_paused"
        control_token.cancel()

        _assert_cancelled_partial_run(lifecycle_worker, thread, control_key)
