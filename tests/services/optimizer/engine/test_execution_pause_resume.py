"""ExecutionService pause/resume/cancel behavior, with OptimizationEngine mocked out."""

import threading
import time
from unittest.mock import patch

import polars as pl
import pytest
from redis import Redis

from osprey.services.optimizer.models.config import OptimizationConfig
from osprey.services.optimizer.models.exceptions import CancellationError
from osprey.services.optimizer.services.execution import ExecutionService
from osprey.services.optimizer.services.orchestration import (
    OptimizationEngine,
    OptimizationMetadata,
    OptimizationResult,
)
from osprey.services.optimizer.utils.cancellation import CancellationToken
from osprey.services.optimizer.utils.job_keys import cancellation_key

from .conftest import wait_until


def simulate_background_worker(
    config: OptimizationConfig,
    control_key: str,
    pause_flag: threading.Event,
    proceed_flag: threading.Event | None = None,
    waiting_flag: threading.Event | None = None,
    redis_url: str | None = None,
) -> str:
    """Simulate a background worker process running optimization.

    Args:
        config: Optimization configuration
        control_key: Redis key for control
        pause_flag: Flag to signal when to check pause
        proceed_flag: Optional flag to wait for before proceeding to next iteration
        waiting_flag: Optional flag to signal when worker is waiting for resume
        redis_url: Redis URL to connect to
    """

    iterations = []

    def mock_run_optimization(self_engine):
        """Mock optimization that respects pause/resume."""
        token = self_engine.cancellation_token

        for i in range(config.n_iterations):
            if token:
                if token.is_paused() and waiting_flag:
                    waiting_flag.set()
                token.wait_if_paused()
                token.check_cancellation()

            iterations.append(f"iter_{i}")

            if i == 0:
                pause_flag.set()
                if proceed_flag:
                    proceed_flag.wait()

        data = pl.DataFrame(
            {
                "var1": [0.1, 0.2, 0.15],
                "charge_ltb": [1.0, 2.0, 2.5],
            }
        )
        best_point = {"var1": 0.15, "charge_ltb": 2.5}
        return data, best_point

    def patched_run(self):
        """Run optimization with patched dependencies."""
        data, best_point = mock_run_optimization(self)
        metadata = OptimizationMetadata(config=self.config)
        return OptimizationResult(data=data, best_point=best_point, metadata=metadata)

    try:
        # Real client against the test server, despite the mock_ name.
        def mock_get_redis(binary: bool = False) -> Redis:  # type: ignore[type-arg]
            """Get Redis client for testing."""
            assert redis_url is not None
            if binary:
                return Redis.from_url(redis_url, decode_responses=False)
            return Redis.from_url(redis_url, decode_responses=True)

        with (
            patch(
                "osprey.services.optimizer.utils.redis_client.get_redis_client",
                side_effect=mock_get_redis,
            ),
            patch.object(OptimizationEngine, "run", patched_run),
        ):
            service = ExecutionService(control_key=control_key)
            # start_optimization propagates the outcome (no swallow): success
            # returns the results dir, a cancel raises CancellationError — exactly
            # what the wrapping JobLifecycle.__exit__ classifies in production.
            service.start_optimization(config.model_copy(update={"on_complete": "hold"}))

        return f"completed:{len(iterations)}"
    except CancellationError:
        return "cancelled"
    except Exception as e:
        import traceback

        return f"error:{e!s}\n{traceback.format_exc()}"


class TestBackgroundPauseResumeSimulation:
    """Test pause/resume in a background task scenario."""

    @pytest.mark.usefixtures("fake_initial_state")
    @pytest.mark.parametrize(
        "post_pause_action,expected_status",
        # "completed:3" pins that ALL n_iterations ran after the resume —
        # a pause must suspend the loop, never drop an iteration.
        [("resume", "completed:3"), ("cancel", "cancelled")],
        ids=["pause_then_resume", "pause_then_cancel"],
    )
    def test_pause_then_action_across_processes(
        self,
        redis_server,
        tmp_path,
        monkeypatch,
        post_pause_action,
        expected_status,
    ):
        monkeypatch.chdir(tmp_path)
        control_key = f"test_control_key_{post_pause_action}"
        redis_client = Redis.from_url(redis_server, decode_responses=True)

        config = OptimizationConfig(
            n_iterations=3,
            environment_config={},
            objective_name="charge_ltb",
            environment_name="test_plugin",
        )

        try:
            # Mirror how the worker derives its token key (get_cancellation_token)
            # so this control-side token coordinates with the same Redis hash.
            control_token = CancellationToken(
                redis_client=redis_client,
                redis_key=cancellation_key(control_key),
                poll_interval=0.05,
            )

            pause_flag = threading.Event()
            proceed_flag = threading.Event()
            result: dict[str, str | None] = {"status": None}

            def worker_thread():
                """Run simulation in background thread."""
                result["status"] = simulate_background_worker(
                    config, control_key, pause_flag, proceed_flag, redis_url=redis_server
                )

            worker = threading.Thread(target=worker_thread, daemon=True)
            worker.start()

            assert pause_flag.wait(timeout=2.0), "Worker should have signaled after first iteration"

            control_token.pause()
            assert control_token.is_paused()

            proceed_flag.set()

            assert wait_until(control_token.is_waiting, timeout=2.0), (
                "Worker should be blocked by wait_if_paused()"
            )

            assert worker.is_alive()

            if post_pause_action == "resume":
                control_token.resume()
            elif post_pause_action == "cancel":
                control_token.cancel()
            else:
                raise ValueError(f"Unknown post_pause_action: {post_pause_action}")

            worker.join(timeout=5.0)

            assert not worker.is_alive(), "Worker should have completed"
            assert result["status"] == expected_status, f"Unexpected status: {result['status']}"
        finally:
            redis_client.delete(cancellation_key(control_key))
            redis_client.close()

    @pytest.mark.usefixtures("fake_initial_state")
    def test_cancel_before_worker_starts_is_honoured(self, redis_server, tmp_path, monkeypatch):
        """A cancel issued before the worker reaches start_optimization must stick.

        The API accepts POST /jobs/{id}/cancel from any non-terminal state, but the
        worker spends ~15 s importing before it runs. start_optimization resets the
        token, so a cancel arriving in that window used to be acknowledged and then
        wiped — the run went ahead on hardware. Note there is no pause counterpart:
        pause is CAS-gated to RUNNING upstream.
        """
        monkeypatch.chdir(tmp_path)
        control_key = "test_control_key_cancel_before_start"
        redis_client = Redis.from_url(redis_server, decode_responses=True)

        config = OptimizationConfig(
            n_iterations=3,
            environment_config={},
            objective_name="charge_ltb",
            environment_name="test_plugin",
        )

        try:
            control_token = CancellationToken(
                redis_client=redis_client,
                redis_key=cancellation_key(control_key),
                poll_interval=0.05,
            )
            result: dict[str, str | None] = {"status": None}

            # Never set: the cancel fires before the engine runs, so the worker
            # exits without reaching the iteration loop that would signal it.
            unreached_pause_flag = threading.Event()

            def worker_thread():
                result["status"] = simulate_background_worker(
                    config, control_key, unreached_pause_flag, redis_url=redis_server
                )

            # Cancel BEFORE the worker exists — the window this test is about.
            control_token.cancel()
            assert control_token.is_cancelled()

            worker = threading.Thread(target=worker_thread, daemon=True)
            worker.start()
            worker.join(timeout=5.0)

            assert not worker.is_alive(), "Worker should have exited"
            assert result["status"] == "cancelled", (
                f"Pre-start cancel was discarded: {result['status']}"
            )
        finally:
            redis_client.delete(cancellation_key(control_key))
            redis_client.close()


class TestWaitIfPausedBlocking:
    """Unit tests for CancellationToken.wait_if_paused() blocking behavior."""

    @pytest.mark.parametrize("n_clients", [1, 2], ids=["same_client", "cross_client"])
    def test_wait_if_paused_blocks_until_resume(self, redis_server, n_clients):
        """Verify that wait_if_paused() actually blocks until resume() is called.

        With two clients the pausing and the waiting token talk to Redis over
        separate connections, proving pause/resume works across processes.
        """
        clients = [Redis.from_url(redis_server, decode_responses=True) for _ in range(n_clients)]
        test_key = f"test_key_block_{n_clients}"
        clients[0].delete(test_key)
        try:
            control = CancellationToken(
                redis_client=clients[0], redis_key=test_key, poll_interval=0.05
            )
            # Same key, last client: identical to `control` for n_clients == 1.
            waiter = CancellationToken(
                redis_client=clients[-1], redis_key=test_key, poll_interval=0.05
            )

            control.pause()

            blocked_duration = []

            def worker():
                """Wait for pause token."""
                start = time.time()
                waiter.wait_if_paused()
                blocked_duration.append(time.time() - start)

            t = threading.Thread(target=worker)
            t.start()

            assert wait_until(waiter.is_waiting, timeout=2.0), (
                "Worker should be reported as waiting"
            )
            assert t.is_alive(), "Worker thread should be active"

            control.resume()
            t.join(timeout=5.0)

            assert not t.is_alive(), "Worker should have completed after resume"
            assert len(blocked_duration) == 1
            assert blocked_duration[0] >= 0.05  # At least one poll interval
        finally:
            clients[0].delete(test_key)
            for client in clients:
                client.close()

    def test_wait_if_paused_returns_immediately_when_not_paused(self, redis_server):
        """Verify that wait_if_paused() returns immediately when not paused."""
        redis_client = Redis.from_url(redis_server, decode_responses=True)
        test_key = "test_key_imm"
        redis_client.delete(test_key)
        try:
            token = CancellationToken(
                redis_client=redis_client, redis_key=test_key, poll_interval=0.05
            )

            start = time.time()
            token.wait_if_paused()
            elapsed = time.time() - start

            assert elapsed < 0.1, f"Should return immediately, took {elapsed:.3f}s"
        finally:
            redis_client.delete(test_key)
            redis_client.close()

    def test_wait_if_paused_raises_on_cancel_while_paused(self, redis_server):
        """Verify that cancel() while blocked in wait_if_paused() raises CancellationError."""
        redis_client = Redis.from_url(redis_server, decode_responses=True)
        test_key = "test_key_cancel_wait"
        redis_client.delete(test_key)
        try:
            token = CancellationToken(
                redis_client=redis_client, redis_key=test_key, poll_interval=0.05
            )

            token.pause()

            exception_raised = []

            def worker():
                """Wait for pause token and catch cancellation."""
                try:
                    token.wait_if_paused()
                except CancellationError:
                    exception_raised.append(True)

            t = threading.Thread(target=worker)
            t.start()

            assert wait_until(token.is_waiting, timeout=2.0), "Worker should be reported as waiting"
            assert t.is_alive(), "Worker thread should be active"

            token.cancel()
            t.join(timeout=5.0)

            assert not t.is_alive(), "Worker should have exited"
            assert len(exception_raised) == 1, "CancellationError should have been raised"
        finally:
            redis_client.delete(test_key)
            redis_client.close()

    def test_wait_if_paused_respects_poll_interval(self, redis_server):
        """Verify that wait_if_paused() respects the configured poll_interval."""
        redis_client = Redis.from_url(redis_server, decode_responses=True)
        poll_interval = 0.05
        # Use unique key to avoid interference from other tests
        test_key = "test_key_poll_interval"
        redis_client.delete(test_key)
        try:
            token = CancellationToken(
                redis_client=redis_client,
                redis_key=test_key,
                poll_interval=poll_interval,
            )

            token.pause()

            target_polls = 3

            def worker():
                """Wait for pause token."""
                token.wait_if_paused()

            t = threading.Thread(target=worker)
            start = time.time()
            t.start()

            assert wait_until(lambda: token.get_poll_count() >= target_polls, timeout=2.0), (
                f"Worker should have polled at least {target_polls} times"
            )
            # Reaching target_polls must take real time: without the sleep between
            # polls the paused loop is an unbounded Redis busy-spin that would hit
            # the target instantly. Lower bound only, so load can never flake it.
            elapsed = time.time() - start
            assert elapsed >= (target_polls - 1) * poll_interval, (
                f"Polled {target_polls} times in only {elapsed:.3f}s"
            )

            token.resume()
            t.join(timeout=5.0)

            assert token.get_poll_count() >= target_polls
        finally:
            redis_client.delete(test_key)
            redis_client.close()
