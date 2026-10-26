"""Tests for api/runner.py — run_optimization_process."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from osprey_optimizer.api.job_tracker import JobTracker, create_job_state
from osprey_optimizer.api.runner import run_optimization_process
from osprey_optimizer.models.enums import JobStatus, JobType
from osprey_optimizer.utils.job_keys import state_prefix
from osprey_optimizer.utils.redis_client import get_redis_client
from osprey_optimizer.utils.serialization import deserialize_obj


@pytest.fixture()
def _runner_mocks():
    """Patch all external dependencies of run_optimization_process."""
    with (
        patch("osprey_optimizer.api.runner.setup_logging"),
        patch("osprey_optimizer.api.runner.JobLifecycle") as mock_lifecycle,
        patch("osprey_optimizer.api.runner.ExecutionService") as mock_exec_cls,
        patch("osprey_optimizer.api.runner.OptimizationConfig") as mock_config_cls,
    ):
        config = MagicMock()
        config.environment_name = "test_env"
        mock_config_cls.model_validate.return_value = config

        mock_lifecycle.return_value.__enter__ = MagicMock()
        mock_lifecycle.return_value.__exit__ = MagicMock(return_value=False)

        yield {
            "config": config,
            "lifecycle": mock_lifecycle,
            "exec_cls": mock_exec_cls,
            "config_cls": mock_config_cls,
        }


class TestRunOptimizationProcess:
    @pytest.mark.parametrize(
        ("results_dir", "expected_run_dir"),
        [
            ("/results", Path("/results")),
            (None, None),
        ],
    )
    def test_wiring_threads_args_to_collaborators(
        self, _runner_mocks, results_dir, expected_run_dir
    ) -> None:
        """job_id, config and results_dir are threaded to the right collaborators.

        Covers both branches of ``Path(results_dir) if results_dir else None``:
        a truthy path is wrapped in a Path, and None passes through as None.
        """
        service = MagicMock()
        service.start_optimization.return_value = expected_run_dir
        _runner_mocks["exec_cls"].return_value = service

        run_optimization_process(
            "job-123", {"environment_name": "test_env"}, results_dir=results_dir
        )

        # The caller's config dict is what gets validated — dropping it would
        # silently run a default config instead of the job's setpoints and objective.
        _runner_mocks["config_cls"].model_validate.assert_called_once_with(
            {"environment_name": "test_env"}
        )
        _runner_mocks["lifecycle"].assert_called_once_with("job-123")
        _runner_mocks["exec_cls"].assert_called_once_with(
            control_key="job-123", specific_run_dir=expected_run_dir
        )
        service.start_optimization.assert_called_once_with(_runner_mocks["config"])

    def test_worker_signal_handlers_installed_before_the_run(self, _runner_mocks) -> None:
        """SIGTERM to the worker must reach the cancel path from the job's first write."""

        def _install() -> None:
            assert not _runner_mocks["lifecycle"].called

        with patch(
            "osprey_optimizer.api.runner.install_worker_signal_handlers", side_effect=_install
        ) as install:
            run_optimization_process("job-123", {"environment_name": "test_env"})

        install.assert_called_once_with()
        _runner_mocks["lifecycle"].assert_called_once_with("job-123")


@pytest.fixture()
def clean_redis(redis_server: str) -> Generator[None, None, None]:
    """Real-Redis backing that flushes the shared DB after the test (no key leaks)."""
    del redis_server  # wanted for the server it starts, not its URL
    yield
    get_redis_client(binary=True).flushdb()


class TestRunOptimizationProcessCrashState:
    """Real-Redis integration: a crash must propagate clean ERROR state via JobLifecycle."""

    @pytest.mark.usefixtures("clean_redis")
    def test_runner_crash_state_propagation(self) -> None:
        """Verify that a crash in the runner correctly updates Redis state via JobLifecycle."""
        job_id = str(uuid.uuid4())
        # As the launcher does before spawning: claim the slot, set RUNNING.
        assert JobTracker().claim_active_job(job_id, 0, job_type=JobType.OPTIMIZATION)
        create_job_state(JobType.OPTIMIZATION, job_id).set_status(JobStatus.RUNNING)
        # Malformed config to ensure we trigger the lifecycle's error handling
        bad_config = {"invalid": "data"}

        with (
            patch(
                "osprey_optimizer.api.runner.OptimizationConfig.model_validate",
                side_effect=ValueError("Simulated Config Crash"),
            ),
            pytest.raises(ValueError, match="Simulated Config Crash"),
        ):
            run_optimization_process(job_id, bad_config)

        redis = get_redis_client(binary=True)

        status_bytes = redis.get(f"{state_prefix(job_id)}:status")
        assert status_bytes is not None
        status = deserialize_obj(status_bytes)
        assert status == JobStatus.ERROR

        # Clean str(exc), not a traceback — the traceback goes to the log, not Redis.
        error_bytes = redis.get(f"{state_prefix(job_id)}:error_message")
        assert error_bytes is not None
        error_msg = deserialize_obj(error_bytes)
        assert error_msg == "Simulated Config Crash"
        assert "Traceback" not in error_msg

        tracker = JobTracker()
        # The active job key is deleted in mark_completed
        assert tracker.get_active_job() is None or tracker.get_active_job() != job_id
