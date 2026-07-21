"""Unit tests for `do_launch`, the bridge's single scan-start choke point."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from osprey.services.bluesky_bridge.plan_runner import FakePlanRunner
from osprey.services.bluesky_bridge.runs import RunRegistry, do_launch


@pytest.fixture
def registry() -> RunRegistry:
    return RunRegistry()


def test_do_launch_builds_and_starts_the_scanner(registry: RunRegistry) -> None:
    run = registry.add(request={"devices": ["motor1"]})
    runner = FakePlanRunner()

    result = do_launch(run, lambda: runner)

    assert result is run
    assert run.launched is True
    assert run.launching is False
    assert run.runner is runner
    assert runner.reinitialize_calls == 1
    assert runner.start_calls == 1
    assert run.status == "running"


def test_do_launch_passes_the_run_request_as_exec_config(registry: RunRegistry) -> None:
    seen: list[object] = []

    class RecordingScanner(FakePlanRunner):
        def reinitialize(self, exec_config: object) -> bool:
            seen.append(exec_config)
            return super().reinitialize(exec_config)

    run = registry.add(request={"devices": ["motor1"]})
    do_launch(run, RecordingScanner)

    assert seen == [{"devices": ["motor1"]}]


def test_do_launch_rejects_a_second_launch(registry: RunRegistry) -> None:
    run = registry.add(request={})
    do_launch(run, FakePlanRunner)

    with pytest.raises(HTTPException) as excinfo:
        do_launch(run, FakePlanRunner)
    assert excinfo.value.status_code == 409


def test_do_launch_rejects_launching_a_stopped_run(registry: RunRegistry) -> None:
    run = registry.add(request={})
    run.stopped = True

    with pytest.raises(HTTPException) as excinfo:
        do_launch(run, FakePlanRunner)
    assert excinfo.value.status_code == 409
    assert run.launched is False


def test_do_launch_rejects_reentrant_launch_in_progress(registry: RunRegistry) -> None:
    run = registry.add(request={})
    run.launching = True  # simulate a launch already underway

    with pytest.raises(HTTPException) as excinfo:
        do_launch(run, FakePlanRunner)
    assert excinfo.value.status_code == 409
    assert run.launched is False


def test_do_launch_returns_500_and_records_error_when_reinitialize_fails(
    registry: RunRegistry,
) -> None:
    run = registry.add(request={})
    runner = FakePlanRunner(reinitialize_fails=True)

    with pytest.raises(HTTPException) as excinfo:
        do_launch(run, lambda: runner)

    assert excinfo.value.status_code == 500
    assert run.launched is False
    assert run.launching is False
    assert run.error is not None
    assert run.status == "error"


def test_do_launch_returns_500_when_runner_factory_raises(registry: RunRegistry) -> None:
    run = registry.add(request={})

    def failing_factory() -> FakePlanRunner:
        raise RuntimeError("could not build runner")

    with pytest.raises(HTTPException) as excinfo:
        do_launch(run, failing_factory)

    assert excinfo.value.status_code == 500
    assert run.launched is False
    assert run.launching is False
    assert run.error == "could not build runner"


def test_do_launch_stops_a_scanner_that_partially_started_then_raised(
    registry: RunRegistry,
) -> None:
    """MANDATORY handoff fix (task 2.7): a `start_run_thread()` that raises after
    partially starting something must not leave a live, untracked, unstoppable
    scan behind — `run.runner` is never published on this path, so nothing
    else could ever call `stop_run_thread()` on it unless `do_launch`
    itself does.
    """
    run = registry.add(request={})

    class PartiallyStartingScanner(FakePlanRunner):
        def start_run_thread(self) -> None:
            # Simulate a real PlanRunner whose thread partially launches before
            # failing to fully come up.
            self._active = True
            raise RuntimeError("thread launch failed after partial start")

    runner = PartiallyStartingScanner()

    with pytest.raises(HTTPException) as excinfo:
        do_launch(run, lambda: runner)

    assert excinfo.value.status_code == 500
    assert run.runner is None
    assert runner.stop_calls == 1
    assert runner.is_run_active() is False


def test_do_launch_does_not_call_stop_when_the_factory_itself_raises(
    registry: RunRegistry,
) -> None:
    """No runner instance exists in this failure mode — there is nothing to stop."""
    run = registry.add(request={})
    stop_calls: list[str] = []

    class RecordingScanner(FakePlanRunner):
        def stop_run_thread(self) -> None:
            stop_calls.append("stop")
            super().stop_run_thread()

    def failing_factory() -> RecordingScanner:
        raise RuntimeError("could not build runner")

    with pytest.raises(HTTPException):
        do_launch(run, failing_factory)

    assert stop_calls == []


def test_a_failed_launch_can_be_retried(registry: RunRegistry) -> None:
    run = registry.add(request={})

    with pytest.raises(HTTPException):
        do_launch(run, lambda: FakePlanRunner(reinitialize_fails=True))
    assert run.error is not None

    runner = FakePlanRunner()
    result = do_launch(run, lambda: runner)

    assert result is run
    assert run.launched is True
    assert run.error is None
    assert run.runner is runner
