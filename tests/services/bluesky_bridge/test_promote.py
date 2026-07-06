"""Unit tests for `do_promote`, the bridge's single scan-start choke point."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from osprey.services.bluesky_bridge.runs import RunRegistry, do_promote
from osprey.services.bluesky_bridge.scanner import FakeScanner


@pytest.fixture
def registry() -> RunRegistry:
    return RunRegistry()


def test_do_promote_builds_and_starts_the_scanner(registry: RunRegistry) -> None:
    run = registry.add(request={"devices": ["motor1"]})
    scanner = FakeScanner()

    result = do_promote(run, lambda: scanner)

    assert result is run
    assert run.promoted is True
    assert run.promoting is False
    assert run.scanner is scanner
    assert scanner.reinitialize_calls == 1
    assert scanner.start_calls == 1
    assert run.status == "running"


def test_do_promote_passes_the_run_request_as_exec_config(registry: RunRegistry) -> None:
    seen: list[object] = []

    class RecordingScanner(FakeScanner):
        def reinitialize(self, exec_config: object) -> bool:
            seen.append(exec_config)
            return super().reinitialize(exec_config)

    run = registry.add(request={"devices": ["motor1"]})
    do_promote(run, RecordingScanner)

    assert seen == [{"devices": ["motor1"]}]


def test_do_promote_rejects_a_second_promotion(registry: RunRegistry) -> None:
    run = registry.add(request={})
    do_promote(run, FakeScanner)

    with pytest.raises(HTTPException) as excinfo:
        do_promote(run, FakeScanner)
    assert excinfo.value.status_code == 409


def test_do_promote_rejects_promoting_a_stopped_run(registry: RunRegistry) -> None:
    run = registry.add(request={})
    run.stopped = True

    with pytest.raises(HTTPException) as excinfo:
        do_promote(run, FakeScanner)
    assert excinfo.value.status_code == 409
    assert run.promoted is False


def test_do_promote_rejects_reentrant_promotion_in_progress(registry: RunRegistry) -> None:
    run = registry.add(request={})
    run.promoting = True  # simulate a promote already underway

    with pytest.raises(HTTPException) as excinfo:
        do_promote(run, FakeScanner)
    assert excinfo.value.status_code == 409
    assert run.promoted is False


def test_do_promote_returns_500_and_records_error_when_reinitialize_fails(
    registry: RunRegistry,
) -> None:
    run = registry.add(request={})
    scanner = FakeScanner(reinitialize_fails=True)

    with pytest.raises(HTTPException) as excinfo:
        do_promote(run, lambda: scanner)

    assert excinfo.value.status_code == 500
    assert run.promoted is False
    assert run.promoting is False
    assert run.error is not None
    assert run.status == "error"


def test_do_promote_returns_500_when_scanner_factory_raises(registry: RunRegistry) -> None:
    run = registry.add(request={})

    def failing_factory() -> FakeScanner:
        raise RuntimeError("could not build scanner")

    with pytest.raises(HTTPException) as excinfo:
        do_promote(run, failing_factory)

    assert excinfo.value.status_code == 500
    assert run.promoted is False
    assert run.promoting is False
    assert run.error == "could not build scanner"


def test_a_failed_promote_can_be_retried(registry: RunRegistry) -> None:
    run = registry.add(request={})

    with pytest.raises(HTTPException):
        do_promote(run, lambda: FakeScanner(reinitialize_fails=True))
    assert run.error is not None

    scanner = FakeScanner()
    result = do_promote(run, lambda: scanner)

    assert result is run
    assert run.promoted is True
    assert run.error is None
    assert run.scanner is scanner
