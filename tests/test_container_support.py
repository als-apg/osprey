"""Unit tests for the shared testcontainers helpers.

These drive ``start_or_skip`` with fake factories — no Docker engine, no
containers, no network — so they run in the plain unit lane.

Every skip case is asserted inside ``pytest.raises(pytest.skip.Exception)``.
A test that merely *calls* code raising ``Skipped`` reports SKIPPED with exit 0
and executes none of its assertions: it looks green while proving nothing.
"""

import pytest
import requests

from tests._container_support import start_or_skip

# ``docker`` is a ``dev``-extra dependency. Importing it at module scope would
# make this file ERROR at collection wherever the extra is absent — the very
# outcome ``_container_support`` keeps its own ``docker`` import lazy to avoid.
# ``importorskip`` degrades to a clean skip instead. The real exception class is
# used deliberately rather than a stand-in: the ordered ``except`` chain is only
# meaningful against the genuine ``ImageNotFound ⊂ NotFound ⊂ APIError ⊂
# RequestException`` hierarchy, which a fake could not reproduce.
docker_errors = pytest.importorskip("docker.errors")


class FakeContainer:
    """Minimal stand-in for a testcontainers container object.

    ``start`` raises the next error queued for it, or succeeds. ``stop`` records
    the call so tests can assert that failed attempts are cleaned up.
    """

    def __init__(self, error: BaseException | None):
        self.error = error
        self.started = False
        self.stop_calls = 0

    def start(self):
        self.started = True
        if self.error is not None:
            raise self.error
        return self

    def stop(self):
        self.stop_calls += 1


class RecordingFactory:
    """Zero-arg factory yielding a fresh ``FakeContainer`` per call.

    Mirrors the contract ``start_or_skip`` relies on: a retry must build a new
    container rather than restart the previous object.
    """

    def __init__(self, *errors: BaseException | None):
        self.errors = list(errors)
        self.containers: list[FakeContainer] = []

    def __call__(self) -> FakeContainer:
        error = self.errors[len(self.containers)]
        container = FakeContainer(error)
        self.containers.append(container)
        return container

    @property
    def call_count(self) -> int:
        return len(self.containers)

    @property
    def stop_calls(self) -> int:
        return sum(c.stop_calls for c in self.containers)


def test_retryable_failure_twice_skips_after_two_attempts():
    """Two consecutive transport errors exhaust the single retry and skip."""
    exc = requests.exceptions.RequestException("connection reset by peer")
    factory = RecordingFactory(exc, requests.exceptions.RequestException("still down"))

    with pytest.raises(pytest.skip.Exception) as ei:
        start_or_skip(factory, label="flaky-db", backoff=0)

    assert "flaky-db" in ei.value.msg
    assert "RequestException" in ei.value.msg
    assert "still down" in ei.value.msg
    assert factory.call_count == 2
    assert factory.stop_calls == 2
    # A retry must not restart the first object — each attempt gets its own.
    assert factory.containers[0] is not factory.containers[1]


def test_retryable_failure_then_success_returns_second_container():
    """The retry returns the freshly built container, not the failed one."""
    factory = RecordingFactory(requests.exceptions.ConnectionError("daemon busy"), None)

    container = start_or_skip(factory, label="flaky-db", backoff=0)

    assert container is factory.containers[1]
    assert factory.call_count == 2
    # Only the failed first attempt is torn down.
    assert factory.containers[0].stop_calls == 1
    assert factory.containers[1].stop_calls == 0
    assert factory.stop_calls == 1


def test_timeout_skips_without_retry():
    """A readiness timeout is not transient — no second 120s wait."""
    factory = RecordingFactory(TimeoutError("container did not become ready"))

    with pytest.raises(pytest.skip.Exception) as ei:
        start_or_skip(factory, label="slow-db", backoff=0)

    assert "slow-db" in ei.value.msg
    assert "TimeoutError" in ei.value.msg
    assert "container did not become ready" in ei.value.msg
    assert factory.call_count == 1
    assert factory.stop_calls == 1


def test_image_not_found_skips_without_retry():
    """``ImageNotFound`` is a ``RequestException`` subclass and must not retry.

    This is the case that proves the ordered ``except`` chain: a tuple-based
    implementation would pull a missing image into the retry arm and pay a
    second full readiness wait for an error that can never clear.
    """
    factory = RecordingFactory(docker_errors.ImageNotFound("no such image: nope:1"))

    with pytest.raises(pytest.skip.Exception) as ei:
        start_or_skip(factory, label="missing-image-db", backoff=0)

    assert "missing-image-db" in ei.value.msg
    assert "ImageNotFound" in ei.value.msg
    assert "no such image: nope:1" in ei.value.msg
    assert factory.call_count == 1
    assert factory.stop_calls == 1
