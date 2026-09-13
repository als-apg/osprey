"""Unit tests for the shared testcontainers helpers.

These drive ``start_or_skip`` with fake factories — no Docker engine, no
containers, no network — so they run in the plain unit lane.

Every skip case is asserted inside ``pytest.raises(pytest.skip.Exception)``.
A test that merely *calls* code raising ``Skipped`` reports SKIPPED with exit 0
and executes none of its assertions: it looks green while proving nothing.
"""

import subprocess

import pytest
import requests

from tests import _container_support
from tests._container_support import (
    docker_cli_unavailable_reason,
    start_or_skip,
    wait_until_ready,
)

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


def test_builtin_connection_error_is_retried_like_a_transport_error():
    """Testcontainers' port-publish race is transient and must get the retry.

    ``DockerClient.get_exposed_port`` raises the BUILTIN ``ConnectionError``
    ("Port mapping for container ... is not available") when the reaper races
    its own port publish. That class is deliberately asserted here to be outside
    the requests hierarchy: it is the whole reason a separate arm is needed, and
    a future refactor that dropped it would send this race back to the
    skip-immediately arm — where it produced an all-skipped module that looked
    exactly like a host with no Docker.
    """
    assert not issubclass(ConnectionError, requests.exceptions.RequestException)

    factory = RecordingFactory(
        ConnectionError("Port mapping for container abc is not available"), None
    )

    container = start_or_skip(factory, label="racing-db", backoff=0)

    assert container is factory.containers[1]
    assert factory.call_count == 2
    assert factory.containers[0].stop_calls == 1
    assert factory.containers[1].stop_calls == 0


def test_builtin_connection_error_twice_skips_after_two_attempts():
    """The retry is bounded for the race exactly as it is for a transport error."""
    factory = RecordingFactory(
        ConnectionError("port mapping not available"),
        ConnectionError("still not available"),
    )

    with pytest.raises(pytest.skip.Exception) as ei:
        start_or_skip(factory, label="racing-db", backoff=0)

    assert "racing-db" in ei.value.msg
    assert "ConnectionError" in ei.value.msg
    assert "still not available" in ei.value.msg
    assert factory.call_count == 2
    assert factory.stop_calls == 2


def test_a_host_without_a_docker_engine_still_skips_on_the_first_attempt():
    """No engine is not a race — it must not buy a second attempt.

    ``docker.from_env()`` raises ``DockerException`` when no daemon answers, and
    that is neither a ``RequestException`` nor a ``ConnectionError``, so it lands
    in the skip-immediately arm. Asserted so widening the retry class never
    quietly starts charging Docker-less contributors a backoff per fixture.
    """
    factory = RecordingFactory(
        docker_errors.DockerException("Error while fetching server API version")
    )

    with pytest.raises(pytest.skip.Exception) as ei:
        start_or_skip(factory, label="no-engine-db", backoff=0)

    assert "DockerException" in ei.value.msg
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


# ---------------------------------------------------------------------------
# docker_cli_unavailable_reason — the three ways "no docker" can be wrong
# ---------------------------------------------------------------------------


def _docker_on_path(monkeypatch) -> None:
    monkeypatch.setattr(_container_support.shutil, "which", lambda _name: "/usr/bin/docker")


def _completed(returncode: int, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["docker", "info"], returncode, stdout="", stderr=stderr)


def test_docker_probe_names_a_missing_cli(monkeypatch):
    monkeypatch.setattr(_container_support.shutil, "which", lambda _name: None)

    assert docker_cli_unavailable_reason() == "docker CLI not on PATH"


def test_docker_probe_names_a_timeout_apart_from_an_absent_daemon(monkeypatch):
    """The #820 case: a daemon that is slow to answer must not read as no daemon."""
    _docker_on_path(monkeypatch)

    def slow_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(_container_support.subprocess, "run", slow_run)

    reason = docker_cli_unavailable_reason(timeout=7)

    assert reason is not None
    assert "timed out after 7s" in reason
    assert "not reachable" not in reason


def test_docker_probe_names_an_unreachable_daemon_by_its_last_line(monkeypatch):
    _docker_on_path(monkeypatch)
    stderr = (
        "Client: Docker Engine - Community\n"
        "Cannot connect to the Docker daemon at unix:///var/run/docker.sock\n"
    )
    monkeypatch.setattr(
        _container_support.subprocess, "run", lambda argv, **kwargs: _completed(1, stderr)
    )

    reason = docker_cli_unavailable_reason()

    assert reason == (
        "docker daemon not reachable: "
        "Cannot connect to the Docker daemon at unix:///var/run/docker.sock"
    )


def test_docker_probe_is_silent_when_the_cli_works(monkeypatch):
    _docker_on_path(monkeypatch)
    monkeypatch.setattr(_container_support.subprocess, "run", lambda argv, **kwargs: _completed(0))

    assert docker_cli_unavailable_reason() is None


# ---------------------------------------------------------------------------
# wait_until_ready
# ---------------------------------------------------------------------------


class CountingProbe:
    """A probe that raises *failures* times before it starts succeeding.

    ``failures=None`` never succeeds, which is the case the deadline is for.
    Spelling it as ``None`` rather than a large count matters: a counted probe
    with no pause between attempts exhausts any plausible count long before a
    short deadline expires, and the test would then be asserting the success
    path under a name that promises the failure one.
    """

    def __init__(self, failures: int | None, error: BaseException | None = None):
        self.remaining = failures
        self.error = error or ConnectionResetError("connection reset by peer")
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
        if self.remaining is None:
            raise self.error
        if self.remaining > 0:
            self.remaining -= 1
            raise self.error


def test_a_probe_that_answers_at_once_is_called_once():
    probe = CountingProbe(failures=0)

    wait_until_ready(probe, "mongodb", interval=0.0)

    assert probe.calls == 1


def test_a_probe_that_answers_on_the_third_try_returns_after_three_calls():
    probe = CountingProbe(failures=2)

    wait_until_ready(probe, "mongodb", interval=0.0)

    assert probe.calls == 3


def test_a_probe_that_never_answers_fails_with_the_label_and_the_last_cause():
    probe = CountingProbe(failures=None, error=ConnectionRefusedError("port not published"))

    with pytest.raises(AssertionError) as caught:
        wait_until_ready(probe, "mongodb-seed", timeout=0.05, interval=0.0)

    message = str(caught.value)
    assert "mongodb-seed" in message
    assert "ConnectionRefusedError" in message
    assert "port not published" in message


# ---------------------------------------------------------------------------
# wait_until_ready — liveness and progress
# ---------------------------------------------------------------------------


class FakeStartedContainer:
    """A stand-in for a started container, driven line by line by the test.

    ``log_lines`` is consumed one entry per read, so a test spells out exactly
    what the subject was doing while the wait watched it: a new line is a boot
    still making progress, a repeat is a subject sitting there quiet. ``status``
    is read the same way and can be changed mid-run to stop the container.
    """

    def __init__(self, log_lines: list[bytes], status: str = "running"):
        self.log_lines = list(log_lines)
        self.status = status
        self.attrs = {"State": {"ExitCode": 137, "Error": ""}}
        self.reads = 0
        self.last_tail: bytes = b""

    # -- the docker-py surface ``_container_liveness`` uses ------------------

    def reload(self) -> None:
        self.reads += 1

    def logs(self, tail: int = 0) -> bytes:
        if self.log_lines:
            self.last_tail = self.log_lines.pop(0)
        return self.last_tail

    # -- the testcontainers surface ----------------------------------------

    def get_wrapped_container(self) -> "FakeStartedContainer":
        return self


def test_a_container_that_keeps_logging_outlives_the_quiet_budget():
    """The regression this whole signal exists for.

    The probe fails for far more rounds than ``timeout`` alone would allow, but
    the container is visibly still booting the whole time, so the wait stays
    with it and sees the subject come up. Measured as a fixed window this run
    would have failed on a slow machine while the subject was still working.
    """
    probe = CountingProbe(failures=10, error=ConnectionRefusedError("connection refused"))
    container = FakeStartedContainer([f"line {n}".encode() for n in range(20)])

    wait_until_ready(probe, "mongodb", timeout=0.05, interval=0.0, container=container)

    assert probe.calls == 11


def test_a_quiet_container_still_gives_up_after_the_quiet_budget():
    probe = CountingProbe(failures=None, error=ConnectionRefusedError("connection refused"))
    container = FakeStartedContainer([b"the one and only line"])

    with pytest.raises(AssertionError) as caught:
        wait_until_ready(probe, "mongodb", timeout=0.05, interval=0.0, container=container)

    assert "without visible progress" in str(caught.value)


def test_a_container_that_exited_fails_at_once_with_its_exit_code():
    """A dead subject must not be reported in the words of a slow one."""
    probe = CountingProbe(failures=None, error=ConnectionRefusedError("connection refused"))
    container = FakeStartedContainer([b"crashing"], status="exited")

    with pytest.raises(AssertionError) as caught:
        wait_until_ready(probe, "mongodb", timeout=30.0, interval=0.0, container=container)

    message = str(caught.value)
    assert "'exited'" in message
    assert "exit code: 137" in message
    assert probe.calls == 1


def test_a_chatty_container_that_never_answers_still_hits_the_ceiling():
    """Progress may extend the wait, never remove its end."""
    probe = CountingProbe(failures=None, error=ConnectionRefusedError("connection refused"))
    container = FakeStartedContainer([f"line {n}".encode() for n in range(10_000)])

    with pytest.raises(AssertionError) as caught:
        wait_until_ready(
            probe, "mongodb", timeout=30.0, interval=0.0, container=container, ceiling=0.05
        )

    assert "ceiling" in str(caught.value)


def test_a_port_that_never_opened_is_reported_apart_from_one_that_refused_the_probe():
    refused = CountingProbe(failures=None, error=ConnectionRefusedError("connection refused"))
    with pytest.raises(AssertionError) as caught:
        wait_until_ready(refused, "mongodb", timeout=0.05, interval=0.0)
    assert "nothing ever listened on the port" in str(caught.value)

    rejected = CountingProbe(failures=None, error=RuntimeError("auth failed"))
    with pytest.raises(AssertionError) as caught:
        wait_until_ready(rejected, "mongodb", timeout=0.05, interval=0.0)
    assert "would not serve the probe" in str(caught.value)


def test_a_wrapped_refusal_is_still_read_as_a_port_that_never_opened():
    """pymongo reports a refused port as a selection timeout, not as a refusal."""
    wrapped = RuntimeError(
        "localhost:56557: [Errno 61] Connection refused (configured timeouts: ...)"
    )
    probe = CountingProbe(failures=None, error=wrapped)

    with pytest.raises(AssertionError) as caught:
        wait_until_ready(probe, "mongodb", timeout=0.05, interval=0.0)

    assert "nothing ever listened on the port" in str(caught.value)


def test_an_unreadable_container_leaves_the_wait_on_the_probe_alone():
    """A daemon too busy to answer is the condition this wait sits through."""

    class UnreadableContainer:
        def get_wrapped_container(self):
            raise RuntimeError("daemon busy")

    probe = CountingProbe(failures=2)

    wait_until_ready(probe, "mongodb", timeout=30.0, interval=0.0, container=UnreadableContainer())

    assert probe.calls == 3
