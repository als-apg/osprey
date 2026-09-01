"""``compose up`` dying on an unhealthy container: the failure report says why.

Compose waits on every ``service_healthy`` edge and, when a dependency's
healthcheck never passes, ends on one line naming the container and nothing
about the cause. The cause is in that container's own log, which the compose
spool does not carry. These pin the translation: the blamed container is read
off the captured output, its last lines are read through the runtime, and the
verb appends them beneath the exception's own line.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from osprey.deployment import runtime_helper, subprocess_capture
from osprey.deployment.errors import CapturedProcessError
from osprey.deployment.runtime_helper import (
    UNHEALTHY_LOG_TAIL_LINES,
    explain_unhealthy_containers,
    unhealthy_containers_in,
)
from osprey.deployment.subprocess_capture import captured_failure_detail

# The abort exactly as Docker Compose v2 renders it at the end of a deploy.
REAL_ABORT = (
    " Container htu-assistant-bluesky-redis  Healthy\n"
    " Container htu-assistant-bluesky-bridge  Error\n"
    "dependency failed to start: container htu-assistant-bluesky-bridge is unhealthy\n"
)

# What that container said before its healthcheck gave up on it.
BRIDGE_TAIL = (
    "ValueError: Channel 'defaults': 'verification' was replaced by 'confirm: true|false'.\n"
    "RuntimeError: refusing to start writable: lane bluesky serves target live, where\n"
    "control_system.limits_checking.database_path could not be read or parsed\n"
)


class _FakeRuntime:
    """A ``subprocess.run`` that answers ``<runtime> logs`` and records the argv."""

    def __init__(self, tail: str | None = BRIDGE_TAIL, returncode: int = 0) -> None:
        self.tail = tail
        self.returncode = returncode
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, self.returncode, stdout=self.tail or "")


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> _FakeRuntime:
    fake = _FakeRuntime()
    monkeypatch.setattr(runtime_helper.subprocess, "run", fake)
    return fake


# ---------------------------------------------------------------------------
# Reading the blamed container off the output


def test_the_blamed_container_is_read_off_the_compose_abort() -> None:
    assert unhealthy_containers_in(REAL_ABORT) == ["htu-assistant-bluesky-bridge"]


def test_a_container_blamed_twice_is_named_once() -> None:
    assert unhealthy_containers_in(REAL_ABORT + REAL_ABORT) == ["htu-assistant-bluesky-bridge"]


def test_output_that_blames_nothing_names_nothing() -> None:
    assert unhealthy_containers_in("Error response from daemon: port is already allocated\n") == []
    assert unhealthy_containers_in("") == []


# ---------------------------------------------------------------------------
# The explanation


def test_the_explanation_carries_the_containers_last_lines(fake_runtime: _FakeRuntime) -> None:
    text = explain_unhealthy_containers(REAL_ABORT, "docker")

    assert text is not None
    assert text.startswith("htu-assistant-bluesky-bridge is unhealthy.")
    # The validator's own words reach the operator, indented under the name.
    assert "  ValueError: Channel 'defaults'" in text
    assert "  RuntimeError: refusing to start writable" in text
    assert fake_runtime.calls == [
        ["docker", "logs", "--tail", str(UNHEALTHY_LOG_TAIL_LINES), "htu-assistant-bluesky-bridge"]
    ]


def test_the_tail_is_read_through_the_runtime_the_deployment_resolved(
    fake_runtime: _FakeRuntime,
) -> None:
    explain_unhealthy_containers(REAL_ABORT, "podman")

    assert fake_runtime.calls[0][0] == "podman"


def test_a_log_that_cannot_be_read_still_names_the_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_helper.subprocess, "run", _FakeRuntime(tail=None, returncode=1))

    text = explain_unhealthy_containers(REAL_ABORT, "docker")

    assert text == "htu-assistant-bluesky-bridge is unhealthy. Run `osprey logs` to see why."


def test_no_runtime_to_ask_still_names_the_container() -> None:
    text = explain_unhealthy_containers(REAL_ABORT, None)

    assert text == "htu-assistant-bluesky-bridge is unhealthy. Run `osprey logs` to see why."


def test_a_runtime_that_does_not_answer_is_not_a_second_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _hang(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 15)

    monkeypatch.setattr(runtime_helper.subprocess, "run", _hang)

    text = explain_unhealthy_containers(REAL_ABORT, "docker")

    assert text is not None
    assert "Run `osprey logs`" in text


def test_output_that_blames_nothing_explains_nothing(fake_runtime: _FakeRuntime) -> None:
    assert explain_unhealthy_containers("something else went wrong\n", "docker") is None
    assert fake_runtime.calls == []


# ---------------------------------------------------------------------------
# From the captured failure to the detail


def _captured(tmp_path: Path, output: str | None) -> CapturedProcessError:
    if output is None:
        return CapturedProcessError(["docker", "compose", "up", "-d"], 1)
    spool = tmp_path / "compose-up.log"
    spool.write_text(output, encoding="utf-8")
    return CapturedProcessError(["docker", "compose", "up", "-d"], 1, spool)


def test_the_detail_is_read_off_the_spool(
    tmp_path: Path, fake_runtime: _FakeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runtime_helper, "get_runtime_command", lambda config=None: ["docker", "compose"]
    )

    detail = captured_failure_detail(_captured(tmp_path, REAL_ABORT), config={"x": 1})

    assert detail is not None
    assert "htu-assistant-bluesky-bridge is unhealthy" in detail
    assert "ValueError" in detail


def test_a_verbose_run_has_no_spool_and_no_detail(tmp_path: Path) -> None:
    assert captured_failure_detail(_captured(tmp_path, None)) is None


def test_a_spool_that_blames_nothing_yields_no_detail(
    tmp_path: Path, fake_runtime: _FakeRuntime
) -> None:
    assert captured_failure_detail(_captured(tmp_path, "port is already allocated\n")) is None
    assert fake_runtime.calls == []


def test_an_unresolvable_runtime_still_names_the_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_runtime(config=None):
        raise RuntimeError("no container runtime")

    monkeypatch.setattr(runtime_helper, "get_runtime_command", _no_runtime)

    detail = captured_failure_detail(_captured(tmp_path, REAL_ABORT))

    assert detail == "htu-assistant-bluesky-bridge is unhealthy. Run `osprey logs` to see why."


def test_an_exception_of_another_type_yields_no_detail() -> None:
    assert captured_failure_detail(RuntimeError("boom")) is None


def test_the_seam_is_exported_beside_its_sibling() -> None:
    assert "captured_failure_detail" in subprocess_capture.__all__
