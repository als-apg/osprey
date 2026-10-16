"""``osprey up`` reports a web terminal ready only once it answers.

``compose up -d`` returns twenty seconds or more before a terminal's app
listens, and nginx — up long before that — answers a browser with a raw 502 in
the meantime. The readiness gate closes that gap: each service's own
healthcheck, run on demand, decides when it is ready; a container that stopped
fails at once; and a window that ends first fails with the ones that are up
named beside the ones that are not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from osprey.cli.phase_reporter import PhaseReporter, install_reporter
from osprey.deployment.web_terminals import readiness
from osprey.deployment.web_terminals.readiness import (
    READY_TIMEOUT_ENV,
    ReadinessTarget,
    WebTerminalsNotReady,
    readiness_targets,
    wait_until_ready,
    web_terminal_ready_timeout,
)


class _Lines(PhaseReporter):
    def __init__(self) -> None:
        super().__init__(color=False)
        self.lines: list[str] = []

    def emit(self, text: str, style: str | None = None) -> None:  # noqa: ARG002
        self.lines.append(text)

    @property
    def steps(self) -> list[str]:
        return [
            line.strip()[2:].split(" (")[0] for line in self.lines if line.lstrip().startswith("· ")
        ]


@pytest.fixture
def reporter():
    rep = _Lines()
    previous = install_reporter(rep)
    rep.phase("Starting")
    try:
        yield rep
    finally:
        install_reporter(previous)


@pytest.fixture(autouse=True)
def _no_timeout_env(monkeypatch):
    monkeypatch.delenv(READY_TIMEOUT_ENV, raising=False)


def _compose(tmp_path: Path, services: dict) -> Path:
    path = tmp_path / "docker-compose.web.yml"
    path.write_text(yaml.safe_dump({"services": services}, sort_keys=False))
    return path


_CURL = ["CMD-SHELL", "curl -fsS http://127.0.0.1:9100/health || exit 1"]


def test_every_service_with_a_healthcheck_is_a_target(tmp_path):
    compose = _compose(
        tmp_path,
        {
            "nginx": {"container_name": "p-nginx", "healthcheck": {"test": _CURL}},
            "web-alice": {"container_name": "p-web-alice", "healthcheck": {"test": _CURL}},
            "auth": {
                "container_name": "p-auth",
                "healthcheck": {"test": ["CMD", "python", "-c", "pass"]},
            },
            "no-check": {"container_name": "p-other"},
        },
    )

    targets = readiness_targets(compose, "p")

    assert [(t.label, t.container) for t in targets] == [
        ("nginx", "p-nginx"),
        ("terminal alice", "p-web-alice"),
        ("auth", "p-auth"),
    ]
    assert targets[1].check == "curl -fsS http://127.0.0.1:9100/health || exit 1"
    assert targets[2].check == "python -c pass"


def test_an_unreadable_document_waits_on_nothing(tmp_path):
    assert readiness_targets(tmp_path / "missing.yml", "p") == []


def test_the_window_defaults_to_five_minutes_and_is_settable(monkeypatch):
    assert web_terminal_ready_timeout({}) == 300.0
    assert web_terminal_ready_timeout({"web_terminal_ready_timeout": 60}) == 60.0
    monkeypatch.setenv(READY_TIMEOUT_ENV, "20")
    assert web_terminal_ready_timeout({"web_terminal_ready_timeout": 60}) == 20.0


_ALICE = ReadinessTarget("terminal alice", "p-web-alice", "true")
_BOB = ReadinessTarget("terminal bob", "p-web-bob", "true")


@pytest.fixture
def host(monkeypatch):
    """A scripted runtime: per container, its state and how many checks fail first."""
    script: dict[str, dict] = {}

    def state(_runtime, container, _env):
        return script[container]["state"]

    def check(_runtime, target, _env):
        entry = script[target.container]
        entry["checks"] = entry.get("checks", 0) + 1
        return entry["checks"] > entry.get("fails", 0)

    monkeypatch.setattr(readiness, "_container_state", state)
    monkeypatch.setattr(readiness, "_check_passes", check)
    monkeypatch.setattr(readiness, "_POLL_S", 0.0)
    monkeypatch.setattr(
        readiness, "container_log_tail", lambda runtime, name: f"{name}: PermissionError"
    )
    return script


def test_each_target_is_reported_as_it_becomes_ready(host, reporter):
    host["p-web-alice"] = {"state": "running", "fails": 3}
    host["p-web-bob"] = {"state": "running"}

    wait_until_ready("podman", [_ALICE, _BOB], {}, timeout=30)

    assert reporter.steps == ["terminal bob ready", "terminal alice ready"]


def test_a_container_that_is_not_running_yet_is_waited_on(host, reporter, monkeypatch):
    host["p-web-alice"] = {"state": "created"}

    def promote(_seconds):
        host["p-web-alice"]["state"] = "running"

    monkeypatch.setattr(readiness.time, "sleep", promote)
    wait_until_ready("podman", [_ALICE], {}, timeout=30)

    assert reporter.steps == ["terminal alice ready"]


@pytest.mark.usefixtures("reporter")
def test_a_stopped_container_fails_at_once_with_its_log(host):
    host["p-web-alice"] = {"state": "running"}
    host["p-web-bob"] = {"state": "exited"}

    with pytest.raises(WebTerminalsNotReady) as excinfo:
        wait_until_ready("podman", [_ALICE, _BOB], {}, timeout=3600)

    message = str(excinfo.value)
    assert "terminal bob stopped before it answered (container exited)" in message
    assert "terminal alice is up and serving" in message
    assert "p-web-bob: PermissionError" in excinfo.value.detail


@pytest.mark.usefixtures("reporter")
def test_a_window_that_ends_first_names_ready_and_not_ready(host, monkeypatch):
    host["p-web-alice"] = {"state": "running"}
    host["p-web-bob"] = {"state": "running", "fails": 10**9}
    clock = iter(range(0, 10**6, 100))
    monkeypatch.setattr(readiness.time, "monotonic", lambda: float(next(clock)))

    with pytest.raises(WebTerminalsNotReady) as excinfo:
        wait_until_ready("podman", [_ALICE, _BOB], {}, timeout=300)

    exc = excinfo.value
    assert exc.not_ready == ["terminal bob"] and exc.ready == ["terminal alice"]
    assert "did not answer within 300s" in str(exc)
    assert "web_terminal_ready_timeout" in str(exc) and READY_TIMEOUT_ENV in str(exc)
    assert "p-web-bob: PermissionError" in exc.detail


def test_nothing_to_wait_on_returns_at_once(reporter):
    wait_until_ready("podman", [], {}, timeout=1)
    assert reporter.steps == []
