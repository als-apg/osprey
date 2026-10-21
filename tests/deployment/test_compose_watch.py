"""The watched, bounded ``compose up``: what it says while it waits, and after.

podman-compose implements ``depends_on: {condition: service_healthy}`` by
retrying ``podman wait --condition=healthy`` once a second with no limit, so a
dependency that never comes up holds ``osprey up`` forever with nothing on the
terminal but a timer. These pin the three answers to that: a still-waiting note
naming the pending containers, a time limit whose failure carries the
diagnosis, and the same diagnosis on Ctrl-C.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from osprey.cli.phase_reporter import PhaseReporter, install_reporter
from osprey.deployment import compose_watch
from osprey.deployment.compose_watch import (
    COMPOSE_UP_TIMEOUT_ENV,
    DEFAULT_COMPOSE_UP_TIMEOUT_S,
    ContainerState,
    compose_up_timeout,
    last_spool_line,
    stall_report,
    watched_compose_up,
)
from osprey.deployment.errors import CapturedProcessTimeout


class _Lines(PhaseReporter):
    def __init__(self) -> None:
        super().__init__(color=False)
        self.lines: list[str] = []

    def emit(self, text: str, style: str | None = None) -> None:  # noqa: ARG002
        self.lines.append(text)


@pytest.fixture
def reporter():
    rep = _Lines()
    previous = install_reporter(rep)
    rep.phase("Starting")
    rep.lines.clear()
    try:
        yield rep
    finally:
        install_reporter(previous)


@pytest.fixture(autouse=True)
def _no_timeout_env(monkeypatch):
    monkeypatch.delenv(COMPOSE_UP_TIMEOUT_ENV, raising=False)


# --- the time limit setting -------------------------------------------------


def test_the_default_limit_is_generous():
    assert compose_up_timeout({}) == DEFAULT_COMPOSE_UP_TIMEOUT_S == 1800.0


def test_the_config_key_sets_the_limit():
    assert compose_up_timeout({"compose_up_timeout": 600}) == 600.0


def test_the_environment_wins_over_the_config(monkeypatch):
    monkeypatch.setenv(COMPOSE_UP_TIMEOUT_ENV, "45")
    assert compose_up_timeout({"compose_up_timeout": 600}) == 45.0


@pytest.mark.parametrize("bad", [0, -5, "soon", True, ""])
def test_a_limit_that_is_not_a_positive_number_is_refused(bad):
    with pytest.raises(ValueError, match="compose_up_timeout"):
        compose_up_timeout({"compose_up_timeout": bad})


def test_a_bad_environment_value_is_refused_by_its_own_name(monkeypatch):
    monkeypatch.setenv(COMPOSE_UP_TIMEOUT_ENV, "ten minutes")
    with pytest.raises(ValueError, match=COMPOSE_UP_TIMEOUT_ENV):
        compose_up_timeout({})


# --- reading container states -------------------------------------------------


@pytest.mark.parametrize(
    ("status", "state", "stuck"),
    [
        ("Up 3 minutes", "", False),
        ("Up 3 minutes (healthy)", "", False),
        ("Up 4 seconds (health: starting)", "starting", False),
        ("Up 4 seconds (starting)", "starting", False),
        ("Up 2 minutes (unhealthy)", "unhealthy", True),
        ("Created", "created", False),
        ("Exited (1) 3 seconds ago", "exited (1)", True),
        ("Restarting (1) 2 seconds ago", "restarting", False),
    ],
)
def test_a_status_reads_as_the_word_an_operator_needs(status, state, stuck):
    container = ContainerState("p-bluesky-web", status)
    assert container.state == state
    assert container.stuck is stuck


def test_the_last_spool_line_is_the_last_thing_compose_said(tmp_path):
    spool = tmp_path / "compose-up.log"
    spool.write_text("first\nError: no container with name or ID found\n\n")
    assert last_spool_line(spool) == "Error: no container with name or ID found"
    assert last_spool_line(None) is None
    assert last_spool_line(tmp_path / "missing.log") is None


def test_a_long_last_line_is_clipped(tmp_path):
    spool = tmp_path / "compose-up.log"
    spool.write_text("x" * 500)
    assert len(last_spool_line(spool)) == 160


def test_containers_are_listed_by_their_compose_project_label(monkeypatch):
    seen: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="a\tUp 1 second\nb\tCreated\n")

    monkeypatch.setattr(compose_watch.subprocess, "run", fake_run)
    containers = compose_watch.project_containers("podman", "proj")

    assert containers == [ContainerState("a", "Up 1 second"), ContainerState("b", "Created")]
    assert "label=com.docker.compose.project=proj" in seen[0]
    assert seen[0][:2] == ["podman", "ps"]


# --- the diagnosis --------------------------------------------------------------


@pytest.fixture
def stalled_host(monkeypatch):
    """A project whose queueserver exited and whose web service waits on it."""
    monkeypatch.setattr(
        compose_watch,
        "project_containers",
        lambda runtime, project: [
            ContainerState("p-redis", "Up 5 minutes (healthy)"),
            ContainerState("p-bluesky-queueserver", "Exited (1) 4 minutes ago"),
            ContainerState("p-bluesky-web", "Created"),
        ],
    )
    monkeypatch.setattr(
        compose_watch,
        "container_log_tail",
        lambda runtime, name, lines: "PermissionError: [Errno 13] /app/config.yml",
    )


@pytest.mark.usefixtures("stalled_host")
def test_the_report_names_what_is_pending_and_why(tmp_path):
    spool = tmp_path / "s.log"
    spool.write_text('Error: no container with name or ID "p-bluesky-queueserver" found\n')

    report = stall_report("podman", "p", spool)

    assert "p-bluesky-queueserver (exited (1)), p-bluesky-web (created)" in report
    assert "p-redis" not in report.splitlines()[0]
    assert 'compose\'s last line: Error: no container with name or ID "p-bluesky-queueserver"' in (
        report
    )
    assert "p-bluesky-queueserver is exited (1). Its last 20 log lines:" in report
    assert "  PermissionError: [Errno 13] /app/config.yml" in report
    # Only a container that cannot get better by waiting has its log read.
    assert "p-bluesky-web is" not in report


@pytest.mark.usefixtures("stalled_host")
def test_the_watch_note_names_the_pending_containers(reporter, tmp_path):
    spool = tmp_path / "s.log"
    spool.write_text("Error: no container\n")

    compose_watch._UpWatcher("podman", "p")(61.0, spool)

    assert reporter.lines == [
        "    still waiting (1m01s): p-bluesky-queueserver (exited (1)), p-bluesky-web (created)",
        "    compose: Error: no container",
    ]


def test_a_runtime_that_will_not_list_still_gets_a_note(reporter, monkeypatch):
    monkeypatch.setattr(compose_watch, "project_containers", lambda runtime, project: None)
    compose_watch._UpWatcher("podman", "p")(30.0, None)
    assert reporter.lines == ["    still waiting (30.0s)"]


# --- around the call ------------------------------------------------------------


@pytest.mark.usefixtures("stalled_host")
def test_a_stopped_up_carries_the_diagnosis_and_the_setting(tmp_path):
    spool = tmp_path / "s.log"
    spool.write_text("waiting\n")
    cmd = ["podman", "compose", "up", "-d"]

    with pytest.raises(CapturedProcessTimeout) as excinfo:
        with watched_compose_up(
            cmd, config={"compose_up_timeout": 120}, env={"COMPOSE_PROJECT_NAME": "p"}
        ) as bounds:
            assert bounds["timeout"] == 120.0
            assert callable(bounds["watch"])
            raise CapturedProcessTimeout(cmd, 120.0, spool)

    detail = excinfo.value.detail
    assert "p-bluesky-queueserver (exited (1))" in detail
    assert "PermissionError" in detail
    assert "compose_up_timeout" in detail and COMPOSE_UP_TIMEOUT_ENV in detail


@pytest.mark.usefixtures("stalled_host")
def test_an_interrupted_up_prints_the_diagnosis_before_propagating(reporter):
    with pytest.raises(KeyboardInterrupt):
        with watched_compose_up(
            ["podman", "compose", "up", "-d"], config={}, env={"COMPOSE_PROJECT_NAME": "p"}
        ):
            raise KeyboardInterrupt

    text = "\n".join(reporter.lines)
    assert "Waiting on: p-bluesky-queueserver (exited (1))" in text
    assert "PermissionError" in text


def test_an_up_that_finishes_passes_straight_through(reporter):
    with watched_compose_up(
        ["docker", "compose", "up", "-d"], config={}, env={"COMPOSE_PROJECT_NAME": "p"}
    ) as bounds:
        assert bounds["watch_interval"] == compose_watch.WATCH_INTERVAL_S
    assert reporter.lines == []


def test_the_spool_path_of_a_report_may_be_absent(monkeypatch):
    monkeypatch.setattr(compose_watch, "project_containers", lambda runtime, project: [])
    assert stall_report("docker", "p", Path("/nonexistent")) == (
        "Waiting on: every container is up."
    )
