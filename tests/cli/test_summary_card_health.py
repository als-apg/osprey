"""The closing line says everything is running only when it is.

``compose up -d`` returns once every dependency edge is satisfied; a container
nothing depends on can go unhealthy after that with the verb already reporting
success. The line under the card asks first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.cli import output
from osprey.cli.phase_reporter import install_reporter
from osprey.cli.summary_card import print_call_to_action

from .test_summary_card import Recorded, RecordingReporter, recording_console


@pytest.fixture
def recorder():
    console, buffer = recording_console()
    previous = install_reporter(RecordingReporter(console))
    try:
        yield Recorded(buffer)
    finally:
        install_reporter(previous)


@pytest.fixture(autouse=True)
def _empty_ledger():
    output.clear_ledger()
    yield
    output.clear_ledger()


@pytest.fixture
def web_repo(tmp_path: Path) -> Path:
    build = tmp_path / "build"
    build.mkdir()
    (build / "config.yml").write_text(
        "project_name: demo\n"
        "deploy:\n  fqdn: demo.example.org\n"
        "modules:\n  web_terminals:\n    enabled: true\n    nginx_port: 8080\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def backend_repo(tmp_path: Path) -> Path:
    build = tmp_path / "build"
    build.mkdir()
    (build / "config.yml").write_text("project_name: demo\n", encoding="utf-8")
    return tmp_path


def _health(monkeypatch: pytest.MonkeyPatch, names: list[str]) -> None:
    monkeypatch.setattr(
        "osprey.deployment.container_lifecycle.unhealthy_containers", lambda repo_root: names
    )


def test_a_healthy_stack_is_told_everything_is_running(
    web_repo: Path, recorder: Recorded, monkeypatch: pytest.MonkeyPatch
) -> None:
    _health(monkeypatch, [])

    print_call_to_action(web_repo, "running")

    assert any(line.startswith("Everything is running. Open ") for line in recorder.lines)


def test_an_unhealthy_container_replaces_the_line(
    web_repo: Path, recorder: Recorded, monkeypatch: pytest.MonkeyPatch
) -> None:
    _health(monkeypatch, ["demo-bluesky-bridge"])

    print_call_to_action(web_repo, "running")

    assert "demo-bluesky-bridge is unhealthy. Run `osprey logs` to see why." in recorder.lines
    assert not any("Everything is running" in line for line in recorder.lines)


def test_two_unhealthy_containers_are_both_named(
    web_repo: Path, recorder: Recorded, monkeypatch: pytest.MonkeyPatch
) -> None:
    _health(monkeypatch, ["demo-a", "demo-b"])

    print_call_to_action(web_repo, "running")

    assert "demo-a, demo-b are unhealthy. Run `osprey logs` to see why." in recorder.lines


def test_a_backend_only_project_still_hears_about_an_unhealthy_container(
    backend_repo: Path, recorder: Recorded, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No landing page means no call to action -- but not no warning."""
    _health(monkeypatch, ["demo-graphdb"])

    print_call_to_action(backend_repo, "running")

    assert "demo-graphdb is unhealthy. Run `osprey logs` to see why." in recorder.lines


def test_a_healthy_backend_only_project_gets_no_line(
    backend_repo: Path, recorder: Recorded, monkeypatch: pytest.MonkeyPatch
) -> None:
    _health(monkeypatch, [])

    print_call_to_action(backend_repo, "running")

    assert recorder.lines == []


def test_the_health_read_belongs_to_the_running_state_alone(
    web_repo: Path, recorder: Recorded, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never(repo_root):
        raise AssertionError("a stopped card asks nothing")

    monkeypatch.setattr("osprey.deployment.container_lifecycle.unhealthy_containers", never)

    print_call_to_action(web_repo, "stopped")

    assert recorder.lines == []
