"""``unhealthy_containers``: this repo's containers whose healthcheck failed.

The read behind the closing line of a detached start. Pinned against the two
listings the supported runtimes emit, scoped by the repo label, and advisory
on every way the host can decline to answer.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from osprey.deployment import container_lifecycle
from osprey.deployment.container_lifecycle import unhealthy_containers

DOCKER_LISTING = "\n".join(
    json.dumps(row)
    for row in [
        {"Names": "demo-bluesky-bridge", "State": "running", "Status": "Up 2 minutes (unhealthy)"},
        {"Names": "demo-bluesky-redis", "State": "running", "Status": "Up 2 minutes (healthy)"},
        {"Names": "demo-web-terminal-1", "State": "running", "Status": "Up 2 minutes"},
        {"Names": "demo-graphdb", "State": "running", "Status": "Up 1 minute (health: starting)"},
    ]
)

PODMAN_LISTING = json.dumps(
    [
        {"Names": ["demo-bluesky-bridge"], "State": "running", "Health": "unhealthy"},
        {"Names": ["demo-bluesky-redis"], "State": "running", "Health": "healthy"},
    ]
)


class _FakePs:
    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, self.returncode, stdout=self.stdout, stderr="")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config=None: ["docker", "compose"]
    )
    return tmp_path


def test_docker_names_the_container_whose_status_says_unhealthy(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakePs(DOCKER_LISTING)
    monkeypatch.setattr(container_lifecycle.subprocess, "run", fake)

    assert unhealthy_containers(repo) == ["demo-bluesky-bridge"]


def test_podman_names_the_container_whose_health_field_says_so(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(container_lifecycle.subprocess, "run", _FakePs(PODMAN_LISTING))

    assert unhealthy_containers(repo) == ["demo-bluesky-bridge"]


def test_the_listing_is_scoped_to_this_repo_and_includes_stopped_containers(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakePs("")
    monkeypatch.setattr(container_lifecycle.subprocess, "run", fake)

    unhealthy_containers(repo)

    (cmd,) = fake.calls
    assert cmd[:3] == ["docker", "ps", "-a"]
    assert cmd[cmd.index("--filter") + 1] == container_lifecycle._repo_label_filter(repo)
    assert cmd[-2:] == ["--format", "json"]


def test_a_healthy_stack_answers_nothing(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    listing = json.dumps({"Names": "demo-redis", "Status": "Up 2 minutes (healthy)"})
    monkeypatch.setattr(container_lifecycle.subprocess, "run", _FakePs(listing))

    assert unhealthy_containers(repo) == []


def test_a_listing_that_fails_is_not_a_failure(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(container_lifecycle.subprocess, "run", _FakePs("", returncode=1))

    assert unhealthy_containers(repo) == []


def test_no_runtime_is_not_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _none(config=None):
        raise RuntimeError("no container runtime")

    monkeypatch.setattr(container_lifecycle, "get_runtime_command", _none)

    assert unhealthy_containers(tmp_path) == []


def test_the_rendered_config_selects_the_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = tmp_path / "build"
    build.mkdir()
    (build / "config.yml").write_text("project_name: demo\ncontainer_runtime: podman\n")
    seen: list[object] = []

    def _select(config=None):
        seen.append(config)
        return ["podman", "compose"]

    monkeypatch.setattr(container_lifecycle, "get_runtime_command", _select)
    fake = _FakePs("[]")
    monkeypatch.setattr(container_lifecycle.subprocess, "run", fake)

    unhealthy_containers(tmp_path)

    assert seen and seen[0]["container_runtime"] == "podman"
    assert fake.calls[0][0] == "podman"


def test_a_listing_with_no_output_at_all_is_not_a_failure(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shape a stubbed runtime hands back: a completed process with no
    stdout. Advisory means total — nothing after the call may raise either."""

    def _bare(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _bare)

    assert unhealthy_containers(repo) == []
