"""``down`` acts on the runtime the as-built config names, or refuses.

The as-built ``build/config.yml`` records the runtime that served the build.
A lifecycle verb reading ``docker`` there must talk to docker: when docker does
not answer it refuses with docker's own remedy, and never lets detection fall
through to a podman that is also installed. The alternative is the failure this
pins: a ``down`` served by podman-compose finds no containers, exits 0, and
reports a stack still running under docker as stopped.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from osprey.deployment import container_lifecycle, runtime_helper

_RENDERED_CONFIG = """\
project_name: pinned
build_dir: ./build
container_runtime: docker
deployed_services:
  - event_dispatcher
services:
  event_dispatcher:
    path: services/event_dispatcher
"""


@pytest.fixture
def pinned_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "pinned"
    build = repo / "build"
    (build / "services" / "event_dispatcher").mkdir(parents=True)
    (repo / "profile.yml").write_text("name: pinned\n")
    (build / "config.yml").write_text(_RENDERED_CONFIG)
    (build / "services" / "docker-compose.yml").write_text("networks:\n  osprey-network:\n")
    (build / "services" / "event_dispatcher" / "docker-compose.yml").write_text(
        "services:\n  event-dispatcher:\n    image: x\n"
    )
    return repo


def _host_where_docker_hangs_and_podman_answers(monkeypatch: pytest.MonkeyPatch) -> list:
    """Both runtimes on PATH; docker's daemon probe times out, podman's succeed."""
    probed: list[list[str]] = []

    def _run(cmd, **kwargs):
        probed.append(list(cmd))
        if cmd[0] == "docker" and cmd[1] == "ps":
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 5))
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(runtime_helper.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(runtime_helper.subprocess, "run", _run)
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    return probed


def test_down_refuses_when_the_pinned_runtime_does_not_answer(monkeypatch, pinned_repo):
    probed = _host_where_docker_hangs_and_podman_answers(monkeypatch)
    compose_runs: list[list[str]] = []
    monkeypatch.setattr(
        container_lifecycle,
        "run_captured",
        lambda cmd, **kwargs: compose_runs.append(list(cmd)),
    )

    with pytest.raises(RuntimeError, match="[Dd]ocker"):
        container_lifecycle.down_deployment(pinned_repo)

    assert compose_runs == [], "no compose command may run against a runtime that was not pinned"
    assert all(cmd[0] == "docker" for cmd in probed), probed
