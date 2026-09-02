"""``osprey up`` dying on an unhealthy container reports what the container said.

The unit half lives in ``tests/deployment/test_unhealthy_container_detail.py``;
this pins the verb: the container's last log lines reach the failure block the
operator reads, beneath the exception's own line, and nothing about the path
changes when there is nothing to add.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.main import cli
from osprey.deployment import runtime_helper
from osprey.deployment.errors import CapturedProcessError

ABORT = "dependency failed to start: container demo-bluesky-bridge is unhealthy\n"
TAIL = "RuntimeError: refusing to start writable: the limits database could not be parsed\n"


@pytest.fixture(autouse=True)
def _container_runtime_is_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "osprey.deployment.runtime_helper.verify_runtime_is_running",
        lambda config=None: (True, ""),
    )


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A deployment repo with a render, every runtime seam stubbed out."""
    from osprey.cli import deploy_cmd, repo_resolver
    from osprey.deployment import container_lifecycle

    build = tmp_path / "build"
    build.mkdir()
    (build / "config.yml").write_text("project_name: demo\n", encoding="utf-8")

    monkeypatch.setattr(repo_resolver, "find_repo_root", lambda start=None: tmp_path)
    monkeypatch.setattr(deploy_cmd, "gate_start_from_build", lambda *a, **k: None)
    monkeypatch.setattr(deploy_cmd, "ensure_repo_env", lambda *a, **k: None)
    monkeypatch.setattr(deploy_cmd, "load_project_config", lambda *a, **k: {})
    monkeypatch.setattr(
        container_lifecycle, "as_built_config_path", lambda root: tmp_path / "build" / "config.yml"
    )
    monkeypatch.setattr(
        runtime_helper, "get_runtime_command", lambda config=None: ["docker", "compose"]
    )
    return tmp_path


def _dies_on_unhealthy(monkeypatch: pytest.MonkeyPatch, repo: Path, output: str) -> None:
    from osprey.deployment import container_lifecycle

    spool = repo / "compose-up.log"
    spool.write_text(output, encoding="utf-8")

    def boom(*args, **kwargs):
        raise CapturedProcessError(["docker", "compose", "up", "-d"], 1, spool)

    monkeypatch.setattr(container_lifecycle, "up_as_built", boom)


def test_the_failure_block_carries_the_containers_log(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dies_on_unhealthy(monkeypatch, repo, ABORT)
    calls: list[list[str]] = []

    def fake_logs(cmd, **kwargs):
        import subprocess

        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout=TAIL)

    monkeypatch.setattr(runtime_helper.subprocess, "run", fake_logs)

    result = CliRunner().invoke(cli, ["up", "-d"])

    assert result.exit_code != 0
    assert "Deployment failed" in result.output
    # The exception's own line is still there ...
    assert "exited 1" in result.output
    # ... and the container's reason follows it.
    assert "demo-bluesky-bridge is unhealthy" in result.output
    assert "refusing to start writable" in result.output
    assert calls == [["docker", "logs", "--tail", "30", "demo-bluesky-bridge"]]


def test_a_failure_that_blames_no_container_is_reported_as_before(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dies_on_unhealthy(monkeypatch, repo, "Error response from daemon: port is already allocated\n")

    def never(cmd, **kwargs):
        raise AssertionError(f"no log should be read: {cmd}")

    monkeypatch.setattr(runtime_helper.subprocess, "run", never)

    result = CliRunner().invoke(cli, ["up", "-d"])

    assert result.exit_code != 0
    assert "Deployment failed" in result.output
    assert "exited 1" in result.output
    assert "unhealthy" not in result.output
