"""``down_deployment`` runs compose through the capture helper, not ``execvpe``.

The ``down`` used to hand itself to ``os.execvpe``, which made its output
impossible to capture: the compose child inherited the terminal and this process
was gone. It now runs through :func:`run_captured`, so the spool file and the
phase reporter both survive the call — while the reason ``execvpe`` (and not
``execvp``) was chosen in the first place, the ``COMPOSE_PROJECT_NAME`` pin,
still has to reach compose.

The attached ``up`` path is deliberately still ``execvpe``: there the operator
WANTS compose's live output, so it is guarded here against an over-broad
conversion.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from osprey.deployment import container_lifecycle
from osprey.deployment.deployment_record import DeploymentRecord, write_deployment_record

#: What the start before each live ``down`` below recorded.
_RECORDED = DeploymentRecord("docker", "myrepo")


def _capture_run(monkeypatch, returncode=0):
    """Patch run_captured in container_lifecycle's namespace; record its call."""
    calls: list[dict] = []

    def _fake(cmd, **kwargs):
        calls.append({"cmd": list(cmd), **kwargs})
        return subprocess.CompletedProcess(list(cmd), returncode)

    monkeypatch.setattr(container_lifecycle, "run_captured", _fake)
    return calls


# ---------------------------------------------------------------------------
# down_deployment — the live path (`osprey down` -> down_verb -> here).
# ---------------------------------------------------------------------------


def _mock_live_down(monkeypatch, tmp_path):
    """Wire down_deployment's as-built reads to fixtures under *tmp_path*."""
    write_deployment_record(tmp_path, _RECORDED)
    monkeypatch.setattr(
        container_lifecycle, "verify_runtime_is_running", lambda config, **_: (True, "")
    )
    config_path = tmp_path / "build" / "config.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("project_name: myrepo\n")

    monkeypatch.setattr(container_lifecycle, "as_built_config_path", lambda root: config_path)
    monkeypatch.setattr(
        container_lifecycle, "load_project_config", lambda p, **k: {"project_name": "myrepo"}
    )
    monkeypatch.setattr(
        container_lifecycle, "as_built_compose_files", lambda config, root: ["docker-compose.yml"]
    )
    monkeypatch.setattr(container_lifecycle, "_web_terminals_enabled", lambda config: False)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )


def test_the_live_down_captures_compose_under_the_pinned_project(monkeypatch, tmp_path):
    """The verb's own compose invocation is spooled, not streamed — with the pin."""
    _mock_live_down(monkeypatch, tmp_path)
    calls = _capture_run(monkeypatch)

    container_lifecycle.down_deployment(tmp_path)

    assert len(calls) == 1, calls
    call = calls[0]
    assert "down" in call["cmd"]
    assert call["spool_name"] == "compose-down"
    assert Path(call["repo_root"]) == tmp_path
    assert call["check"] is False
    assert call["env"]["COMPOSE_PROJECT_NAME"] == _RECORDED.compose_project
    # The rest of the environment still comes through — the pin is an overlay on
    # os.environ, not a replacement for it (compose reads PATH, DOCKER_HOST, ...).
    assert "PATH" in call["env"]


def test_the_live_down_reports_its_step(monkeypatch, tmp_path):
    """The verb's phase gets one line for the containers it stopped."""
    _mock_live_down(monkeypatch, tmp_path)
    _capture_run(monkeypatch)
    steps: list[str] = []
    monkeypatch.setattr(container_lifecycle, "_report_step", steps.append)

    container_lifecycle.down_deployment(tmp_path)

    assert steps == ["containers stopped"]


def test_a_failed_live_down_still_says_the_containers_may_be_running(monkeypatch, tmp_path):
    """The consequence a bare CapturedProcessError would not state.

    The spool is still written and noted on the reporter, so the failure line
    replays it; this path adds what a non-zero ``down`` MEANS.
    """
    _mock_live_down(monkeypatch, tmp_path)
    _capture_run(monkeypatch, returncode=1)
    steps: list[str] = []
    monkeypatch.setattr(container_lifecycle, "_report_step", steps.append)

    with pytest.raises(RuntimeError, match="may still be running"):
        container_lifecycle.down_deployment(tmp_path)

    assert steps == [], "a failed down stopped nothing to report"


def test_attached_up_still_execvpes(monkeypatch, tmp_path):
    """Non-regression: the conversion is scoped to ``down``.

    A non-detached ``up`` hands the terminal to compose on purpose — the
    operator asked to watch it — so that call site must still exec.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: ({"deployed_services": ["event_dispatcher"]}, ["docker-compose.yml"]),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(container_lifecycle, "_ensure_service_tokens", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "_build_project_image", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "log_endpoint_summary", lambda *a, **k: None)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(
        container_lifecycle,
        "run_captured",
        lambda cmd, **k: subprocess.CompletedProcess(list(cmd), 0),
    )
    execd: dict = {}
    monkeypatch.setattr(
        container_lifecycle.os,
        "execvpe",
        lambda file, args, env: execd.update(args=list(args), env=env),
    )
    # execvp too: a call that lost its env would otherwise replace the test
    # process instead of failing the assertion below.
    monkeypatch.setattr(
        container_lifecycle.os,
        "execvp",
        lambda file, args: pytest.fail("attached up must exec WITH its env"),
    )

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=False)

    assert "up" in execd["args"], execd
    assert execd["env"]["COMPOSE_PROJECT_NAME"]
