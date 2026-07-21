"""Unit tests for the web-terminal deploy orchestration entrypoints.

Covers ``osprey.deployment.web_terminals.provision`` in isolation: the web
stack's own ``compose down``. The deploy_up-entry orchestration that wires
the provisioning modules together lives in
``tests/deployment/test_container_lifecycle.py``; the split-out provisioning
steps have their own modules and test files (``test_persona_images.py``,
``test_env_production.py``, ``test_postup_hooks.py``).
"""

from __future__ import annotations

import os

from osprey.deployment.web_terminals import provision


class _FakeCompletedProcess:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ---------------------------------------------------------------------------
# deploy_down_web_terminals -- the web stack's own `compose down`
# ---------------------------------------------------------------------------


def test_deploy_down_web_terminals_runs_compose_down_on_web_file(monkeypatch, tmp_path):
    """With a rendered docker-compose.web.yml at the project root, the web
    stack gets its own `compose -f docker-compose.web.yml down` under the
    pinned compose project — the mirror of deploy_up_web_terminals' second
    invocation. Without it the fixed-name `<prefix>-web-<user>`/`<prefix>-nginx`
    containers outlive every `osprey deploy down` and the next web-terminals
    deploy on the host dies at `up` with a container-name Conflict."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docker-compose.web.yml").write_text("services: {}\n", encoding="utf-8")

    recorded: dict = {}

    def _fake_run(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded["env"] = kwargs.get("env")
        return _FakeCompletedProcess()

    monkeypatch.setattr(provision.subprocess, "run", _fake_run)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["docker", "compose"])

    provision.deploy_down_web_terminals(
        {"project_name": "myproj"}, dict(os.environ), ["--env-file", ".env"]
    )

    assert recorded["cmd"] == [
        "docker",
        "compose",
        "-f",
        "docker-compose.web.yml",
        "--env-file",
        ".env",
        "down",
    ]
    assert recorded["env"]["COMPOSE_PROJECT_NAME"] == "myproj"


def test_deploy_down_web_terminals_noop_without_web_file(monkeypatch, tmp_path):
    """No rendered web compose file (nothing was ever deployed from this root,
    or the render predates web terminals) → no compose invocation at all."""
    monkeypatch.chdir(tmp_path)

    def _unexpected_run(cmd, **kwargs):
        raise AssertionError(f"unexpected subprocess.run: {cmd}")

    monkeypatch.setattr(provision.subprocess, "run", _unexpected_run)

    provision.deploy_down_web_terminals({"project_name": "myproj"}, dict(os.environ), [])
