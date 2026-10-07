"""The Web Terminal lifespan compares a managed policy with the launch environment.

A managed-policy ``env`` block outranks everything the deployment sets, so the
lifespan refuses to start when a policy provider variable differs from the value
the terminal's agent is launched with. A policy that agrees with it is no reason
to refuse. With no provider configured there is nothing to agree with, so any
policy provider key refuses.

Drives the real app factory through the TestClient lifespan with the real
``inject_provider_env``; only the policy read is pinned.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.agent_runner.provider_env import load_provider_spec
from osprey.cli.templates.manager import TemplateManager
from osprey.interfaces.web_terminal import app as web_app

_CBORG_CONFIG = "claude_code:\n  provider: cborg\n"


@pytest.fixture(autouse=True)
def _restore_environ():
    """The real injection writes to this interpreter's ``os.environ``."""
    environ = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(environ)


def _pin_policy(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    """Make the managed-policy read return ``env`` from a fixed source file."""
    policy = {var: (value, "/etc/claude-code/managed-settings.json") for var, value in env.items()}
    monkeypatch.setattr(
        "osprey.agent_runner.provider_env.read_managed_policy_env",
        lambda paths=None: policy,
    )


def _project(tmp_path: Path, config: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A flat project (render and secrets zone coincide) with *config*."""
    (tmp_path / "config.yml").write_text(config, encoding="utf-8")
    (tmp_path / "_agent_data").mkdir(exist_ok=True)
    monkeypatch.setattr(TemplateManager, "regen_if_drift", lambda self, pd: [])
    monkeypatch.setattr(
        web_app, "_load_web_config", lambda *_a, **_k: {"watch_dir": str(tmp_path / "_agent_data")}
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _serve(project: Path) -> None:
    """Enter and leave the real lifespan for *project*."""
    app = web_app.create_app(
        config_path=str(project / "config.yml"),
        shell_command="echo",
        project_dir=str(project),
    )
    with TestClient(app):
        pass


def test_a_policy_equal_to_the_launch_value_serves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CBORG_API_KEY", "sk-cborg")
    project = _project(tmp_path, _CBORG_CONFIG, monkeypatch)
    spec = load_provider_spec(project, include_telemetry=False)
    assert spec is not None
    _pin_policy(monkeypatch, ANTHROPIC_BASE_URL=spec.env_block["ANTHROPIC_BASE_URL"])

    _serve(project)


def test_a_disagreeing_policy_refuses_to_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CBORG_API_KEY", "sk-cborg")
    project = _project(tmp_path, _CBORG_CONFIG, monkeypatch)
    _pin_policy(monkeypatch, ANTHROPIC_BASE_URL="https://elsewhere.example.org")

    with pytest.raises(RuntimeError, match="Refusing to start the Web Terminal"):
        _serve(project)


def test_a_policy_key_refuses_when_no_provider_is_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path, "control_system:\n  type: mock\n", monkeypatch)
    _pin_policy(monkeypatch, ANTHROPIC_MODEL="anything")

    with pytest.raises(RuntimeError, match="Refusing to start the Web Terminal"):
        _serve(project)
