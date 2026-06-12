"""Layer 1 of the #244 fix: the web server re-syncs Claude Code artifacts on launch.

Drives the real app factory through the TestClient lifespan and asserts the
on-launch regen runs against the resolved project dir (and fails open if it raises,
so a regen error never prevents the server from starting).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from osprey.cli.templates.manager import TemplateManager
from osprey.interfaces.web_terminal.app import create_app


def _project(tmp_path):
    (tmp_path / "_agent_data").mkdir(exist_ok=True)
    return tmp_path


def test_lifespan_regenerates_on_launch(tmp_path, monkeypatch):
    project = _project(tmp_path)
    calls: list[Path] = []
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    monkeypatch.chdir(project)
    monkeypatch.setattr(
        TemplateManager, "regen_if_drift", lambda self, pd: calls.append(Path(pd)) or []
    )

    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={"watch_dir": str(project / "_agent_data")},
    ):
        app = create_app(shell_command="echo", project_dir=str(project))
        with TestClient(app):
            pass

    assert calls, "web launch must call regen_if_drift"
    # Uses the canonical (resolved) project dir the PTY runs in.
    assert calls[0] == project.resolve()


def test_lifespan_fails_open_when_regen_raises(tmp_path, monkeypatch):
    project = _project(tmp_path)
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    monkeypatch.chdir(project)

    def boom(self, pd):
        raise RuntimeError("regen exploded")

    monkeypatch.setattr(TemplateManager, "regen_if_drift", boom)

    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={"watch_dir": str(project / "_agent_data")},
    ):
        app = create_app(shell_command="echo", project_dir=str(project))
        # Entering the lifespan must not raise despite the regen failure.
        with TestClient(app):
            pass
