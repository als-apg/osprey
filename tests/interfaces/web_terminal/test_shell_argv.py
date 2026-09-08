"""``web_terminal.shell`` is argv in the ``--reload`` reader too.

``osprey web`` resolves the PTY argv in :mod:`osprey.cli.web_cmd`, but uvicorn's
factory bypass under ``--reload`` never reaches that code — the app's own
lifespan reads the key instead. Both readers must reach the same argv, or a
configured harness loses its arguments in exactly one of the two ways a
deployment can be started.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal import app as web_app


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "_agent_data").mkdir(exist_ok=True)
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _shell_command_for(project, monkeypatch, shell) -> list[str]:
    """Start the app with ``web_terminal.shell`` set to *shell*, return the argv."""
    monkeypatch.setattr(
        web_app,
        "_load_web_config",
        lambda *_a, **_k: {"shell": shell, "watch_dir": str(project / "_agent_data")},
    )
    monkeypatch.setattr(web_app, "_log_claude_cli_versions", lambda *_a, **_k: None)

    app = web_app.create_app(
        config_path=str(project / "config.yml"),
        project_dir=str(project),
    )
    with (
        patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/abs/harness"),
        TestClient(app),
    ):
        return list(app.state.shell_command)


def test_string_shell_keeps_its_arguments(project, monkeypatch):
    assert _shell_command_for(project, monkeypatch, "harness --profile ops") == [
        "/abs/harness",
        "--profile",
        "ops",
    ]


def test_list_shell_keeps_its_arguments(project, monkeypatch):
    assert _shell_command_for(project, monkeypatch, ["harness", "--profile", "ops"]) == [
        "/abs/harness",
        "--profile",
        "ops",
    ]
