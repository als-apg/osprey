"""The Web Terminal's agents export to the agent record under the terminal's login.

The lifespan resolves the provider spec — which resolves the OTLP block with
it — from the container's environment. With the record's address and a record
login, the terminal's agents send their log events to the record's
``/v1/logs``; without a login the terminal still starts and its agents export
nothing. A profile ``claude_code.telemetry`` block chooses neither.

Drives the real app factory through the TestClient lifespan; only
``inject_provider_env`` is stubbed, because it mutates this interpreter's own
``os.environ`` and these assertions are about what was resolved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from osprey.agent_runner import provider_env
from osprey.cli.templates.manager import TemplateManager
from osprey.interfaces.web_terminal import app as web_app


@pytest.fixture(autouse=True)
def _a_terminal_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host-networked terminal: the record on loopback, no ambient login."""
    monkeypatch.delenv("OSPREY_AGENT_RECORD_LOGIN", raising=False)
    monkeypatch.delenv("OSPREY_AGENT_RECORD_OFF", raising=False)
    monkeypatch.setenv("OSPREY_AGENT_RECORD_URL", "http://127.0.0.1:10061")


@pytest.fixture
def injected(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """The specs the lifespan handed to ``inject_provider_env``, injection stubbed."""
    seen: list[Any] = []

    # ``inject_provider_env``'s signature: the lifespan names ``project_dir``.
    def _record(_env, spec, *, project_dir=None):  # noqa: ARG001
        seen.append(spec)
        return []

    monkeypatch.setattr(provider_env, "inject_provider_env", _record)
    return seen


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A flat project whose telemetry block names the deployment's gateway.

    Flat, so the render and the secrets zone coincide — the container shape.
    """
    (tmp_path / "config.yml").write_text(
        "claude_code:\n"
        "  provider: anthropic\n"
        "  telemetry:\n"
        "    enabled: true\n"
        "    backend: gateway\n",
        encoding="utf-8",
    )
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
        shell_command=["echo"],
        project_dir=str(project),
    )
    with TestClient(app):
        pass


def test_a_missing_login_starts_and_exports_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, injected: list[Any]
) -> None:
    """Without a record login the terminal serves, and its agents send nowhere."""
    project = _project(tmp_path, monkeypatch)

    _serve(project)

    env = injected[0].env_block
    assert env["CLAUDE_CODE_ENABLE_TELEMETRY"] == "0"
    assert env["OTEL_LOGS_EXPORTER"] == "none"
    assert env["OTEL_METRICS_EXPORTER"] == "none"
    assert env["OTEL_TRACES_EXPORTER"] == "none"
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in env
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in env


def test_a_login_exports_to_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, injected: list[Any]
) -> None:
    """The login reaches the container by its own env_file; with it the terminal exports."""
    project = _project(tmp_path, monkeypatch)
    monkeypatch.setenv("OSPREY_AGENT_RECORD_LOGIN", "alice:s3cret")

    _serve(project)

    env = injected[0].env_block
    assert env["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://127.0.0.1:10061"
    assert env["OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"] == "http://127.0.0.1:10061/v1/logs"
    assert "Authorization=Basic " in env["OTEL_EXPORTER_OTLP_HEADERS"]
