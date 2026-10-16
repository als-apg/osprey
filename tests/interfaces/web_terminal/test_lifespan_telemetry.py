"""The Web Terminal refuses to start without its telemetry gateway login.

The lifespan resolves the provider spec — which resolves the telemetry block
with it — so a terminal whose container carries no ``OSPREY_TELEMETRY_LOGIN``
refuses at startup rather than serving agents whose telemetry the gateway
would drop. With the login, the same config resolves and the terminal exports.

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
from osprey.build.claude_code_telemetry import ObservabilityCredentialError
from osprey.cli.templates.manager import TemplateManager
from osprey.interfaces.web_terminal import app as web_app


@pytest.fixture(autouse=True)
def _a_terminal_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host-networked terminal: no ambient login, the gateway on loopback."""
    monkeypatch.delenv("OSPREY_TELEMETRY_LOGIN", raising=False)
    monkeypatch.setenv("OSPREY_OTEL_GATEWAY_HOST", "127.0.0.1")
    monkeypatch.setenv("OSPREY_OTEL_GATEWAY_PORT", "10051")


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
    """A flat project whose telemetry block sends to the deployment's gateway.

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


def test_a_missing_login_refuses_to_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, injected: list[Any]
) -> None:
    """Serving without the login would hide a broken pipeline behind a healthy page."""
    project = _project(tmp_path, monkeypatch)

    with pytest.raises(ObservabilityCredentialError) as caught:
        _serve(project)

    assert caught.value.unresolved_vars == ("OSPREY_TELEMETRY_LOGIN",)
    assert injected == []


def test_a_login_keeps_telemetry_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, injected: list[Any]
) -> None:
    """The login reaches the container by its own env_file; with it the terminal exports."""
    project = _project(tmp_path, monkeypatch)
    monkeypatch.setenv("OSPREY_TELEMETRY_LOGIN", "alice:s3cret")

    _serve(project)

    env = injected[0].env_block
    assert env["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://127.0.0.1:10051"
    assert "Authorization=Basic " in env["OTEL_EXPORTER_OTLP_HEADERS"]
