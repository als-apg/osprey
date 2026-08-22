"""The Web Terminal starts on a project whose telemetry token is not issued yet.

The shipped telemetry block names ``${ZO_INGEST_SA_TOKEN}`` with no fallback,
and the store mints that token only when ``osprey up`` starts it. The lifespan
resolves the provider spec — which resolves the telemetry block with it — so
before the first deploy that read refuses and the server exits at startup:
"Application startup failed. Exiting.", on every launch, for a value no
operator can supply.

Same rule as ``osprey chat`` and the headless agent-run primitives: the
carve-out is exactly as wide as the store-issued registry. A credential an
operator does have to set still refuses the start, because a terminal serving
agents whose telemetry silently goes nowhere is the worse outcome.

Drives the real app factory through the TestClient lifespan; only
``inject_provider_env`` is stubbed, because it mutates this interpreter's own
``os.environ`` and these assertions are about what was resolved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from osprey.build import claude_code_resolver
from osprey.build.claude_code_telemetry import ObservabilityCredentialError
from osprey.cli.templates.manager import TemplateManager
from osprey.interfaces.web_terminal import app as web_app


@pytest.fixture(autouse=True)
def _no_ambient_store_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer machine that happens to export the token would hide the bug."""
    monkeypatch.delenv("ZO_INGEST_SA_TOKEN", raising=False)
    monkeypatch.delenv("OPERATOR_OTLP_SECRET", raising=False)


@pytest.fixture
def injected(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """The specs the lifespan handed to ``inject_provider_env``, injection stubbed."""
    seen: list[Any] = []

    def _record(env, spec, *, project_dir=None):
        seen.append(spec)
        return []

    monkeypatch.setattr(claude_code_resolver, "inject_provider_env", _record)
    return seen


def _project(tmp_path: Path, password: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A flat project whose telemetry block names *password* as its secret.

    Flat, so the render and the secrets zone coincide — the container shape,
    which is where the crash-loop was observed.
    """
    (tmp_path / "config.yml").write_text(
        "claude_code:\n"
        "  provider: anthropic\n"
        "  telemetry:\n"
        "    enabled: true\n"
        "    backend: openobserve\n"
        "    openobserve:\n"
        "      user: ingest@example.com\n"
        f"      password: {password}\n"
        "      org: default\n",
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
        shell_command="echo",
        project_dir=str(project),
    )
    with TestClient(app):
        pass


def test_the_token_under_test_is_really_store_issued() -> None:
    """Guards every assertion below from passing for the wrong reason."""
    from osprey.deployment.container_lifecycle import _STORE_ISSUED_VARS

    assert "ZO_INGEST_SA_TOKEN" in _STORE_ISSUED_VARS
    assert "OPERATOR_OTLP_SECRET" not in _STORE_ISSUED_VARS


def test_an_unissued_store_credential_serves_without_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, injected: list[Any]
) -> None:
    project = _project(tmp_path, "${ZO_INGEST_SA_TOKEN}", monkeypatch)

    _serve(project)

    assert injected, "the lifespan must resolve a provider spec"
    # Degraded, not deferred: an exporter without its auth header would post to
    # an auth-gated store and drop every span, so the whole block goes.
    assert "CLAUDE_CODE_ENABLE_TELEMETRY" not in injected[0].env_block
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in injected[0].env_block
    # The provider half of the same read is untouched — the terminal still routes.
    assert injected[0].env_block["ANTHROPIC_MODEL"]


def test_the_operator_is_told_which_verb_issues_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    injected: list[Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Silence would read as "this deployment has no telemetry configured"."""
    project = _project(tmp_path, "${ZO_INGEST_SA_TOKEN}", monkeypatch)

    with caplog.at_level("WARNING", logger="osprey.interfaces.web_terminal.app"):
        _serve(project)

    assert "ZO_INGEST_SA_TOKEN" in caplog.text
    assert "osprey up" in caplog.text


def test_an_operator_supplied_credential_still_refuses_to_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, injected: list[Any]
) -> None:
    """The carve-out reads the registry, not "unresolved" — this one is a real
    missing secret, and serving without it would hide a broken pipeline."""
    project = _project(tmp_path, "${OPERATOR_OTLP_SECRET}", monkeypatch)

    with pytest.raises(ObservabilityCredentialError):
        _serve(project)

    assert injected == []


def test_a_mixed_refusal_is_not_deferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, injected: list[Any]
) -> None:
    """One name in the set the operator does have to supply and the whole set
    stands refused — a server told "nothing to do here" about half of it leaves
    the other half to be discovered as a store that never authenticates."""
    project = _project(tmp_path, "${OPERATOR_OTLP_SECRET}${ZO_INGEST_SA_TOKEN}", monkeypatch)

    with pytest.raises(ObservabilityCredentialError) as caught:
        _serve(project)

    assert caught.value.unresolved_vars == ("OPERATOR_OTLP_SECRET", "ZO_INGEST_SA_TOKEN")
    assert injected == []


def test_a_blank_credential_still_refuses_to_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, injected: list[Any]
) -> None:
    """No variable is named at all, so there is nothing a deploy could issue."""
    project = _project(tmp_path, '""', monkeypatch)

    with pytest.raises(ObservabilityCredentialError):
        _serve(project)

    assert injected == []


def test_a_resolvable_token_keeps_telemetry_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, injected: list[Any]
) -> None:
    """The deferral is about absence only — once the deploy has passed the token
    into the container, the same config resolves and the terminal exports."""
    project = _project(tmp_path, "${ZO_INGEST_SA_TOKEN}", monkeypatch)
    (project / ".env").write_text("ZO_INGEST_SA_TOKEN=issued-by-the-store\n", encoding="utf-8")

    _serve(project)

    assert injected[0].env_block["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert "Authorization=Basic " in injected[0].env_block["OTEL_EXPORTER_OTLP_HEADERS"]
