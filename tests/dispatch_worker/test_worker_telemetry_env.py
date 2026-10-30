"""The dispatch worker's agent receives the telemetry block the resolver emits.

The worker resolves its provider env through the same resolver every launch
path uses, then hands the SDK ``build_clean_env()``; the SDK builds the CLI's
environment as ``os.environ`` minus ``CLAUDECODE``, overlaid with that dict.
These tests pin that the record's OTLP block the resolver emits reaches the CLI
through both steps.
"""

from __future__ import annotations

import os

from osprey.agent_runner.clean_env import build_clean_env
from osprey.build.claude_code_telemetry import (
    CONTENT_GATES,
    LOGS_PATH,
    RECORD_LOGIN_ENV,
    RECORD_URL_ENV,
    TELEMETRY_ENV_VARS,
)
from osprey.mcp_server.dispatch_worker import dispatch_api

_CONFIG = """\
claude_code:
  provider: anthropic
"""

_RECORD_URL = "http://agent-record:4318"

_EXPECTED = {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "OTEL_LOGS_EXPORTER": "otlp",
    "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT": f"{_RECORD_URL}{LOGS_PATH}",
    "OTEL_METRICS_EXPORTER": "none",
    "OTEL_TRACES_EXPORTER": "none",
    "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "0",
    **dict.fromkeys(CONTENT_GATES, "0"),
}


def _isolated_environ(monkeypatch, tmp_path) -> dict[str, str]:
    """Swap ``os.environ`` for a copy free of inherited telemetry state."""
    fake = {k: v for k, v in os.environ.items() if k not in TELEMETRY_ENV_VARS}
    fake.pop("OSPREY_CONFIG", None)
    fake.pop("ANTHROPIC_BASE_URL", None)
    fake.update(
        {
            "OSPREY_PROJECT_DIR": str(tmp_path),
            "CONFIG_FILE": str(tmp_path / "build" / "config.yml"),
            "ANTHROPIC_API_KEY": "sk-ant",
            RECORD_URL_ENV: _RECORD_URL,
            RECORD_LOGIN_ENV: "dispatch-worker-1:s3cret",
        }
    )
    monkeypatch.setattr(os, "environ", fake)
    return fake


def _render(tmp_path) -> None:
    render = tmp_path / "build"
    render.mkdir()
    (render / "config.yml").write_text(_CONFIG)


def _cli_env(environ: dict[str, str]) -> dict[str, str]:
    """The env the SDK hands the CLI: ``os.environ`` minus ``CLAUDECODE``, overlaid."""
    return {
        **{k: v for k, v in environ.items() if k != "CLAUDECODE"},
        **build_clean_env(),
    }


def test_worker_sdk_env_sends_logs_to_the_record(tmp_path, monkeypatch):
    """The resolver's record block survives into the env the SDK hands the CLI."""
    _render(tmp_path)
    environ = _isolated_environ(monkeypatch, tmp_path)

    dispatch_api._inject_provider_env_once()

    for name, value in _EXPECTED.items():
        assert environ[name] == value, name
    assert environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == _RECORD_URL

    cli_env = _cli_env(environ)
    for name, value in _EXPECTED.items():
        assert cli_env[name] == value, name


def test_worker_exports_traces_and_content_as_off_over_a_stale_export(tmp_path, monkeypatch):
    """An inherited trace exporter or content gate reaches the CLI switched off."""
    _render(tmp_path)
    environ = _isolated_environ(monkeypatch, tmp_path)
    environ["OTEL_TRACES_EXPORTER"] = "otlp"
    environ["CLAUDE_CODE_ENHANCED_TELEMETRY_BETA"] = "1"
    environ["OTEL_LOG_TOOL_CONTENT"] = "1"

    dispatch_api._inject_provider_env_once()

    cli_env = _cli_env(environ)
    assert cli_env["OTEL_LOGS_EXPORTER"] == "otlp"
    assert cli_env["OTEL_TRACES_EXPORTER"] == "none"
    assert cli_env["CLAUDE_CODE_ENHANCED_TELEMETRY_BETA"] == "0"
    assert cli_env["OTEL_LOG_TOOL_CONTENT"] == "0"
