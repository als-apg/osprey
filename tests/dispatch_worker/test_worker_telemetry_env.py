"""The dispatch worker's agent receives the tracing and tool-content switches.

The worker resolves its provider env through the same resolver every launch
path uses, then hands the SDK ``build_clean_env()``; the SDK builds the CLI's
environment as ``os.environ`` minus ``CLAUDECODE``, overlaid with that dict.
These tests pin that the telemetry block the resolver emits reaches the CLI
through both steps.
"""

from __future__ import annotations

import os

from osprey.agent_runner.clean_env import build_clean_env
from osprey.build.claude_code_telemetry import TELEMETRY_ENV_VARS
from osprey.mcp_server.dispatch_worker import dispatch_api

_TRACING_CONFIG = """\
claude_code:
  provider: anthropic
  telemetry:
    enabled: true
    backend: openobserve
    openobserve:
      user: root@example.com
      password: secret
    content_max_length: 262144
"""

_EXPECTED = {
    "OTEL_TRACES_EXPORTER": "otlp",
    "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",
    "OTEL_LOG_TOOL_CONTENT": "1",
    "CLAUDE_CODE_OTEL_CONTENT_MAX_LENGTH": "262144",
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
            "OSPREY_OTEL_OPENOBSERVE_HOST": "openobserve",
            "OSPREY_OTEL_OPENOBSERVE_PORT": "5080",
        }
    )
    monkeypatch.setattr(os, "environ", fake)
    return fake


def test_worker_sdk_env_carries_traces_and_tool_content(tmp_path, monkeypatch):
    """The resolver's tracing block survives into the env the SDK hands the CLI."""
    render = tmp_path / "build"
    render.mkdir()
    (render / "config.yml").write_text(_TRACING_CONFIG)
    environ = _isolated_environ(monkeypatch, tmp_path)

    dispatch_api._inject_provider_env_once()

    for name, value in _EXPECTED.items():
        assert environ[name] == value, name
    assert environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://openobserve:5080/api/default"

    cli_env = {
        **{k: v for k, v in environ.items() if k != "CLAUDECODE"},
        **build_clean_env(),
    }
    for name, value in _EXPECTED.items():
        assert cli_env[name] == value, name
