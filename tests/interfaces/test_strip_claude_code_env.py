"""Tests for the shared ``CLAUDE_CODE_*`` env-strip helper.

Verifies that both the operator (SDK) path and the interactive PTY path strip
Claude Code internal session variables while preserving the telemetry master
switch (``CLAUDE_CODE_ENABLE_TELEMETRY``) and unrelated variables (including
``OTEL_*``, which does not carry the stripped prefix).
"""

from __future__ import annotations

from osprey.interfaces.web_terminal.env_utils import strip_claude_code_env
from osprey.interfaces.web_terminal.operator_session import build_clean_env
from osprey.interfaces.web_terminal.pty_manager import build_pty_env


def test_strips_internal_session_markers():
    """Keys with the CLAUDECODE / CLAUDE_CODE_ prefixes are removed."""
    result = strip_claude_code_env(
        {
            "CLAUDE_CODE_FOO": "1",
            "CLAUDE_CODE_ENTRYPOINT": "cli",
            "CLAUDECODE": "1",
            "PATH": "/usr/bin",
        }
    )
    assert "CLAUDE_CODE_FOO" not in result
    assert "CLAUDE_CODE_ENTRYPOINT" not in result
    assert "CLAUDECODE" not in result
    assert result["PATH"] == "/usr/bin"


def test_preserves_telemetry_master_switch():
    """The telemetry master switch survives the strip on both paths."""
    result = strip_claude_code_env(
        {
            "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
            "CLAUDE_CODE_FOO": "1",
        }
    )
    assert result["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert "CLAUDE_CODE_FOO" not in result


def test_preserves_otel_and_unrelated_keys():
    """OTEL_* keys (no stripped prefix) and unrelated vars pass through."""
    result = strip_claude_code_env(
        {
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4317",
            "OTEL_METRIC_EXPORT_INTERVAL": "5000",
            "HOME": "/home/op",
            "CLAUDE_CODE_FOO": "1",
        }
    )
    assert result["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://collector:4317"
    assert result["OTEL_METRIC_EXPORT_INTERVAL"] == "5000"
    assert result["HOME"] == "/home/op"
    assert "CLAUDE_CODE_FOO" not in result


def test_returns_a_copy():
    """The helper does not mutate its input mapping."""
    source = {"CLAUDE_CODE_FOO": "1", "PATH": "/usr/bin"}
    result = strip_claude_code_env(source)
    assert result is not source
    assert "CLAUDE_CODE_FOO" in source  # original untouched


def test_build_clean_env_retains_telemetry_switch(monkeypatch):
    """The operator (SDK) path retains the telemetry master switch."""
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_TELEMETRY", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")

    env = build_clean_env()

    assert env["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert "CLAUDE_CODE_ENTRYPOINT" not in env
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://collector:4317"


def test_build_pty_env_retains_telemetry_switch(monkeypatch):
    """The interactive PTY path retains the telemetry master switch."""
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_TELEMETRY", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")

    env = build_pty_env()

    assert env["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert "CLAUDE_CODE_ENTRYPOINT" not in env
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://collector:4317"
    # Behavior preserved: terminal type vars still set.
    assert env["TERM"] == "xterm-256color"
    assert env["COLORTERM"] == "truecolor"


def test_build_pty_env_applies_extra_env_last(monkeypatch):
    """Caller-supplied extra_env overlays the resolved environment."""
    monkeypatch.setenv("CLAUDE_CODE_FOO", "1")
    env = build_pty_env({"MY_VAR": "hello", "TERM": "dumb"})
    assert env["MY_VAR"] == "hello"
    assert env["TERM"] == "dumb"  # extra_env wins over the default
    assert "CLAUDE_CODE_FOO" not in env
