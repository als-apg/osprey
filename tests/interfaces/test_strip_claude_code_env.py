"""Tests for the shared ``CLAUDE_CODE_*`` env-strip helper.

Verifies that both the operator (SDK) path and the interactive PTY path strip
Claude Code internal session variables while preserving the telemetry master
switch (``CLAUDE_CODE_ENABLE_TELEMETRY``) and unrelated variables (including
``OTEL_*``, which does not carry the stripped prefix).

Also verifies the second deny step in the same shared helper: the sensitive
credential names owned by ``osprey.utils.sensitive_env`` are removed from every
child environment, while operational keys (``PATH``, ``OSPREY_WEB_PORT``)
survive.

The last section verifies the deliberate exception to that deny step: the
**panel token** is put back on both web-terminal child paths, because the
agent's MCP panel tools and its hooks are answered 401 without it. Those tests
drive the real seams — a real ``create_app`` for the credentials, the real
``_build_extra_env``/``build_pty_env`` pair for the PTY child, the real
``build_operator_child_env`` for the SDK child — rather than asserting against a
synthesised dict, because the defect they pin was that nothing connected the
seams, not that any one of them was wrong.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from osprey.agent_runner.clean_env import (
    build_base_child_env,
    build_clean_env,
    strip_claude_code_env,
)
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


def test_build_base_child_env_drops_sensitive_credentials(monkeypatch):
    """Subprocess children never inherit the shared sensitive-credential set."""
    monkeypatch.setenv("OSPREY_TERMINAL_SECRET", "operator-secret")
    monkeypatch.setenv("OSPREY_PANEL_TOKEN", "panel-token")
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "dispatcher-token")
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "launch-token")

    env = build_base_child_env()

    assert "OSPREY_TERMINAL_SECRET" not in env
    assert "OSPREY_PANEL_TOKEN" not in env
    assert "EVENT_DISPATCHER_TOKEN" not in env
    assert "BLUESKY_LAUNCH_TOKEN" not in env


def test_build_base_child_env_matches_launch_token_suffix(monkeypatch):
    """A future bridge following the ``*_LAUNCH_TOKEN`` convention is covered."""
    monkeypatch.setenv("SOME_FUTURE_BRIDGE_LAUNCH_TOKEN", "t")

    assert "SOME_FUTURE_BRIDGE_LAUNCH_TOKEN" not in build_base_child_env()


def test_sensitive_strip_preserves_operational_keys(monkeypatch):
    """PATH and OSPREY_WEB_PORT survive the sensitive strip."""
    monkeypatch.setenv("OSPREY_WEB_PORT", "8080")
    monkeypatch.setenv("OSPREY_TERMINAL_SECRET", "operator-secret")

    env = build_base_child_env()

    assert env["OSPREY_WEB_PORT"] == "8080"
    assert env.get("PATH"), "PATH must survive the strip"
    assert "OSPREY_TERMINAL_SECRET" not in env


def test_build_clean_env_drops_sensitive_credentials(monkeypatch):
    """The SDK/dispatch consumer inherits the base helper's deny step."""
    monkeypatch.setenv("OSPREY_TERMINAL_SECRET", "operator-secret")
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "dispatcher-token")

    env = build_clean_env()

    assert "OSPREY_TERMINAL_SECRET" not in env
    assert "EVENT_DISPATCHER_TOKEN" not in env


def test_build_pty_env_drops_sensitive_credentials(monkeypatch):
    """The interactive PTY path inherits the same deny step."""
    monkeypatch.setenv("OSPREY_PANEL_TOKEN", "panel-token")
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "launch-token")

    env = build_pty_env()

    assert "OSPREY_PANEL_TOKEN" not in env
    assert "BLUESKY_LAUNCH_TOKEN" not in env


def test_sensitive_strip_does_not_mutate_process_environment(monkeypatch):
    """Building a child env leaves this process's own environment intact."""
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "dispatcher-token")

    build_base_child_env()

    assert os.environ["EVENT_DISPATCHER_TOKEN"] == "dispatcher-token"


# -- the panel token: the one credential deliberately put back --------------


@pytest.fixture
def single_user_launch(monkeypatch):
    """A clean single-user launch: no carriers set, no credentials settled yet.

    ``setenv`` then ``delenv`` rather than a bare ``delenv``: monkeypatch only
    records an undo entry for a name it actually removed, so a key the code
    under test writes itself (``mint_and_announce`` publishes the operator
    secret) would otherwise survive into the next test.
    """
    from osprey.interfaces.web_auth import (
        BIND_HOST_ENV,
        OPERATOR_SECRET_ENV,
        PANEL_TOKEN_ENV,
        reset_web_credentials,
    )

    for name in (OPERATOR_SECRET_ENV, PANEL_TOKEN_ENV, BIND_HOST_ENV):
        monkeypatch.setenv(name, "__unset_by_test__")
        monkeypatch.delenv(name)
    reset_web_credentials()
    yield
    reset_web_credentials()


def test_pty_child_gets_the_panel_token_and_never_the_operator_secret(single_user_launch):
    """The interactive terminal's child holds the weak credential, not the strong one.

    ``build_pty_env`` hands its result to ``Popen(env=...)`` as the child's
    complete environment, so whatever is missing here is missing from the agent
    session and from every MCP server it starts. Before the panel token was
    overlaid, that meant every panel-tier call the agent made — the panel tools,
    the SessionStart context hook, the workspace-delta hook, the approval hook's
    focus POST — went out with no bearer and was answered 401 in silence.
    """
    from osprey.interfaces.web_auth import (
        OPERATOR_SECRET_ENV,
        PANEL_TOKEN_ENV,
        get_web_credentials,
        mint_and_announce,
    )
    from osprey.interfaces.web_terminal.app import create_app
    from osprey.interfaces.web_terminal.routes.websocket import _build_extra_env

    mint_and_announce("127.0.0.1", 8087)  # the launcher, publishing the carrier
    app = create_app()  # ... which becomes the server, closing it again
    credentials = get_web_credentials(app)

    extra_env = _build_extra_env(
        SimpleNamespace(app=app), claude_session_id=None, telemetry_session_id=None
    )
    env = build_pty_env(extra_env)

    assert env[PANEL_TOKEN_ENV] == credentials.panel_token
    assert OPERATOR_SECRET_ENV not in env
    assert credentials.operator_secret not in env.values()
    # Negative control: a gutted env would satisfy every absence above.
    assert env["TERM"] == "xterm-256color"


def test_sdk_operator_child_gets_the_panel_token_and_never_the_operator_secret(
    single_user_launch, tmp_path
):
    """The chat/operator SDK session's env carries the same weak credential.

    The SDK overlays this dict onto ``os.environ`` rather than replacing it, so
    the assertion about the operator secret is made against that merge — the
    absence of a key from the overlay would prove nothing on its own.
    """
    from osprey.interfaces.web_auth import (
        OPERATOR_SECRET_ENV,
        PANEL_TOKEN_ENV,
        get_web_credentials,
        mint_and_announce,
    )
    from osprey.interfaces.web_terminal.app import create_app
    from osprey.interfaces.web_terminal.operator_session import build_operator_child_env

    mint_and_announce("127.0.0.1", 8087)
    app = create_app()
    credentials = get_web_credentials(app)

    overlay = build_operator_child_env(str(tmp_path))
    sdk_child_env = {**os.environ, **overlay}

    assert overlay[PANEL_TOKEN_ENV] == credentials.panel_token
    assert OPERATOR_SECRET_ENV not in sdk_child_env
    assert credentials.operator_secret not in sdk_child_env.values()
    assert "PATH" in sdk_child_env


def test_both_web_terminal_child_paths_agree_on_the_panel_token(single_user_launch, tmp_path):
    """PTY and SDK hand the child the SAME token — the one the server verifies.

    Two seams re-introducing the credential independently is exactly the shape
    that drifts. Pinning them equal to each other *and* to the holder means a
    change to either one that invents its own value fails here.
    """
    from osprey.interfaces.web_auth import PANEL_TOKEN_ENV, get_web_credentials
    from osprey.interfaces.web_terminal.app import create_app
    from osprey.interfaces.web_terminal.operator_session import build_operator_child_env
    from osprey.interfaces.web_terminal.routes.websocket import _build_extra_env

    app = create_app()
    expected = get_web_credentials(app).panel_token

    pty = build_pty_env(_build_extra_env(SimpleNamespace(app=app), None, None))
    sdk = build_operator_child_env(str(tmp_path))

    assert pty[PANEL_TOKEN_ENV] == expected
    assert sdk[PANEL_TOKEN_ENV] == expected
