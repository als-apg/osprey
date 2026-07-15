"""Tests for the C3 loopback-bind enforcement in `osprey web`.

Per the multi-user compose (`docker-compose.web.yml.j2`), every per-user
container declares `OSPREY_TERMINAL_BIND_HOST=127.0.0.1` so nginx is the
ONLY off-host path (criterion C3). Nothing previously read that env var, and
a legacy image CMD passing `--host 0.0.0.0` would silently punch through the
reverse-proxy chokepoint. `resolve_bind_host()` makes the declared env
authoritative over both `--host` and config, while leaving single-user
`osprey web` (no declared env) free to honor `--host 0.0.0.0` verbatim.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from osprey.cli.web_cmd import DECLARED_BIND_ENV, resolve_bind_host, web


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def _isolate_bind_and_port_env(monkeypatch):
    """Start each test from a clean slate for both env vars under test.

    The foreground `web()` path unconditionally does a REAL
    ``os.environ["OSPREY_WEB_PORT"] = str(port)`` (for child PTY/MCP
    processes) — not through ``monkeypatch``. A plain
    ``monkeypatch.delenv(key, raising=False)`` on a key that's already absent
    records NO undo entry, so that later direct mutation is never rolled
    back and leaks into subsequent tests. Forcing a ``setenv`` first
    guarantees monkeypatch tracks the key and restores the true pre-test
    state (present or absent) at teardown, regardless of what the app wrote
    in between.
    """
    for _key in ("OSPREY_WEB_PORT", DECLARED_BIND_ENV):
        monkeypatch.setenv(_key, "__unset_by_test_fixture__")
        monkeypatch.delenv(_key)


def _free_port() -> int:
    """Reserve then release an OS-assigned port so nothing is listening on it."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestResolveBindHost:
    """Pure resolver: declared env authoritative; CLI/config only as fallback."""

    def test_declared_bind_env_overrides_explicit_host(self):
        assert resolve_bind_host("0.0.0.0", None, {DECLARED_BIND_ENV: "127.0.0.1"}) == "127.0.0.1"

    def test_single_user_host_0000_honored_without_env(self):
        assert resolve_bind_host("0.0.0.0", None, {}) == "0.0.0.0"

    def test_deliberate_public_optout(self):
        """A deployment CAN declare 0.0.0.0 itself — the invariant is
        "declared wins", not "loopback is force-pinned no matter what"."""
        assert resolve_bind_host("127.0.0.1", None, {DECLARED_BIND_ENV: "0.0.0.0"}) == "0.0.0.0"

    def test_falls_back_to_config_then_default(self):
        assert resolve_bind_host(None, "10.0.0.5", {}) == "10.0.0.5"
        assert resolve_bind_host(None, None, {}) == "127.0.0.1"


class TestWebCommandHonorsDeclaredBindEnv:
    """The load-bearing wiring guard: the reconciled host must actually reach
    the server entrypoint, not just the pure resolver."""

    def _stub_launch(self, monkeypatch):
        monkeypatch.setattr("osprey.interfaces.web_terminal.run_web", lambda **_kw: None)
        monkeypatch.setattr("osprey.mcp_env.load_dotenv_from_project", lambda: None)

    def test_multiuser_env_pins_loopback_reaches_run_web(self, runner, monkeypatch):
        """The scenario this whole fix exists for: a stale/hostile image CMD
        passes --host 0.0.0.0, but the multi-user container has declared
        OSPREY_TERMINAL_BIND_HOST=127.0.0.1. The host that reaches run_web
        must be 127.0.0.1, NOT 0.0.0.0 — otherwise nginx is no longer the
        only off-host path."""
        monkeypatch.setenv(DECLARED_BIND_ENV, "127.0.0.1")
        captured = {}

        def _fake_run_web(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr("osprey.interfaces.web_terminal.run_web", _fake_run_web)
        monkeypatch.setattr("osprey.mcp_env.load_dotenv_from_project", lambda: None)

        result = runner.invoke(
            web,
            [
                "--host",
                "0.0.0.0",
                "--port",
                str(_free_port()),
                "--shell",
                "true",
                "--skip-preflight",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert captured.get("host") == "127.0.0.1"

    def test_notice_printed_when_declared_env_overrides_flag(self, runner, monkeypatch):
        monkeypatch.setenv(DECLARED_BIND_ENV, "127.0.0.1")
        self._stub_launch(monkeypatch)

        result = runner.invoke(
            web,
            [
                "--host",
                "0.0.0.0",
                "--port",
                str(_free_port()),
                "--shell",
                "true",
                "--skip-preflight",
            ],
            catch_exceptions=False,
        )

        assert "NOTICE" in result.output
        assert DECLARED_BIND_ENV in result.output

    def test_single_user_no_env_keeps_0000(self, runner, monkeypatch):
        """Without the declared env (single-user `osprey web`), --host 0.0.0.0
        must still work exactly as before."""
        monkeypatch.delenv(DECLARED_BIND_ENV, raising=False)
        captured = {}

        def _fake_run_web(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr("osprey.interfaces.web_terminal.run_web", _fake_run_web)
        monkeypatch.setattr("osprey.mcp_env.load_dotenv_from_project", lambda: None)

        result = runner.invoke(
            web,
            [
                "--host",
                "0.0.0.0",
                "--port",
                str(_free_port()),
                "--shell",
                "true",
                "--skip-preflight",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert captured.get("host") == "0.0.0.0"


class TestPortEnvvar:
    """OSPREY_WEB_PORT fills in when --port is absent; --port still wins."""

    def _stub_launch(self, monkeypatch):
        monkeypatch.setattr("osprey.interfaces.web_terminal.run_web", lambda **_kw: None)
        monkeypatch.setattr("osprey.mcp_env.load_dotenv_from_project", lambda: None)
        monkeypatch.delenv("OSPREY_TERMINAL_BIND_HOST", raising=False)

    def test_env_port_honored_when_flag_absent(self, runner, monkeypatch):
        self._stub_launch(monkeypatch)
        env_port = _free_port()
        monkeypatch.setenv("OSPREY_WEB_PORT", str(env_port))
        captured = {}
        monkeypatch.setattr(
            "osprey.interfaces.web_terminal.run_web", lambda **kw: captured.update(kw)
        )

        result = runner.invoke(web, ["--shell", "true", "--skip-preflight"], catch_exceptions=False)

        assert result.exit_code == 0
        assert captured.get("port") == env_port

    def test_explicit_port_flag_wins_over_env(self, runner, monkeypatch):
        self._stub_launch(monkeypatch)
        monkeypatch.setenv("OSPREY_WEB_PORT", str(_free_port()))
        flag_port = _free_port()
        captured = {}
        monkeypatch.setattr(
            "osprey.interfaces.web_terminal.run_web", lambda **kw: captured.update(kw)
        )

        result = runner.invoke(
            web,
            ["--port", str(flag_port), "--shell", "true", "--skip-preflight"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert captured.get("port") == flag_port
