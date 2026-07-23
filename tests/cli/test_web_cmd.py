"""Tests for osprey web CLI command (detach/stop and backward compat)."""

from __future__ import annotations

import os
import signal
import socket
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from osprey.cli.web_cmd import (
    LOG_FILE,
    PID_FILE,
    _preflight,
    _read_pid,
    _resolve_web_shell_command,
    _start_detached,
    _wait_for_server,
    _write_pid,
    web,
)


@pytest.fixture
def runner():
    return CliRunner()


# -- shell-command resolution ---------------------------------------------


class TestResolveWebShellCommand:
    """_resolve_web_shell_command wires build_claude_launch_argv into the PTY."""

    @patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/abs/claude")
    def test_default_resolves_path_and_keeps_setting_sources(self, mock_resolve):
        """No pin, no override: bare ``claude`` is resolved to an absolute path
        AND the launcher's --setting-sources project flag is preserved.

        Regression guard: an earlier ``len(argv) == 1`` heuristic dropped into
        the pinned branch once the launcher started appending flags, leaving
        ``claude`` unresolved on a stripped PATH and silently discarding the
        provider-isolation flag.
        """
        cmd = _resolve_web_shell_command({}, None, {})

        assert cmd == ["/abs/claude", "--setting-sources", "project"]
        mock_resolve.assert_called_once_with("claude")

    @patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/abs/claude")
    def test_pinned_left_to_path_lookup(self, mock_resolve):
        """A cli_version pin yields the npx prefix, unresolved, flag preserved."""
        cmd = _resolve_web_shell_command({"cli_version": "2.1.146"}, None, {})

        assert cmd == [
            "npx",
            "-y",
            "@anthropic-ai/claude-code@2.1.146",
            "--setting-sources",
            "project",
        ]
        mock_resolve.assert_not_called()

    @patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/abs/custom")
    def test_shell_override_defeats_pin(self, mock_resolve):
        cmd = _resolve_web_shell_command({"cli_version": "2.1.146"}, "my-shell", {})

        assert cmd == ["/abs/custom"]
        mock_resolve.assert_called_once_with("my-shell")


# -- help / backward compat ------------------------------------------------


def test_web_help_shows_detach_and_stop(runner: CliRunner):
    result = runner.invoke(web, ["--help"])
    assert result.exit_code == 0
    assert "--detach" in result.output
    assert "stop" in result.output


# -- _read_pid --------------------------------------------------------------


def test_read_pid_missing(tmp_path: Path):
    assert _read_pid(tmp_path) is None


def test_read_pid_valid(tmp_path: Path):
    pid = os.getpid()  # current process — guaranteed alive
    (tmp_path / PID_FILE).write_text(str(pid))
    assert _read_pid(tmp_path) == pid


def test_read_pid_stale(tmp_path: Path):
    (tmp_path / PID_FILE).write_text("999999999")
    with patch("osprey.cli.web_cmd.os.kill", side_effect=ProcessLookupError):
        result = _read_pid(tmp_path)
    assert result is None
    assert not (tmp_path / PID_FILE).exists()


def test_read_pid_corrupt(tmp_path: Path):
    (tmp_path / PID_FILE).write_text("not-a-number")
    assert _read_pid(tmp_path) is None
    assert not (tmp_path / PID_FILE).exists()


# -- _write_pid -------------------------------------------------------------


def test_write_pid(tmp_path: Path):
    _write_pid(tmp_path, 42)
    assert (tmp_path / PID_FILE).read_text() == "42"


# -- _wait_for_server -------------------------------------------------------


def test_wait_for_server_success():
    proc = MagicMock()
    proc.poll.return_value = None

    with patch("osprey.cli.web_cmd.socket.create_connection") as mock_conn:
        mock_conn.return_value.__enter__ = MagicMock()
        mock_conn.return_value.__exit__ = MagicMock()
        assert _wait_for_server("127.0.0.1", 8087, proc, timeout=2.0) is True


def test_wait_for_server_timeout():
    proc = MagicMock()
    proc.poll.return_value = None

    with patch("osprey.cli.web_cmd.socket.create_connection", side_effect=OSError):
        assert _wait_for_server("127.0.0.1", 8087, proc, timeout=0.5) is False


def test_wait_for_server_early_crash():
    proc = MagicMock()
    proc.poll.return_value = 1  # process already exited

    assert _wait_for_server("127.0.0.1", 8087, proc, timeout=5.0) is False


# -- detach -----------------------------------------------------------------


@patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/bin/fake-claude")
@patch("osprey.cli.web_cmd._preflight", return_value=([], []))
@patch("osprey.cli.web_cmd._wait_for_server", return_value=True)
@patch("osprey.cli.web_cmd.subprocess.Popen")
@patch("osprey.cli.web_cmd.get_config_value", return_value={})
def test_detach_spawns_subprocess(
    mock_config, mock_popen, mock_wait, mock_preflight, mock_resolve, tmp_path, runner
):
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_popen.return_value = mock_proc

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(web, ["--detach"])

    assert result.exit_code == 0
    mock_popen.assert_called_once()
    call_kwargs = mock_popen.call_args
    assert call_kwargs.kwargs["start_new_session"] is True


@patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/bin/fake-claude")
@patch("osprey.cli.web_cmd._preflight", return_value=([], []))
@patch("osprey.cli.web_cmd._wait_for_server", return_value=True)
@patch("osprey.cli.web_cmd.subprocess.Popen")
@patch("osprey.cli.web_cmd.get_config_value", return_value={})
def test_detach_writes_pid_file(
    mock_config, mock_popen, mock_wait, mock_preflight, mock_resolve, tmp_path, runner
):
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_popen.return_value = mock_proc

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        result = runner.invoke(web, ["--detach"])
        pid_content = (Path(td) / PID_FILE).read_text()

    assert result.exit_code == 0
    assert pid_content == "12345"


@patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/bin/fake-claude")
@patch("osprey.cli.web_cmd._preflight", return_value=([], []))
@patch("osprey.cli.web_cmd._read_pid", return_value=99999)
@patch("osprey.cli.web_cmd.get_config_value", return_value={})
def test_detach_idempotent_when_running(
    mock_config, mock_read_pid, mock_preflight, mock_resolve, runner, tmp_path
):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(web, ["--detach"])

    assert result.exit_code == 0
    assert "already running" in result.output


@patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/bin/fake-claude")
@patch("osprey.cli.web_cmd._preflight", return_value=([], []))
@patch("osprey.cli.web_cmd._wait_for_server", return_value=True)
@patch("osprey.cli.web_cmd.subprocess.Popen")
@patch("osprey.cli.web_cmd.get_config_value", return_value={})
def test_detach_cleans_stale_pid(
    mock_config, mock_popen, mock_wait, mock_preflight, mock_resolve, tmp_path, runner
):
    mock_proc = MagicMock()
    mock_proc.pid = 55555
    mock_popen.return_value = mock_proc

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        # Write a stale PID file
        (Path(td) / PID_FILE).write_text("999999999")

        # _read_pid will call os.kill which raises ProcessLookupError for stale
        with patch("osprey.cli.web_cmd.os.kill", side_effect=ProcessLookupError):
            result = runner.invoke(web, ["--detach"])

    assert result.exit_code == 0
    # Should have started a new server
    mock_popen.assert_called_once()


@patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/bin/fake-claude")
@patch("osprey.cli.web_cmd._preflight", return_value=([], []))
@patch("osprey.cli.web_cmd._wait_for_server", return_value=True)
@patch("osprey.cli.web_cmd.subprocess.Popen")
@patch("osprey.cli.web_cmd.get_config_value", return_value={})
def test_detach_shows_url_and_pid(
    mock_config, mock_popen, mock_wait, mock_preflight, mock_resolve, tmp_path, runner
):
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_popen.return_value = mock_proc

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(web, ["--detach"])

    assert "PID 12345" in result.output
    assert "http://127.0.0.1:8087" in result.output
    assert "osprey web stop" in result.output


# -- stop -------------------------------------------------------------------


def test_stop_kills_process(tmp_path, runner):
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        (Path(td) / PID_FILE).write_text("12345")
        (Path(td) / LOG_FILE).write_text("some log")

        with patch("osprey.cli.web_cmd.os.kill") as mock_kill:
            result = runner.invoke(web, ["stop"])

        mock_kill.assert_called_once_with(12345, signal.SIGTERM)
        assert result.exit_code == 0
        assert "Stopped" in result.output
        assert not (Path(td) / PID_FILE).exists()
        assert not (Path(td) / LOG_FILE).exists()


def test_stop_no_pid_file(tmp_path, runner):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(web, ["stop"])

    assert result.exit_code == 0
    assert "No running" in result.output


def test_stop_stale_pid(tmp_path, runner):
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        (Path(td) / PID_FILE).write_text("999999999")

        with patch("osprey.cli.web_cmd.os.kill", side_effect=ProcessLookupError):
            result = runner.invoke(web, ["stop"])

        assert result.exit_code == 0
        assert "not found" in result.output
        assert not (Path(td) / PID_FILE).exists()


# -- pre-flight (Task 1.1: --skip-preflight, _preflight, Probe 1) -----------
#
# Probe 1 is the companion port-collision guard: before binding its own port,
# `osprey web` TCP-connect-probes every companion panel port the lifespan
# will actually launch (see `_load_panel_config` / `_make_auto_launch_checker`
# / `_make_config_reader`). A listener already on one of those ports is
# foreign — at best it steals a panel's tab, at worst it silently proxies
# another project's data into this UI.


def _free_port() -> int:
    """Reserve then release an OS-assigned port so nothing is listening on it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _hold_port(port: int, host: str = "127.0.0.1") -> socket.socket:
    """Bind and listen on *port* so it looks taken to a connect-probe."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(1)
    return sock


def _patch_config(monkeypatch, config: dict) -> None:
    """Point every load_osprey_config() call site the pre-flight touches at *config*.

    `_load_panel_config()` (web_terminal/app.py) does a per-call lazy import of
    `osprey.utils.workspace.load_osprey_config`, while `server_launcher.py`
    binds it once at module-import time — both need patching independently to
    control panel/port resolution deterministically and offline.
    """
    monkeypatch.setattr("osprey.utils.workspace.load_osprey_config", lambda: config)
    monkeypatch.setattr("osprey.infrastructure.server_launcher.load_osprey_config", lambda: config)


def _spy_config_readers(monkeypatch) -> list[str]:
    """Wrap server_launcher._make_config_reader to record which config_key gets probed.

    Used to prove an excluded panel's port is never even resolved (as opposed
    to resolved-but-not-held) — avoids depending on the framework's default
    companion ports (8085/8086/8092/...) actually being free on the test host,
    which they may not be if a real `osprey web` happens to be running there.
    """
    from osprey.infrastructure import server_launcher

    probed_keys: list[str] = []
    original_make_reader = server_launcher._make_config_reader

    def _wrapping_make_reader(defn):
        reader = original_make_reader(defn)

        def _wrapped():
            probed_keys.append(defn.config_key)
            return reader()

        return _wrapped

    monkeypatch.setattr(server_launcher, "_make_config_reader", _wrapping_make_reader)
    return probed_keys


class TestPreflightCompanionPortCollision:
    """Probe 1: held companion ports are reported; free/excluded ports are not."""

    def test_held_artifact_port_reported(self, monkeypatch):
        """A companion port the lifespan WILL launch, already held, is a finding."""
        port = _free_port()
        _patch_config(monkeypatch, {"artifact_server": {"port": port}})

        held = _hold_port(port)
        try:
            failures, warnings = _preflight({}, None, "127.0.0.1", 8087)
        finally:
            held.close()

        assert len(failures) == 1
        assert str(port) in failures[0]
        assert "artifact" in failures[0]
        assert f"lsof -i :{port}" in failures[0]
        assert warnings == []

    def test_free_ports_clean_pass(self, monkeypatch):
        """With no companion ports held, _preflight returns no findings."""
        port = _free_port()
        _patch_config(monkeypatch, {"artifact_server": {"port": port}})

        assert _preflight({}, None, "127.0.0.1", 8087) == ([], [])

    def test_require_section_unmet_panel_excluded(self, monkeypatch):
        """Enabled-but-not-launched panel (require_section unmet) is not probed."""
        # "channel-finder" is enabled in web.panels, but the channel_finder
        # top-level section (which gates auto_launch/require_section) is
        # absent — the lifespan would never actually call
        # _launch_channel_finder_server, so its port must never be resolved.
        artifact_port = _free_port()
        _patch_config(
            monkeypatch,
            {
                "artifact_server": {"port": artifact_port},
                "web": {"panels": {"channel-finder": True}},
            },
        )
        probed_keys = _spy_config_readers(monkeypatch)

        failures, warnings = _preflight({}, None, "127.0.0.1", 8087)

        assert failures == []
        assert warnings == []
        assert "channel_finder" not in probed_keys

    def test_disabled_panel_excluded(self, monkeypatch):
        """A panel absent from web.panels is not probed even if its port is held."""
        artifact_port = _free_port()
        _patch_config(
            monkeypatch, {"artifact_server": {"port": artifact_port}}
        )  # ariel not enabled
        probed_keys = _spy_config_readers(monkeypatch)

        failures, warnings = _preflight({}, None, "127.0.0.1", 8087)

        assert failures == []
        assert warnings == []
        assert "ariel" not in probed_keys

    def test_no_network_or_registry_calls(self, monkeypatch):
        """Probe 1 never starts a companion server or hits its /health endpoint."""
        port = _free_port()
        _patch_config(monkeypatch, {"artifact_server": {"port": port}})

        from osprey.infrastructure.server_launcher import ServerLauncher

        def _boom(*_args, **_kwargs):
            raise AssertionError("Probe 1 must not touch the network or launch a server")

        monkeypatch.setattr(ServerLauncher, "_launch_in_thread", _boom)
        monkeypatch.setattr(ServerLauncher, "_is_running", _boom)
        monkeypatch.setattr("urllib.request.urlopen", _boom)

        assert _preflight({}, None, "127.0.0.1", 8087) == ([], [])

    def test_panel_id_mapping_covers_every_registry_server(self):
        """Every FRAMEWORK_WEB_SERVERS entry except `artifact` (always launched,
        never gated on web.panels) must have a `_PANEL_ID_FOR_REGISTRY_KEY`
        mapping — a missing entry makes the probe silently skip that panel, so
        a foreign listener on its port would go unreported."""
        from osprey.cli.web_cmd import _PANEL_ID_FOR_REGISTRY_KEY
        from osprey.registry.web import FRAMEWORK_WEB_SERVERS

        expected_keys = set(FRAMEWORK_WEB_SERVERS) - {"artifact"}
        assert set(_PANEL_ID_FOR_REGISTRY_KEY) == expected_keys


class TestWebCommandPreflightWiring:
    """`osprey web`: consolidated report + exit 1, or --skip-preflight bypass."""

    def _stub_launch(self, monkeypatch):
        """Prevent the real server from starting once pre-flight passes."""
        monkeypatch.setattr("osprey.interfaces.web_terminal.run_web", lambda **_kw: None)
        monkeypatch.setattr("osprey.mcp_env.load_dotenv_from_project", lambda: None)

    def test_held_companion_port_aborts_before_bind(self, runner, monkeypatch):
        artifact_port = _free_port()
        _patch_config(monkeypatch, {"artifact_server": {"port": artifact_port}})
        self._stub_launch(monkeypatch)
        own_port = _free_port()

        held = _hold_port(artifact_port)
        try:
            result = runner.invoke(
                web, ["--port", str(own_port), "--shell", "true"], catch_exceptions=False
            )
        finally:
            held.close()

        assert result.exit_code == 1
        assert str(artifact_port) in result.output
        assert "artifact" in result.output
        assert f"lsof -i :{artifact_port}" in result.output

    def test_free_ports_clean_launch(self, runner, monkeypatch):
        artifact_port = _free_port()
        _patch_config(monkeypatch, {"artifact_server": {"port": artifact_port}})
        self._stub_launch(monkeypatch)
        own_port = _free_port()

        result = runner.invoke(
            web, ["--port", str(own_port), "--shell", "true"], catch_exceptions=False
        )

        assert result.exit_code == 0

    def test_skip_preflight_bypasses_held_port(self, runner, monkeypatch):
        artifact_port = _free_port()
        _patch_config(monkeypatch, {"artifact_server": {"port": artifact_port}})
        self._stub_launch(monkeypatch)
        own_port = _free_port()

        held = _hold_port(artifact_port)
        try:
            result = runner.invoke(
                web,
                ["--port", str(own_port), "--shell", "true", "--skip-preflight"],
                catch_exceptions=False,
            )
        finally:
            held.close()

        assert result.exit_code == 0
        assert str(artifact_port) not in result.output


# -- pre-flight (Task 1.2: Probe 2 auth-secret, Probe 3 config/settings) ---
#
# Probe 2 resolves the provider spec and checks whether its auth secret is
# resolvable (env or .env) before launch — a proxy provider without one is a
# hard failure, direct Anthropic without a key is only a warning (subscription
# / OAuth login still works). Probe 3 does a dedicated JSON/YAML parse of
# .claude/settings.json and config.yml so a syntax error surfaces as a
# pre-flight failure instead of load_osprey_config() silently degrading to {}.


def _stub_spec(**overrides):
    """Build a minimal ClaudeCodeModelSpec for Probe 2 tests without a real config.yml."""
    from osprey.build.claude_code_resolver import ClaudeCodeModelSpec

    defaults = {
        "provider": "als-apg",
        "auth_env_var": "ANTHROPIC_AUTH_TOKEN",
        "auth_secret_env": "ALS_APG_API_KEY",
        "needs_proxy": True,
    }
    defaults.update(overrides)
    return ClaudeCodeModelSpec(**defaults)


class TestPreflightAuthSecret:
    """Probe 2: the resolved provider's auth secret must be resolvable before launch."""

    def _stub_launch(self, monkeypatch):
        monkeypatch.setattr("osprey.interfaces.web_terminal.run_web", lambda **_kw: None)
        monkeypatch.setattr("osprey.mcp_env.load_dotenv_from_project", lambda: None)

    def _stub_clean_ports(self, monkeypatch):
        artifact_port = _free_port()
        _patch_config(monkeypatch, {"artifact_server": {"port": artifact_port}})

    def test_proxy_provider_missing_secret_aborts(self, runner, monkeypatch, tmp_path):
        self._stub_launch(monkeypatch)
        self._stub_clean_ports(monkeypatch)
        monkeypatch.setattr(
            "osprey.build.claude_code_resolver.load_provider_spec", lambda *_a, **_kw: _stub_spec()
        )
        monkeypatch.delenv("ALS_APG_API_KEY", raising=False)
        own_port = _free_port()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                web, ["--port", str(own_port), "--shell", "true"], catch_exceptions=False
            )

        assert result.exit_code == 1
        assert "ALS_APG_API_KEY" in result.output
        assert "als-apg" in result.output

    def test_proxy_provider_empty_secret_aborts(self, runner, monkeypatch, tmp_path):
        """An exported-but-empty secret counts as absent, matching `osprey claude status`.

        `KEY in os.environ` would treat `export ALS_APG_API_KEY=` as present and
        let the launch proceed into a runtime auth error; the truthiness check
        (`bool(os.environ.get(...))`) treats it as missing and hard-fails here.
        """
        self._stub_launch(monkeypatch)
        self._stub_clean_ports(monkeypatch)
        monkeypatch.setattr(
            "osprey.build.claude_code_resolver.load_provider_spec", lambda *_a, **_kw: _stub_spec()
        )
        monkeypatch.setenv("ALS_APG_API_KEY", "")
        own_port = _free_port()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                web, ["--port", str(own_port), "--shell", "true"], catch_exceptions=False
            )

        assert result.exit_code == 1
        assert "ALS_APG_API_KEY" in result.output

    def test_proxy_provider_secret_in_dotenv_only_passes(self, runner, monkeypatch, tmp_path):
        """Secret defined ONLY in .env (not os.environ) must still count as present.

        Guards the ordering bug: load_dotenv_from_project() doesn't run until
        after pre-flight on the foreground path, so a naive os.environ-only
        check would false-fail a healthy proxy launch.
        """
        self._stub_launch(monkeypatch)
        self._stub_clean_ports(monkeypatch)
        monkeypatch.setattr(
            "osprey.build.claude_code_resolver.load_provider_spec", lambda *_a, **_kw: _stub_spec()
        )
        monkeypatch.delenv("ALS_APG_API_KEY", raising=False)
        own_port = _free_port()

        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            (Path(td) / ".env").write_text("ALS_APG_API_KEY=secret-from-dotenv\n")
            result = runner.invoke(
                web, ["--port", str(own_port), "--shell", "true"], catch_exceptions=False
            )

        assert result.exit_code == 0
        assert "ALS_APG_API_KEY" not in result.output

    def test_direct_anthropic_missing_key_warns_no_abort(self, runner, monkeypatch, tmp_path):
        self._stub_launch(monkeypatch)
        self._stub_clean_ports(monkeypatch)
        spec = _stub_spec(
            provider="anthropic",
            auth_env_var="ANTHROPIC_API_KEY",
            auth_secret_env="ANTHROPIC_API_KEY",
            needs_proxy=False,
        )
        monkeypatch.setattr(
            "osprey.build.claude_code_resolver.load_provider_spec", lambda *_a, **_kw: spec
        )
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        own_port = _free_port()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                web, ["--port", str(own_port), "--shell", "true"], catch_exceptions=False
            )

        assert result.exit_code == 0
        assert "WARNING" in result.output
        assert "ANTHROPIC_API_KEY" in result.output

    def test_no_provider_configured_skipped(self, runner, monkeypatch, tmp_path):
        self._stub_launch(monkeypatch)
        self._stub_clean_ports(monkeypatch)
        monkeypatch.setattr(
            "osprey.build.claude_code_resolver.load_provider_spec", lambda *_a, **_kw: None
        )
        own_port = _free_port()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                web, ["--port", str(own_port), "--shell", "true"], catch_exceptions=False
            )

        assert result.exit_code == 0
        assert "auth secret" not in result.output

    def test_no_provider_network_call(self, runner, monkeypatch, tmp_path):
        """Probe 2 never calls a provider's check_health / completion path."""
        self._stub_launch(monkeypatch)
        self._stub_clean_ports(monkeypatch)
        monkeypatch.setattr(
            "osprey.build.claude_code_resolver.load_provider_spec", lambda *_a, **_kw: _stub_spec()
        )
        monkeypatch.setenv("ALS_APG_API_KEY", "present")

        def _boom(*_a, **_kw):
            raise AssertionError("Probe 2 must not call provider.check_health")

        monkeypatch.setattr("osprey.models.providers.base.BaseProvider.check_health", _boom)
        own_port = _free_port()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                web, ["--port", str(own_port), "--shell", "true"], catch_exceptions=False
            )

        assert result.exit_code == 0


class TestPreflightConfigValidity:
    """Probe 3: config.yml and .claude/settings.json must at least parse."""

    def _stub_launch(self, monkeypatch):
        monkeypatch.setattr("osprey.interfaces.web_terminal.run_web", lambda **_kw: None)
        monkeypatch.setattr("osprey.mcp_env.load_dotenv_from_project", lambda: None)
        # No provider configured — keep Probe 2 out of these tests' way.
        monkeypatch.setattr(
            "osprey.build.claude_code_resolver.load_provider_spec", lambda *_a, **_kw: None
        )

    def _stub_clean_ports(self, monkeypatch):
        artifact_port = _free_port()
        _patch_config(monkeypatch, {"artifact_server": {"port": artifact_port}})

    def test_malformed_settings_json_aborts(self, runner, monkeypatch, tmp_path):
        self._stub_launch(monkeypatch)
        self._stub_clean_ports(monkeypatch)
        own_port = _free_port()

        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            settings_path = Path(td) / ".claude" / "settings.json"
            settings_path.parent.mkdir(parents=True)
            settings_path.write_text("{not valid json")
            result = runner.invoke(
                web, ["--port", str(own_port), "--shell", "true"], catch_exceptions=False
            )

        assert result.exit_code == 1
        assert "settings.json" in result.output
        assert "invalid JSON" in result.output

    def test_absent_settings_json_skipped(self, runner, monkeypatch, tmp_path):
        self._stub_launch(monkeypatch)
        self._stub_clean_ports(monkeypatch)
        own_port = _free_port()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                web, ["--port", str(own_port), "--shell", "true"], catch_exceptions=False
            )

        assert result.exit_code == 0
        assert "settings.json" not in result.output

    def test_malformed_config_yml_aborts(self, runner, monkeypatch, tmp_path):
        self._stub_launch(monkeypatch)
        self._stub_clean_ports(monkeypatch)
        own_port = _free_port()

        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            (Path(td) / "config.yml").write_text("key: [unclosed\n")
            result = runner.invoke(
                web, ["--port", str(own_port), "--shell", "true"], catch_exceptions=False
            )

        assert result.exit_code == 1
        assert "config.yml" in result.output
        assert "invalid YAML" in result.output


class TestDetachSkipsPreflightInChild:
    """The detached re-spawn must not re-run pre-flight in the child."""

    @patch("osprey.cli.web_cmd._wait_for_server", return_value=True)
    @patch("osprey.cli.web_cmd.subprocess.Popen")
    def test_child_argv_gets_skip_preflight(self, mock_popen, mock_wait, tmp_path):
        mock_proc = MagicMock()
        mock_proc.pid = 4242
        mock_popen.return_value = mock_proc

        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            _start_detached("127.0.0.1", 8087, None, None)
        finally:
            os.chdir(cwd)

        cmd = mock_popen.call_args.args[0]
        assert "--skip-preflight" in cmd
        assert "--detach" not in cmd
