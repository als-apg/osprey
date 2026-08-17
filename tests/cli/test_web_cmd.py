"""Tests for the ``osprey web`` CLI command (detach/stop, pre-flight, discovery).

``osprey web`` serves a *deployment*, not a directory it was handed: it walks up
to the nearest ``profile.yml`` and launches that repo's ``build/``. So the unit
under test here is always a repo with a render in it, and the two zones stay
distinguishable — the render holds ``config.yml`` and ``.claude/``, the repo
root holds ``.env`` and the detached server's PID/log pair under ``var/``.
"""

from __future__ import annotations

import os
import signal
import socket
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from osprey.cli.web_cmd import (
    DECLARED_BIND_ENV,
    DECLARED_WEB_PORT_ENV,
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
from tests.cli._lifecycle_build import stub_build
from tests.cli._scoped_subprocess import patch_subprocess


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def deployment(lifecycle_repo: Path) -> Path:
    """An exemplar repo with a render — the shape every launch test needs."""
    stub_build(lifecycle_repo, config="web: {}\n")
    return lifecycle_repo


def seed_pid(root: Path, text: str) -> Path:
    """Write a PID file as a running server would, creating the STATE zone.

    ``PID_FILE`` is a path under ``var/`` rather than a bare basename, so a test
    that writes it has to make the directory the same way ``_write_pid`` does.
    """
    path = root / PID_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


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
    seed_pid(tmp_path, str(pid))
    assert _read_pid(tmp_path) == pid


def test_read_pid_stale(tmp_path: Path):
    seed_pid(tmp_path, "999999999")
    with patch("osprey.cli.web_cmd.os.kill", side_effect=ProcessLookupError):
        result = _read_pid(tmp_path)
    assert result is None
    assert not (tmp_path / PID_FILE).exists()


def test_read_pid_corrupt(tmp_path: Path):
    seed_pid(tmp_path, "not-a-number")
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


# -- deployment resolution --------------------------------------------------
#
# `osprey web` resolves THE deployment it serves exactly once, up front, in
# _resolve_render(): walk up to the nearest profile.yml (or start from --repo),
# then serve that repo's build/. These tests pin that single mechanism — the
# loud failure with nothing rendered, the banner, the OSPREY_CONFIG publication
# for children, and the always-explicit --repo in the detached child's argv.
# (TestRepoFlagIsAuthoritative below pins the other half: what --repo overrides.)


class TestDeploymentResolution:
    """`osprey web` must resolve a real render or refuse to start.

    The web terminal's panel set, companion servers, and provider wiring all
    come from the rendered config.yml. A launch without one used to silently
    degrade to universal panels only — these tests pin the loud failure and the
    explicit repo forwarding that prevent that.
    """

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        # web() writes OSPREY_WEB_PORT and OSPREY_CONFIG directly into
        # os.environ for its child processes. setenv-then-delenv forces
        # monkeypatch to record an undo entry even when the key started
        # absent, so those in-test writes are always rolled back instead of
        # leaking into later tests (same trick as test_web_bind.py's
        # _isolate_bind_and_port_env).
        for key in ("OSPREY_CONFIG", "OSPREY_WEB_PORT"):
            monkeypatch.setenv(key, "__unset_by_test_fixture__")
            monkeypatch.delenv(key)

    def test_outside_any_deployment_refuses_to_start(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(web, [])

        assert result.exit_code == 1
        assert "No OSPREY deployment repo found" in result.output

    def test_repo_without_a_render_refuses(self, runner, lifecycle_repo):
        """A repo that has never been built is a launch error naming the fix.

        Not a silent fallback: without a render there is no panel set, no
        companion server and no provider, so the terminal would come up as a
        mystery shell rather than this deployment's agent.
        """
        result = runner.invoke(web, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 1
        assert "no build found" in result.output
        assert "osprey build" in result.output

    @patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/bin/fake-claude")
    @patch("osprey.cli.web_cmd._preflight", return_value=([], []))
    @patch("osprey.cli.web_cmd._wait_for_server", return_value=True)
    @patch_subprocess("osprey.cli.web_cmd", popen=True)
    @patch("osprey.cli.web_cmd.get_config_value", return_value={})
    def test_repo_flag_resolves_a_deployment_from_elsewhere(
        self,
        mock_config,
        fake_subprocess,
        mock_wait,
        mock_preflight,
        mock_resolve,
        tmp_path,
        runner,
        deployment,
    ):
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        fake_subprocess.Popen.return_value = mock_proc

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(web, ["--detach", "--repo", str(deployment)])

        assert result.exit_code == 0
        child_argv = fake_subprocess.Popen.call_args.args[0]
        assert "--repo" in child_argv
        assert str(deployment.resolve()) in child_argv

    @patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/bin/fake-claude")
    @patch("osprey.cli.web_cmd._preflight", return_value=([], []))
    @patch("osprey.cli.web_cmd._wait_for_server", return_value=True)
    @patch_subprocess("osprey.cli.web_cmd", popen=True)
    @patch("osprey.cli.web_cmd.get_config_value", return_value={})
    def test_detach_child_argv_always_carries_the_repo(
        self,
        mock_config,
        fake_subprocess,
        mock_wait,
        mock_preflight,
        mock_resolve,
        monkeypatch,
        runner,
        deployment,
    ):
        """Even without --repo, the child command must name the deployment.

        The detached child's argv is what `ps` shows — and what humans and
        agents copy to restart the server. Without it the identity would ride
        only in the inherited cwd, which by then is the *render*, not a repo.
        """
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        fake_subprocess.Popen.return_value = mock_proc

        monkeypatch.chdir(deployment)
        result = runner.invoke(web, ["--detach"])

        assert result.exit_code == 0
        child_argv = fake_subprocess.Popen.call_args.args[0]
        assert "--project" not in child_argv
        assert child_argv[child_argv.index("--repo") + 1] == str(deployment)

    @patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/bin/fake-claude")
    @patch("osprey.cli.web_cmd._preflight", return_value=([], []))
    @patch("osprey.cli.web_cmd._wait_for_server", return_value=True)
    @patch_subprocess("osprey.cli.web_cmd", popen=True)
    @patch("osprey.cli.web_cmd.get_config_value", return_value={})
    def test_banner_names_the_repo_and_the_render(
        self,
        mock_config,
        fake_subprocess,
        mock_wait,
        mock_preflight,
        mock_resolve,
        monkeypatch,
        runner,
        deployment,
    ):
        """Both, because they are different directories.

        An operator reading only "build/…" could not tell which deployment they
        are in; reading only the repo could not tell whether the render is the
        one they just made.
        """
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        fake_subprocess.Popen.return_value = mock_proc

        monkeypatch.chdir(deployment)
        result = runner.invoke(web, ["--detach"])

        assert result.exit_code == 0
        # One section, so the two paths line up as a column under padded labels.
        assert f"Repo    {deployment}" in result.stdout
        assert f"Build   {deployment / 'build'}" in result.stdout

    def test_ambient_osprey_config_does_not_choose_the_deployment(
        self, runner, tmp_path, monkeypatch, deployment
    ):
        """The env var is a publication for children, never a discovery input.

        An export left over from whichever deployment the operator last worked
        in must not decide what this command serves — it is overwritten with
        the render the walk actually found.
        """
        stale = tmp_path / "stale-config.yml"
        stale.write_text("web: {}\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(stale))

        captured = {}

        def _capture_run_web(**kw):
            captured["kwargs"] = kw
            captured["cwd"] = os.getcwd()

        monkeypatch.setattr("osprey.interfaces.web_terminal.run_web", _capture_run_web)
        monkeypatch.setattr("osprey.mcp_env.load_dotenv_from_project", lambda: None)
        monkeypatch.chdir(deployment)

        result = runner.invoke(web, ["--shell", "true", "--skip-preflight"], catch_exceptions=False)

        assert result.exit_code == 0
        render = (deployment / "build").resolve()
        assert Path(captured["kwargs"]["config_path"]).resolve() == render / "config.yml"
        # The render is also the working directory: the PTY's agent CLI treats
        # its cwd as the project root.
        assert Path(captured["cwd"]).resolve() == render

    def test_bare_launch_publishes_osprey_config_for_children(
        self, runner, monkeypatch, deployment
    ):
        """Even a plain `osprey web` inside a repo must publish OSPREY_CONFIG.

        The --reload worker re-imports create_app() with no arguments, and
        child PTY shells / MCP servers resolve the deployment through the
        environment — without the publication they all fall back to their own
        cwd, which the operator may have changed by then.
        """
        observed = {}

        def _capture_run_web(**_kw):
            observed["osprey_config"] = os.environ.get("OSPREY_CONFIG")

        monkeypatch.setattr("osprey.interfaces.web_terminal.run_web", _capture_run_web)
        monkeypatch.setattr("osprey.mcp_env.load_dotenv_from_project", lambda: None)
        monkeypatch.chdir(deployment)

        result = runner.invoke(web, ["--shell", "true", "--skip-preflight"], catch_exceptions=False)

        assert result.exit_code == 0
        expected = (deployment / "build" / "config.yml").resolve()
        assert Path(observed["osprey_config"]).resolve() == expected


# -- detach -----------------------------------------------------------------


@patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/bin/fake-claude")
@patch("osprey.cli.web_cmd._preflight", return_value=([], []))
@patch("osprey.cli.web_cmd._wait_for_server", return_value=True)
@patch_subprocess("osprey.cli.web_cmd", popen=True)
@patch("osprey.cli.web_cmd.get_config_value", return_value={})
def test_detach_spawns_subprocess(
    mock_config,
    fake_subprocess,
    mock_wait,
    mock_preflight,
    mock_resolve,
    monkeypatch,
    runner,
    deployment,
):
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    fake_subprocess.Popen.return_value = mock_proc

    monkeypatch.chdir(deployment)
    result = runner.invoke(web, ["--detach"])

    assert result.exit_code == 0
    fake_subprocess.Popen.assert_called_once()
    call_kwargs = fake_subprocess.Popen.call_args
    assert call_kwargs.kwargs["start_new_session"] is True


@patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/bin/fake-claude")
@patch("osprey.cli.web_cmd._preflight", return_value=([], []))
@patch("osprey.cli.web_cmd._wait_for_server", return_value=True)
@patch_subprocess("osprey.cli.web_cmd", popen=True)
@patch("osprey.cli.web_cmd.get_config_value", return_value={})
def test_detach_writes_pid_file_into_the_state_zone(
    mock_config,
    fake_subprocess,
    mock_wait,
    mock_preflight,
    mock_resolve,
    monkeypatch,
    runner,
    deployment,
):
    """``var/``, not the repo root and not the render.

    The STATE zone is git-ignored (so a running server leaves `git status`
    clean), host-local, and survives the next ``osprey build`` — which wipes
    the render out from under a running server.
    """
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    fake_subprocess.Popen.return_value = mock_proc

    monkeypatch.chdir(deployment)
    result = runner.invoke(web, ["--detach"])

    assert result.exit_code == 0
    assert (deployment / "var" / "osprey-web.pid").read_text() == "12345"
    assert not (deployment / ".osprey-web.pid").exists()


@patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/bin/fake-claude")
@patch("osprey.cli.web_cmd._preflight", return_value=([], []))
@patch("osprey.cli.web_cmd._read_pid", return_value=99999)
@patch("osprey.cli.web_cmd.get_config_value", return_value={})
def test_detach_idempotent_when_running(
    mock_config, mock_read_pid, mock_preflight, mock_resolve, monkeypatch, runner, deployment
):
    monkeypatch.chdir(deployment)
    result = runner.invoke(web, ["--detach"])

    assert result.exit_code == 0
    assert "already running" in result.output


@patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/bin/fake-claude")
@patch("osprey.cli.web_cmd._preflight", return_value=([], []))
@patch("osprey.cli.web_cmd._wait_for_server", return_value=True)
@patch_subprocess("osprey.cli.web_cmd", popen=True)
@patch("osprey.cli.web_cmd.get_config_value", return_value={})
def test_detach_cleans_stale_pid(
    mock_config,
    fake_subprocess,
    mock_wait,
    mock_preflight,
    mock_resolve,
    monkeypatch,
    runner,
    deployment,
):
    mock_proc = MagicMock()
    mock_proc.pid = 55555
    fake_subprocess.Popen.return_value = mock_proc

    seed_pid(deployment, "999999999")
    monkeypatch.chdir(deployment)

    # _read_pid will call os.kill which raises ProcessLookupError for stale
    with patch("osprey.cli.web_cmd.os.kill", side_effect=ProcessLookupError):
        result = runner.invoke(web, ["--detach"])

    assert result.exit_code == 0
    # Should have started a new server
    fake_subprocess.Popen.assert_called_once()


@patch("osprey.utils.shell_resolver.resolve_shell_command", return_value="/bin/fake-claude")
@patch("osprey.cli.web_cmd._preflight", return_value=([], []))
@patch("osprey.cli.web_cmd._wait_for_server", return_value=True)
@patch_subprocess("osprey.cli.web_cmd", popen=True)
@patch("osprey.cli.web_cmd.get_config_value", return_value={})
def test_detach_shows_url_and_pid(
    mock_config,
    fake_subprocess,
    mock_wait,
    mock_preflight,
    mock_resolve,
    monkeypatch,
    runner,
    deployment,
):
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    fake_subprocess.Popen.return_value = mock_proc

    monkeypatch.chdir(deployment)
    result = runner.invoke(web, ["--detach"])

    assert "PID 12345" in result.output
    assert "http://127.0.0.1:8087" in result.output
    assert "osprey web stop" in result.output


# -- stop -------------------------------------------------------------------


def test_stop_kills_process(monkeypatch, runner, deployment):
    seed_pid(deployment, "12345")
    log_path = deployment / LOG_FILE
    log_path.write_text("some log")
    monkeypatch.chdir(deployment)

    with patch("osprey.cli.web_cmd.os.kill") as mock_kill:
        result = runner.invoke(web, ["stop"])

    mock_kill.assert_called_once_with(12345, signal.SIGTERM)
    assert result.exit_code == 0
    assert "Stopped" in result.output
    assert not (deployment / PID_FILE).exists()
    assert not log_path.exists()


def test_stop_no_pid_file(monkeypatch, runner, deployment):
    monkeypatch.chdir(deployment)
    result = runner.invoke(web, ["stop"])

    assert result.exit_code == 0
    assert "No running" in result.output


def test_stop_stale_pid(monkeypatch, runner, deployment):
    seed_pid(deployment, "999999999")
    monkeypatch.chdir(deployment)

    with patch("osprey.cli.web_cmd.os.kill", side_effect=ProcessLookupError):
        result = runner.invoke(web, ["stop"])

    assert result.exit_code == 0
    assert "not found" in result.output
    assert not (deployment / PID_FILE).exists()


# -- pre-flight (Task 1.1: --skip-preflight, _preflight, Probe 1) -----------
#
# Probe 1 is the companion port-collision guard: before binding its own port,
# `osprey web` TCP-connect-probes every companion panel port the lifespan
# will actually launch (see `_load_panel_config` / `_make_auto_launch_checker`
# / `resolve_web_server_address`). A listener already on one of those ports is
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
    """Wrap resolve_web_server_address to record which server keys get probed.

    Used to prove an excluded panel's port is never even resolved (as opposed
    to resolved-but-not-held) — avoids depending on the framework's default
    companion ports (8085/8086/8092/...) actually being free on the test host,
    which they may not be if a real `osprey web` happens to be running there.
    """
    from osprey.registry import web as registry_web

    probed_keys: list[str] = []
    original_resolve = registry_web.resolve_web_server_address

    def _spying_resolve(key, config=None):
        probed_keys.append(key)
        return original_resolve(key, config)

    monkeypatch.setattr(registry_web, "resolve_web_server_address", _spying_resolve)
    return probed_keys


def _preflight_at(root: Path):
    """Run pre-flight against a repo at *root*, with the render one level down.

    Probes 2 and 3 need the two zones separately — the provider comes from the
    render, its secret from the repo's ``.env`` — so the helper spells the pair
    once instead of at every Probe-1 call site, which cares about neither.
    """
    build = root / "build"
    return _preflight({}, root, build, build / "config.yml", "127.0.0.1", 8087)


class TestPreflightCompanionPortCollision:
    """Probe 1: held companion ports are reported; free/excluded ports are not."""

    def test_held_artifact_port_reported(self, monkeypatch, tmp_path):
        """A companion port the lifespan WILL launch, already held, is a finding."""
        port = _free_port()
        _patch_config(monkeypatch, {"artifact_server": {"port": port}})

        held = _hold_port(port)
        try:
            failures, warnings = _preflight_at(tmp_path)
        finally:
            held.close()

        assert len(failures) == 1
        assert str(port) in failures[0]
        assert "artifact" in failures[0]
        assert f"lsof -i :{port}" in failures[0]
        assert warnings == []

    def test_free_ports_clean_pass(self, monkeypatch, tmp_path):
        """With no companion ports held, _preflight returns no findings."""
        port = _free_port()
        _patch_config(monkeypatch, {"artifact_server": {"port": port}})

        assert _preflight_at(tmp_path) == ([], [])

    def test_require_section_unmet_panel_excluded(self, monkeypatch, tmp_path):
        """Enabled-but-not-launched panel (require_section unmet) is not probed."""
        # "channel-finder" is enabled in web.panels, but the channel_finder
        # top-level section (which gates auto_launch/require_section) is
        # absent — the lifespan's launcher would skip it, so its port must
        # never be resolved.
        artifact_port = _free_port()
        _patch_config(
            monkeypatch,
            {
                "artifact_server": {"port": artifact_port},
                "web": {"panels": {"channel-finder": True}},
            },
        )
        probed_keys = _spy_config_readers(monkeypatch)

        failures, warnings = _preflight_at(tmp_path)

        assert failures == []
        assert warnings == []
        assert "channel_finder" not in probed_keys

    def test_disabled_panel_excluded(self, monkeypatch, tmp_path):
        """A panel absent from web.panels is not probed even if its port is held."""
        artifact_port = _free_port()
        _patch_config(
            monkeypatch, {"artifact_server": {"port": artifact_port}}
        )  # ariel not enabled
        probed_keys = _spy_config_readers(monkeypatch)

        failures, warnings = _preflight_at(tmp_path)

        assert failures == []
        assert warnings == []
        assert "ariel" not in probed_keys

    def test_no_network_or_registry_calls(self, monkeypatch, tmp_path):
        """Probe 1 never starts a companion server or hits its /health endpoint."""
        port = _free_port()
        _patch_config(monkeypatch, {"artifact_server": {"port": port}})

        from osprey.infrastructure.server_launcher import ServerLauncher

        def _boom(*_args, **_kwargs):
            raise AssertionError("Probe 1 must not touch the network or launch a server")

        monkeypatch.setattr(ServerLauncher, "_launch_in_thread", _boom)
        monkeypatch.setattr(ServerLauncher, "_is_running", _boom)
        monkeypatch.setattr("urllib.request.urlopen", _boom)

        assert _preflight_at(tmp_path) == ([], [])

    def test_panel_id_mapping_covers_every_registry_server(self):
        """The probe resolves each server's panel id off the registry entry.

        A server whose panel id the probe cannot resolve is silently skipped, so
        a foreign listener on its port goes unreported. The probe used to carry
        its own key → id table, and that table had already lost the gallery's
        `artifacts` entry; reading `WebServerDefinition.panel_id` means every
        registered server is covered by construction.
        """
        import inspect

        from osprey.cli import web_cmd
        from osprey.profiles.web_panels import BUILTIN_PANELS
        from osprey.registry.web import FRAMEWORK_WEB_SERVERS

        assert {defn.panel_id for defn in FRAMEWORK_WEB_SERVERS.values()} == BUILTIN_PANELS
        assert "_PANEL_ID_FOR_REGISTRY_KEY" not in inspect.getsource(web_cmd)


class TestWebCommandPreflightWiring:
    """`osprey web`: consolidated report + exit 1, or --skip-preflight bypass."""

    def _stub_launch(self, monkeypatch):
        """Prevent the real server from starting once pre-flight passes."""
        monkeypatch.setattr("osprey.interfaces.web_terminal.run_web", lambda **_kw: None)
        monkeypatch.setattr("osprey.mcp_env.load_dotenv_from_project", lambda: None)

    def test_held_companion_port_aborts_before_bind(self, runner, monkeypatch, deployment):
        artifact_port = _free_port()
        _patch_config(monkeypatch, {"artifact_server": {"port": artifact_port}})
        self._stub_launch(monkeypatch)
        own_port = _free_port()

        held = _hold_port(artifact_port)
        try:
            result = runner.invoke(
                web,
                ["--repo", str(deployment), "--port", str(own_port), "--shell", "true"],
                catch_exceptions=False,
            )
        finally:
            held.close()

        assert result.exit_code == 1
        assert str(artifact_port) in result.output
        assert "artifact" in result.output
        assert f"lsof -i :{artifact_port}" in result.output

    def test_free_ports_clean_launch(self, runner, monkeypatch, deployment):
        artifact_port = _free_port()
        _patch_config(monkeypatch, {"artifact_server": {"port": artifact_port}})
        self._stub_launch(monkeypatch)
        own_port = _free_port()

        result = runner.invoke(
            web,
            ["--repo", str(deployment), "--port", str(own_port), "--shell", "true"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0

    def test_skip_preflight_bypasses_held_port(self, runner, monkeypatch, deployment):
        artifact_port = _free_port()
        _patch_config(monkeypatch, {"artifact_server": {"port": artifact_port}})
        self._stub_launch(monkeypatch)
        own_port = _free_port()

        held = _hold_port(artifact_port)
        try:
            result = runner.invoke(
                web,
                [
                    "--repo",
                    str(deployment),
                    "--port",
                    str(own_port),
                    "--shell",
                    "true",
                    "--skip-preflight",
                ],
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

    def test_proxy_provider_missing_secret_aborts(self, runner, monkeypatch, deployment):
        self._stub_launch(monkeypatch)
        self._stub_clean_ports(monkeypatch)
        monkeypatch.setattr(
            "osprey.build.claude_code_resolver.load_provider_spec", lambda *_a, **_kw: _stub_spec()
        )
        monkeypatch.delenv("ALS_APG_API_KEY", raising=False)
        own_port = _free_port()

        result = runner.invoke(
            web,
            ["--repo", str(deployment), "--port", str(own_port), "--shell", "true"],
            catch_exceptions=False,
        )

        assert result.exit_code == 1
        assert "ALS_APG_API_KEY" in result.output
        assert "als-apg" in result.output

    def test_proxy_provider_empty_secret_aborts(self, runner, monkeypatch, deployment):
        """An exported-but-empty secret counts as absent, matching `osprey status`.

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

        result = runner.invoke(
            web,
            ["--repo", str(deployment), "--port", str(own_port), "--shell", "true"],
            catch_exceptions=False,
        )

        assert result.exit_code == 1
        assert "ALS_APG_API_KEY" in result.output

    def test_proxy_provider_secret_in_dotenv_only_passes(self, runner, monkeypatch, deployment):
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

        # The SECRETS zone is the repo root, NOT the render: `.env` has to
        # survive `osprey build` wiping build/, so that is where the probe must
        # look for a secret that lives only on disk.
        (deployment / ".env").write_text("ALS_APG_API_KEY=secret-from-dotenv\n")

        result = runner.invoke(
            web,
            ["--repo", str(deployment), "--port", str(own_port), "--shell", "true"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "ALS_APG_API_KEY" not in result.output

    def test_direct_anthropic_missing_key_warns_no_abort(self, runner, monkeypatch, deployment):
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

        result = runner.invoke(
            web,
            ["--repo", str(deployment), "--port", str(own_port), "--shell", "true"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        # A pre-flight warning is trouble, so it goes to stderr wearing the
        # renderer's warning mark rather than a hand-typed "WARNING:" prefix.
        assert "⚠" in result.stderr
        assert "ANTHROPIC_API_KEY" in result.stderr

    def test_no_provider_configured_skipped(self, runner, monkeypatch, deployment):
        self._stub_launch(monkeypatch)
        self._stub_clean_ports(monkeypatch)
        monkeypatch.setattr(
            "osprey.build.claude_code_resolver.load_provider_spec", lambda *_a, **_kw: None
        )
        own_port = _free_port()

        result = runner.invoke(
            web,
            ["--repo", str(deployment), "--port", str(own_port), "--shell", "true"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "auth secret" not in result.output

    def test_no_provider_network_call(self, runner, monkeypatch, deployment):
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

        result = runner.invoke(
            web,
            ["--repo", str(deployment), "--port", str(own_port), "--shell", "true"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0

    def test_credential_error_reported_not_hidden(self, runner, monkeypatch, deployment):
        """A telemetry-credential failure explains itself instead of vanishing.

        `ObservabilityCredentialError` subclasses `ValueError`, so the broad
        skip below would swallow it and leave pre-flight silent about a read
        that never happened. The launch still proceeds: telemetry credentials
        say nothing about whether the terminal can authenticate.
        """
        from osprey.build.claude_code_telemetry import ObservabilityCredentialError

        self._stub_launch(monkeypatch)
        self._stub_clean_ports(monkeypatch)

        def _raise(*_a, **_kw):
            raise ObservabilityCredentialError(
                "openobserve.user / openobserve.password are missing or blank"
            )

        monkeypatch.setattr("osprey.build.claude_code_resolver.load_provider_spec", _raise)
        own_port = _free_port()

        result = runner.invoke(
            web,
            ["--repo", str(deployment), "--port", str(own_port), "--shell", "true"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "⚠" in result.stderr
        assert "provider auth check skipped" in result.stderr
        assert "openobserve.password" in result.stderr

    def test_unrelated_value_error_still_skips_quietly(self, runner, monkeypatch, deployment):
        """The narrow arm must not widen the catch it sits in front of.

        An unknown provider name (or any other plain `ValueError`) is Probe 3's
        diagnosis to make, so Probe 2 keeps saying nothing about it.
        """
        self._stub_launch(monkeypatch)
        self._stub_clean_ports(monkeypatch)

        def _raise(*_a, **_kw):
            raise ValueError("unknown provider 'nowhere'")

        monkeypatch.setattr("osprey.build.claude_code_resolver.load_provider_spec", _raise)
        own_port = _free_port()

        result = runner.invoke(
            web,
            ["--repo", str(deployment), "--port", str(own_port), "--shell", "true"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "provider auth check skipped" not in result.output
        assert "provider auth check skipped" not in result.stderr
        assert "nowhere" not in result.output


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

    def test_malformed_settings_json_aborts(self, runner, monkeypatch, deployment):
        self._stub_launch(monkeypatch)
        self._stub_clean_ports(monkeypatch)
        own_port = _free_port()

        # `.claude/` is part of the RENDER — it is what the PTY's agent reads,
        # and `osprey build` writes it — so that is where the probe must parse.
        settings_path = deployment / "build" / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text("{not valid json")

        result = runner.invoke(
            web,
            ["--repo", str(deployment), "--port", str(own_port), "--shell", "true"],
            catch_exceptions=False,
        )

        assert result.exit_code == 1
        assert "settings.json" in result.output
        assert "invalid JSON" in result.output

    def test_absent_settings_json_skipped(self, runner, monkeypatch, deployment):
        self._stub_launch(monkeypatch)
        self._stub_clean_ports(monkeypatch)
        own_port = _free_port()

        result = runner.invoke(
            web,
            ["--repo", str(deployment), "--port", str(own_port), "--shell", "true"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "settings.json" not in result.output

    def test_malformed_config_yml_aborts(self, runner, monkeypatch, deployment):
        self._stub_launch(monkeypatch)
        self._stub_clean_ports(monkeypatch)
        own_port = _free_port()

        (deployment / "build" / "config.yml").write_text("key: [unclosed\n")

        result = runner.invoke(
            web,
            ["--repo", str(deployment), "--port", str(own_port), "--shell", "true"],
            catch_exceptions=False,
        )

        assert result.exit_code == 1
        assert "config.yml" in result.output
        assert "invalid YAML" in result.output


class TestDetachSkipsPreflightInChild:
    """The detached re-spawn must not re-run pre-flight in the child."""

    @patch("osprey.cli.web_cmd._wait_for_server", return_value=True)
    @patch_subprocess("osprey.cli.web_cmd", popen=True)
    def test_child_argv_gets_skip_preflight(
        self, fake_subprocess, mock_wait, tmp_path, monkeypatch
    ):
        mock_proc = MagicMock()
        mock_proc.pid = 4242
        fake_subprocess.Popen.return_value = mock_proc

        monkeypatch.chdir(tmp_path)
        _start_detached("127.0.0.1", 8087, None, tmp_path)

        cmd = fake_subprocess.Popen.call_args.args[0]
        assert "--skip-preflight" in cmd
        assert "--detach" not in cmd
        # The resolved repo always rides in the child argv (see
        # TestDeploymentResolution.test_detach_child_argv_always_carries_the_repo
        # for the CLI-level pin).
        assert "--repo" in cmd


# -- --repo makes the deployment authoritative ------------------------------
#
# `osprey web --repo X` run from outside X must behave exactly like standing
# inside X. Regression guard for the reported failure where the deployment's
# `.env` was never loaded, so `${ARGO_API_KEY}` in its config stayed an
# unexpanded literal and the agent came up without provider credentials. Two
# independent sites resolved "the project" from the process cwd rather than
# from the resolved repo: web_cmd's pre-lifespan `.env` load and the lifespan's
# `app.state.config_path` (which gates provider env injection).
#
# The split makes this sharp: the `.env` is at the repo root and the config is
# in the render, so a launch that gets only one of them right is a visible
# failure rather than a coincidence.


@pytest.fixture
def restore_env_and_cwd():
    """Undo the process-global state a launch legitimately mutates.

    The command chdirs into the render and bulk-loads the deployment `.env`
    into os.environ, both of which outlive the CliRunner invocation and would
    leak into later tests.
    """
    saved_cwd = os.getcwd()
    saved_env = dict(os.environ)
    yield
    os.chdir(saved_cwd)
    os.environ.clear()
    os.environ.update(saved_env)


def _deployment_with_secret(repo: Path, port: int) -> Path:
    """Give *repo* a render whose provider key comes from the repo-root `.env`.

    Deliberately split across the two zones: the ``${VAR}`` placeholder is in
    the render's config, the value is in the SECRETS zone at the repo root.
    Resolving it requires getting both directories right.
    """
    stub_build(
        repo,
        config=(
            "system:\n"
            "  name: repro\n"
            "web_terminal:\n"
            "  host: 127.0.0.1\n"
            f"  port: {port}\n"
            "api:\n"
            "  providers:\n"
            "    argo:\n"
            "      base_url: https://argo.example/v1\n"
            "      api_key: ${OSPREY_TEST_PROJECT_KEY}\n"
        ),
    )
    (repo / ".env").write_text("OSPREY_TEST_PROJECT_KEY=sk-from-project-dotenv\n")
    return repo


class TestRepoFlagIsAuthoritative:
    """--repo must win over the process cwd for every deployment-scoped lookup."""

    def _invoke_from_elsewhere(self, tmp_path, repo, runner, monkeypatch, extra_args=()):
        """Run `osprey web --repo <repo>` with cwd set to a directory outside it.

        Returns the kwargs run_web was called with, plus the cwd and the
        OSPREY_TEST_PROJECT_KEY value observed at the moment run_web ran --
        i.e. the state the server (and every child PTY) would actually inherit.
        """
        port = _free_port()
        _deployment_with_secret(repo, port)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()

        monkeypatch.delenv("OSPREY_TEST_PROJECT_KEY", raising=False)
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.delenv("CONFIG_FILE", raising=False)
        # `web` publishes the bound port to child processes via OSPREY_WEB_PORT,
        # which is also --port's envvar — so an earlier test's invocation would
        # otherwise supply the port and mask the config read under test.
        monkeypatch.delenv("OSPREY_WEB_PORT", raising=False)
        monkeypatch.delenv(DECLARED_WEB_PORT_ENV, raising=False)
        monkeypatch.delenv(DECLARED_BIND_ENV, raising=False)

        observed = {}

        def _capture_run_web(**kwargs):
            observed["kwargs"] = kwargs
            observed["cwd"] = os.getcwd()
            observed["key"] = os.environ.get("OSPREY_TEST_PROJECT_KEY")
            observed["osprey_config"] = os.environ.get("OSPREY_CONFIG")

        monkeypatch.setattr("osprey.interfaces.web_terminal.run_web", _capture_run_web)
        monkeypatch.chdir(elsewhere)

        result = runner.invoke(
            web,
            ["--repo", str(repo), "--shell", "true", "--skip-preflight", *extra_args],
            catch_exceptions=False,
        )
        return result, observed, repo, port

    def test_repo_dotenv_is_loaded(
        self, tmp_path, lifecycle_repo, runner, monkeypatch, restore_env_and_cwd
    ):
        """The deployment's .env reaches os.environ before the server starts.

        This is the reported bug: without it, the render's ${VAR} placeholders
        stay literal and the provider has no credentials. The `.env` is at the
        repo root while the config that references it is in the render, so this
        only passes if both zones resolved to the same deployment.
        """
        result, observed, _repo, _port = self._invoke_from_elsewhere(
            tmp_path, lifecycle_repo, runner, monkeypatch
        )

        assert result.exit_code == 0
        assert observed["key"] == "sk-from-project-dotenv"

    def test_the_render_becomes_the_working_directory(
        self, tmp_path, lifecycle_repo, runner, monkeypatch, restore_env_and_cwd
    ):
        """The agent CLI treats its cwd as the project root, and the render is it.

        `.claude/`, `.mcp.json` and CLAUDE.md all live in build/, so anything
        else as the working directory hands the PTY a different project.
        """
        _result, observed, repo, _port = self._invoke_from_elsewhere(
            tmp_path, lifecycle_repo, runner, monkeypatch
        )

        assert Path(observed["cwd"]).resolve() == (repo / "build").resolve()

    def test_render_web_terminal_port_is_honored(
        self, tmp_path, lifecycle_repo, runner, monkeypatch, restore_env_and_cwd
    ):
        """web_terminal.port from the *resolved* render, not the 8087 default."""
        _result, observed, _repo, port = self._invoke_from_elsewhere(
            tmp_path, lifecycle_repo, runner, monkeypatch
        )

        assert observed["kwargs"]["port"] == port

    def test_osprey_config_points_at_the_render(
        self, tmp_path, lifecycle_repo, runner, monkeypatch, restore_env_and_cwd
    ):
        """Child processes (PTY shells, MCP servers) resolve config via OSPREY_CONFIG."""
        _result, observed, repo, _port = self._invoke_from_elsewhere(
            tmp_path, lifecycle_repo, runner, monkeypatch
        )

        expected = (repo / "build" / "config.yml").resolve()
        assert Path(observed["osprey_config"]).resolve() == expected

    def test_explicit_flag_beats_stale_osprey_config_export(
        self, tmp_path, lifecycle_repo, runner, monkeypatch, restore_env_and_cwd
    ):
        """A stale OSPREY_CONFIG in the shell must not defeat an explicit --repo."""
        stale = tmp_path / "stale"
        stale.mkdir()
        (stale / "config.yml").write_text("system:\n  name: stale\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(stale / "config.yml"))

        _result, observed, repo, _port = self._invoke_from_elsewhere(
            tmp_path, lifecycle_repo, runner, monkeypatch
        )

        expected = (repo / "build" / "config.yml").resolve()
        assert Path(observed["osprey_config"]).resolve() == expected
        assert observed["key"] == "sk-from-project-dotenv"
