"""Tests for the C3 loopback-bind enforcement in `osprey web`.

Per the multi-user compose (`docker-compose.web.yml.j2`), every per-user
container declares `OSPREY_TERMINAL_BIND_HOST=127.0.0.1` so nginx is the
ONLY off-host path (criterion C3). Without a reader for that env var, a stale
image CMD passing `--host 0.0.0.0` would silently punch through the
reverse-proxy chokepoint. `resolve_bind_host()` makes the declared env
authoritative over both `--host` and config, while leaving single-user
`osprey web` (no declared env) free to honor `--host 0.0.0.0` verbatim.
"""

from __future__ import annotations

import contextlib
import os
import socket

import pytest
from click.testing import CliRunner

# Imported for effect: ``server_launcher`` builds its ``_launchers`` dict once,
# at import, by comprehending ``FRAMEWORK_WEB_SERVERS``. A test below stands
# that registry in as ``{}`` — so whichever test imports the launcher module
# FIRST decides whether every later test sees the real launchers or none at
# all. Importing it here, at collection, takes that decision away from test
# order.
import osprey.infrastructure.server_launcher  # noqa: F401
from osprey.cli.web_cmd import (
    DECLARED_BIND_ENV,
    DECLARED_WEB_PORT_ENV,
    resolve_bind_host,
    resolve_web_port,
    web,
)
from osprey.port_layout import DEFAULT_PORT_BASE
from tests.cli._lifecycle_build import stub_build


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
    from osprey.interfaces.web_auth import OPERATOR_SECRET_ENV, reset_web_credentials

    # ``web()`` now mints the operator secret straight into ``os.environ`` and
    # into the process-wide web-credentials holder. Both outlive a CliRunner
    # invocation, so isolate the env carrier the same setenv-then-delenv way as
    # the port keys, and reset the holder around every test — otherwise the
    # first launch would decide (and leak) the secret for every test after it,
    # which is what masked a launch that should have refused.
    for _key in (
        "OSPREY_CONFIG",
        "OSPREY_WEB_PORT",
        DECLARED_BIND_ENV,
        DECLARED_WEB_PORT_ENV,
        OPERATOR_SECRET_ENV,
    ):
        monkeypatch.setenv(_key, "__unset_by_test_fixture__")
        monkeypatch.delenv(_key)
    reset_web_credentials()
    yield
    reset_web_credentials()


@pytest.fixture(autouse=True)
def _inside_a_deployment(lifecycle_repo, monkeypatch):
    """Satisfy `web()`'s deployment resolution for every launch-path test here.

    `osprey web` refuses to start outside a deployment repo, or inside one with
    nothing rendered (a configless launch silently serves a panel-less
    terminal). These tests exercise bind and port resolution, not discovery, so
    they stand in a repo with a minimal render and let the walk-up rule do its
    normal thing.

    The env var is deliberately NOT how this is arranged: `OSPREY_CONFIG` is a
    publication `web` makes for its children, not a way of telling it where to
    look, and a test that set it would be pinning a contract that does not
    exist.
    """
    stub_build(lifecycle_repo, config="web: {}\n")
    monkeypatch.chdir(lifecycle_repo)


def _free_port() -> int:
    """Reserve then release an OS-assigned port so nothing is listening on it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def _occupied_port(host: str = "127.0.0.1"):
    """Hold a real listener on an OS-assigned port for the block's duration.

    A second bind to this exact ``(host, port)`` fails with ``EADDRINUSE`` even
    with ``SO_REUSEADDR`` (that only shares TIME_WAIT sockets, not two live
    binds), so it deterministically drives the ``osprey web`` pre-flight bind
    into its busy-port branch.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    sock.listen(1)
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


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
        # The multi-user shape supplies the operator secret (deploy .env); a
        # declared bind host with no secret is a refuse-to-mint error, so this
        # bind-resolution test must stand in the real container shape.
        monkeypatch.setenv("OSPREY_TERMINAL_SECRET", "deploy-supplied-secret")
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
        # Declared bind host => container shape => the deployment supplies the
        # secret; without it the launch refuses to mint before reaching the bind.
        monkeypatch.setenv("OSPREY_TERMINAL_SECRET", "deploy-supplied-secret")
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

        # The notice is a warning now, so it lands on stderr under the ⚠ mark.
        assert "is authoritative" in result.stderr
        assert DECLARED_BIND_ENV in result.stderr

    def test_single_user_no_env_keeps_0000(self, runner, monkeypatch):
        """Without the declared env (single-user `osprey web`), --host 0.0.0.0
        must still be honored verbatim."""
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


class TestResolveWebPort:
    """Pure resolver: declared env authoritative; CLI/config only as fallback."""

    def test_declared_web_port_env_overrides_explicit_port(self):
        assert (
            resolve_web_port(
                9000, None, base=DEFAULT_PORT_BASE, env={DECLARED_WEB_PORT_ENV: "9001"}
            )
            == 9001
        )

    def test_no_env_honors_explicit_port(self):
        assert resolve_web_port(9000, None, base=DEFAULT_PORT_BASE, env={}) == 9000

    def test_deliberate_matching_declaration(self):
        """A deployment CAN declare the same port the flag already requests —
        the invariant is "declared wins", not "flag is always rejected"."""
        assert (
            resolve_web_port(
                9001, None, base=DEFAULT_PORT_BASE, env={DECLARED_WEB_PORT_ENV: "9001"}
            )
            == 9001
        )

    def test_falls_back_to_config_then_layout(self):
        assert resolve_web_port(None, 9002, base=DEFAULT_PORT_BASE, env={}) == 9002
        assert (
            resolve_web_port(None, None, base=DEFAULT_PORT_BASE, env={}) == DEFAULT_PORT_BASE + 100
        )

    def test_terminal_fallback_follows_the_caller_s_base(self):
        """The last resort is the layout's ``web`` slot at the base the CALLER
        resolved — not at the layout's own default. A deployment that moved its
        block must not have its terminal land in the block it moved out of."""
        assert resolve_web_port(None, None, base=20000, env={}) == 20100


class TestWebCommandHonorsDeclaredWebPortEnv:
    """The load-bearing wiring guard: the reconciled port must actually reach
    the server entrypoint, not just the pure resolver."""

    def _stub_launch(self, monkeypatch):
        monkeypatch.setattr("osprey.interfaces.web_terminal.run_web", lambda **_kw: None)
        monkeypatch.setattr("osprey.mcp_env.load_dotenv_from_project", lambda: None)

    def test_multiuser_env_pins_port_reaches_run_web(self, runner, monkeypatch):
        """A stale/hostile image CMD passes a mismatched --port, but the
        multi-user container has declared OSPREY_TERMINAL_WEB_PORT. The port
        that reaches run_web must be the declared one, NOT the flag —
        otherwise nginx's per-user upstream mapping desyncs from the
        container's actual listener."""
        declared_port = _free_port()
        monkeypatch.setenv(DECLARED_WEB_PORT_ENV, str(declared_port))
        captured = {}

        def _fake_run_web(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr("osprey.interfaces.web_terminal.run_web", _fake_run_web)
        monkeypatch.setattr("osprey.mcp_env.load_dotenv_from_project", lambda: None)

        result = runner.invoke(
            web,
            [
                "--port",
                str(_free_port()),
                "--shell",
                "true",
                "--skip-preflight",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert captured.get("port") == declared_port

    def test_notice_printed_when_declared_env_overrides_port_flag(self, runner, monkeypatch):
        declared_port = _free_port()
        monkeypatch.setenv(DECLARED_WEB_PORT_ENV, str(declared_port))
        self._stub_launch(monkeypatch)

        result = runner.invoke(
            web,
            [
                "--port",
                str(_free_port()),
                "--shell",
                "true",
                "--skip-preflight",
            ],
            catch_exceptions=False,
        )

        # The notice is a warning now, so it lands on stderr under the ⚠ mark.
        assert "is authoritative" in result.stderr
        assert DECLARED_WEB_PORT_ENV in result.stderr

    def test_single_user_no_env_keeps_explicit_port(self, runner, monkeypatch):
        """Without the declared env (single-user `osprey web`), --port must
        still be honored verbatim."""
        monkeypatch.delenv(DECLARED_WEB_PORT_ENV, raising=False)
        captured = {}

        def _fake_run_web(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr("osprey.interfaces.web_terminal.run_web", _fake_run_web)
        monkeypatch.setattr("osprey.mcp_env.load_dotenv_from_project", lambda: None)

        flag_port = _free_port()
        result = runner.invoke(
            web,
            ["--port", str(flag_port), "--shell", "true", "--skip-preflight"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert captured.get("port") == flag_port


class TestBusyPortFallback:
    """An UNSPECIFIED port auto-moves off a busy default (single-user QoL); a
    PINNED port — explicit ``--port`` or a DECLARED multi-user port — still
    hard-fails, so nginx's per-user upstream can never be silently desynced by
    a moved listener.
    """

    def _capture_run_web(self, monkeypatch) -> dict:
        captured: dict = {}
        monkeypatch.setattr(
            "osprey.interfaces.web_terminal.run_web", lambda **kw: captured.update(kw)
        )
        monkeypatch.setattr("osprey.mcp_env.load_dotenv_from_project", lambda: None)
        return captured

    def _fake_config(self, monkeypatch, **wt) -> None:
        """Pin config `web_terminal.{host,port}` so the bound host is
        deterministic (127.0.0.1) and matches ``_occupied_port``."""
        wt.setdefault("host", "127.0.0.1")
        monkeypatch.setattr(
            "osprey.cli.web_cmd.get_config_value",
            lambda key, default=None: (
                wt if key == "web_terminal" else ({} if default is None else default)
            ),
        )

    def test_unspecified_port_auto_moves_off_busy_default(self, runner, monkeypatch):
        captured = self._capture_run_web(monkeypatch)
        with _occupied_port() as busy:
            self._fake_config(monkeypatch, port=busy)
            result = runner.invoke(
                web, ["--shell", "true", "--skip-preflight"], catch_exceptions=False
            )

        assert result.exit_code == 0
        # Moved to some OTHER free port, and published it for children.
        assert captured.get("port") not in (None, busy)
        assert "in use" in result.output

    def test_explicit_port_flag_still_hard_fails(self, runner, monkeypatch):
        self._capture_run_web(monkeypatch)
        with _occupied_port() as busy:
            self._fake_config(monkeypatch)
            result = runner.invoke(
                web,
                ["--port", str(busy), "--shell", "true", "--skip-preflight"],
                catch_exceptions=False,
            )

        assert result.exit_code != 0
        assert "already in use" in result.output

    def test_declared_port_still_hard_fails(self, runner, monkeypatch):
        """A DECLARED multi-user port that's busy must error, never silently
        move — a moved listener would desync nginx's per-user upstream."""
        self._capture_run_web(monkeypatch)
        with _occupied_port() as busy:
            self._fake_config(monkeypatch)
            monkeypatch.setenv(DECLARED_WEB_PORT_ENV, str(busy))
            result = runner.invoke(
                web, ["--shell", "true", "--skip-preflight"], catch_exceptions=False
            )

        assert result.exit_code != 0
        assert "already in use" in result.output


class TestCookieNameAgreesWithTheTerminal:
    """``osprey chat`` and ``osprey web`` must name the same session cookie.

    ``session_cookie_name`` appends ``OSPREY_WEB_PORT`` to the cookie's base
    name, and ``chat`` publishes that value from the same resolver ``web``
    binds by. Both therefore have to take the port base from the deployment's
    own config: a chat session that fell back to the layout's default base
    while the terminal bound inside a moved block would name two cookies, and
    signing in at one would not sign the operator in at the other.
    """

    def test_chat_publishes_the_port_web_binds(self, runner, lifecycle_repo, monkeypatch):
        from osprey.cli.chat_cmd import _launch_companion_servers
        from osprey.interfaces.common_middleware import WEB_PORT_ENV

        build = stub_build(lifecycle_repo, config="deployment:\n  port_base: 20000\n")
        # No companions registered: this is about the port the launch
        # publishes, not about what it manages to start.
        monkeypatch.setattr("osprey.registry.web.FRAMEWORK_WEB_SERVERS", {})
        monkeypatch.setattr("osprey.mcp_env.load_dotenv_from_project", lambda: None)
        captured: dict = {}
        monkeypatch.setattr(
            "osprey.interfaces.web_terminal.run_web", lambda **kw: captured.update(kw)
        )

        _launch_companion_servers(build)
        chat_port = os.environ[WEB_PORT_ENV]
        # ``web`` reads this same key as the ``--port`` envvar fallback, so the
        # two resolutions must be independent to be worth comparing.
        monkeypatch.delenv(WEB_PORT_ENV)

        result = runner.invoke(web, ["--shell", "true", "--skip-preflight"], catch_exceptions=False)

        assert result.exit_code == 0
        assert captured.get("port") == 20100
        assert chat_port == str(captured["port"])


class TestCompanionProbeAttribution:
    """A companion port held by THIS repo's own roster is not a foreign listener.

    Single-user `osprey web` and multi-user roster user 0 both take index 0 of
    every port family, so a repo with `modules.web_terminals.enabled: true`
    cannot run the two side by side at one base. Sending that operator to
    `lsof` to rediscover their own deployment is the wrong diagnosis, so the
    probe attributes the port and names the two escapes that exist.
    """

    #: A base well clear of the default block, so the ports this class binds
    #: cannot collide with a real deployment on the developer's host.
    BASE = 21000
    #: The artifact gallery's index-0 slot at :attr:`BASE`. Artifacts is the one
    #: UNIVERSAL panel, so it is probed with no `web.panels` config at all.
    ARTIFACT_PORT = BASE + 200

    def _render(self, lifecycle_repo, monkeypatch, *, roster: bool) -> None:
        """Stand this repo's render on ``BASE``, with the roster on or off."""
        from osprey.utils.workspace import reset_config_cache

        build = stub_build(
            lifecycle_repo,
            config=(
                "claude_code:\n  provider: anthropic\n"
                f"deployment:\n  port_base: {self.BASE}\n"
                f"modules:\n  web_terminals:\n    enabled: {str(roster).lower()}\n"
            ),
        )
        monkeypatch.setenv("OSPREY_CONFIG", str(build / "config.yml"))
        reset_config_cache()

    @contextlib.contextmanager
    def _listener_on_artifact_port(self):
        """Hold a listener on the artifact index-0 port, or skip the test.

        The port is a fixed number rather than an OS-assigned one — that is the
        whole point, since the attribution keys off the port EQUALLING the
        family's index-0 slot — so it can genuinely be unavailable on a busy
        host. That is a fact about the machine, not a failure of the probe.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", self.ARTIFACT_PORT))
            sock.listen(1)
        except OSError as exc:
            sock.close()
            pytest.skip(f"port {self.ARTIFACT_PORT} unavailable on this host: {exc}")
        try:
            yield
        finally:
            sock.close()

    def test_companion_port_held_by_the_roster_is_attributed(self, lifecycle_repo, monkeypatch):
        from osprey.cli.web_cmd import _probe_companion_ports

        self._render(lifecycle_repo, monkeypatch, roster=True)
        with self._listener_on_artifact_port():
            failures = _probe_companion_ports()

        assert len(failures) == 1, failures
        finding = failures[0]
        assert "multi-user roster (user 0) owns this port" in finding
        # The port is named as the family's index-0 slot at THIS base, not as
        # an anonymous busy number.
        assert f"'artifact' family's index-0 slot at port base {self.BASE}" in finding
        assert str(self.ARTIFACT_PORT) in finding
        # Both escapes, and no `lsof` hunt for a process the operator owns.
        assert "artifact_server.port" in finding
        assert "--port" in finding
        assert "lsof" not in finding

    def test_companion_port_held_without_a_roster_stays_foreign(self, lifecycle_repo, monkeypatch):
        """Same listener, same port — but with the web tier off, nothing in this
        repo can account for it, so the foreign-listener wording is unchanged."""
        from osprey.cli.web_cmd import _probe_companion_ports

        self._render(lifecycle_repo, monkeypatch, roster=False)
        with self._listener_on_artifact_port():
            failures = _probe_companion_ports()

        assert len(failures) == 1, failures
        finding = failures[0]
        assert "is already in use by another process" in finding
        assert f"lsof -i :{self.ARTIFACT_PORT}" in finding
        assert "roster" not in finding

    def test_companion_probe_is_clean_when_nothing_holds_the_port(
        self, lifecycle_repo, monkeypatch
    ):
        """The roster flag alone reports nothing: attribution needs a listener."""
        from osprey.cli.web_cmd import _probe_companion_ports

        self._render(lifecycle_repo, monkeypatch, roster=True)
        assert _probe_companion_ports() == []
