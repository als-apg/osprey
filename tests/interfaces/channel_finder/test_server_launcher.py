"""Tests for data-driven web server launcher (registry.web + server_launcher)."""

from __future__ import annotations

import socket
import time
from unittest.mock import MagicMock, patch

from osprey.registry.web import FRAMEWORK_WEB_SERVERS


def _free_port() -> int:
    """Reserve then release an OS-assigned port so nothing is listening on it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _make_launcher(host: str, port: int):
    """Build a ServerLauncher wired to a fixed (host, port), auto-launch on."""
    from osprey.infrastructure.server_launcher import ServerLauncher

    return ServerLauncher(
        name="Test Server",
        config_reader=lambda: (host, port),
        auto_launch_checker=lambda: True,
        app_factory=lambda: object(),
    )


class TestMakeConfigReader:
    """Tests for _make_config_reader using catalog entries."""

    def test_defaults_when_section_empty(self):
        """Config reader returns default host/port when config section is empty."""
        from osprey.infrastructure.server_launcher import _make_config_reader

        reader = _make_config_reader(FRAMEWORK_WEB_SERVERS["channel_finder"])
        with patch(
            "osprey.infrastructure.server_launcher.load_osprey_config",
            return_value={},
        ):
            host, port = reader()
        assert host == "127.0.0.1"
        assert port == 8092

    def test_custom_values_with_web_subkey(self):
        """Config reader navigates config_web_subkey correctly."""
        from osprey.infrastructure.server_launcher import _make_config_reader

        reader = _make_config_reader(FRAMEWORK_WEB_SERVERS["channel_finder"])
        with patch(
            "osprey.infrastructure.server_launcher.load_osprey_config",
            return_value={
                "channel_finder": {"web": {"host": "0.0.0.0", "port": 9999}},
            },
        ):
            host, port = reader()
        assert host == "0.0.0.0"
        assert port == 9999

    def test_flat_config_key(self):
        """Config reader works for servers without config_web_subkey."""
        from osprey.infrastructure.server_launcher import _make_config_reader

        reader = _make_config_reader(FRAMEWORK_WEB_SERVERS["artifact"])
        with patch(
            "osprey.infrastructure.server_launcher.load_osprey_config",
            return_value={"artifact_server": {"host": "10.0.0.1", "port": 7777}},
        ):
            host, port = reader()
        assert host == "10.0.0.1"
        assert port == 7777


class TestMakeAutoLaunchChecker:
    """Tests for _make_auto_launch_checker using catalog entries."""

    def test_require_section_missing(self):
        """Auto-launch returns False when require_section=True and section is absent."""
        from osprey.infrastructure.server_launcher import _make_auto_launch_checker

        checker = _make_auto_launch_checker(FRAMEWORK_WEB_SERVERS["channel_finder"])
        with patch(
            "osprey.infrastructure.server_launcher.load_osprey_config",
            return_value={},
        ):
            assert checker() is False

    def test_require_section_present(self):
        """Auto-launch returns True when section exists (default auto_launch=True)."""
        from osprey.infrastructure.server_launcher import _make_auto_launch_checker

        checker = _make_auto_launch_checker(FRAMEWORK_WEB_SERVERS["channel_finder"])
        with patch(
            "osprey.infrastructure.server_launcher.load_osprey_config",
            return_value={"channel_finder": {"pipeline_mode": "in_context"}},
        ):
            assert checker() is True

    def test_no_require_section(self):
        """Auto-launch returns True even when section is empty if require_section=False."""
        from osprey.infrastructure.server_launcher import _make_auto_launch_checker

        checker = _make_auto_launch_checker(FRAMEWORK_WEB_SERVERS["artifact"])
        with patch(
            "osprey.infrastructure.server_launcher.load_osprey_config",
            return_value={},
        ):
            assert checker() is True


class TestBackwardCompatAliases:
    """Named ensure_* functions remain importable."""

    def test_ensure_channel_finder_server_exists(self):
        from osprey.infrastructure.server_launcher import ensure_channel_finder_server

        assert callable(ensure_channel_finder_server)

    def test_ensure_artifact_server_exists(self):
        from osprey.infrastructure.server_launcher import ensure_artifact_server

        assert callable(ensure_artifact_server)

    def test_ensure_ariel_server_exists(self):
        from osprey.infrastructure.server_launcher import ensure_ariel_server

        assert callable(ensure_ariel_server)


class TestEnsureRunningOwnership:
    """ensure_running must own the port, not trust a bare /health 200.

    Regression coverage for the false-positive described in issue #327: a
    stale/foreign responder answering /health made the launcher skip binding,
    leaving the panel unbacked (proxy 502) after a restart.
    """

    def test_launches_when_port_free_despite_health_200(self):
        """A /health 200 with no real listener must NOT suppress the launch.

        This is the core bug: `_is_running()` returning True (stale/foreign
        responder) previously short-circuited the launch even though nothing
        was actually bound to the port.
        """
        launcher = _make_launcher("127.0.0.1", _free_port())

        with (
            patch.object(launcher, "_is_running", return_value=True),
            patch.object(launcher, "_launch_in_thread") as mock_launch,
        ):
            launcher.ensure_running()

        mock_launch.assert_called_once()

    def test_launches_when_port_genuinely_free(self):
        """No listener + no responder → launch and own the port."""
        launcher = _make_launcher("127.0.0.1", _free_port())

        with (
            patch.object(launcher, "_is_running", return_value=False),
            patch.object(launcher, "_launch_in_thread") as mock_launch,
        ):
            launcher.ensure_running()

        mock_launch.assert_called_once()

    def test_waits_out_dying_predecessor_then_launches(self):
        """A predecessor that releases the port during the grace window is waited out."""
        launcher = _make_launcher("127.0.0.1", _free_port())
        # Held on the first probe, released on the second (predecessor shut down).
        listener_states = iter([True, False])

        with (
            patch.object(
                launcher,
                "_port_has_listener",
                side_effect=lambda *_: next(listener_states, False),
            ),
            patch.object(launcher, "_launch_in_thread") as mock_launch,
            patch("osprey.infrastructure.server_launcher.time.sleep"),
        ):
            launcher.ensure_running()

        mock_launch.assert_called_once()

    def test_defers_to_persistent_external_server(self):
        """A port held for the full grace window that serves /health → defer, don't launch."""
        launcher = _make_launcher("127.0.0.1", _free_port())

        with (
            patch.object(launcher, "_port_has_listener", return_value=True),
            patch.object(launcher, "_is_running", return_value=True),
            patch.object(launcher, "_launch_in_thread") as mock_launch,
            patch("osprey.infrastructure.server_launcher.time.sleep"),
        ):
            launcher.ensure_running()

        mock_launch.assert_not_called()
        assert launcher._launched is True

    def test_warns_when_port_held_by_non_responder(self):
        """Port held for the full grace window with no /health → loud warning, no silent skip."""
        launcher = _make_launcher("127.0.0.1", _free_port())

        with (
            patch.object(launcher, "_port_has_listener", return_value=True),
            patch.object(launcher, "_is_running", return_value=False),
            patch.object(launcher, "_launch_in_thread") as mock_launch,
            patch("osprey.infrastructure.server_launcher.time.sleep"),
            patch("osprey.infrastructure.server_launcher.logger.warning") as mock_warn,
        ):
            launcher.ensure_running()

        mock_launch.assert_not_called()
        mock_warn.assert_called_once()

    def test_non_responder_does_not_latch_and_self_heals(self):
        """A non-responder outcome must not latch, so a later call can self-heal.

        Regression guard for the artifact_store per-save relaunch pattern: if the
        first call hit a foreign holder, a later call (after the port frees) must
        still launch instead of being permanently short-circuited.
        """
        launcher = _make_launcher("127.0.0.1", _free_port())

        # First call: held by a non-responder → warn, throttle, no launch, no latch.
        with (
            patch.object(launcher, "_port_has_listener", return_value=True),
            patch.object(launcher, "_is_running", return_value=False),
            patch.object(launcher, "_launch_in_thread") as mock_launch,
            patch("osprey.infrastructure.server_launcher.time.sleep"),
        ):
            launcher.ensure_running()
        mock_launch.assert_not_called()
        assert launcher._launched is False

        # Cooldown elapsed and the port is now free → launch (self-heal).
        launcher._retry_not_before = 0.0
        with (
            patch.object(launcher, "_port_has_listener", return_value=False),
            patch.object(launcher, "_launch_in_thread") as mock_launch2,
        ):
            launcher.ensure_running()
        mock_launch2.assert_called_once()

    def test_cooldown_short_circuits_reprobe(self):
        """Within the retry cooldown, ensure_running must not re-probe (avoids grace cost)."""
        launcher = _make_launcher("127.0.0.1", _free_port())
        launcher._retry_not_before = time.monotonic() + 1000  # far future

        with (
            patch.object(launcher, "_port_has_listener") as mock_probe,
            patch.object(launcher, "_launch_in_thread") as mock_launch,
        ):
            launcher.ensure_running()

        mock_probe.assert_not_called()
        mock_launch.assert_not_called()

    def test_dead_thread_health_200_not_marked_launched(self):
        """Post-launch: a /health 200 from a dead thread (foreign responder) is not trusted."""
        launcher = _make_launcher("127.0.0.1", _free_port())
        dead_thread = MagicMock()
        dead_thread.is_alive.return_value = False

        with (
            patch(
                "osprey.infrastructure.server_launcher.threading.Thread",
                return_value=dead_thread,
            ),
            patch.object(launcher, "_is_running", return_value=True),
            patch("osprey.infrastructure.server_launcher.time.sleep"),
        ):
            launcher._launch_in_thread("127.0.0.1", 12345)

        assert launcher._launched is False


class TestLoopbackFor:
    """Wildcard bind hosts are normalized to a client-reachable loopback."""

    def test_wildcard_ipv4(self):
        from osprey.infrastructure.server_launcher import _loopback_for

        assert _loopback_for("0.0.0.0") == "127.0.0.1"
        assert _loopback_for("") == "127.0.0.1"

    def test_wildcard_ipv6(self):
        from osprey.infrastructure.server_launcher import _loopback_for

        assert _loopback_for("::") == "::1"

    def test_concrete_host_passthrough(self):
        from osprey.infrastructure.server_launcher import _loopback_for

        assert _loopback_for("10.0.0.5") == "10.0.0.5"
        assert _loopback_for("127.0.0.1") == "127.0.0.1"


class TestPortHasListener:
    """The connect-probe distinguishes a bound port from a free one."""

    def test_returns_false_for_free_port(self):
        launcher = _make_launcher("127.0.0.1", _free_port())
        host, port = launcher._config_reader()
        assert launcher._port_has_listener(host, port) is False

    def test_returns_true_for_bound_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            launcher = _make_launcher("127.0.0.1", port)
            assert launcher._port_has_listener("127.0.0.1", port) is True
