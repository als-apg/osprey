"""Tests for data-driven web server launcher (registry.web + server_launcher).

This module is the launcher's ownership STATE MACHINE: which of the three
outcomes — launch, adopt, refuse — each situation reaches, and what a second
call does afterwards. The probes those transitions are built from have their
own matrices in ``tests/infrastructure/test_server_launcher.py`` (family and
errno rows for ``_port_is_bindable``, the adoption verdict rows, the repeat
refusal accounting); they are not repeated here, and the probes are mocked
throughout so the transitions do not depend on a kernel.
"""

from __future__ import annotations

import socket
import time
from unittest.mock import MagicMock, patch

from osprey.registry.web import FRAMEWORK_WEB_SERVERS, framework_web_port_default


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


def _probes(unauthenticated: int | None, credentialed: int | None = None):
    """Return a ``_probe_status`` stand-in answering per credential offered."""

    def _probe(_host, _port, secret=None):
        return credentialed if secret else unauthenticated

    return _probe


class TestLauncherConfigReader:
    """Each launcher's config reader resolves its own catalog entry's address."""

    @staticmethod
    def _reader(key: str, config: dict):
        from osprey.infrastructure import server_launcher

        reader = server_launcher._launchers[key]._config_reader
        with patch("osprey.utils.workspace.load_osprey_config", return_value=config):
            return reader()

    def test_defaults_when_section_empty(self):
        """Config reader returns default host/port when config section is empty.

        The port is not a constant: with no ``deployment.port_base`` in the
        config it is the Channel Finder's slot at the layout's default base.
        """
        host, port = self._reader("channel_finder", {})
        assert host == "127.0.0.1"
        assert port == framework_web_port_default("channel_finder")

    def test_custom_values_with_web_subkey(self):
        """Config reader navigates config_web_subkey correctly."""
        host, port = self._reader(
            "channel_finder",
            {"channel_finder": {"web": {"host": "0.0.0.0", "port": 9999}}},
        )
        assert host == "0.0.0.0"
        assert port == 9999

    def test_flat_config_key(self):
        """Config reader works for servers without config_web_subkey."""
        host, port = self._reader(
            "artifact", {"artifact_server": {"host": "10.0.0.1", "port": 7777}}
        )
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
    leaving the panel unbacked (proxy 502) after a restart. Ownership is now
    decided by whether ``_port_is_bindable`` says this process could bind the
    port; a held port is stood down for only when a credentialed probe pair
    attributes the listener to this deployment.
    """

    def test_launches_when_port_bindable_despite_health_200(self):
        """A /health 200 over a bindable port must NOT suppress the launch.

        This is the core of #327: ``/health`` is exempt from the auth gate, so
        every OSPREY checkout on the machine answers it 200 and it says nothing
        about whether this port is ours. The bind question is the only one asked.
        """
        launcher = _make_launcher("127.0.0.1", _free_port())

        with (
            patch.object(launcher, "_port_is_bindable", return_value=True),
            patch.object(launcher, "_port_answers_connect", return_value=False),
            patch.object(launcher, "_is_running", return_value=True),
            patch.object(launcher, "_launch_in_thread") as mock_launch,
        ):
            launcher.ensure_running()

        mock_launch.assert_called_once()

    def test_launches_when_port_genuinely_free(self):
        """Bindable, nothing answering → launch and own the port."""
        launcher = _make_launcher("127.0.0.1", _free_port())

        with (
            patch.object(launcher, "_port_is_bindable", return_value=True),
            patch.object(launcher, "_port_answers_connect", return_value=False),
            patch.object(launcher, "_is_running", return_value=False),
            patch.object(launcher, "_launch_in_thread") as mock_launch,
        ):
            launcher.ensure_running()

        mock_launch.assert_called_once()

    def test_waits_out_dying_predecessor_then_launches(self):
        """A predecessor that releases the port during the grace window is waited out."""
        launcher = _make_launcher("127.0.0.1", _free_port())
        # Unbindable on the first probe, bindable on the second (predecessor gone).
        bindable = iter([False, True])

        with (
            patch.object(
                launcher,
                "_port_is_bindable",
                side_effect=lambda *_: next(bindable, True),
            ),
            patch.object(launcher, "_port_answers_connect", return_value=False),
            patch.object(launcher, "_launch_in_thread") as mock_launch,
            patch("osprey.infrastructure.server_launcher.time.sleep"),
        ):
            launcher.ensure_running()

        mock_launch.assert_called_once()

    def test_adopts_a_held_port_whose_listener_accepts_our_secret(self):
        """401 unauthenticated + non-401 credentialed → our own panel; stand down.

        This is the only shape that justifies not launching: the listener gates,
        and it accepts THIS process's operator secret, so it is this deployment's
        panel under a shared ``OSPREY_TERMINAL_SECRET``.
        """
        launcher = _make_launcher("127.0.0.1", _free_port())

        with (
            patch.object(launcher, "_port_is_bindable", return_value=False),
            patch.object(launcher, "_operator_secret", return_value="operator-secret"),
            patch.object(launcher, "_probe_status", side_effect=_probes(401, 200)),
            patch.object(launcher, "_launch_in_thread") as mock_launch,
            patch("osprey.infrastructure.server_launcher.time.sleep"),
            patch("osprey.infrastructure.server_launcher.logger.warning") as mock_warn,
        ):
            launcher.ensure_running()

        mock_launch.assert_not_called()
        assert launcher._launched is True
        mock_warn.assert_not_called()

    def test_refuses_a_held_port_that_serves_health_but_cannot_be_attributed(self):
        """A /health 200 does not earn a stand-down either — it is refused, not adopted.

        The superseded contract deferred to any listener that answered /health,
        which handed the operator a panel backed by a server this process cannot
        talk to. An ungated 200 to the adoption probe now refuses: no launch (the
        port is not ours to take), no latch, and a warning naming the conflict.
        """
        launcher = _make_launcher("127.0.0.1", _free_port())

        with (
            patch.object(launcher, "_port_is_bindable", return_value=False),
            patch.object(launcher, "_operator_secret", return_value="operator-secret"),
            patch.object(launcher, "_probe_status", side_effect=_probes(200)),
            patch.object(launcher, "_is_running", return_value=True),
            patch.object(launcher, "_launch_in_thread") as mock_launch,
            patch("osprey.infrastructure.server_launcher.time.sleep"),
            patch("osprey.infrastructure.server_launcher.logger.warning") as mock_warn,
        ):
            launcher.ensure_running()

        mock_launch.assert_not_called()
        assert launcher._launched is False
        assert launcher._retry_not_before > 0
        mock_warn.assert_called_once()

    def test_warns_when_port_held_by_a_listener_that_never_answers(self):
        """Held for the full grace window, nothing answering → warn, no silent skip."""
        launcher = _make_launcher("127.0.0.1", _free_port())

        with (
            patch.object(launcher, "_port_is_bindable", return_value=False),
            patch.object(launcher, "_operator_secret", return_value="operator-secret"),
            patch.object(launcher, "_probe_status", side_effect=_probes(None)),
            patch.object(launcher, "_launch_in_thread") as mock_launch,
            patch("osprey.infrastructure.server_launcher.time.sleep"),
            patch("osprey.infrastructure.server_launcher.logger.warning") as mock_warn,
        ):
            launcher.ensure_running()

        mock_launch.assert_not_called()
        assert launcher._launched is False
        mock_warn.assert_called_once()

    def test_a_refusal_does_not_latch_and_self_heals(self):
        """A refusal must not latch, so a later call can self-heal.

        Regression guard for the artifact_store per-save relaunch pattern: if the
        first call hit a foreign holder, a later call (after the port frees) must
        still launch instead of being permanently short-circuited.
        """
        launcher = _make_launcher("127.0.0.1", _free_port())

        # First call: held by an unattributable listener → warn, throttle, no latch.
        with (
            patch.object(launcher, "_port_is_bindable", return_value=False),
            patch.object(launcher, "_operator_secret", return_value="operator-secret"),
            patch.object(launcher, "_probe_status", side_effect=_probes(None)),
            patch.object(launcher, "_launch_in_thread") as mock_launch,
            patch("osprey.infrastructure.server_launcher.time.sleep"),
        ):
            launcher.ensure_running()
        mock_launch.assert_not_called()
        assert launcher._launched is False

        # Cooldown elapsed and the port is now bindable → launch (self-heal).
        launcher._retry_not_before = 0.0
        with (
            patch.object(launcher, "_port_is_bindable", return_value=True),
            patch.object(launcher, "_port_answers_connect", return_value=False),
            patch.object(launcher, "_launch_in_thread") as mock_launch2,
        ):
            launcher.ensure_running()
        mock_launch2.assert_called_once()

    def test_a_second_held_port_pass_skips_the_grace_window(self):
        """The window is a predecessor's, and a refused port has no predecessor.

        Once a refusal has established that the holder is staying, a per-save
        caller must not pay 2.5s of sleeps under the launcher's lock on every
        call. The bind question is still re-asked first, so a freed port still
        launches (covered above).
        """
        launcher = _make_launcher("127.0.0.1", _free_port())
        sleep = MagicMock()

        for _pass in range(2):
            launcher._retry_not_before = 0.0
            with (
                patch.object(launcher, "_port_is_bindable", return_value=False),
                patch.object(launcher, "_operator_secret", return_value="operator-secret"),
                patch.object(launcher, "_probe_status", side_effect=_probes(None)),
                patch.object(launcher, "_launch_in_thread"),
                patch("osprey.infrastructure.server_launcher.time.sleep", sleep),
            ):
                launcher.ensure_running()
            if _pass == 0:
                assert sleep.call_count == launcher._release_grace_attempts
                sleep.reset_mock()

        sleep.assert_not_called()

    def test_cooldown_short_circuits_reprobe(self):
        """Within the retry cooldown, ensure_running must not re-probe (avoids grace cost)."""
        launcher = _make_launcher("127.0.0.1", _free_port())
        launcher._retry_not_before = time.monotonic() + 1000  # far future

        with (
            patch.object(launcher, "_port_is_bindable") as mock_probe,
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


class TestTheTwoProbesAnswerDifferentQuestions:
    """One verdict, one diagnostic — the split the state machine is built on.

    Only the contrast is pinned here, on the one situation where both probes
    agree there is a real listener. The family, address and errno rows for each
    probe live in ``tests/infrastructure/test_server_launcher.py``.
    """

    def test_a_free_port_is_bindable_and_answers_nothing(self):
        launcher = _make_launcher("127.0.0.1", _free_port())
        host, port = launcher._config_reader()

        assert launcher._port_is_bindable(host, port) is True
        assert launcher._port_answers_connect(host, port) is False

    def test_a_bound_port_is_unbindable_and_answers_a_connect(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            launcher = _make_launcher("127.0.0.1", port)

            assert launcher._port_is_bindable("127.0.0.1", port) is False
            assert launcher._port_answers_connect("127.0.0.1", port) is True
