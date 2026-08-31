"""Unit tests for the generic ``ServerLauncher`` plumbing.

The ownership state machine (port-free / held-then-freed / held-throughout) is
covered in ``tests/interfaces/channel_finder/test_server_launcher.py``. This
module targets the parts not exercised there: the launcher table's wiring to the
shared address resolver, the data-driven callback builders (``_resolve_dotted``,
``_make_app_factory`` import/kwargs/error branches), the ``ensure_web_server``
dispatch table, and the two ``ensure_running`` short-circuits (auto-launch
disabled, already launched).
"""

from __future__ import annotations

import email.message
import errno
import inspect
import io
import os
import socket
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.response import addinfourl

import pytest

from osprey.infrastructure import server_launcher
from osprey.infrastructure.server_launcher import (
    ServerLauncher,
    _make_app_factory,
    _resolve_dotted,
    ensure_web_server,
)
from osprey.registry.web import (
    FRAMEWORK_WEB_SERVERS,
    WebServerConfigDepthError,
    WebServerDefinition,
    framework_web_port_default,
    resolve_web_server_address,
)


def _free_port() -> int:
    """Reserve then release an OS-assigned port so nothing is listening on it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _make_launcher(host: str = "127.0.0.1", port: int | None = None) -> ServerLauncher:
    """Build a ServerLauncher wired to a fixed (host, port), auto-launch on."""
    settled = _free_port() if port is None else port
    return ServerLauncher(
        name="Test Server",
        config_reader=lambda: (host, settled),
        auto_launch_checker=lambda: True,
        app_factory=lambda: object(),
    )


def _probes(unauthenticated: int | None, credentialed: int | None = None):
    """Return a ``_probe_status`` stand-in answering per credential.

    The credentialed answer is keyed on the ``secret`` argument, which is also
    how a test can see whether the secret was offered at all.
    """

    def _probe(_host, _port, secret=None):
        return credentialed if secret else unauthenticated

    return _probe


def _rendered(mock_log) -> str:
    """Render a lazily-formatted ``logger`` call into the line an operator reads."""
    call = mock_log.call_args
    return call.args[0] % call.args[1:]


def _lines(mock_log) -> list[str]:
    """Render every call to a lazily-formatted ``logger`` method, in order."""
    return [call.args[0] % call.args[1:] for call in mock_log.call_args_list]


class _FakeSocket:
    """A stand-in for a socket that answers ``bind`` from a script.

    Records what the probe asked for — family, socket type, every ``setsockopt``
    and the exact address handed to ``bind`` — so a test can pin the syscall
    shape on a host whose kernel would answer differently.
    """

    def __init__(self, family, sock_type, log, bind_error):
        self._log = log
        self._bind_error = bind_error
        log["family"] = family
        log["type"] = sock_type

    def __enter__(self) -> _FakeSocket:
        return self

    def __exit__(self, *_exc) -> bool:
        self._log["closed"] = True
        return False

    def setsockopt(self, level, option, value) -> None:
        self._log["sockopts"].append((level, option, value))

    def bind(self, address) -> None:
        self._log["bound"] = address
        # Snapshot the options as they stood when bind was called: an option set
        # afterwards would not have applied to this bind, so ordering is part of
        # the contract, not an incidental detail.
        self._log["sockopts_at_bind"] = list(self._log["sockopts"])
        if self._bind_error is not None:
            raise self._bind_error


def _fake_socket(bind_error: OSError | None = None):
    """Return a ``socket.socket`` stand-in and the log it writes to."""
    log: dict = {
        "family": None,
        "type": None,
        "sockopts": [],
        "sockopts_at_bind": [],
        "bound": None,
        "closed": False,
    }

    def _factory(family, sock_type):
        return _FakeSocket(family, sock_type, log, bind_error)

    return _factory, log


def _require_ipv6_loopback() -> None:
    """Skip unless this host can actually bind ``::1`` — CI images vary.

    Asked from inside the test rather than from a ``skipif`` condition: a
    ``skipif`` argument is evaluated at import, and opening a socket while the
    module is merely being collected is what ``test_import_time_audit`` forbids.
    """
    if not socket.has_ipv6:
        pytest.skip("host has no IPv6 support")
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
            sock.bind(("::1", 0))
    except OSError as exc:
        pytest.skip(f"host has no usable IPv6 loopback ({exc})")


class _Fake302Handler(urllib.request.HTTPHandler):
    """Answer any ``http://`` request with a 302, opening no socket.

    Subclasses ``HTTPHandler`` so ``build_opener`` drops the real transport,
    which is what keeps this test off the network.
    """

    def http_open(self, req):  # noqa: D102 - the class docstring says it
        headers = email.message.Message()
        headers["Location"] = "http://127.0.0.1:9/somewhere-else"
        response = addinfourl(io.BytesIO(b""), headers, req.full_url, 302)
        response.msg = "Found"
        return response


def _defn(**overrides) -> WebServerDefinition:
    """Build a WebServerDefinition with test-friendly defaults."""
    base = {
        "name": "Test Server",
        "factory_path": "types:SimpleNamespace",
        "config_key": "test_server",
        "panel_id": "test-server",
    }
    base.update(overrides)
    return WebServerDefinition(**base)


# ---------------------------------------------------------------------------
# Address resolution — one shared derivation, wired into every launcher
# ---------------------------------------------------------------------------


class TestLauncherAddressResolution:
    """Config/default/env resolution itself is covered by the resolver's own tests
    (``tests/mcp_server/test_artifact_port_resolution.py``). What matters here is
    that the launcher table is actually wired to it, per key."""

    def test_env_var_name_derives_from_config_key(self):
        assert _defn(config_key="artifact_server").port_env_var == "OSPREY_ARTIFACT_SERVER_PORT"

    def test_launcher_reader_is_the_shared_resolver_bound_to_its_key(self, monkeypatch):
        monkeypatch.setattr(
            "osprey.utils.workspace.load_osprey_config",
            lambda: {"artifact_server": {"host": "10.0.0.1", "port": 9000}},
        )
        monkeypatch.setenv("OSPREY_ARTIFACT_SERVER_PORT", "9191")

        host, port = server_launcher._launchers["artifact"]._config_reader()
        assert (host, port) == ("10.0.0.1", 9191)  # env wins over the configured 9000

    def test_every_launcher_resolves_its_own_key(self, monkeypatch):
        monkeypatch.setattr("osprey.utils.workspace.load_osprey_config", lambda: {})
        for key, defn in FRAMEWORK_WEB_SERVERS.items():
            monkeypatch.delenv(defn.port_env_var, raising=False)
            _host, port = server_launcher._launchers[key]._config_reader()
            # The default is computed from the key rather than carried on the
            # definition: a definition does not know which port family it is
            # filed under, and the family is what the layout names.
            assert port == framework_web_port_default(key), f"{key} resolved another server's port"


# ---------------------------------------------------------------------------
# auto_launch — read at exactly one depth, and never silently at the wrong one
# ---------------------------------------------------------------------------


class TestAutoLaunchNesting:
    """``auto_launch: false`` must never leave the panel launching.

    The six companion servers disagree about depth: three take
    ``<section>.web.auto_launch``, three take ``<section>.auto_launch``. Reading
    the wrong depth returned the *default* — so ``ariel.auto_launch: false``
    (correct: ``ariel.web.auto_launch``) read back as ``True`` and the panel the
    operator had just switched off started anyway.
    """

    @staticmethod
    def _checker(key: str, config: dict, monkeypatch) -> bool:
        monkeypatch.setattr(
            "osprey.infrastructure.server_launcher.load_osprey_config", lambda: config
        )
        return server_launcher._make_auto_launch_checker(FRAMEWORK_WEB_SERVERS[key])()

    @pytest.mark.parametrize(
        ("key", "config"),
        [
            ("ariel", {"ariel": {"web": {"auto_launch": False}}}),
            ("system_health", {"health": {"web": {"auto_launch": False}}}),
            ("artifact", {"artifact_server": {"auto_launch": False}}),
            ("okf", {"facility_knowledge": {"bundle_path": "x", "auto_launch": False}}),
        ],
    )
    def test_auto_launch_false_at_the_read_depth_disables_the_panel(self, key, config, monkeypatch):
        assert self._checker(key, config, monkeypatch) is False

    @pytest.mark.parametrize(
        ("key", "config"),
        [
            # Nested readers: the key written one level too shallow.
            ("ariel", {"ariel": {"auto_launch": False}}),
            ("channel_finder", {"channel_finder": {"auto_launch": False}}),
            ("system_health", {"health": {"auto_launch": False}}),
            # Flat readers: the key written one level too deep.
            ("artifact", {"artifact_server": {"web": {"auto_launch": False}}}),
            ("lattice_dashboard", {"lattice_dashboard": {"web": {"auto_launch": False}}}),
            ("okf", {"facility_knowledge": {"bundle_path": "x", "web": {"auto_launch": False}}}),
        ],
    )
    def test_auto_launch_false_at_the_wrong_depth_never_starts_the_panel(
        self, key, config, monkeypatch
    ):
        """The one thing that must not happen is a silent ``True``."""
        with pytest.raises(WebServerConfigDepthError) as excinfo:
            self._checker(key, config, monkeypatch)
        assert "auto_launch" in str(excinfo.value)

    def test_the_error_names_both_the_wrong_key_and_the_right_one(self, monkeypatch):
        with pytest.raises(WebServerConfigDepthError) as excinfo:
            self._checker("ariel", {"ariel": {"auto_launch": False}}, monkeypatch)
        message = str(excinfo.value)
        assert "ariel.auto_launch" in message
        assert "ariel.web.auto_launch" in message

    def test_a_misplaced_port_is_refused_too(self, monkeypatch):
        """``artifact_server.web.port`` was inert — the gallery stayed on 8086."""
        monkeypatch.setattr(
            "osprey.utils.workspace.load_osprey_config",
            lambda: {"artifact_server": {"web": {"port": 9999}}},
        )
        with pytest.raises(WebServerConfigDepthError, match=r"artifact_server\.web\.port"):
            server_launcher._launchers["artifact"]._config_reader()

    def test_keys_at_the_read_depth_are_untouched(self, monkeypatch):
        """The check must not flag the legitimate spelling of either shape."""
        monkeypatch.setattr(
            "osprey.utils.workspace.load_osprey_config",
            lambda: {
                "ariel": {"web": {"port": 1111}, "database": {"uri": "postgres://x"}},
                "artifact_server": {"port": 2222, "categories": {}},
            },
        )
        assert resolve_web_server_address("ariel")[1] == 1111
        assert resolve_web_server_address("artifact")[1] == 2222


# ---------------------------------------------------------------------------
# _resolve_dotted — nested config traversal
# ---------------------------------------------------------------------------


class TestResolveDotted:
    def test_traverses_nested_keys(self):
        cfg = {"a": {"b": {"c": 42}}}
        assert _resolve_dotted(cfg, "a.b.c") == 42

    def test_missing_key_returns_none(self):
        assert _resolve_dotted({"a": {"b": {}}}, "a.b.c") is None

    def test_non_dict_midway_returns_none(self):
        # ``a`` resolves to an int, so ``a.b`` cannot continue.
        assert _resolve_dotted({"a": 5}, "a.b") is None

    def test_single_key(self):
        assert _resolve_dotted({"only": "value"}, "only") == "value"


# ---------------------------------------------------------------------------
# _make_app_factory — dynamic import, kwargs, error handling
# ---------------------------------------------------------------------------


class TestMakeAppFactory:
    def test_basic_factory_invocation(self):
        factory = _make_app_factory(_defn(factory_path="types:SimpleNamespace"))
        app = factory()
        assert isinstance(app, SimpleNamespace)

    def test_pass_workspace_forwards_workspace_root(self):
        factory = _make_app_factory(
            _defn(factory_path="types:SimpleNamespace", pass_workspace=True)
        )
        app = factory(workspace_root="/tmp/ws")
        assert app.workspace_root == "/tmp/ws"

    def test_factory_config_kwargs_resolved_from_config(self, monkeypatch):
        defn = _defn(
            factory_path="types:SimpleNamespace",
            factory_config_kwargs={"bundle_path": "facility_knowledge.bundle_path"},
        )
        monkeypatch.setattr(
            server_launcher,
            "load_osprey_config",
            lambda: {"facility_knowledge": {"bundle_path": "/data/okf"}},
        )
        app = _make_app_factory(defn)()
        assert app.bundle_path == "/data/okf"

    def test_import_error_uses_custom_message(self):
        defn = _defn(
            factory_path="osprey._does_not_exist_xyz:create_app",
            import_error_message="install the extra to enable this panel",
        )
        with pytest.raises(ImportError, match="install the extra"):
            _make_app_factory(defn)()

    def test_import_error_without_custom_message_propagates(self):
        defn = _defn(factory_path="osprey._does_not_exist_xyz:create_app")
        with pytest.raises(ImportError) as exc:
            _make_app_factory(defn)()
        # The original import error, not a rewritten one.
        assert "install the extra" not in str(exc.value)


# ---------------------------------------------------------------------------
# ensure_web_server / named aliases — dispatch table
# ---------------------------------------------------------------------------


class TestEnsureWebServerDispatch:
    def test_dispatches_to_the_named_launcher(self, monkeypatch):
        fake = MagicMock()
        monkeypatch.setitem(server_launcher._launchers, "artifact", fake)
        ensure_web_server("artifact")
        fake.ensure_running.assert_called_once_with()

    def test_unknown_key_raises_keyerror(self):
        with pytest.raises(KeyError):
            ensure_web_server("no-such-server")

    def test_named_alias_targets_expected_key(self, monkeypatch):
        fake = MagicMock()
        monkeypatch.setitem(server_launcher._launchers, "ariel", fake)
        server_launcher.ensure_ariel_server()
        fake.ensure_running.assert_called_once_with()


# ---------------------------------------------------------------------------
# ensure_running — the two early-exit guards
# ---------------------------------------------------------------------------


class TestEnsureRunningShortCircuits:
    def _launcher(self, auto_launch: bool) -> ServerLauncher:
        return ServerLauncher(
            name="Guarded",
            config_reader=MagicMock(return_value=("127.0.0.1", _free_port())),
            auto_launch_checker=lambda: auto_launch,
            app_factory=lambda: object(),
        )

    def test_no_launch_when_auto_launch_disabled(self):
        launcher = self._launcher(auto_launch=False)
        with patch.object(launcher, "_launch_in_thread") as mock_launch:
            launcher.ensure_running()
        mock_launch.assert_not_called()
        launcher._config_reader.assert_not_called()

    def test_no_launch_when_already_launched(self):
        launcher = self._launcher(auto_launch=True)
        launcher._launched = True
        with patch.object(launcher, "_launch_in_thread") as mock_launch:
            launcher.ensure_running()
        mock_launch.assert_not_called()
        launcher._config_reader.assert_not_called()


# ---------------------------------------------------------------------------
# _is_running — /health probe distinct from the connect-probe
# ---------------------------------------------------------------------------


class TestIsRunning:
    def test_free_port_is_not_running(self):
        launcher = ServerLauncher(
            name="Probe",
            config_reader=lambda: ("127.0.0.1", _free_port()),
            auto_launch_checker=lambda: True,
            app_factory=lambda: object(),
        )
        host, port = launcher._config_reader()
        assert launcher._is_running(host, port) is False


# ---------------------------------------------------------------------------
# Adoption — a held port is stood down for only on credentialed attribution
# ---------------------------------------------------------------------------


class TestHeldPortAdoption:
    """A port we cannot bind is adopted only when a credential says it is ours.

    Standing down for a listener means the panel is served by a process this
    one does not own. That is right when the listener is this deployment's own
    panel under a shared operator secret, and wrong for anything else — an
    unrelated server, or one holding a different secret, leaves the operator
    with a panel whose calls are refused. The evidence is two-sided: a 401 to
    an unauthenticated probe (it gates) AND a non-401 to a credentialed one (it
    accepts OUR credential). ``/health`` cannot supply either half.
    """

    @staticmethod
    def _run(launcher, probe, secret="operator-secret"):
        """Drive ensure_running down the held-port path with *probe* answering."""
        with (
            patch.object(launcher, "_port_is_bindable", return_value=False),
            patch.object(launcher, "_operator_secret", return_value=secret),
            patch.object(launcher, "_probe_status", side_effect=probe),
            patch.object(launcher, "_launch_in_thread") as mock_launch,
            patch("osprey.infrastructure.server_launcher.time.sleep"),
            patch("osprey.infrastructure.server_launcher.logger.warning") as mock_warn,
            patch("osprey.infrastructure.server_launcher.logger.info") as mock_info,
        ):
            launcher.ensure_running()
        return mock_launch, mock_warn, mock_info

    def test_gated_listener_accepting_our_secret_is_adopted(self):
        """401 unauthenticated + 200 credentialed → our own panel; stand down and latch."""
        launcher = _make_launcher()

        mock_launch, mock_warn, mock_info = self._run(launcher, _probes(401, 200))

        mock_launch.assert_not_called()
        assert launcher._launched is True
        mock_warn.assert_not_called()
        assert "adopting" in _rendered(mock_info)

    def test_ungated_listener_answering_200_is_refused(self):
        """An open remote serving anyone is not ours, however healthy it looks."""
        host, port = "127.0.0.1", _free_port()
        launcher = _make_launcher(host, port)

        mock_launch, mock_warn, _ = self._run(launcher, _probes(200))

        mock_launch.assert_not_called()
        assert launcher._launched is False
        mock_warn.assert_called_once()
        line = _rendered(mock_warn)
        assert f"{host}:{port}" in line
        assert "HTTP 200" in line
        assert "not attempted" in line

    def test_listener_holding_a_different_secret_is_refused(self):
        """401 to both probes: it gates, but not for us — the credentials diverged."""
        launcher = _make_launcher()

        mock_launch, mock_warn, _ = self._run(launcher, _probes(401, 401))

        mock_launch.assert_not_called()
        assert launcher._launched is False
        mock_warn.assert_called_once()
        assert "HTTP 401" in _rendered(mock_warn)

    def test_listener_that_never_answers_is_refused(self):
        """A non-HTTP process holding the port cannot be attributed either."""
        launcher = _make_launcher()

        mock_launch, mock_warn, _ = self._run(launcher, _probes(None))

        mock_launch.assert_not_called()
        assert launcher._launched is False
        mock_warn.assert_called_once()
        assert "no answer" in _rendered(mock_warn)

    def test_a_credentialed_probe_that_goes_unanswered_is_refused(self):
        """401 then silence: the listener gates, but stopped answering mid-verdict.

        Attribution needs a non-401 *answer* to the credentialed probe, and no
        answer is not one — a listener that gates and then times out is exactly
        as unattributable as one that refuses our secret.
        """
        launcher = _make_launcher()

        mock_launch, mock_warn, _ = self._run(launcher, _probes(401, None))

        mock_launch.assert_not_called()
        assert launcher._launched is False
        mock_warn.assert_called_once()
        line = _rendered(mock_warn)
        assert "HTTP 401" in line
        assert "credentialed probe: no answer" in line

    def test_missing_local_operator_secret_is_refused(self):
        """With no credential of our own there is no attribution to make."""
        launcher = _make_launcher()

        mock_launch, mock_warn, _ = self._run(launcher, _probes(401, 200), secret=None)

        mock_launch.assert_not_called()
        assert launcher._launched is False
        mock_warn.assert_called_once()
        assert "no operator secret" in _rendered(mock_warn)

    def test_our_secret_is_never_offered_to_an_ungated_listener(self):
        """A listener that served an ungated 200 must not be handed the secret.

        It has already demonstrated that it checks no credential, so sending
        one teaches us nothing and discloses it to a stranger.
        """
        launcher = _make_launcher()
        offered: list[str | None] = []

        def _record(_host, _port, secret=None):
            offered.append(secret)
            return 200 if secret is None else 200

        self._run(launcher, _record)

        assert offered == [None]

    def test_a_refusal_does_not_latch_and_self_heals_when_the_port_frees(self):
        """The per-save caller must recover once the conflicting listener exits."""
        launcher = _make_launcher()

        self._run(launcher, _probes(200))
        assert launcher._launched is False
        assert launcher._retry_not_before > 0

        launcher._retry_not_before = 0.0
        with (
            patch.object(launcher, "_port_is_bindable", return_value=True),
            patch.object(launcher, "_launch_in_thread") as mock_launch,
        ):
            launcher.ensure_running()
        mock_launch.assert_called_once()

    def test_a_health_200_alone_never_suppresses_a_launch(self):
        """Issue #327, restated against the adoption path: /health is not evidence.

        The port is bindable, so it is ours to take — and no probe of any kind,
        credentialed or not, may be consulted to talk us out of the launch.
        """
        launcher = _make_launcher()

        with (
            patch.object(launcher, "_port_is_bindable", return_value=True),
            patch.object(launcher, "_port_answers_connect", return_value=False),
            patch.object(launcher, "_is_running", return_value=True),
            patch.object(launcher, "_probe_status") as mock_probe,
            patch.object(launcher, "_launch_in_thread") as mock_launch,
        ):
            launcher.ensure_running()

        mock_launch.assert_called_once()
        mock_probe.assert_not_called()

    def test_a_clean_start_pays_for_no_credentialed_probe(self):
        """The probes are a cost of the degraded path only."""
        launcher = _make_launcher()

        with (
            patch.object(launcher, "_port_is_bindable", return_value=True),
            patch.object(launcher, "_port_answers_connect", return_value=False),
            patch.object(launcher, "_operator_secret") as mock_secret,
            patch.object(launcher, "_launch_in_thread"),
        ):
            launcher.ensure_running()

        mock_secret.assert_not_called()


class TestAdoptionProbeContract:
    """The probe must read the auth gate, not a credential-free surface."""

    def test_the_probe_path_is_not_exempt_from_the_gate(self):
        """An exempt path answers 200 for any OSPREY process, so it proves nothing.

        The set pin and the predicate pin are both needed: ``is_exempt_path`` is
        what the gate actually asks, and it exempts more than the set — every
        ``STATIC_MOUNT_PREFIXES`` mount too — so a probe path that stayed out of
        the set could still be waved through by the predicate.
        """
        from osprey.interfaces.common_middleware import EXEMPT_PATHS, is_exempt_path

        assert server_launcher._ADOPTION_PROBE_PATH not in EXEMPT_PATHS
        assert is_exempt_path(server_launcher._ADOPTION_PROBE_PATH) is False

    def test_the_probe_carries_the_secret_in_the_header_the_gate_reads(self):
        """Mirrored rather than imported (infrastructure/ does not import interfaces/),
        so the two spellings need a pin."""
        from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER
        from osprey.interfaces.web_auth import OPERATOR_SECRET_ENV

        assert server_launcher._OPERATOR_SECRET_HEADER == OPERATOR_SECRET_HEADER
        assert server_launcher._OPERATOR_SECRET_ENV == OPERATOR_SECRET_ENV

    def test_the_environment_carrier_is_read_when_it_is_still_present(self, monkeypatch):
        monkeypatch.setenv("OSPREY_TERMINAL_SECRET", "  carried-secret  ")
        assert _make_launcher()._operator_secret() == "carried-secret"

    def test_an_unanswered_probe_is_no_status_rather_than_a_status(self, monkeypatch):
        """A transport-level failure is None, not a status. No socket is opened:
        the point under test is the mapping, and a real connect to a "free" port
        is a race against whatever the OS hands out next."""
        opener = MagicMock()
        opener.open.side_effect = urllib.error.URLError("connection refused")
        monkeypatch.setattr(server_launcher, "_probe_opener", lambda *extra: opener)

        assert _make_launcher()._probe_status("127.0.0.1", 9) is None

    def test_a_redirect_is_the_listeners_own_status_and_is_not_followed(self, monkeypatch):
        """A 3xx must be reported as itself.

        The default opener would follow it and report the *destination's*
        status as though this port had answered it — and would re-send the
        operator-secret header to a host the listener under suspicion chose.
        Both are settled by refusing to redirect, so the 302 surfaces as 302.
        """
        real_opener = server_launcher._probe_opener
        monkeypatch.setattr(
            server_launcher,
            "_probe_opener",
            lambda *extra: real_opener(_Fake302Handler(), *extra),
        )

        assert _make_launcher()._probe_status("127.0.0.1", 9, "operator-secret") == 302


class TestRepeatRefusalsAreCheapAndQuiet:
    """The default topology re-enters the refusal path on every artifact save.

    Once the cooldown expires, ``artifact_store`` calls ``ensure_running`` again
    on the next save — for as long as the conflicting listener holds the port.
    The first refusal is news and must cost what it costs; every repeat of the
    same verdict must cost neither the grace window (2.5s of sleeps under the
    launcher's lock) nor another warning in the operator's log.
    """

    @staticmethod
    def _refuse(launcher, probe, sleep_mock, secret="operator-secret"):
        """Run one full held-port pass with *probe* answering; return the loggers."""
        with (
            patch.object(launcher, "_port_is_bindable", return_value=False),
            patch.object(launcher, "_operator_secret", return_value=secret),
            patch.object(launcher, "_probe_status", side_effect=probe),
            patch.object(launcher, "_launch_in_thread"),
            patch("osprey.infrastructure.server_launcher.time.sleep", sleep_mock),
            patch("osprey.infrastructure.server_launcher.logger.warning") as mock_warn,
            patch("osprey.infrastructure.server_launcher.logger.info") as mock_info,
        ):
            launcher.ensure_running()
        return mock_warn, mock_info

    def test_the_second_refusal_pays_no_grace_window(self):
        """A refused port is held by an established listener, not a departing one."""
        launcher = _make_launcher()
        sleep = MagicMock()

        self._refuse(launcher, _probes(200), sleep)
        assert sleep.call_count == launcher._release_grace_attempts

        sleep.reset_mock()
        launcher._retry_not_before = 0.0
        self._refuse(launcher, _probes(200), sleep)
        sleep.assert_not_called()

    def test_an_unchanged_repeat_verdict_drops_to_info(self):
        launcher = _make_launcher()
        sleep = MagicMock()

        first_warn, _ = self._refuse(launcher, _probes(200), sleep)
        first_warn.assert_called_once()

        launcher._retry_not_before = 0.0
        repeat_warn, repeat_info = self._refuse(launcher, _probes(200), sleep)
        repeat_warn.assert_not_called()
        repeat_info.assert_called_once()
        assert "cannot attribute" in _rendered(repeat_info)

    def test_a_changed_verdict_warns_again(self):
        """Demotion is per verdict, not per port: new evidence is new news."""
        launcher = _make_launcher()
        sleep = MagicMock()
        answers = {"unauthenticated": 200, "credentialed": None}

        def _probe(_host, _port, secret=None):
            return answers["credentialed"] if secret else answers["unauthenticated"]

        first_warn, _ = self._refuse(launcher, _probe, sleep)
        first_warn.assert_called_once()
        assert "HTTP 200" in _rendered(first_warn)

        answers.update(unauthenticated=401, credentialed=401)
        launcher._retry_not_before = 0.0
        second_warn, _ = self._refuse(launcher, _probe, sleep)
        second_warn.assert_called_once()
        assert "HTTP 401" in _rendered(second_warn)

    def test_a_freed_port_still_launches_after_a_refusal(self):
        """Skipping the grace window must not cost the self-heal: every call
        re-asks the bind question first, and that is the one that matters."""
        launcher = _make_launcher()
        self._refuse(launcher, _probes(200), MagicMock())

        launcher._retry_not_before = 0.0
        with (
            patch.object(launcher, "_port_is_bindable", return_value=True),
            patch.object(launcher, "_launch_in_thread") as mock_launch,
        ):
            launcher.ensure_running()
        mock_launch.assert_called_once()

    def test_an_unattributable_listener_is_not_declared_unbacked(self):
        """With no identity of its own, the launcher must not narrate the panel.

        A gating listener on this port is equally consistent with this
        deployment's own hub-owned panel — the common case when the caller is
        the MCP server the agent spawned, which never held a carrier — and with
        a stranger's server. Advising the operator to stop it, or telling them
        the panel is unbacked, would be a guess stated as fact.
        """
        # Pinned port: the line renders host:port, and an OS-assigned port whose
        # digits happen to contain "502" would fail the status-code assertion.
        launcher = _make_launcher(port=8080)
        _, mock_warn, _ = TestHeldPortAdoption._run(launcher, _probes(401, 200), secret=None)

        line = _rendered(mock_warn)
        assert "no operator secret" in line
        assert "502" not in line
        assert "unbacked" not in line
        assert "stop the other" not in line


class TestOperatorSecretIsSideEffectFree:
    """Asking "do I hold an identity?" must never be what creates one.

    ``get_web_credentials`` populates: in a process that never held a carrier it
    MINTS an operator secret and panel token nothing else in the deployment
    recognises, and pops ``OSPREY_PANEL_TOKEN`` out of ``os.environ`` on the
    way. The most frequent caller of ``ensure_*_server`` is exactly such a
    process — the MCP server spawned under the agent, on every artifact save —
    so the launcher reads the holder with the peeking accessor instead.
    """

    @staticmethod
    def _unpopulated_holder(monkeypatch):
        """Point the process holder at "not populated", restoring it afterwards."""
        from osprey.interfaces import web_auth

        for name in (
            "OSPREY_TERMINAL_SECRET",
            "OSPREY_TERMINAL_BIND_HOST",
            "OSPREY_TERMINAL_SESSION_LIFETIME",
            "OSPREY_TERMINAL_SESSION_STORE_DIR",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("OSPREY_PANEL_TOKEN", "carried-panel-token")
        # ``monkeypatch.setattr`` restores the real holder at teardown, so a
        # test here cannot decide the credentials for the tests that follow.
        monkeypatch.setattr(web_auth, "_CREDENTIALS", None)
        return web_auth

    def test_no_carrier_and_no_holder_is_no_secret(self, monkeypatch):
        web_auth = self._unpopulated_holder(monkeypatch)

        assert _make_launcher()._operator_secret() is None
        assert web_auth._CREDENTIALS is None

    def test_reading_the_secret_leaves_the_panel_token_carrier_alone(self, monkeypatch):
        """``_populate`` pops it unconditionally; peeking must not.

        The MCP process is handed a re-introduced panel token deliberately, and
        popping it here would race the panel-auth latch and strip the carrier
        from every child spawned after the probe.
        """
        self._unpopulated_holder(monkeypatch)

        _make_launcher()._operator_secret()

        assert os.environ["OSPREY_PANEL_TOKEN"] == "carried-panel-token"

    def test_peek_web_credentials_never_populates(self, monkeypatch):
        web_auth = self._unpopulated_holder(monkeypatch)

        assert web_auth.peek_web_credentials() is None
        assert web_auth._CREDENTIALS is None
        assert os.environ["OSPREY_PANEL_TOKEN"] == "carried-panel-token"

    def test_peek_web_credentials_returns_a_populated_holder_unchanged(self, monkeypatch):
        """Peeking is not "always None": it reports what is genuinely there."""
        web_auth = self._unpopulated_holder(monkeypatch)
        settled = web_auth.get_web_credentials()

        assert web_auth.peek_web_credentials() is settled
        assert _make_launcher()._operator_secret() == settled.operator_secret


class TestAnUnavailableAuthGateGetsItsOwnAdvice:
    """A 503 is an OSPREY gate that could not populate its own credentials.

    The generic refusal advice — "a foreign server, or one holding a different
    secret; the panel will be unbacked (502); stop the other server" — is wrong
    twice over for that listener. It is almost certainly not foreign, and the
    thing to fix is its configuration, not its existence. Naming the wrong
    remedy costs the operator the debugging session, so the 503 verdict says
    what it actually knows and nothing more.
    """

    def test_a_503_names_the_configuration_fault_not_a_foreign_server(self):
        # Pinned port: the line renders host:port, and an OS-assigned port whose
        # digits happen to contain "502" would fail the status-code assertion.
        launcher = _make_launcher(port=8080)

        _, mock_warn, _ = TestHeldPortAdoption._run(launcher, _probes(503))

        line = _rendered(mock_warn)
        assert "HTTP 503" in line
        assert "OSPREY_TERMINAL_SECRET" in line
        assert "502" not in line
        assert "unbacked" not in line
        assert "stop the other" not in line
        assert "foreign server" not in line

    def test_the_recorded_503_outcome_reads_flat_rather_than_nested(self):
        """The outcome phrase is the demotion key AND a fragment of prose.

        It is compared against the next refusal to decide warn-vs-info, and it
        is written to be read: spelling it "not attempted (the gate is
        unavailable (503))" nested one parenthesis inside another.
        """
        launcher = _make_launcher()

        TestHeldPortAdoption._run(launcher, _probes(503))

        status, outcome = launcher._last_refusal
        assert status == 503
        assert "unavailable, HTTP 503" in outcome
        assert "(503)" not in outcome

    def test_a_503_is_refused_without_the_secret_ever_being_offered(self):
        """A gate answering for nobody is not handed this process's identity."""
        launcher = _make_launcher()
        offered: list[str | None] = []

        def _record(_host, _port, secret=None):
            offered.append(secret)
            return 503

        mock_launch, _, _ = TestHeldPortAdoption._run(launcher, _record)

        assert offered == [None]
        mock_launch.assert_not_called()
        assert launcher._launched is False


class TestRefusalMemoryIsScopedToOneHeldPortEpisode:
    """A committed launch ends the episode that ``_refused_once`` describes.

    Both suppressions the refusal memory drives are correct only WITHIN one
    continuous held-port episode: the grace window is skipped because the holder
    is established, and the repeat warning is demoted because the verdict is
    unchanged. Once this process has bound the port itself, the holder is gone —
    so a later conflict is a new episode, with a fresh predecessor that deserves
    its window and a verdict that is news again even if it reads the same.
    """

    def test_committing_a_launch_clears_the_refusal_memory(self):
        launcher = _make_launcher()
        launcher._refused_once = True
        launcher._last_refusal = (200, "not attempted")

        with (
            patch("osprey.infrastructure.server_launcher.threading.Thread"),
            patch("osprey.infrastructure.server_launcher.time.sleep"),
            patch.object(launcher, "_is_running", return_value=True),
        ):
            launcher._launch_in_thread("127.0.0.1", 4321)

        assert launcher._refused_once is False
        assert launcher._last_refusal is None

    def test_a_launch_between_two_refusals_restores_the_window_and_the_warning(self):
        launcher = _make_launcher()
        sleep = MagicMock()

        first_warn, _ = TestRepeatRefusalsAreCheapAndQuiet._refuse(launcher, _probes(200), sleep)
        first_warn.assert_called_once()
        assert sleep.call_count == launcher._release_grace_attempts

        # The port frees and we take it — then the thread dies (a crashing app
        # factory, say), so the launcher is eligible to try again.
        launcher._retry_not_before = 0.0
        dead_thread = MagicMock()
        dead_thread.is_alive.return_value = False
        with (
            patch.object(launcher, "_port_is_bindable", return_value=True),
            patch.object(launcher, "_port_answers_connect", return_value=False),
            patch(
                "osprey.infrastructure.server_launcher.threading.Thread",
                return_value=dead_thread,
            ),
            patch("osprey.infrastructure.server_launcher.time.sleep"),
        ):
            launcher.ensure_running()
        assert launcher._launched is False

        # A new listener takes the port and returns the SAME verdict as the
        # first one. It is still news, and it still gets a predecessor's window.
        sleep.reset_mock()
        second_warn, second_info = TestRepeatRefusalsAreCheapAndQuiet._refuse(
            launcher, _probes(200), sleep
        )
        assert sleep.call_count == launcher._release_grace_attempts
        second_warn.assert_called_once()
        second_info.assert_not_called()


# ---------------------------------------------------------------------------
# _port_is_bindable — the ownership verdict, pinned syscall-shape then errno
# ---------------------------------------------------------------------------


class TestPortIsBindableAsksTheKernelToBind:
    """What the probe ASKS is pinned without asking a real kernel.

    The verdict must read the same on macOS and Linux, and the two genuinely
    disagree about which real binds collide — BSD refuses a wildcard bind
    against a specific listener, Linux allows it when both sockets set
    ``SO_REUSEADDR`` — so the syscall shape and the errno mapping are pinned
    against a stand-in socket. Real sockets appear once per family below, as a
    smoke test that the stand-in describes the real thing.
    """

    @staticmethod
    def _probe(host: str, port: int = 4321, bind_error: OSError | None = None):
        """Run ``_port_is_bindable`` against a scripted socket; return (verdict, log).

        The launcher is built OUTSIDE the patch: the stand-in answers only the
        probe's use of the socket API, and nothing else may be routed into it.
        """
        launcher = _make_launcher(host, port)
        factory, log = _fake_socket(bind_error)
        with patch("osprey.infrastructure.server_launcher.socket.socket", factory):
            verdict = launcher._port_is_bindable(host, port)
        return verdict, log

    @pytest.mark.parametrize(
        ("host", "family"),
        [
            ("127.0.0.1", socket.AF_INET),
            ("0.0.0.0", socket.AF_INET),
            ("", socket.AF_INET),
            ("localhost", socket.AF_INET),
            ("::1", socket.AF_INET6),
            ("::", socket.AF_INET6),
            ("fe80::1%en0", socket.AF_INET6),
        ],
    )
    def test_the_address_family_follows_the_host_spelling(self, host, family):
        """``AF_INET6`` iff the host string contains ``':'`` — no name lookup.

        Deliberate: the probe must ask the same question uvicorn will, and
        uvicorn is handed this same string. A hostname therefore gets an
        IPv4-only probe even where it resolves to both families; the launcher
        documents that, and the conflict surfaces at launch instead.
        """
        _verdict, log = self._probe(host)

        assert log["family"] == family
        assert log["type"] == socket.SOCK_STREAM

    @pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0", "", "::", "::1", "10.0.0.5"])
    def test_the_exact_configured_address_is_bound_with_no_loopback_substitution(self, host):
        """A wildcard probe binds the wildcard, never a substituted loopback.

        Probing ``127.0.0.1`` on behalf of a ``0.0.0.0`` bind would miss every
        listener holding another local address — the server's bind would then
        fail where the probe said it would succeed, which is precisely the
        disagreement between probe and server that #327 was about.
        """
        _verdict, log = self._probe(host, port=4321)

        assert log["bound"] == (host, 4321)

    def test_so_reuseaddr_is_set_before_the_bind_is_attempted(self):
        """Mirrors uvicorn's listener, so a ``TIME_WAIT`` remnant reads as bindable
        for the probe exactly as it is for the server — and never delays startup
        in the grace loop. Set after the bind it would answer a question about a
        socket that no longer exists."""
        _verdict, log = self._probe("127.0.0.1")

        assert (socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) in log["sockopts_at_bind"]

    @pytest.mark.parametrize("bind_error", [None, OSError(errno.EADDRINUSE, "in use")])
    def test_the_probe_socket_is_always_closed(self, bind_error):
        """The probe runs on every ``ensure_running`` call, including the per-save
        ones — a leaked descriptor per call would outlive the answer."""
        _verdict, log = self._probe("127.0.0.1", bind_error=bind_error)

        assert log["closed"] is True

    @pytest.mark.parametrize(
        ("bind_error", "bindable"),
        [
            (None, True),
            (OSError(errno.EADDRINUSE, "Address already in use"), False),
            (OSError(errno.EACCES, "Permission denied"), False),
            (OSError(errno.EAFNOSUPPORT, "Address family not supported"), True),
            (OSError(errno.EADDRNOTAVAIL, "Can't assign requested address"), True),
            (OSError(errno.EINVAL, "Invalid argument"), True),
            (OSError(None, "an OSError carrying no errno at all"), True),
        ],
        ids=["bound", "eaddrinuse", "eacces", "eafnosupport", "eaddrnotavail", "einval", "noerrno"],
    )
    @pytest.mark.parametrize("host", ["127.0.0.1", "::1"], ids=["ipv4", "ipv6"])
    def test_only_eaddrinuse_and_eacces_mean_the_port_is_taken(self, bind_error, bindable, host):
        """Everything else says the PROBE could not answer, not that the port is held.

        The asymmetry is deliberate and load-bearing: a probe defect that read
        as "taken" would silently suppress the launch and leave the panel
        unbacked with no listener at all — the #327 failure by another route.
        Read as "free", the launch goes ahead and any real conflict surfaces as
        uvicorn's own bind failure, which is a diagnosable event.
        """
        verdict, _log = self._probe(host, bind_error=bind_error)

        assert verdict is bindable

    @pytest.mark.parametrize(
        ("bind_error", "warns"),
        [
            (None, False),
            (OSError(errno.EADDRINUSE, "Address already in use"), False),
            (OSError(errno.EACCES, "Permission denied"), False),
            (OSError(errno.EAFNOSUPPORT, "Address family not supported"), True),
            (OSError(errno.EINVAL, "Invalid argument"), True),
        ],
        ids=["bound", "eaddrinuse", "eacces", "eafnosupport", "einval"],
    )
    def test_an_inconclusive_probe_says_so_once_and_a_verdict_says_nothing(self, bind_error, warns):
        """Treating a port as free on an unanswerable probe is auditable, not silent."""
        launcher = _make_launcher("127.0.0.1", 4321)
        factory, _log = _fake_socket(bind_error)
        with (
            patch("osprey.infrastructure.server_launcher.socket.socket", factory),
            patch("osprey.infrastructure.server_launcher.logger.warning") as mock_warn,
        ):
            launcher._port_is_bindable("127.0.0.1", 4321)

        assert mock_warn.called is warns
        if warns:
            line = _rendered(mock_warn)
            assert "127.0.0.1:4321" in line
            assert "treating the port as free" in line


class TestPortIsBindableAgainstARealKernel:
    """One smoke row per family: the stand-in above must describe the real thing.

    Only the two unambiguous cases are asked of a real kernel — an exact-address
    listener holds its own address, and a released port is free — because those
    are the rows macOS and Linux answer identically. The platform-divergent ones
    (a wildcard probe against a specific listener) stay mocked.
    """

    def test_an_ipv4_listener_holds_its_own_address(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            assert _make_launcher()._port_is_bindable("127.0.0.1", port) is False

    def test_a_released_ipv4_port_is_bindable(self):
        assert _make_launcher()._port_is_bindable("127.0.0.1", _free_port()) is True

    def test_an_ipv6_listener_holds_its_own_address(self):
        _require_ipv6_loopback()
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as srv:
            srv.bind(("::1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            assert _make_launcher()._port_is_bindable("::1", port) is False

    def test_a_released_ipv6_port_is_bindable(self):
        _require_ipv6_loopback()
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
            sock.bind(("::1", 0))
            port = sock.getsockname()[1]
        assert _make_launcher()._port_is_bindable("::1", port) is True


class TestPortAnswersConnectIsDiagnosticOnly:
    """The connect probe answers a different question, and substitutes for it."""

    @pytest.mark.parametrize(
        ("host", "destination"),
        [
            ("0.0.0.0", "127.0.0.1"),
            ("", "127.0.0.1"),
            ("::", "::1"),
            ("10.0.0.5", "10.0.0.5"),
        ],
    )
    def test_a_wildcard_is_substituted_here_and_only_here(self, host, destination):
        """The contrast with the bind probe is the point, not an inconsistency.

        A wildcard is not a valid client destination on macOS/BSD, and a server
        on the wildcard accepts the loopback anyway — so the *reachability*
        question is asked at the loopback. The *ownership* question is not:
        see ``test_the_exact_configured_address_is_bound_with_no_loopback_substitution``.
        """
        launcher = _make_launcher(host, 4321)
        seen: dict = {}

        def _connect(address, timeout=None):
            seen["address"] = address
            seen["timeout"] = timeout
            raise OSError(errno.ECONNREFUSED, "connection refused")

        with patch("osprey.infrastructure.server_launcher.socket.create_connection", _connect):
            assert launcher._port_answers_connect(host, 4321) is False

        assert seen["address"] == (destination, 4321)
        assert seen["timeout"] == 1

    def test_an_accepted_connect_is_reported_as_answered(self):
        launcher = _make_launcher("127.0.0.1", 4321)
        with patch(
            "osprey.infrastructure.server_launcher.socket.create_connection",
            return_value=MagicMock(),
        ):
            assert launcher._port_answers_connect("127.0.0.1", 4321) is True


class TestBindableAndConnectableTogetherDecideTheLaunch:
    """The bind decides; the connect only narrates. Four rows, both families."""

    @staticmethod
    def _run(host: str, bindable: bool, answers_connect: bool):
        launcher = _make_launcher(host, 4321)
        with (
            patch.object(launcher, "_port_is_bindable", return_value=bindable),
            patch.object(
                launcher, "_port_answers_connect", return_value=answers_connect
            ) as mock_connect,
            patch.object(launcher, "_operator_secret", return_value=None),
            patch.object(launcher, "_probe_status", side_effect=_probes(None)),
            patch.object(launcher, "_launch_in_thread") as mock_launch,
            patch("osprey.infrastructure.server_launcher.time.sleep"),
            patch("osprey.infrastructure.server_launcher.logger.warning"),
            patch("osprey.infrastructure.server_launcher.logger.info") as mock_info,
        ):
            launcher.ensure_running()
        return launcher, mock_launch, mock_connect, mock_info

    @pytest.mark.parametrize("host", ["127.0.0.1", "::1"], ids=["ipv4", "ipv6"])
    @pytest.mark.parametrize("answers_connect", [True, False], ids=["connectable", "silent"])
    def test_a_bindable_port_launches_whatever_answers_a_connect(self, host, answers_connect):
        """A listener that does not contend for the bind does not own the port.

        A Docker Desktop host-loopback pass-through answers a connect on a port
        the container can still bind. Standing down for it would leave the panel
        with no server of its own.
        """
        _launcher, mock_launch, _mock_connect, mock_info = self._run(
            host, bindable=True, answers_connect=answers_connect
        )

        mock_launch.assert_called_once()
        narrated = any("does not block our bind" in line for line in _lines(mock_info))
        assert narrated is answers_connect

    @pytest.mark.parametrize("host", ["127.0.0.1", "::1"], ids=["ipv4", "ipv6"])
    def test_an_unbindable_port_goes_to_the_adoption_verdict_without_a_connect(self, host):
        """Reachability has no vote once the bind has failed, so it is not asked."""
        launcher, mock_launch, mock_connect, _mock_info = self._run(
            host, bindable=False, answers_connect=True
        )

        mock_launch.assert_not_called()
        assert launcher._launched is False
        mock_connect.assert_not_called()


# ---------------------------------------------------------------------------
# getsource drift guard — ownership is decided by a bind, not by reachability
# ---------------------------------------------------------------------------


def test_the_ownership_verdict_is_a_bind_and_the_connect_is_only_a_diagnostic():
    """Source-level backstop for issue #327, behind the behavioural pins above.

    The behaviour tests fix what the probe answers; this fixes how it may find
    out. A future refactor that reintroduced a ``/health`` request or a connect
    into the verdict could still satisfy a mocked matrix while restoring exactly
    the false positive #327 was about — a stale or foreign responder talking the
    launcher out of owning a port it could have bound.
    """
    verdict = inspect.getsource(ServerLauncher._port_is_bindable)
    # ``.bind(`` rather than ``bind``: the docstring says "bind" a dozen times,
    # so the bare substring would pass on a body that no longer binds anything.
    assert ".bind(" in verdict
    assert "urlopen" not in verdict
    assert "create_connection" not in verdict

    diagnostic = inspect.getsource(ServerLauncher._port_answers_connect)
    assert "create_connection" in diagnostic
    assert "urlopen" not in diagnostic
