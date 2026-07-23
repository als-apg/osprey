"""Unit tests for the generic ``ServerLauncher`` plumbing.

The ownership state machine (port-free / held-then-freed / held-throughout) is
covered in ``tests/interfaces/channel_finder/test_server_launcher.py``. This
module targets the parts not exercised there: the data-driven callback builders
(``_make_config_reader`` env override, ``_resolve_dotted``, ``_make_app_factory``
import/kwargs/error branches), the ``ensure_web_server`` dispatch table, and the
two ``ensure_running`` short-circuits (auto-launch disabled, already launched).
"""

from __future__ import annotations

import inspect
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from osprey.infrastructure import server_launcher
from osprey.infrastructure.server_launcher import (
    ServerLauncher,
    _make_app_factory,
    _make_config_reader,
    _resolve_dotted,
    ensure_web_server,
)
from osprey.registry.web import WebServerDefinition


def _free_port() -> int:
    """Reserve then release an OS-assigned port so nothing is listening on it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _defn(**overrides) -> WebServerDefinition:
    """Build a WebServerDefinition with test-friendly defaults."""
    base = {
        "name": "Test Server",
        "factory_path": "types:SimpleNamespace",
        "config_key": "test_server",
    }
    base.update(overrides)
    return WebServerDefinition(**base)


# ---------------------------------------------------------------------------
# _make_config_reader — env override on top of config
# ---------------------------------------------------------------------------


class TestConfigReaderEnvOverride:
    def test_env_var_overrides_configured_port(self, monkeypatch):
        defn = _defn(port_default=8080)
        reader = _make_config_reader(defn)
        monkeypatch.setattr(
            server_launcher,
            "load_osprey_config",
            lambda: {"test_server": {"host": "10.0.0.1", "port": 9000}},
        )
        monkeypatch.setenv(defn.port_env_var, "9191")

        host, port = reader()
        assert host == "10.0.0.1"
        assert port == 9191  # env wins over the configured 9000

    def test_empty_env_var_is_ignored(self, monkeypatch):
        defn = _defn(port_default=8080)
        reader = _make_config_reader(defn)
        monkeypatch.setattr(server_launcher, "load_osprey_config", lambda: {})
        monkeypatch.setenv(defn.port_env_var, "")  # exported-but-empty

        _host, port = reader()
        assert port == 8080  # falls back to the default, not int("")

    def test_env_var_name_derives_from_config_key(self):
        assert _defn(config_key="artifact_server").port_env_var == "OSPREY_ARTIFACT_SERVER_PORT"


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
# getsource drift guard — ownership is decided by a TCP connect-probe
# ---------------------------------------------------------------------------


def test_ownership_probe_uses_tcp_connect_not_health():
    """Regression guard for issue #327: ``_port_has_listener`` must answer the
    ownership question by opening a TCP connection, never by trusting a
    ``/health`` 200 (which a stale/foreign responder can fake)."""
    src = inspect.getsource(ServerLauncher._port_has_listener)
    assert "create_connection" in src
    # Must not fall back to the urllib ``/health`` request used by ``_is_running``.
    assert "urlopen" not in src
