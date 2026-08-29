"""Every panel entry point lands on its layout slot, at the base the config names.

The acceptance criterion for the panel entry points: launched with no ``--port``
and no configured port, ``osprey web`` binds the ``web`` slot, ``osprey ariel
web`` the ``ariel`` slot, and so on — and a project that sets
``deployment.port_base`` moves every one of them by exactly that much. The old
failure this pins is a default frozen at one deployment's numbers: on a host
running two, the second deployment's panels would advertise the first one's
ports.

Nothing here starts a server. Each entry point's *derivation* is driven
directly — the signature default a programmatic caller gets, and the resolver
the CLI consults — because that derivation is the whole of what this feature
changed; binding a socket would only prove uvicorn works.
"""

from __future__ import annotations

import pytest

from osprey.port_layout import DEFAULT_PORT_BASE, default_port
from osprey.registry.web import resolve_web_server_address

#: A base no deployment would be on by accident, so an assertion against it
#: cannot pass by coincidence with the layout's own default.
MOVED_BASE = 20000

#: Registry key → the ``osprey`` verb that launches that panel, for test ids
#: that name the command an operator would type.
PANEL_ENTRY_POINTS = {
    "artifact": "osprey artifacts web",
    "ariel": "osprey ariel web",
    "channel_finder": "osprey channel-finder web",
}


@pytest.fixture(autouse=True)
def _no_port_env(monkeypatch):
    """Clear every panel's ``OSPREY_<KEY>_PORT`` override.

    The env var wins over both the config and the layout, so a variable left in
    the developer's environment would silently answer every assertion below.
    """
    from osprey.registry.web import FRAMEWORK_WEB_SERVERS

    for definition in FRAMEWORK_WEB_SERVERS.values():
        monkeypatch.delenv(definition.port_env_var, raising=False)
    monkeypatch.delenv("OSPREY_WEB_PORT", raising=False)


class TestSignatureDefaults:
    """The ``port=`` default each panel's programmatic entry point carries.

    A signature default is the one place :data:`DEFAULT_PORT_BASE` is the right
    base: it is read at import time, when there is no config to resolve. Every
    real caller passes the port it resolved instead, which the CLI tests below
    cover.
    """

    @staticmethod
    def _default(func) -> int:
        import inspect

        return inspect.signature(func).parameters["port"].default

    def test_web_terminal_run_web(self):
        from osprey.interfaces.web_terminal.app import run_web

        assert self._default(run_web) == default_port("web", 0, base=DEFAULT_PORT_BASE)

    def test_ariel_run_web(self):
        from osprey.interfaces.ariel import run_web

        assert self._default(run_web) == default_port("ariel", 0, base=DEFAULT_PORT_BASE)

    def test_artifacts_run_server(self):
        from osprey.interfaces.artifacts import run_server

        assert self._default(run_server) == default_port("artifact", 0, base=DEFAULT_PORT_BASE)


class TestPanelCliDefaults:
    """What ``osprey <panel> web`` binds with no ``--port`` and no configured port.

    Each of these CLI commands defaults ``--port`` to ``None`` and asks
    :func:`resolve_web_server_address` for the answer, so driving the resolver
    with the command's own registry key is driving the command's default.
    """

    @pytest.mark.parametrize("key", sorted(PANEL_ENTRY_POINTS), ids=PANEL_ENTRY_POINTS.get)
    def test_default_base_project(self, key):
        """A project that never mentions ``deployment.port_base``."""
        _, port = resolve_web_server_address(key, {})

        assert port == default_port(key, 0, base=DEFAULT_PORT_BASE)

    @pytest.mark.parametrize("key", sorted(PANEL_ENTRY_POINTS), ids=PANEL_ENTRY_POINTS.get)
    def test_moved_base_project(self, key):
        """``deployment.port_base: 20000`` moves the panel by exactly that much."""
        config = {"deployment": {"port_base": MOVED_BASE}}

        _, port = resolve_web_server_address(key, config)

        assert port == default_port(key, 0, base=MOVED_BASE)

    @pytest.mark.parametrize("key", sorted(PANEL_ENTRY_POINTS), ids=PANEL_ENTRY_POINTS.get)
    def test_configured_port_still_wins(self, key):
        """A port written in the config is honoured verbatim, base or no base."""
        from osprey.registry.web import FRAMEWORK_WEB_SERVERS

        definition = FRAMEWORK_WEB_SERVERS[key]
        section = {"port": 31000}
        if definition.config_web_subkey:
            section = {definition.config_web_subkey: section}
        config = {"deployment": {"port_base": MOVED_BASE}, definition.config_key: section}

        _, port = resolve_web_server_address(key, config)

        assert port == 31000


class TestWebTerminalCli:
    """``osprey web`` — the terminal itself, whose port is resolved in the CLI."""

    @staticmethod
    def _resolved(config: dict) -> int:
        from osprey.cli.web_cmd import resolve_web_port
        from osprey.port_layout import resolve_port_base

        return resolve_web_port(None, None, base=resolve_port_base(config), env={})

    def test_default_base_project(self):
        assert self._resolved({}) == default_port("web", 0, base=DEFAULT_PORT_BASE)

    def test_moved_base_project(self):
        config = {"deployment": {"port_base": MOVED_BASE}}

        assert self._resolved(config) == default_port("web", 0, base=MOVED_BASE)


class TestWebTerminalUrlFromMcp:
    """``mcp_server.http.web_terminal_url`` — what an MCP tool links back to.

    It reads the config itself rather than being handed one, so it is driven
    here through the loader it calls.
    """

    @staticmethod
    def _url(monkeypatch, config: dict) -> str:
        from osprey.mcp_server import http
        from osprey.utils import workspace

        monkeypatch.setattr(workspace, "load_osprey_config", lambda *a, **k: config)
        return http.web_terminal_url()

    def test_default_base_project(self, monkeypatch):
        url = self._url(monkeypatch, {})

        assert url == f"http://127.0.0.1:{default_port('web', 0, base=DEFAULT_PORT_BASE)}"

    def test_moved_base_project(self, monkeypatch):
        url = self._url(monkeypatch, {"deployment": {"port_base": MOVED_BASE}})

        assert url == f"http://127.0.0.1:{default_port('web', 0, base=MOVED_BASE)}"

    def test_configured_port_still_wins(self, monkeypatch):
        config = {
            "deployment": {"port_base": MOVED_BASE},
            "web_terminal": {"port": 31000},
        }

        assert self._url(monkeypatch, config) == "http://127.0.0.1:31000"
