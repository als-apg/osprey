"""Tests for the ``osprey theme-lab`` CLI command.

Hermetic: no browser is ever opened and no server ever serves — the uvicorn
server and the browser-opening helper are patched out. The command still binds
its listening socket, so an explicit port comes from ``reserve_port``.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

import pytest

from osprey.cli.main import LazyGroup, cli
from osprey.cli.theme_lab_cmd import THEME_LAB_PATH, create_app, theme_lab
from osprey_connectors_tests._harness._ports import bound_socket

BROWSER_HELPER = "osprey.interfaces.web_terminal.app._open_browser_when_ready"


# -- registration ----------------------------------------------------------


class TestRegistration:
    """The command must be listed AND resolvable through LazyGroup."""

    def test_root_help_lists_theme_lab(self, runner):
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "theme-lab" in result.output

    def test_lazy_group_resolves_hyphenated_name(self):
        """``theme-lab`` is not a Python identifier, so the default
        ``getattr(mod, cmd_name)`` path in LazyGroup.get_command cannot find it.

        Regression guard: without the explicit ``theme-lab`` branch this raises
        AttributeError the moment anyone actually runs the command, which a
        help-listing test alone does not catch.
        """
        group = LazyGroup(name="osprey")

        command = group.get_command(mock.Mock(), "theme-lab")

        assert command is not None
        assert command.name == "theme-lab"

    def test_command_help_through_group(self, runner):
        result = runner.invoke(cli, ["theme-lab", "--help"])

        assert result.exit_code == 0
        assert "--no-browser" in result.output


# -- serving ---------------------------------------------------------------


class TestServing:
    """Flag handling around the server the command starts."""

    @pytest.fixture
    def served(self, monkeypatch) -> dict:
        """What the stubbed uvicorn server was handed: its port and its sockets."""
        import uvicorn

        seen: dict = {}

        class _Server:
            def __init__(self, config):
                seen["config"] = config

            def run(self, sockets=None):
                seen["sockets"] = sockets
                seen["listening_port"] = sockets[0].getsockname()[1]

        monkeypatch.setattr(uvicorn, "Server", _Server)
        return seen

    def test_no_browser_serves_without_opening_one(self, runner, served):
        with mock.patch(BROWSER_HELPER) as mock_browser:
            result = runner.invoke(theme_lab, ["--no-browser"])

        assert result.exit_code == 0
        assert not mock_browser.called
        port = served["listening_port"]
        assert port != 0
        assert served["config"].host == "127.0.0.1"
        assert served["config"].port == port
        assert f"http://127.0.0.1:{port}{THEME_LAB_PATH}" in result.output

    def test_published_port_is_the_bound_socket_s(self, runner, served):
        from osprey.interfaces.common_middleware import WEB_PORT_ENV

        with mock.patch(BROWSER_HELPER):
            result = runner.invoke(theme_lab, ["--no-browser"])

        assert result.exit_code == 0
        assert os.environ.get(WEB_PORT_ENV) == str(served["listening_port"])

    def test_port_flag_is_honoured(self, runner, served, reserve_port):
        port = reserve_port()
        with mock.patch(BROWSER_HELPER):
            result = runner.invoke(theme_lab, ["--port", str(port), "--no-browser"])

        assert result.exit_code == 0
        assert served["listening_port"] == port
        assert f"http://127.0.0.1:{port}{THEME_LAB_PATH}" in result.output

    def test_a_held_port_is_refused(self, runner, served):
        with bound_socket("127.0.0.1") as held:
            held.listen(1)
            port = held.getsockname()[1]
            result = runner.invoke(theme_lab, ["--port", str(port), "--no-browser"])

        assert result.exit_code == 1
        assert f"Cannot serve on port {port}" in result.output
        assert "sockets" not in served

    @pytest.mark.usefixtures("served")
    def test_browser_opened_at_the_root_token_url_not_the_theme_lab_page(
        self, runner, reserve_port
    ):
        """The login URL must land on ``/``, the one path the gate inspects.

        ``THEME_LAB_PATH`` sits under the ``/design-system`` mount, which
        ``is_exempt_path`` clears before a credential is looked at — so a
        ``?token=`` appended there is never read and never exchanged, and the
        page loads with no session behind it. ``/`` is gated, so the gate
        performs the exchange and then the app's own root route forwards the
        now-authenticated browser to the Theme Lab.
        """
        port = reserve_port()
        with mock.patch(BROWSER_HELPER) as mock_browser:
            result = runner.invoke(theme_lab, ["--port", str(port)])

        assert result.exit_code == 0
        mock_browser.assert_called_once()
        opened = mock_browser.call_args.args[0]
        # The token is a freshly minted secret, so assert the shape, not a value.
        assert opened.startswith(f"http://127.0.0.1:{port}/?token=")
        assert THEME_LAB_PATH not in opened

    @pytest.mark.usefixtures("served")
    def test_announced_login_url_lands_on_a_gated_path(self, runner):
        """The printed URL is exchangeable — its path is not an exempt one."""
        from osprey.interfaces.common_middleware import TOKEN_EXCHANGE_PATHS, is_exempt_path

        with mock.patch(BROWSER_HELPER) as mock_browser:
            runner.invoke(theme_lab, [])

        opened = mock_browser.call_args.args[0]
        path = urlsplit(opened).path
        assert path in TOKEN_EXCHANGE_PATHS
        assert not is_exempt_path(path)

    @pytest.mark.usefixtures("served")
    def test_settled_port_is_published_for_the_cookie_name(self, runner, reserve_port):
        """``OSPREY_WEB_PORT`` is set before the app is built.

        The session cookie's name carries the port, because cookies ignore
        ports and two OSPREY servers on one host would otherwise overwrite each
        other's session. Without this publication the name falls back to the
        bare base and they collide.
        """
        from osprey.interfaces.common_middleware import WEB_PORT_ENV, session_cookie_name

        port = reserve_port()
        with mock.patch(BROWSER_HELPER):
            runner.invoke(theme_lab, ["--port", str(port), "--no-browser"])

        assert os.environ.get(WEB_PORT_ENV) == str(port)
        assert session_cookie_name() == session_cookie_name(port)

    def test_browser_failure_still_serves_and_prints_url(self, runner, served):
        """A headless host must get a usable URL, not a dead command."""
        with mock.patch(BROWSER_HELPER, side_effect=RuntimeError("no display")):
            result = runner.invoke(theme_lab, [])

        assert result.exit_code == 0
        port = served["listening_port"]
        assert f"http://127.0.0.1:{port}{THEME_LAB_PATH}" in result.output


# -- app factory -----------------------------------------------------------


class TestCreateApp:
    """create_app() is shared with the browser suite, so it must stand alone."""

    def test_mounts_design_system(self):
        app = create_app()

        mounts = {route.path: route for route in app.routes if hasattr(route, "app")}
        assert "/design-system" in mounts

    def test_design_system_mount_serves_packaged_static_dir(self):
        from osprey.interfaces import design_system

        app = create_app()

        mount = next(r for r in app.routes if getattr(r, "path", None) == "/design-system")
        served = Path(mount.app.directory).resolve()
        assert served == (Path(design_system.__file__).parent / "static").resolve()

    def test_root_redirects_to_theme_lab_page(self):
        app = create_app()

        paths = [getattr(route, "path", None) for route in app.routes]
        assert "/" in paths
