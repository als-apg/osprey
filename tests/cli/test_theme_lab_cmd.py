"""Tests for the ``osprey theme-lab`` CLI command.

Hermetic: no browser is ever opened and no server is ever bound — ``uvicorn.run``
and the browser-opening helper are patched out.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

import pytest
from click.testing import CliRunner

from osprey.cli.main import LazyGroup, cli
from osprey.cli.theme_lab_cmd import THEME_LAB_PATH, create_app, theme_lab

BROWSER_HELPER = "osprey.interfaces.web_terminal.app._open_browser_when_ready"


@pytest.fixture
def runner():
    return CliRunner()


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
    """Flag handling around the uvicorn call."""

    def test_no_browser_serves_without_opening_one(self, runner):
        with (
            mock.patch("uvicorn.run") as mock_run,
            mock.patch(BROWSER_HELPER) as mock_browser,
        ):
            result = runner.invoke(theme_lab, ["--no-browser"])

        assert result.exit_code == 0
        assert not mock_browser.called
        _, kwargs = mock_run.call_args
        assert kwargs["host"] == "127.0.0.1"
        assert isinstance(kwargs["port"], int)
        assert f":{kwargs['port']}{THEME_LAB_PATH}" in result.output

    def test_port_flag_is_honoured(self, runner):
        with (
            mock.patch("uvicorn.run") as mock_run,
            mock.patch(BROWSER_HELPER),
        ):
            result = runner.invoke(theme_lab, ["--port", "9123", "--no-browser"])

        assert result.exit_code == 0
        assert mock_run.call_args.kwargs["port"] == 9123
        assert f"http://127.0.0.1:9123{THEME_LAB_PATH}" in result.output

    def test_browser_opened_at_the_root_token_url_not_the_theme_lab_page(self, runner):
        """The login URL must land on ``/``, the one path the gate inspects.

        ``THEME_LAB_PATH`` sits under the ``/design-system`` mount, which
        ``is_exempt_path`` clears before a credential is looked at — so a
        ``?token=`` appended there is never read and never exchanged, and the
        page loads with no session behind it. ``/`` is gated, so the gate
        performs the exchange and then the app's own root route forwards the
        now-authenticated browser to the Theme Lab.
        """
        with (
            mock.patch("uvicorn.run"),
            mock.patch(BROWSER_HELPER) as mock_browser,
        ):
            result = runner.invoke(theme_lab, ["--port", "9124"])

        assert result.exit_code == 0
        mock_browser.assert_called_once()
        opened = mock_browser.call_args.args[0]
        # The token is a freshly minted secret, so assert the shape, not a value.
        assert opened.startswith("http://127.0.0.1:9124/?token=")
        assert THEME_LAB_PATH not in opened

    def test_announced_login_url_lands_on_a_gated_path(self, runner):
        """The printed URL is exchangeable — its path is not an exempt one."""
        from osprey.interfaces.common_middleware import TOKEN_EXCHANGE_PATHS, is_exempt_path

        with (
            mock.patch("uvicorn.run"),
            mock.patch(BROWSER_HELPER) as mock_browser,
        ):
            runner.invoke(theme_lab, ["--port", "9126"])

        opened = mock_browser.call_args.args[0]
        path = urlsplit(opened).path
        assert path in TOKEN_EXCHANGE_PATHS
        assert not is_exempt_path(path)

    def test_settled_port_is_published_for_the_cookie_name(self, runner):
        """``OSPREY_WEB_PORT`` is set before the app is built.

        The session cookie's name carries the port, because cookies ignore
        ports and two OSPREY servers on one host would otherwise overwrite each
        other's session. Without this publication the name falls back to the
        bare base and they collide.
        """
        from osprey.interfaces.common_middleware import WEB_PORT_ENV, session_cookie_name

        with (
            mock.patch("uvicorn.run"),
            mock.patch(BROWSER_HELPER),
        ):
            runner.invoke(theme_lab, ["--port", "9127", "--no-browser"])

        assert os.environ.get(WEB_PORT_ENV) == "9127"
        assert session_cookie_name() == session_cookie_name(9127)

    def test_browser_failure_still_serves_and_prints_url(self, runner):
        """A headless host must get a usable URL, not a dead command."""
        with (
            mock.patch("uvicorn.run") as mock_run,
            mock.patch(BROWSER_HELPER, side_effect=RuntimeError("no display")),
        ):
            result = runner.invoke(theme_lab, ["--port", "9125"])

        assert result.exit_code == 0
        assert mock_run.called
        assert f"http://127.0.0.1:9125{THEME_LAB_PATH}" in result.output


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
