"""Tests for the interactive-menu display helpers.

These are pure Rich/console output functions. Tests patch the module ``console``
and ``builtins.input`` (the terminal boundary) and assert on *content and
structure* of what would be shown — not exact ANSI/box-drawing art, which is
brittle. The goal is: the right context subtitles appear, custom banners are
honored, version is surfaced, and the help screens block on ENTER.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from osprey.cli import menu_display


def _printed_text(mock_console) -> str:
    """Join the stringified first positional arg of every console.print call."""
    parts = []
    for call in mock_console.print.call_args_list:
        if call.args:
            parts.append(str(call.args[0]))
    return "\n".join(parts)


class TestShowBanner:
    def test_default_interactive_banner_shows_menu_subtitle(self):
        mock_console = MagicMock()
        with patch.object(menu_display, "console", mock_console):
            # No project cwd config → default banner path.
            with patch("osprey.utils.config.get_config_value", return_value=None):
                menu_display.show_banner(context="interactive")

        text = _printed_text(mock_console)
        assert "Interactive Menu System" in text

    def test_chat_context_shows_slash_command_hint(self):
        mock_console = MagicMock()
        with patch.object(menu_display, "console", mock_console):
            with patch("osprey.utils.config.get_config_value", return_value=None):
                menu_display.show_banner(context="chat")

        text = _printed_text(mock_console)
        assert "/help" in text

    def test_version_is_surfaced_when_available(self):
        mock_console = MagicMock()
        with patch.object(menu_display, "console", mock_console):
            with patch("osprey.utils.config.get_config_value", return_value=None):
                with patch("osprey.__version__", "9.9.9", create=True):
                    menu_display.show_banner(context="interactive")

        assert "v9.9.9" in _printed_text(mock_console)

    def test_custom_banner_from_config_path_is_used(self):
        mock_console = MagicMock()
        with patch.object(menu_display, "console", mock_console):
            with patch(
                "osprey.utils.config.get_config_value", return_value="MY CUSTOM BANNER"
            ) as mock_get:
                menu_display.show_banner(context="interactive", config_path="/some/config.yml")

        # The config-path branch queries cli.banner with the explicit path.
        assert mock_get.called
        assert "MY CUSTOM BANNER" in _printed_text(mock_console)

    def test_config_load_failure_falls_back_to_default_banner(self):
        mock_console = MagicMock()
        with patch.object(menu_display, "console", mock_console):
            with patch("osprey.utils.config.get_config_value", side_effect=RuntimeError("boom")):
                # Must not raise — the CLI banner should always render.
                menu_display.show_banner(context="interactive", config_path="/x/config.yml")

        # Default banner still carries the framework tagline.
        assert "Osprey Framework" in _printed_text(mock_console)


class TestShowSuccessArt:
    def test_prints_success_marker(self):
        mock_console = MagicMock()
        with patch.object(menu_display, "console", mock_console):
            menu_display.show_success_art()

        assert "SUCCESS" in _printed_text(mock_console)


class TestHelpScreens:
    def test_deploy_help_lists_all_actions_and_blocks_on_enter(self):
        mock_console = MagicMock()
        with patch.object(menu_display, "console", mock_console):
            with patch("osprey.utils.config.get_config_value", return_value=None):
                with patch("builtins.input") as mock_input:
                    menu_display.show_deploy_help()

        text = _printed_text(mock_console)
        # Every deploy action gets an entry in the help screen.
        for action in ["up", "down", "status", "restart", "build", "rebuild", "clean"]:
            assert action in text
        mock_console.clear.assert_called_once()
        mock_input.assert_called_once()

    def test_root_help_covers_getting_started(self):
        mock_console = MagicMock()
        with patch.object(menu_display, "console", mock_console):
            with patch("osprey.utils.config.get_config_value", return_value=None):
                with patch("builtins.input") as mock_input:
                    menu_display.handle_help_action_root()

        text = _printed_text(mock_console)
        assert "Select a project" in text
        assert "Create new project" in text
        assert "Typical Workflow" in text
        mock_input.assert_called_once()

    def test_project_help_covers_project_commands(self):
        mock_console = MagicMock()
        with patch.object(menu_display, "console", mock_console):
            with patch("osprey.utils.config.get_config_value", return_value=None):
                with patch("builtins.input") as mock_input:
                    menu_display.handle_help_action()

        text = _printed_text(mock_console)
        for cmd in ["deploy", "health", "generate", "config", "registry", "init"]:
            assert cmd in text
        mock_input.assert_called_once()
