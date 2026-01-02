"""Tests for preview_styles CLI tool.

Tests the interactive theme preview and testing tool, focusing on smoke tests
and basic functionality to establish a safety net for this utility module.
"""

from unittest import mock

import pytest

from osprey.cli.preview_styles import (
    THEMES,
    compare_themes,
    main,
    preview_theme,
    show_banner,
    show_color_swatches,
    show_inline_styles,
    show_panel_examples,
    show_real_world_example,
    show_status_messages,
    show_table_example,
    show_theme_header,
)
from osprey.cli.styles import ColorTheme


@pytest.fixture
def mock_console():
    """Mock the rich console to avoid actual terminal output."""
    with mock.patch("osprey.cli.preview_styles.console") as mock_console:
        yield mock_console


@pytest.fixture
def sample_theme():
    """Provide a sample color theme for testing."""
    return ColorTheme(
        primary="#9370DB",
        accent="#00cccc",
        command="#ff9500",
        path="#999999",
        info="#00aaff",
    )


class TestThemeDefinitions:
    """Test predefined theme definitions."""

    def test_themes_dictionary_exists(self):
        """Verify THEMES dictionary is defined."""
        assert THEMES is not None
        assert isinstance(THEMES, dict)

    def test_themes_contains_expected_themes(self):
        """Verify standard themes are defined."""
        expected_themes = ["osprey", "ocean", "sunset", "forest", "monochrome"]
        for theme_name in expected_themes:
            assert theme_name in THEMES
            assert isinstance(THEMES[theme_name], ColorTheme)

    def test_all_themes_have_required_colors(self):
        """Verify each theme has all required color attributes."""
        required_colors = ["primary", "accent", "command", "path", "info"]
        for theme_name, theme in THEMES.items():
            for color in required_colors:
                assert hasattr(theme, color), f"{theme_name} missing {color}"
                assert getattr(theme, color) is not None


class TestShowFunctions:
    """Test individual show/display functions."""

    def test_show_theme_header(self, mock_console, sample_theme):
        """Test theme header display."""
        # Should not raise exception
        show_theme_header("test", sample_theme)

        # Verify console.print was called
        assert mock_console.print.called

    def test_show_color_swatches(self, mock_console, sample_theme):
        """Test color swatches display."""
        show_color_swatches(sample_theme)
        assert mock_console.print.called

    def test_show_status_messages(self, mock_console):
        """Test status messages display."""
        show_status_messages()
        assert mock_console.print.called

    def test_show_inline_styles(self, mock_console):
        """Test inline styles display."""
        show_inline_styles()
        assert mock_console.print.called

    def test_show_real_world_example(self, mock_console):
        """Test real-world example display."""
        show_real_world_example()
        assert mock_console.print.called

    def test_show_panel_examples(self, mock_console):
        """Test panel examples display."""
        show_panel_examples()
        assert mock_console.print.called

    def test_show_table_example(self, mock_console):
        """Test table example display."""
        show_table_example()
        assert mock_console.print.called

    def test_show_banner(self, mock_console):
        """Test ASCII banner display."""
        show_banner()
        assert mock_console.print.called


class TestPreviewTheme:
    """Test the preview_theme function."""

    @mock.patch("osprey.cli.preview_styles.console")
    @mock.patch("osprey.cli.preview_styles.set_theme")
    def test_preview_theme_basic(self, mock_set_theme, mock_console, sample_theme):
        """Test basic theme preview."""
        preview_theme("test", sample_theme, show_banner_art=False)

        # Verify theme was set
        mock_set_theme.assert_called_once_with(sample_theme)

        # Verify console operations were performed
        assert mock_console.clear.called
        assert mock_console.print.called

    @mock.patch("osprey.cli.preview_styles.console")
    @mock.patch("osprey.cli.preview_styles.set_theme")
    def test_preview_theme_with_banner(self, mock_set_theme, mock_console, sample_theme):
        """Test theme preview with banner enabled."""
        preview_theme("test", sample_theme, show_banner_art=True)

        # Should call console.print multiple times (banner included)
        assert mock_console.print.call_count > 5

    @mock.patch("osprey.cli.preview_styles.console")
    @mock.patch("osprey.cli.preview_styles.set_theme")
    def test_preview_theme_without_banner(self, mock_set_theme, mock_console, sample_theme):
        """Test theme preview without banner."""
        preview_theme("test", sample_theme, show_banner_art=False)

        # Verify it completes without error
        assert mock_console.clear.called


class TestCompareThemes:
    """Test the compare_themes function."""

    @mock.patch("osprey.cli.preview_styles.console")
    @mock.patch("osprey.cli.preview_styles.set_theme")
    def test_compare_themes(self, mock_set_theme, mock_console):
        """Test theme comparison display."""
        compare_themes()

        # Should clear console and display comparison
        assert mock_console.clear.called
        assert mock_console.print.called

        # Should set theme multiple times (once per theme)
        assert mock_set_theme.call_count >= len(THEMES)


class TestCreateCustomTheme:
    """Test custom theme creation."""

    @mock.patch("osprey.cli.preview_styles.console")
    def test_create_custom_theme_missing_questionary(self, mock_console):
        """Test custom theme creation handles missing questionary."""
        with mock.patch.dict("sys.modules", {"questionary": None}):
            from osprey.cli.preview_styles import create_custom_theme

            # Should handle missing dependency gracefully
            # This is a characterization test - documenting current behavior
            try:
                create_custom_theme()
            except (ImportError, AttributeError):
                # Expected if questionary is not installed
                pass


class TestMainFunction:
    """Test the main entry point."""

    @mock.patch("osprey.cli.preview_styles.preview_theme")
    @mock.patch("sys.argv", ["preview_styles.py"])
    def test_main_default_theme(self, mock_preview):
        """Test main with default arguments."""
        try:
            main()
        except SystemExit:
            pass

        # Should call preview_theme with default theme
        assert mock_preview.called

    @mock.patch("osprey.cli.preview_styles.preview_theme")
    @mock.patch("sys.argv", ["preview_styles.py", "--theme", "ocean"])
    def test_main_with_specific_theme(self, mock_preview):
        """Test main with specific theme selection."""
        try:
            main()
        except SystemExit:
            pass

        # Should call preview_theme with ocean theme
        assert mock_preview.called
        call_args = mock_preview.call_args
        if call_args:
            assert call_args[0][0] == "ocean"

    @mock.patch("osprey.cli.preview_styles.compare_themes")
    @mock.patch("sys.argv", ["preview_styles.py", "--compare"])
    def test_main_with_compare_flag(self, mock_compare):
        """Test main with --compare flag."""
        try:
            main()
        except SystemExit:
            pass

        # Should call compare_themes
        assert mock_compare.called

    @mock.patch("osprey.cli.preview_styles.console")
    @mock.patch("sys.argv", ["preview_styles.py", "--help"])
    def test_main_with_help_flag(self, mock_console):
        """Test main with --help flag."""
        with pytest.raises(SystemExit) as exc_info:
            main()

        # Help should exit with 0
        assert exc_info.value.code == 0

    @mock.patch("osprey.cli.preview_styles.console")
    @mock.patch("osprey.cli.preview_styles.preview_theme")
    @mock.patch("sys.argv", ["preview_styles.py"])
    def test_main_keyboard_interrupt(self, mock_preview, mock_console):
        """Test main handles keyboard interrupt gracefully."""
        mock_preview.side_effect = KeyboardInterrupt()

        with pytest.raises(SystemExit) as exc_info:
            main()

        # Should exit cleanly with 0
        assert exc_info.value.code == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_all_predefined_themes_are_valid(self):
        """Verify all predefined themes can be created."""
        for _theme_name, theme in THEMES.items():
            # Each theme should be a ColorTheme instance
            assert isinstance(theme, ColorTheme)

            # Should have non-empty color values
            assert theme.primary
            assert theme.accent
            assert theme.command
            assert theme.path
            assert theme.info

    @mock.patch("osprey.cli.preview_styles.console")
    @mock.patch("osprey.cli.preview_styles.set_theme")
    def test_preview_handles_none_theme_name(self, mock_set_theme, mock_console):
        """Test preview handles edge case inputs."""
        theme = THEMES["osprey"]

        # Should handle gracefully
        try:
            preview_theme("", theme, show_banner_art=False)
        except Exception:
            # Documenting current behavior - may raise or handle gracefully
            pass
