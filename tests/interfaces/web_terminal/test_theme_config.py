"""Tests for `web.theme` config resolution and SSR no-FOUC rendering.

Task 1.10 (web-theme-config): `web.theme` in ``config.yml`` (top-level `web`
section, separate from the console-only `cli.theme`) is resolved to a
concrete baked theme id and server-rendered onto `<html data-theme>` so the
generated `theme-boot.js` (Task 1.8) first-paints with no flash.

Covers:
    - `resolve_theme_id` (pure resolver): family -> family's dark id,
      concrete id passthrough, unknown -> warn + fallback to osprey's dark id.
    - The render path: GET "/" contains the expected `data-theme="..."`.
"""

from __future__ import annotations

import dataclasses
import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.design_system.generator.emit_js import ThemeManifestEntry
from osprey.interfaces.design_system.theme_config import (
    DEFAULT_WEB_THEME,
    configured_web_theme,
    resolve_configured_web_theme,
    resolve_pinned_mode,
    resolve_theme_id,
)
from osprey.interfaces.web_terminal.app import create_app
from tests.interfaces.web_terminal._started_app import started_client

# A synthetic manifest mirroring the real baked tokens.js THEMES: the
# `main` family (dark/light) plus a `high-contrast` family (dark/light).
_ENTRIES = [
    ThemeManifestEntry(id="dark", label="Dark", mode="dark", family="main"),
    ThemeManifestEntry(id="light", label="Light", mode="light", family="main"),
    ThemeManifestEntry(
        id="high-contrast-dark", label="High Contrast Dark", mode="dark", family="high-contrast"
    ),
    ThemeManifestEntry(
        id="high-contrast-light", label="High Contrast Light", mode="light", family="high-contrast"
    ),
]
_DEFAULTS = {
    "main": {"dark": "dark", "light": "light"},
    "high-contrast": {"dark": "high-contrast-dark", "light": "high-contrast-light"},
}


class TestResolveWebThemeId:
    """Pure resolver: config value -> concrete baked theme id.

    Every answer must be a concrete id in the manifest — the contract
    theme-boot.js's ``isValidId`` depends on: a family-only or unknown id
    server-rendered onto ``<html data-theme>`` would silently fall through to
    OS-auto instead of honoring config.
    """

    @pytest.mark.parametrize(
        ("configured", "expected"),
        [
            ("high-contrast", "high-contrast-dark"),
            ("main", "dark"),
            ("light", "light"),
            ("high-contrast-light", "high-contrast-light"),
            ("dark", "dark"),
            ("bogus", "dark"),
            ("", "dark"),
            (None, "dark"),
        ],
    )
    def test_resolves_to_a_concrete_id(self, configured, expected):
        assert resolve_theme_id(configured, _ENTRIES, _DEFAULTS) == expected

    def test_unknown_value_warns_and_falls_back_to_osprey_dark(self, caplog):
        """An unrecognized value logs a warning and falls back to osprey's dark id."""
        with caplog.at_level(logging.WARNING):
            result = resolve_theme_id("nonsense", _ENTRIES, _DEFAULTS)

        assert result == "dark"
        assert any(
            "nonsense" in record.message and record.levelno == logging.WARNING
            for record in caplog.records
        ), "expected a WARNING mentioning the unknown value"


# ---- Render path: GET "/" server-renders the resolved data-theme ----


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


class TestRenderedDataTheme:
    @pytest.mark.parametrize(
        ("env", "configured", "theme", "mode"),
        [
            pytest.param(None, "high-contrast", "high-contrast-dark", None, id="family"),
            pytest.param(None, "light", "light", "light", id="concrete-id"),
            pytest.param(None, "high-contrast-light", "high-contrast-light", "light", id="pin"),
            pytest.param(None, "nonsense", "dark", None, id="unknown"),
            pytest.param("high-contrast-light", "main", "high-contrast-light", "light", id="env"),
            pytest.param("high-contrast", "main", "high-contrast-dark", None, id="env-family"),
        ],
    )
    @pytest.mark.parametrize(
        "path",
        [
            pytest.param("/", id="index"),
            pytest.param("/static/session.html", id="session-page"),
        ],
    )
    def test_rendered_theme(self, workspace_dir, monkeypatch, env, configured, theme, mode, path):
        """``web.theme`` (or ``OSPREY_WEB_THEME`` over it) reaches ``<html>``.

        ``data-theme-mode`` is rendered only when the value pinned a mode:
        absence is what tells the hub to stay on 'auto', so a family or a
        fallback emits no attribute at all.
        """
        if env is not None:
            monkeypatch.setenv("OSPREY_WEB_THEME", env)
        with started_client(workspace_dir, config_values={"web.theme": configured}) as client:
            body = client.get(path).text

        assert f'data-theme="{theme}"' in body
        if mode is None:
            assert "data-theme-mode" not in body
        else:
            assert f'data-theme-mode="{mode}"' in body

    def test_missing_config_yml_fails_open_to_dark(self, workspace_dir):
        """No config.yml at all (FileNotFoundError from get_config_value) -> fallback 'dark'."""
        with (
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value={"watch_dir": str(workspace_dir)},
            ),
            patch(
                "osprey.utils.config.get_config_value",
                side_effect=FileNotFoundError("no config.yml found"),
            ),
            TestClient(create_app(shell_command=["echo"])) as client,
        ):
            body = client.get("/").text
            assert 'data-theme="dark"' in body

    def test_display_menu_mounted(self, workspace_dir):
        """The hub header mounts the shared ``<osprey-display-menu>``.

        The always-visible ``<osprey-theme-switcher>`` and the old binary
        ``#theme-toggle`` button are both gone from the hub page — theme
        controls live inside the component's popover card (session.html keeps
        the shared switcher component). The hub projects its own Settings row
        into the card, so that id is still rendered.
        """
        with started_client(workspace_dir) as client:
            body = client.get("/").text

        assert '<osprey-display-menu id="display-menu">' in body
        assert 'id="display-menu-settings"' in body
        assert 'id="display-menu-btn"' not in body
        assert "<osprey-theme-switcher>" not in body
        assert 'id="theme-toggle"' not in body


# ---- The mode pin: family vs concrete id ----


class TestResolveWebThemePinnedMode:
    """A configured value either states a mode or leaves it to the OS."""

    def test_concrete_id_pins_its_own_mode(self):
        assert resolve_pinned_mode("high-contrast-light", _ENTRIES) == "light"
        assert resolve_pinned_mode("dark", _ENTRIES) == "dark"

    def test_family_pins_nothing(self):
        """A family states a palette only — light/dark stays the operator's OS call."""
        assert resolve_pinned_mode("high-contrast", _ENTRIES) is None
        assert resolve_pinned_mode("main", _ENTRIES) is None

    def test_unknown_value_pins_nothing(self):
        """An unknown value falls back; a fallback must not pose as stated intent."""
        assert resolve_pinned_mode("nonsense", _ENTRIES) is None
        assert resolve_pinned_mode("", _ENTRIES) is None


# ---- The shared chain: environment -> web.theme -> id + pin + family ----


class TestConfiguredWebTheme:
    """`configured_web_theme()` — the raw value, read once for every surface.

    The web terminal (above) and the artifact gallery both server-render a
    theme, and both used to spell this precedence out for themselves; it now
    lives in the design system so the two cannot drift apart.
    """

    def test_env_var_outranks_config(self, monkeypatch):
        monkeypatch.setenv("OSPREY_WEB_THEME", "desy-light")
        with patch("osprey.utils.config.get_config_value", return_value="main") as get_value:
            assert configured_web_theme() == "desy-light"
        get_value.assert_not_called()  # the config is not even read

    def test_blank_env_var_falls_through_to_config(self, monkeypatch):
        """An empty env var is 'unset', not 'the empty theme'."""
        monkeypatch.setenv("OSPREY_WEB_THEME", "   ")
        with patch("osprey.utils.config.get_config_value", return_value="retro") as get_value:
            assert configured_web_theme() == "retro"
        get_value.assert_called_once_with("web.theme", "main")

    def test_absent_env_var_reads_web_theme_with_the_main_default(self, monkeypatch):
        monkeypatch.delenv("OSPREY_WEB_THEME", raising=False)
        with patch("osprey.utils.config.get_config_value", return_value="main") as get_value:
            assert configured_web_theme() == DEFAULT_WEB_THEME
        get_value.assert_called_once_with("web.theme", "main")

    def test_empty_config_value_falls_back_to_the_default(self, monkeypatch):
        """`web.theme:` with nothing after it reads as None, not as a theme name."""
        monkeypatch.delenv("OSPREY_WEB_THEME", raising=False)
        with patch("osprey.utils.config.get_config_value", return_value=None):
            assert configured_web_theme() == DEFAULT_WEB_THEME

    def test_config_read_error_propagates(self, monkeypatch):
        """No config primed: the callers disagree about what to do (the terminal
        renders a fallback theme, the gallery serves unpinned pages), so this
        must not decide for them."""
        monkeypatch.delenv("OSPREY_WEB_THEME", raising=False)
        with (
            patch(
                "osprey.utils.config.get_config_value",
                side_effect=FileNotFoundError("no config.yml found"),
            ),
            pytest.raises(FileNotFoundError),
        ):
            configured_web_theme()


class TestResolveConfiguredWebTheme:
    """`resolve_configured_web_theme()` — id, pin and family in one read.

    Resolved against the real baked token tree, since the point of the helper
    is that every surface sees the same registry.
    """

    def test_concrete_id_resolves_to_itself_and_pins_its_mode(self, monkeypatch):
        monkeypatch.setenv("OSPREY_WEB_THEME", "high-contrast-light")
        resolved = resolve_configured_web_theme()
        assert (resolved.id, resolved.pinned_mode, resolved.family) == (
            "high-contrast-light",
            "light",
            "high-contrast",
        )

    def test_family_resolves_to_its_dark_id_but_pins_nothing(self, monkeypatch):
        monkeypatch.setenv("OSPREY_WEB_THEME", "high-contrast")
        resolved = resolve_configured_web_theme()
        assert resolved.id == "high-contrast-dark"
        assert resolved.pinned_mode is None
        assert resolved.family == "high-contrast"

    def test_unknown_value_falls_back_without_posing_as_a_pin(self, monkeypatch):
        monkeypatch.setenv("OSPREY_WEB_THEME", "nonsense")
        resolved = resolve_configured_web_theme()
        assert resolved.id == "dark"
        assert resolved.pinned_mode is None
        assert resolved.family == "main"

    def test_explicit_value_bypasses_the_environment(self, monkeypatch):
        monkeypatch.setenv("OSPREY_WEB_THEME", "high-contrast-light")
        assert resolve_configured_web_theme("light").id == "light"

    def test_result_is_immutable(self):
        """A resolved theme is read once at startup and shared; it must not be
        editable in place by whatever reads it later."""
        resolved = resolve_configured_web_theme("light")
        with pytest.raises(dataclasses.FrozenInstanceError):
            resolved.id = "dark"  # type: ignore[misc]
