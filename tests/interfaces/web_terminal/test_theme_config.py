"""Tests for `web.theme` config resolution and SSR no-FOUC rendering.

Task 1.10 (web-theme-config): `web.theme` in ``config.yml`` (top-level `web`
section, separate from the console-only `cli.theme`) is resolved to a
concrete baked theme id and server-rendered onto `<html data-theme>` so the
generated `theme-boot.js` (Task 1.8) first-paints with no flash.

Covers:
    - `resolve_web_theme_id` (pure resolver): family -> family's dark id,
      concrete id passthrough, unknown -> warn + fallback to osprey's dark id.
    - The render path: GET "/" contains the expected `data-theme="..."`.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.design_system.generator.emit_js import ThemeManifestEntry
from osprey.interfaces.web_terminal.app import create_app, resolve_web_theme_id

# A synthetic manifest mirroring the real baked tokens.js THEMES: the
# `osprey` family (dark/light) plus a `high-contrast` family (dark/light).
_ENTRIES = [
    ThemeManifestEntry(id="dark", label="Dark", mode="dark", family="osprey"),
    ThemeManifestEntry(id="light", label="Light", mode="light", family="osprey"),
    ThemeManifestEntry(
        id="high-contrast-dark", label="High Contrast Dark", mode="dark", family="high-contrast"
    ),
    ThemeManifestEntry(
        id="high-contrast-light", label="High Contrast Light", mode="light", family="high-contrast"
    ),
]
_DEFAULTS = {
    "osprey": {"dark": "dark", "light": "light"},
    "high-contrast": {"dark": "high-contrast-dark", "light": "high-contrast-light"},
}


class TestResolveWebThemeId:
    """Pure resolver: config value -> concrete baked theme id."""

    def test_family_resolves_to_family_dark_id(self):
        """A family name resolves to that family's dark id (the SSR default)."""
        assert resolve_web_theme_id("high-contrast", _ENTRIES, _DEFAULTS) == "high-contrast-dark"

    def test_osprey_family_resolves_to_dark(self):
        assert resolve_web_theme_id("osprey", _ENTRIES, _DEFAULTS) == "dark"

    def test_concrete_id_passes_through(self):
        """A concrete id (e.g. pinning a specific mode) is used as-is."""
        assert resolve_web_theme_id("light", _ENTRIES, _DEFAULTS) == "light"
        assert resolve_web_theme_id("high-contrast-light", _ENTRIES, _DEFAULTS) == (
            "high-contrast-light"
        )

    def test_unknown_value_warns_and_falls_back_to_osprey_dark(self, caplog):
        """An unrecognized value logs a warning and falls back to osprey's dark id."""
        with caplog.at_level(logging.WARNING):
            result = resolve_web_theme_id("nonsense", _ENTRIES, _DEFAULTS)

        assert result == "dark"
        assert any(
            "nonsense" in record.message and record.levelno == logging.WARNING
            for record in caplog.records
        ), "expected a WARNING mentioning the unknown value"

    def test_unknown_value_never_raises(self):
        """The resolver never raises on bad input — it only warns and falls back."""
        try:
            resolve_web_theme_id("", _ENTRIES, _DEFAULTS)
            resolve_web_theme_id(None, _ENTRIES, _DEFAULTS)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - failure path
            pytest.fail(f"resolve_web_theme_id raised unexpectedly: {exc}")

    def test_result_is_always_a_valid_baked_id(self):
        """Whatever is returned must be one of the concrete ids in the manifest.

        This is the contract theme-boot.js's `isValidId` depends on: an
        invalid/family-only id server-rendered onto `<html data-theme>`
        would silently fall through to OS-auto instead of honoring config.
        """
        valid_ids = {entry.id for entry in _ENTRIES}
        for configured in ("osprey", "high-contrast", "dark", "high-contrast-light", "bogus"):
            assert resolve_web_theme_id(configured, _ENTRIES, _DEFAULTS) in valid_ids


# ---- Render path: GET "/" server-renders the resolved data-theme ----


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


def _make_client(workspace_dir, configured_theme):
    """TestClient whose lifespan resolves `web.theme` = configured_theme."""
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.utils.config.get_config_value",
            return_value=configured_theme,
        ),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as c:
            yield c


class TestRenderedDataTheme:
    def test_family_config_renders_family_dark_id(self, workspace_dir):
        gen = _make_client(workspace_dir, "high-contrast")
        client = next(gen)
        try:
            body = client.get("/").text
            assert 'data-theme="high-contrast-dark"' in body
        finally:
            next(gen, None)

    def test_concrete_id_config_renders_as_is(self, workspace_dir):
        gen = _make_client(workspace_dir, "light")
        client = next(gen)
        try:
            body = client.get("/").text
            assert 'data-theme="light"' in body
        finally:
            next(gen, None)

    def test_unknown_config_renders_osprey_dark_fallback(self, workspace_dir):
        gen = _make_client(workspace_dir, "nonsense")
        client = next(gen)
        try:
            body = client.get("/").text
            assert 'data-theme="dark"' in body
        finally:
            next(gen, None)

    def test_looks_up_web_theme_key_with_osprey_default(self, workspace_dir):
        """The lifespan resolves the theme from `web.theme` with default 'osprey'.

        The lifespan reads other `web.*` keys too (e.g. the chat-pool bounds),
        so this asserts the theme *contract* — that `web.theme` is queried with
        the 'osprey' default and the rendered result reflects it — rather than
        that the theme read is the lifespan's only config lookup.
        """
        with (
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value={"watch_dir": str(workspace_dir)},
            ),
            patch(
                "osprey.utils.config.get_config_value", return_value="osprey"
            ) as mock_get_config_value,
            TestClient(create_app(shell_command="echo")) as client,
        ):
            body = client.get("/").text

        mock_get_config_value.assert_any_call("web.theme", "osprey")
        assert 'data-theme="dark"' in body

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
            TestClient(create_app(shell_command="echo")) as client,
        ):
            body = client.get("/").text
            assert 'data-theme="dark"' in body

    def test_switcher_element_mounted(self, workspace_dir):
        """The binary theme-toggle button is replaced by <osprey-theme-switcher>."""
        gen = _make_client(workspace_dir, "osprey")
        client = next(gen)
        try:
            body = client.get("/").text
            assert "<osprey-theme-switcher></osprey-theme-switcher>" in body
            assert 'id="theme-toggle"' not in body
        finally:
            next(gen, None)
