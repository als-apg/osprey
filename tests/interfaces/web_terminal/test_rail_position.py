"""Tests for `web.rail_position` config resolution, SSR stamping, and API echo.

`web.rail_position` in ``config.yml`` (top-level `web` section) selects where
the panel rail sits — ``left`` (the redesign's icon-rail column) or ``top``
(the same rail rendered as a horizontal strip under the header, the
arrangement operators know from the pre-redesign tab bar). It is resolved to a
concrete position and server-rendered onto ``<html data-rail-position>`` so
the pre-paint rail-boot script first-paints the right orientation with no
flash. ``GET /api/panels`` also echoes the resolved position, but first paint
must never depend on that API field — the SSR attribute is the authoritative
rung.

When the key is absent the position comes from the active theme family
(``FAMILY_RAIL_DEFAULTS``): the ``retro`` family restores the pre-redesign
look, and the horizontal tab strip is part of that look. An explicit
``web.rail_position`` outranks the family — a deployment that states a
position keeps it in every theme.

Covers:
    - `resolve_rail_position` (pure resolver): valid-position passthrough,
      unknown -> warn + fall back to what the family implies, never raises.
    - `family_rail_default`: the family -> position map, and its fallback.
    - The render path: GET "/" contains the expected `data-rail-position="..."`,
      including the retro-family default with no rail config at all.
    - The API path: GET "/api/panels" carries the resolved `rail_position`
      plus the coupling the browser needs to follow a live theme switch.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import (
    DEFAULT_RAIL_POSITION,
    FAMILY_RAIL_DEFAULTS,
    RAIL_POSITIONS,
    create_app,
    family_rail_default,
    resolve_rail_position,
)


class TestResolveRailPosition:
    """Pure resolver: config value -> concrete rail position."""

    def test_left_passes_through(self):
        assert resolve_rail_position("left") == "left"

    def test_top_passes_through(self):
        assert resolve_rail_position("top") == "top"

    def test_default_is_left(self):
        """The default position is the redesign's left rail column."""
        assert DEFAULT_RAIL_POSITION == "left"
        assert RAIL_POSITIONS == ("left", "top")

    def test_unknown_value_warns_and_falls_back_to_default(self, caplog):
        """An unrecognized value logs a warning and falls back to the default."""
        with caplog.at_level(logging.WARNING):
            result = resolve_rail_position("sideways")

        assert result == DEFAULT_RAIL_POSITION
        assert any(
            "sideways" in record.message and record.levelno == logging.WARNING
            for record in caplog.records
        ), "expected a WARNING mentioning the unknown value"

    def test_absent_key_does_not_warn(self, caplog):
        """``None`` means "key absent", which is normal — not a misconfiguration."""
        with caplog.at_level(logging.WARNING):
            resolve_rail_position(None)

        assert not [r for r in caplog.records if "rail_position" in r.message]

    @pytest.mark.parametrize("bad", ["", None, 3])
    def test_bad_values_never_raise(self, bad):
        """The resolver never raises on bad input — it only warns and falls back."""
        assert resolve_rail_position(bad) == DEFAULT_RAIL_POSITION

    def test_result_is_always_a_valid_position(self):
        """Whatever is returned must be one of the concrete supported positions.

        This is the contract the pre-paint rail-boot rung depends on: an
        invalid position server-rendered onto `<html data-rail-position>`
        would leave the client with nothing real to honor.
        """
        for configured in ("left", "top", "", "bogus", None):
            assert resolve_rail_position(configured) in RAIL_POSITIONS  # type: ignore[arg-type]


class TestFamilyRailDefault:
    """The theme-family -> rail-position coupling."""

    def test_retro_family_implies_the_top_rail(self):
        """Retro is the pre-redesign look, tab bar included."""
        assert family_rail_default("retro") == "top"
        assert FAMILY_RAIL_DEFAULTS["retro"] == "top"

    def test_other_families_get_the_default(self):
        assert family_rail_default("main") == DEFAULT_RAIL_POSITION
        assert family_rail_default("high-contrast") == DEFAULT_RAIL_POSITION

    def test_unknown_and_missing_family_get_the_default(self):
        assert family_rail_default("nonesuch") == DEFAULT_RAIL_POSITION
        assert family_rail_default(None) == DEFAULT_RAIL_POSITION

    def test_every_mapped_position_is_a_real_position(self):
        """A typo in the map would server-render an attribute nothing honors."""
        assert set(FAMILY_RAIL_DEFAULTS.values()) <= set(RAIL_POSITIONS)


class TestResolveRailPositionWithFamily:
    """Explicit config outranks the family; an absent key defers to it."""

    def test_absent_key_follows_the_retro_family(self):
        assert resolve_rail_position(None, "retro") == "top"

    def test_absent_key_follows_the_main_family(self):
        assert resolve_rail_position(None, "main") == "left"

    @pytest.mark.parametrize("configured", ["left", "top"])
    def test_explicit_config_outranks_the_family(self, configured):
        """A deployment that states a position keeps it in every theme."""
        assert resolve_rail_position(configured, "retro") == configured
        assert resolve_rail_position(configured, "main") == configured

    def test_unknown_config_falls_through_to_the_family(self):
        """A typo is treated as absent, not as a reason to ignore the theme."""
        assert resolve_rail_position("sideways", "retro") == "top"


# ---- Render + API paths: startup resolves web.rail_position from config ----


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


def _make_client(workspace_dir, configured_position, configured_theme="main"):
    """TestClient whose lifespan resolves `web.rail_position` = configured_position.

    ``configured_position`` of ``None`` omits the ``rail_position`` key
    entirely, exercising the "key absent -> default" path. ``load_osprey_config``
    is patched because the lifespan reads the top-level ``web`` section through
    it (the same reader the panel loaders use); with no ``panels`` key only the
    universal panels are enabled.
    """
    web_section: dict = {}
    if configured_position is not None:
        web_section["rail_position"] = configured_position
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"web": web_section},
        ),
        # web.theme is read through a different reader than the `web` section
        # above; the rail block consults the family it resolves to.
        patch("osprey.utils.config.get_config_value", return_value=configured_theme),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as c:
            yield c


class TestRenderedDataRailPosition:
    def test_left_config_renders_left(self, workspace_dir):
        gen = _make_client(workspace_dir, "left")
        client = next(gen)
        try:
            body = client.get("/").text
            assert 'data-rail-position="left"' in body
        finally:
            next(gen, None)

    def test_top_config_renders_top(self, workspace_dir):
        gen = _make_client(workspace_dir, "top")
        client = next(gen)
        try:
            body = client.get("/").text
            assert 'data-rail-position="top"' in body
        finally:
            next(gen, None)

    def test_unknown_config_renders_default_fallback(self, workspace_dir):
        gen = _make_client(workspace_dir, "sideways")
        client = next(gen)
        try:
            body = client.get("/").text
            assert f'data-rail-position="{DEFAULT_RAIL_POSITION}"' in body
        finally:
            next(gen, None)

    def test_missing_key_renders_default(self, workspace_dir):
        """No `web.rail_position` key at all -> the default position is rendered."""
        gen = _make_client(workspace_dir, None)
        client = next(gen)
        try:
            body = client.get("/").text
            assert f'data-rail-position="{DEFAULT_RAIL_POSITION}"' in body
        finally:
            next(gen, None)


class TestPanelsPayloadRailPosition:
    def test_payload_carries_resolved_top_position(self, workspace_dir):
        gen = _make_client(workspace_dir, "top")
        client = next(gen)
        try:
            payload = client.get("/api/panels").json()
            assert payload["rail_position"] == "top"
        finally:
            next(gen, None)

    def test_payload_carries_resolved_left_position(self, workspace_dir):
        gen = _make_client(workspace_dir, "left")
        client = next(gen)
        try:
            payload = client.get("/api/panels").json()
            assert payload["rail_position"] == "left"
        finally:
            next(gen, None)

    def test_payload_unknown_position_falls_back_to_default(self, workspace_dir):
        gen = _make_client(workspace_dir, "sideways")
        client = next(gen)
        try:
            payload = client.get("/api/panels").json()
            assert payload["rail_position"] == DEFAULT_RAIL_POSITION
        finally:
            next(gen, None)


class TestRetroFamilyMovesTheRail:
    """The coupling, end to end through the server."""

    def test_retro_theme_with_no_rail_config_renders_the_top_rail(self, workspace_dir):
        """Selecting Retro alone restores the pre-redesign tab-bar arrangement."""
        gen = _make_client(workspace_dir, None, configured_theme="retro")
        client = next(gen)
        try:
            body = client.get("/").text
            assert 'data-theme="retro-dark"' in body
            assert 'data-rail-position="top"' in body
        finally:
            next(gen, None)

    def test_retro_theme_respects_an_explicit_left_rail(self, workspace_dir):
        """A deployment that pinned the rail keeps it, retro or not."""
        gen = _make_client(workspace_dir, "left", configured_theme="retro")
        client = next(gen)
        try:
            body = client.get("/").text
            assert 'data-rail-position="left"' in body
        finally:
            next(gen, None)

    def test_main_theme_with_no_rail_config_keeps_the_left_rail(self, workspace_dir):
        gen = _make_client(workspace_dir, None, configured_theme="main")
        client = next(gen)
        try:
            assert 'data-rail-position="left"' in client.get("/").text
        finally:
            next(gen, None)


class TestPanelsPayloadRailCoupling:
    """What the browser needs to follow a live theme-family switch."""

    def test_payload_carries_the_family_coupling(self, workspace_dir):
        gen = _make_client(workspace_dir, None)
        client = next(gen)
        try:
            payload = client.get("/api/panels").json()
            assert payload["family_rail_defaults"] == dict(FAMILY_RAIL_DEFAULTS)
        finally:
            next(gen, None)

    def test_unconfigured_rail_is_not_reported_as_configured(self, workspace_dir):
        """An absent key leaves the rail free to follow the theme."""
        gen = _make_client(workspace_dir, None)
        client = next(gen)
        try:
            assert client.get("/api/panels").json()["rail_position_configured"] is False
        finally:
            next(gen, None)

    @pytest.mark.parametrize("configured", ["left", "top"])
    def test_configured_rail_is_reported_as_configured(self, workspace_dir, configured):
        """An explicit position tells the browser not to move the rail on a theme switch."""
        gen = _make_client(workspace_dir, configured)
        client = next(gen)
        try:
            assert client.get("/api/panels").json()["rail_position_configured"] is True
        finally:
            next(gen, None)

    def test_unknown_configured_rail_is_not_reported_as_configured(self, workspace_dir):
        """A typo resolves like an absent key, and reports like one too."""
        gen = _make_client(workspace_dir, "sideways")
        client = next(gen)
        try:
            assert client.get("/api/panels").json()["rail_position_configured"] is False
        finally:
            next(gen, None)
