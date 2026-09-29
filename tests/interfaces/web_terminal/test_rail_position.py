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

import pytest

from osprey.interfaces.web_terminal import app as web_app
from osprey.interfaces.web_terminal.app import (
    DEFAULT_RAIL_POSITION,
    FAMILY_RAIL_DEFAULTS,
    RAIL_POSITIONS,
    family_rail_default,
    resolve_rail_position,
)
from tests.interfaces.web_terminal._started_app import started_client


class TestResolveRailPosition:
    """Pure resolver: config value -> concrete rail position."""

    @pytest.mark.parametrize("configured", ["left", "top"])
    def test_a_real_position_passes_through(self, configured):
        assert resolve_rail_position(configured) == configured

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

        assert not [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and r.name == web_app.logger.name
        ]

    @pytest.mark.parametrize("bad", ["", None, 3])
    def test_bad_values_never_raise(self, bad):
        """The resolver never raises on bad input — it only warns and falls back."""
        assert resolve_rail_position(bad) == DEFAULT_RAIL_POSITION


class TestFamilyRailDefault:
    """The theme-family -> rail-position coupling."""

    def test_retro_family_implies_the_top_rail(self):
        """Retro is the pre-redesign look, tab bar included."""
        assert family_rail_default("retro") == "top"

    @pytest.mark.parametrize("family", ["main", "high-contrast", "nonesuch", None])
    def test_other_families_get_the_default(self, family):
        assert family_rail_default(family) == DEFAULT_RAIL_POSITION

    def test_every_mapped_position_is_a_real_position(self):
        """A typo in the map would server-render an attribute nothing honors."""
        assert set(FAMILY_RAIL_DEFAULTS.values()) <= set(RAIL_POSITIONS)


class TestResolveRailPositionWithFamily:
    """Explicit config outranks the family; an absent key defers to it."""

    @pytest.mark.parametrize(("family", "expected"), [("retro", "top"), ("main", "left")])
    def test_absent_key_follows_the_family(self, family, expected):
        assert resolve_rail_position(None, family) == expected

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


def _started(workspace_dir, configured_position, configured_theme="main"):
    """Start the app with ``web.rail_position`` (``None`` omits the key) and ``web.theme``."""
    web = {} if configured_position is None else {"rail_position": configured_position}
    return started_client(workspace_dir, web=web, config_values={"web.theme": configured_theme})


@pytest.mark.parametrize(
    ("configured", "position", "is_configured"),
    [
        pytest.param("left", "left", True, id="left"),
        pytest.param("top", "top", True, id="top"),
        # A typo resolves like an absent key, and reports like one too.
        pytest.param("sideways", "left", False, id="unknown"),
        pytest.param(None, "left", False, id="absent"),
    ],
)
def test_rail_reaches_page_and_payload(workspace_dir, configured, position, is_configured):
    """The SSR attribute, the echoed position, and whether config stated it.

    ``rail_position_configured`` tells the browser whether a live theme-family
    switch may move the rail: an explicit position outranks the family.
    """
    with _started(workspace_dir, configured) as client:
        body = client.get("/").text
        payload = client.get("/api/panels").json()

    assert f'data-rail-position="{position}"' in body
    assert payload["rail_position"] == position
    assert payload["rail_position_configured"] is is_configured


class TestRetroFamilyMovesTheRail:
    """The coupling, end to end through the server."""

    def test_retro_theme_with_no_rail_config_renders_the_top_rail(self, workspace_dir):
        """Selecting Retro alone restores the pre-redesign tab-bar arrangement."""
        with _started(workspace_dir, None, configured_theme="retro") as client:
            body = client.get("/").text

        assert 'data-theme="retro-dark"' in body
        assert 'data-rail-position="top"' in body

    def test_retro_theme_respects_an_explicit_left_rail(self, workspace_dir):
        """A deployment that pinned the rail keeps it, retro or not."""
        with _started(workspace_dir, "left", configured_theme="retro") as client:
            assert 'data-rail-position="left"' in client.get("/").text


def test_payload_carries_the_family_coupling(workspace_dir):
    """What the browser needs to follow a live theme-family switch."""
    with _started(workspace_dir, None) as client:
        payload = client.get("/api/panels").json()

    assert payload["family_rail_defaults"] == dict(FAMILY_RAIL_DEFAULTS)
