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

Covers:
    - `resolve_rail_position` (pure resolver): valid-position passthrough,
      unknown -> warn + fallback to the default, never raises.
    - The render path: GET "/" contains the expected `data-rail-position="..."`.
    - The API path: GET "/api/panels" carries the resolved `rail_position`.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import (
    DEFAULT_RAIL_POSITION,
    RAIL_POSITIONS,
    create_app,
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


# ---- Render + API paths: startup resolves web.rail_position from config ----


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


def _make_client(workspace_dir, configured_position):
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
