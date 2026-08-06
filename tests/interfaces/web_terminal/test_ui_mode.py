"""Tests for `web.ui_mode` config resolution, SSR stamping, and API echo.

Task 5.1 (ui-mode-config-api): `web.ui_mode` in ``config.yml`` (top-level
`web` section) selects the web UI surface — ``expert`` (full split-pane
terminal workspace) or ``simple`` (pared-down operator layout). It is resolved
to a concrete mode and server-rendered onto ``<html data-ui-mode>`` so the
pre-paint mode-boot script (Task 5.2) first-paints in the right mode with no
flash. ``GET /api/panels`` also echoes the resolved mode, but first paint must
never depend on that API field — the SSR attribute is the authoritative rung.

Covers:
    - `resolve_ui_mode` (pure resolver): valid-mode passthrough, unknown ->
      warn + fallback to the default mode, never raises.
    - The render path: GET "/" contains the expected `data-ui-mode="..."`.
    - The API path: GET "/api/panels" carries the resolved `ui_mode`.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import (
    DEFAULT_UI_MODE,
    UI_MODES,
    create_app,
    resolve_ui_mode,
)


class TestResolveUiMode:
    """Pure resolver: config value -> concrete UI mode."""

    def test_expert_passes_through(self):
        assert resolve_ui_mode("expert") == "expert"

    def test_simple_passes_through(self):
        assert resolve_ui_mode("simple") == "simple"

    def test_default_is_expert(self):
        """The default mode is the full expert surface, never the reduced one."""
        assert DEFAULT_UI_MODE == "expert"
        assert DEFAULT_UI_MODE in UI_MODES

    def test_unknown_value_warns_and_falls_back_to_default(self, caplog):
        """An unrecognized value logs a warning and falls back to the default mode."""
        with caplog.at_level(logging.WARNING):
            result = resolve_ui_mode("nonsense")

        assert result == DEFAULT_UI_MODE
        assert any(
            "nonsense" in record.message and record.levelno == logging.WARNING
            for record in caplog.records
        ), "expected a WARNING mentioning the unknown value"

    def test_empty_and_none_never_raise(self):
        """The resolver never raises on bad input — it only warns and falls back."""
        try:
            assert resolve_ui_mode("") == DEFAULT_UI_MODE
            assert resolve_ui_mode(None) == DEFAULT_UI_MODE  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - failure path
            pytest.fail(f"resolve_ui_mode raised unexpectedly: {exc}")

    def test_result_is_always_a_valid_mode(self):
        """Whatever is returned must be one of the concrete supported modes.

        This is the contract the pre-paint mode-boot rung depends on: an
        invalid mode server-rendered onto `<html data-ui-mode>` would leave
        the client with nothing real to honor.
        """
        for configured in ("expert", "simple", "", "bogus", None):
            assert resolve_ui_mode(configured) in UI_MODES  # type: ignore[arg-type]


# ---- Render + API paths: startup resolves web.ui_mode from config ----


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


def _make_client(workspace_dir, configured_mode):
    """TestClient whose lifespan resolves `web.ui_mode` = configured_mode.

    ``configured_mode`` of ``None`` omits the ``ui_mode`` key entirely, exercising
    the "key absent -> default" path. ``load_osprey_config`` is patched because
    the lifespan reads the top-level ``web`` section through it (the same reader
    the panel loaders use); with no ``panels`` key only the universal panels are
    enabled.
    """
    web_section: dict = {}
    if configured_mode is not None:
        web_section["ui_mode"] = configured_mode
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


class TestRenderedDataUiMode:
    def test_expert_config_renders_expert(self, workspace_dir):
        gen = _make_client(workspace_dir, "expert")
        client = next(gen)
        try:
            body = client.get("/").text
            assert 'data-ui-mode="expert"' in body
        finally:
            next(gen, None)

    def test_simple_config_renders_simple(self, workspace_dir):
        gen = _make_client(workspace_dir, "simple")
        client = next(gen)
        try:
            body = client.get("/").text
            assert 'data-ui-mode="simple"' in body
        finally:
            next(gen, None)

    def test_unknown_config_renders_default_fallback(self, workspace_dir):
        gen = _make_client(workspace_dir, "nonsense")
        client = next(gen)
        try:
            body = client.get("/").text
            assert f'data-ui-mode="{DEFAULT_UI_MODE}"' in body
        finally:
            next(gen, None)

    def test_missing_key_renders_default(self, workspace_dir):
        """No `web.ui_mode` key at all -> the default mode is rendered."""
        gen = _make_client(workspace_dir, None)
        client = next(gen)
        try:
            body = client.get("/").text
            assert f'data-ui-mode="{DEFAULT_UI_MODE}"' in body
        finally:
            next(gen, None)


class TestPanelsPayloadUiMode:
    def test_payload_carries_resolved_simple_mode(self, workspace_dir):
        gen = _make_client(workspace_dir, "simple")
        client = next(gen)
        try:
            payload = client.get("/api/panels").json()
            assert payload["ui_mode"] == "simple"
        finally:
            next(gen, None)

    def test_payload_carries_resolved_expert_mode(self, workspace_dir):
        gen = _make_client(workspace_dir, "expert")
        client = next(gen)
        try:
            payload = client.get("/api/panels").json()
            assert payload["ui_mode"] == "expert"
        finally:
            next(gen, None)

    def test_payload_unknown_mode_falls_back_to_default(self, workspace_dir):
        gen = _make_client(workspace_dir, "nonsense")
        client = next(gen)
        try:
            payload = client.get("/api/panels").json()
            assert payload["ui_mode"] == DEFAULT_UI_MODE
        finally:
            next(gen, None)
