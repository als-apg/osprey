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

import pytest

from osprey.interfaces.web_terminal.app import DEFAULT_UI_MODE, resolve_ui_mode
from tests.interfaces.web_terminal._started_app import started_client


class TestResolveUiMode:
    """Pure resolver: config value -> concrete UI mode, never raising."""

    @pytest.mark.parametrize(
        ("configured", "expected"),
        [("expert", "expert"), ("simple", "simple"), ("", "expert"), (None, "expert")],
    )
    def test_resolves_to_a_mode(self, configured, expected):
        assert resolve_ui_mode(configured) == expected

    def test_unknown_value_warns_and_falls_back_to_default(self, caplog):
        """An unrecognized value logs a warning and falls back to the default mode."""
        with caplog.at_level(logging.WARNING):
            result = resolve_ui_mode("nonsense")

        assert result == DEFAULT_UI_MODE
        assert any(
            "nonsense" in record.message and record.levelno == logging.WARNING
            for record in caplog.records
        ), "expected a WARNING mentioning the unknown value"


# ---- Render + API paths: startup resolves web.ui_mode from config ----


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        pytest.param("expert", "expert", id="expert"),
        pytest.param("simple", "simple", id="simple"),
        pytest.param("nonsense", "expert", id="unknown"),
        # The default is the full surface, never the reduced one.
        pytest.param(None, "expert", id="absent"),
    ],
)
def test_ui_mode_reaches_page_and_payload(workspace_dir, configured, expected):
    """The SSR attribute is the first-paint rung; the payload mirrors it."""
    web = {} if configured is None else {"ui_mode": configured}
    with started_client(workspace_dir, web=web) as client:
        body = client.get("/").text
        payload = client.get("/api/panels").json()

    assert f'data-ui-mode="{expected}"' in body
    assert payload["ui_mode"] == expected
