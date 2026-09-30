"""Tests for `web.tour` resolution, the env override, and the API echo.

The onboarding tour's invite policy (``once`` / ``always`` / ``never``) is
resolved once at startup from ``OSPREY_WEB_TOUR`` (the per-user roster path,
which outranks config — the ``OSPREY_WEB_THEME`` precedence) falling back to
``web.tour``. ``GET /api/panels`` echoes the resolved policy together with
the derived capability list for the tour's "Ask in plain language" card; the
browser renders those facts and never invents its own.

Covers:
    - `resolve_tour_policy` (pure resolver): valid-policy passthrough,
      unknown -> warn + fallback to the default, never raises.
    - The API path: GET "/api/panels" carries ``tour.policy`` (config,
      env-override, unknown-fallback, key-absent), ``tour.capabilities``
      (core executor lines, ARIEL logbook line, and never a reading line —
      the browser derives that from the active target's kind) and
      ``tour.logbook``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import (
    DEFAULT_TOUR_POLICY,
    create_app,
    resolve_tour_policy,
)
from tests.interfaces.web_terminal._started_app import started_client


class TestResolveTourPolicy:
    """Pure resolver: config/env value -> concrete invite policy."""

    @pytest.mark.parametrize(
        ("configured", "expected"),
        [
            ("once", "once"),
            ("always", "always"),
            ("never", "never"),
            ("", "once"),
            (None, "once"),
        ],
    )
    def test_resolves_to_a_policy(self, configured, expected):
        assert resolve_tour_policy(configured) == expected

    def test_unknown_value_warns_and_falls_back_to_default(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = resolve_tour_policy("nonsense")

        assert result == DEFAULT_TOUR_POLICY
        assert any(
            "nonsense" in record.message and record.levelno == logging.WARNING
            for record in caplog.records
        ), "expected a WARNING mentioning the unknown value"


# ---- API path: startup resolves web.tour / OSPREY_WEB_TOUR from config ----


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


def _tour_payload(workspace_dir, web: dict) -> dict:
    with started_client(workspace_dir, web=web) as client:
        return client.get("/api/panels").json()["tour"]


@contextmanager
def _started_over(workspace_dir, config: dict):
    """Start the app with ``load_osprey_config`` answering *config* whole.

    :func:`started_client` answers the ``web`` section only; the capability
    check below needs a top-level section beside it.
    """
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch("osprey.utils.workspace.load_osprey_config", return_value=config),
    ):
        with TestClient(create_app(shell_command=["echo"])) as client:
            yield client


class TestPanelsPayloadTour:
    @pytest.mark.parametrize(
        ("configured", "expected"),
        [("once", "once"), ("always", "always"), ("never", "never"), ("nonsense", "once")],
    )
    def test_payload_carries_configured_policy(self, workspace_dir, configured, expected):
        assert _tour_payload(workspace_dir, {"tour": configured})["policy"] == expected

    def test_missing_key_resolves_to_default(self, workspace_dir):
        assert _tour_payload(workspace_dir, {})["policy"] == "once"

    @pytest.mark.parametrize(("env", "expected"), [("never", "never"), ("sometimes", "once")])
    def test_env_override_outranks_config(self, workspace_dir, monkeypatch, env, expected):
        """OSPREY_WEB_TOUR (the per-user roster path) wins over web.tour."""
        monkeypatch.setenv("OSPREY_WEB_TOUR", env)
        assert _tour_payload(workspace_dir, {"tour": "always"})["policy"] == expected


class TestPanelsPayloadTourCapabilities:
    def test_baseline_capabilities_without_control_system(self, workspace_dir):
        """No control_system, no ARIEL: the core executor lines, and no logbook."""
        tour = _tour_payload(workspace_dir, {})
        assert tour["capabilities"] == ["run Python analysis", "make plots"]
        assert tour["logbook"] is False

    def test_control_system_adds_no_read_line(self, workspace_dir):
        """A configured connector says nothing about what is behind it.

        ``control_system.type`` is set on a mock deployment too, so the server
        never claims a reading capability; the browser derives that wording
        from the active control target's kind.
        """
        config = {"web": {}, "control_system": {"type": "mock"}}
        with _started_over(workspace_dir, config) as client:
            tour = client.get("/api/panels").json()["tour"]
        assert tour["capabilities"] == ["run Python analysis", "make plots"]

    def test_ariel_panel_adds_the_logbook_line_last(self, workspace_dir):
        tour = _tour_payload(workspace_dir, {"panels": {"ariel": True}})
        assert tour["capabilities"] == [
            "run Python analysis",
            "make plots",
            "search the logbook",
        ]
        assert tour["logbook"] is True
