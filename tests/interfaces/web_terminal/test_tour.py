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
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import (
    DEFAULT_TOUR_POLICY,
    TOUR_POLICIES,
    create_app,
    resolve_tour_policy,
)


class TestResolveTourPolicy:
    """Pure resolver: config/env value -> concrete invite policy."""

    def test_valid_policies_pass_through(self):
        for policy in TOUR_POLICIES:
            assert resolve_tour_policy(policy) == policy

    def test_default_is_once(self):
        assert DEFAULT_TOUR_POLICY == "once"
        assert DEFAULT_TOUR_POLICY in TOUR_POLICIES

    def test_unknown_value_warns_and_falls_back_to_default(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = resolve_tour_policy("nonsense")

        assert result == DEFAULT_TOUR_POLICY
        assert any(
            "nonsense" in record.message and record.levelno == logging.WARNING
            for record in caplog.records
        ), "expected a WARNING mentioning the unknown value"

    def test_empty_and_none_never_raise(self):
        try:
            assert resolve_tour_policy("") == DEFAULT_TOUR_POLICY
            assert resolve_tour_policy(None) == DEFAULT_TOUR_POLICY
        except Exception as exc:  # pragma: no cover - failure path
            pytest.fail(f"resolve_tour_policy raised unexpectedly: {exc}")

    def test_result_is_always_a_valid_policy(self):
        for configured in ("once", "always", "never", "", "bogus", None):
            assert resolve_tour_policy(configured) in TOUR_POLICIES


# ---- API path: startup resolves web.tour / OSPREY_WEB_TOUR from config ----


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


def _make_client(workspace_dir, config: dict):
    """TestClient whose lifespan reads *config* through ``load_osprey_config``.

    Mirrors test_ui_mode._make_client: the lifespan reads the top-level
    sections through ``load_osprey_config`` (the same reader the panel
    loaders use).
    """
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value=config,
        ),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as c:
            yield c


def _tour_payload(workspace_dir, config: dict) -> dict:
    gen = _make_client(workspace_dir, config)
    client = next(gen)
    try:
        return client.get("/api/panels").json()["tour"]
    finally:
        next(gen, None)


class TestPanelsPayloadTour:
    def test_payload_carries_configured_policy(self, workspace_dir):
        for policy in TOUR_POLICIES:
            tour = _tour_payload(workspace_dir, {"web": {"tour": policy}})
            assert tour["policy"] == policy

    def test_missing_key_resolves_to_default(self, workspace_dir):
        tour = _tour_payload(workspace_dir, {"web": {}})
        assert tour["policy"] == DEFAULT_TOUR_POLICY

    def test_unknown_policy_falls_back_to_default(self, workspace_dir):
        tour = _tour_payload(workspace_dir, {"web": {"tour": "nonsense"}})
        assert tour["policy"] == DEFAULT_TOUR_POLICY

    def test_env_override_outranks_config(self, workspace_dir, monkeypatch):
        """OSPREY_WEB_TOUR (the per-user roster path) wins over web.tour."""
        monkeypatch.setenv("OSPREY_WEB_TOUR", "never")
        tour = _tour_payload(workspace_dir, {"web": {"tour": "always"}})
        assert tour["policy"] == "never"

    def test_unknown_env_value_falls_back_to_default(self, workspace_dir, monkeypatch):
        monkeypatch.setenv("OSPREY_WEB_TOUR", "sometimes")
        tour = _tour_payload(workspace_dir, {"web": {"tour": "always"}})
        assert tour["policy"] == DEFAULT_TOUR_POLICY


class TestPanelsPayloadTourCapabilities:
    def test_baseline_capabilities_without_control_system(self, workspace_dir):
        """No control_system, no ARIEL: the core executor lines only."""
        tour = _tour_payload(workspace_dir, {"web": {}})
        assert tour["capabilities"] == ["run Python analysis", "make plots"]

    def test_control_system_adds_no_read_line(self, workspace_dir):
        """A configured connector says nothing about what is behind it.

        ``control_system.type`` is set on a mock deployment too, so the server
        never claims a reading capability; the browser derives that wording
        from the active control target's kind.
        """
        tour = _tour_payload(workspace_dir, {"web": {}, "control_system": {"type": "mock"}})
        assert tour["capabilities"] == ["run Python analysis", "make plots"]

    def test_ariel_panel_adds_the_logbook_line_last(self, workspace_dir):
        tour = _tour_payload(
            workspace_dir,
            {"web": {"panels": {"ariel": True}}, "control_system": {"type": "mock"}},
        )
        assert tour["capabilities"] == [
            "run Python analysis",
            "make plots",
            "search the logbook",
        ]


class TestPanelsPayloadTourLogbook:
    """``tour.logbook`` mirrors ARIEL panel availability as its own fact."""

    def test_logbook_false_without_the_ariel_panel(self, workspace_dir):
        tour = _tour_payload(workspace_dir, {"web": {}})
        assert tour["logbook"] is False

    def test_logbook_true_with_the_ariel_panel(self, workspace_dir):
        tour = _tour_payload(workspace_dir, {"web": {"panels": {"ariel": True}}})
        assert tour["logbook"] is True
