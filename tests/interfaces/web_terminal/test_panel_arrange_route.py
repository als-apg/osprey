"""Tests for ``POST /api/panel-arrange`` (`routes/panels.py`).

The arrange route is the single server operation behind both the agent's
``arrange_workspace`` verb and the human "Layouts" click. It validates the
requested end state against the live panel inventory, updates rail membership
and the recorded focus, and broadcasts exactly one ``panel_arrange`` frame that
every client applies as a deterministic rebuild of the service-tile region.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.routes.panels import router


def _make_client(**state) -> TestClient:
    """A minimal app exposing the panels router with a stub broadcaster.

    Defaults give three known panels (two built-ins plus a custom one) and an
    empty rail; individual tests override via keyword arguments.
    """
    app = FastAPI()
    app.include_router(router)
    app.state.broadcaster = MagicMock()
    app.state.enabled_panels = {"ariel", "lattice", "artifacts"}
    app.state.custom_panels = [{"id": "grafana", "label": "GRAFANA", "url": "http://10.0.0.5:3000"}]
    app.state.visible_panels = []
    app.state.panel_presets = []
    for key, value in state.items():
        setattr(app.state, key, value)
    return TestClient(app)


def _frame(client: TestClient) -> dict:
    """The single broadcast frame the request produced."""
    broadcaster = client.app.state.broadcaster
    broadcaster.broadcast.assert_called_once()
    return broadcaster.broadcast.call_args[0][0]


class TestTilesPath:
    def test_arranges_requested_tiles_in_order(self):
        client = _make_client()
        resp = client.post("/api/panel-arrange", json={"tiles": ["lattice", "ariel"]})
        assert resp.status_code == 200
        assert resp.json() == {
            "status": "ok",
            "tiles": ["lattice", "ariel"],
            "focus": None,
            "preset": None,
            "prune_rail": False,
        }

    def test_broadcasts_one_arrange_frame(self):
        client = _make_client()
        client.post("/api/panel-arrange", json={"tiles": ["lattice", "ariel"]})
        assert _frame(client) == {"type": "panel_arrange", "tiles": ["lattice", "ariel"]}

    def test_listed_non_members_are_added_to_the_rail(self):
        client = _make_client(visible_panels=["artifacts"])
        client.post("/api/panel-arrange", json={"tiles": ["lattice", "ariel"]})
        assert client.app.state.visible_panels == ["artifacts", "lattice", "ariel"]

    def test_omitted_panels_keep_rail_membership(self):
        """A tiles request never removes a rail entry — only presets prune."""
        client = _make_client(visible_panels=["artifacts", "grafana"])
        client.post("/api/panel-arrange", json={"tiles": ["ariel"]})
        assert client.app.state.visible_panels == ["artifacts", "grafana", "ariel"]

    def test_duplicate_ids_collapse_to_first_occurrence(self):
        client = _make_client()
        resp = client.post("/api/panel-arrange", json={"tiles": ["ariel", "lattice", "ariel"]})
        assert resp.json()["tiles"] == ["ariel", "lattice"]

    def test_custom_panel_ids_are_arrangeable(self):
        client = _make_client()
        resp = client.post("/api/panel-arrange", json={"tiles": ["grafana"]})
        assert resp.status_code == 200
        assert resp.json()["tiles"] == ["grafana"]

    def test_source_agent_is_passed_through(self):
        client = _make_client()
        client.post("/api/panel-arrange", json={"tiles": ["ariel"], "source": "agent"})
        assert _frame(client)["source"] == "agent"

    def test_source_key_omitted_when_not_supplied(self):
        client = _make_client()
        client.post("/api/panel-arrange", json={"tiles": ["ariel"]})
        assert "source" not in _frame(client)

    def test_prune_rail_key_omitted_on_the_tiles_path(self):
        client = _make_client()
        client.post("/api/panel-arrange", json={"tiles": ["ariel"]})
        assert "prune_rail" not in _frame(client)


class TestFocus:
    def test_requested_focus_is_recorded_and_broadcast(self):
        client = _make_client()
        resp = client.post(
            "/api/panel-arrange", json={"tiles": ["lattice", "ariel"], "focus": "ariel"}
        )
        assert resp.json()["focus"] == "ariel"
        assert client.app.state.active_panel == "ariel"
        assert _frame(client)["focus"] == "ariel"

    def test_first_tile_becomes_active_when_no_focus_requested(self):
        """An arrangement always lands focus — it never strands the old one."""
        client = _make_client(active_panel="artifacts")
        client.post("/api/panel-arrange", json={"tiles": ["lattice", "ariel"]})
        assert client.app.state.active_panel == "lattice"

    def test_no_focus_request_still_omits_focus_from_the_broadcast(self):
        """The client's healthy-fallback rule stays in charge of the screen."""
        client = _make_client(active_panel="artifacts")
        client.post("/api/panel-arrange", json={"tiles": ["lattice", "ariel"]})
        assert "focus" not in _frame(client)

    def test_focus_less_response_still_reports_no_requested_focus(self):
        """The response echoes what was *asked*, not the recorded fallback."""
        client = _make_client()
        resp = client.post("/api/panel-arrange", json={"tiles": ["lattice", "ariel"]})
        assert resp.json()["focus"] is None

    def test_focus_outside_the_arranged_tiles_is_rejected(self):
        client = _make_client()
        resp = client.post("/api/panel-arrange", json={"tiles": ["ariel"], "focus": "lattice"})
        assert resp.status_code == 422
        assert "lattice" in resp.json()["detail"]

    def test_unknown_focus_id_is_rejected(self):
        client = _make_client()
        resp = client.post("/api/panel-arrange", json={"tiles": ["ariel"], "focus": "nope"})
        assert resp.status_code == 422


class TestPresetPath:
    def test_preset_resolves_members_server_side(self):
        client = _make_client(panel_presets=[{"name": "Injection", "panels": ["lattice", "ariel"]}])
        resp = client.post("/api/panel-arrange", json={"preset": "Injection"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["tiles"] == ["lattice", "ariel"]
        assert body["preset"] == "Injection"
        assert body["prune_rail"] is True

    def test_preset_broadcast_carries_prune_rail(self):
        client = _make_client(panel_presets=[{"name": "L1", "panels": ["ariel"]}])
        client.post("/api/panel-arrange", json={"preset": "L1"})
        assert _frame(client) == {
            "type": "panel_arrange",
            "tiles": ["ariel"],
            "prune_rail": True,
        }

    def test_preset_prunes_the_rail_to_its_members(self):
        client = _make_client(
            visible_panels=["artifacts", "lattice", "grafana"],
            panel_presets=[{"name": "L1", "panels": ["lattice", "ariel"]}],
        )
        client.post("/api/panel-arrange", json={"preset": "L1"})
        # Surviving members keep their rail order; new members are appended.
        assert client.app.state.visible_panels == ["lattice", "ariel"]

    def test_unknown_members_are_filtered_fail_safe(self):
        """Mirrors ``computePresetDiff``: a typo'd member is skipped, not fatal."""
        client = _make_client(panel_presets=[{"name": "L1", "panels": ["ariel", "typo"]}])
        resp = client.post("/api/panel-arrange", json={"preset": "L1"})
        assert resp.status_code == 200
        assert resp.json()["tiles"] == ["ariel"]

    def test_terminal_member_is_filtered_out(self):
        client = _make_client(panel_presets=[{"name": "L1", "panels": ["terminal", "ariel"]}])
        resp = client.post("/api/panel-arrange", json={"preset": "L1"})
        assert resp.json()["tiles"] == ["ariel"]

    def test_unknown_preset_lists_available_names(self):
        client = _make_client(
            panel_presets=[
                {"name": "Injection", "panels": ["ariel"]},
                {"name": "Tuning", "panels": ["lattice"]},
            ]
        )
        resp = client.post("/api/panel-arrange", json={"preset": "Nope"})
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "Injection" in detail and "Tuning" in detail
        client.app.state.broadcaster.broadcast.assert_not_called()

    def test_preset_with_no_known_member_is_rejected(self):
        """Never strand an empty workspace when every member is unknown."""
        client = _make_client(panel_presets=[{"name": "L1", "panels": ["typo", "gone"]}])
        resp = client.post("/api/panel-arrange", json={"preset": "L1"})
        assert resp.status_code == 422
        assert "ariel" in resp.json()["detail"]  # valid ids are named
        client.app.state.broadcaster.broadcast.assert_not_called()

    def test_focus_may_target_a_resolved_preset_member(self):
        client = _make_client(panel_presets=[{"name": "L1", "panels": ["lattice", "ariel"]}])
        resp = client.post("/api/panel-arrange", json={"preset": "L1", "focus": "ariel"})
        assert resp.status_code == 200
        assert client.app.state.active_panel == "ariel"


class TestActiveStaysConsistentWithVisible:
    """``active`` must never name a panel the same payload calls invisible.

    A human "Layouts" click is a focus-less preset arrange, so before the fix
    a preset could prune the previously-active panel out of the rail while
    ``active_panel`` still pointed at it — a contradiction that reached
    ``list_panels`` and both workspace hooks.
    """

    def test_focus_less_preset_arrange_leaves_active_inside_visible(self):
        client = _make_client(
            visible_panels=["artifacts"],
            active_panel="artifacts",
            panel_presets=[{"name": "Injection", "panels": ["lattice", "ariel"]}],
        )

        resp = client.post("/api/panel-arrange", json={"preset": "Injection"})
        assert resp.status_code == 200

        body = client.get("/api/panels").json()
        assert body["active"] in body["visible"], (
            f"active={body['active']!r} is not in visible={body['visible']!r}"
        )
        assert body["active"] == "lattice"
        assert "artifacts" not in body["visible"]

    def test_surviving_rail_member_does_not_steal_the_recorded_focus(self):
        """Active follows TILE order, not rail order, when the two diverge.

        Pruning keeps a surviving member's rail position, so the rail can lead
        with a panel that is not the arrangement's first tile. The recorded
        focus must still be the first tile, matching what a client's
        healthy-fallback rule picks.
        """
        client = _make_client(
            visible_panels=["artifacts", "ariel"],
            active_panel="artifacts",
            panel_presets=[{"name": "Machine setup", "panels": ["lattice", "ariel"]}],
        )

        client.post("/api/panel-arrange", json={"preset": "Machine setup"})

        body = client.get("/api/panels").json()
        assert body["visible"] == ["ariel", "lattice"]  # survivor keeps its slot
        assert body["active"] == "lattice"  # but focus follows the tile order
        assert body["active"] in body["visible"]

    def test_focus_less_tiles_arrange_leaves_active_inside_visible(self):
        client = _make_client(visible_panels=["artifacts"], active_panel="artifacts")

        client.post("/api/panel-arrange", json={"tiles": ["grafana", "ariel"]})

        body = client.get("/api/panels").json()
        assert body["active"] in body["visible"]
        assert body["active"] == "grafana"

    def test_explicit_focus_still_wins_over_the_first_tile(self):
        client = _make_client(visible_panels=["artifacts"], active_panel="artifacts")

        client.post("/api/panel-arrange", json={"tiles": ["lattice", "ariel"], "focus": "ariel"})

        body = client.get("/api/panels").json()
        assert body["active"] == "ariel"
        assert body["active"] in body["visible"]


class TestValidation:
    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"focus": "ariel"},
            {"tiles": ["ariel"], "preset": "L1"},
        ],
        ids=["neither", "focus-only", "both"],
    )
    def test_exactly_one_of_tiles_or_preset(self, payload):
        client = _make_client(panel_presets=[{"name": "L1", "panels": ["ariel"]}])
        resp = client.post("/api/panel-arrange", json=payload)
        assert resp.status_code == 422
        client.app.state.broadcaster.broadcast.assert_not_called()

    def test_empty_tiles_is_rejected(self):
        client = _make_client()
        resp = client.post("/api/panel-arrange", json={"tiles": []})
        assert resp.status_code == 422
        client.app.state.broadcaster.broadcast.assert_not_called()

    def test_unknown_tile_id_lists_valid_ids(self):
        client = _make_client()
        resp = client.post("/api/panel-arrange", json={"tiles": ["ariel", "nope"]})
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "nope" in detail
        for valid in ("ariel", "lattice", "artifacts", "grafana"):
            assert valid in detail

    def test_terminal_tile_is_rejected(self):
        client = _make_client()
        resp = client.post("/api/panel-arrange", json={"tiles": ["ariel", "terminal"]})
        assert resp.status_code == 422
        assert "terminal" in resp.json()["detail"]

    def test_terminal_is_rejected_even_when_a_custom_panel_squats_the_id(self):
        """The terminal check is explicit, not a side effect of id validation."""
        client = _make_client(
            custom_panels=[{"id": "terminal", "label": "T", "url": "http://10.0.0.5:1"}]
        )
        resp = client.post("/api/panel-arrange", json={"tiles": ["terminal"]})
        assert resp.status_code == 422

    def test_rejected_requests_leave_state_untouched(self):
        client = _make_client(visible_panels=["artifacts"], active_panel="artifacts")
        client.post("/api/panel-arrange", json={"tiles": ["nope"], "focus": "nope"})
        assert client.app.state.visible_panels == ["artifacts"]
        assert client.app.state.active_panel == "artifacts"
        client.app.state.broadcaster.broadcast.assert_not_called()
