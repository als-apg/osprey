"""Tests for ``POST /api/panel-layout`` and the layout half of ``GET /api/panels``.

The layout report is how a browser's dock tells the server what the operator's
workspace actually looks like: the dock layout tree (its tiles, the panel each
one shows, the terminal) plus the page's ``client_id``. The server validates it
with ``dock_layout`` and OSPREY's one-panel-per-tile rule, records the reporting
client as the current one and mints a ``rev`` that the ack returns. It is
report-only: last-writer-wins, never broadcast, and content-deduped so an
applied arrangement converges instead of ping-ponging.
"""

from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.routes.panels import router

from .conftest import bare_route_app

LAYOUT = {
    "split": "row",
    "items": [
        {"tile": "1", "tabs": ["terminal"], "showing": "terminal", "size": 0.4},
        {"tile": "2", "tabs": ["ariel"], "showing": "ariel", "size": 0.6},
    ],
}
#: LAYOUT with LATTICE in tile 2, in place of ARIEL.
REPLACED = {
    "split": "row",
    "items": [
        {"tile": "1", "tabs": ["terminal"], "showing": "terminal", "size": 0.4},
        {"tile": "2", "tabs": ["lattice"], "showing": "lattice", "size": 0.6},
    ],
}
TERMINAL_ONLY = {"tile": "1", "tabs": ["terminal"], "showing": "terminal"}


def _make_client(**state) -> TestClient:
    """A minimal app exposing the panels router with a stub broadcaster."""
    defaults = {
        "enabled_panels": {"ariel", "lattice", "artifacts"},
        "custom_panels": [{"id": "grafana", "label": "GRAFANA", "url": "http://10.0.0.5:3000"}],
    }
    return TestClient(bare_route_app(router, **{**defaults, **state}))


def _report(client: TestClient, layout, dock: bool = True, client_id: str = "page-a"):
    return client.post(
        "/api/panel-layout", json={"layout": layout, "dock": dock, "client_id": client_id}
    )


def _panels(client: TestClient) -> dict:
    return client.get("/api/panels").json()


class TestStore:
    def test_first_report_is_stored_with_the_client_and_a_minted_rev(self):
        client = _make_client()
        resp = _report(client, LAYOUT)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["updated"] is True
        assert body["dock"] is True
        assert body["layout"] == LAYOUT
        assert isinstance(body["rev"], int) and body["rev"] >= 1
        state = client.app.state
        assert state.dock_layout == LAYOUT
        assert state.dock_layout_rev == body["rev"]
        assert state.dock_client_id == "page-a"
        assert isinstance(state.layout_ts, float)

    def test_the_terminal_key_is_accepted(self):
        client = _make_client()
        assert _report(client, TERMINAL_ONLY).status_code == 200
        assert client.app.state.dock_layout == TERMINAL_ONLY

    def test_a_changed_layout_mints_a_higher_rev(self):
        client = _make_client()
        first = _report(client, LAYOUT).json()["rev"]
        second = _report(client, REPLACED).json()
        assert second["updated"] is True
        assert second["rev"] > first

    def test_the_last_reporting_client_becomes_current(self):
        client = _make_client()
        _report(client, LAYOUT, client_id="page-a")
        _report(client, TERMINAL_ONLY, client_id="page-b")
        assert client.app.state.dock_client_id == "page-b"
        assert client.app.state.dock_layout == TERMINAL_ONLY

    def test_the_stored_layout_is_normalised(self):
        client = _make_client()
        sizeless = {
            "split": "row",
            "items": [
                {"tile": "1", "tabs": ["terminal"], "showing": "terminal"},
                {"tile": "2", "tabs": ["ariel"], "showing": "ariel"},
            ],
        }
        _report(client, sizeless)
        sizes = [item["size"] for item in client.app.state.dock_layout["items"]]
        assert sizes == pytest.approx([0.5, 0.5])

    def test_never_broadcasts(self):
        client = _make_client()
        _report(client, LAYOUT)
        _report(client, TERMINAL_ONLY)
        _report(client, None, dock=False)
        client.app.state.broadcaster.broadcast.assert_not_called()


class TestDockLessReportsUnknownLayout:
    """A client without a dock shell reports presence, not an empty screen."""

    def test_dock_false_records_unknown_and_is_never_current(self):
        client = _make_client()
        body = _report(client, None, dock=False).json()
        assert body["updated"] is True
        assert body["layout"] is None
        assert body["dock"] is False
        assert client.app.state.dock_layout is None
        assert client.app.state.dock_client_id is None
        assert client.app.state.layout_dock is False
        assert isinstance(client.app.state.layout_ts, float)

    def test_dock_false_ignores_any_layout_it_carries(self):
        client = _make_client()
        assert _report(client, LAYOUT, dock=False).json()["layout"] is None

    def test_dock_true_with_a_null_layout_is_an_empty_dock(self):
        client = _make_client()
        resp = _report(client, None, dock=True)
        assert resp.status_code == 200
        assert resp.json()["layout"] is None
        assert client.app.state.layout_dock is True
        assert client.app.state.dock_client_id == "page-a"


class TestDedupe:
    def test_identical_report_is_a_no_op_that_returns_the_same_rev(self):
        client = _make_client()
        first = _report(client, LAYOUT).json()
        first_ts = client.app.state.layout_ts

        again = _report(client, copy.deepcopy(LAYOUT)).json()

        assert again["updated"] is False
        assert again["rev"] == first["rev"]
        assert client.app.state.layout_ts == first_ts

    def test_the_same_layout_from_another_client_moves_current_without_a_rev(self):
        client = _make_client()
        first = _report(client, LAYOUT, client_id="page-a").json()
        again = _report(client, LAYOUT, client_id="page-b").json()
        assert again["updated"] is False
        assert again["rev"] == first["rev"]
        assert client.app.state.dock_client_id == "page-b"

    def test_successive_dock_less_reports_dedupe(self):
        client = _make_client()
        _report(client, None, dock=False)
        assert _report(client, None, dock=False).json()["updated"] is False

    def test_sizes_within_the_engine_tolerance_are_the_same_report(self):
        """The dock's read-back of a tree it just applied differs by rounding only.

        The browser dedupes such a report (``sameReport``); the server must agree,
        or the read-back mints a revision between an agent's two steps and the
        second is refused ``stale_base`` for a workspace nobody changed.
        """
        client = _make_client()
        first = _report(client, LAYOUT).json()
        drifted = copy.deepcopy(LAYOUT)
        drifted["items"][0]["size"] = 0.4004
        drifted["items"][1]["size"] = 0.5996

        again = _report(client, drifted).json()

        assert again["updated"] is False
        assert again["rev"] == first["rev"]
        assert _panels(client)["layout"]["items"][0]["size"] == 0.4

    def test_a_size_change_past_the_tolerance_mints_a_rev(self):
        client = _make_client()
        first = _report(client, LAYOUT).json()
        resized = copy.deepcopy(LAYOUT)
        resized["items"][0]["size"] = 0.45
        resized["items"][1]["size"] = 0.55

        again = _report(client, resized).json()

        assert again["updated"] is True
        assert again["rev"] == first["rev"] + 1

    def test_another_panel_in_a_tile_mints_a_rev(self):
        client = _make_client()
        first = _report(client, LAYOUT).json()

        assert _report(client, REPLACED).json()["rev"] == first["rev"] + 1

    def test_a_dock_report_after_a_dock_less_one_is_news(self):
        client = _make_client()
        _report(client, None, dock=False)
        assert _report(client, TERMINAL_ONLY).json()["updated"] is True


class TestValidation:
    def test_unknown_key_is_rejected_and_lists_valid_ids(self):
        client = _make_client()
        bad = copy.deepcopy(LAYOUT)
        bad["items"][1].update(tabs=["nope"], showing="nope")
        resp = _report(client, bad)
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "nope" in detail
        for valid in ("ariel", "lattice", "artifacts", "grafana"):
            assert valid in detail

    def test_a_tile_holding_two_panels_is_refused_one_panel_per_tile(self):
        """A valid tree by the engine's grammar, but OSPREY's dock shows one panel per tile."""
        client = _make_client()
        rev = _report(client, LAYOUT).json()["rev"]
        crowded = copy.deepcopy(LAYOUT)
        crowded["items"][1].update(tabs=["lattice", "ariel"], showing="ariel")

        resp = _report(client, crowded)

        assert resp.status_code == 422
        assert resp.json()["detail"] == (
            "layout.items[1].tabs: one_panel_per_tile: a tile holds one panel; to show "
            "several, put them side by side in tiles of their own"
        )
        assert client.app.state.dock_layout == LAYOUT
        assert client.app.state.dock_layout_rev == rev

    def test_a_c_layout_refusal_names_the_reason(self):
        client = _make_client()
        bad = copy.deepcopy(LAYOUT)
        bad["items"][1]["showing"] = "artifacts"
        resp = _report(client, bad)
        assert resp.status_code == 422
        assert "showing_not_in_tabs" in resp.json()["detail"]

    def test_the_old_flat_tiles_body_is_refused(self):
        client = _make_client()
        resp = client.post("/api/panel-layout", json={"tiles": ["ariel"], "dock": True})
        assert resp.status_code == 422

    def test_rejected_report_leaves_stored_state_untouched(self):
        client = _make_client()
        rev = _report(client, LAYOUT).json()["rev"]
        stored_ts = client.app.state.layout_ts
        bad = copy.deepcopy(LAYOUT)
        bad["items"][0]["tabs"] = []
        assert _report(client, bad).status_code == 422
        assert client.app.state.dock_layout == LAYOUT
        assert client.app.state.dock_layout_rev == rev
        assert client.app.state.layout_ts == stored_ts

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param({"layout": TERMINAL_ONLY, "client_id": "a"}, id="no-dock"),
            pytest.param({"layout": TERMINAL_ONLY, "dock": True}, id="no-client"),
            pytest.param(
                {"layout": TERMINAL_ONLY, "dock": True, "client_id": ""}, id="empty-client"
            ),
        ],
    )
    def test_dock_and_client_id_are_required(self, body):
        client = _make_client()
        assert client.post("/api/panel-layout", json=body).status_code == 422


def _report_popouts(client: TestClient, layout, popouts: list[str]):
    return client.post(
        "/api/panel-layout",
        json={"layout": layout, "dock": True, "client_id": "page-a", "popouts": popouts},
    )


class TestPopouts:
    """A pane in a window of its own: any service panel, and the Agent pane."""

    def test_a_detached_service_panel_is_recorded_beside_the_layout(self):
        client = _make_client()
        body = _report_popouts(client, TERMINAL_ONLY, ["ariel"]).json()
        assert body["popouts"] == ["ariel"]
        assert client.app.state.dock_popouts == ["ariel"]

    def test_the_agent_pane_may_be_detached(self):
        client = _make_client()
        only_ariel = {"tile": "2", "tabs": ["ariel"], "showing": "ariel"}
        resp = _report_popouts(client, only_ariel, ["terminal"])
        assert resp.status_code == 200
        assert resp.json()["popouts"] == ["terminal"]

    @pytest.mark.parametrize("key", ["ariel", "terminal"])
    def test_a_pane_both_docked_and_detached_is_refused(self, key):
        client = _make_client()
        resp = _report_popouts(client, LAYOUT, [key])
        assert resp.status_code == 422
        assert "cannot also be in the layout" in resp.json()["detail"]

    def test_an_unknown_pop_out_is_refused(self):
        client = _make_client()
        assert _report_popouts(client, TERMINAL_ONLY, ["nope"]).status_code == 422

    def test_a_null_layout_with_the_agent_detached_is_an_empty_dock(self):
        """Detaching the Agent pane from the default layout leaves the dock empty."""
        client = _make_client()
        resp = _report_popouts(client, None, ["terminal"])
        assert resp.status_code == 200
        body = resp.json()
        assert body["layout"] is None
        assert body["popouts"] == ["terminal"]
        assert body["dock"] is True
        assert client.app.state.dock_client_id == "page-a"
        panels = _panels(client)
        assert panels["layout"] is None
        assert panels["layout_dock"] is True
        assert panels["popouts"] == ["terminal"]

    def test_the_last_detached_pane_going_clears_the_stored_pop_outs(self):
        """Its window closed over an empty dock: the server must stop naming it."""
        client = _make_client()
        rev = _report_popouts(client, None, ["terminal"]).json()["rev"]
        resp = _report_popouts(client, None, [])
        assert resp.status_code == 200
        assert resp.json()["rev"] == rev + 1
        panels = _panels(client)
        assert panels["layout"] is None
        assert panels["popouts"] == []
        assert panels["layout_dock"] is True


class TestGetPanels:
    def test_never_reported_is_null_layout_age_and_dock(self):
        body = _panels(_make_client())
        assert body["layout"] is None
        assert body["rev"] == 0
        assert body["layout_age_s"] is None
        assert body["layout_dock"] is None

    def test_layout_and_rev_reflect_the_last_report(self):
        client = _make_client()
        rev = _report(client, LAYOUT).json()["rev"]
        body = _panels(client)
        assert body["layout"] == LAYOUT
        assert body["rev"] == rev

    def test_a_dock_report_is_known_layout_with_a_fresh_age(self):
        client = _make_client()
        _report(client, LAYOUT)
        body = _panels(client)
        assert body["layout_dock"] is True
        assert 0 <= body["layout_age_s"] < 60

    def test_a_terminal_only_dock_is_a_known_layout_not_null(self):
        client = _make_client()
        _report(client, TERMINAL_ONLY)
        body = _panels(client)
        assert body["layout"] == TERMINAL_ONLY
        assert body["layout_dock"] is True

    def test_a_dock_less_report_is_null_layout_with_an_age(self):
        client = _make_client()
        _report(client, None, dock=False)
        body = _panels(client)
        assert body["layout"] is None
        assert body["layout_dock"] is False
        assert body["layout_age_s"] is not None
