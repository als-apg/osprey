"""Tests for ``POST /api/panel-layout`` (`routes/panels.py`).

The layout report is how a browser tells the server which service tiles are
actually on screen. It is report-only: last-writer-wins, never broadcast to
other clients, and content-deduped so an applied arrangement converges instead
of ping-ponging reports (the server half of the convergence contract).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.routes.panels import router

from .conftest import bare_route_app


def _make_client(**state) -> TestClient:
    """A minimal app exposing the panels router with a stub broadcaster."""
    defaults = {
        "enabled_panels": {"ariel", "lattice", "artifacts"},
        "custom_panels": [{"id": "grafana", "label": "GRAFANA", "url": "http://10.0.0.5:3000"}],
    }
    return TestClient(bare_route_app(router, **{**defaults, **state}))


def _report(client: TestClient, tiles: list[str], dock: bool = True):
    return client.post("/api/panel-layout", json={"tiles": tiles, "dock": dock})


class TestStore:
    def test_first_report_is_stored_with_a_timestamp(self):
        client = _make_client()
        resp = _report(client, ["lattice", "ariel"])
        assert resp.status_code == 200
        assert resp.json() == {
            "status": "ok",
            "tiles": ["lattice", "ariel"],
            "dock": True,
            "updated": True,
        }
        assert client.app.state.open_tiles == ["lattice", "ariel"]
        assert client.app.state.open_tiles_dock is True
        assert isinstance(client.app.state.open_tiles_ts, float)

    def test_never_broadcasts(self):
        client = _make_client()
        _report(client, ["ariel"])
        _report(client, ["lattice"])
        _report(client, [])
        client.app.state.broadcaster.broadcast.assert_not_called()


class TestDockLessReportsUnknownOccupancy:
    """A client without a dock shell reports presence, not an empty screen.

    Its ``{"tiles": [], "dock": false}`` is a capability signal — it cannot see
    tile order at all. Recording that ``[]`` verbatim would let it clobber a
    dock client's live list with a confident "nothing is open", so it is stored
    as unknown occupancy instead.
    """

    def test_dock_false_records_unknown_not_empty(self):
        client = _make_client()
        _report(client, [], dock=False)
        assert client.app.state.open_tiles is None
        assert client.app.state.open_tiles_dock is False
        # Someone IS watching — that fact is fresh news even without order.
        assert isinstance(client.app.state.open_tiles_ts, float)

    def test_dock_false_response_echoes_the_recorded_unknown(self):
        client = _make_client()
        body = _report(client, [], dock=False).json()
        assert body["tiles"] is None
        assert body["dock"] is False
        assert body["updated"] is True


class TestDedupe:
    def test_identical_report_is_a_no_op(self):
        client = _make_client()
        _report(client, ["ariel", "lattice"])
        first_ts = client.app.state.open_tiles_ts

        resp = _report(client, ["ariel", "lattice"])

        assert resp.status_code == 200
        assert resp.json() == {
            "status": "ok",
            "tiles": ["ariel", "lattice"],
            "dock": True,
            "updated": False,
        }
        assert client.app.state.open_tiles_ts == first_ts
        assert client.app.state.open_tiles == ["ariel", "lattice"]

    def test_reordered_tiles_are_a_different_report(self):
        client = _make_client()
        _report(client, ["ariel", "lattice"])
        first_ts = client.app.state.open_tiles_ts

        resp = _report(client, ["lattice", "ariel"])

        assert resp.json()["updated"] is True
        assert client.app.state.open_tiles == ["lattice", "ariel"]
        assert client.app.state.open_tiles_ts >= first_ts

    def test_changed_dock_flag_is_a_different_report(self):
        client = _make_client()
        _report(client, ["ariel"], dock=True)
        first_ts = client.app.state.open_tiles_ts

        resp = _report(client, ["ariel"], dock=False)

        assert resp.json()["updated"] is True
        assert client.app.state.open_tiles_dock is False
        assert client.app.state.open_tiles is None
        assert client.app.state.open_tiles_ts >= first_ts

    def test_successive_dock_less_reports_dedupe(self):
        """Both record the same unknown occupancy, so the second is a no-op."""
        client = _make_client()
        _report(client, [], dock=False)
        first_ts = client.app.state.open_tiles_ts

        resp = _report(client, [], dock=False)

        assert resp.json()["updated"] is False
        assert client.app.state.open_tiles_ts == first_ts

    def test_known_empty_and_unknown_are_not_the_same_report(self):
        """[] from a dock client must not dedupe against a dock-less unknown."""
        client = _make_client()
        _report(client, [], dock=False)

        resp = _report(client, [], dock=True)

        assert resp.json()["updated"] is True
        assert client.app.state.open_tiles == []

    def test_first_empty_report_is_not_deduped_against_the_never_reported_state(self):
        """``dock`` starts unset, so even an empty first report is real news."""
        client = _make_client()
        resp = _report(client, [], dock=True)
        assert resp.json()["updated"] is True
        assert client.app.state.open_tiles_ts is not None


class TestValidation:
    def test_unknown_id_is_rejected_and_lists_valid_ids(self):
        client = _make_client()
        resp = _report(client, ["ariel", "nope"])
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "nope" in detail
        for valid in ("ariel", "lattice", "artifacts", "grafana"):
            assert valid in detail

    def test_terminal_is_rejected_even_when_a_custom_panel_squats_the_id(self):
        client = _make_client(
            custom_panels=[{"id": "terminal", "label": "T", "url": "http://10.0.0.5:1"}]
        )
        assert _report(client, ["terminal"]).status_code == 422

    def test_rejected_report_leaves_stored_state_untouched(self):
        client = _make_client()
        _report(client, ["ariel"])
        stored_ts = client.app.state.open_tiles_ts

        assert _report(client, ["ariel", "nope"]).status_code == 422

        assert client.app.state.open_tiles == ["ariel"]
        assert client.app.state.open_tiles_ts == stored_ts

    @pytest.mark.parametrize(
        "body",
        [pytest.param({"tiles": []}, id="no-dock"), pytest.param({"dock": True}, id="no-tiles")],
    )
    def test_both_fields_are_required(self, body):
        """No default for ``dock``: a client that omits it must not be read as a dock."""
        client = _make_client()
        assert client.post("/api/panel-layout", json=body).status_code == 422
