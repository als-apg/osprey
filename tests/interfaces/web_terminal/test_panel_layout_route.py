"""Tests for ``POST /api/panel-layout`` (`routes/panels.py`).

The layout report is how a browser tells the server which service tiles are
actually on screen. It is report-only: last-writer-wins, never broadcast to
other clients, and content-deduped so an applied arrangement converges instead
of ping-ponging reports (the server half of the convergence contract).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.routes.panels import router


def _make_client(**state) -> TestClient:
    """A minimal app exposing the panels router with a stub broadcaster."""
    app = FastAPI()
    app.include_router(router)
    app.state.broadcaster = MagicMock()
    app.state.enabled_panels = {"ariel", "lattice", "artifacts"}
    app.state.custom_panels = [{"id": "grafana", "label": "GRAFANA", "url": "http://10.0.0.5:3000"}]
    for key, value in state.items():
        setattr(app.state, key, value)
    return TestClient(app)


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

    def test_reported_order_is_preserved(self):
        """The list is a spatial reading order, not a set — order is the payload."""
        client = _make_client()
        _report(client, ["ariel", "grafana", "lattice"])
        assert client.app.state.open_tiles == ["ariel", "grafana", "lattice"]

    def test_empty_report_is_valid(self):
        """Every service tile closed is a real state, not an error."""
        client = _make_client()
        resp = _report(client, [])
        assert resp.status_code == 200
        assert client.app.state.open_tiles == []
        assert isinstance(client.app.state.open_tiles_ts, float)

    def test_last_writer_wins(self):
        client = _make_client()
        _report(client, ["ariel"])
        _report(client, ["lattice", "grafana"])
        assert client.app.state.open_tiles == ["lattice", "grafana"]

    def test_stored_list_is_a_copy_of_the_request(self):
        """Mutating the response list must not reach into app.state."""
        client = _make_client()
        resp = _report(client, ["ariel"])
        resp.json()["tiles"].append("lattice")
        assert client.app.state.open_tiles == ["ariel"]

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

    def test_dock_false_still_stamps_a_timestamp(self):
        """Someone IS watching — that fact is fresh news even without order."""
        client = _make_client()
        _report(client, [], dock=False)
        assert isinstance(client.app.state.open_tiles_ts, float)

    def test_dock_false_response_echoes_the_recorded_unknown(self):
        client = _make_client()
        body = _report(client, [], dock=False).json()
        assert body["tiles"] is None
        assert body["dock"] is False
        assert body["updated"] is True

    def test_dock_false_does_not_clobber_a_dock_list_with_a_fake_empty(self):
        """The regression this exists for: the list is dropped, not falsified."""
        client = _make_client()
        _report(client, ["ariel", "lattice"], dock=True)

        _report(client, [], dock=False)

        assert client.app.state.open_tiles is None  # unknown, never []

    def test_dock_true_report_restores_known_occupancy(self):
        client = _make_client()
        _report(client, [], dock=False)
        _report(client, ["ariel"], dock=True)
        assert client.app.state.open_tiles == ["ariel"]
        assert client.app.state.open_tiles_dock is True

    def test_dock_false_ignores_any_tiles_it_sends(self):
        """Occupancy is unknown regardless of payload — the flag decides."""
        client = _make_client()
        _report(client, ["ariel"], dock=False)
        assert client.app.state.open_tiles is None

    def test_dock_true_empty_is_still_known_empty(self):
        """The dock:true path is untouched: [] keeps meaning known-empty."""
        client = _make_client()
        _report(client, [], dock=True)
        assert client.app.state.open_tiles == []


class TestDedupe:
    def test_identical_report_is_a_no_op(self):
        client = _make_client()
        _report(client, ["ariel", "lattice"])
        first_ts = client.app.state.open_tiles_ts

        resp = _report(client, ["ariel", "lattice"])

        assert resp.status_code == 200
        assert resp.json()["updated"] is False
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

    def test_deduped_report_echoes_the_stored_state(self):
        client = _make_client()
        _report(client, ["ariel"])
        body = _report(client, ["ariel"]).json()
        assert body["tiles"] == ["ariel"]
        assert body["dock"] is True


class TestValidation:
    def test_unknown_id_is_rejected_and_lists_valid_ids(self):
        client = _make_client()
        resp = _report(client, ["ariel", "nope"])
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "nope" in detail
        for valid in ("ariel", "lattice", "artifacts", "grafana"):
            assert valid in detail

    def test_terminal_tile_is_rejected(self):
        client = _make_client()
        resp = _report(client, ["terminal", "ariel"])
        assert resp.status_code == 422
        assert "terminal" in resp.json()["detail"]

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

    def test_dock_is_required(self):
        client = _make_client()
        assert client.post("/api/panel-layout", json={"tiles": []}).status_code == 422

    def test_tiles_is_required(self):
        client = _make_client()
        assert client.post("/api/panel-layout", json={"dock": True}).status_code == 422
