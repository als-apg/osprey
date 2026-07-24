"""Tests for the Lattice Dashboard FastAPI routes.

Exercises the REST surface with a TestClient.  Subprocess-spawning compute
calls are monkeypatched so no worker processes are launched, and figure
endpoints read pre-seeded raw JSON to drive the real figure adapters.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.lattice_dashboard.app import _SSEBroadcaster, create_app
from osprey.interfaces.lattice_dashboard.compute import ComputeManager
from osprey.interfaces.lattice_dashboard.state import LatticeState


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """Workspace root plus a client, with compute launches neutralized."""
    monkeypatch.setattr(ComputeManager, "refresh_fast", lambda self: ["optics"])
    monkeypatch.setattr(ComputeManager, "refresh_verification", lambda self: ["da", "lma"])
    monkeypatch.setattr(ComputeManager, "refresh_one", lambda self, name: None)
    app = create_app(workspace_root=tmp_path)
    return tmp_path, TestClient(app)


def _seed_state_with_families(root):
    state = LatticeState(root / "lattice")
    s = LatticeState._empty_state()
    s["base_lattice"] = "/fake.mat"
    s["families"] = {"QF": {"type": "quadrupole", "param": "K", "value": 1.0}}
    state.save(s)
    return state


class TestHealthAndState:
    def test_health(self, ws):
        _, client = ws
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["service"] == "lattice_dashboard"

    def test_get_state_injects_settings(self, ws):
        _, client = ws
        r = client.get("/api/state")
        assert r.status_code == 200
        assert "settings" in r.json()

    def test_init_error_returns_400(self, ws):
        _, client = ws
        # No monkeypatch of initialize → real pyAT load of a bad path fails
        r = client.post("/api/state/init", json={"lattice_path": "/does/not/exist.mat"})
        assert r.status_code == 400

    def test_init_success(self, ws, monkeypatch):
        root, client = ws
        monkeypatch.setattr(
            LatticeState, "initialize", lambda self, path: {"base_lattice": path, "families": {}}
        )
        r = client.post("/api/state/init", json={"lattice_path": "/fake.mat"})
        assert r.status_code == 200
        assert r.json()["base_lattice"] == "/fake.mat"


class TestParam:
    def test_unknown_family_404(self, ws):
        _, client = ws
        r = client.post("/api/state/param", json={"family": "ZZ", "value": 1.0})
        assert r.status_code == 404

    def test_set_param_success(self, ws):
        root, client = ws
        _seed_state_with_families(root)
        r = client.post("/api/state/param", json={"family": "QF", "value": 2.3})
        assert r.status_code == 200
        assert r.json()["overrides"]["QF"] == 2.3


class TestRefresh:
    def test_refresh_fast(self, ws):
        _, client = ws
        r = client.post("/api/refresh")
        assert r.status_code == 200
        assert r.json()["launched"] == ["optics"]

    def test_refresh_figure_valid(self, ws):
        _, client = ws
        r = client.post("/api/refresh/optics")
        assert r.status_code == 200
        assert r.json()["launched"] == ["optics"]

    def test_refresh_figure_unknown_404(self, ws):
        _, client = ws
        r = client.post("/api/refresh/bogus")
        assert r.status_code == 404

    def test_verify(self, ws):
        _, client = ws
        r = client.post("/api/verify")
        assert r.status_code == 200
        assert r.json()["launched"] == ["da", "lma"]


RAW_FIXTURES = {
    "optics": {
        "s_pos": [0.0, 1.0, 2.0],
        "beta_x": [10.0, 12.0, 11.0],
        "beta_y": [5.0, 6.0, 5.5],
        "eta_x": [0.1, 0.15, 0.12],
        "baseline": None,
    },
    "chromaticity": {
        "dp": [-0.01, 0.0, 0.01],
        "nux": [0.49, 0.48, 0.47],
        "nuy": [0.11, 0.10, 0.09],
        "baseline": None,
    },
    "resonance": {"nux": 0.48, "nuy": 0.10, "baseline_nux": None, "baseline_nuy": None},
    "da": {
        "da_x": [0.01, 0.0, -0.01, 0.0, 0.01],
        "da_y": [0.0, 0.01, 0.0, -0.01, 0.0],
        "area_mm2": 42.0,
        "nturns": 256,
        "baseline": None,
    },
    "lma": {
        "s_pos": [0.0, 1.0, 2.0],
        "dp_plus": [0.03, 0.028, 0.03],
        "dp_minus": [0.025, 0.024, 0.025],
        "lattice_elements": [
            {"s_start": 0.0, "s_end": 0.5, "type": "quadrupole", "name": "QF", "strength": 1.0}
        ],
        "n_sectors": 1,
        "baseline": None,
    },
    "footprint": {
        "nux": [0.25, 0.251],
        "nuy": [0.15, 0.151],
        "amps": [1.0, 2.0],
        "diffusion": [-8.0, -6.0],
        "design_tune": [0.25, 0.15],
        "baseline_tune": None,
        "baseline": None,
        "n_amp": 3,
    },
}


class TestFigures:
    def test_unknown_figure_404(self, ws):
        _, client = ws
        r = client.get("/api/figures/bogus")
        assert r.status_code == 404

    def test_not_yet_computed_404(self, ws):
        _, client = ws
        r = client.get("/api/figures/optics")
        assert r.status_code == 404

    @pytest.mark.parametrize("name", list(RAW_FIXTURES))
    def test_figure_builds_from_raw(self, ws, name):
        root, client = ws
        state = LatticeState(root / "lattice")
        (state.figures_dir / f"{name}.json").write_text(json.dumps(RAW_FIXTURES[name]))

        r = client.get(f"/api/figures/{name}")
        assert r.status_code == 200
        payload = r.json()
        # Adapter → build_figure → figure_to_dict yields a Plotly figure dict
        assert "data" in payload
        assert "layout" in payload

    def test_get_data_returns_raw(self, ws):
        root, client = ws
        state = LatticeState(root / "lattice")
        (state.figures_dir / "optics.json").write_text(json.dumps(RAW_FIXTURES["optics"]))
        r = client.get("/api/data/optics")
        assert r.status_code == 200
        assert r.json()["s_pos"] == [0.0, 1.0, 2.0]

    def test_get_data_unknown_404(self, ws):
        _, client = ws
        assert client.get("/api/data/bogus").status_code == 404

    def test_get_data_not_computed_404(self, ws):
        _, client = ws
        assert client.get("/api/data/optics").status_code == 404


class TestBaselineAndSettings:
    def test_set_and_clear_baseline(self, ws):
        _, client = ws
        assert client.post("/api/baseline").status_code == 200
        assert client.delete("/api/baseline").status_code == 200

    def test_get_settings(self, ws):
        _, client = ws
        r = client.get("/api/settings")
        assert r.status_code == 200
        assert "da" in r.json()

    def test_update_settings(self, ws):
        _, client = ws
        r = client.put("/api/settings", json={"settings": {"da": {"nturns": 1024}}})
        assert r.status_code == 200
        assert r.json()["da"]["nturns"] == 1024

    def test_update_settings_clamps_out_of_range(self, ws):
        _, client = ws
        r = client.put("/api/settings", json={"settings": {"da": {"nturns": 999999}}})
        assert r.status_code == 200
        # Clamped to the validation ceiling (8192)
        assert r.json()["da"]["nturns"] == 8192

    def test_reset_settings(self, ws):
        _, client = ws
        client.put("/api/settings", json={"settings": {"da": {"nturns": 1024}}})
        r = client.delete("/api/settings")
        assert r.status_code == 200
        assert r.json()["settings"]["da"]["nturns"] == 512


class TestSSEBroadcaster:
    """Unit-level coverage of the SSE fan-out helper."""

    def test_subscribe_broadcast_unsubscribe(self):
        b = _SSEBroadcaster()
        q = b.subscribe()
        b.broadcast({"type": "ping"})
        assert q.get_nowait() == {"type": "ping"}
        b.unsubscribe(q)
        b.broadcast({"type": "after"})
        assert q.empty()

    def test_double_unsubscribe_is_safe(self):
        b = _SSEBroadcaster()
        q = b.subscribe()
        b.unsubscribe(q)
        b.unsubscribe(q)  # must not raise

    def test_full_queue_drops_silently(self):
        b = _SSEBroadcaster()
        q = b.subscribe()
        # Fill the queue to capacity; the extra broadcast must be dropped, not raise
        with contextlib.suppress(asyncio.QueueFull):
            while True:
                q.put_nowait({"x": 1})
        b.broadcast({"type": "overflow"})
        assert q.full()
