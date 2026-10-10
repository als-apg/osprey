"""Tests for the Lattice Dashboard FastAPI routes.

Exercises the REST surface with a TestClient over a synthetic render whose SR
model is selected at startup. Compute launches are monkeypatched so no worker
process runs, and figure endpoints read raw JSON seeded under the current key
to drive the real figure adapters.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from fastapi.testclient import TestClient
from tests.interfaces.lattice_dashboard.test_app import _write_render, settle

from osprey.interfaces.lattice_dashboard.app import _SSEBroadcaster, create_app
from osprey.interfaces.lattice_dashboard.compute import ComputeManager
from osprey.interfaces.lattice_dashboard.workers._base import save_data
from osprey_connectors.process import ExitCause


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """Workspace root plus a client over a render serving SR, with compute launches neutralized."""
    monkeypatch.setattr(ComputeManager, "refresh_fast", lambda self: ["optics"])
    monkeypatch.setattr(ComputeManager, "refresh_verification", lambda self: ["da", "lma"])
    monkeypatch.setattr(ComputeManager, "refresh_one", lambda self, name: True)
    render = _write_render(tmp_path / "render", served=["SR"], models={"SR": {"solve": "periodic"}})
    with TestClient(create_app(workspace_root=tmp_path, render_root=render)) as client:
        settle(client)
        yield tmp_path, client


def _seed(client, name, raw):
    """Store *raw* as figure *name* under the key of the inputs on screen."""
    state = client.app.state.lattice
    key = state.figure_key(name)
    job = {"key": key, "job_id": 0, "deck_sha256": state.selection.deck_sha256}
    save_data(job, raw, state.figure_path(name, key))
    return key


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


class TestParam:
    def test_unknown_family_404(self, ws):
        _, client = ws
        r = client.post("/api/state/param", json={"family": "ZZ", "value": 1.0})
        assert r.status_code == 404

    def test_set_param_success(self, ws):
        _, client = ws
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
        assert r.json()["status"] == "not_computed"

    @pytest.mark.parametrize("name", list(RAW_FIXTURES))
    def test_figure_builds_from_raw(self, ws, name):
        _, client = ws
        key = _seed(client, name, RAW_FIXTURES[name])

        r = client.get(f"/api/figures/{name}")
        assert r.status_code == 200
        payload = r.json()
        assert payload["status"] == "ready"
        assert payload["key"] == key
        # Adapter → build_figure → figure_to_dict yields a Plotly figure dict
        assert "data" in payload["figure"]
        assert "layout" in payload["figure"]

    def test_get_data_returns_raw(self, ws):
        _, client = ws
        key = _seed(client, "optics", RAW_FIXTURES["optics"])
        r = client.get("/api/data/optics")
        assert r.status_code == 200
        assert r.json()["key"] == key
        assert r.json()["data"]["s_pos"] == [0.0, 1.0, 2.0]

    def test_get_data_unknown_404(self, ws):
        _, client = ws
        assert client.get("/api/data/bogus").status_code == 404

    def test_get_data_not_computed_404(self, ws):
        _, client = ws
        assert client.get("/api/data/optics").status_code == 404


class TestSummaryFreshness:
    """The stat chips' numbers follow the magnet overrides.

    The optics worker recomputes them on the ring it actually tracked, and
    the state's summary reads them from the optics figure of the current key.
    """

    def test_state_summary_reflects_worker_recompute(self, ws, fake_slots):
        _, client = ws
        state = client.app.state.lattice
        _seed(
            client, "optics", {**RAW_FIXTURES["optics"], "summary_updates": {"tunes": [0.3, 0.2]}}
        )
        assert client.get("/api/state").json()["summary"]["tunes"] == [0.3, 0.2]

        client.post("/api/state/param", json={"family": "QF", "value": 2.3})
        assert "tunes" not in client.get("/api/state").json()["summary"]

        # The optics worker finishes on the override-applied ring
        async def recompute():
            manager = ComputeManager(state, _SSEBroadcaster(), fake_slots)
            manager._launch("optics")
            job = fake_slots.current("optics")
            spec = state.job_spec("optics")
            save_data(
                spec,
                {
                    **RAW_FIXTURES["optics"],
                    "summary_updates": {
                        "tunes": [0.4412, 0.3107],
                        "chromaticity": [-2.4, -1.8],
                        "beta_max": [14.0, 9.0],
                    },
                },
                state.figure_path("optics", spec["key"]),
            )
            job.finish(ExitCause.COMPLETED)
            for _ in range(5):
                await asyncio.sleep(0)
            return manager.figure_status("optics")

        assert client.portal.call(recompute)["status"] == "ready"

        summary = client.get("/api/state").json()["summary"]
        assert summary["tunes"] == [0.4412, 0.3107]
        assert summary["chromaticity"] == [-2.4, -1.8]
        assert summary["beta_max"] == [14.0, 9.0]
        assert summary["energy_gev"] == pytest.approx(2.0)

    def test_extra_key_does_not_disturb_the_figure(self, ws):
        """The figure adapter ignores the summary block the summary reads."""
        _, client = ws
        _seed(client, "optics", {**RAW_FIXTURES["optics"], "summary_updates": {"tunes": [0.44]}})

        r = client.get("/api/figures/optics")
        assert r.status_code == 200
        assert "data" in r.json()["figure"]


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
