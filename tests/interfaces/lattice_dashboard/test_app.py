"""The lattice dashboard app over a render's simulator view.

Each app is built over a synthetic render under ``tmp_path``: ``facility.json``
with its ``models``, and ``data/simulator/`` holding ``served_models.json`` and
a pyAT JSON deck per deck-bearing model. Worker launches go to a recording
``Popen`` stand-in, so no subprocess runs.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.lattice_dashboard import app as app_mod
from osprey.interfaces.lattice_dashboard import compute as compute_mod
from osprey.interfaces.lattice_dashboard.app import create_app
from osprey.interfaces.lattice_dashboard.catalog import NO_SERVED_MODEL_TEXT, NO_VIEW_TEXT
from osprey.interfaces.lattice_dashboard.state import (
    ALL_FIGURES,
    FAST_FIGURES,
    SINGLE_PASS_UNAVAILABLE,
)

at = pytest.importorskip("at")

TWISS_IN = {"beta": [8.0, 4.0], "alpha": [0.0, 0.0]}


def _deck(kf: float = 1.0) -> at.Lattice:
    d = at.Drift("DR", 0.5)
    qf = at.Quadrupole("QF", 0.2, kf)
    qd = at.Quadrupole("QD", 0.2, -kf)
    b = at.Dipole("BM", 0.5, np.pi / 8)
    return at.Lattice([qf, d, b, d, qd, d, b, d] * 2, name="FODO", energy=2e9)


def _write_render(root, *, served, models):
    """Write a render: ``models`` is ``{name: settings.pyat or None}`` (None = no deck)."""
    decks = root / "data" / "simulator" / "decks"
    decks.mkdir(parents=True)
    records = [{"name": "texture", "engine": "texture"}]
    for name, pyat in models.items():
        record = {"name": name, "engine": "pyat"}
        if pyat is not None:
            record["deck"] = f"decks/{name}.json"
            record["settings"] = {"pyat": pyat}
            _deck().save(str(decks / f"{name}.json"))
        records.append(record)
    (root / "facility.json").write_text(json.dumps({"models": records}))
    (root / "data" / "simulator" / "served_models.json").write_text(
        json.dumps({"schema": "osprey.facility.served_models/1", "models": served})
    )
    return root


class _RecordingPopen:
    """A worker process that never runs; each launch records its module."""

    launched: list[str] = []

    def __init__(self, cmd, stdout=None, stderr=None):  # noqa: ARG002
        _RecordingPopen.launched.append(cmd[2].rsplit(".", 1)[-1])
        self.pid = 0
        self.returncode = None

    def poll(self):
        return 0

    def communicate(self, timeout=None):  # noqa: ARG002
        return (b"", b"")


@pytest.fixture(autouse=True)
def launched(monkeypatch):
    """The figure workers launched, by name, with no subprocess or monitor thread."""
    _RecordingPopen.launched = []
    monkeypatch.setattr(compute_mod.subprocess, "Popen", _RecordingPopen)
    monkeypatch.setattr(compute_mod.ComputeManager, "_monitor_worker", lambda *a, **k: None)
    return _RecordingPopen.launched


@pytest.fixture
def render(tmp_path):
    """SR (periodic) and TRANSFER (single_pass) served, BOOSTER unserved, SPARE deckless."""
    return _write_render(
        tmp_path / "render",
        served=["SR", "TRANSFER", "texture"],
        models={
            "BOOSTER": {"solve": "periodic"},
            "SPARE": None,
            "SR": {"solve": "periodic"},
            "TRANSFER": {"solve": "single_pass", "twiss_in": TWISS_IN},
        },
    )


@pytest.fixture
def client(tmp_path, render):
    return TestClient(create_app(workspace_root=tmp_path / "ws", render_root=render))


class TestLaunch:
    def test_launcher_starts_without_a_section_and_no_worker(
        self, tmp_path, render, monkeypatch, launched
    ):
        """The launcher's own checker and factory, over a config with no section."""
        from osprey.infrastructure import server_launcher
        from osprey.registry.web import FRAMEWORK_WEB_SERVERS

        definition = FRAMEWORK_WEB_SERVERS["lattice_dashboard"]
        monkeypatch.setattr(server_launcher, "load_osprey_config", lambda: {})
        monkeypatch.setattr(app_mod, "default_config_path", lambda: str(render / "config.yml"))

        assert server_launcher._make_auto_launch_checker(definition)() is True
        app = server_launcher._make_app_factory(definition)(workspace_root=tmp_path / "ws")
        response = TestClient(app).get("/health")

        assert response.status_code == 200
        assert response.json()["service"] == "lattice_dashboard"
        assert launched == []
        assert not (tmp_path / "ws" / "lattice" / "state.json").exists()


class TestNoView:
    def test_no_config_loaded_gives_the_no_view_state(self, tmp_path, monkeypatch, launched):
        monkeypatch.setattr(app_mod, "default_config_path", lambda: None)
        client = TestClient(create_app(workspace_root=tmp_path))

        state = client.get("/api/state").json()

        assert state["notice"] == NO_VIEW_TEXT
        assert state["model"] is None
        assert client.get("/api/models").json() == []
        assert launched == []

    def test_render_without_the_view(self, tmp_path, launched):
        client = TestClient(create_app(workspace_root=tmp_path / "ws", render_root=tmp_path))

        assert client.get("/api/state").json()["notice"] == NO_VIEW_TEXT
        assert launched == []

    def test_texture_only_render(self, tmp_path, launched):
        render = _write_render(tmp_path / "render", served=["texture"], models={})
        client = TestClient(create_app(workspace_root=tmp_path / "ws", render_root=render))

        state = client.get("/api/state").json()

        assert state["notice"] == NO_SERVED_MODEL_TEXT
        assert state["model"] is None
        assert launched == []


class TestModels:
    def test_models_read_from_the_render(self, client):
        models = client.get("/api/models").json()

        assert models == [
            {"name": "SR", "served": True, "solve": "periodic", "selected": True},
            {"name": "TRANSFER", "served": True, "solve": "single_pass", "selected": False},
            {"name": "BOOSTER", "served": False, "solve": "periodic", "selected": False},
        ]

    def test_first_request_loads_the_first_served_deck(self, client, render, launched):
        state = client.get("/api/state").json()

        assert state["model"] == "SR"
        assert state["base_lattice"] == str(render / "data/simulator/decks/SR.json")
        assert state["notice"] is None
        assert state["fast_figures"] == list(FAST_FIGURES)
        assert "tunes" in state["summary"]
        assert launched == list(FAST_FIGURES)

    def test_second_request_does_not_reload(self, client, launched):
        client.get("/api/state")
        client.post("/api/state/param", json={"family": "QF", "value": 1.05})
        launched.clear()

        state = client.get("/api/state").json()

        assert state["overrides"] == {"QF": 1.05}
        assert launched == []

    def test_changed_deck_reloads(self, client, render, launched):
        client.get("/api/state")
        client.post("/api/state/param", json={"family": "QF", "value": 1.05})
        _deck(kf=1.1).save(str(render / "data/simulator/decks/SR.json"))
        launched.clear()

        state = client.get("/api/state").json()

        assert state["overrides"] == {}
        assert state["families"]["QF"]["value"] == pytest.approx(1.1)
        assert launched == list(FAST_FIGURES)

    def test_select_clears_overrides_and_keeps_settings(self, client):
        client.get("/api/state")
        client.post("/api/state/param", json={"family": "QF", "value": 1.05})
        client.put("/api/settings", json={"settings": {"da": {"nturns": 1024}}})

        r = client.post("/api/models/select", json={"name": "TRANSFER"})
        assert r.status_code == 200

        state = client.get("/api/state").json()
        assert state["model"] == "TRANSFER"
        assert state["overrides"] == {}
        assert state["baseline"]["overrides"] == {}
        assert state["settings"]["da"]["nturns"] == 1024
        assert [m["name"] for m in client.get("/api/models").json() if m["selected"]] == [
            "TRANSFER"
        ]

    def test_select_unknown_model_404(self, client):
        assert client.post("/api/models/select", json={"name": "SPARE"}).status_code == 404

    def test_select_unserved_model(self, client):
        assert client.post("/api/models/select", json={"name": "BOOSTER"}).json()["model"] == (
            "BOOSTER"
        )


_OPTICS_RAW = {"s_pos": [0.0, 1.0], "beta_x": [1.0, 2.0], "beta_y": [2.0, 1.0], "eta_x": [0.0, 0.1]}


class TestModelSwitch:
    @pytest.fixture
    def figures(self, tmp_path):
        return tmp_path / "ws" / "lattice" / "figures"

    def test_switch_clears_figures_until_the_worker_writes(self, client, figures):
        client.get("/api/state")
        (figures / "optics.json").write_text(json.dumps(_OPTICS_RAW))
        (figures / "da.json").write_text(json.dumps({"da_x": [], "da_y": [], "area_mm2": 0}))
        assert client.get("/api/figures/optics").status_code == 200

        client.post("/api/models/select", json={"name": "TRANSFER"})

        r = client.get("/api/figures/optics")
        assert r.status_code == 404
        assert r.json()["detail"] == "Figure not yet computed: optics"
        assert list(figures.glob("*.json")) == []
        (figures / "optics.json").write_text(json.dumps(_OPTICS_RAW))
        assert client.get("/api/figures/optics").status_code == 200

    def test_switch_cancels_the_running_workers(self, client, monkeypatch):
        client.get("/api/state")
        cancelled = []
        monkeypatch.setattr(
            compute_mod.ComputeManager, "cancel_all", lambda self: cancelled.append(True)
        )

        client.post("/api/models/select", json={"name": "TRANSFER"})

        assert cancelled == [True]

    def test_unknown_model_keeps_the_figures(self, client, figures):
        client.get("/api/state")
        (figures / "optics.json").write_text(json.dumps(_OPTICS_RAW))

        assert client.post("/api/models/select", json={"name": "SPARE"}).status_code == 404
        assert client.get("/api/figures/optics").status_code == 200


class TestSinglePass:
    @pytest.fixture
    def transfer(self, client, launched):
        client.get("/api/state")
        client.post("/api/models/select", json={"name": "TRANSFER"})
        launched.clear()
        return client

    def test_state_names_optics_only(self, transfer):
        state = transfer.get("/api/state").json()

        assert state["fast_figures"] == ["optics"]
        assert "tunes" not in state["summary"]
        assert "chromaticity" not in state["summary"]
        assert state["twiss_in"] == TWISS_IN

    def test_refresh_launches_only_optics(self, transfer, launched):
        r = transfer.post("/api/refresh")

        assert r.json()["launched"] == ["optics"]
        assert launched == ["optics"]

    def test_chromaticity_figure_409(self, transfer):
        r = transfer.get("/api/figures/chromaticity")

        assert r.status_code == 409
        assert r.json()["detail"] == SINGLE_PASS_UNAVAILABLE

    @pytest.mark.parametrize("name", [n for n in ALL_FIGURES if n != "optics"])
    def test_every_route_refuses_figures_beyond_optics(self, transfer, launched, name):
        for method, path in (
            ("get", f"/api/figures/{name}"),
            ("get", f"/api/data/{name}"),
            ("post", f"/api/refresh/{name}"),
        ):
            r = getattr(transfer, method)(path)
            assert r.status_code == 409, path
            assert r.json()["detail"] == SINGLE_PASS_UNAVAILABLE
        assert launched == []

    def test_verify_409(self, transfer, launched):
        assert transfer.post("/api/verify").status_code == 409
        assert launched == []

    def test_unknown_figure_stays_404(self, transfer):
        assert transfer.get("/api/figures/tune").status_code == 404

    def test_optics_still_refreshes(self, transfer, launched):
        assert transfer.post("/api/refresh/optics").status_code == 200
        assert launched == ["optics"]
