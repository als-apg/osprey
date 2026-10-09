"""The lattice dashboard app over a render's simulator view.

Each app is built over a synthetic render under ``tmp_path``: its simulator
view ``data/simulator/`` holds ``variables.json`` with the models and a pyAT
JSON deck per deck-bearing model. Worker launches go to ``FakeSlots``
(conftest), so no subprocess runs.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.lattice_dashboard import app as app_mod
from osprey.interfaces.lattice_dashboard.app import create_app
from osprey.interfaces.lattice_dashboard.catalog import NO_SERVED_MODEL_TEXT, NO_VIEW_TEXT
from osprey.interfaces.lattice_dashboard.state import (
    ALL_FIGURES,
    FAST_FIGURES,
    SINGLE_PASS_UNAVAILABLE,
    VERIFICATION_FIGURES,
)
from osprey.interfaces.lattice_dashboard.workers._base import save_data
from osprey_connectors.process import ExitCause

at = pytest.importorskip("at")

TWISS_IN = {"beta": [8.0, 4.0], "alpha": [0.0, 0.0]}


def _deck(kf: float = 1.0) -> at.Lattice:
    d = at.Drift("DR", 0.5)
    qf = at.Quadrupole("QF", 0.2, kf)
    qd = at.Quadrupole("QD", 0.2, -kf)
    b = at.Dipole("BM", 0.5, np.pi / 8)
    return at.Lattice([qf, d, b, d, qd, d, b, d] * 2, name="FODO", energy=2e9)


def _write_render(root, *, served, models):
    """Write a render's simulator view.

    ``models`` is ``{name: settings.pyat or None}`` (None = no deck); a model is
    served when ``served`` names it, and the texture model is listed too.
    """
    view = root / "data" / "simulator"
    (view / "decks").mkdir(parents=True, exist_ok=True)
    records = [
        {
            "name": "texture",
            "engine": "texture",
            "served": "texture" in served,
            "settings": {},
            "deck": None,
        }
    ]
    for name, pyat in models.items():
        deck = None
        if pyat is not None:
            deck = f"decks/{name}.json"
            _deck().save(str(view / deck))
        records.append(
            {
                "name": name,
                "engine": "pyat",
                "served": name in served,
                "settings": {"pyat": pyat} if pyat is not None else {},
                "deck": deck,
            }
        )
    (view / "variables.json").write_text(
        json.dumps({"schema": "osprey.facility.variables/1", "models": records, "channels": []})
    )
    return root


@pytest.fixture(autouse=True)
def launched(fake_slots):
    """The figure workers launched, by name, with no subprocess."""
    return fake_slots.launched


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


def settle(client) -> dict:
    """Return the state once no resolve is loading."""
    deadline = time.monotonic() + 20
    while True:
        state = client.get("/api/state").json()
        if state["selection"]["status"] != "loading":
            return state
        assert time.monotonic() < deadline, "the selection never resolved"
        time.sleep(0.02)


def select(client, name: str) -> dict:
    """Select model *name* and return the state once it resolved."""
    response = client.post("/api/models/select", json={"name": name})
    assert response.status_code == 200, response.text
    return settle(client)


def job_spec(client, slots, name: str) -> dict:
    """Return the job file figure *name*'s current job was started with."""
    job = slots.current(name)

    async def started():
        for _ in range(500):
            if job.argv is not None:
                return
            await asyncio.sleep(0.01)

    client.portal.call(started)
    return json.loads(Path(job.argv[-2]).read_text())


def land(client, slots, name: str, data: dict | None = None, *, cause=ExitCause.COMPLETED):
    """End figure *name*'s current job: write *data* as its output, then exit with *cause*."""
    job = slots.current(name)

    async def finish():
        for _ in range(500):
            if job.argv is not None or job._exit is not None:
                break
            await asyncio.sleep(0.01)
        if data is not None:
            spec = json.loads(Path(job.argv[-2]).read_text())
            save_data(spec, data, Path(job.argv[-1]))
        job.finish(cause, 0 if cause is ExitCause.COMPLETED else 1, "boom")
        for _ in range(10):
            await asyncio.sleep(0)

    client.portal.call(finish)


@pytest.fixture
def client(tmp_path, render):
    with TestClient(create_app(workspace_root=tmp_path / "ws", render_root=render)) as client:
        settle(client)
        yield client


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
        assert not (tmp_path / "ws" / "lattice").exists()


class TestNoView:
    def test_no_config_loaded_gives_the_no_view_state(self, tmp_path, monkeypatch, launched):
        monkeypatch.setattr(app_mod, "default_config_path", lambda: None)
        with TestClient(create_app(workspace_root=tmp_path)) as client:
            state = settle(client)

            assert state["notice"] == NO_VIEW_TEXT
            assert state["selection"]["model"] is None
            assert state["selection"]["status"] == "none"
            assert client.get("/api/models").json() == []
        assert launched == []

    def test_render_without_the_view(self, tmp_path, launched):
        app = create_app(workspace_root=tmp_path / "ws", render_root=tmp_path)
        with TestClient(app) as client:
            assert settle(client)["notice"] == NO_VIEW_TEXT
        assert launched == []

    def test_texture_only_render(self, tmp_path, launched):
        render = _write_render(tmp_path / "render", served=["texture"], models={})
        app = create_app(workspace_root=tmp_path / "ws", render_root=render)
        with TestClient(app) as client:
            state = settle(client)

        assert state["notice"] == NO_SERVED_MODEL_TEXT
        assert state["selection"]["model"] is None
        assert launched == []


class TestModels:
    def test_models_read_from_the_render(self, client):
        models = client.get("/api/models").json()

        assert models == [
            {"name": "SR", "served": True, "solve": "periodic", "selected": True, "error": None},
            {
                "name": "TRANSFER",
                "served": True,
                "solve": "single_pass",
                "selected": False,
                "error": None,
            },
            {
                "name": "BOOSTER",
                "served": False,
                "solve": "periodic",
                "selected": False,
                "error": None,
            },
        ]

    def test_startup_selects_the_first_served_deck(self, client, launched):
        state = client.get("/api/state").json()

        assert state["selection"] == {
            "model": "SR",
            "deck_sha256": state["selection"]["deck_sha256"],
            "status": "ready",
            "error": None,
            "capabilities": {
                "figures": list(ALL_FIGURES),
                "fast_figures": list(FAST_FIGURES),
                "verify": True,
            },
        }
        assert state["notice"] is None
        assert state["summary"]["energy_gev"] == pytest.approx(2.0)
        assert "tunes" not in state["summary"]
        assert launched == list(FAST_FIGURES)

    def test_changed_deck_reloads(self, client, render, launched):
        client.post("/api/state/param", json={"family": "QF", "value": 1.05})
        _deck(kf=1.1).save(str(render / "data/simulator/decks/SR.json"))
        launched.clear()

        client.get("/api/state")
        state = settle(client)

        assert state["overrides"] == {}
        assert state["families"]["QF"]["value"] == pytest.approx(1.1)
        assert launched == list(FAST_FIGURES)

    def test_select_clears_overrides_and_keeps_settings(self, client):
        client.post("/api/state/param", json={"family": "QF", "value": 1.05})
        client.put("/api/settings", json={"settings": {"da": {"nturns": 1024}}})

        state = select(client, "TRANSFER")

        assert state["selection"]["model"] == "TRANSFER"
        assert state["overrides"] == {}
        assert state["baseline"]["overrides"] == {}
        assert state["settings"]["da"]["nturns"] == 1024
        assert [m["name"] for m in client.get("/api/models").json() if m["selected"]] == [
            "TRANSFER"
        ]

    def test_select_answers_at_once_with_the_model_loading(self, client):
        body = client.post("/api/models/select", json={"name": "TRANSFER"}).json()

        assert body["model"] == "TRANSFER"
        assert body["status"] == "loading"

    def test_select_answers_while_a_resolve_runs(self, client, monkeypatch):
        release = threading.Event()
        resolve = app_mod.resolve_selection

        def blocked(*args, **kwargs):
            release.wait(10)
            return resolve(*args, **kwargs)

        monkeypatch.setattr(app_mod, "resolve_selection", blocked)
        try:
            client.post("/api/models/select", json={"name": "TRANSFER"})
            answers = []
            second = threading.Thread(
                target=lambda: answers.append(
                    client.post("/api/models/select", json={"name": "SR"}).status_code
                )
            )
            second.start()
            second.join(timeout=5)

            assert answers == [200]
        finally:
            release.set()
        assert settle(client)["selection"]["model"] == "SR"

    def test_select_unknown_model_404(self, client):
        assert client.post("/api/models/select", json={"name": "SPARE"}).status_code == 404

    def test_select_unserved_model(self, client):
        assert select(client, "BOOSTER")["selection"]["model"] == "BOOSTER"


class TestCatalog:
    def test_bad_pyat_settings_name_the_engine_stop(self, tmp_path):
        render = _write_render(
            tmp_path / "render",
            served=["SR", "BAD"],
            models={"SR": {"solve": "periodic"}, "BAD": {"solve": "periodic", "bogus": 1}},
        )
        app = create_app(workspace_root=tmp_path / "ws", render_root=render)
        with TestClient(app) as client:
            settle(client)
            models = {m["name"]: m for m in client.get("/api/models").json()}
            state = select(client, "BAD")

        assert models["SR"]["error"] is None
        assert models["BAD"]["solve"] is None
        assert "engine-invalid" in models["BAD"]["error"]
        assert "bogus" in models["BAD"]["error"]
        assert state["selection"]["status"] == "failed"
        assert state["selection"]["error"] == models["BAD"]["error"]

    def test_the_job_carries_the_engine_normalised_twiss_in(self, client, fake_slots):
        select(client, "TRANSFER")

        spec = job_spec(client, fake_slots, "optics")

        twiss_in = spec["prepared"]["twiss_in"]
        assert twiss_in["beta"] == TWISS_IN["beta"]
        assert twiss_in["closed_orbit"] == [0.0] * 6
        assert spec["key"] == client.get("/api/state").json()["figures"]["optics"]["key"]
        assert spec["job_id"] == fake_slots.current("optics").job
        assert spec["baseline_overrides"] == {}


class TestSelection:
    def test_get_state_launches_no_worker(self, client, launched):
        launched.clear()

        client.get("/api/state")
        client.get("/api/models")
        settle(client)

        assert launched == []

    def test_model_gone_from_render_is_no_model(self, client, render):
        select(client, "BOOSTER")
        _write_render(
            render,
            served=["SR", "TRANSFER", "texture"],
            models={"SR": {"solve": "periodic"}, "TRANSFER": {"solve": "periodic"}},
        )

        client.get("/api/state")
        state = settle(client)

        assert state["selection"]["model"] is None
        assert state["selection"]["status"] == "none"
        assert state["families"] == {}
        assert client.get("/api/figures/optics").status_code == 404

    def test_served_only_rebuild_reresolves(self, client, render, launched):
        client.post("/api/state/param", json={"family": "QF", "value": 1.05})
        launched.clear()
        _write_render(
            render,
            served=["SR", "TRANSFER", "BOOSTER", "texture"],
            models={
                "BOOSTER": {"solve": "periodic"},
                "SR": {"solve": "periodic"},
                "TRANSFER": {"solve": "single_pass", "twiss_in": TWISS_IN},
            },
        )

        client.get("/api/models")
        state = settle(client)

        served = {m["name"]: m["served"] for m in client.get("/api/models").json()}
        assert served["BOOSTER"] is True
        assert state["selection"]["model"] == "SR"
        assert state["overrides"] == {"QF": 1.05}
        assert launched == []

    def test_load_failure_is_status_failed_not_500(self, client, render):
        (render / "data/simulator/decks/BOOSTER.json").write_text("not a deck {")

        response = client.post("/api/models/select", json={"name": "BOOSTER"})
        state = settle(client)

        assert response.status_code == 200
        assert state["selection"]["model"] == "BOOSTER"
        assert state["selection"]["status"] == "failed"
        assert state["selection"]["error"]
        assert client.get("/api/state").status_code == 200


_OPTICS_RAW = {
    "s_pos": [0.0, 1.0],
    "beta_x": [1.0, 2.0],
    "beta_y": [2.0, 1.0],
    "eta_x": [0.0, 0.1],
    "baseline": None,
    "summary_updates": {"tunes": [0.31, 0.21], "chromaticity": [1.0, 2.0], "beta_max": [2, 2]},
}


class TestKeyedFigures:
    def test_switch_answers_not_computed_until_the_new_key_lands(self, client, fake_slots):
        land(client, fake_slots, "optics", _OPTICS_RAW)
        assert client.get("/api/figures/optics").status_code == 200

        select(client, "TRANSFER")

        r = client.get("/api/figures/optics")
        assert r.status_code == 404
        assert r.json()["status"] in ("computing", "not_computed")
        land(client, fake_slots, "optics", _OPTICS_RAW)
        r = client.get("/api/figures/optics")
        assert r.status_code == 200
        assert r.json()["key"] == client.get("/api/state").json()["figures"]["optics"]["key"]

    def test_reselect_unchanged_deck_serves_the_old_figure_again(self, client, fake_slots):
        land(client, fake_slots, "optics", _OPTICS_RAW)
        sr_key = client.get("/api/figures/optics").json()["key"]

        select(client, "TRANSFER")
        select(client, "SR")

        r = client.get("/api/figures/optics")
        assert r.status_code == 200
        assert r.json()["key"] == sr_key

    def test_override_makes_the_figure_stale_not_served(self, client, fake_slots):
        land(client, fake_slots, "optics", _OPTICS_RAW)
        assert client.get("/api/state").json()["summary"]["tunes"] == [0.31, 0.21]

        client.post("/api/state/param", json={"family": "QF", "value": 1.05})

        r = client.get("/api/figures/optics")
        assert r.status_code == 404
        assert r.json()["status"] == "stale"
        state = client.get("/api/state").json()
        assert state["figures"]["optics"]["status"] == "stale"
        assert "tunes" not in state["summary"]

    def test_failed_recompute_is_not_served_as_current(self, client, fake_slots):
        land(client, fake_slots, "optics", _OPTICS_RAW)
        client.post("/api/state/param", json={"family": "QF", "value": 1.05})
        client.post("/api/refresh/optics")

        land(client, fake_slots, "optics", cause=ExitCause.FAILED)

        r = client.get("/api/figures/optics")
        assert r.status_code == 404
        assert r.json()["status"] == "failed"
        assert r.json()["error"] == "Worker exited with code 1: boom"

    def test_deck_change_via_sync_never_serves_the_old_deck(self, client, render, fake_slots):
        land(client, fake_slots, "optics", _OPTICS_RAW)
        _deck(kf=1.1).save(str(render / "data/simulator/decks/SR.json"))

        client.get("/api/state")
        settle(client)

        r = client.get("/api/figures/optics")
        assert r.status_code == 404
        assert r.json()["status"] in ("computing", "not_computed")

    def test_select_broadcasts_no_figure_error(self, client, monkeypatch):
        events = []
        monkeypatch.setattr(
            app_mod._SSEBroadcaster, "broadcast", lambda self, data: events.append(data)
        )

        select(client, "TRANSFER")
        select(client, "SR")

        assert [e for e in events if e["type"] == "figure_error"] == []

    def test_unknown_model_keeps_the_figures(self, client, fake_slots):
        land(client, fake_slots, "optics", _OPTICS_RAW)

        assert client.post("/api/models/select", json={"name": "SPARE"}).status_code == 404
        assert client.get("/api/figures/optics").status_code == 200


class TestSinglePass:
    @pytest.fixture
    def transfer(self, client, launched):
        select(client, "TRANSFER")
        launched.clear()
        return client

    def test_state_names_optics_only(self, transfer):
        state = transfer.get("/api/state").json()

        assert state["selection"]["capabilities"] == {
            "figures": ["optics"],
            "fast_figures": ["optics"],
            "verify": False,
        }
        assert "tunes" not in state["summary"]
        assert "chromaticity" not in state["summary"]

    def test_refresh_launches_only_optics(self, transfer, launched):
        r = transfer.post("/api/refresh")

        assert r.json()["launched"] == ["optics"]
        assert launched == ["optics"]

    def test_chromaticity_figure_409(self, transfer):
        r = transfer.get("/api/figures/chromaticity")

        assert r.status_code == 409
        assert r.json() == {"status": "unavailable", "reason": SINGLE_PASS_UNAVAILABLE}

    @pytest.mark.parametrize("name", [n for n in ALL_FIGURES if n != "optics"])
    def test_every_route_refuses_figures_beyond_optics(self, transfer, launched, name):
        for method, path in (
            ("get", f"/api/figures/{name}"),
            ("get", f"/api/data/{name}"),
            ("post", f"/api/refresh/{name}"),
        ):
            r = getattr(transfer, method)(path)
            assert r.status_code == 409, path
            assert r.json()["reason"] == SINGLE_PASS_UNAVAILABLE
        assert launched == []

    def test_verify_409(self, transfer, launched):
        assert transfer.post("/api/verify").status_code == 409
        assert launched == []

    def test_unknown_figure_stays_404(self, transfer):
        assert transfer.get("/api/figures/tune").status_code == 404

    def test_optics_still_refreshes(self, transfer, launched):
        assert transfer.post("/api/refresh/optics").status_code == 200
        assert launched == ["optics"]


class TestUnservedIsAMark:
    @pytest.fixture
    def booster(self, client, launched):
        launched.clear()
        select(client, "BOOSTER")
        return client

    def test_state_names_every_fast_figure(self, booster):
        selection = booster.get("/api/state").json()["selection"]

        assert selection["model"] == "BOOSTER"
        assert "served" not in selection
        assert selection["capabilities"]["fast_figures"] == list(FAST_FIGURES)

    @pytest.mark.usefixtures("booster")
    def test_select_launches_every_fast_figure(self, launched):
        assert launched == list(FAST_FIGURES)

    def test_resonance_figure_is_not_refused(self, booster):
        r = booster.get("/api/figures/resonance")

        assert r.status_code == 404
        assert r.json()["status"] in ("computing", "not_computed")

    def test_verify_launches_da_and_lma(self, booster, launched):
        launched.clear()
        r = booster.post("/api/verify")

        assert r.status_code == 200
        assert r.json()["launched"] == list(VERIFICATION_FIGURES)
        assert launched == list(VERIFICATION_FIGURES)

    def test_models_keep_the_served_mark(self, booster):
        models = {m["name"]: m["served"] for m in booster.get("/api/models").json()}

        assert models == {"SR": True, "TRANSFER": True, "BOOSTER": False}
