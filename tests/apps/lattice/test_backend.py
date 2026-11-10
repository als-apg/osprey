"""The backend: one computation per deck and settings, read by every caller, kept on disk."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from lattice_app.config import Settings, load_settings
from lattice_app.jobs import Jobs, WorkerFailed, build_request, run_worker
from lattice_app.server import NO_MODELS, create_app
from lattice_app.tools import thin

OPTICS = {
    "scalars": {"tune_x": 14.13, "tune_y": 6.22, "chromaticity_x": float("nan")},
    "units": {},
    "tables": {
        "twiss": {
            "columns": {
                "s": [0.0, 1.0, 2.0],
                "beta_x": [1.0, 2.0, 1.0],
                "beta_y": [2.0, 1.0, 2.0],
                "eta_x": [0.0, 0.1, 0.0],
                "name": ["a", "b", "c"],
            },
            "units": {"s": "m"},
        }
    },
}


class Runner:
    """A worker that answers what it is told and counts its starts."""

    def __init__(self, answer: Any = OPTICS) -> None:
        self.answer = answer
        self.jobs: list[Mapping[str, Any]] = []
        self.release = asyncio.Event()
        self.release.set()

    async def __call__(self, _command: tuple[str, ...], job: Mapping[str, Any], _deadline: float):
        self.jobs.append(job)
        await self.release.wait()
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _settings(root: Path, **over: Any) -> Settings:
    (root / "sr.lte").write_text("ring: line=(q)\n")
    (root / "booster.lte").write_text("booster: line=(q)\n")
    config = {
        "models": [
            {"name": "sr", "label": "Storage ring", "deck": "sr.lte"},
            {"name": "booster", "deck": "booster.lte", "energies_gev": [0.425, 6.0]},
        ],
        **over,
    }
    return load_settings(config, root)


def _client(tmp_path: Path, runner: Runner, **over: Any) -> TestClient:
    app = create_app(_settings(tmp_path, **over), cache=tmp_path / "cache", runner=runner)
    return TestClient(app)


def test_the_state_lists_models_and_computes_nothing(tmp_path: Path) -> None:
    runner = Runner()
    with _client(tmp_path, runner) as client:
        state = client.get("/api/state").json()

    assert state["error"] is None
    assert [model["name"] for model in state["models"]] == ["sr", "booster"]
    assert state["models"][0]["deck"] == "sr.lte"
    assert state["selected"] == {"model": "sr", "energy_gev": None}
    assert state["results"]["da"]["status"] == "absent"
    assert state["results"]["da"]["deadline_s"] == 300
    assert runner.jobs == []


def test_a_result_is_computed_once_for_every_caller(tmp_path: Path) -> None:
    runner = Runner()
    with _client(tmp_path, runner) as client:
        answers = [
            client.get("/api/result", params={"analysis": "optics", "wait_s": 5}).json()
            for _ in range(20)
        ]
        state = client.get("/api/state").json()

    assert len(runner.jobs) == 1
    assert {answer["status"] for answer in answers} == {"ready"}
    assert answers[0]["tables"]["twiss"]["columns"]["beta_x"] == [1.0, 2.0, 1.0]
    assert answers[0]["engine"] == "pyat"
    # A non-finite number is null in the document, so it is strict JSON.
    assert answers[0]["scalars"] == {"tune_x": 14.13, "tune_y": 6.22, "chromaticity_x": None}
    # The state carries a ready result's scalars.
    assert state["results"]["optics"]["scalars"]["tune_x"] == 14.13


def test_a_restart_reads_the_kept_result(tmp_path: Path) -> None:
    first = Runner()
    with _client(tmp_path, first) as client:
        client.get("/api/result", params={"analysis": "optics", "wait_s": 5})
    second = Runner()
    with _client(tmp_path, second) as client:
        answer = client.get("/api/result", params={"analysis": "optics"}).json()

    assert answer["status"] == "ready"
    assert second.jobs == []


def test_a_changed_deck_or_setting_is_a_new_result(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    sr = settings.models[0]
    key = build_request(sr, "da", None).key

    (tmp_path / "sr.lte").write_text("ring: line=(q, q)\n")
    assert build_request(sr, "da", None).key != key

    reduced = _settings(tmp_path, analyses={"da": {"settings": {"nturns": 128}}}).models[0]
    assert build_request(reduced, "da", None).key != build_request(sr, "da", None).key
    booster = settings.models[1]
    assert build_request(booster, "da", 0.425).key != build_request(booster, "da", 6.0).key


def test_two_models_over_one_deck_and_setup_share_a_result(tmp_path: Path) -> None:
    (tmp_path / "a.lte").write_text("x")
    (tmp_path / "b.lte").write_text("x")
    a, b = load_settings(
        {"models": [{"name": "a", "deck": "a.lte"}, {"name": "b", "deck": "b.lte"}]}, tmp_path
    ).models
    assert build_request(a, "optics", None).key == build_request(b, "optics", None).key


def test_a_failure_stays_failed_until_retried(tmp_path: Path) -> None:
    runner = Runner(WorkerFailed("stopped at its deadline of 300 s"))
    with _client(tmp_path, runner) as client:
        params = {"analysis": "da", "wait_s": 5}
        first = client.get("/api/result", params=params).json()
        again = client.get("/api/result", params=params).json()
        assert len(runner.jobs) == 1
        runner.answer = {"tables": {"aperture": {"columns": {"x_mm": [1.0], "y_mm": [2.0]}}}}
        retried = client.get("/api/result", params={**params, "retry": "true"}).json()

    assert first["status"] == again["status"] == "failed"
    assert first["error"] == "stopped at its deadline of 300 s"
    assert retried["status"] == "ready"
    assert len(runner.jobs) == 2


def test_an_answer_without_the_analysis_table_fails_with_the_reason(tmp_path: Path) -> None:
    with _client(tmp_path, Runner({"tables": {}})) as client:
        answer = client.get("/api/result", params={"analysis": "optics", "wait_s": 5}).json()

    assert answer["status"] == "failed"
    assert answer["error"] == "the engine's output lacks table twiss"


def test_a_running_result_answers_running_not_an_error(tmp_path: Path) -> None:
    runner = Runner()
    runner.release.clear()
    with _client(tmp_path, runner) as client:
        answer = client.get("/api/result", params={"analysis": "lma"}).json()
        export = client.get("/api/export", params={"analysis": "lma"})

    assert answer["status"] in {"queued", "running"}
    assert "tables" not in answer
    assert export.status_code == 409
    assert "not ready; read its result first" in export.json()["detail"]


def test_the_energy_picks_its_own_result(tmp_path: Path) -> None:
    runner = Runner()
    with _client(tmp_path, runner) as client:
        low = client.get("/api/result", params={"analysis": "optics", "model": "booster"}).json()
        high = client.get(
            "/api/result", params={"analysis": "optics", "model": "booster", "energy_gev": 6}
        ).json()
        refused = client.get(
            "/api/result", params={"analysis": "optics", "model": "booster", "energy_gev": 3}
        )

    assert (low["energy_gev"], high["energy_gev"]) == (0.425, 6.0)
    assert low["key"] != high["key"]
    assert refused.status_code == 400
    assert refused.json()["detail"] == ("model booster offers no energy 3 GeV; it offers: 0.425, 6")


def test_what_the_deployment_does_not_offer_is_refused_by_name(tmp_path: Path) -> None:
    over = {"analyses": {"footprint": {"enabled": False}}}
    with _client(tmp_path, Runner(), **over) as client:
        unknown_model = client.get("/api/state", params={"model": "linac"})
        unknown = client.get("/api/result", params={"analysis": "orbit"})
        unoffered = client.get("/api/result", params={"analysis": "footprint"})
        fmt = client.get("/api/export", params={"analysis": "optics", "format": "xlsx"})

    assert unknown_model.status_code == 404
    assert unknown_model.json()["detail"] == "there is no model linac; the models: sr, booster"
    assert unknown.status_code == 404
    assert unoffered.status_code == 400
    assert "does not offer footprint" in unoffered.json()["detail"]
    assert fmt.status_code == 400


def test_a_missing_deck_is_the_analysis_failure_not_the_panels(tmp_path: Path) -> None:
    settings = load_settings({"models": [{"name": "sr", "deck": "gone.lte"}]}, tmp_path)
    with TestClient(create_app(settings, cache=tmp_path / "cache", runner=Runner())) as client:
        state = client.get("/api/state")
        result = client.get("/api/result", params={"analysis": "optics"})

    assert state.status_code == 200
    assert state.json()["results"]["optics"]["status"] == "failed"
    assert "gone.lte" in state.json()["results"]["optics"]["error"]
    assert result.status_code == 409


def test_no_model_says_where_to_configure_one(tmp_path: Path) -> None:
    app = create_app(Settings(models=()), cache=tmp_path / "cache", runner=Runner())
    with TestClient(app) as client:
        state = client.get("/api/state").json()
        result = client.get("/api/result", params={"analysis": "optics"})

    assert state["error"] == NO_MODELS
    assert state["selected"] is None
    assert result.status_code == 409


def test_precompute_starts_every_offered_analysis_at_startup(tmp_path: Path) -> None:
    runner = Runner()
    over = {"precompute": True, "analyses": {"lma": {"enabled": False}}, "max_workers": 8}
    with _client(tmp_path, runner, **over) as client:
        client.get("/api/result", params={"analysis": "optics", "wait_s": 5})
        client.get("/api/result", params={"analysis": "da", "model": "booster", "wait_s": 5})

    # Four analyses of the ring, and four at each of the booster's two energies. The two
    # decks differ, so no result is shared.
    assert len(runner.jobs) == 12


def test_the_export_is_the_result_as_a_file(tmp_path: Path) -> None:
    with _client(tmp_path, Runner()) as client:
        client.get("/api/result", params={"analysis": "optics", "wait_s": 5})
        csv = client.get("/api/export", params={"analysis": "optics"})
        sdds = client.get("/api/export", params={"analysis": "optics", "format": "sdds"})
        missing = client.get("/api/export", params={"analysis": "optics", "table": "orbit"})

    assert csv.headers["content-disposition"] == 'attachment; filename="sr-optics-twiss.csv"'
    lines = csv.text.splitlines()
    assert "# model = sr" in lines and "# engine = pyat" in lines and "# tune_x = 14.13" in lines
    header = lines.index("s [m],beta_x,beta_y,eta_x,name")
    assert lines[header + 1 :] == ["0.0,1.0,2.0,0.0,a", "1.0,2.0,1.0,0.1,b", "2.0,1.0,2.0,0.0,c"]

    text = sdds.text
    assert text.startswith("SDDS1\n")
    assert "&parameter name=tune_x, type=double &end" in text
    assert '&column name=s, type=double, units="m" &end' in text
    assert "&column name=name, type=string &end" in text
    assert text.endswith('3\n0.0 1.0 2.0 0.0 "a"\n1.0 2.0 1.0 0.1 "b"\n2.0 1.0 2.0 0.0 "c"\n')
    assert missing.status_code == 404


def test_the_panel_and_its_files_are_served(tmp_path: Path) -> None:
    with _client(tmp_path, Runner()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert 'id="lattice"' in client.get("/").text
        assert client.get("/static/panel.js").status_code == 200


async def test_workers_beyond_the_limit_queue(tmp_path: Path) -> None:
    runner = Runner()
    runner.release.clear()
    settings = _settings(tmp_path)
    jobs = Jobs(tmp_path / "cache", 1, runner)
    first = build_request(settings.models[0], "optics", None)
    second = build_request(settings.models[1], "optics", 6.0)

    jobs.ensure(first)
    jobs.ensure(second)
    jobs.ensure(first)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert jobs.status(first.key)["status"] == "running"
    assert jobs.status(second.key)["status"] == "queued"
    runner.release.set()
    await jobs.wait(second.key, 5)
    assert jobs.status(second.key)["status"] == "ready"
    await jobs.close()


def _engine(tmp_path: Path, body: str) -> tuple[str, ...]:
    script = tmp_path / "engine.py"
    script.write_text(body)
    return (sys.executable, str(script))


async def test_a_worker_is_a_command_given_a_job_file_and_an_output_file(tmp_path: Path) -> None:
    command = _engine(
        tmp_path,
        "import json, sys\n"
        "job = json.load(open(sys.argv[1]))\n"
        "json.dump({'scalars': {'echo': job['analysis']}}, open(sys.argv[2], 'w'))\n",
    )

    assert await run_worker(command, {"analysis": "optics"}, 30) == {"scalars": {"echo": "optics"}}


async def test_a_worker_past_its_deadline_is_stopped_and_says_what_to_change(
    tmp_path: Path,
) -> None:
    command = _engine(tmp_path, "import time\ntime.sleep(60)\n")

    with pytest.raises(WorkerFailed, match="stopped at its deadline of 0.3 s; raise the"):
        await run_worker(command, {"analysis": "da"}, 0.3)


async def test_a_failed_worker_reports_what_it_wrote_to_stderr(tmp_path: Path) -> None:
    command = _engine(
        tmp_path, "import sys\nprint('no beamline RING', file=sys.stderr)\nsys.exit(1)\n"
    )

    with pytest.raises(WorkerFailed, match="no beamline RING"):
        await run_worker(command, {"analysis": "optics"}, 30)
    with pytest.raises(WorkerFailed, match="the engine could not start"):
        await run_worker((str(tmp_path / "absent"),), {"analysis": "optics"}, 30)


def test_a_long_table_is_thinned_evenly_and_keeps_its_ends() -> None:
    table = {"columns": {"s": list(range(101)), "beta_x": list(range(101))}, "units": {"s": "m"}}

    thinned = thin(table, 5)

    assert thinned["columns"]["s"] == [0, 25, 50, 75, 100]
    assert (thinned["rows_total"], thinned["rows_shown"]) == (101, 5)
    assert thin(table, 200)["rows_shown"] == 101
