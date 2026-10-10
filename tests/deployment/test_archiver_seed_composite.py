"""The deploy-time seed holds what the simulator serves.

The base seed reads every sample from the archive composite of the render's
simulator view, at the active set and anchor the scenario state file records:
a physics monitor's sample carries its engine's readout faults, a stand-in
deployment seeds the same history as a sandbox, and an ``at_offset`` event sits
at the persisted anchor whenever the seed happens to run.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from osprey.deployment import container_lifecycle
from osprey_connectors.simulation import archive as archive_module
from osprey_connectors.simulation import composite as composite_module
from osprey_connectors.simulation.archive import (
    DATE_FIELD,
    SeedKnobs,
    build,
    seed_base,
    synthesize_documents,
)
from osprey_connectors.simulation.composite import Composite
from osprey_connectors.simulation.state import write_active_state
from osprey_connectors.simulation.view import SimulatorView
from osprey_connectors.standin import archive_belongs_to_standin
from osprey_connectors.workspace import resolve_simulation_state_dir
from tests._simulator_view import write_texture_view

if TYPE_CHECKING:
    from tests._builds import BuiltProject

T0 = datetime(2026, 3, 14, 9, 26, 53, tzinfo=UTC)

#: An anchor on the coarse cadence, so the spike's peak is a grid timestamp.
ANCHOR = datetime(2026, 3, 14, 9, 26, tzinfo=UTC)

BPM_X = "SR:DIAG:BPM:17:POSITION:X"
BPM_Y = "SR:DIAG:BPM:17:POSITION:Y"

PRESSURE = "SR:VAC:IP07:PRESSURE"
SPIKE_OFFSET_S = -1800.0
SPIKE = {"shape": "spike", "at_offset": SPIKE_OFFSET_S, "amplitude": 5e-9, "width": 60.0}

#: A day of coarse history and an hour of dense: enough grid for a spike.
KNOBS = SeedKnobs(retention_days=1, hot_span_hours=1, hot_cadence_sec=10, tail_cadence_sec=60)


class _Collection:
    """The store calls a base seed makes, held in memory."""

    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []

    def create_index(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def insert_many(self, documents: list[dict[str, Any]], **kwargs: Any) -> None:
        del kwargs
        self.documents.extend(documents)

    def replace_one(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def samples(self) -> dict[datetime, dict[str, Any]]:
        return {
            document[DATE_FIELD]: {
                key: value for key, value in document.items() if key not in (DATE_FIELD, "expireAt")
            }
            for document in self.documents
        }


def _activate(root: Path, config: dict[str, Any], names: list[str], anchor: datetime) -> None:
    """Record ``names`` and ``anchor`` in the project's scenario state file, as `sim apply` does."""
    view = json.loads((SimulatorView.path_for_project(root) / "scenarios.json").read_text())
    targets = {scenario["name"]: set() for scenario in view["scenarios"]}
    path = resolve_simulation_state_dir(config, root) / "active_scenarios"
    write_active_state(path, targets, names, anchor=anchor)


# -- the demo's physics monitors -----------------------------------------------


@pytest.fixture
def demo_render(built_control_assistant: BuiltProject, tmp_path: Path) -> Path:
    """A deployment repo whose render holds the demo's simulator view."""
    render = tmp_path / "build"
    for name, data in built_control_assistant.outputs[0].files.items():
        if name.startswith("data/simulator/"):
            target = render / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    (render / "config.yml").write_text("{}\n", encoding="utf-8")
    return tmp_path


@pytest.mark.slow
def test_a_seeded_monitor_sample_under_a_readout_fault_is_the_served_reading(
    demo_render: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(composite_module, "default_config_path", lambda: None)
    _activate(demo_render, {}, ["bpm-polarity"], T0)
    view = demo_render / "build" / "data" / "simulator"
    state = resolve_simulation_state_dir({}, demo_render)
    instant = T0.timestamp() - 600.0

    archive = container_lifecycle._archiver_seed_inputs({}, demo_render)
    (document,) = synthesize_documents(archive, np.asarray([instant]))
    served = Composite(view, state_dir=state, clock=lambda: instant, model_log=False)
    nominal = Composite(view, clock=lambda: instant, model_log=False)

    for plane in (BPM_X, BPM_Y):
        assert document[plane] == served.get(plane)
        assert document[plane] != nominal.get(plane)


@pytest.mark.slow
def test_a_stand_in_deployment_seeds_the_history_a_sandbox_seeds(
    demo_render: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(composite_module, "default_config_path", lambda: None)
    standin = {
        "services": {"live_standin": {"port": 5064}},
        "deployed_services": ["archiver_recorder"],
    }
    assert archive_belongs_to_standin(standin)
    times = np.arange(T0.timestamp() - 300.0, T0.timestamp(), 10.0)

    with_standin = synthesize_documents(
        container_lifecycle._archiver_seed_inputs(standin, demo_render), times
    )
    without = synthesize_documents(
        container_lifecycle._archiver_seed_inputs({}, demo_render), times
    )

    assert with_standin == without


# -- the anchor -----------------------------------------------------------------


def test_two_seeds_at_different_wall_clock_times_hold_one_past(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The event sits at the persisted anchor plus its offset, not at the time of the seed."""
    write_texture_view(
        tmp_path,
        {PRESSURE: {"nominal": 1e-9}},
        {"burst": {"archiver": [{"channel": PRESSURE, "events": [SPIKE]}]}},
    )
    _activate(tmp_path, {}, ["burst"], ANCHOR)
    seeds = []
    for wall_clock in (ANCHOR + timedelta(minutes=5), ANCHOR + timedelta(hours=2)):
        monkeypatch.setattr(archive_module.time, "time", wall_clock.timestamp)
        collection = _Collection()
        seed_base(
            collection,  # type: ignore[arg-type]
            container_lifecycle._archiver_seed_inputs({}, tmp_path),
            KNOBS,
            t0=wall_clock,
        )
        seeds.append(collection.samples())

    shared = sorted(set(seeds[0]) & set(seeds[1]))
    assert len(shared) > 100
    assert [seeds[0][stamp] for stamp in shared] == [seeds[1][stamp] for stamp in shared]
    peak = max(shared, key=lambda stamp: seeds[0][stamp][PRESSURE])
    assert peak.timestamp() == ANCHOR.timestamp() + SPIKE_OFFSET_S
    assert seeds[0][peak][PRESSURE] == pytest.approx(6e-9)


def test_the_archive_composite_places_the_event_from_the_anchor_it_is_given(
    tmp_path: Path,
) -> None:
    view = write_texture_view(
        tmp_path,
        {PRESSURE: {"nominal": 1e-9}},
        {"burst": {"archiver": [{"channel": PRESSURE, "events": [SPIKE]}]}},
    )
    at = [T0.timestamp() + SPIKE_OFFSET_S]

    assert build(view, ["burst"], anchor_s=T0.timestamp()).series(PRESSURE, at) == [
        pytest.approx(6e-9)
    ]


# -- the rewrite, against a real store -------------------------------------------

TEMPERATURE = "SR:RF:CAV01:TEMP:BODY"
FLAG = "SR:VAC:IP07:STATUS:FAULT"
SETPOINT = "SR:MAG:HCM:01:CURRENT:SP"

STORE_CHANNELS = {
    PRESSURE: {"nominal": 1e-9},
    TEMPERATURE: {"nominal": 25.0},
    FLAG: {"value_type": "bool", "nominal": "FALSE"},
    SETPOINT: {"role": "setpoint", "nominal": 5.0},
}
#: What a spike of amplitude 3 still adds at four sigmas, where the rewrite cuts its window.
CUT = 3.0 * math.exp(-8.0)

STORE_SCENARIOS = {
    "burst": {"archiver": [{"channel": PRESSURE, "events": [SPIKE]}]},
    "warm": {
        "archiver": [
            {
                "channel": TEMPERATURE,
                "events": [{**SPIKE, "at_offset": -900.0, "amplitude": 3.0}],
            }
        ]
    },
    "trip": {
        "archiver": [
            {"channel": FLAG, "events": [{"shape": "step", "at_offset": -1800.0, "to": "TRUE"}]}
        ]
    },
}


@pytest.fixture(scope="module")
def mongo_store():
    from tests._container_support import is_docker_available
    from tests._mongo_container import started_mongo

    if not is_docker_available():
        pytest.skip("Docker not available — needed to rewrite a real store.")
    username, password = "composeuser", "composepass123"
    with started_mongo("mongodb-seed-composite", username=username, password=password) as (
        host,
        port,
    ):
        yield {"host": host, "port": port, "username": username, "password": password}


@pytest.fixture
def store(tmp_path: Path, mongo_store: dict[str, Any]):
    """A project whose texture view is seeded into an empty collection."""
    import yaml
    from pymongo import MongoClient

    root = tmp_path / "proj"
    write_texture_view(root, STORE_CHANNELS, STORE_SCENARIOS)
    config = {
        "project_name": "seed-composite",
        "project_root": str(root),
        "archiver": {
            "type": "mongodb_archiver",
            "mongodb_archiver": {
                "host": mongo_store["host"],
                "port": mongo_store["port"],
                "name": "seed_composite_db",
                "collection": "pv_history",
                "auth": {
                    "source": "admin",
                    "username": mongo_store["username"],
                    "password_env": "MONGO_ROOT_PASSWORD",
                },
                "timeout_s": 10,
            },
        },
        "va_archiver": {
            "retention_days": KNOBS.retention_days,
            "hot_span_hours": KNOBS.hot_span_hours,
            "hot_cadence_sec": KNOBS.hot_cadence_sec,
            "tail_cadence_sec": KNOBS.tail_cadence_sec,
        },
    }
    (root / "config.yml").write_text(yaml.safe_dump(config))
    (root / ".env").write_text(f"MONGO_ROOT_PASSWORD={mongo_store['password']}\n")
    client: Any = MongoClient(
        host=mongo_store["host"],
        port=mongo_store["port"],
        username=mongo_store["username"],
        password=mongo_store["password"],
        authSource="admin",
    )
    collection = client["seed_composite_db"]["pv_history"]
    collection.drop()
    seed_base(collection, build(root / "data" / "simulator", []), KNOBS, t0=ANCHOR, chunk_size=256)
    try:
        yield root, config, collection
    finally:
        collection.drop()
        client.close()


def _apply(root: Path, names: list[str]):
    from osprey.simulation.apply import apply_scenarios

    return apply_scenarios(root, names, seed_logbook=False, now=ANCHOR)


def _stored(collection) -> dict[datetime, dict[str, Any]]:
    return {
        document[DATE_FIELD]: {
            key: value for key, value in document.items() if key not in ("_id", DATE_FIELD)
        }
        for document in collection.find({DATE_FIELD: {"$exists": True}})
    }


def _reseed(root: Path, collection) -> None:
    collection.drop()
    seed_base(collection, build(root / "data" / "simulator", []), KNOBS, t0=ANCHOR, chunk_size=256)


def test_a_writes_journal_planted_before_a_rewrite_changes_no_sample(store) -> None:
    from osprey_connectors.control_system.va_in_process_connector import (
        JOURNAL_DIR,
        JOURNAL_FILE,
        active_set_sha256,
    )

    root, config, collection = store
    _apply(root, ["burst"])
    clean = _stored(collection)
    _reseed(root, collection)
    journal = resolve_simulation_state_dir(config, root) / JOURNAL_DIR / JOURNAL_FILE
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(
        json.dumps(
            {
                "active_set_sha256": active_set_sha256(["nominal", "burst"]),
                "seq": 1,
                "writes": [[1, SETPOINT, 1.0]],
            }
        ),
        encoding="utf-8",
    )

    result = _apply(root, ["burst"])

    assert result.archiver.updated > 0
    assert _stored(collection) == clean
    assert {sample[SETPOINT] for sample in clean.values() if SETPOINT in sample} == {5.0}


def test_a_rewritten_window_over_a_stored_flag_keeps_its_false_samples(store) -> None:
    root, _config, collection = store
    collection.update_many({FLAG: 0}, {"$set": {FLAG: False}})

    _apply(root, ["trip"])

    step = ANCHOR.timestamp() - 1800.0
    samples = [
        (document[DATE_FIELD].replace(tzinfo=UTC).timestamp(), document[FLAG])
        for document in collection.find({FLAG: {"$exists": True}})
    ]
    before = [value for moment, value in samples if moment < step]
    after = [value for moment, value in samples if moment >= step]
    assert before and after
    assert all(value is False for value in before)
    assert all(value is True for value in after)


def test_applying_one_set_then_another_leaves_none_of_the_first_s_events(store) -> None:
    """Every sample is the second set's, to the spike's four-sigma cut; the first
    set's channel is back on its held value everywhere."""
    from osprey.simulation.apply import DENSIFIED_FIELD

    root, _config, collection = store
    _apply(root, ["burst"])

    _apply(root, ["warm"])

    stored = _stored(collection)
    archive = build(root / "data" / "simulator", ["warm"], anchor_s=ANCHOR.timestamp())
    moments = sorted(stored)
    expected = archive.samples(
        archive.addresses, [moment.replace(tzinfo=UTC).timestamp() for moment in moments]
    )
    for index, moment in enumerate(moments):
        for address, value in stored[moment].items():
            if address in expected:
                assert value == pytest.approx(expected[address][index], abs=CUT), address
    assert {sample[PRESSURE] for sample in stored.values() if PRESSURE in sample} == {1e-9}
    assert not collection.count_documents({DENSIFIED_FIELD: True, PRESSURE: {"$exists": True}})
