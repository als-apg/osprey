"""The archive composite: a simulator view's history at one active set's start state."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from lume.model import LUMEModel
from lume.variables import ScalarVariable, Variable

from osprey_connectors.simulation import composite as composite_module
from osprey_connectors.simulation import series
from osprey_connectors.simulation.archive import build
from osprey_connectors.simulation.composite import Composite

if TYPE_CHECKING:
    from tests._builds import BuiltProject

T0 = 1_760_000_000.0
HOUR = np.arange(T0, T0 + 3600.0, 1.0)
BPM_X = "SR:DIAG:BPM:01:POSITION:X"


# -- a stub engine that counts its builds and its reads ------------------------


class CountingModel(LUMEModel):
    """Writes held as inputs, readbacks at their wiring default; every read is counted."""

    def __init__(self, wiring: list[Mapping[str, Any]], active: Mapping[str, Any]) -> None:
        self._variables: dict[str, Variable] = {}
        self._defaults: dict[str, float] = {}
        for entry in wiring:
            address = str(entry["address"])
            writable = entry.get("direction") == "write"
            self._variables[address] = ScalarVariable(name=address, read_only=not writable)
            self._defaults[address] = float(active.get(address, entry.get("default", 0.0)))
        self.inputs: dict[str, float] = {}
        self.reads = 0
        self.reset()

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    def reset(self) -> None:
        self.inputs = dict(self._defaults)

    def _set(self, values: dict[str, Any]) -> None:
        self.inputs.update({name: float(value) for name, value in values.items()})

    def _get(self, names: list[str]) -> dict[str, Any]:
        self.reads += 1
        return {name: self.inputs[name] for name in names}


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """The ``stub`` engine; ``models`` holds every model it built."""
    models: list[CountingModel] = []

    def build_model(model, wiring, deck, settings, active=None):
        del model, deck, settings
        built = CountingModel(list(wiring), active or {})
        models.append(built)
        return built

    stub = SimpleNamespace(build=build_model, models=models)
    real = Composite._engine
    monkeypatch.setattr(
        Composite, "_engine", staticmethod(lambda name: stub if name == "stub" else real(name))
    )
    monkeypatch.setattr(composite_module, "default_config_path", lambda: None)
    return stub


# -- a synthetic view ----------------------------------------------------------


def _channel(address: str, owner: str = "texture", **fields: Any) -> dict[str, Any]:
    role = fields.pop("role", "readback")
    return {
        "address": address,
        "role": role,
        "pair": fields.pop("pair", address if role == "setpoint" else None),
        "value_type": fields.pop("value_type", "float"),
        "unit": None,
        "description": None,
        "writable": fields.pop("writable", False),
        "value_range": None,
        "owner": owner,
        **fields,
    }


def _view(
    path: Path,
    *,
    scenarios: list[dict[str, Any]] | None = None,
    seeds: Mapping[str, Any] | None = None,
) -> Path:
    channels = [
        _channel("M:SP", "M", role="setpoint", writable=True),
        _channel("M:RB", "M"),
        _channel("T:SP", role="setpoint", pair="T:RB", writable=True),
        _channel("T:RB"),
        _channel("T:NOISY"),
        _channel("T:MODE", value_type="enum", options=["OFF", "STANDBY", "ON"]),
        _channel("T:FLAG", value_type="bool"),
    ]
    documents = {
        "served_models.json": {"models": ["M", "texture"]},
        "addresses.json": {
            "channels": sorted(channel["address"] for channel in channels),
            "status": ["T:SIM:M:STATUS"],
        },
        "variables.json": {
            "code": "T",
            "models": [
                {
                    "name": "M",
                    "engine": "stub",
                    "served": True,
                    "settings": {},
                    "deck": None,
                    "wiring": [
                        {"id": "1", "address": "M:SP", "direction": "write", "default": 2.0},
                        {"id": "2", "address": "M:RB", "direction": "read", "default": 4.0},
                    ],
                },
            ],
            "channels": sorted(channels, key=lambda channel: channel["address"]),
        },
        "seeds.json": {
            "seeds": {
                "T:SP": {"nominal": 5.0},
                "T:NOISY": {"nominal": 10.0, "noise": 0.5},
                "M:RB": {"noise": 0.1, "drift": {"amplitude": 0.2, "period_s": 600}},
                "T:MODE": {"nominal": "ON"},
                "T:FLAG": {"nominal": "TRUE"},
                **(seeds or {}),
            }
        },
        "scenarios.json": {"scenarios": [{"name": "nominal"}, *(scenarios or [])]},
    }
    view = path / "simulator"
    view.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        (view / name).write_text(json.dumps(document), encoding="utf-8")
    return view


# -- one read per active set ---------------------------------------------------


def test_the_composite_is_built_and_read_once_per_active_set(
    tmp_path: Path, engine: SimpleNamespace
) -> None:
    archive = build(_view(tmp_path), [])

    for address in archive.addresses:
        archive.series(address, HOUR)
        archive.series(address, HOUR + 7200.0)

    assert len(engine.models) == 1
    assert engine.models[0].reads == 1


def test_an_instance_other_than_the_virtual_accelerator_is_refused(
    tmp_path: Path, engine: SimpleNamespace
) -> None:
    del engine
    with pytest.raises(ValueError, match="virtual_accelerator"):
        build(_view(tmp_path), [], instance="live_standin")


def test_an_address_outside_the_view_is_refused(tmp_path: Path, engine: SimpleNamespace) -> None:
    del engine
    archive = build(_view(tmp_path), [])

    with pytest.raises(KeyError, match="NOT:A:CHANNEL is not a channel of the simulator view"):
        archive.series("NOT:A:CHANNEL", HOUR)


# -- the samples ---------------------------------------------------------------


def test_a_texture_channel_archives_what_the_texture_serves_at_each_instant(
    tmp_path: Path, engine: SimpleNamespace
) -> None:
    del engine
    view = _view(tmp_path)
    archive = build(view, [])
    times = HOUR[:50]

    archived = archive.series("T:NOISY", times)
    served = [Composite(view, clock=lambda t=t: float(t)).get("T:NOISY") for t in times]

    assert archived == served


def test_a_physics_readback_carries_drift_and_keyed_noise_on_its_held_value(
    tmp_path: Path, engine: SimpleNamespace
) -> None:
    del engine
    archive = build(_view(tmp_path), [])
    key = series.channel_key_bytes("M:RB")
    expected = (
        4.0
        + series.wander(key, HOUR, 0.2, 600.0)
        + 0.1 * series.keyed_normals(key, np.rint(HOUR * 1000.0).astype(np.int64))
    )

    assert archive.held("M:RB") == 4.0
    assert archive.series("M:RB", HOUR) == pytest.approx(expected.tolist(), rel=1e-12)


def test_the_active_writes_are_the_start_state(tmp_path: Path, engine: SimpleNamespace) -> None:
    del engine
    view = _view(tmp_path, scenarios=[{"name": "lift", "overrides": {"M:SP": 7.0, "T:SP": 9.0}}])

    archive = build(view, ["lift"])

    assert archive.series("M:SP", HOUR[:3]) == [7.0, 7.0, 7.0]
    assert archive.series("T:RB", HOUR[:3]) == [9.0, 9.0, 9.0]


def test_a_relative_noise_replacement_scales_with_the_held_value(
    tmp_path: Path, engine: SimpleNamespace
) -> None:
    del engine
    noisy = [{"name": "noisy", "noise": {"T:NOISY": {"noise": 0.01}}}]
    times = HOUR[:200]
    spreads = {}
    for nominal in (10.0, 40.0):
        view = _view(
            tmp_path / str(nominal), scenarios=noisy, seeds={"T:NOISY": {"nominal": nominal}}
        )
        archived = np.asarray(build(view, ["noisy"]).series("T:NOISY", times))
        spreads[nominal] = float(np.std(archived))

    assert spreads[10.0] > 0.0
    assert spreads[40.0] == pytest.approx(4.0 * spreads[10.0], rel=1e-9)


def test_a_relative_noise_replacement_matches_the_texture_sample_for_sample(
    tmp_path: Path, engine: SimpleNamespace
) -> None:
    del engine
    view = _view(tmp_path, scenarios=[{"name": "noisy", "noise": {"T:NOISY": {"noise": 0.01}}}])
    state = tmp_path / "state"
    state.mkdir()
    (state / "active_scenarios").write_text("nominal\nnoisy\n", encoding="utf-8")
    times = HOUR[:50]

    archived = build(view, ["noisy"]).series("T:NOISY", times)
    served = [
        Composite(view, state_dir=state, clock=lambda t=t: float(t)).get("T:NOISY") for t in times
    ]

    assert archived == served


def test_an_archiver_event_moves_the_level_the_motion_rides_on(
    tmp_path: Path, engine: SimpleNamespace
) -> None:
    del engine
    spike = {"shape": "spike", "at_offset": -1800.0, "amplitude": 50.0, "width": 60.0}
    view = _view(
        tmp_path,
        scenarios=[{"name": "burst", "archiver": [{"channel": "T:RB", "events": [spike]}]}],
    )

    archive = build(view, ["burst"], anchor_s=T0 + 3600.0)
    samples = np.asarray(archive.series("T:RB", HOUR))

    assert samples[1800] == pytest.approx(55.0)
    assert samples[0] == pytest.approx(5.0)
    assert int(np.argmax(samples)) == 1800


def test_a_step_event_on_an_enum_channel_archives_option_indices(
    tmp_path: Path, engine: SimpleNamespace
) -> None:
    del engine
    step = {"shape": "step", "at_offset": -10.0, "to": "STANDBY"}
    view = _view(
        tmp_path,
        scenarios=[{"name": "trip", "archiver": [{"channel": "T:MODE", "events": [step]}]}],
    )

    archive = build(view, ["trip"], anchor_s=T0 + 20.0)

    assert archive.series("T:MODE", HOUR[:12]) == [2] * 10 + [1] * 2
    assert archive.series("T:FLAG", HOUR[:2]) == [1, 1]


def test_scenarios_writing_one_target_twice_are_archived_without_them(
    tmp_path: Path, engine: SimpleNamespace
) -> None:
    del engine
    spike = {"shape": "spike", "at_offset": 0.0, "amplitude": 50.0, "width": 60.0}
    view = _view(
        tmp_path,
        scenarios=[
            {
                "name": "a",
                "overrides": {"T:SP": 9.0},
                "archiver": [{"channel": "T:RB", "events": [spike]}],
            },
            {"name": "b", "overrides": {"T:SP": 8.0}},
        ],
    )

    archive = build(view, ["a", "b"], anchor_s=T0)

    assert archive.series("T:RB", [T0]) == [5.0]


def test_a_journal_planted_before_a_replay_gives_identical_samples(
    tmp_path: Path, engine: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    del engine
    view = _view(tmp_path)
    before = build(view, []).series("T:RB", HOUR[:20])
    planted: list[Path] = []

    class PlantedComposite(Composite):
        """Plants a session write of ``T:SP`` in the state directory it is handed."""

        def __init__(self, view_dir: Path | str, **kwargs: Any) -> None:
            state = Path(kwargs["state_dir"])
            active = (state / "active_scenarios").read_bytes()
            journal = state / "mock" / "writes.json"
            journal.parent.mkdir(parents=True)
            journal.write_text(
                json.dumps(
                    {
                        "active_set_sha256": hashlib.sha256(active).hexdigest(),
                        "seq": 1,
                        "writes": [[1, "T:SP", 1.0]],
                    }
                ),
                encoding="utf-8",
            )
            planted.append(journal)
            super().__init__(view_dir, **kwargs)

    monkeypatch.setattr(composite_module, "Composite", PlantedComposite)

    after = build(view, []).series("T:RB", HOUR[:20])

    assert planted
    assert after == before == [5.0] * 20


# -- what the served composite reads -------------------------------------------


def _flip(model: Any, readings: Mapping[str, float], t_ms: int) -> dict[str, float]:
    """A readout that reports every reading with its sign flipped."""
    del model, t_ms
    return {address: -value for address, value in readings.items()}


def test_a_physics_readback_archives_the_engine_readout_the_composite_serves(
    tmp_path: Path, engine: SimpleNamespace
) -> None:
    engine.readout = _flip
    view = _view(tmp_path)
    times = HOUR[:50]

    archived = build(view, []).series("M:RB", times)
    served = [Composite(view, clock=lambda t=t: float(t)).get("M:RB") for t in times]

    assert archived == served
    assert all(sample < 0.0 for sample in archived)


def test_a_physics_setpoint_archives_its_held_value_without_motion(
    tmp_path: Path, engine: SimpleNamespace
) -> None:
    del engine
    view = _view(
        tmp_path, seeds={"M:SP": {"noise": 0.1, "drift": {"amplitude": 1.0, "period_s": 60}}}
    )
    times = HOUR[:20]

    archived = build(view, []).series("M:SP", times)
    served = [Composite(view, clock=lambda t=t: float(t)).get("M:SP") for t in times]

    assert archived == served == [2.0] * 20


def test_a_physics_model_that_fails_to_build_stops_the_archive(
    tmp_path: Path, engine: SimpleNamespace
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("deck unreadable")

    engine.build = refuse

    with pytest.raises(RuntimeError, match="M: deck unreadable"):
        build(_view(tmp_path), [])


def test_building_an_archive_appends_nothing_to_the_model_log(
    tmp_path: Path, engine: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    del engine
    logs = tmp_path / "var" / "simulator"
    monkeypatch.setattr(composite_module, "log_dir", lambda: logs)
    view = _view(
        tmp_path,
        scenarios=[
            {"name": "a", "overrides": {"T:SP": 9.0}},
            {"name": "b", "overrides": {"T:SP": 8.0}},
        ],
    )

    build(view, ["a", "b"]).series("M:RB", HOUR[:3])

    assert not logs.exists()


def test_importing_the_archive_module_loads_no_lume() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, osprey_connectors.simulation.archive;"
            "assert not [m for m in sys.modules if m == 'lume' or m.startswith('lume.')];"
            "assert 'osprey_connectors.simulation.composite' not in sys.modules;"
            "print('CLEAN')",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "CLEAN"


# -- the demo's monitor ----------------------------------------------------------


@pytest.fixture
def demo_view(built_control_assistant: BuiltProject, tmp_path: Path) -> Path:
    prefix = "data/simulator/"
    for name, data in built_control_assistant.outputs[0].files.items():
        if name.startswith(prefix):
            target = tmp_path / "simulator" / name[len(prefix) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return tmp_path / "simulator"


@pytest.mark.slow
def test_an_hour_of_a_quiet_monitor_stays_on_its_held_value_inside_its_clamp(
    demo_view: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(composite_module, "default_config_path", lambda: None)
    seeds_path = demo_view / "seeds.json"
    seeds = json.loads(seeds_path.read_text(encoding="utf-8"))
    record = seeds["seeds"][BPM_X]
    sigma = float(record["noise"])
    drift = series.wander(
        series.channel_key_bytes(BPM_X),
        HOUR,
        float(record["drift"]["amplitude"]),
        float(record["drift"]["period_s"]),
    )
    max_drift = float(np.max(np.abs(drift)))
    held = float(build(demo_view, []).held(BPM_X))
    band = max_drift + 10.0 * sigma
    record["clamp"] = [held - band, held + band]
    seeds_path.write_text(json.dumps(seeds), encoding="utf-8")

    archive = build(demo_view, [])
    samples = np.asarray(archive.series(BPM_X, HOUR))

    assert float(archive.held(BPM_X)) == held
    assert abs(float(np.mean(samples)) - held) <= 4.0 * sigma / np.sqrt(len(samples)) + max_drift
    assert np.all((samples >= held - band) & (samples <= held + band))
