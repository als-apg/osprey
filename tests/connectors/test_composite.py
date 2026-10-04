"""The composite over a synthetic simulator view and over the demo's."""

from __future__ import annotations

import ast
import json
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from lume.exceptions import ReadOnlyError
from lume.model import LUMEModel
from lume.variables import ScalarVariable, Variable

from osprey_connectors.simulation import composite as composite_module
from osprey_connectors.simulation import series
from osprey_connectors.simulation.composite import (
    LOG_RECORD_MAX_BYTES,
    STATUS_MAX_BYTES,
    Composite,
)

if TYPE_CHECKING:
    from tests._builds import BuiltProject

REPO_ROOT = Path(__file__).resolve().parents[2]
T0 = 1_760_000_000.0
STATUS = "T:SIM:M:STATUS"


# -- a stub engine -------------------------------------------------------------


class StubModel(LUMEModel):
    """One wiring's channels: settable writes, monitor readings at a held truth."""

    def __init__(self, wiring: list[Mapping[str, Any]], active: Mapping[str, Any]) -> None:
        self.truth: dict[str, float] = {}
        self.monitors: dict[str, dict[str, str]] = {}
        self._variables: dict[str, Variable] = {}
        self._defaults: dict[str, float] = {}
        for entry in wiring:
            address = str(entry["address"])
            if entry.get("direction") == "write":
                self._defaults[address] = float(active.get(address, entry.get("default", 0.0)))
                self._variables[address] = ScalarVariable(name=address, read_only=False)
            else:
                self._variables[address] = ScalarVariable(name=address, read_only=True)
                self.truth[address] = float(entry.get("default", 0.0))
                axis = (entry.get("engine") or {}).get("axis")
                if axis is not None and entry.get("element") is not None:
                    self.monitors.setdefault(str(entry["element"]), {})[axis] = address
                    name = f"{address}/polarity"
                    self._defaults[name] = float(active.get(name, 1.0))
                    self._variables[name] = ScalarVariable(name=name, read_only=False)
        self._defaults["knob"] = 1.0
        self._variables["knob"] = ScalarVariable(name="knob", read_only=False)
        self.inputs: dict[str, float] = {}
        self.reads = 0
        self.reset()

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    def reset(self) -> None:
        self.inputs = dict(self._defaults)

    def _set(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            if float(value) > 100.0:
                raise ValueError(f"{name} cannot reach {value}")
        self.inputs.update({name: float(value) for name, value in values.items()})

    def _get(self, names: list[str]) -> dict[str, Any]:
        self.reads += 1
        if self.inputs.get("knob") == -1.0:
            raise RuntimeError("the solve lost the beam")
        return {name: self.inputs.get(name, self.truth.get(name)) for name in names}


def _stub_build(model, wiring, deck, settings, active=None):
    del model, deck
    if settings and settings.get("fail"):
        raise RuntimeError(settings["fail"])
    return StubModel(list(wiring), active or {})


def _stub_readout(model, values, t_ms):
    del t_ms
    read = dict(values)
    for element, axes in model.monitors.items():
        missing = [address for address in axes.values() if address not in values]
        if len(missing) == len(axes):
            continue
        if missing:
            raise ValueError(f"monitor {element!r} values hold no reading of {missing}")
        for address in axes.values():
            read[address] = values[address] * model.inputs[f"{address}/polarity"]
    return read


STUB = SimpleNamespace(
    build=_stub_build,
    readout=_stub_readout,
    error_text=lambda exc: " ".join(str(exc).split()),
)


@pytest.fixture(autouse=True)
def _stub_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    real = Composite._engine

    def engine(name: str) -> Any:
        return STUB if name == "stub" else real(name)

    monkeypatch.setattr(Composite, "_engine", staticmethod(engine))
    monkeypatch.setattr(composite_module, "default_config_path", lambda: None)


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
        "value_range": fields.pop("value_range", None),
        "owner": owner,
        **fields,
    }


def _wiring() -> list[dict[str, Any]]:
    return [
        {"id": "1", "address": "M:SP", "direction": "write", "default": 2.0},
        {"id": "2", "address": "M:STUCK:SP", "direction": "write", "default": 1.0},
        {"id": "3", "address": "M:BPM:X", "direction": "read", "element": "B1"}
        | {"engine": {"axis": "x"}, "default": 0.25},
        {"id": "4", "address": "M:BPM:Y", "direction": "read", "element": "B1"}
        | {"engine": {"axis": "y"}, "default": -0.5},
        {"id": "5", "address": "M:RB", "direction": "read", "default": 4.0},
        {"id": "6", "address": "M:STATE", "direction": "read", "default": 0},
    ]


def _view(
    path: Path,
    *,
    settings: Mapping[str, Any] | None = None,
    engine: str = "stub",
    scenarios: list[dict[str, Any]] | None = None,
) -> Path:
    channels = [
        _channel("M:SP", "M", role="setpoint", writable=True),
        _channel("M:STUCK:SP", "M", role="setpoint", writable=True),
        _channel("M:BPM:X", "M"),
        _channel("M:BPM:Y", "M"),
        _channel("M:RB", "M"),
        _channel("M:STATE", "M", value_type="enum", options=["OFF", "ON"]),
        _channel("T:SP", role="setpoint", pair="T:RB", writable=True),
        _channel("T:RB"),
        _channel("T:STEP:SP", role="setpoint", pair="T:STEP:RB", writable=True),
        _channel("T:STEP:RB", value_type="int"),
        _channel("T:NOISY"),
        _channel("T:TRACE", role="setpoint", value_type="waveform", shape=[2, 2], writable=True),
        _channel("T:LOCKED", role="setpoint"),
    ]
    documents = {
        "served_models.json": {"models": ["M", "texture"]},
        "addresses.json": {
            "channels": sorted(channel["address"] for channel in channels),
            "status": [STATUS],
        },
        "variables.json": {
            "code": "T",
            "models": [
                {
                    "name": "M",
                    "engine": engine,
                    "served": True,
                    "settings": dict(settings or {}),
                    "deck": None,
                    "wiring": _wiring(),
                },
                {
                    "name": "texture",
                    "engine": "texture",
                    "served": True,
                    "settings": {},
                    "deck": None,
                    "wiring": [],
                },
            ],
            "channels": sorted(channels, key=lambda channel: channel["address"]),
        },
        "seeds.json": {
            "seeds": {
                "T:SP": {"nominal": 5.0},
                "T:NOISY": {"nominal": 10.0, "noise": 0.5},
                "M:RB": {"noise": 0.1},
                "M:BPM:X": {"drift": {"amplitude": 0.01, "period_s": 600}},
                "M:BPM:Y": {"drift": {"amplitude": 0.01, "period_s": 600}},
                "T:STEP:SP": {"nominal": 3.0},
            }
        },
        "scenarios.json": {"scenarios": [{"name": "nominal"}, *(scenarios or [])]},
    }
    view = path / "simulator"
    view.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        (view / name).write_text(json.dumps(document), encoding="utf-8")
    return view


def _activate(state: Path, *names: str) -> None:
    state.mkdir(parents=True, exist_ok=True)
    target = state / "active_scenarios"
    before = target.stat().st_mtime_ns if target.exists() else 0
    target.write_text("\n".join(names or ("nominal",)) + "\n", encoding="utf-8")
    os.utime(target, ns=(before + 1_000_000_000, before + 1_000_000_000))


def _composite(tmp_path: Path, t_s: float = T0, **view: Any) -> Composite:
    return Composite(_view(tmp_path, **view), state_dir=tmp_path / "state", clock=lambda: t_s)


# -- the variables -------------------------------------------------------------


def test_supported_variables_are_the_channels_and_the_status(tmp_path: Path) -> None:
    composite = _composite(tmp_path)
    addresses = json.loads((tmp_path / "simulator" / "addresses.json").read_text())

    assert set(composite.supported_variables) == {*addresses["channels"], *addresses["status"]}
    assert composite.get(STATUS) == "ok"
    assert composite.models == ["M"]


def test_a_childs_own_variables_are_reached_through_model_get_and_set(tmp_path: Path) -> None:
    composite = _composite(tmp_path)

    assert "knob" not in composite.supported_variables
    assert "M/knob" not in composite.supported_variables
    assert composite.model_get(["M/knob"]) == {"M/knob": 1.0}
    composite.model_set({"M/knob": 3.0})
    assert composite.model_get(["M/knob"]) == {"M/knob": 3.0}
    with pytest.raises(ValueError, match="no variable"):
        composite.model_get(["M/M:SP"])
    with pytest.raises(ValueError):
        composite.get("M/knob")


def _model_set_callers(path: Path) -> set[str]:
    """The functions of ``path`` that call ``model_set``, ``<module>`` at top level."""
    callers: set[str] = set()

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                visit(child, child.name)
                continue
            if isinstance(child, ast.Call):
                func = child.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if name == "model_set":
                    callers.add(scope)
            visit(child, scope)

    visit(ast.parse(path.read_text(encoding="utf-8")), "<module>")
    return callers


COMPOSITE_SOURCE = "packages/osprey-connectors/src/osprey_connectors/simulation/composite.py"


def test_model_set_is_called_only_by_the_model_surface_and_the_rebuild() -> None:
    allowed = {"src/osprey/services/virtual_accelerator/serving/model_surface.py", COMPOSITE_SOURCE}
    callers = {
        path.relative_to(REPO_ROOT).as_posix(): found
        for root in ("src", "packages")
        for path in (REPO_ROOT / root).rglob("*.py")
        if (found := _model_set_callers(path))
    }

    assert set(callers) <= allowed
    assert callers.get(COMPOSITE_SOURCE, set()) <= {"_rebuild"}


# -- reads ---------------------------------------------------------------------


def test_a_physics_readback_is_truth_plus_motion_then_clamped(tmp_path: Path) -> None:
    composite = _composite(tmp_path)
    noise = series.keyed_normals(series.channel_key_bytes("M:RB"), np.array([round(T0 * 1000)]))

    assert composite.get("M:RB") == pytest.approx(4.0 + 0.1 * noise[0])
    assert composite.held(["M:RB"]) == {"M:RB": 4.0}


def test_a_monitor_read_on_one_plane_reads_its_partner_through_the_readout(
    tmp_path: Path,
) -> None:
    composite = _composite(tmp_path)

    both = composite.get(["M:BPM:X", "M:BPM:Y"])

    assert composite.get("M:BPM:Y") == both["M:BPM:Y"]
    assert composite.get("M:BPM:X") == both["M:BPM:X"]


def test_a_texture_waveform_is_set_nested_and_reads_flat(tmp_path: Path) -> None:
    composite = _composite(tmp_path)

    composite.set({"T:TRACE": [[1, 2], [3, 4]]})

    assert composite.get("T:TRACE") == [1.0, 2.0, 3.0, 4.0]
    assert composite.held(["T:TRACE"]) == {"T:TRACE": [1.0, 2.0, 3.0, 4.0]}


# -- writes --------------------------------------------------------------------


def test_a_set_reaches_the_physics_child_and_the_texture(tmp_path: Path) -> None:
    composite = _composite(tmp_path)

    composite.set({"M:SP": 7, "T:SP": 6.5})

    assert composite.held(["M:SP", "T:SP", "T:RB"]) == {"M:SP": 7.0, "T:SP": 6.5, "T:RB": 6.5}


def test_a_texture_refusal_restores_the_physics_childs_inputs(tmp_path: Path) -> None:
    composite = _composite(tmp_path)

    with pytest.raises(ValueError, match="int"):
        composite.set({"M:SP": 9.0, "T:STEP:SP": 3.5})

    assert composite.held(["M:SP", "T:STEP:SP"]) == {"M:SP": 2.0, "T:STEP:SP": 3.0}


def test_an_engine_refusal_fails_the_write_only(tmp_path: Path) -> None:
    composite = _composite(tmp_path)

    with pytest.raises(ValueError, match="cannot reach"):
        composite.set({"M:SP": 500.0})

    assert composite.held(["M:SP"]) == {"M:SP": 2.0}
    assert composite.status("M") == "ok"


def test_a_read_only_channel_and_a_status_are_refused(tmp_path: Path) -> None:
    composite = _composite(tmp_path)

    with pytest.raises(ReadOnlyError):
        composite.set({"T:LOCKED": 1.0})
    with pytest.raises(ReadOnlyError):
        composite.set({STATUS: "fine"})
    with pytest.raises(ValueError, match="not a valid float"):
        composite.set({"M:SP": "high"})


def test_an_empty_set_reaches_no_child(tmp_path: Path) -> None:
    composite = _composite(tmp_path)
    child = composite._children["M"].model
    assert child is not None
    reads = child.reads

    composite.set({})

    assert child.inputs["M:SP"] == 2.0
    assert child.reads == reads


def test_reset_returns_every_child_to_its_start(tmp_path: Path) -> None:
    composite = _composite(tmp_path)
    composite.set({"M:SP": 8.0, "T:SP": 1.0})

    composite.reset()

    assert composite.held(["M:SP", "T:SP"]) == {"M:SP": 2.0, "T:SP": 5.0}


# -- a failed child ------------------------------------------------------------


def test_a_child_whose_build_raises_is_failed_and_never_raises(tmp_path: Path) -> None:
    composite = _composite(tmp_path, settings={"fail": "the deck has no stable orbit"})

    assert composite.get(STATUS) == "the deck has no stable orbit"
    assert math.isnan(composite.get("M:RB"))
    assert composite.get("M:STATE") == "OFF"
    assert composite.get("M:SP") == 2.0
    composite.set({"M:SP": 9.0})
    assert composite.get("M:SP") == 9.0
    assert composite.output_severity(["M:RB", "M:SP", "T:SP", STATUS]) == {
        "M:RB": {"condition": "udf"},
        "M:SP": {"condition": "udf"},
    }
    assert composite.get("T:SP") == 5.0


def test_a_child_whose_read_raises_fails_with_the_engine_text(tmp_path: Path) -> None:
    composite = _composite(tmp_path)
    composite.get("M:STATE")
    composite.model_set({"M/knob": -1.0})

    assert math.isnan(composite.get("M:RB"))
    assert composite.status("M") == "the solve lost the beam"
    composite.reset()
    assert composite.status("M") == "ok"


def test_an_unregistered_engine_is_a_failed_child_naming_it(tmp_path: Path) -> None:
    composite = _composite(tmp_path, engine="no-such-engine")

    assert "no-such-engine" in composite.status("M")


def test_a_long_engine_text_is_capped_in_utf8_bytes(tmp_path: Path) -> None:
    message = "Ä" * 1024
    composite = _composite(tmp_path, settings={"fail": message})

    status = composite.status("M")

    assert len(status.encode("utf-8")) <= STATUS_MAX_BYTES
    assert status.endswith("…")
    assert message.startswith(status[:-1])


# -- the model log -------------------------------------------------------------


def test_without_a_loaded_config_no_log_file_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    _composite(tmp_path, settings={"fail": "broken"})

    assert not list(tmp_path.rglob("*.log"))


def test_the_log_appends_one_line_per_record_widened_past_the_umask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "build").mkdir()
    config = tmp_path / "build" / "config.yml"
    monkeypatch.setattr(composite_module, "default_config_path", lambda: str(config))
    previous = os.umask(0o022)
    try:
        _composite(tmp_path, settings={"fail": "x" * 6000})
        _composite(tmp_path, settings={"fail": "broken"})
    finally:
        os.umask(previous)
    log = tmp_path / "var" / "simulator" / "M.log"

    raw = log.read_bytes().splitlines(keepends=True)
    records = [json.loads(line) for line in raw]
    assert stat.S_IMODE(log.stat().st_mode) == 0o664
    assert [record["error"][:6] for record in records] == ["xxxxxx", "broken"]
    assert all(len(line) < LOG_RECORD_MAX_BYTES for line in raw)
    assert {(record["instance"], record["pid"]) for record in records} == {
        ("inprocess", os.getpid())
    }
    assert not (tmp_path / "var" / "simulator" / "texture.log").exists()


# -- the active scenarios ------------------------------------------------------


def test_a_scenario_change_rebuilds_the_children_at_its_writes(tmp_path: Path) -> None:
    scenarios = [
        {
            "name": "flip",
            "overrides": {"M:SP": 6.0, "T:SP": 9.0},
            "faults": {"M": {"writes": {"M:BPM:Y": {"polarity": -1}, "M:STUCK:SP": "stuck"}}},
        }
    ]
    composite = _composite(tmp_path, scenarios=scenarios)
    plain = composite.get("M:BPM:Y")
    composite.set({"M:SP": 1.0})

    _activate(tmp_path / "state", "flip")

    assert composite.get("M:BPM:Y") == pytest.approx(-plain)
    assert composite.held(["M:SP", "T:SP", "T:RB"]) == {"M:SP": 6.0, "T:SP": 9.0, "T:RB": 9.0}
    composite.set({"M:STUCK:SP": 50.0})
    assert composite.held(["M:STUCK:SP"]) == {"M:STUCK:SP": 1.0}
    composite.set({"M:SP": 3.0})
    composite.reset()
    assert composite.held(["M:SP"]) == {"M:SP": 6.0}
    _activate(tmp_path / "state")
    assert composite.get("M:BPM:Y") == pytest.approx(plain)
    assert composite.held(["M:SP"]) == {"M:SP": 2.0}


def test_a_failed_child_is_rebuilt_on_a_scenario_change(tmp_path: Path) -> None:
    scenarios = [{"name": "cooler", "overrides": {"T:SP": 4.0}}]
    composite = _composite(tmp_path, scenarios=scenarios)
    composite.model_set({"M/knob": -1.0})
    composite.get("M:RB")
    assert composite.status("M") != "ok"

    _activate(tmp_path / "state", "cooler")

    assert composite.status("M") == "ok"


def test_scenarios_touching_one_target_twice_serve_without_them(tmp_path: Path) -> None:
    scenarios = [
        {"name": "a", "overrides": {"T:SP": 1.0}},
        {"name": "b", "overrides": {"T:SP": 2.0}},
    ]
    composite = _composite(tmp_path, scenarios=scenarios)

    _activate(tmp_path / "state", "a", "b")

    assert composite.held(["T:SP"]) == {"T:SP": 5.0}


def _driver(t_s: float) -> float:
    return float(series.wander(series.driver_key_bytes("d"), np.array([t_s]), 1.0, 300.0)[0])


def test_a_scenarios_drivers_move_the_readbacks_it_couples(tmp_path: Path) -> None:
    scenarios = [
        {
            "name": "thermal",
            "drivers": {"d": {"kind": "wander", "amplitude": 1.0, "period_s": 300}},
            "couple": {
                "T:RB": [{"driver": "d", "gain": 0.5}],
                "M:RB": [{"driver": "d", "gain": 2.0}],
            },
            "noise": {"M:RB": {"noise": 0.0, "noise_abs": 0.0}},
        }
    ]
    composite = _composite(tmp_path, scenarios=scenarios)
    before = composite.get(["T:RB", "M:RB"])

    _activate(tmp_path / "state", "thermal")

    assert composite.get(["T:RB", "M:RB"]) == pytest.approx(
        {"T:RB": 5.0 + 0.5 * _driver(T0), "M:RB": 4.0 + 2.0 * _driver(T0)}
    )
    _activate(tmp_path / "state")
    assert composite.get(["T:RB", "M:RB"]) == before


# -- the demo's simulator view -------------------------------------------------


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
def test_bpm_polarity_negates_the_demo_monitor_on_both_planes(
    demo_view: Path, tmp_path: Path
) -> None:
    y = "SR:DIAG:BPM:17:POSITION:Y"
    x = "SR:DIAG:BPM:17:POSITION:X"
    plain = Composite(demo_view, clock=lambda: T0)
    _activate(tmp_path / "state", "bpm-polarity")
    faulted = Composite(demo_view, state_dir=tmp_path / "state", clock=lambda: T0)

    alone = faulted.get(y)
    both = faulted.get([x, y])

    assert faulted.status("SR") == "ok"
    assert plain.held([y])[y] == pytest.approx(0.0, abs=1e-12)
    assert plain.get(y) != 0.0
    assert alone == both[y]
    assert alone == pytest.approx(-plain.get(y))
    assert both[x] == pytest.approx(-plain.get(x))
