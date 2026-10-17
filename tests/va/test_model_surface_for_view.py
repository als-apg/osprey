"""The model RPC's verbs answered over a simulator view and its composite.

``ModelSurface.for_view`` keys the surface on the view's address set:
``addresses.json`` channels and status addresses are the served side, and a
physics model's own variables are reached as ``<model>/<name>`` through the
composite's ``model_get`` and ``model_set``. These run over a real
:class:`~osprey_connectors.simulation.composite.Composite` built from a
hand-written view with a stub engine, never a mock of the composite.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from lume.model import LUMEModel
from lume.variables import ScalarVariable, Variable

from osprey.services.virtual_accelerator.serving.model_rpc import ModelRpcError
from osprey.services.virtual_accelerator.serving.model_surface import (
    SURFACE_MODEL_ONLY,
    SURFACE_SERVED,
    WRITES_DISABLED,
    ModelSurface,
)
from osprey.services.virtual_accelerator.serving.runner_config import periodic_addresses
from osprey_connectors.simulation import DEFAULT_TICK_S, archive
from osprey_connectors.simulation import composite as composite_module
from osprey_connectors.simulation.composite import Composite
from osprey_connectors.simulation.view import SCHEMAS, SimulatorView
from tests._builds import BuiltProject

TOKEN = "s3cret"
STATUS = "T:SIM:M:STATUS"
CHANNELS = ("M:BPM:X", "M:SP", "T:RB", "T:SP")
MODEL_VARIABLES = ("M/knob", "M/gain")
T0 = 1_760_000_000.0


class StubModel(LUMEModel):
    """One wiring's channels plus two model-only variables, ``knob`` and ``gain``."""

    def __init__(self, wiring: list[Mapping[str, Any]], active: Mapping[str, Any]) -> None:
        self._variables: dict[str, Variable] = {}
        self._defaults: dict[str, float] = {"knob": 1.0}
        for entry in wiring:
            address = str(entry["address"])
            writes = entry.get("direction") == "write"
            self._variables[address] = ScalarVariable(name=address, read_only=not writes)
            self._defaults[address] = float(active.get(address, entry.get("default", 0.0)))
        self._variables["knob"] = ScalarVariable(name="knob", read_only=False)
        self._variables["gain"] = ScalarVariable(name="gain", read_only=True)
        self.inputs: dict[str, float] = {}
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
        return {name: self.inputs.get(name, 2.5) for name in names}


def _build(model, wiring, deck, settings, active=None):
    del model, deck
    if settings and settings.get("fail"):
        raise RuntimeError(settings["fail"])
    return StubModel(list(wiring), active or {})


STUB = SimpleNamespace(build=_build, error_text=lambda exc: " ".join(str(exc).split()))


@pytest.fixture(autouse=True)
def _stub_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    real = Composite._engine

    def engine(name: str) -> Any:
        return STUB if name == "stub" else real(name)

    monkeypatch.setattr(Composite, "_engine", staticmethod(engine))
    monkeypatch.setattr(composite_module, "default_config_path", lambda: None)


def _channel(address: str, owner: str = "texture", **fields: Any) -> dict[str, Any]:
    role = fields.pop("role", "readback")
    return {
        "address": address,
        "role": role,
        "pair": fields.pop("pair", address if role == "setpoint" else None),
        "value_type": "float",
        "unit": fields.pop("unit", None),
        "description": None,
        "writable": fields.pop("writable", False),
        "value_range": fields.pop("value_range", None),
        "owner": owner,
        "on": None,
    }


def _view(path: Path, settings: Mapping[str, Any] | None = None) -> Path:
    channels = [
        _channel("M:BPM:X", "M", unit="mm"),
        _channel("M:SP", "M", role="setpoint", writable=True, value_range=[-5.0, 5.0]),
        _channel("T:RB"),
        _channel("T:SP", role="setpoint", pair="T:RB", writable=True),
    ]
    addresses = {"channels": sorted(c["address"] for c in channels), "status": [STATUS]}
    documents = {
        "served_models.json": {"models": ["M", "texture"]},
        "addresses.json": addresses,
        "variables.json": {
            "code": "T",
            "models": [
                {
                    "name": "M",
                    "engine": "stub",
                    "served": True,
                    "settings": dict(settings or {}),
                    "deck": None,
                    "wiring": [
                        {"id": "1", "address": "M:SP", "direction": "write", "default": 2.0}
                        | {"role": "setpoint", "plane": None, "refresh": "pass"},
                        {"id": "2", "address": "M:BPM:X", "direction": "read", "default": 0.5}
                        | {"role": "monitor", "plane": "x", "refresh": "pass"},
                    ],
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
            "channels": channels,
        },
        "seeds.json": {"seeds": {"T:SP": {"nominal": 5.0}}},
        "scenarios.json": {"scenarios": [{"name": "nominal"}]},
    }
    view = path / "simulator"
    view.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        (view / name).write_text(
            json.dumps({"schema": SCHEMAS[name], **document}), encoding="utf-8"
        )
    return view


def _surface(
    tmp_path: Path,
    *,
    token: str | None = TOKEN,
    settings: Mapping[str, Any] | None = None,
    clock: Iterable[float] = (10.0, 12.5),
    failed_pass_tolerance: int = 3,
) -> tuple[ModelSurface, Composite]:
    view = SimulatorView.open(_view(tmp_path, settings))
    composite = Composite(view, state_dir=tmp_path / "state", clock=lambda: T0)
    surface = ModelSurface.for_view(
        composite,
        view,
        instance="va-1",
        endpoint="va-1:5075",
        model_write_token=token,
        failed_pass_tolerance=failed_pass_tolerance,
        clock=iter(clock).__next__,
    )
    return surface, composite


# -- read verbs ----------------------------------------------------------------


def test_info_lists_the_served_addresses_then_the_model_variables(tmp_path: Path) -> None:
    surface, _ = _surface(tmp_path)

    info = surface.info()

    assert set(info) == {"variables"}
    assert [entry["name"] for entry in info["variables"]] == [*CHANNELS, STATUS, *MODEL_VARIABLES]
    for entry in info["variables"]:
        assert set(entry) == {"name", "unit", "value_range", "read_only", "shape", "surface"}
        assert entry["shape"] is None
    surfaces = {entry["name"]: entry["surface"] for entry in info["variables"]}
    assert surfaces == {
        **dict.fromkeys([*CHANNELS, STATUS], SURFACE_SERVED),
        **dict.fromkeys(MODEL_VARIABLES, SURFACE_MODEL_ONLY),
    }
    by_name = {entry["name"]: entry for entry in info["variables"]}
    assert by_name["M:BPM:X"]["unit"] == "mm"
    assert by_name["M:SP"]["read_only"] is False
    assert by_name[STATUS]["read_only"] is True
    assert by_name["M/knob"]["read_only"] is False
    assert by_name["M/gain"]["read_only"] is True


def test_info_of_a_failed_model_lists_no_model_variable(tmp_path: Path) -> None:
    surface, _ = _surface(tmp_path, settings={"fail": "the deck has no stable orbit"})

    names = [entry["name"] for entry in surface.info()["variables"]]

    assert names == [*CHANNELS, STATUS]


def test_status_carries_the_view_keyset(tmp_path: Path) -> None:
    surface, _ = _surface(tmp_path)
    surface.record_cycle(4.0)
    surface.record_queue_depth(2)

    assert surface.status() == {
        "instance": "va-1",
        "endpoint": "va-1:5075",
        "last_cycle_ms": 4.0,
        "queue_depth": 2,
        "uptime_s": 2.5,
        "last_refused_write": None,
        "last_failed_pass": None,
        "health": {
            "state": "serving",
            "last_pass": None,
            "passes_ok": 0,
            "passes_failed": 0,
            "consecutive_failed": 0,
            "failed_pass_tolerance": 3,
            "last_failed_pass": None,
        },
    }


def test_a_recorded_pass_failure_is_reported_with_its_uptime(tmp_path: Path) -> None:
    surface, _ = _surface(tmp_path, clock=(10.0, 11.5, 14.0))
    surface.record_pass("the deck has no stable orbit")

    status = surface.status()

    assert status["last_failed_pass"] == {"error": "the deck has no stable orbit", "uptime_s": 1.5}
    assert status["uptime_s"] == 4.0


def test_status_answers_from_the_health_record(tmp_path: Path) -> None:
    surface, _ = _surface(tmp_path, clock=(10.0, 11.0, 12.0, 13.0, 14.0), failed_pass_tolerance=1)
    surface.record_pass("first")
    document = surface.record_pass("second")
    surface.record_pass(None)

    status = surface.status()

    assert document["state"] == "failed"
    assert status["health"] == surface.health.document()
    assert status["health"]["state"] == "serving"
    assert (status["health"]["passes_ok"], status["health"]["passes_failed"]) == (1, 2)
    assert status["last_failed_pass"] == status["health"]["last_failed_pass"]
    assert status["last_failed_pass"] == {"error": "second", "uptime_s": 2.0}


def test_get_reads_a_served_address_and_a_model_variable(tmp_path: Path) -> None:
    surface, composite = _surface(tmp_path)

    values = surface.get(["M/knob", "T:SP", "M:SP"])

    assert list(values) == ["M/knob", "T:SP", "M:SP"]
    assert values["M/knob"] == 1.0
    assert values["T:SP"] == composite.get("T:SP") == 5.0
    assert values["M:SP"] == 2.0


def test_get_of_an_unknown_name_carries_the_composite_text(tmp_path: Path) -> None:
    surface, composite = _surface(tmp_path)

    with pytest.raises(ModelRpcError) as caught:
        surface.get(["M/nope"])

    with pytest.raises(ValueError) as expected:
        composite.model_get(["M/nope"])
    assert str(caught.value) == str(expected.value)


def test_diff_iterates_the_view_channels(tmp_path: Path) -> None:
    surface, composite = _surface(tmp_path)
    served = {address: f"served {address}" for address in CHANNELS}

    diff = surface.diff(served.__getitem__)

    assert list(diff) == list(CHANNELS)
    held = composite.held(list(CHANNELS))
    for address in CHANNELS:
        assert diff[address] == {"served": served[address], "truth": held[address]}


# -- write verbs ---------------------------------------------------------------


def test_set_writes_a_model_variable(tmp_path: Path) -> None:
    surface, composite = _surface(tmp_path)

    assert surface.set({"M/knob": 3.0}, TOKEN) == ["M/knob"]

    assert composite.model_get(["M/knob"]) == {"M/knob": 3.0}
    assert surface.status()["last_refused_write"] is None


@pytest.mark.parametrize(
    ("token", "configured", "text"),
    [
        (None, TOKEN, "no write token was presented"),
        ("wrong", TOKEN, "the write token does not match"),
        (TOKEN, None, WRITES_DISABLED),
    ],
    ids=["missing", "wrong", "disabled"],
)
def test_set_without_the_token_is_refused(
    tmp_path: Path, token: str | None, configured: str | None, text: str
) -> None:
    surface, composite = _surface(tmp_path, token=configured)

    with pytest.raises(ModelRpcError, match=text):
        surface.set({"M/knob": 3.0}, token)

    assert composite.model_get(["M/knob"]) == {"M/knob": 1.0}
    assert text in surface.status()["last_refused_write"]


def test_set_of_a_served_address_is_refused(tmp_path: Path) -> None:
    surface, composite = _surface(tmp_path)

    with pytest.raises(ModelRpcError, match="a served address, written through the control"):
        surface.set({"M:SP": 1.0, "M/knob": 3.0}, TOKEN)

    assert composite.model_get(["M/knob"]) == {"M/knob": 1.0}
    assert composite.get("M:SP") == 2.0


def test_set_of_a_non_finite_value_is_refused(tmp_path: Path) -> None:
    surface, composite = _surface(tmp_path)

    with pytest.raises(ModelRpcError, match="not a finite value: M/knob"):
        surface.set({"M/knob": math.inf}, TOKEN)

    assert composite.model_get(["M/knob"]) == {"M/knob": 1.0}


def test_set_the_model_refuses_carries_its_text(tmp_path: Path) -> None:
    surface, composite = _surface(tmp_path)

    with pytest.raises(ModelRpcError, match="knob cannot reach 500.0") as caught:
        surface.set({"M/knob": 500.0}, TOKEN)

    assert surface.status()["last_refused_write"] == str(caught.value)
    assert composite.model_get(["M/knob"]) == {"M/knob": 1.0}


def test_reset_restores_a_model_variable_and_leaves_a_setpoint_as_written(
    tmp_path: Path,
) -> None:
    surface, composite = _surface(tmp_path)
    surface.set({"M/knob": 3.0}, TOKEN)
    composite.set({"M:SP": 4.0})

    assert surface.reset(TOKEN) == ["M/knob"]

    assert composite.model_get(["M/knob"]) == {"M/knob": 1.0}
    assert composite.held(["M:SP"]) == {"M:SP": 4.0}


def test_reset_with_nothing_drifted_writes_nothing(tmp_path: Path) -> None:
    surface, _ = _surface(tmp_path)

    assert surface.reset(TOKEN) == []


def test_reset_without_the_token_is_refused_on_the_token(tmp_path: Path) -> None:
    surface, _ = _surface(tmp_path)

    with pytest.raises(ModelRpcError, match="no write token was presented"):
        surface.reset(None)


# -- a failed model ------------------------------------------------------------


def test_a_failed_model_answers_its_variables_with_its_status(tmp_path: Path) -> None:
    surface, composite = _surface(tmp_path, settings={"fail": "the deck has no stable orbit"})
    expected = "has failed: " + composite.status("M")

    with pytest.raises(ModelRpcError) as read:
        surface.get(["M/knob"])
    with pytest.raises(ModelRpcError) as write:
        surface.set({"M/knob": 3.0}, TOKEN)

    assert expected in str(read.value)
    assert expected in str(write.value)
    assert surface.status()["last_refused_write"] == str(write.value)
    assert surface.get(["T:SP"]) == {"T:SP": 5.0}
    assert math.isnan(surface.get(["M:BPM:X"])["M:BPM:X"])


# -- the SEK waveform on the example tree's view --------------------------------

SEK_WAVEFORM = "SR:MAG:SEPTUM:01:CURRENT:WAVEFORM"
SEK_SP = "SR:MAG:SEPTUM:01:CURRENT:SP"
SEK_ON = "SR:MAG:SEPTUM:01:STATUS:ON"


class _Clock:
    """Epoch seconds a test moves by hand."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _demo_surface(
    built: BuiltProject,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seeds: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[ModelSurface, Composite, _Clock]:
    """The surface over the built example view, every physics child a stub engine.

    The waveform is texture-owned, so the physics engines have no part in it;
    stubbing them keeps the composite fast to build. ``seeds`` replaces those
    addresses' seed records in a copy of the view; the shared build is never
    written.
    """
    monkeypatch.setattr(Composite, "_engine", staticmethod(lambda name: STUB))
    view = SimulatorView.of_render(built.build_dir)
    if seeds:
        copy = tmp_path / "simulator"
        shutil.copytree(view.path, copy)
        document = json.loads((copy / "seeds.json").read_text(encoding="utf-8"))
        document["seeds"].update(seeds)
        (copy / "seeds.json").write_text(json.dumps(document), encoding="utf-8")
        view = SimulatorView.open(copy)
    clock = _Clock(T0)
    composite = Composite(view, state_dir=tmp_path / "state", clock=clock)
    surface = ModelSurface.for_view(
        composite,
        view,
        instance="va-1",
        endpoint="va-1:5075",
        model_write_token=TOKEN,
        failed_pass_tolerance=3,
        clock=iter((10.0, 12.5)).__next__,
    )
    return surface, composite, clock


def _waveform(surface: ModelSurface) -> np.ndarray:
    return np.asarray(surface.get([SEK_WAVEFORM])[SEK_WAVEFORM], dtype=np.float64)


def test_info_describes_the_sek_waveform_as_a_500_element_array(
    built_control_assistant: BuiltProject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    surface, _, _ = _demo_surface(built_control_assistant, tmp_path, monkeypatch)

    by_name = {entry["name"]: entry for entry in surface.info()["variables"]}

    assert by_name[SEK_WAVEFORM]["shape"] == [500]
    assert by_name[SEK_WAVEFORM]["surface"] == SURFACE_SERVED
    assert by_name[SEK_WAVEFORM]["read_only"] is True
    assert by_name[SEK_WAVEFORM]["unit"] == "A"
    assert by_name[SEK_SP]["shape"] is None


def test_two_publishes_one_tick_apart_differ(
    built_control_assistant: BuiltProject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    surface, _, clock = _demo_surface(built_control_assistant, tmp_path, monkeypatch)

    first = _waveform(surface)
    clock.now += DEFAULT_TICK_S
    second = _waveform(surface)

    assert first.shape == second.shape == (500,)
    assert np.all(np.isfinite(first))
    assert first.max() > 0.0
    assert not np.array_equal(first, second)


def test_the_sek_waveform_is_refreshed_by_every_pass_of_the_runner(
    built_control_assistant: BuiltProject,
) -> None:
    """A periodic pass reads the whole roster; a write pass drops only the periodic addresses."""
    view = SimulatorView.of_render(built_control_assistant.build_dir)

    assert SEK_WAVEFORM in view.channels()
    assert SEK_WAVEFORM not in periodic_addresses(view)


def test_writing_the_sek_setpoint_scales_the_next_peak(
    built_control_assistant: BuiltProject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    surface, composite, clock = _demo_surface(built_control_assistant, tmp_path, monkeypatch)
    nominal = float(surface.get([SEK_SP])[SEK_SP])
    clock.now += DEFAULT_TICK_S
    before = _waveform(surface)

    # A Channel Access put of a served setpoint reaches the composite as this
    # set; the model RPC's own ``set`` refuses a served address.
    composite.set({SEK_SP: nominal / 2})
    after = _waveform(surface)

    assert after.max() == pytest.approx(before.max() / 2, rel=1e-9)
    np.testing.assert_allclose(after, before / 2, rtol=1e-9, atol=1e-12)


def test_sek_off_publishes_all_zeros(
    built_control_assistant: BuiltProject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The on status is a read-only readback, so the septum is switched off at its seed."""
    surface, _, clock = _demo_surface(
        built_control_assistant, tmp_path, monkeypatch, seeds={SEK_ON: {"nominal": "FALSE"}}
    )

    first = _waveform(surface)
    clock.now += DEFAULT_TICK_S
    second = _waveform(surface)

    assert first.shape == second.shape == (500,)
    assert np.array_equal(first, np.zeros(500))
    assert np.array_equal(second, np.zeros(500))


# -- a readback gate on a hand-written texture view ------------------------------

GATE_SP = "G:PADDLE:SP"
GATE_RB = "G:PADDLE:RB"
GATED_SUM = "G:BPM:SUM"
GATED_VALID = "G:BPM:STATUS:VALID"
GATED_COUNT = "G:PADDLES:IN"
GATED = (GATED_SUM, GATED_VALID, GATED_COUNT)


def _gate_view(path: Path, *, paddle: str = "FALSE") -> Path:
    """A texture-only view: one paddle whose in-state gates a sum, a valid flag and a count."""
    channels = [
        _channel(GATE_SP, role="setpoint", pair=GATE_RB, writable=True) | {"value_type": "bool"},
        _channel(GATE_RB) | {"value_type": "bool"},
        _channel(GATED_SUM),
        _channel(GATED_VALID) | {"value_type": "bool"},
        _channel(GATED_COUNT) | {"value_type": "int"},
    ]
    inputs = {GATE_RB: "TRUE"}
    documents = {
        "served_models.json": {"models": ["texture"]},
        "addresses.json": {"channels": sorted(c["address"] for c in channels), "status": []},
        "variables.json": {
            "code": "G",
            "models": [
                {
                    "name": "texture",
                    "engine": "texture",
                    "served": True,
                    "settings": {},
                    "deck": None,
                    "wiring": [],
                }
            ],
            "channels": channels,
        },
        "seeds.json": {
            "seeds": {
                GATE_SP: {"nominal": paddle},
                GATED_SUM: {
                    "nominal": 3.0,
                    "noise": {"absolute": 0.001},
                    "gate": {"inputs": inputs, "gated": 0.01},
                },
                GATED_VALID: {"nominal": "TRUE", "gate": {"inputs": inputs, "gated": "FALSE"}},
                GATED_COUNT: {"nominal": 0, "gate": {"inputs": inputs, "gated": 1}},
            }
        },
        "scenarios.json": {"scenarios": [{"name": "nominal"}]},
    }
    view = path / "simulator"
    view.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        (view / name).write_text(
            json.dumps({"schema": SCHEMAS[name], **document}), encoding="utf-8"
        )
    return view


def _gate_surface(tmp_path: Path, **view: str) -> tuple[ModelSurface, Composite, _Clock]:
    opened = SimulatorView.open(_gate_view(tmp_path, **view))
    clock = _Clock(T0)
    composite = Composite(opened, state_dir=tmp_path / "state", clock=clock)
    surface = ModelSurface.for_view(
        composite,
        opened,
        instance="va-1",
        endpoint="va-1:5075",
        model_write_token=TOKEN,
        failed_pass_tolerance=3,
        clock=iter((10.0, 12.5)).__next__,
    )
    return surface, composite, clock


def test_the_surface_reads_the_gated_value_on_the_next_tick(tmp_path: Path) -> None:
    surface, composite, clock = _gate_surface(tmp_path)
    open_sum = surface.get([GATED_SUM])[GATED_SUM]
    assert surface.get([GATED_VALID, GATED_COUNT]) == {GATED_VALID: "TRUE", GATED_COUNT: 0}

    composite.set({GATE_SP: "TRUE"})
    clock.now += DEFAULT_TICK_S
    gated = surface.get(list(GATED))

    assert 0.0 < gated[GATED_SUM] < open_sum / 10
    assert gated[GATED_VALID] == "FALSE"
    assert gated[GATED_COUNT] == 1

    composite.set({GATE_SP: "FALSE"})
    clock.now += DEFAULT_TICK_S
    assert surface.get([GATED_VALID, GATED_COUNT]) == {GATED_VALID: "TRUE", GATED_COUNT: 0}
    assert surface.get([GATED_SUM])[GATED_SUM] == pytest.approx(3.0, abs=0.01)


def test_the_held_value_a_subscriber_compares_carries_the_gate(tmp_path: Path) -> None:
    """An in-process subscription fires when a channel's held value changes."""
    surface, composite, _ = _gate_surface(tmp_path)
    before = composite.held(list(GATED))

    composite.set({GATE_SP: "TRUE"})
    after = composite.held(list(GATED))

    assert before == {GATED_SUM: 3.0, GATED_VALID: "TRUE", GATED_COUNT: 0}
    assert after == {GATED_SUM: pytest.approx(0.03), GATED_VALID: "FALSE", GATED_COUNT: 1}
    diff = surface.diff(lambda address: None)
    assert diff[GATED_VALID]["truth"] == "FALSE"


@pytest.mark.parametrize("paddle", ["FALSE", "TRUE"])
def test_the_seeded_value_at_t_equals_the_live_value_at_t(tmp_path: Path, paddle: str) -> None:
    _, composite, _ = _gate_surface(tmp_path, paddle=paddle)
    live = composite.get(list(GATED))

    seeded = archive.build(tmp_path / "simulator", ["nominal"]).samples(list(GATED), [T0])

    labels = ["FALSE", "TRUE"]
    assert seeded[GATED_SUM] == [pytest.approx(live[GATED_SUM], rel=1e-12)]
    assert seeded[GATED_VALID] == [labels.index(live[GATED_VALID])]
    assert seeded[GATED_COUNT] == [live[GATED_COUNT]]
    assert (live[GATED_VALID] == "FALSE") is (paddle == "TRUE")
    assert seeded[GATED_SUM][0] > 0.0
