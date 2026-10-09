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
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
from osprey_connectors.simulation import composite as composite_module
from osprey_connectors.simulation.composite import Composite
from osprey_connectors.simulation.view import SCHEMAS

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


def _view(path: Path, settings: Mapping[str, Any] | None = None) -> tuple[Path, dict[str, Any]]:
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
    return view, addresses


def _surface(
    tmp_path: Path,
    *,
    token: str | None = TOKEN,
    settings: Mapping[str, Any] | None = None,
    clock: Iterable[float] = (10.0, 12.5),
    failed_pass_tolerance: int = 3,
) -> tuple[ModelSurface, Composite]:
    view, addresses = _view(tmp_path, settings)
    composite = Composite(view, state_dir=tmp_path / "state", clock=lambda: T0)
    surface = ModelSurface.for_view(
        composite,
        addresses,
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
        assert set(entry) == {"name", "unit", "value_range", "read_only", "surface"}
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
