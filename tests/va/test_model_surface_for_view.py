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
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from lume.model import LUMEModel
from lume.variables import ScalarVariable, Variable

from osprey.services.virtual_accelerator.serving.model_rpc import ModelRpcError
from osprey.services.virtual_accelerator.serving.model_surface import (
    SURFACE_SERVED,
    WRITES_DISABLED,
    ModelSurface,
)
from osprey_connectors.simulation import composite as composite_module
from osprey_connectors.simulation.composite import Composite

TOKEN = "s3cret"
STATUS = "T:SIM:M:STATUS"
CHANNELS = ("M:BPM:X", "M:SP", "T:RB", "T:SP")
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
                        {"id": "1", "address": "M:SP", "direction": "write", "default": 2.0},
                        {"id": "2", "address": "M:BPM:X", "direction": "read", "default": 0.5},
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
        (view / name).write_text(json.dumps(document), encoding="utf-8")
    return view, addresses


def _surface(
    tmp_path: Path, *, token: str | None = TOKEN, settings: Mapping[str, Any] | None = None
) -> tuple[ModelSurface, Composite]:
    view, addresses = _view(tmp_path, settings)
    composite = Composite(view, state_dir=tmp_path / "state", clock=lambda: T0)
    surface = ModelSurface.for_view(
        composite,
        addresses,
        instance="va-1",
        endpoint="va-1:5075",
        model_write_token=token,
        clock=iter([10.0, 12.5]).__next__,
    )
    return surface, composite


# -- read verbs ----------------------------------------------------------------


def test_info_lists_every_served_address_and_the_status(tmp_path: Path) -> None:
    surface, _ = _surface(tmp_path)

    info = surface.info()

    assert set(info) == {"variables"}
    assert [entry["name"] for entry in info["variables"]] == [*CHANNELS, STATUS]
    for entry in info["variables"]:
        assert set(entry) == {"name", "unit", "value_range", "read_only", "surface"}
        assert entry["surface"] == SURFACE_SERVED
    by_name = {entry["name"]: entry for entry in info["variables"]}
    assert by_name["M:BPM:X"]["unit"] == "mm"
    assert by_name["M:SP"]["read_only"] is False
    assert by_name[STATUS]["read_only"] is True


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
    }


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


def test_reset_is_refused_naming_why(tmp_path: Path) -> None:
    surface, composite = _surface(tmp_path)
    surface.set({"M/knob": 3.0}, TOKEN)

    with pytest.raises(ModelRpcError, match="reset"):
        surface.reset(TOKEN)

    assert composite.model_get(["M/knob"]) == {"M/knob": 3.0}


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
