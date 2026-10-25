"""A command is written with channels of its own model only.

The composite restores a child by writing its previous inputs back, which
cannot take back what a command did. A batch that carries a command beside a
channel of another owner is therefore refused before any child is written.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from lume.model import LUMEModel
from lume.variables import ScalarVariable, Variable

from osprey_connectors.simulation import composite as composite_module
from osprey_connectors.simulation.composite import Composite, WriteRefusedError
from osprey_connectors.simulation.view import SCHEMAS

T0 = 1_760_000_000.0
REFUSED_ABOVE = 100.0
START = {
    "M:ARM": 0.0,
    "M:FIRE": 0.0,
    "M:SP": 2.0,
    "N:SP": 2.0,
    "T:SP": 5.0,
}


# -- a stub engine whose command cannot be taken back ---------------------------


class StubModel(LUMEModel):
    """Setpoints of one wiring; a batch raising ``FIRE`` fires a shot that stays fired."""

    def __init__(self, name: str, wiring: list[Mapping[str, Any]]) -> None:
        self.fire = f"{name}:FIRE"
        self.shots = 0
        self.batches: list[dict[str, float]] = []
        self._defaults = {
            str(entry["address"]): float(entry.get("default", 0.0))
            for entry in wiring
            if entry.get("direction") == "write"
        }
        self._variables: dict[str, Variable] = {
            address: ScalarVariable(name=address, read_only=False) for address in self._defaults
        }
        self.inputs: dict[str, float] = {}
        self.reset()

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    def reset(self) -> None:
        self.inputs = dict(self._defaults)

    def _set(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            if float(value) > REFUSED_ABOVE:
                raise ValueError(f"{name} cannot reach {value}")
        batch = {name: float(value) for name, value in values.items()}
        self.batches.append(batch)
        if batch.get(self.fire, 0.0) > self.inputs.get(self.fire, 0.0):
            self.shots += 1
        self.inputs.update(batch)

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {name: self.inputs[name] for name in names}


def _build(model, wiring, deck, settings, active=None):
    del deck, settings, active
    return StubModel(model, list(wiring))


@pytest.fixture(autouse=True)
def _stub_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = SimpleNamespace(build=_build)
    monkeypatch.setattr(Composite, "_engine", staticmethod(lambda name: engine))
    monkeypatch.setattr(composite_module, "default_config_path", lambda: None)


# -- a synthetic view: the command owner sorts before the refusing model --------


def _channel(address: str, owner: str) -> dict[str, Any]:
    return {
        "address": address,
        "role": "setpoint",
        "pair": address,
        "value_type": "float",
        "unit": None,
        "description": None,
        "writable": True,
        "value_range": None,
        "owner": owner,
        "on": None,
    }


def _record(address: str, default: float, *, command: bool = False) -> dict[str, Any]:
    record = {
        "id": address,
        "address": address,
        "direction": "write",
        "default": default,
        "role": "setpoint",
        "plane": None,
        "refresh": "pass",
    }
    return record | {"command": True} if command else record


def _model(name: str, wiring: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": name,
        "engine": "stub",
        "served": True,
        "settings": {},
        "deck": None,
        "wiring": wiring,
    }


def _view(path: Path, *, marked: bool = True) -> Path:
    channels = [
        _channel("M:ARM", "M"),
        _channel("M:FIRE", "M"),
        _channel("M:SP", "M"),
        _channel("N:SP", "N"),
        _channel("T:SP", "texture"),
    ]
    documents = {
        "served_models.json": {"models": ["M", "N", "texture"]},
        "addresses.json": {
            "channels": sorted(channel["address"] for channel in channels),
            "status": ["T:SIM:M:STATUS", "T:SIM:N:STATUS"],
        },
        "variables.json": {
            "code": "T",
            "models": [
                _model(
                    "M",
                    [
                        _record("M:ARM", START["M:ARM"], command=marked),
                        _record("M:FIRE", START["M:FIRE"], command=marked),
                        _record("M:SP", START["M:SP"]),
                    ],
                ),
                _model("N", [_record("N:SP", START["N:SP"])]),
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
        "seeds.json": {"seeds": {"T:SP": {"nominal": START["T:SP"]}}},
        "scenarios.json": {"scenarios": [{"name": "nominal"}]},
    }
    view = path / "simulator"
    view.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        (view / name).write_text(
            json.dumps({"schema": SCHEMAS[name], **document}), encoding="utf-8"
        )
    return view


def _composite(tmp_path: Path, **view: Any) -> Composite:
    return Composite(
        _view(tmp_path, **view),
        state_dir=tmp_path / "state",
        log_dir=tmp_path / "log",
        clock=lambda: T0,
    )


def _stub(composite: Composite, model: str) -> StubModel:
    stub = composite._children[model].model
    assert isinstance(stub, StubModel)
    return stub


def _untouched(composite: Composite) -> None:
    """No child saw a write and every channel holds its start value."""
    assert _stub(composite, "M").batches == []
    assert _stub(composite, "N").batches == []
    assert _stub(composite, "M").shots == 0
    assert composite.held(list(START)) == START


# -- what is refused -----------------------------------------------------------


def test_a_shot_does_not_survive_a_batch_a_later_model_refuses(tmp_path: Path) -> None:
    composite = _composite(tmp_path)

    with pytest.raises(WriteRefusedError) as refused:
        composite.set({"M:FIRE": 1.0, "N:SP": 500.0})

    assert str(refused.value) == (
        "a command is written with channels of its own model only: "
        "M:FIRE is a command of M, N:SP a channel of N"
    )
    assert refused.value.address is None
    _untouched(composite)


def test_a_command_beside_another_models_channel_is_refused_whatever_the_values(
    tmp_path: Path,
) -> None:
    composite = _composite(tmp_path)

    with pytest.raises(WriteRefusedError, match="M:FIRE is a command of M, N:SP a channel of N"):
        composite.set({"N:SP": 3.0, "M:FIRE": 1.0})

    _untouched(composite)


def test_a_command_beside_a_texture_channel_is_refused(tmp_path: Path) -> None:
    composite = _composite(tmp_path)

    with pytest.raises(WriteRefusedError, match="M:ARM is a command of M, T:SP a channel of") as (
        refused
    ):
        composite.set({"T:SP": 6.0, "M:ARM": 1.0})

    assert refused.value.address is None
    _untouched(composite)


def test_a_command_at_its_held_value_beside_another_owner_is_refused(tmp_path: Path) -> None:
    composite = _composite(tmp_path)

    with pytest.raises(WriteRefusedError, match="a command is written with channels of its own"):
        composite.set({"M:FIRE": START["M:FIRE"], "T:SP": 6.0})

    _untouched(composite)


def test_a_value_its_channel_refuses_is_named_before_the_batch_is_judged(
    tmp_path: Path,
) -> None:
    composite = _composite(tmp_path)

    with pytest.raises(WriteRefusedError, match="not a valid float") as refused:
        composite.set({"M:FIRE": 1.0, "N:SP": "high"})

    assert refused.value.address == "N:SP"
    _untouched(composite)


# -- what stays legal ----------------------------------------------------------


def test_a_batch_of_one_models_commands_fires_once(tmp_path: Path) -> None:
    composite = _composite(tmp_path)

    composite.set({"M:ARM": 1.0, "M:FIRE": 1.0})

    stub = _stub(composite, "M")
    assert stub.batches == [{"M:ARM": 1.0, "M:FIRE": 1.0}]
    assert stub.shots == 1
    assert _stub(composite, "N").batches == []


def test_a_command_beside_a_setpoint_of_its_own_model_is_written(tmp_path: Path) -> None:
    composite = _composite(tmp_path)

    composite.set({"M:SP": 3.0, "M:FIRE": 1.0})

    assert _stub(composite, "M").shots == 1
    assert composite.held(["M:SP", "M:FIRE"]) == {"M:SP": 3.0, "M:FIRE": 1.0}


def test_a_command_alone_is_written(tmp_path: Path) -> None:
    composite = _composite(tmp_path)

    composite.set({"M:FIRE": 1.0})

    assert _stub(composite, "M").shots == 1


def test_a_refusal_of_the_commands_own_model_names_what_that_model_refused(
    tmp_path: Path,
) -> None:
    composite = _composite(tmp_path)

    with pytest.raises(WriteRefusedError, match="M:SP cannot reach 500.0"):
        composite.set({"M:SP": 500.0, "M:FIRE": 1.0})

    _untouched(composite)


def test_a_batch_without_a_command_still_spans_owners(tmp_path: Path) -> None:
    composite = _composite(tmp_path)

    composite.set({"M:SP": 3.0, "N:SP": 4.0, "T:SP": 6.0})

    assert composite.held(["M:SP", "N:SP", "T:SP"]) == {"M:SP": 3.0, "N:SP": 4.0, "T:SP": 6.0}
    assert _stub(composite, "M").shots == 0


def test_a_view_marking_no_command_refuses_no_batch_for_its_owners(tmp_path: Path) -> None:
    composite = _composite(tmp_path, marked=False)

    composite.set({"M:FIRE": 1.0, "N:SP": 3.0})

    assert composite.held(["M:FIRE", "N:SP"]) == {"M:FIRE": 1.0, "N:SP": 3.0}
