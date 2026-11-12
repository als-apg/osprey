"""The composite's delivery of model inputs: producers before consumers, baselines, failures."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from lume.model import LUMEModel
from lume.variables import ScalarVariable, Variable

from osprey_connectors.simulation import composite as composite_module
from osprey_connectors.simulation.view import Model

# The composite under test delivers model inputs; connectors <=2026.11.9 (the
# floor lane's) read none from the view. Removed by the floor bump past
# 2026.11.9.
if not hasattr(Model, "inputs"):
    pytest.skip(
        "the installed connectors predate model inputs (<=2026.11.9)",
        allow_module_level=True,
    )

from osprey_connectors.simulation.composite import Composite
from osprey_connectors.simulation.view import REFRESH, SCHEMAS
from tests._simulator_view import view_channel
from tests.connectors._view_records import activate

PASS, PERIODIC, FAST, EVENT = REFRESH
T0 = 1_760_000_000.0
SHOT = {"kind": "shot"}
CHARGE = {"kind": "charge"}


# -- a stub engine: OUT counts the advances, SP holds what is written ----------


class CountingModel(LUMEModel):
    """``<name>:SP`` a setpoint; ``<name>:OUT`` the advances since build or reset, plus SP."""

    def __init__(self, name: str, wiring: list[Mapping[str, Any]]) -> None:
        self.name = name
        self._variables: dict[str, Variable] = {
            str(entry["address"]): ScalarVariable(
                name=str(entry["address"]), read_only=entry.get("direction") != "write"
            )
            for entry in wiring
        }
        self.sp = 0.0
        self.advances = 0
        self.reset()

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    def reset(self) -> None:
        self.sp = 0.0
        self.advances = 0

    def _set(self, values: dict[str, Any]) -> None:
        self.sp = float(values[f"{self.name}:SP"])

    def _get(self, names: list[str]) -> dict[str, Any]:
        read = {f"{self.name}:SP": self.sp, f"{self.name}:OUT": self.sp + self.advances}
        return {name: read[name] for name in names}


CALLS: list[tuple[Any, ...]] = []
#: Model names whose ``advance`` raises.
FAIL_ADVANCE: set[str] = set()
#: Model names whose ``take_inputs`` raises.
FAIL_TAKE: set[str] = set()


def _build(model, wiring, deck, settings, active=None):
    del deck, settings, active
    CALLS.append(("build", model))
    return CountingModel(model, list(wiring))


def _advance(model: CountingModel, t_s: float) -> None:
    CALLS.append(("advance", model.name, t_s))
    if model.name in FAIL_ADVANCE:
        raise RuntimeError(f"{model.name} lost its beam")
    model.advances += 1


def _take_inputs(
    model: CountingModel,
    inputs: Sequence[tuple[Mapping[str, Any], Any]],
    t_s: float,
    *,
    baseline: bool,
) -> None:
    pairs = [(dict(record), value) for record, value in inputs]
    CALLS.append(("take_inputs", model.name, pairs, t_s, baseline))
    if model.name in FAIL_TAKE:
        raise ValueError(f"{model.name} cannot read its inputs")


STUB = SimpleNamespace(build=_build, advance=_advance, take_inputs=_take_inputs)


@pytest.fixture(autouse=True)
def _stub_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    CALLS.clear()
    FAIL_ADVANCE.clear()
    FAIL_TAKE.clear()
    real = Composite._engine

    def engine(name: str) -> Any:
        return STUB if name == "stub" else real(name)

    monkeypatch.setattr(Composite, "_engine", staticmethod(engine))
    monkeypatch.setattr(composite_module, "default_config_path", lambda: None)


# -- a synthetic view ----------------------------------------------------------


def _described(role: str, refresh: str) -> dict[str, Any]:
    return {"engine_role": role, "plane": None, "refresh": refresh}


def _view(path: Path, models: Mapping[str, Sequence[tuple[str, Mapping[str, Any]]]]) -> Path:
    """A view of stub models; ``models`` maps each name to its ``(address, engine)`` inputs."""
    entries: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []
    for name, inputs in models.items():
        entry: dict[str, Any] = {
            "name": name,
            "engine": "stub",
            "served": True,
            "settings": {},
            "deck": None,
            "wiring": [
                {"id": f"{name}1", "address": f"{name}:SP", "direction": "write", "default": 0.0}
                | _described("setpoint", PASS),
                {"id": f"{name}2", "address": f"{name}:OUT", "direction": "read"}
                | _described("output", PASS),
            ],
        }
        if inputs:
            entry["inputs"] = [{"address": address, "engine": engine} for address, engine in inputs]
        entries.append(entry)
        channels.append(view_channel(f"{name}:SP", name, role="setpoint", writable=True))
        channels.append(view_channel(f"{name}:OUT", name))
    entries.append(
        {
            "name": "texture",
            "engine": "texture",
            "served": True,
            "settings": {},
            "deck": None,
            "wiring": [],
        }
    )
    channels.append(view_channel("T:SP", role="setpoint", writable=True))
    documents = {
        "served_models.json": {"models": [*sorted(models), "texture"]},
        "addresses.json": {
            "channels": sorted(channel["address"] for channel in channels),
            "status": [f"T:SIM:{name}:STATUS" for name in sorted(models)],
        },
        "variables.json": {
            "code": "T",
            "models": sorted(entries, key=lambda entry: entry["name"]),
            "channels": sorted(channels, key=lambda channel: channel["address"]),
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


class _Clock:
    def __init__(self, t_s: float = T0) -> None:
        self.t_s = t_s

    def __call__(self) -> float:
        return self.t_s


def _composite(
    tmp_path: Path,
    clock: _Clock,
    models: Mapping[str, Sequence[tuple[str, Mapping[str, Any]]]],
    *,
    evolve: bool = True,
) -> Composite:
    return Composite(
        _view(tmp_path, models), state_dir=tmp_path / "state", clock=clock, evolve=evolve
    )


def _takes(model: str | None = None) -> list[tuple[Any, ...]]:
    """Each ``take_inputs`` call as ``(model, pairs, t_s, baseline)``."""
    return [
        call[1:]
        for call in CALLS
        if call[0] == "take_inputs" and (model is None or call[1] == model)
    ]


def _names() -> list[tuple[str, str]]:
    """Each call as ``(function, model)``, in call order."""
    return [(call[0], call[1]) for call in CALLS]


def _pair(address: str, engine: Mapping[str, Any], value: float) -> tuple[dict[str, Any], float]:
    return ({"address": address, "engine": dict(engine)}, value)


# -- no inputs: today's order and calls ----------------------------------------


def test_without_inputs_the_children_keep_name_order_and_no_delivery_is_made(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock, {"B": (), "A": (), "C": ()})
    clock.t_s = T0 + 1.0
    composite.get(["A:OUT"])
    composite.set({"B:SP": 2.0})
    composite.reset()

    assert composite.models == ["A", "B", "C"]
    assert _takes() == []
    assert _names() == [
        ("build", "A"), ("advance", "A"),
        ("build", "B"), ("advance", "B"),
        ("build", "C"), ("advance", "C"),
        ("advance", "A"), ("advance", "B"), ("advance", "C"),
        ("advance", "A"), ("advance", "B"), ("advance", "C"),
        ("advance", "A"), ("advance", "B"), ("advance", "C"),
    ]  # fmt: skip


def test_without_inputs_a_composite_that_does_not_evolve_makes_no_call(tmp_path: Path) -> None:
    composite = _composite(tmp_path, _Clock(), {"A": (), "B": ()}, evolve=False)
    composite.set({"A:SP": 1.0})
    composite.reset()

    assert _names() == [("build", "A"), ("build", "B")]


# -- producers before consumers ------------------------------------------------


@pytest.mark.parametrize(("producer", "consumer"), [("A", "Z"), ("Z", "A")])
def test_a_consumer_reads_its_producer_as_it_stands_after_the_same_pass(
    tmp_path: Path, producer: str, consumer: str
) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock, {producer: (), consumer: ((f"{producer}:OUT", SHOT),)})
    assert _names() == [
        ("build", producer), ("advance", producer),
        ("build", consumer), ("take_inputs", consumer), ("advance", consumer),
    ]  # fmt: skip
    assert _takes(consumer) == [(consumer, [_pair(f"{producer}:OUT", SHOT, 1.0)], T0, True)]

    CALLS.clear()
    clock.t_s = T0 + 1.0
    composite.get([f"{consumer}:OUT"])

    assert _names() == [
        ("advance", producer), ("take_inputs", consumer), ("advance", consumer),
    ]  # fmt: skip
    assert _takes(consumer) == [(consumer, [_pair(f"{producer}:OUT", SHOT, 2.0)], T0 + 1.0, False)]


def test_a_chain_is_ordered_producer_first_whatever_the_names(tmp_path: Path) -> None:
    composite = _composite(
        tmp_path,
        _Clock(),
        {"A": (("M:OUT", SHOT),), "M": (("Z:OUT", SHOT),), "Z": (), "B": ()},
    )

    assert [name for called, name in _names() if called == "build"] == ["B", "Z", "M", "A"]
    assert composite.models == ["A", "B", "M", "Z"]


def test_a_client_write_to_a_producer_reaches_its_consumer_on_the_next_pass(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock, {"P": (), "C": (("P:OUT", SHOT),)})
    composite.set({"P:SP": 10.0})
    CALLS.clear()

    composite.get(["C:OUT"])

    assert _takes("C") == [("C", [_pair("P:OUT", SHOT, 13.0)], T0, False)]


def test_inputs_are_delivered_in_declared_order_with_their_engine_blocks(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    _composite(
        tmp_path,
        clock,
        {"P": (), "Q": (), "C": (("Q:OUT", CHARGE), ("P:OUT", SHOT))},
    )

    assert _takes("C") == [
        ("C", [_pair("Q:OUT", CHARGE, 1.0), _pair("P:OUT", SHOT, 1.0)], T0, True)
    ]


# -- baselines -----------------------------------------------------------------


def _baselines(model: str) -> list[bool]:
    return [take[3] for take in _takes(model)]


def test_the_first_delivery_after_build_is_a_baseline_and_later_ones_are_not(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock, {"P": (), "C": (("P:OUT", SHOT),)})
    clock.t_s = T0 + 1.0
    composite.get(["C:OUT"])
    composite.advance()

    assert _baselines("C") == [True, False, False]


def test_reset_makes_the_next_delivery_a_baseline(tmp_path: Path) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock, {"P": (), "C": (("P:OUT", SHOT),)})
    composite.advance()
    CALLS.clear()

    composite.reset()
    composite.advance()

    assert _names()[:3] == [("advance", "P"), ("take_inputs", "C"), ("advance", "C")]
    assert _baselines("C") == [True, False]
    assert _takes("C")[0][1] == [_pair("P:OUT", SHOT, 1.0)]


def test_a_rebuilt_producer_makes_the_next_delivery_a_baseline(tmp_path: Path) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock, {"P": (), "C": (("P:OUT", SHOT),)})
    FAIL_ADVANCE.add("P")
    composite.advance()
    assert composite.status("P") != "ok"
    FAIL_ADVANCE.clear()
    CALLS.clear()

    composite.reset()

    assert _names()[:4] == [
        ("build", "P"), ("advance", "P"), ("take_inputs", "C"), ("advance", "C"),
    ]  # fmt: skip
    assert _takes("C") == [("C", [_pair("P:OUT", SHOT, 1.0)], T0, True)]


def test_a_producer_back_after_a_failure_is_delivered_again_as_a_baseline(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    composite = _composite(
        tmp_path, clock, {"P": (), "Q": (), "C": (("P:OUT", SHOT), ("Q:OUT", CHARGE))}
    )
    composite.advance()
    FAIL_ADVANCE.add("P")
    composite.advance()
    FAIL_ADVANCE.clear()
    CALLS.clear()

    activate(tmp_path / "state", "nominal")
    composite.advance()

    assert _baselines("C") == [True, False]
    assert [record["address"] for record, _ in _takes("C")[0][1]] == ["P:OUT", "Q:OUT"]


def test_a_scenario_change_makes_the_next_delivery_a_baseline(tmp_path: Path) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock, {"P": (), "C": (("P:OUT", SHOT),)})
    composite.advance()
    CALLS.clear()

    activate(tmp_path / "state", "nominal")
    composite.advance()

    assert _names()[:5] == [
        ("build", "P"), ("advance", "P"),
        ("build", "C"), ("take_inputs", "C"), ("advance", "C"),
    ]  # fmt: skip
    assert _baselines("C") == [True, False]


# -- a composite that does not evolve ------------------------------------------


def test_a_composite_that_does_not_evolve_delivers_at_build_reset_and_set_as_baselines(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock, {"P": (), "C": (("P:OUT", SHOT),)}, evolve=False)
    assert _names() == [("build", "P"), ("build", "C"), ("take_inputs", "C")]
    assert _takes("C") == [("C", [_pair("P:OUT", SHOT, 0.0)], T0, True)]

    CALLS.clear()
    composite.get(["C:OUT"])
    composite.held(["C:OUT"])
    assert CALLS == []

    composite.set({"P:SP": 4.0})
    assert _takes("C") == [("C", [_pair("P:OUT", SHOT, 4.0)], T0, True)]

    CALLS.clear()
    composite.reset()
    assert _takes("C") == [("C", [_pair("P:OUT", SHOT, 0.0)], T0, True)]
    assert not [name for name in _names() if name[0] == "advance"]


def test_a_refused_write_in_a_composite_that_does_not_evolve_delivers_nothing(
    tmp_path: Path,
) -> None:
    composite = _composite(tmp_path, _Clock(), {"P": (), "C": (("P:OUT", SHOT),)}, evolve=False)
    CALLS.clear()

    with pytest.raises(ValueError):
        composite.set({"P:SP": "not a number"})

    assert _takes() == []


# -- failures ------------------------------------------------------------------


def test_a_failed_producers_input_is_left_out(tmp_path: Path) -> None:
    clock = _Clock()
    composite = _composite(
        tmp_path, clock, {"P": (), "Q": (), "C": (("P:OUT", SHOT), ("Q:OUT", CHARGE))}
    )
    FAIL_ADVANCE.add("P")
    CALLS.clear()

    composite.advance()

    assert composite.status("P") != "ok"
    assert _takes("C") == [("C", [_pair("Q:OUT", CHARGE, 2.0)], T0, False)]


def test_no_call_is_made_when_every_producer_failed(tmp_path: Path) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock, {"P": (), "C": (("P:OUT", SHOT),)})
    FAIL_ADVANCE.add("P")
    CALLS.clear()

    composite.advance()
    composite.advance()

    assert _takes() == []
    assert [name for name in _names() if name[1] == "C"] == [("advance", "C")] * 2
    assert composite.status("C") == "ok"


def test_a_raise_in_take_inputs_fails_the_consumer_like_any_time_call(tmp_path: Path) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock, {"P": (), "C": (("P:OUT", SHOT),)})
    FAIL_TAKE.add("C")
    CALLS.clear()

    composite.advance()

    assert composite.status("C") == "C cannot read its inputs"
    assert ("advance", "C") not in _names()
    assert composite.output_severity(["C:OUT"]) == {"C:OUT": {"condition": "udf"}}
    assert composite.status("P") == "ok"


def test_a_raise_in_take_inputs_at_build_fails_the_consumer(tmp_path: Path) -> None:
    FAIL_TAKE.add("C")
    composite = _composite(tmp_path, _Clock(), {"P": (), "C": (("P:OUT", SHOT),)}, evolve=False)

    assert composite.status("C") == "C cannot read its inputs"
