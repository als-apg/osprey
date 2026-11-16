"""Model inputs end to end: facility file, build, simulator view, evolving composite.

Proves requirement 1 (a model may declare inputs that another model produces,
and the consumer reads them from the same pass) through the whole path: a
synthetic facility whose time-driven producer feeds a consumer, both test-only
engines, is built by the facility pipeline (``run_stages``), rendered into its
simulator view (``render_facility_outputs``) and loaded from that view into an
evolving :class:`Composite`. The model names are chosen so name order alone
would run the consumer first.

It also proves the structural half of the no-input performance criterion: a
facility that declares no inputs keeps its children in name order and makes
no delivery call.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import at
import pytest
import yaml
from lume.model import LUMEModel
from lume.variables import ScalarVariable, Variable

from osprey.facility.build import LATER_STAGES
from osprey.facility.render import render_facility_outputs
from osprey.facility.validate import run_stages
from osprey.simulation.engines import ENTRY_POINT_GROUP
from osprey.simulation.engines import pyat as pyat_engine
from osprey_connectors.simulation import composite as composite_module
from osprey_connectors.simulation.composite import Composite

PRODUCER_ENGINE = "fake_model_input_producer_engine"
CONSUMER_ENGINE = "fake_model_input_consumer_engine"
#: Named so that name order alone would run the consumer first.
PRODUCER = "zring"
CONSUMER = "aline"
PRODUCED = "ZRING:X"
CONSUMED = "ALINE:X"
SHOT = {"role": "shot"}
T0 = 1_760_000_000.0

#: Every engine call, in call order: ``(function, model, ...)``.
CALLS: list[tuple[Any, ...]] = []


class StubModel(LUMEModel):
    """Reads its wired channels: the producer counts its advances, the consumer echoes its input."""

    def __init__(self, name: str, wiring: Sequence[Mapping[str, Any]]) -> None:
        self.name = name
        self._variables: dict[str, Variable] = {
            str(entry["address"]): ScalarVariable(name=str(entry["address"]), read_only=True)
            for entry in wiring
        }
        self.value = 0.0

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    def reset(self) -> None:
        self.value = 0.0

    def _set(self, values: dict[str, Any]) -> None:
        del values

    def _get(self, names: list[str]) -> dict[str, Any]:
        return dict.fromkeys(names, self.value)


def _build(name, wiring, deck, settings, active=None):
    del deck, settings, active
    CALLS.append(("build", name))
    return StubModel(name, list(wiring))


def _producer_advance(model: StubModel, t_s: float) -> None:
    CALLS.append(("advance", model.name, t_s))
    model.value += 1.0


def _consumer_advance(model: StubModel, t_s: float) -> None:
    CALLS.append(("advance", model.name, t_s))


def _take_inputs(
    model: StubModel,
    inputs: Sequence[tuple[Mapping[str, Any], Any]],
    t_s: float,
    *,
    baseline: bool,
) -> None:
    pairs = [(dict(record), value) for record, value in inputs]
    CALLS.append(("take_inputs", model.name, pairs, t_s, baseline))
    model.value = float(pairs[0][1])


def _like_pyat(name: str) -> types.ModuleType:
    """A module carrying pyat's facility-build functions, so the build checks it as pyat."""
    module = types.ModuleType(name)
    for key, value in vars(pyat_engine).items():
        if not key.startswith("__"):
            setattr(module, key, value)
    module.build = _build
    for hook in ("take_inputs", "describe_input"):
        if hasattr(module, hook):
            delattr(module, hook)
    return module


@pytest.fixture(autouse=True)
def engines(monkeypatch: pytest.MonkeyPatch) -> None:
    """``producer`` advances and counts; ``consumer`` takes its input and echoes it."""
    producer = _like_pyat(PRODUCER_ENGINE)
    producer.advance = _producer_advance
    consumer = _like_pyat(CONSUMER_ENGINE)
    consumer.advance = _consumer_advance
    consumer.take_inputs = _take_inputs
    consumer.describe_input = lambda record: None
    monkeypatch.setitem(sys.modules, PRODUCER_ENGINE, producer)
    monkeypatch.setitem(sys.modules, CONSUMER_ENGINE, consumer)
    found = metadata.EntryPoints(
        [
            metadata.EntryPoint(name="pyat", value=pyat_engine.__name__, group=ENTRY_POINT_GROUP),
            metadata.EntryPoint(name="producer", value=PRODUCER_ENGINE, group=ENTRY_POINT_GROUP),
            metadata.EntryPoint(name="consumer", value=CONSUMER_ENGINE, group=ENTRY_POINT_GROUP),
        ]
    )

    def entry_points(**selection: Any) -> metadata.EntryPoints:
        return found.select(**selection) if selection else found

    monkeypatch.setattr(metadata, "entry_points", entry_points)
    monkeypatch.setattr(composite_module, "default_config_path", lambda: None)
    CALLS.clear()


# -- the facility --------------------------------------------------------------


def _save_deck(path: Path) -> None:
    lattice = at.Lattice(
        [
            at.Drift("D0", 1.0),
            at.Quadrupole("Q", 0.2, 1.2),
            at.Drift("D1", 1.8),
            at.Monitor("BPM1"),
        ],
        energy=3e9,
        particle="electron",
        periodicity=1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    at.save_lattice(lattice, str(path))


def _model(name: str, engine: str, address: str, inputs: list[Any] | None) -> dict[str, Any]:
    model: dict[str, Any] = {
        "name": name,
        "engine": engine,
        "deck": f"decks/{name}.json",
        "wiring": [{"address": address, "element": "BPM1", "engine": {"axis": "x"}}],
    }
    if inputs is not None:
        model["inputs"] = inputs
    return model


def _simulator_view(tmp_path: Path, *, with_inputs: bool) -> Path:
    """Build the two-model facility and render it; the render's simulator view."""
    inputs = [{"address": PRODUCED, "engine": SHOT}] if with_inputs else None
    files: dict[str, Any] = {
        "records/channels.yaml": [
            {"id": address, "unit": "mm"} for address in (PRODUCED, CONSUMED)
        ],
        "models.yaml": [
            _model(PRODUCER, "producer", PRODUCED, None),
            _model(CONSUMER, "consumer", CONSUMED, inputs),
        ],
    }
    facility_dir = tmp_path / "facility"
    for rel, data in files.items():
        path = facility_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    for model in files["models.yaml"]:
        _save_deck(facility_dir / model["deck"])
    report = run_stages(facility_dir, project_name="p", later=LATER_STAGES)
    assert report.ok, [error.format_message() for error in report.errors]
    document = report.validated.document
    assert document is not None
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    render_facility_outputs(render_dir, document, {}, facility_dir)
    return render_dir / "data" / "simulator"


class _Clock:
    def __init__(self) -> None:
        self.t_s = T0

    def __call__(self) -> float:
        return self.t_s


def _composite(tmp_path: Path, clock: _Clock, *, with_inputs: bool) -> Composite:
    view = _simulator_view(tmp_path, with_inputs=with_inputs)
    return Composite(view, state_dir=tmp_path / "state", clock=clock, evolve=True, model_log=False)


def _names() -> list[tuple[str, str]]:
    return [(call[0], call[1]) for call in CALLS]


def _takes() -> list[tuple[Any, ...]]:
    """Each ``take_inputs`` call as ``(pairs, t_s, baseline)``."""
    return [call[2:] for call in CALLS if call[0] == "take_inputs"]


def _pair(value: float) -> tuple[dict[str, Any], float]:
    return ({"address": PRODUCED, "engine": dict(SHOT)}, value)


# -- with inputs ---------------------------------------------------------------


def test_the_consumer_reads_the_producers_value_from_the_same_pass(tmp_path: Path) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock, with_inputs=True)

    assert sorted(composite.models) == [CONSUMER, PRODUCER]
    assert _names() == [
        ("build", PRODUCER), ("advance", PRODUCER),
        ("build", CONSUMER), ("take_inputs", CONSUMER), ("advance", CONSUMER),
    ]  # fmt: skip
    assert _takes() == [([_pair(1.0)], T0, True)]

    CALLS.clear()
    clock.t_s = T0 + 1.0
    read = composite.get([PRODUCED, CONSUMED])

    assert _names() == [
        ("advance", PRODUCER), ("take_inputs", CONSUMER), ("advance", CONSUMER),
    ]  # fmt: skip
    assert _takes() == [([_pair(2.0)], T0 + 1.0, False)]
    assert read == {PRODUCED: 2.0, CONSUMED: 2.0}


def test_a_reset_is_delivered_as_a_baseline(tmp_path: Path) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock, with_inputs=True)
    clock.t_s = T0 + 1.0
    composite.advance()
    assert [take[2] for take in _takes()] == [True, False]
    CALLS.clear()

    composite.reset()
    clock.t_s = T0 + 2.0
    composite.advance()

    assert _names()[:3] == [("advance", PRODUCER), ("take_inputs", CONSUMER), ("advance", CONSUMER)]
    assert _takes() == [([_pair(1.0)], T0 + 1.0, True), ([_pair(2.0)], T0 + 2.0, False)]


# -- without inputs ------------------------------------------------------------


def test_a_facility_without_inputs_keeps_name_order_and_makes_no_delivery(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock, with_inputs=False)
    clock.t_s = T0 + 1.0
    composite.get([CONSUMED])
    composite.reset()

    assert _takes() == []
    assert _names() == [
        ("build", CONSUMER), ("advance", CONSUMER),
        ("build", PRODUCER), ("advance", PRODUCER),
        ("advance", CONSUMER), ("advance", PRODUCER),
        ("advance", CONSUMER), ("advance", PRODUCER),
    ]  # fmt: skip
