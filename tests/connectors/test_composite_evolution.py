"""The composite's evolution: a time-driven child advanced by the injected clock."""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from lume.model import LUMEModel
from lume.variables import ScalarVariable, Variable

from osprey_connectors.simulation import composite as composite_module
from osprey_connectors.simulation.channel_faults import STUCK
from osprey_connectors.simulation.composite import Composite
from osprey_connectors.simulation.view import REFRESH, SCHEMAS
from tests._simulator_view import view_channel
from tests.connectors._view_records import activate

PASS, PERIODIC, FAST, EVENT = REFRESH
T0 = 1_760_000_000.0
#: How long after a client write to ``M:REQ`` the stub's step falls due.
STEP_S = 0.5
COMPOSITE_SOURCE = Path(composite_module.__file__)


# -- a time-driven stub engine -------------------------------------------------


class TimedModel(LUMEModel):
    """A request setpoint, a fast reading of elapsed time, and one step per request."""

    def __init__(self, wiring: list[Mapping[str, Any]], active: Mapping[str, Any]) -> None:
        self._variables: dict[str, Variable] = {}
        self._defaults: dict[str, float] = {}
        for entry in wiring:
            address = str(entry["address"])
            writable = entry.get("direction") == "write"
            self._variables[address] = ScalarVariable(name=address, read_only=not writable)
            if writable:
                self._defaults[address] = float(active.get(address, entry.get("default", 0.0)))
        self.inputs: dict[str, float] = {}
        self.anchor: float | None = None
        self.now: float | None = None
        self.pending: list[float] = []
        self.steps = 0
        self.fired: set[str] = set()
        self.reset()

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    def reset(self) -> None:
        self.inputs = dict(self._defaults)
        self.anchor = None
        self.now = None
        self.pending = []
        self.steps = 0
        self.fired = set()

    def _set(self, values: dict[str, Any]) -> None:
        self.inputs.update({name: float(value) for name, value in values.items()})

    def _get(self, names: list[str]) -> dict[str, Any]:
        read = {
            "M:FAST": 0.0 if self.now is None else self.now - float(self.anchor or 0.0),
            "M:STEP": float(self.steps),
        }
        return {name: self.inputs.get(name, read.get(name)) for name in names}


CALLS: list[tuple[str, Any]] = []


def _build(model, wiring, deck, settings, active=None):
    del model, deck, settings
    CALLS.append(("build", None))
    return TimedModel(list(wiring), active or {})


def _advance(model: TimedModel, t_s: float) -> None:
    CALLS.append(("advance", t_s))
    if model.anchor is None:
        model.anchor = t_s
    model.now = t_s
    due = [deadline for deadline in model.pending if deadline <= t_s]
    model.pending = [deadline for deadline in model.pending if deadline > t_s]
    if due:
        model.steps += len(due)
        model.fired.add("M:STEP")


def _next_deadline(model: TimedModel) -> float | None:
    CALLS.append(("next_deadline", None))
    return min(model.pending) if model.pending else None


def _emitted(model: TimedModel) -> set[str]:
    CALLS.append(("emitted", None))
    fired, model.fired = model.fired, set()
    return fired


def _client_write(model: TimedModel, values: Mapping[str, Any], t_s: float) -> None:
    CALLS.append(("client_write", dict(values)))
    if "M:REQ" in values:
        model.pending.append(t_s + STEP_S)


TIMED = SimpleNamespace(
    build=_build,
    advance=_advance,
    next_deadline=_next_deadline,
    emitted=_emitted,
    client_write=_client_write,
)


def _forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("a composite that does not evolve called the time contract")


UNEVOLVED = SimpleNamespace(
    build=_build,
    advance=_forbidden,
    next_deadline=_forbidden,
    emitted=_forbidden,
    client_write=_forbidden,
)


@pytest.fixture(autouse=True)
def _timed_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    CALLS.clear()
    real = Composite._engine
    engines = {"timed": TIMED, "unevolved": UNEVOLVED}

    def engine(name: str) -> Any:
        return engines[name] if name in engines else real(name)

    monkeypatch.setattr(Composite, "_engine", staticmethod(engine))
    monkeypatch.setattr(composite_module, "default_config_path", lambda: None)


# -- a synthetic view ----------------------------------------------------------


def _described(role: str, refresh: str) -> dict[str, Any]:
    return {"role": role, "plane": None, "refresh": refresh}


def _view(path: Path, *, engine: str = "timed", scenarios: list[dict[str, Any]] | None = None):
    wiring = [
        {"id": "1", "address": "M:REQ", "direction": "write", "default": 0.0}
        | _described("setpoint", PASS),
        {"id": "2", "address": "M:FAST", "direction": "read"} | _described("output", FAST),
        {"id": "3", "address": "M:STEP", "direction": "read"} | _described("output", EVENT),
    ]
    channels = [
        view_channel("M:REQ", "M", role="setpoint", writable=True),
        view_channel("M:FAST", "M"),
        view_channel("M:STEP", "M"),
        view_channel("T:SP", role="setpoint", writable=True),
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
                    "engine": engine,
                    "served": True,
                    "settings": {},
                    "deck": None,
                    "wiring": wiring,
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
        "seeds.json": {"seeds": {"T:SP": {"nominal": 5.0}}},
        "scenarios.json": {"scenarios": [{"name": "nominal"}, *(scenarios or [])]},
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


def _composite(tmp_path: Path, clock: _Clock, *, evolve: bool = True, **view: Any) -> Composite:
    return Composite(
        _view(tmp_path, **view), state_dir=tmp_path / "state", clock=clock, evolve=evolve
    )


def _calls(name: str) -> list[Any]:
    return [argument for called, argument in CALLS if called == name]


# -- evolution -----------------------------------------------------------------


def test_a_built_child_is_advanced_at_once_to_anchor_its_time(tmp_path: Path) -> None:
    clock = _Clock()
    _composite(tmp_path, clock)

    assert [called for called, _ in CALLS] == ["build", "advance"]
    assert _calls("advance") == [T0]


def test_the_fast_channel_is_served_as_the_test_clock_advances_it(tmp_path: Path) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock)

    assert composite.get("M:FAST") == 0.0
    clock.t_s = T0 + 0.1
    assert composite.get("M:FAST") == pytest.approx(0.1)
    clock.t_s = T0 + 2.5
    assert composite.get(["M:FAST"]) == {"M:FAST": pytest.approx(2.5)}
    assert _calls("advance")[-1] == T0 + 2.5


def test_advance_alone_moves_the_child_to_the_clock(tmp_path: Path) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock)
    clock.t_s = T0 + 1.0

    composite.advance()

    assert _calls("advance")[-1] == T0 + 1.0


def test_a_client_write_reaches_client_write_and_schedules_a_step(tmp_path: Path) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock)
    clock.t_s = T0 + 1.0

    composite.set({"M:REQ": 1.0, "T:SP": 2.0})

    assert _calls("client_write") == [{"M:REQ": 1.0}]
    assert _calls("advance")[-1] == T0 + 1.0
    assert composite.next_deadline() == T0 + 1.0 + STEP_S
    assert composite.emitted() == frozenset()

    clock.t_s = T0 + 1.0 + STEP_S
    assert composite.get("M:STEP") == 1.0
    assert composite.emitted() == frozenset({"M:STEP"})
    assert composite.emitted() == frozenset()
    assert composite.next_deadline() is None


def test_a_write_the_child_refuses_reaches_no_client_write(tmp_path: Path) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock)

    with pytest.raises(ValueError):
        composite.set({"M:REQ": "not a number"})

    assert _calls("client_write") == []


def test_restore_puts_state_back_without_a_client_write(tmp_path: Path) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock)
    clock.t_s = T0 + 1.0

    composite.restore({"M:REQ": 3.0, "T:SP": 7.0})

    assert composite.get(["M:REQ", "T:SP"]) == {"M:REQ": 3.0, "T:SP": 7.0}
    assert _calls("client_write") == []
    assert composite.next_deadline() is None
    assert _calls("advance")[-1] == T0 + 1.0


def test_the_start_state_and_a_scenario_write_reach_no_client_write(tmp_path: Path) -> None:
    clock = _Clock()
    scenarios = [{"name": "request", "overrides": {"M:REQ": 4.0}}]
    composite = _composite(tmp_path, clock, scenarios=scenarios)
    activate(tmp_path / "state", "request")

    assert composite.get("M:REQ") == 4.0
    assert _calls("build") == [None, None]
    assert _calls("client_write") == []
    assert composite.next_deadline() is None


def test_a_write_under_a_stuck_fault_reaches_no_client_write(tmp_path: Path) -> None:
    clock = _Clock()
    scenarios = [{"name": "held", "channel_faults": {"M:REQ": STUCK}}]
    composite = _composite(tmp_path, clock, scenarios=scenarios)
    activate(tmp_path / "state", "held")

    composite.set({"M:REQ": 1.0})

    assert composite.get("M:REQ") == 1.0
    assert _calls("client_write") == []
    assert composite.next_deadline() is None


def test_a_rebuild_drops_pending_deadlines_and_advances_the_new_child(tmp_path: Path) -> None:
    clock = _Clock()
    scenarios = [{"name": "other", "overrides": {"M:REQ": 0.0}}]
    composite = _composite(tmp_path, clock, scenarios=scenarios)
    composite.set({"M:REQ": 1.0})
    assert composite.next_deadline() == T0 + STEP_S

    clock.t_s = T0 + 0.2
    activate(tmp_path / "state", "other")

    assert composite.next_deadline() is None
    built = max(index for index, (called, _) in enumerate(CALLS) if called == "build")
    assert CALLS[built + 1] == ("advance", T0 + 0.2)
    assert composite.get("M:FAST") == 0.0


def test_a_reset_mid_sequence_is_followed_by_an_advance_and_no_client_write(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock)
    composite.set({"M:REQ": 1.0})
    assert len(_calls("client_write")) == 1
    clock.t_s = T0 + 0.2
    CALLS.clear()

    composite.reset()

    assert CALLS == [("advance", T0 + 0.2)]
    assert composite.next_deadline() is None
    assert composite.get("M:FAST") == 0.0


def test_the_earliest_deadline_and_the_emitted_union_span_every_child(tmp_path: Path) -> None:
    clock = _Clock()
    composite = _composite(tmp_path, clock)
    child = composite._children["M"]
    assert isinstance(child.model, TimedModel)
    child.model.pending = [T0 + 3.0, T0 + 2.0]
    child.model.fired = {"M:STEP", "M:FAST"}

    assert composite.next_deadline() == T0 + 2.0
    assert composite.emitted() == frozenset({"M:STEP", "M:FAST"})


def test_a_child_without_the_time_contract_is_never_advanced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _Clock()
    plain = SimpleNamespace(build=_build)
    monkeypatch.setattr(Composite, "_engine", staticmethod(lambda name: plain))
    composite = _composite(tmp_path, clock)

    composite.set({"M:REQ": 1.0})
    composite.reset()

    assert composite.next_deadline() is None
    assert composite.emitted() == frozenset()
    assert composite.get("M:FAST") == 0.0
    assert [called for called, _ in CALLS] == ["build"]


# -- without evolve ------------------------------------------------------------


def test_a_composite_that_does_not_evolve_never_calls_the_time_contract(tmp_path: Path) -> None:
    clock = _Clock()
    scenarios = [{"name": "request", "overrides": {"M:REQ": 4.0}}]
    composite = _composite(tmp_path, clock, evolve=False, engine="unevolved", scenarios=scenarios)
    clock.t_s = T0 + 5.0

    assert composite.get("M:FAST") == 0.0
    composite.set({"M:REQ": 1.0})
    composite.restore({"M:REQ": 2.0})
    composite.advance()
    composite.reset()
    activate(tmp_path / "state", "request")
    assert composite.get("M:REQ") == 4.0
    assert composite.next_deadline() is None
    assert composite.emitted() == frozenset()
    assert [called for called, _ in CALLS] == ["build", "build"]


def test_evolve_is_off_by_default(tmp_path: Path) -> None:
    composite = Composite(_view(tmp_path, engine="unevolved"), clock=_Clock())

    composite.set({"M:REQ": 1.0})

    assert composite.next_deadline() is None


# -- the source ----------------------------------------------------------------


def test_the_composite_restates_no_refresh_word_as_a_constant() -> None:
    """A refresh word the composite needs comes from the view module, never a copy here."""
    tree = ast.parse(COMPOSITE_SOURCE.read_text(encoding="utf-8"))
    restated = sorted(
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign | ast.AnnAssign)
        and isinstance(node.value, ast.Constant)
        and node.value.value in REFRESH
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    )

    assert restated == []
