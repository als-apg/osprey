"""The injection chain model: a deterministic state machine under an explicit clock."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest
from lume.exceptions import ReadOnlyError
from lume.model import LUMEModel

from osprey.simulation.engines import injection_model
from osprey.simulation.engines.injection_model import (
    CHARGE,
    COUNTER,
    CURRENT,
    EFFICIENCY,
    InjectionModel,
    Step,
)

T0 = 1024.0
PERIOD = 2.0
WIDTH = 1.0
S = T0 + 0.0625  # a shot start, binary-exact like every time here
PER_CHARGE = 3.0

START = {
    "src": 4.0,
    "eff0": 0.5,
    "eff1": 0.25,
    "stored": 10.0,
    "count": 7.0,
    "count2": 1.0,
}


def _steps() -> list[Step]:
    """Declared out of time order on purpose; ``count2`` ties ``count``."""
    return [
        Step("charge0", CHARGE, 0.25, stage=0),
        Step("count", COUNTER, 0.125),
        Step("count2", COUNTER, 0.125),
        Step("eff0_read", EFFICIENCY, 0.375, stage=0),
        Step("charge1", CHARGE, 0.5, stage=1),
        Step("eff1_read", EFFICIENCY, 0.625, stage=1),
        Step("stored_read", CURRENT, 0.75),
    ]


def _model(**overrides: Any) -> InjectionModel:
    kwargs: dict[str, Any] = {
        "request": "req",
        "source_charge": "src",
        "efficiencies": ["eff0", "eff1"],
        "stored_current": "stored",
        "steps": _steps(),
        "currents": ["stored_read", "stored_fast"],
        "busy": ["busy"],
        "period_s": PERIOD,
        "width_s": WIDTH,
        "current_per_charge": PER_CHARGE,
        "start": START,
    }
    kwargs.update(overrides)
    return InjectionModel(**kwargs)


def _shot(model: InjectionModel, t_s: float = S) -> None:
    model.advance(T0)
    model.advance(t_s)
    model.set({"req": 1.0})
    model.client_write({"req": 1.0}, t_s)


EVENT_ADDRESSES = ["count", "count2", "charge0", "eff0_read", "charge1", "eff1_read"]


# -- shape ---------------------------------------------------------------------


def test_the_model_is_a_lume_model_with_setpoints_and_readings() -> None:
    model = _model()
    assert isinstance(model, LUMEModel)
    variables = model.supported_variables
    for name in ("req", "src", "eff0", "eff1", "stored"):
        assert variables[name].read_only is False
    for name in (*EVENT_ADDRESSES, "stored_read", "stored_fast", "busy"):
        assert variables[name].read_only is True


def test_start_values_are_served_and_absent_ones_read_zero() -> None:
    model = _model()
    values = model.get(["src", "eff0", "stored", "count", "charge0", "busy", "req"])
    assert values == {
        "src": 4.0,
        "eff0": 0.5,
        "stored": 10.0,
        "count": 7.0,
        "charge0": 0.0,
        "busy": 0.0,
        "req": 0.0,
    }
    assert model.get(["stored_read", "stored_fast"]) == {"stored_read": 10.0, "stored_fast": 10.0}


def test_a_reading_refuses_a_write() -> None:
    with pytest.raises(ReadOnlyError):
        _model().set({"count": 1.0})


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"period_s": 0.0}, id="zero-period"),
        pytest.param({"width_s": 0.0}, id="zero-width"),
        pytest.param({"width_s": PERIOD}, id="width-not-below-period"),
        pytest.param({"steps": [Step("x", "bogus", 0.1)]}, id="unknown-step-kind"),
        pytest.param({"steps": [Step("x", CHARGE, 0.1)]}, id="charge-without-stage"),
        pytest.param({"steps": [Step("x", CHARGE, 0.1, stage=2)]}, id="stage-out-of-range"),
        pytest.param({"steps": [Step("x", COUNTER, -0.1)]}, id="negative-offset"),
    ],
)
def test_an_inconsistent_chain_is_refused(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _model(**overrides)


# -- one shot ------------------------------------------------------------------


def test_one_shot_applies_its_steps_at_their_offsets_in_declared_order() -> None:
    model = _model()
    _shot(model)
    model.emitted()
    seen: list[list[str]] = []
    for offset in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875):
        model.advance(S + offset - 1e-9)
        seen.append(model.emitted())
        model.advance(S + offset)
        seen.append(model.emitted())
    assert seen == [
        [],
        ["count", "count2"],
        [],
        ["charge0"],
        [],
        ["eff0_read"],
        [],
        ["charge1"],
        [],
        ["eff1_read"],
        [],
        [],  # the current step changes a pass channel: nothing emitted
        [],
        [],
    ]


def test_each_step_value_follows_the_chain() -> None:
    model = _model()
    _shot(model)
    model.advance(S + 0.875)  # after every step
    values = model.get(["count", "count2", "charge0", "eff0_read", "charge1", "eff1_read"])
    assert values == {
        "count": 8.0,
        "count2": 2.0,
        "charge0": 4.0 * 0.5,
        "eff0_read": 0.5,
        "charge1": 4.0 * 0.5 * 0.25,
        "eff1_read": 0.25,
    }


def test_the_current_step_adds_final_charge_times_current_per_charge() -> None:
    model = _model()
    _shot(model)
    model.advance(S + 0.75 - 1e-9)
    assert model.get("stored_read") == 10.0
    model.advance(S + 0.75)
    expected = 10.0 + 4.0 * 0.5 * 0.25 * PER_CHARGE
    assert model.get(["stored", "stored_read", "stored_fast"]) == {
        "stored": expected,
        "stored_read": expected,
        "stored_fast": expected,
    }


def test_a_write_to_the_stored_current_parameter_sets_the_stored_current() -> None:
    model = _model()
    model.advance(T0)
    model.set({"stored": 42.0})
    assert model.get(["stored_read", "stored_fast"]) == {"stored_read": 42.0, "stored_fast": 42.0}


def test_a_shot_ends_at_its_last_step_and_a_new_request_then_starts_another() -> None:
    model = _model()
    _shot(model)
    assert model.in_shot is True
    model.advance(S + 0.75 - 1e-9)
    assert model.in_shot is True
    model.advance(S + 0.75)
    assert model.in_shot is False
    model.client_write({"req": 1.0}, S + 1.0)
    assert model.in_shot is True
    model.advance(S + 1.75)
    assert model.get("count") == 9.0


def test_parameters_are_latched_at_shot_start_and_a_later_write_shows_in_the_next_shot() -> None:
    model = _model()
    _shot(model)
    model.set({"eff0": 0.8, "src": 2.0})
    model.advance(S + 0.75)
    assert model.get(["charge0", "eff0_read"]) == {"charge0": 2.0, "eff0_read": 0.5}
    model.client_write({"req": 1.0}, S + 1.0)
    model.advance(S + 1.75)
    assert model.get(["charge0", "eff0_read"]) == {"charge0": 2.0 * 0.8, "eff0_read": 0.8}


def test_a_request_during_a_shot_is_ignored() -> None:
    model = _model()
    _shot(model)
    model.set({"req": 2.0})
    model.client_write({"req": 2.0}, S + 0.0625)  # a moment later
    assert model.get("req") == 2.0  # the write itself is accepted
    model.advance(T0 + 5 * PERIOD)
    assert model.get(["count", "count2"]) == {"count": 8.0, "count2": 2.0}


def test_a_set_of_the_request_channel_starts_no_shot() -> None:
    model = _model()
    model.advance(T0)
    model.set({"req": 1.0})  # as restore and rollback do
    assert model.in_shot is False
    model.advance(T0 + 5 * PERIOD)
    assert model.get("count") == 7.0
    assert set(model.emitted()) == {"busy"}


def test_a_client_write_without_the_request_starts_no_shot() -> None:
    model = _model()
    model.advance(T0)
    model.client_write({"src": 5.0}, T0)
    assert model.in_shot is False


def test_reset_mid_shot_produces_no_further_steps_and_restores_start_values() -> None:
    model = _model()
    _shot(model)
    model.advance(S + 0.25)
    assert model.get("charge0") == 2.0
    model.set({"src": 9.0})
    model.reset()
    assert model.in_shot is False
    assert model.next_deadline() is None
    assert model.emitted() == []
    assert model.get(["count", "charge0", "src", "busy"]) == {
        "count": 7.0,
        "charge0": 0.0,
        "src": 4.0,
        "busy": 0.0,
    }
    model.advance(S + 0.75)
    assert model.get(["count", "charge1", "stored_read"]) == {
        "count": 7.0,
        "charge1": 0.0,
        "stored_read": 10.0,
    }
    assert model.emitted() == ["busy"]  # re-anchored: only the cycle's rise


def test_a_step_at_offset_zero_applies_at_the_request() -> None:
    model = _model(steps=[Step("count", COUNTER, 0.0)])
    model.advance(T0)
    model.emitted()
    model.client_write({"req": 1.0}, T0 + 0.1)
    assert model.get("count") == 8.0
    assert model.emitted() == ["count"]
    assert model.in_shot is False


# -- the cycle signal ----------------------------------------------------------


def test_the_cycle_rises_at_the_first_advance_and_each_period_for_its_width() -> None:
    model = _model()
    trace: list[tuple[float, float, list[str]]] = []
    for t in (0.0, 0.5, 1.0, 1.25, 1.875, 2.0, 2.875, 3.0, 4.0):
        model.advance(T0 + t)
        trace.append((t, model.get("busy"), model.emitted()))
    assert trace == [
        (0.0, 1.0, ["busy"]),
        (0.5, 1.0, []),
        (1.0, 0.0, ["busy"]),
        (1.25, 0.0, []),
        (1.875, 0.0, []),
        (2.0, 1.0, ["busy"]),
        (2.875, 1.0, []),
        (3.0, 0.0, ["busy"]),
        (4.0, 1.0, ["busy"]),
    ]


def test_the_cycle_runs_the_same_with_and_without_shots() -> None:
    quiet, shooting = _model(), _model()
    quiet.advance(T0)
    _shot(shooting, T0)
    for t in (0.25, 0.5, 1.0, 2.0, 2.5, 3.0):
        quiet.advance(T0 + t)
        shooting.advance(T0 + t)
        assert quiet.get("busy") == shooting.get("busy")
        assert ("busy" in quiet.emitted()) == ("busy" in shooting.emitted())


def test_an_advance_over_several_edges_emits_the_cycle_once() -> None:
    model = _model()
    model.advance(T0)
    model.emitted()
    model.advance(T0 + 3 * PERIOD + 0.25)
    assert model.emitted() == ["busy"]
    assert model.get("busy") == 1.0


# -- the time contract ---------------------------------------------------------


def test_nothing_happens_before_the_first_advance() -> None:
    model = _model()
    assert model.next_deadline() is None
    assert model.emitted() == []


def test_the_emitted_set_is_cleared_as_returned() -> None:
    model = _model()
    _shot(model)
    model.advance(S + 0.25)
    assert model.emitted() == ["busy", "count", "count2", "charge0"]
    assert model.emitted() == []


def test_several_steps_due_in_one_advance_are_applied_in_order() -> None:
    model = _model()
    model.advance(T0)
    model.emitted()
    model.client_write({"req": 1.0}, S)
    model.advance(S + 0.5625)
    assert model.emitted() == ["count", "count2", "charge0", "eff0_read", "charge1"]


def test_next_deadline_is_the_earlier_of_the_next_edge_and_the_next_step() -> None:
    model = _model()
    model.advance(T0)
    assert model.next_deadline() == T0 + WIDTH
    model.client_write({"req": 1.0}, S)
    assert model.next_deadline() == S + 0.125
    model.advance(S + 0.125)
    assert model.next_deadline() == S + 0.25
    model.advance(S + 0.75)
    assert model.next_deadline() == T0 + WIDTH
    model.advance(T0 + WIDTH)
    assert model.next_deadline() == T0 + PERIOD


def test_a_clock_reading_earlier_than_the_last_one_changes_nothing() -> None:
    model = _model()
    model.advance(T0)
    model.advance(T0 + WIDTH)
    model.emitted()
    model.advance(T0 + 0.5)
    assert model.emitted() == []
    assert model.get("busy") == 0.0


def _run(sequence: list[tuple[str, Any]]) -> list[Any]:
    model = _model()
    out: list[Any] = []
    names = list(model.supported_variables)
    for kind, arg in sequence:
        if kind == "advance":
            model.advance(arg)
        elif kind == "set":
            model.set(arg[0])
        else:
            model.client_write(arg[0], arg[1])
        out.append((model.get(names), model.emitted(), model.next_deadline()))
    return out


def test_identical_input_sequences_give_identical_output() -> None:
    sequence: list[tuple[str, Any]] = [
        ("advance", T0),
        ("set", ({"req": 1.0},)),
        ("write", ({"req": 1.0}, T0 + 0.01)),
        ("advance", T0 + 0.33),
        ("set", ({"eff1": 0.9},)),
        ("write", ({"req": 1.0}, T0 + 0.34)),
        ("advance", T0 + 1.7),
        ("set", ({"stored": 3.0},)),
        ("write", ({"req": 1.0}, T0 + 1.8)),
        ("advance", T0 + 2.3),
        ("advance", T0 + 9.0),
    ]
    assert _run(sequence) == _run(sequence)


# -- facility neutrality -------------------------------------------------------


def test_the_module_names_no_facility_and_holds_no_number_beyond_zero_and_one() -> None:
    path = Path(injection_model.__file__)
    source = path.read_text()
    assert not re.search(
        r"\b(als|alsu|slac|lcls|nsls|aps|esrf|spear|booster|linac)\b", source, re.IGNORECASE
    )
    numbers = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    }
    assert numbers <= {0, 1, 2}
