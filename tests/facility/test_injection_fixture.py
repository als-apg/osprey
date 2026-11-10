"""The injection fixture facility builds, and its view serves the chain it states.

The fixture under ``tests/facility/fixtures/injection`` is the deckless
injection chain later tests serve live. These cases pin what those tests rely
on: the build and render succeed from a copy under the test's own directory;
the view serves exactly the chain's channels, each with its role, refresh class
and direction; every step lands well inside the busy window; and a composite
over the view, before any write, serves each parameter's start value.
"""

from __future__ import annotations

import pytest

import osprey_connectors.simulation.composite as composite_module
from osprey_connectors.simulation.composite import Composite
from osprey_connectors.simulation.view import SimulatorView
from tests.facility._injection_fixture import (
    BUSY,
    CHARGES,
    COUNTERS,
    CURRENT,
    CURRENT_FAST,
    EFFICIENCIES,
    MODEL,
    OFFSETS,
    PARAMETERS,
    READINGS,
    REQUEST,
    SETPOINTS,
    injection_view,
)

#: Each channel's ``(role, refresh, direction)`` in the built view.
EXPECTED = {
    REQUEST: ("setpoint", "pass", "write"),
    **dict.fromkeys(PARAMETERS, ("setpoint", "pass", "write")),
    **dict.fromkeys((*COUNTERS, *CHARGES, *EFFICIENCIES, BUSY), ("output", "event", "read")),
    CURRENT: ("output", "pass", "read"),
    CURRENT_FAST: ("output", "fast", "read"),
}


@pytest.fixture(scope="module")
def view(tmp_path_factory: pytest.TempPathFactory) -> SimulatorView:
    return SimulatorView.open(injection_view(tmp_path_factory.mktemp("injection")))


@pytest.fixture(autouse=True)
def _no_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """No stray config resolves a model log directory for the composite."""
    monkeypatch.setattr(composite_module, "default_config_path", lambda: None)


def test_the_fixture_builds_and_serves_exactly_the_chains_channels(view: SimulatorView) -> None:
    assert set(view.channels()) == {*SETPOINTS, *READINGS}
    assert len(SETPOINTS) + len(READINGS) == len(view.channels())
    assert [model.name for model in view.physics_models()] == [MODEL]
    assert view.model(MODEL).engine == "injection"
    assert {view.channel(address).owner for address in view.channels()} == {MODEL}


def test_the_fixture_states_the_chain_the_task_asks_for() -> None:
    assert len(PARAMETERS) == 4
    assert all(start != 0 for start in PARAMETERS.values())
    assert (len(COUNTERS), len(CHARGES), len(EFFICIENCIES)) == (2, 2, 2)


def test_each_channel_carries_its_role_refresh_and_direction(view: SimulatorView) -> None:
    bindings = {b.address: b for b in view.bindings(model=MODEL)}
    assert set(bindings) == set(EXPECTED)
    found = {a: (b.role, b.refresh, b.direction) for a, b in bindings.items()}
    assert found == EXPECTED


def test_setpoints_are_writable_and_readings_are_not(view: SimulatorView) -> None:
    assert all(view.channel(address).writable for address in SETPOINTS)
    assert not any(view.channel(address).writable for address in READINGS)


def test_every_step_lands_early_inside_the_busy_window(view: SimulatorView) -> None:
    cycle = view.model(MODEL).settings["injection"]["cycle"]
    assert set(OFFSETS) == {*COUNTERS, *CHARGES, *EFFICIENCIES, CURRENT}
    assert all(0 < offset for offset in OFFSETS.values())
    assert max(OFFSETS.values()) <= cycle["width_s"] / 5
    assert cycle["width_s"] < cycle["period_s"] <= 1.0


def test_a_composite_serves_each_parameters_start_value(view: SimulatorView) -> None:
    composite = Composite(view, model_log=False)
    assert composite.get(list(PARAMETERS)) == PARAMETERS
    assert composite.get(REQUEST) == 0.0


def test_both_currents_start_at_the_stored_current_parameter(view: SimulatorView) -> None:
    composite = Composite(view, model_log=False)
    stored = PARAMETERS["INJ:RING:CURRENT:SP"]
    assert composite.get([CURRENT, CURRENT_FAST]) == {CURRENT: stored, CURRENT_FAST: stored}
