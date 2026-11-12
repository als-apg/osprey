"""The turn-by-turn fixture facility builds, its view carries the ring's inputs, and it is calibrated.

The fixture under ``tests/facility/fixtures/turn_by_turn`` is the facility
later tests serve live: an injection chain feeding the example ring on pyAT.
These cases pin what those tests rely on: the build and render succeed from a
copy under the test's own directory; the chain's stored current may be zero
and its ring shot counter steps after the ring charge; the ring carries every
turn-by-turn record and takes the chain's shot, charge and stored current as
inputs; and the ring's aperture and bunch sizes give the loss windows the
shot tests measure, tracked with the tracking core on the fixture's own deck
and settings.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
import yaml

at = pytest.importorskip("at")

from osprey.simulation.engines import pyat as engine  # noqa: E402
from osprey.simulation.engines import pyat_turns as tracking  # noqa: E402
from osprey.simulation.engines.pyat_turn_settings import TurnSettings  # noqa: E402
from osprey_connectors.simulation.view import SimulatorView  # noqa: E402
from tests.facility._turn_by_turn_fixture import (  # noqa: E402
    ARMED,
    ARMS,
    CHAIN,
    CHARGE,
    CURRENT,
    CURRENT_SETPOINT,
    DECK,
    FIXTURE,
    INPUTS,
    MONITORS,
    PILOT_TONE,
    RF_SWITCH,
    RING,
    SAMPLES,
    SHOT,
    STORED,
    TURN_SUM,
    TURN_X,
    TURN_Y,
    TURNS,
    TURNS_TRACKED,
    turn_by_turn_view,
)

#: The shots the calibration is checked on; each draws its own jitter.
SHOTS = (1, 2, 3)


def _stub_take_inputs(model: Any, inputs: Any, t_s: float, *, baseline: bool) -> None:
    """Stands in for the pyat engine's ``take_inputs`` until the engine states one."""


@pytest.fixture(scope="module")
def view(tmp_path_factory: pytest.TempPathFactory) -> SimulatorView:
    # The facility build refuses inputs on an engine without ``take_inputs``.
    # This fixture checks the tree, not the engine: while the pyat engine
    # states none yet, a no-op stands in for the build alone.
    with pytest.MonkeyPatch.context() as patch:
        if not callable(getattr(engine, "take_inputs", None)):
            patch.setattr(engine, "take_inputs", _stub_take_inputs, raising=False)
        return SimulatorView.open(turn_by_turn_view(tmp_path_factory.mktemp("turn_by_turn")))


def _ring_settings() -> Any:
    models = yaml.safe_load((FIXTURE / "models.yaml").read_text())
    return next(model for model in models if model["name"] == RING)["settings"]


@pytest.fixture(scope="module")
def block() -> TurnSettings:
    prepared = engine.prepare(DECK, _ring_settings(), model=RING)
    assert prepared.turn_by_turn is not None
    return prepared.turn_by_turn


@pytest.fixture(scope="module")
def ring() -> Any:
    return at.load_lattice(str(DECK))


def _alive(ring: Any, block: TurnSettings, *, rf_on: bool, shot: int) -> np.ndarray:
    """The share alive per monitor and turn for one shot, tracked as the fixture states it."""
    lattice = copy.deepcopy(ring)
    if not rf_on:
        for index in lattice.get_uint32_index(at.RFCavity):
            lattice[index].Voltage = 0.0
    assert block.aperture is not None
    half_axes = (block.aperture.x, block.aperture.y)
    tracking.write_apertures(lattice, **{block.aperture.shape: half_axes})
    shape = tracking.declared_bunch(
        ring,
        block.injection_index,
        block.particles,
        emittance_x=block.emittance_x,
        emittance_y=block.emittance_y,
        sigma_delta=block.sigma_delta,
        sigma_z=block.sigma_z,
    )
    monitors = np.asarray(ring.get_uint32_index(at.Monitor), dtype=np.int64)
    reading = tracking.track_shot(
        lattice,
        block.injection_index,
        monitors,
        block.launch,
        block.turns,
        shape=shape,
        key=RING,
        shot=shot,
        jitter=block.jitter,
    )
    return reading.alive


# -- the build and the view ---------------------------------------------------


def test_the_fixture_builds_both_models(view: SimulatorView) -> None:
    assert [model.name for model in view.physics_models()] == [CHAIN, RING]
    assert view.model(CHAIN).engine == "injection"
    assert view.model(RING).engine == "pyat"


def test_the_view_carries_the_rings_inputs(view: SimulatorView) -> None:
    inputs = {item["engine"]["signal"]: item["address"] for item in view.model(RING).inputs}
    assert inputs == INPUTS
    assert {view.channel(address).owner for address in INPUTS.values()} == {CHAIN}
    assert view.model(CHAIN).inputs == ()


# -- the chain ----------------------------------------------------------------


def test_the_chains_stored_current_may_be_zero(view: SimulatorView) -> None:
    bindings = {b.address: b for b in view.bindings(model=CHAIN)}
    assert bindings[CURRENT_SETPOINT].record["engine"]["start"] == 0.0
    band = view.channel(CURRENT_SETPOINT).value_range
    assert band is not None
    assert band[0] == 0


def test_the_ring_shot_counter_steps_after_the_ring_charge(view: SimulatorView) -> None:
    bindings = {b.address: b for b in view.bindings(model=CHAIN)}
    assert bindings[SHOT].record["engine"]["signal"] == "counter"
    assert bindings[CHARGE].record["engine"]["signal"] == "charge"
    assert bindings[CURRENT].record["engine"]["signal"] == "current"
    assert (
        bindings[SHOT].record["engine"]["offset_s"] > bindings[CHARGE].record["engine"]["offset_s"]
    )


# -- the ring -----------------------------------------------------------------


def test_the_ring_carries_every_turn_by_turn_record(view: SimulatorView) -> None:
    signals = {
        b.address: (b.record["engine"].get("signal"), b.element, b.direction)
        for b in view.bindings(model=RING)
    }
    assert signals[RF_SWITCH] == ("rf_switch", None, "write")
    assert signals[PILOT_TONE] == ("pilot_tone", None, "write")
    assert signals[TURNS_TRACKED] == ("turns_tracked", None, "read")
    assert [signals[address][0::2] for address in STORED] == [("stored", "write")] * 2
    for table, signal, direction in (
        (ARMS, "arm", "write"),
        (ARMED, "armed", "read"),
        (TURN_X, "turn_x", "read"),
        (TURN_Y, "turn_y", "read"),
        (TURN_SUM, "turn_sum", "read"),
    ):
        assert tuple(table) == MONITORS
        for element, address in table.items():
            assert signals[address] == (signal, element, direction)


def test_every_monitor_of_the_deck_has_its_records(ring: Any) -> None:
    deck_monitors = [ring[int(index)].FamName for index in ring.get_uint32_index(at.Monitor)]
    assert tuple(deck_monitors) == MONITORS


def test_each_turn_array_is_a_waveform_of_the_rings_samples(
    view: SimulatorView, block: TurnSettings
) -> None:
    assert (block.turns, block.samples) == (TURNS, SAMPLES)
    for table in (TURN_X, TURN_Y, TURN_SUM):
        for address in table.values():
            channel = view.channel(address)
            assert (channel.value_type, channel.shape) == ("waveform", (SAMPLES,))
            assert not channel.writable


def test_the_ring_keeps_its_orbit_monitors_and_correctors(view: SimulatorView) -> None:
    bindings = {b.address: b for b in view.bindings(model=RING)}
    for element in MONITORS:
        for axis in ("X", "Y"):
            assert bindings[f"SR:DIAG:BPM:{element[3:]}:POSITION:{axis}"].element == element
    correctors = [
        b for b in bindings.values() if b.record["engine"].get("attribute") == "KickAngle"
    ]
    assert len(correctors) == 288
    setpoints = [b.address for b in correctors if b.engine_role == "setpoint"]
    assert len(setpoints) == 144
    assert all(view.channel(address).writable for address in setpoints)


# -- the calibration ----------------------------------------------------------


@pytest.mark.parametrize("shot", SHOTS)
def test_rf_off_the_bunch_is_lost_between_turns_110_and_210(
    ring: Any, block: TurnSettings, shot: int
) -> None:
    alive = _alive(ring, block, rf_on=False, shot=shot)
    assert alive.shape == (len(MONITORS), TURNS)
    assert alive[:, 110 - 1].min() > 0.8
    assert alive[:, 210 - 1].max() < 0.5
    assert alive[:, TURNS - 1].max() == 0.0


@pytest.mark.parametrize("shot", SHOTS)
def test_rf_on_the_bunch_survives_every_turn(ring: Any, block: TurnSettings, shot: int) -> None:
    alive = _alive(ring, block, rf_on=True, shot=shot)
    assert alive.shape == (len(MONITORS), TURNS)
    assert alive.min() >= 0.95
