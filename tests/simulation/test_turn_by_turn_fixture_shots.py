"""The turn-by-turn fixture's shots behave as the TxT manual shows, end to end.

Each case builds the turn-by-turn fixture facility, serves it from an
evolving composite on a fake clock, and drives it as a client would: it
writes the monitors' arm records, the ring's switches and the chain's
request, and reads the ring's records back. No case reaches into an engine:
the injection chain makes every shot and the ring takes it as its inputs.

One case per behaviour (requirements 3 to 10): the armed handshake; a shot
while not armed, arming twice, a reset and a scenario change; the bunch lost
with the RF off and kept with it on; the stored beam before the trigger; the
pilot tone; shot-to-shot variation and reproduction in a second process; a
corrector change; and the noise where the sum is at the floor.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

at = pytest.importorskip("at")

from osprey.simulation.engines import pyat as engine  # noqa: E402
from osprey.simulation.engines.pyat_turn_readout import TurnReadout  # noqa: E402
from osprey_connectors.simulation.composite import Composite  # noqa: E402
from tests.facility._turn_by_turn_fixture import (  # noqa: E402
    ARMED,
    ARMS,
    CURRENT_SETPOINT,
    DECK,
    FIXTURE,
    MONITORS,
    PILOT_TONE,
    REQUEST,
    RF_SWITCH,
    RING,
    SAMPLES,
    TURN_SUM,
    TURN_X,
    TURN_Y,
    TURNS,
    TURNS_TRACKED,
    turn_by_turn_view,
)

#: The fake clock's start, in seconds.
T0 = 1000.0

#: The time from a request until every chain step has landed, in seconds.
SHOT_SETTLES_S = 0.1

#: The chain's cycle period, so each request finds the chain idle.
CYCLE_S = 1.0

#: Every turn array's address.
ARRAYS = frozenset((*TURN_X.values(), *TURN_Y.values(), *TURN_SUM.values()))

#: The stored current a stored-beam case sets on the chain.
STORED_CURRENT = 5.0

#: The stored current one shot adds: ring charge 1.5 * 0.8 * 0.6 at 2.0 per charge.
SHOT_CURRENT = 1.5 * 0.8 * 0.6 * 2.0

#: The corrector a corrector case changes, and the change, in its hardware unit (A).
CORRECTOR = "HCM36"
CORRECTOR_SETPOINT = "SR:MAG:HCM:36:CURRENT:SP"
CORRECTOR_CHANGE_A = 200.0

#: The scenario the render always carries.
SCENARIO = "still"


def _sample(turn: int, trigger_sample: int) -> int:
    """The array index of turn ``turn`` (turn 1 is the first injected turn)."""
    return trigger_sample + turn - 1


def _ring_settings() -> Any:
    models = yaml.safe_load((FIXTURE / "models.yaml").read_text())
    return next(model for model in models if model["name"] == RING)["settings"]


@pytest.fixture(scope="module")
def block() -> Any:
    prepared = engine.prepare(DECK, _ring_settings(), model=RING)
    assert prepared.turn_by_turn is not None
    return prepared.turn_by_turn


@pytest.fixture(scope="module")
def readout(block: Any) -> TurnReadout:
    return block.readout


@pytest.fixture(scope="module")
def view(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return turn_by_turn_view(tmp_path_factory.mktemp("turn_by_turn_shots"))


class _Clock:
    def __init__(self) -> None:
        self.t_s = T0

    def __call__(self) -> float:
        return self.t_s


class _Ring:
    """The fixture served from an evolving composite, driven by its records."""

    def __init__(self, view: Path, *, state_dir: Path | None = None) -> None:
        self.clock = _Clock()
        self.composite = Composite(
            view, state_dir=state_dir, clock=self.clock, evolve=True, model_log=False
        )
        self.requests = 0
        self.composite.emitted()

    def later(self, seconds: float = CYCLE_S) -> None:
        self.clock.t_s += seconds

    def write(self, values: dict[str, float]) -> frozenset[str]:
        """Write ``values`` a cycle later; the names that changed."""
        self.later()
        self.composite.set(values)
        return self.composite.emitted()

    def arm(self) -> frozenset[str]:
        return self.write({ARMS[monitor]: 1.0 for monitor in MONITORS})

    def shoot(self) -> frozenset[str]:
        """Request a shot, let every chain step land, advance; the names that changed."""
        self.requests += 1
        self.write({REQUEST: float(self.requests)})
        self.later(SHOT_SETTLES_S)
        self.composite.advance()
        return self.composite.emitted()

    def settle(self) -> frozenset[str]:
        """One more advance at the same time; the names that changed."""
        self.composite.advance()
        return self.composite.emitted()

    def read(self, name: str) -> Any:
        return self.composite.get(name)

    def armed(self) -> np.ndarray:
        return np.array([float(self.read(ARMED[monitor])) for monitor in MONITORS])

    def arrays(self, addresses: dict[str, str]) -> np.ndarray:
        """One array per monitor, in deck order, as ``(monitors, samples)``."""
        return np.array([np.asarray(self.read(addresses[m]), dtype=float) for m in MONITORS])

    def all_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.arrays(TURN_X), self.arrays(TURN_Y), self.arrays(TURN_SUM)


def _armed_shot(view: Path, writes: dict[str, float] | None = None, shots: int = 1) -> _Ring:
    """A ring with ``writes`` applied, then ``shots`` armed shots."""
    ring = _Ring(view)
    if writes:
        ring.write(writes)
    for _ in range(shots):
        ring.arm()
        ring.shoot()
        ring.settle()
    return ring


@pytest.fixture(scope="module")
def rf_off(view: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _armed_shot(view, {RF_SWITCH: 0.0}).all_arrays()


@pytest.fixture(scope="module")
def rf_on(view: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _armed_shot(view, {RF_SWITCH: 1.0}).all_arrays()


# -- requirement 3: the armed handshake -----------------------------------------


def test_an_armed_shot_publishes_its_arrays_first_then_armed_drops_to_0(view: Path) -> None:
    ring = _Ring(view)
    assert (ring.armed() == 0.0).all()

    armed_up = ring.arm()
    assert set(ARMED.values()) <= armed_up
    assert (ring.armed() == 1.0).all()

    ring.requests += 1
    ring.write({REQUEST: float(ring.requests)})
    ring.later(SHOT_SETTLES_S)
    ring.composite.advance()
    # The arrays are readable while armed still reads 1.
    assert float(ring.read(TURNS_TRACKED)) == TURNS
    x, y, total = ring.all_arrays()
    assert x.shape == y.shape == total.shape == (len(MONITORS), SAMPLES)
    assert (total > 0.5).any(axis=1).all()
    assert (ring.armed() == 1.0).all()

    landed = ring.composite.emitted()
    assert ARRAYS <= landed
    assert TURNS_TRACKED in landed
    assert not set(ARMED.values()) & landed

    # Once the arrays are out, the next advance drops armed to 0, on its own.
    assert (ring.armed() == 0.0).all()
    dropped = ring.composite.emitted()
    assert set(ARMED.values()) <= dropped
    assert not ARRAYS & dropped


# -- requirement 4: unarmed, armed twice, reset, scenario change ----------------


def test_a_shot_while_not_armed_leaves_the_arrays_unchanged(view: Path) -> None:
    ring = _Ring(view)
    before = ring.all_arrays()

    landed = ring.shoot()
    landed |= ring.settle()

    assert not ARRAYS & landed
    assert float(ring.read(TURNS_TRACKED)) == 0.0
    for start, now in zip(before, ring.all_arrays(), strict=True):
        np.testing.assert_array_equal(now, start)


def test_arming_twice_is_harmless(view: Path) -> None:
    ring = _Ring(view)
    ring.arm()

    again = ring.arm()
    assert not set(ARMED.values()) & again
    assert (ring.armed() == 1.0).all()

    assert ARRAYS <= ring.shoot()
    ring.settle()
    assert (ring.armed() == 0.0).all()
    after_one = ring.all_arrays()

    # One shot spent the arming: the next finds nothing armed.
    landed = ring.shoot() | ring.settle()
    assert not ARRAYS & landed
    for one, now in zip(after_one, ring.all_arrays(), strict=True):
        np.testing.assert_array_equal(now, one)


def _no_shot_follows(ring: _Ring) -> None:
    """Advances without a request track nothing, whatever the delivery after the change."""
    landed: frozenset[str] = frozenset()
    before = ring.all_arrays()
    tracked = float(ring.read(TURNS_TRACKED))
    for _ in range(3):
        ring.later()
        landed |= ring.settle()
    ring.arm()
    ring.later()
    landed |= ring.settle()
    assert not ARRAYS & landed
    assert float(ring.read(TURNS_TRACKED)) == tracked
    for start, now in zip(before, ring.all_arrays(), strict=True):
        np.testing.assert_array_equal(now, start)


def test_a_reset_clears_armed_and_is_not_a_shot(view: Path) -> None:
    ring = _Ring(view)
    ring.arm()
    ring.shoot()
    ring.settle()
    ring.arm()
    assert (ring.armed() == 1.0).all()

    ring.composite.reset()
    ring.composite.emitted()

    assert (ring.armed() == 0.0).all()
    _no_shot_follows(ring)


def _activate(state_dir: Path, *names: str) -> None:
    """Write the active set, moving the file's modification time forward."""
    path = state_dir / "active_scenarios"
    before = path.stat().st_mtime_ns if path.exists() else 0
    path.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")
    stamp = max(before + 1_000_000, path.stat().st_mtime_ns)
    os.utime(path, ns=(stamp, stamp))


def test_a_scenario_change_clears_armed_and_is_not_a_shot(view: Path, tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    ring = _Ring(view, state_dir=state_dir)
    ring.arm()
    ring.shoot()
    ring.settle()
    ring.arm()
    assert (ring.armed() == 1.0).all()

    _activate(state_dir, SCENARIO)
    ring.later()

    assert (ring.armed() == 0.0).all()
    ring.composite.emitted()
    _no_shot_follows(ring)


# -- requirement 5: RF off the bunch is lost and decoheres; RF on it survives ---


def _oscillation(x: np.ndarray, first: int, last: int, trigger_sample: int) -> np.ndarray:
    """Each monitor's betatron amplitude over turns ``first..last``: the std about a line.

    The line takes out the slow orbit drift of the energy loss, so what is
    left is the oscillation of the centroid about it.
    """
    window = x[:, _sample(first, trigger_sample) : _sample(last, trigger_sample) + 1]
    turns = np.arange(window.shape[1])
    fit = np.polynomial.polynomial.polyfit(turns, window.T, 1)
    line = np.polynomial.polynomial.polyval(turns, fit)
    return np.asarray((window - line).std(axis=1))


def test_rf_off_the_sum_decays_to_the_floor_and_x_decoheres(
    rf_off: tuple[np.ndarray, np.ndarray, np.ndarray], block: Any, readout: TurnReadout
) -> None:
    x, _, total = rf_off
    trigger = block.trigger_sample
    first = total[:, _sample(1, trigger)]

    assert (total[:, _sample(110, trigger)] > 0.5 * first).all()
    assert (total[:, _sample(210, trigger)] < 0.5 * first).all()
    np.testing.assert_array_equal(total[:, _sample(TURNS, trigger)], readout.sum_floor)

    early = _oscillation(x, 1, 20, trigger)
    late = _oscillation(x, 161, 180, trigger)
    assert (late / early <= 0.5).all()


def test_rf_on_the_sum_holds_every_turn(
    rf_on: tuple[np.ndarray, np.ndarray, np.ndarray], block: Any
) -> None:
    _, _, total = rf_on
    trigger = block.trigger_sample
    first = total[:, _sample(1, trigger) : _sample(1, trigger) + 1]

    assert (total[:, trigger:] >= 0.95 * first).all()


# -- requirement 6: the stored beam before the trigger ---------------------------


def _pre_trigger(view: Path, *, rf: float, current: float, trigger_sample: int) -> np.ndarray:
    ring = _armed_shot(view, {RF_SWITCH: rf, CURRENT_SETPOINT: current})
    return ring.arrays(TURN_SUM)[:, :trigger_sample]


def test_with_rf_on_stored_current_shows_before_the_trigger(
    view: Path, block: Any, readout: TurnReadout
) -> None:
    pre = _pre_trigger(view, rf=1.0, current=STORED_CURRENT, trigger_sample=block.trigger_sample)

    level = STORED_CURRENT * readout.sum_per_stored
    assert level > readout.sum_floor + 10 * readout.sum_noise
    assert (pre > readout.sum_floor).all()
    assert abs(pre.mean() - level) < readout.sum_noise
    assert np.abs(pre - level).max() < 6 * readout.sum_noise


def test_with_rf_off_the_pre_trigger_sum_is_the_floor_whatever_the_current(
    view: Path, block: Any, readout: TurnReadout
) -> None:
    pre = _pre_trigger(view, rf=0.0, current=STORED_CURRENT, trigger_sample=block.trigger_sample)

    np.testing.assert_array_equal(pre, readout.sum_floor)


def test_at_zero_current_the_pre_trigger_sum_is_the_floor(
    view: Path, block: Any, readout: TurnReadout
) -> None:
    pre = _pre_trigger(view, rf=1.0, current=0.0, trigger_sample=block.trigger_sample)

    np.testing.assert_array_equal(pre, readout.sum_floor)


# -- requirement 7: the pilot tone -----------------------------------------------


def test_the_pilot_tone_spikes_every_tone_period_in_x_y_and_sum_and_off_none(
    view: Path, readout: TurnReadout
) -> None:
    tone_on = _armed_shot(view, {PILOT_TONE: 1.0}).all_arrays()
    tone_off = _armed_shot(view, {PILOT_TONE: 0.0}).all_arrays()
    on_tone = np.arange(SAMPLES) % readout.tone_period == 0

    for on, off in zip(tone_on, tone_off, strict=True):
        spikes = on - off
        assert (spikes[:, on_tone] != 0.0).all()
        np.testing.assert_array_equal(spikes[:, ~on_tone], 0.0)
    sum_spikes = tone_on[2] - tone_off[2]
    assert (sum_spikes[:, on_tone] > 0.0).all()
    assert np.median(sum_spikes[:, on_tone]) > 10 * readout.sum_noise


# -- requirement 8: shot to shot, and across processes ----------------------------


def test_consecutive_shots_differ_by_jitter_and_noise_and_the_stored_sum_grows(
    view: Path, block: Any, readout: TurnReadout
) -> None:
    ring = _Ring(view)
    trigger = block.trigger_sample
    shots = []
    for _ in range(2):
        ring.arm()
        ring.shoot()
        ring.settle()
        shots.append(ring.all_arrays())
    (x1, y1, sum1), (x2, y2, sum2) = shots

    first = _sample(1, trigger)
    for one, two, jitter in ((x1, x2, block.jitter[0]), (y1, y2, block.jitter[1])):
        moved = np.abs(two[:, first] - one[:, first])
        # Noise alone moves a sample by about position_noise; jitter by far more.
        assert np.median(moved) > 3 * readout.position_noise
        assert moved.max() < 100 * jitter
    # The first shot's stored current was zero; the second holds one shot's worth.
    np.testing.assert_array_equal(sum1[:, :trigger], readout.sum_floor)
    grown = sum2[:, :trigger].mean() - sum1[:, :trigger].mean()
    assert abs(grown - (SHOT_CURRENT * readout.sum_per_stored - readout.sum_floor)) < (
        readout.sum_noise
    )


#: A second process: serve the view, run the same writes, print every array.
_CHILD = """
import json, sys
from pathlib import Path
import numpy as np
from osprey_connectors.simulation.composite import Composite
view, writes = Path(sys.argv[1]), json.loads(sys.argv[2])
clock = {"t": 1000.0}
composite = Composite(view, clock=lambda: clock["t"], evolve=True, model_log=False)
composite.emitted()
for step in writes:
    clock["t"] += step["dt"]
    if step["values"]:
        composite.set(step["values"])
    else:
        composite.advance()
    composite.emitted()
print(json.dumps({name: [v.hex() for v in np.asarray(composite.get(name), float).tolist()]
                  for name in sys.argv[3:]}))
"""


def _writes() -> list[dict[str, Any]]:
    """Two armed shots with RF and tone on: arm, request, settle, twice."""
    arm = {ARMS[monitor]: 1.0 for monitor in MONITORS}
    steps: list[dict[str, Any]] = [{"dt": CYCLE_S, "values": {PILOT_TONE: 1.0}}]
    for shot in (1.0, 2.0):
        steps += [
            {"dt": CYCLE_S, "values": arm},
            {"dt": CYCLE_S, "values": {REQUEST: shot}},
            {"dt": SHOT_SETTLES_S, "values": {}},
            {"dt": 0.0, "values": {}},
        ]
    return steps


def test_the_same_writes_give_identical_arrays_in_two_python_processes(view: Path) -> None:
    names = [TURN_X["BPM01"], TURN_Y["BPM37"], TURN_SUM["BPM72"]]
    writes = _writes()

    def run() -> dict[str, list[str]]:
        done = subprocess.run(
            [sys.executable, "-c", _CHILD, str(view), json.dumps(writes), *names],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(done.stdout.strip().splitlines()[-1])

    first, second = run(), run()
    assert first == second

    ring = _Ring(view)
    for step in writes:
        ring.later(step["dt"])
        if step["values"]:
            ring.composite.set(step["values"])
        else:
            ring.composite.advance()
        ring.composite.emitted()
    here = {name: [v.hex() for v in np.asarray(ring.read(name), float).tolist()] for name in names}
    assert here == first
    assert float(ring.read(TURNS_TRACKED)) == TURNS


# -- requirement 9: a corrector change --------------------------------------------


def test_a_corrector_change_between_shots_moves_the_first_turn_downstream_only(
    view: Path, block: Any, readout: TurnReadout
) -> None:
    lattice = at.load_lattice(str(DECK))
    names = [element.FamName for element in lattice]
    corrector = names.index(CORRECTOR)
    injection = block.injection_index
    where = np.array([names.index(monitor) for monitor in MONITORS])
    downstream = where > corrector
    upstream = (injection < where) & (where < corrector)
    assert downstream.sum() > 10 and upstream.sum() > 10

    def second_shot(change: float) -> np.ndarray:
        ring = _Ring(view)
        ring.arm()
        ring.shoot()
        ring.settle()
        ring.write({CORRECTOR_SETPOINT: change})
        ring.arm()
        ring.shoot()
        ring.settle()
        return ring.arrays(TURN_X)[:, _sample(1, block.trigger_sample)]

    held, changed = second_shot(0.0), second_shot(CORRECTOR_CHANGE_A)

    # Upstream the shot is the same shot: same jitter, same noise.
    assert np.abs(changed[upstream] - held[upstream]).max() < readout.position_noise
    moved = np.abs(changed[downstream] - held[downstream])
    assert np.median(moved) > 5 * block.jitter[0]


# -- requirement 10: noise where the sum is at the floor ---------------------------


def test_where_the_sum_is_at_the_floor_x_and_y_read_as_noise(
    rf_off: tuple[np.ndarray, np.ndarray, np.ndarray], block: Any, readout: TurnReadout
) -> None:
    x, y, total = rf_off
    trigger = block.trigger_sample
    floor = total <= readout.sum_floor
    # Before the trigger (no stored beam with the RF off) and after the loss.
    assert floor[:, :trigger].all()
    assert floor[:, _sample(TURNS, trigger)].all()

    for axis in (x, y):
        noise = axis[floor]
        assert 0.8 * readout.floor_noise < noise.std() < 1.2 * readout.floor_noise
        assert abs(noise.mean()) < 0.1 * readout.floor_noise
        # Much larger than the noise on a beam the monitor sees.
        assert noise.std() > 50 * readout.position_noise
