"""One monitor's turn arrays from tracked centroids and alive shares.

Every effect of :func:`monitor_turn_arrays` is checked on its own -- the
layout, the sum, the stored beam, the floor, the position mix, the noise and
the tone -- and the output is checked to be the same in a second Python
process, since every draw is keyed on address, shot and sample index.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from osprey.simulation.engines.pyat_turn_readout import (
    NOISE_SUBKEY,
    TONE_SPIKE_M,
    TONE_SUBKEY,
    TurnReadout,
    monitor_turn_arrays,
    sample_draws,
)

ADDRESSES = ("SR:BPM1:TBT:X", "SR:BPM1:TBT:Y", "SR:BPM1:TBT:SUM")
TURNS = 50
TRIGGER = 10


def quiet(**changes: float) -> TurnReadout:
    """A readout with no noise and no tone, ``changes`` applied."""
    fields = {
        "sum_per_charge": 2.0,
        "sum_per_stored": 0.5,
        "sum_floor": 0.03,
        "position_noise": 0.0,
        "sum_noise": 0.0,
        "floor_noise": 0.0,
        "tone_period": 10,
        "tone_amplitude": 0.0,
    }
    fields.update(changes)
    return TurnReadout(**fields)


def arrays(readout: TurnReadout, **changes):
    """``monitor_turn_arrays`` on a beam centred at (1 mm, -2 mm) that survives."""
    call = {
        "alive": np.ones(TURNS),
        "centroid_x": np.full(TURNS, 1e-3),
        "centroid_y": np.full(TURNS, -2e-3),
        "trigger_sample": TRIGGER,
        "charge": 1.0,
        "stored_current": 0.0,
        "rf_on": True,
        "tone_on": False,
        "addresses": ADDRESSES,
        "shot": 3,
        "readout": readout,
    }
    call.update(changes)
    return monitor_turn_arrays(**call)


def test_the_arrays_hold_the_samples_before_the_trigger_then_one_per_turn() -> None:
    x, y, total = arrays(quiet())
    for array in (x, y, total):
        assert array.shape == (TRIGGER + TURNS,)
        assert array.dtype == np.float64
    assert np.all(total[:TRIGGER] == 0.03)
    assert np.all(total[TRIGGER:] == 2.0)
    np.testing.assert_allclose(x[TRIGGER:], 1e-3)
    np.testing.assert_allclose(y[TRIGGER:], -2e-3)


def test_a_zero_trigger_sample_starts_with_the_first_turn() -> None:
    x, _, total = arrays(quiet(), trigger_sample=0)
    assert total.shape == (TURNS,)
    assert total[0] == 2.0
    assert x[0] == pytest.approx(1e-3)


def test_the_sum_is_charge_times_alive_share_times_its_calibration() -> None:
    alive = np.linspace(1.0, 0.5, TURNS)
    _, _, total = arrays(quiet(sum_per_charge=4.0), alive=alive, charge=0.25)
    np.testing.assert_allclose(total[TRIGGER:], 0.25 * alive * 4.0)


def test_the_stored_beam_adds_to_the_sum_with_rf_on_also_before_the_trigger() -> None:
    _, _, total = arrays(quiet(), stored_current=2.0)
    np.testing.assert_allclose(total[:TRIGGER], 2.0 * 0.5)
    np.testing.assert_allclose(total[TRIGGER:], 2.0 + 1.0)


def test_with_rf_off_the_stored_beam_is_gone_and_the_pre_trigger_sum_is_the_floor() -> None:
    _, _, total = arrays(quiet(), stored_current=2.0, rf_on=False)
    assert np.all(total[:TRIGGER] == 0.03)
    np.testing.assert_allclose(total[TRIGGER:], 2.0)


def test_the_sum_never_falls_below_the_floor() -> None:
    alive = np.concatenate([np.ones(10), np.zeros(TURNS - 10)])
    _, _, total = arrays(quiet(sum_noise=0.5), alive=alive, charge=0.001)
    assert np.all(total >= 0.03)
    assert np.all(total[TRIGGER + 10 :] >= 0.03)


def test_a_lost_bunch_reads_the_floor_from_the_turn_it_is_lost() -> None:
    alive = np.concatenate([np.ones(5), np.zeros(TURNS - 5)])
    nan = np.where(alive > 0, 1e-3, np.nan)
    _, _, total = arrays(quiet(), alive=alive, centroid_x=nan, centroid_y=nan)
    np.testing.assert_allclose(total[TRIGGER : TRIGGER + 5], 2.0)
    assert np.all(total[TRIGGER + 5 :] == 0.03)


def test_zero_charge_reads_the_floor_with_no_stored_beam() -> None:
    _, _, total = arrays(quiet(), charge=0.0)
    assert np.all(total == 0.03)


def test_positions_are_the_sum_weighted_mix_of_stored_beam_at_zero_and_the_bunch() -> None:
    x, y, _ = arrays(quiet(), stored_current=2.0)
    # Injected sum 2.0, stored sum 1.0: two thirds of the bunch's centroid.
    np.testing.assert_allclose(x[TRIGGER:], 1e-3 * 2.0 / 3.0)
    np.testing.assert_allclose(y[TRIGGER:], -2e-3 * 2.0 / 3.0)
    # Before the trigger only the stored beam, on axis.
    np.testing.assert_allclose(x[:TRIGGER], 0.0)


def test_position_noise_adds_keyed_draws_where_beam_is_seen() -> None:
    clean_x, _, _ = arrays(quiet())
    noisy_x, _, _ = arrays(quiet(position_noise=1e-5))
    expected = 1e-5 * sample_draws(ADDRESSES[0], NOISE_SUBKEY, 3, TRIGGER + TURNS)
    np.testing.assert_allclose(noisy_x[TRIGGER:] - clean_x[TRIGGER:], expected[TRIGGER:])


def test_at_the_floor_positions_are_floor_noise_much_larger_than_the_beam() -> None:
    readout = quiet(position_noise=1e-6, floor_noise=1e-2)
    alive = np.concatenate([np.ones(20), np.zeros(TURNS - 20)])
    x, y, _ = arrays(readout, alive=alive)
    draws_x = sample_draws(ADDRESSES[0], NOISE_SUBKEY, 3, TRIGGER + TURNS)
    np.testing.assert_allclose(x[:TRIGGER], 1e-2 * draws_x[:TRIGGER])
    np.testing.assert_allclose(x[TRIGGER + 20 :], 1e-2 * draws_x[TRIGGER + 20 :])
    floor_spread = np.std(np.concatenate([x[:TRIGGER], x[TRIGGER + 20 :]]))
    beam_spread = np.std(x[TRIGGER : TRIGGER + 20])
    assert floor_spread > 100 * beam_spread
    assert np.std(y[TRIGGER + 20 :]) > abs(-2e-3)


def test_noise_is_keyed_on_address_shot_and_sample_index() -> None:
    readout = quiet(position_noise=1e-5, sum_noise=0.01)
    first = arrays(readout)
    again = arrays(readout)
    other_shot = arrays(readout, shot=4)
    for a, b in zip(first, again, strict=True):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[0], other_shot[0])
    # x and y read different streams because their addresses differ.
    assert not np.allclose(first[0] - 1e-3, first[1] + 2e-3)
    other_address = arrays(readout, addresses=("A:X", ADDRESSES[1], ADDRESSES[2]))
    assert not np.array_equal(first[0], other_address[0])
    np.testing.assert_array_equal(first[1], other_address[1])


def test_with_the_tone_flag_off_there_are_no_spikes() -> None:
    readout = quiet(tone_amplitude=1.0)
    clean = arrays(quiet())
    off = arrays(readout, tone_on=False)
    for a, b in zip(clean, off, strict=True):
        np.testing.assert_array_equal(a, b)


def test_with_the_tone_flag_on_periodic_spikes_appear_in_all_three() -> None:
    readout = quiet(tone_amplitude=1.0, tone_period=10)
    clean = arrays(quiet())
    toned = arrays(readout, tone_on=True)
    n = TRIGGER + TURNS
    on = np.arange(n) % 10 == 0
    for axis, (a, b) in enumerate(zip(clean, toned, strict=True)):
        delta = b - a
        assert np.all(delta[~on] == 0.0), axis
        assert np.all(delta[on] != 0.0), axis
    spikes_x = TONE_SPIKE_M * sample_draws(ADDRESSES[0], TONE_SUBKEY, 3, n)
    np.testing.assert_allclose((toned[0] - clean[0])[on], spikes_x[on])
    assert np.all(toned[2][on] > clean[2][on])


def test_a_toned_sum_stays_above_the_floor() -> None:
    _, _, total = arrays(quiet(tone_amplitude=1.0), tone_on=True, charge=0.0)
    assert np.all(total >= 0.03)


def test_the_inputs_must_be_one_sample_per_turn() -> None:
    with pytest.raises(ValueError, match="turn"):
        arrays(quiet(), centroid_x=np.zeros(TURNS - 1))
    with pytest.raises(ValueError, match="trigger"):
        arrays(quiet(), trigger_sample=-1)


_CHILD = """
import json, sys
import numpy as np
from osprey.simulation.engines.pyat_turn_readout import TurnReadout, monitor_turn_arrays
readout = TurnReadout(sum_per_charge=2.0, sum_per_stored=0.5, sum_floor=0.03,
    position_noise=1e-5, sum_noise=0.01, floor_noise=1e-3, tone_period=7,
    tone_amplitude=0.5)
turns = 40
alive = np.concatenate([np.linspace(1, 0.2, 30), np.zeros(10)])
cx = np.where(alive > 0, np.linspace(1e-3, -1e-3, turns), np.nan)
cy = np.where(alive > 0, np.linspace(-5e-4, 5e-4, turns), np.nan)
out = monitor_turn_arrays(alive=alive, centroid_x=cx, centroid_y=cy, trigger_sample=8,
    charge=0.7, stored_current=1.5, rf_on=True, tone_on=True,
    addresses=("SR:BPM1:TBT:X", "SR:BPM1:TBT:Y", "SR:BPM1:TBT:SUM"), shot=12,
    readout=readout)
print(json.dumps([[v.hex() for v in a.tolist()] for a in out]))
"""


def test_the_output_is_identical_in_a_second_python_process() -> None:
    def run() -> list:
        done = subprocess.run(
            [sys.executable, "-c", _CHILD], capture_output=True, text=True, check=True
        )
        return json.loads(done.stdout)

    first = run()
    second = run()
    assert first == second
    exec_globals: dict = {}
    exec(_CHILD.replace("print(", "result = ("), exec_globals)
    here = [[v.hex() for v in a.tolist()] for a in exec_globals["out"]]
    assert here == first
