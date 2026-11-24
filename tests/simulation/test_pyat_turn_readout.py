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
from pathlib import Path

import numpy as np
import pytest

from osprey.simulation.engines import pyat_pickup
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


def test_the_monitors_faults_act_on_the_beam_position_before_floor_and_tone() -> None:
    def faults(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return 2.0 * x + 1e-4, y - 3e-4

    alive = np.ones(TURNS)
    alive[30:] = 0.0
    readout = quiet(position_noise=1e-5, floor_noise=1e-3, tone_amplitude=1.0)
    plain = arrays(readout, alive=alive, tone_on=True)
    read = arrays(readout, alive=alive, tone_on=True, faults=faults)
    beam = np.zeros(TRIGGER + TURNS, dtype=bool)
    beam[TRIGGER : TRIGGER + 30] = True
    tone = np.arange(TRIGGER + TURNS) % 10 == 0
    seen = beam & ~tone
    # The position noise is on the beam position the faults read.
    np.testing.assert_allclose(read[0][seen], 2.0 * plain[0][seen] + 1e-4, atol=1e-15)
    np.testing.assert_allclose(read[1][seen], plain[1][seen] - 3e-4, atol=1e-15)
    # With no beam there is nothing to read: the floor noise, untouched.
    np.testing.assert_array_equal(read[0][~beam], plain[0][~beam])
    np.testing.assert_array_equal(read[1][~beam], plain[1][~beam])
    np.testing.assert_array_equal(read[2], plain[2])


def test_without_faults_the_arrays_are_the_ones_of_a_perfect_monitor() -> None:
    readout = quiet(position_noise=1e-5, floor_noise=1e-3, tone_amplitude=1.0)
    plain = arrays(readout, tone_on=True)
    same = arrays(readout, tone_on=True, faults=lambda x, y: (x, y))
    for one, other in zip(plain, same, strict=True):
        np.testing.assert_array_equal(one, other)


def test_a_sum_gain_of_one_is_the_perfect_monitors_sum_bit_for_bit() -> None:
    readout = quiet(position_noise=1e-5, sum_noise=0.01, floor_noise=1e-3, tone_amplitude=1.0)
    alive = np.linspace(1.0, 0.0, TURNS)
    plain = arrays(readout, alive=alive, stored_current=0.3, tone_on=True)
    unit = arrays(readout, alive=alive, stored_current=0.3, tone_on=True, sum_gain=1.0)
    for one, other in zip(plain, unit, strict=True):
        np.testing.assert_array_equal(one, other)


def test_a_sum_gain_scales_the_injected_and_the_stored_beam_sum() -> None:
    alive = np.linspace(1.0, 0.5, TURNS)
    _, _, plain = arrays(quiet(), alive=alive, stored_current=0.4)
    _, _, low = arrays(quiet(), alive=alive, stored_current=0.4, sum_gain=0.8)
    _, _, high = arrays(quiet(), alive=alive, stored_current=0.4, sum_gain=1.2)
    np.testing.assert_allclose(low, 0.8 * plain, rtol=1e-15)
    np.testing.assert_allclose(high / low, 1.5, rtol=1e-14)
    # Before the trigger the sum is the stored beam's alone.
    np.testing.assert_allclose(high[:TRIGGER], 1.2 * 0.4 * 0.5, rtol=1e-15)


def test_a_sum_gain_leaves_the_positions_and_their_mix_alone() -> None:
    readout = quiet(position_noise=1e-5, floor_noise=1e-3)
    alive = np.ones(TURNS)
    alive[30:] = 0.0
    x, y, _ = arrays(readout, alive=alive, stored_current=0.2)
    x_gained, y_gained, _ = arrays(readout, alive=alive, stored_current=0.2, sum_gain=0.1)
    np.testing.assert_array_equal(x_gained, x)
    np.testing.assert_array_equal(y_gained, y)


def test_the_sum_noise_is_added_after_the_gain_at_its_own_width() -> None:
    readout = quiet(sum_noise=0.01, sum_floor=0.0)
    _, _, clean = arrays(quiet(sum_floor=0.0))
    _, _, plain = arrays(readout)
    _, _, gained = arrays(readout, sum_gain=3.0)
    np.testing.assert_allclose(gained - 3.0 * clean, plain - clean, atol=1e-14)
    assert 0.007 < (gained - 3.0 * clean)[TRIGGER:].std() < 0.013


def test_the_floor_holds_under_a_gained_sum() -> None:
    _, _, total = arrays(quiet(), charge=0.01, sum_gain=0.5)
    assert np.all(total == 0.03)


# A monitor that states a pickup is read button by button.

PICKUP_RADIUS = 0.016
STATED_PICKUP = {
    "shape": "round",
    "radius": PICKUP_RADIUS,
    "buttons": [45.0, 135.0, 225.0, 315.0],
    "half_width": 14.0,
    "x": {"plus": [1, 4], "minus": [2, 3]},
    "y": {"plus": [1, 2], "minus": [3, 4]},
}


def refuse(detail: str, remedy: str) -> Exception:
    return ValueError(f"{detail}; {remedy}")


def pickup(**changes):
    return pyat_pickup.parse({**STATED_PICKUP, **changes}, "pickup", Path("."), refuse)


def reading(buttons: np.ndarray, signs: np.ndarray, scale: float) -> float:
    return float(scale * (signs @ buttons) / buttons.sum())


def test_a_centred_beam_on_a_pickup_reads_the_perfect_monitors_arrays() -> None:
    readout = quiet(position_noise=1e-5, sum_noise=1e-3, floor_noise=2e-3, tone_amplitude=1.0)
    centred = {
        "centroid_x": np.zeros(TURNS),
        "centroid_y": np.zeros(TURNS),
        "alive": np.linspace(1.0, 0.0, TURNS),
        "stored_current": 1.5,
        "tone_on": True,
    }
    plain = arrays(readout, **centred)
    read = arrays(readout, pickup=pickup(), **centred)
    for with_pickup, without in zip(read, plain, strict=True):
        np.testing.assert_allclose(with_pickup, without, rtol=1e-14, atol=1e-18)


def test_the_stored_beam_alone_reads_zero_on_a_pickup() -> None:
    x, y, total = arrays(quiet(), pickup=pickup(), charge=0.0, stored_current=4.0)
    np.testing.assert_allclose(total, 2.0, rtol=1e-14)
    np.testing.assert_allclose(x, 0.0, atol=1e-18)
    np.testing.assert_allclose(y, 0.0, atol=1e-18)


def test_a_small_offset_reads_true_and_a_large_one_reads_short_on_a_pickup() -> None:
    small = arrays(
        quiet(), pickup=pickup(), centroid_x=np.full(TURNS, 2e-6), centroid_y=np.full(TURNS, -3e-6)
    )
    np.testing.assert_allclose(small[0][TRIGGER:], 2e-6, rtol=1e-6)
    np.testing.assert_allclose(small[1][TRIGGER:], -3e-6, rtol=1e-6)
    large = arrays(
        quiet(), pickup=pickup(), centroid_x=np.full(TURNS, 8e-3), centroid_y=np.zeros(TURNS)
    )
    assert np.all((large[0][TRIGGER:] > 4e-3) & (large[0][TRIGGER:] < 7.5e-3))
    np.testing.assert_allclose(large[1][TRIGGER:], 0.0, atol=1e-15)


def test_the_arrays_are_the_difference_over_sum_of_the_button_responses() -> None:
    stated = pickup()
    centroid_x = np.linspace(-6e-3, 6e-3, TURNS)
    centroid_y = np.linspace(4e-3, -5e-3, TURNS)
    x, y, total = arrays(quiet(), pickup=stated, centroid_x=centroid_x, centroid_y=centroid_y)
    for turn in (0, 7, 31, TURNS - 1):
        buttons = stated.response(centroid_x[turn], centroid_y[turn])
        assert total[TRIGGER + turn] == pytest.approx(2.0 * buttons.sum(), rel=1e-13)
        assert x[TRIGGER + turn] == pytest.approx(
            reading(buttons, stated.signs_x, stated.scale_x), rel=1e-12
        )
        assert y[TRIGGER + turn] == pytest.approx(
            reading(buttons, stated.signs_y, stated.scale_y), rel=1e-12
        )


def test_the_sum_is_larger_towards_a_button_and_smaller_on_the_axes() -> None:
    offset = 6e-3
    on_axis = arrays(
        quiet(), pickup=pickup(), centroid_x=np.full(TURNS, offset), centroid_y=np.zeros(TURNS)
    )[2]
    to_button = arrays(
        quiet(),
        pickup=pickup(),
        centroid_x=np.full(TURNS, offset / np.sqrt(2.0)),
        centroid_y=np.full(TURNS, offset / np.sqrt(2.0)),
    )[2]
    assert np.all(on_axis[TRIGGER:] < 2.0)
    assert np.all(to_button[TRIGGER:] > 2.0)
    assert to_button[TRIGGER] - 2.0 == pytest.approx(2.0 - on_axis[TRIGGER], rel=0.05)


def test_the_stored_and_the_injected_beam_add_at_the_buttons() -> None:
    stated = pickup()
    x, y, total = arrays(
        quiet(),
        pickup=stated,
        centroid_x=np.full(TURNS, 6e-3),
        centroid_y=np.full(TURNS, 5e-3),
        stored_current=3.0,
    )
    buttons = 2.0 * stated.response(6e-3, 5e-3) + 1.5 * stated.centre
    assert total[TRIGGER] == pytest.approx(buttons.sum(), rel=1e-13)
    assert x[TRIGGER] == pytest.approx(reading(buttons, stated.signs_x, stated.scale_x), rel=1e-12)
    assert y[TRIGGER] == pytest.approx(reading(buttons, stated.signs_y, stated.scale_y), rel=1e-12)
    # Not the sum-weighted mix of the two beams' own readings.
    alone = reading(stated.response(6e-3, 5e-3), stated.signs_x, stated.scale_x)
    assert abs(x[TRIGGER] - alone * 2.0 / 3.5) > 1e-5
    np.testing.assert_allclose(x[:TRIGGER], 0.0, atol=1e-18)


def test_a_weak_button_shifts_a_centred_beams_reading_by_the_first_order_amount() -> None:
    stated = pickup()
    centred = {"centroid_x": np.zeros(TURNS), "centroid_y": np.zeros(TURNS)}
    x, y, total = arrays(quiet(), pickup=stated, button_gains=[0.95, 1.0, 1.0, 1.0], **centred)
    # Button 1 adds to both planes: a share d weaker shifts each by -scale * d / 4.
    np.testing.assert_allclose(x[TRIGGER:], -stated.scale_x * 0.05 / 4.0, rtol=2e-2)
    np.testing.assert_allclose(y[TRIGGER:], -stated.scale_y * 0.05 / 4.0, rtol=2e-2)
    np.testing.assert_allclose(x[TRIGGER:], -stated.scale_x * 0.05 / 3.95, rtol=1e-12)
    np.testing.assert_allclose(total[TRIGGER:], 2.0 * 3.95 / 4.0, rtol=1e-13)
    other = arrays(quiet(), pickup=stated, button_gains=[1.0, 0.95, 1.0, 1.0], **centred)
    np.testing.assert_allclose(other[0][TRIGGER:], stated.scale_x * 0.05 / 3.95, rtol=1e-12)
    np.testing.assert_allclose(other[1][TRIGGER:], -stated.scale_y * 0.05 / 3.95, rtol=1e-12)


def test_unit_button_gains_are_the_pickup_without_gains_bit_for_bit() -> None:
    readout = quiet(position_noise=1e-5, sum_noise=1e-3)
    plain = arrays(readout, pickup=pickup())
    gained = arrays(readout, pickup=pickup(), button_gains=[1.0, 1.0, 1.0, 1.0])
    for with_gains, without in zip(gained, plain, strict=True):
        np.testing.assert_array_equal(with_gains, without)


def test_dead_buttons_read_as_no_beam() -> None:
    readout = quiet(floor_noise=2e-3)
    x, y, total = arrays(readout, pickup=pickup(), button_gains=[0.0, 0.0, 0.0, 0.0])
    assert np.all(total == 0.03)
    for position, address in ((x, ADDRESSES[0]), (y, ADDRESSES[1])):
        assert np.all(np.isfinite(position))
        np.testing.assert_array_equal(
            position, 2e-3 * sample_draws(address, NOISE_SUBKEY, 3, TRIGGER + TURNS)
        )


def test_beam_is_seen_on_the_button_sum_with_its_gains_before_the_sum_gain() -> None:
    readout = quiet(floor_noise=2e-3, sum_floor=1.0)
    weak = arrays(readout, pickup=pickup(), button_gains=[0.4] * 4, sum_gain=5.0)
    floor = 2e-3 * sample_draws(ADDRESSES[0], NOISE_SUBKEY, 3, TRIGGER + TURNS)
    np.testing.assert_array_equal(weak[0], floor)
    np.testing.assert_allclose(weak[2][TRIGGER:], 5.0 * 0.8 * weak_sum_share(), rtol=1e-13)
    seen = arrays(readout, pickup=pickup(), button_gains=[0.6] * 4, sum_gain=0.1)
    assert np.all(seen[0][TRIGGER:] != floor[TRIGGER:])
    assert np.all(seen[2][TRIGGER:] == 1.0)


def weak_sum_share() -> float:
    return float(pickup().response(1e-3, -2e-3).sum())


def test_the_sum_gain_scales_a_pickups_sum_and_leaves_its_positions() -> None:
    plain = arrays(quiet(), pickup=pickup(), stored_current=1.0)
    gained = arrays(quiet(), pickup=pickup(), stored_current=1.0, sum_gain=0.8)
    np.testing.assert_array_equal(gained[0], plain[0])
    np.testing.assert_array_equal(gained[1], plain[1])
    np.testing.assert_allclose(gained[2], 0.8 * plain[2], rtol=1e-14)


def test_the_monitors_faults_act_on_what_the_pickup_reads() -> None:
    def faults(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return 1.1 * x + 1e-4, -y

    plain = arrays(quiet(), pickup=pickup())
    faulted = arrays(quiet(), pickup=pickup(), faults=faults)
    np.testing.assert_allclose(faulted[0][TRIGGER:], 1.1 * plain[0][TRIGGER:] + 1e-4, rtol=1e-14)
    np.testing.assert_allclose(faulted[1][TRIGGER:], -plain[1][TRIGGER:], rtol=1e-14)
    np.testing.assert_array_equal(faulted[2], plain[2])


def test_position_noise_is_added_to_a_pickups_reading_where_beam_is_seen() -> None:
    plain = arrays(quiet(), pickup=pickup())
    noisy = arrays(quiet(position_noise=1e-5), pickup=pickup())
    draws = sample_draws(ADDRESSES[0], NOISE_SUBKEY, 3, TRIGGER + TURNS)
    np.testing.assert_allclose(noisy[0][TRIGGER:], plain[0][TRIGGER:] + 1e-5 * draws[TRIGGER:])
    np.testing.assert_array_equal(noisy[0][:TRIGGER], 0.0)


def test_a_turn_with_no_centroid_reads_the_floor_on_a_pickup() -> None:
    alive = np.ones(TURNS)
    alive[20:] = 0.0
    centroid = np.full(TURNS, 1e-3)
    centroid[20:] = np.nan
    x, y, total = arrays(
        quiet(), pickup=pickup(), alive=alive, centroid_x=centroid, centroid_y=centroid
    )
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(total))
    assert np.all(total[TRIGGER + 20 :] == 0.03)
    assert np.all(x[TRIGGER + 20 :] == 0.0)


def test_a_table_pickup_reads_as_the_round_one_it_tabulates(tmp_path) -> None:
    grid = np.linspace(-8e-3, 8e-3, 33)
    mesh_x, mesh_y = np.meshgrid(grid, grid)
    values = pyat_pickup.round_response(
        mesh_x, mesh_y, PICKUP_RADIUS, np.radians(STATED_PICKUP["buttons"]), np.radians(14.0)
    )
    (tmp_path / "pickup.json").write_text(
        json.dumps({"x": grid.tolist(), "y": grid.tolist(), "buttons": values.tolist()})
    )

    tabulated = pyat_pickup.parse(
        {
            "shape": "table",
            "table": "pickup.json",
            "buttons": 4,
            "interpolation": "bicubic",
            "x": STATED_PICKUP["x"],
            "y": STATED_PICKUP["y"],
        },
        "pickup",
        tmp_path,
        refuse,
    )
    centroid_x = np.linspace(-6e-3, 6e-3, TURNS)
    centroid_y = np.linspace(4e-3, -5e-3, TURNS)
    beam = {"centroid_x": centroid_x, "centroid_y": centroid_y}
    analytic = arrays(quiet(), pickup=pickup(), **beam)
    read = arrays(quiet(), pickup=tabulated, **beam)
    np.testing.assert_allclose(read[2], analytic[2], rtol=1e-4)
    np.testing.assert_allclose(read[0], analytic[0], rtol=3e-3, atol=2e-6)
    np.testing.assert_allclose(read[1], analytic[1], rtol=3e-3, atol=2e-6)
