"""The ``burst`` and ``loss`` archiver event shapes.

A vacuum burst jumps at its instant and pumps down over seconds; a beam loss
drops the stored current in one step and only top-off refills it, one injector
shot a second (0.75 mA on a 500 mA ceiling) from the first refill instant after
the loss. Neither moves a sample before its instant: an effect never precedes
its cause.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from osprey_connectors.simulation.series import apply_events, loss_refill_bound

ANCHOR = 1_762_300_000.0
TOPOFF = {"ceiling": 500.0, "lifetime_h": 7.0, "period_s": 31.0}
BASE = 3.7e-10
BURST = {
    "shape": "burst",
    "at_offset": -600.0,
    "amplitude": 1.5e-7,
    "decay_s": 5.0,
    "tail_share": 0.1,
    "tail_decay_s": 25.0,
}
LOSS = {"shape": "loss", "at_offset": -600.0, "amount": 18.0}
AT = ANCHOR - 600.0
#: One injector shot: 0.15% of the 500 mA ceiling.
SHOT = 0.75


def _grid(before: float, after: float, step: float = 1.0, at: float = AT) -> np.ndarray:
    return at + np.arange(-before, after + step, step)


def _burst(t: np.ndarray, event: dict = BURST) -> np.ndarray:
    """The burst's excursion above a flat baseline at the instants ``t``."""
    return apply_events(np.full(t.size, BASE), [event], t.size, t, ANCHOR) - BASE


def _current(t: np.ndarray, topoff: dict | None = TOPOFF, at: float = AT) -> np.ndarray:
    """A flat 500 mA level with the loss applied, at the instants ``t``."""
    event = dict(LOSS, at_offset=at - ANCHOR)
    return apply_events(np.full(t.size, 500.0), [event], t.size, t, ANCHOR, None, topoff)


class TestBurst:
    def test_nothing_moves_before_the_instant(self):
        t = _grid(300, 0)
        assert np.all(_burst(t)[:-1] == 0.0)

    def test_the_peak_is_the_amplitude_at_the_instant(self):
        assert _burst(np.array([AT]))[0] == pytest.approx(1.5e-7)

    def test_the_rise_is_one_sample_and_the_fall_takes_many(self):
        """A Gaussian rises as slowly as it falls; a burst rises within one sample."""
        t = _grid(120, 600)
        value = _burst(t)
        peak = int(np.argmax(value))
        rise = np.flatnonzero(value[:peak] < 0.1 * value[peak])
        fall = np.flatnonzero(value[peak:] < 0.1 * value[peak])
        rise_s = t[peak] - t[rise[-1]]
        fall_s = t[peak + fall[0]] - t[peak]
        assert rise_s <= 1.0
        assert fall_s >= 10 * rise_s

    def test_the_slow_tail_outlasts_the_fast_pump_down(self):
        t = np.array([AT + 60.0])
        fast_only = {k: v for k, v in BURST.items() if not k.startswith("tail_")}
        assert _burst(t)[0] > 1e3 * _burst(t, fast_only)[0]
        assert _burst(t)[0] == pytest.approx(
            1.5e-7 * (0.9 * math.exp(-12) + 0.1 * math.exp(-60 / 25))
        )


class TestLoss:
    def test_it_drops_in_one_step_at_the_instant(self):
        current = _current(_grid(5, 0))
        assert list(current[:-1]) == [500.0] * 5
        assert current[-1] == 482.0

    def test_without_top_off_it_never_recovers(self):
        assert np.all(_current(_grid(0, 7200, 60.0), topoff=None) == 482.0)
        assert loss_refill_bound(LOSS, None) is None

    def test_it_holds_until_the_first_refill_instant_after_it(self):
        period = TOPOFF["period_s"]
        refill = (math.floor(AT / period) + 1) * period
        t = _grid(0, 120, 0.5)
        current = _current(t)
        assert np.all(current[t < refill] == 482.0)
        assert current[t == refill][0] == pytest.approx(482.0 + SHOT)

    def test_top_off_refills_it_one_shot_a_second(self):
        """The current climbs only in shot-sized steps at the injection rate, never smoothly."""
        t = _grid(0, 120, 0.25)
        current = _current(t)
        steps = np.diff(current)
        assert np.all(steps >= -1e-12), "the current never falls back"
        rises = steps[steps > 1e-12]
        assert np.allclose(rises, SHOT)
        assert np.allclose(np.diff(t[1:][steps > 1e-12]), 1.0)
        assert current[-1] == 500.0
        assert rises.size == 24

    def test_the_refill_bound_covers_every_phase(self):
        bound = loss_refill_bound(LOSS, TOPOFF)
        for shift in np.arange(0.0, TOPOFF["period_s"], 0.5):
            at = AT + shift
            t = at + np.arange(0.0, bound + 1.0, 0.5)
            current = _current(t, at=at)
            assert current[t >= at + bound].min() == 500.0, shift
