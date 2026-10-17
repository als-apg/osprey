"""The pulsed-septum trace: shape, linearity, enable, drift and the shape digest."""

from __future__ import annotations

import numpy as np
import pytest

from osprey_connectors.simulation import pulse

T0 = 1_760_000_000.0
ADDRESS = "SR:INJ:SEK:01:CURRENT:WAVEFORM"
THIRTY_MINUTES = np.arange(0.0, 1800.0, 5.0)


def _rise_edge(values: np.ndarray) -> int:
    """First sample at or above half the peak."""
    return int(np.argmax(values >= 0.5 * values.max()))


def _edge_times(values: np.ndarray, low: float, high: float) -> tuple[int, int]:
    above_low = np.flatnonzero(values >= low)
    above_high = np.flatnonzero(values >= high)
    return int(above_low[0]), int(above_high[0])


def test_the_trace_has_the_declared_length():
    values = pulse.trace(ADDRESS, T0, 5.0, True)
    assert values.shape == (500,)
    assert pulse.TRACE_POINTS == 500
    assert values.dtype == np.float64


def test_the_same_inputs_give_the_identical_trace():
    first = pulse.trace(ADDRESS, T0 + 12.345, 3.0, True)
    second = pulse.trace(ADDRESS, T0 + 12.345, 3.0, True)
    np.testing.assert_array_equal(first, second)


def test_time_is_taken_at_whole_milliseconds():
    np.testing.assert_array_equal(
        pulse.trace(ADDRESS, T0 + 1.0001, 3.0, True), pulse.trace(ADDRESS, T0 + 1.0, 3.0, True)
    )


def test_a_different_instant_or_address_gives_a_different_trace():
    base = pulse.trace(ADDRESS, T0, 3.0, True)
    assert not np.array_equal(base, pulse.trace(ADDRESS, T0 + 1.0, 3.0, True))
    assert not np.array_equal(base, pulse.trace("SR:INJ:SEN:01:CURRENT:WAVEFORM", T0, 3.0, True))


@pytest.mark.parametrize("offset", [0.0, 97.0, 1234.5])
def test_the_peak_ratio_equals_the_driver_ratio(offset):
    low = pulse.trace(ADDRESS, T0 + offset, 2.0, True)
    high = pulse.trace(ADDRESS, T0 + offset, 7.0, True)
    assert high.max() / low.max() == pytest.approx(3.5, rel=0.01)


def test_the_peak_follows_the_driver_level():
    values = pulse.trace(ADDRESS, T0, 4.0, True)
    assert values.max() == pytest.approx(4.0, rel=0.03)


def test_a_disabled_supply_reads_flat_zero():
    values = pulse.trace(ADDRESS, T0, 5.0, False)
    assert values.shape == (500,)
    assert np.all(values == 0.0)


def test_the_pulse_rises_faster_than_it_falls():
    values = pulse.trace(ADDRESS, T0, 5.0, True)
    peak_index = int(np.argmax(values))
    peak = values.max()
    rise_10, rise_90 = _edge_times(values, 0.1 * peak, 0.9 * peak)
    tail = values[peak_index:]
    fall_90 = peak_index + int(np.flatnonzero(tail <= 0.9 * peak)[0])
    fall_10 = peak_index + int(np.flatnonzero(tail <= 0.1 * peak)[0])
    assert rise_90 - rise_10 < fall_10 - fall_90


def test_the_flat_top_droops_slightly():
    clean = pulse._unit_shape(np.arange(500, dtype=np.float64) - pulse.RISE_START_SAMPLE)
    flat_start = int(pulse.RISE_START_SAMPLE + pulse.RISE_SAMPLES)
    flat_end = int(flat_start + pulse.FLAT_TOP_SAMPLES) - 1
    assert clean[flat_start] > clean[flat_end] > 0.9
    assert clean[0] == 0.0


def test_the_peak_drifts_over_thirty_minutes_by_at_most_three_percent():
    peaks = np.array([pulse.trace(ADDRESS, T0 + dt, 5.0, True).max() for dt in THIRTY_MINUTES])
    spread = (peaks.max() - peaks.min()) / peaks.mean()
    assert 0.0 < spread <= 0.03


def test_the_rise_edge_drifts_by_a_bounded_number_of_samples():
    edges = np.array(
        [_rise_edge(pulse.trace(ADDRESS, T0 + dt, 5.0, True)) for dt in THIRTY_MINUTES]
    )
    shift = int(edges.max() - edges.min())
    assert 1 <= shift <= pulse.MAX_TIMING_DRIFT_SAMPLES


def test_the_shape_constants_are_module_constants():
    for name in pulse.SHAPE_CONSTANT_NAMES:
        assert isinstance(getattr(pulse, name), (int, float))
    public_numbers = {
        name
        for name, value in vars(pulse).items()
        if name.isupper() and not name.startswith("_") and isinstance(value, (int, float))
    }
    assert public_numbers == set(pulse.SHAPE_CONSTANT_NAMES)


def test_shape_version_is_the_digest_of_the_current_constants():
    assert pulse.shape_version() == pulse.SHAPE_VERSION
    assert isinstance(pulse.SHAPE_VERSION, str) and pulse.SHAPE_VERSION


@pytest.mark.parametrize("name", pulse.SHAPE_CONSTANT_NAMES)
def test_changing_any_shape_constant_changes_shape_version(monkeypatch, name):
    original = getattr(pulse, name)
    monkeypatch.setattr(pulse, name, original + 1 if isinstance(original, int) else original * 1.1)
    assert pulse.shape_version() != pulse.SHAPE_VERSION
