"""Direct unit tests for the pure synthesis primitives in ``simulation.series``.

These cover the stateless functions in isolation (no engine). The engine-driven
behavior is exercised separately in ``test_series.py``; this file locks the
module's standalone contract so the extracted primitives can be refactored
without going through the full engine each time.
"""

import hashlib
import math
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from osprey.simulation import series as series_module
from osprey.simulation.expressions import ExpressionError
from osprey.simulation.series import (
    apply_events,
    clamp,
    daily_occurrences,
    epoch_seconds_array,
    event_positions,
    keyed_normals,
    pv_key_bytes,
    ref_value,
    string_series,
    wander,
)

# --------------------------------------------------------------------------
# Deliberately-slow per-sample reference implementation of the pinned draw
# construction. It lives here (not in src/) on purpose: it is the independent
# witness that the vectorized batch path consumes exactly two words per sample
# and therefore stays pointwise-deterministic. Pure Python ints, one sample at
# a time, no numpy — the opposite implementation strategy from the real one.
# --------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_MIX_A = 0xBF58476D1CE4E5B9
_MIX_B = 0x94D049BB133111EB
_TWO_POW_M53 = 2.0**-53


def _reference_mix64(z: int) -> int:
    """splitmix64 finalizer over Python ints."""
    z = ((z ^ (z >> 30)) * _MIX_A) & _MASK64
    z = ((z ^ (z >> 27)) * _MIX_B) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def _reference_key_words(pv_key: bytes) -> tuple[int, int]:
    digest = hashlib.sha256(pv_key).digest()
    return int.from_bytes(digest[0:8], "big"), int.from_bytes(digest[8:16], "big")


def _reference_words(pv_key: bytes, counter: int) -> tuple[int, int]:
    """The two words consumed by a single sample."""
    k0, k1 = _reference_key_words(pv_key)
    base = (counter * _GAMMA) & _MASK64
    return _reference_mix64(base ^ k0), _reference_mix64(base ^ k1)


def _reference_normal(pv_key: bytes, counter: int) -> float:
    w0, w1 = _reference_words(pv_key, counter)
    u1 = ((w0 >> 11) | 1) * _TWO_POW_M53
    u2 = (w1 >> 11) * _TWO_POW_M53
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _unxor_shift(y: int, shift: int) -> int:
    """Invert ``x -> x ^ (x >> shift)`` by fixed-point iteration."""
    x = y
    for _ in range(64 // shift + 1):
        x = y ^ (x >> shift)
    return x


def _reference_unmix64(z: int) -> int:
    """Inverse of :func:`_reference_mix64` (the finalizer is a bijection)."""
    z = _unxor_shift(z, 31)
    z = (z * pow(_MIX_B, -1, 1 << 64)) & _MASK64
    z = _unxor_shift(z, 27)
    z = (z * pow(_MIX_A, -1, 1 << 64)) & _MASK64
    return _unxor_shift(z, 30)


def _counter_producing_zero_word(pv_key: bytes, lane: int) -> int:
    """The unique counter whose ``lane`` word mixes to exactly zero."""
    key_word = _reference_key_words(pv_key)[lane]
    base = _reference_unmix64(0) ^ key_word
    return (base * pow(_GAMMA, -1, 1 << 64)) & _MASK64


class TestClamp:
    def test_within_bounds_unchanged(self):
        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_clamps_to_min_and_max(self):
        assert clamp(-1.0, 0.0, 10.0) == 0.0
        assert clamp(99.0, 0.0, 10.0) == 10.0

    def test_optional_bounds(self):
        assert clamp(-50.0, None, 10.0) == -50.0
        assert clamp(50.0, 0.0, None) == 50.0
        assert clamp(123.0, None, None) == 123.0


class TestEpochSecondsArray:
    def test_datetime_objects(self):
        dts = [datetime(2026, 1, 1, 0, 0, 0), datetime(2026, 1, 1, 1, 0, 0)]
        out = epoch_seconds_array(dts)
        assert out is not None
        np.testing.assert_allclose(out, [dts[0].timestamp(), dts[1].timestamp()])

    def test_epoch_floats_pass_through(self):
        out = epoch_seconds_array([100.0, 200, 300.5])
        np.testing.assert_allclose(out, [100.0, 200.0, 300.5])

    def test_non_convertible_returns_none(self):
        assert epoch_seconds_array(["not-a-time", 1.0]) is None

    def test_bool_is_rejected(self):
        # bool is an int subclass but must not be treated as an epoch second.
        assert epoch_seconds_array([True]) is None


class TestRefValue:
    def test_numeric_lookup(self):
        ref_series = {"PV:A": np.array([1.0, 2.0, 3.0])}
        assert ref_value(ref_series, "PV:A", 2) == 3.0

    def test_string_series_raises(self):
        ref_series = {"PV:STATE": ["OPEN", "CLOSED"]}
        with pytest.raises(ExpressionError, match="cannot be used in an expression"):
            ref_value(ref_series, "PV:STATE", 0)


class TestEventPositions:
    def test_fraction_position(self):
        t_frac = np.linspace(0.0, 1.0, 5)
        positions = event_positions({"shape": "step", "at": 0.5, "to": 1.0}, t_frac, None, 0.0)
        assert len(positions) == 1
        axis, at = positions[0]
        assert at == 0.5
        assert axis is t_frac

    def test_offset_anchored_needs_abs_axis(self):
        t_frac = np.linspace(0.0, 1.0, 5)
        assert event_positions({"at_offset": 10.0}, t_frac, None, 100.0) == []

    def test_offset_anchored_with_abs_axis(self):
        t_frac = np.linspace(0.0, 1.0, 5)
        t_abs = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        positions = event_positions({"at_offset": 10.0}, t_frac, t_abs, 100.0)
        assert len(positions) == 1
        axis, at = positions[0]
        assert at == 110.0
        assert axis is t_abs

    def test_time_of_day_needs_abs_axis(self):
        t_frac = np.linspace(0.0, 1.0, 5)
        assert event_positions({"at_time": "12:00:00"}, t_frac, None, 0.0) == []


class TestDailyOccurrences:
    """``at_time`` occurrences are placed in the facility timezone, not the box's
    local zone. The tests pass an explicit ``tz`` and build expected positions in
    that same zone, then assert the result is independent of the host ``$TZ``."""

    def test_one_per_calendar_date_in_window(self):
        tz = ZoneInfo("UTC")
        start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=tz)
        end = datetime(2026, 1, 2, 23, 0, 0, tzinfo=tz)
        t_abs = np.array([start.timestamp(), end.timestamp()])
        occ = daily_occurrences("12:00:00", t_abs, tz=tz)
        assert len(occ) == 2
        assert occ[0] == datetime(2026, 1, 1, 12, 0, 0, tzinfo=tz).timestamp()
        assert occ[1] == datetime(2026, 1, 2, 12, 0, 0, tzinfo=tz).timestamp()

    def test_occurrence_outside_window_excluded(self):
        # Window ends before noon on the only date → no occurrence.
        tz = ZoneInfo("UTC")
        start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=tz)
        end = datetime(2026, 1, 1, 6, 0, 0, tzinfo=tz)
        t_abs = np.array([start.timestamp(), end.timestamp()])
        assert daily_occurrences("12:00:00", t_abs, tz=tz) == []

    def test_positions_are_box_tz_independent(self):
        """A UTC ``tz`` yields UTC-noon epochs even when the host ``$TZ`` is a
        wildly different zone — this is the regression that broke the archiver
        ``at_time`` benchmark on non-UTC deploy boxes."""
        tz = ZoneInfo("UTC")
        start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=tz)
        end = datetime(2026, 1, 1, 23, 0, 0, tzinfo=tz)
        t_abs = np.array([start.timestamp(), end.timestamp()])
        expected = datetime(2026, 1, 1, 12, 0, 0, tzinfo=tz).timestamp()

        old_tz = os.environ.get("TZ")
        try:
            os.environ["TZ"] = "Pacific/Kiritimati"  # UTC+14, maximally far from UTC
            time.tzset()
            occ = daily_occurrences("12:00:00", t_abs, tz=tz)
        finally:
            if old_tz is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = old_tz
            time.tzset()

        assert occ == [expected]

    def test_dst_boundary_one_occurrence_per_date(self):
        """Characterization: across a spring-forward DST boundary the daily walk
        still yields exactly one (local-noon) occurrence per calendar date — no
        skipped or duplicated date. Noon is unambiguous (well clear of the 02:00
        gap), so this pins the current contract and would catch a future cursor
        refactor that miscounts dates across the transition."""
        ny = ZoneInfo("America/New_York")
        # 2026-03-08 is the US spring-forward date (02:00 -> 03:00 EST->EDT).
        start = datetime(2026, 3, 7, 0, 0, tzinfo=ny)
        end = datetime(2026, 3, 10, 23, 0, tzinfo=ny)
        t_abs = np.array([start.timestamp(), end.timestamp()])

        occ = daily_occurrences("12:00:00", t_abs, tz=ny)

        assert len(occ) == 4  # Mar 7, 8, 9, 10
        dates = {datetime.fromtimestamp(e, ny).date() for e in occ}
        assert len(dates) == 4  # one per distinct calendar date
        for epoch in occ:
            local = datetime.fromtimestamp(epoch, ny)
            assert (local.hour, local.minute) == (12, 0)


class TestStringSeries:
    def test_step_switches_value(self):
        out = string_series("OPEN", [{"shape": "step", "at": 0.5, "to": "CLOSED"}], 5, None, 0.0)
        # t_frac = [0, .25, .5, .75, 1]; switch at >= 0.5 → last 3 entries.
        assert out == ["OPEN", "OPEN", "CLOSED", "CLOSED", "CLOSED"]

    def test_non_step_events_ignored_for_strings(self):
        out = string_series("OPEN", [{"shape": "spike", "at": 0.5}], 3, None, 0.0)
        assert out == ["OPEN", "OPEN", "OPEN"]


class TestApplyEvents:
    def test_empty_events_returns_input(self):
        series = np.array([1.0, 2.0, 3.0])
        assert apply_events(series, [], 3, None, 0.0) is series

    def test_step(self):
        series = np.zeros(5)
        out = apply_events(series, [{"shape": "step", "at": 0.5, "to": 9.0}], 5, None, 0.0)
        np.testing.assert_allclose(out, [0.0, 0.0, 9.0, 9.0, 9.0])

    def test_ramp_interpolates(self):
        series = np.zeros(5)
        out = apply_events(
            series, [{"shape": "ramp", "at": 0.0, "until": 1.0, "to": 4.0}], 5, None, 0.0
        )
        np.testing.assert_allclose(out, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_spike_adds_gaussian_bump(self):
        series = np.zeros(11)
        out = apply_events(
            series, [{"shape": "spike", "at": 0.5, "amplitude": 2.0, "width": 0.1}], 11, None, 0.0
        )
        # Peak at the center of the window.
        assert out.argmax() == 5
        assert out[5] == pytest.approx(2.0, rel=1e-6)
        # Input is not mutated (apply_events copies).
        np.testing.assert_array_equal(series, np.zeros(11))


class TestPvKeyBytes:
    def test_is_sha256_of_the_utf8_name(self):
        assert pv_key_bytes("SR:BPM1:X") == hashlib.sha256(b"SR:BPM1:X").digest()

    def test_stable_and_distinct_per_name(self):
        assert pv_key_bytes("A") == pv_key_bytes("A")
        assert pv_key_bytes("A") != pv_key_bytes("B")

    def test_builtin_hash_is_not_used(self):
        """``hash()`` is PYTHONHASHSEED-randomized per process, so a key derived
        from it would silently change between runs. Same-name keys must be
        reproducible against a literal digest computed outside this process."""
        expected = bytes.fromhex("07b788d9e5c120d3471cb7245b835ab1bfd5ac347f77e92488b177f617ecaf2f")
        assert pv_key_bytes("SR:HCM:1:SP") == expected


class TestKeyedNormals:
    """The pinned fixed-consumption draw construction (FR12).

    ``Generator.normal(size=n)`` is forbidden on this path: it uses ziggurat
    rejection sampling with *variable* stream consumption, so a batch draw from
    a window-start counter does not equal the per-sample draws — pointwise
    determinism breaks invisibly. These tests pin the alternative.
    """

    def test_matches_per_sample_reference_exactly(self):
        key = pv_key_bytes("SR:BPM:07:X")
        counters = np.arange(1_764_500_000_000, 1_764_500_000_000 + 500, 100, dtype=np.uint64)

        out = keyed_normals(key, counters)

        expected = np.array([_reference_normal(key, int(c)) for c in counters])
        np.testing.assert_array_equal(out, expected)

    def test_word_stage_matches_reference_exactly(self):
        """Exactness anchor below the transcendentals: if the normals ever
        disagree while the words and uniforms still match bit-for-bit, the cause
        is a 1-ULP libm/SIMD difference, not a broken construction."""
        key = pv_key_bytes("SR:BPM:07:Y")
        counters = np.arange(1_764_500_000_000, 1_764_500_000_000 + 200, dtype=np.uint64)

        w0, w1 = series_module._keyed_words(key, counters)

        expected = [_reference_words(key, int(c)) for c in counters]
        np.testing.assert_array_equal(w0, np.array([w[0] for w in expected], dtype=np.uint64))
        np.testing.assert_array_equal(w1, np.array([w[1] for w in expected], dtype=np.uint64))

    def test_consumes_exactly_two_words_per_sample(self):
        """A batch of n samples uses the n-th sample's own counter, never a
        running stream position: drawing one sample in isolation reproduces its
        value inside any batch."""
        key = pv_key_bytes("SR:HCM:12:SP")
        counters = np.arange(9_000_000, 9_000_000 + 64, dtype=np.uint64)

        batch = keyed_normals(key, counters)

        for i in (0, 1, 17, 63):
            single = keyed_normals(key, np.array([counters[i]], dtype=np.uint64))
            assert single[0] == batch[i]

    def test_pointwise_stable_across_separate_calls(self):
        key = pv_key_bytes("SR:VCM:03:SP")
        counters = np.array([1, 2, 3, 1_700_000_000_000], dtype=np.uint64)

        np.testing.assert_array_equal(keyed_normals(key, counters), keyed_normals(key, counters))

    def test_pointwise_stable_under_slicing_and_reordering(self):
        key = pv_key_bytes("SR:VCM:04:SP")
        counters = np.arange(2_000_000, 2_000_100, dtype=np.uint64)

        full = keyed_normals(key, counters)

        np.testing.assert_array_equal(keyed_normals(key, counters[10:40]), full[10:40])
        np.testing.assert_array_equal(keyed_normals(key, counters[::-1]), full[::-1])
        order = np.array([73, 5, 5, 99, 0], dtype=np.intp)
        np.testing.assert_array_equal(keyed_normals(key, counters[order]), full[order])

    def test_overlapping_windows_agree_on_shared_counters(self):
        """Two windows that share timestamps must agree exactly on the overlap —
        the property that window-start-seeded batch draws cannot provide."""
        key = pv_key_bytes("SR:DCCT:CURRENT")
        window_a = np.arange(1_764_000_000_000, 1_764_000_060_000, 1000, dtype=np.uint64)
        window_b = np.arange(1_764_000_030_000, 1_764_000_090_000, 1000, dtype=np.uint64)

        draws_a = keyed_normals(key, window_a)
        draws_b = keyed_normals(key, window_b)

        np.testing.assert_array_equal(draws_a[30:], draws_b[:30])

    def test_distinct_keys_give_distinct_draws(self):
        counters = np.arange(500, dtype=np.uint64)

        a = keyed_normals(pv_key_bytes("SR:BPM:01:X"), counters)
        b = keyed_normals(pv_key_bytes("SR:BPM:01:Y"), counters)

        assert not np.any(a == b)
        assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.15

    def test_zero_mixed_word_stays_finite(self):
        """Open-interval mapping: ``u1 = ((w >> 11) | 1) * 2**-53`` keeps
        ``u1 > 0`` even for the one counter whose word mixes to exactly zero, so
        ``log(u1)`` can never be ``-inf``. A shared-mapping defect here would be
        invisible to the equivalence test, hence a dedicated test."""
        key = pv_key_bytes("SR:BPM:99:X")

        for lane in (0, 1):
            counter = _counter_producing_zero_word(key, lane)
            assert _reference_words(key, counter)[lane] == 0  # the setup is real

            counters = np.array([counter], dtype=np.uint64)
            w0, w1 = series_module._keyed_words(key, counters)
            assert int((w0 if lane == 0 else w1)[0]) == 0

            out = keyed_normals(key, counters)
            assert np.isfinite(out).all()
            assert out[0] == _reference_normal(key, counter)
            # sqrt(-2 * log(2**-53)) is the largest magnitude the mapping admits.
            assert abs(float(out[0])) <= math.sqrt(2.0 * 53.0 * math.log(2.0))

    def test_no_inf_or_nan_over_a_large_sweep(self):
        key = pv_key_bytes("SR:BPM:42:X")
        counters = np.arange(200_000, dtype=np.uint64)

        assert np.isfinite(keyed_normals(key, counters)).all()

    def test_distribution_sanity(self):
        key = pv_key_bytes("SR:BPM:11:X")
        counters = np.arange(100_000, dtype=np.uint64)

        out = keyed_normals(key, counters)

        assert abs(float(out.mean())) < 0.02
        assert float(out.std()) == pytest.approx(1.0, abs=0.02)
        # Box-Muller cosine branch is symmetric; tails should be populated.
        assert float(np.abs(out).max()) > 3.5

    def test_accepts_float_counters_by_rounding(self):
        """Callers derive counters as ``epoch_seconds * 1000``, which arrives as
        float64; the draw must key on the rounded integer millisecond."""
        key = pv_key_bytes("SR:BPM:05:X")

        out = keyed_normals(key, np.array([1_764_000_000_123.4, 1_764_000_000_122.5]))

        assert out[0] == _reference_normal(key, 1_764_000_000_123)
        assert out[1] == _reference_normal(key, 1_764_000_000_122)  # rint: half to even

    def test_accepts_signed_and_negative_counters(self):
        """Sample-index fallback counters are plain int64; pre-epoch timestamps
        make them negative. Both must key deterministically rather than raise."""
        key = pv_key_bytes("SR:BPM:06:X")
        counters = np.array([-2, -1, 0, 1], dtype=np.int64)

        out = keyed_normals(key, counters)

        assert np.isfinite(out).all()
        np.testing.assert_array_equal(out, keyed_normals(key, counters))
        assert out[2] == _reference_normal(key, 0)
        assert out[0] == _reference_normal(key, (-2) & _MASK64)

    def test_shape_dtype_and_empty_input(self):
        key = pv_key_bytes("SR:BPM:08:X")

        out = keyed_normals(key, np.arange(6, dtype=np.uint64).reshape(2, 3))
        assert out.shape == (2, 3)
        assert out.dtype == np.float64

        empty = keyed_normals(key, np.array([], dtype=np.uint64))
        assert empty.shape == (0,)
        assert empty.dtype == np.float64

    def test_subkeys_are_independent_streams(self):
        """Relative and absolute noise draw from the same counters; a distinct
        subkey keeps the two streams from being the same number twice."""
        counters = np.arange(1000, dtype=np.uint64)
        base = pv_key_bytes("SR:BPM:09:X")

        rel = keyed_normals(base + b":noise", counters)
        absolute = keyed_normals(base + b":noise_abs", counters)

        assert not np.any(rel == absolute)

    def test_perf_144_channels_by_10000_samples(self):
        """FR12 perf sanity: the pointwise-correct per-generator loop this
        construction replaces cost ~11.6 s for the same workload."""
        counters = np.arange(1_764_000_000_000, 1_764_000_000_000 + 10_000, dtype=np.uint64)

        start = time.perf_counter()
        for channel in range(144):
            keyed_normals(pv_key_bytes(f"SR:SIM:{channel}:X"), counters)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"144 x 10,000 draws took {elapsed:.3f}s (budget 1.0s)"

    def test_construction_uses_no_numpy_generator(self):
        """``series`` is the deterministic core: a stateful numpy generator in it
        would reintroduce stream-position-dependent draws. The guard matches
        call-shaped tokens so prose may still explain the trap it prevents."""
        source = Path(series_module.__file__).read_text(encoding="utf-8")

        assert "np.random" not in source
        assert "default_rng(" not in source
        assert ".normal(" not in source


class TestWanderComponents:
    """Structural pins on the component stack behind :func:`wander`."""

    def test_component_count_is_in_the_declared_range(self):
        periods, phases, weights = series_module._wander_components(pv_key_bytes("PV:A"), 3600.0)

        assert 4 <= len(periods) <= 6
        assert len(phases) == len(periods)
        assert len(weights) == len(periods)

    def test_period_s_is_the_slowest_period(self):
        """``period_s`` names the slowest component, not the only one."""
        periods, _, _ = series_module._wander_components(pv_key_bytes("PV:A"), 3600.0)

        assert periods[0] == 3600.0
        assert np.all(np.diff(periods) < 0.0)  # strictly decreasing

    def test_periods_span_three_to_four_octaves_log_spaced(self):
        periods, _, _ = series_module._wander_components(pv_key_bytes("PV:A"), 3600.0)

        octaves = np.log2(periods[0] / periods[-1])
        assert 3.0 <= octaves <= 4.0
        # Log-spaced => equal octave steps between neighbours.
        steps = np.diff(np.log2(periods))
        np.testing.assert_allclose(steps, steps[0], rtol=1e-12)

    def test_period_ratios_are_not_simple_rationals(self):
        """Incommensurate in the practical sense: no pair of components shares a
        short common period, so the stack does not visibly repeat."""
        periods, _, _ = series_module._wander_components(pv_key_bytes("PV:A"), 3600.0)

        for i in range(len(periods)):
            for j in range(i + 1, len(periods)):
                ratio = periods[i] / periods[j]
                for q in range(1, 5):
                    for p in range(1, 4 * q + 1):
                        assert abs(ratio - p / q) > 0.02, f"{ratio} ~= {p}/{q}"

    def test_weights_are_normalized_and_slowest_dominates(self):
        """Envelope bound comes from ``sum(weights) == 1`` with positive weights,
        and the log-space jitter is narrower than the decay step, so the slowest
        component dominates for EVERY channel rather than only on average."""
        for name in [f"PV:{i}" for i in range(50)]:
            _, _, weights = series_module._wander_components(pv_key_bytes(name), 3600.0)

            assert float(weights.sum()) == pytest.approx(1.0, abs=1e-12)
            assert np.all(weights > 0.0)
            assert np.all(np.diff(weights) < 0.0)

    def test_phases_cover_the_circle_and_differ_per_channel(self):
        phases = np.concatenate(
            [
                series_module._wander_components(pv_key_bytes(f"PV:{i}"), 3600.0)[1]
                for i in range(60)
            ]
        )

        assert np.all(phases >= 0.0)
        assert np.all(phases < 2.0 * np.pi)
        assert phases.min() < 0.5 and phases.max() > 2.0 * np.pi - 0.5
        assert len(np.unique(phases)) == len(phases)


class TestWander:
    """The declarative baseline-texture primitive.

    One component covers both timescales: with ``period_s`` of many hours the
    slowest terms read as a distinct quasi-static per-channel offset inside any
    short query window, while the faster octaves supply visible motion.
    """

    def test_envelope_is_bounded_by_amplitude(self):
        amplitude = 25.0
        t = np.linspace(1.7e9, 1.7e9 + 200_000.0, 20_001)

        for name in [f"SR:BPM:{i}:X" for i in range(20)]:
            out = wander(pv_key_bytes(name), t, amplitude, 3600.0)

            assert np.all(np.abs(out) <= amplitude)
            assert np.isfinite(out).all()

    def test_bound_is_structural_not_clipping(self):
        """Doubling the amplitude doubles every sample exactly. A clipped
        implementation is nonlinear at the bound and fails this."""
        key = pv_key_bytes("SR:BPM:3:X")
        t = np.linspace(1.7e9, 1.7e9 + 50_000.0, 5_001)

        single = wander(key, t, 1.0, 3600.0)
        double = wander(key, t, 2.0, 3600.0)

        np.testing.assert_array_equal(double, 2.0 * single)
        # Nothing piles up at the bound the way a clip would.
        assert np.abs(single).max() < 1.0

    def test_deterministic_per_channel_and_time(self):
        key = pv_key_bytes("SR:BPM:4:X")
        t = np.linspace(1.7e9, 1.7e9 + 10_000.0, 1_001)

        np.testing.assert_array_equal(wander(key, t, 10.0, 3600.0), wander(key, t, 10.0, 3600.0))
        # Order-independent: a reordered request returns reordered values.
        order = np.array([500, 3, 999, 0, 3], dtype=np.intp)
        np.testing.assert_array_equal(
            wander(key, t[order], 10.0, 3600.0), wander(key, t, 10.0, 3600.0)[order]
        )

    def test_overlapping_windows_agree_exactly_on_shared_timestamps(self):
        """The two windows are built independently (not sliced from a common
        array) so the shared timestamps are equal by value, and the texture must
        agree bit-for-bit there — this is what makes overlapping archiver
        queries consistent."""
        key = pv_key_bytes("SR:BPM:5:X")
        window_a = np.arange(1_764_000_000.0, 1_764_000_600.0, 5.0)
        window_b = np.arange(1_764_000_300.0, 1_764_000_900.0, 5.0)

        draws_a = wander(key, window_a, 10.0, 3600.0)
        draws_b = wander(key, window_b, 10.0, 3600.0)

        shared = 60  # 300 s of overlap at 5 s spacing
        np.testing.assert_array_equal(window_a[shared:], window_b[:shared])
        np.testing.assert_array_equal(draws_a[shared:], draws_b[:shared])

    def test_long_window_mean_is_about_zero(self):
        """Texture is ambient motion, not a fake static baseline: over many
        slow periods it averages out, so the declared ``value: 0.0`` still reads
        as zero on a daily mean."""
        amplitude = 40.0
        period_s = 3600.0
        t = np.linspace(1.7e9, 1.7e9 + 400 * period_s, 200_001)

        for name in ["SR:BPM:1:X", "SR:BPM:2:Y", "SR:HCM:9:CURRENT:RB"]:
            out = wander(pv_key_bytes(name), t, amplitude, period_s)

            assert abs(float(out.mean())) < 0.02 * amplitude

    def test_distinct_channels_get_distinct_offsets_in_a_short_window(self):
        """Family-uniform declared parameters, per-channel apparent offset — the
        offsets come from the seeded phases, not from per-entry amplitudes."""
        amplitude = 30.0
        t = np.linspace(1.7e9, 1.7e9 + 60.0, 61)

        offsets = np.array(
            [
                float(wander(pv_key_bytes(f"SR:BPM:{i}:X"), t, amplitude, 86400.0).mean())
                for i in range(144)
            ]
        )

        assert len(np.unique(offsets)) == len(offsets)
        # The spread is a visible fraction of the envelope, not a hairline.
        assert float(offsets.std()) > 0.1 * amplitude

    def test_short_window_drift_is_small_when_period_dominates_the_window(self):
        amplitude = 50.0
        t = np.linspace(1.7e9, 1.7e9 + 60.0, 61)  # 60 s window vs a 24 h period

        for name in [f"SR:BPM:{i}:X" for i in range(20)]:
            out = wander(pv_key_bytes(name), t, amplitude, 86400.0)

            assert float(out.max() - out.min()) < 0.1 * amplitude

    def test_motion_is_visible_over_a_window_spanning_the_period(self):
        """Guards the degenerate pass: a constant (or all-zero) texture would
        satisfy the envelope, determinism and drift tests above."""
        amplitude = 30.0
        period_s = 3600.0
        t = np.linspace(1.7e9, 1.7e9 + 2 * period_s, 4_001)

        for name in [f"SR:BPM:{i}:X" for i in range(20)]:
            out = wander(pv_key_bytes(name), t, amplitude, period_s)

            assert float(out.max() - out.min()) > 0.3 * amplitude

    def test_does_not_repeat_at_the_slowest_period(self):
        """Incommensurate components mean the stack is not periodic in
        ``period_s`` — a harmonic stack would repeat exactly one period later."""
        key = pv_key_bytes("SR:BPM:6:X")
        period_s = 3600.0
        t = np.linspace(1.7e9, 1.7e9 + period_s, 2_001)

        now = wander(key, t, 20.0, period_s)
        one_period_later = wander(key, t + period_s, 20.0, period_s)

        assert float(np.abs(now - one_period_later).max()) > 0.1 * 20.0

    def test_faster_octaves_move_within_a_window_far_shorter_than_period_s(self):
        """The quasi-static term is not the whole story: inside a window that is
        a small fraction of ``period_s`` there is still real motion, which is
        what puts wander back on the BPM plots."""
        amplitude = 30.0
        t = np.linspace(1.7e9, 1.7e9 + 1800.0, 1_801)  # 30 min of a 24 h period

        spans = [
            float(np.ptp(wander(pv_key_bytes(f"SR:BPM:{i}:X"), t, amplitude, 86400.0)))
            for i in range(20)
        ]

        assert min(spans) > 0.001 * amplitude
        assert float(np.median(spans)) > 0.01 * amplitude

    def test_is_pure_and_does_not_mutate_input(self):
        key = pv_key_bytes("SR:BPM:7:X")
        t = np.linspace(1.7e9, 1.7e9 + 1000.0, 101)
        original = t.copy()

        first = wander(key, t, 10.0, 3600.0)
        wander(pv_key_bytes("SR:BPM:8:X"), t, 99.0, 60.0)  # interleaved other call
        second = wander(key, t, 10.0, 3600.0)

        np.testing.assert_array_equal(t, original)
        np.testing.assert_array_equal(first, second)

    def test_zero_amplitude_is_flat(self):
        out = wander(pv_key_bytes("SR:BPM:9:X"), np.linspace(1.7e9, 1.7e9 + 100.0, 11), 0.0, 60.0)

        np.testing.assert_array_equal(out, np.zeros(11))

    def test_shape_dtype_and_empty_input(self):
        key = pv_key_bytes("SR:BPM:10:X")

        out = wander(key, np.full((2, 3), 1.7e9), 5.0, 3600.0)
        assert out.shape == (2, 3)
        assert out.dtype == np.float64

        empty = wander(key, np.array([]), 5.0, 3600.0)
        assert empty.shape == (0,)
        assert empty.dtype == np.float64

    def test_scalar_live_read_matches_the_synthesized_sample_exactly(self):
        """The live-read path samples texture at a single wall-clock instant while
        synthesis passes a whole window. Both go through this one function, so a
        live read at time ``t`` and the synthesized sample at ``t`` must be the
        same number — bit-for-bit, not merely close."""
        key = pv_key_bytes("SR:BPM:12:X")
        now = 1_764_000_000.5

        live = wander(key, now, 30.0, 86400.0)
        window = np.array([now - 1.0, now, now + 1.0])
        synthesized = wander(key, window, 30.0, 86400.0)

        assert float(live) == float(synthesized[1])
        assert np.asarray(live).shape == ()

    def test_non_positive_period_is_rejected(self):
        """A zero/negative period would divide to inf and put NaN in the series —
        the exact class of silent corruption this feature exists to remove."""
        key = pv_key_bytes("SR:BPM:11:X")
        t = np.linspace(1.7e9, 1.7e9 + 100.0, 11)

        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match="period_s"):
                wander(key, t, 10.0, bad)
