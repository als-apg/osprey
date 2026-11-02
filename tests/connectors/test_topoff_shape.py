"""The top-off sawtooth: shape, held level, motion, envelope and fingerprint.

The archive view and the limits band of a built facility are the framework's
and are tested beside its facility build.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from osprey_connectors.simulation import envelope, topoff
from osprey_connectors.simulation.archive import SeedKnobs, seed_fingerprint
from osprey_connectors.simulation.texture import TextureModel

T0 = 1_760_000_010.0  # a whole multiple of PERIOD_S: a refill instant
CURRENT = "SR:DCCT:CURRENT"
LIFETIME = "SR:DCCT:LIFETIME"
CEILING = 500.0
LIFETIME_H = 8.0
PERIOD_S = 30.0
TAU = LIFETIME_H * 3600.0
RECORD = {"ceiling": CEILING, "lifetime_h": LIFETIME_H, "period_s": PERIOD_S}
# 1 - exp(-x) for a small x, taken without cancellation.
MEAN = (TAU / PERIOD_S) * -math.expm1(-PERIOD_S / TAU)
LEVEL = CEILING * MEAN
SEEDS = {
    CURRENT: {"topoff": RECORD},
    LIFETIME: {"topoff_lifetime": {"of": CURRENT}},
}


def _channel(address: str) -> dict[str, Any]:
    return {
        "address": address,
        "role": "readback",
        "pair": None,
        "value_type": "float",
        "unit": None,
        "description": None,
        "writable": False,
        "value_range": None,
        "owner": "texture",
    }


def _model(t_s: float = T0, seeds: dict[str, Any] | None = None) -> TextureModel:
    seeds = SEEDS if seeds is None else seeds
    variables = {
        "schema": "simulator-variables",
        "code": "T",
        "models": [],
        "channels": [_channel(address) for address in sorted(seeds)],
    }
    clock = {"t": t_s}
    model = TextureModel(variables, {"seeds": seeds}, clock=lambda: clock["t"])
    model.clock = clock  # type: ignore[attr-defined]
    return model


def _read(model: TextureModel, address: str, t_s: float) -> float:
    model.clock["t"] = t_s  # type: ignore[attr-defined]
    return float(model.get([address])[address])


assert T0 % PERIOD_S == 0.0


# --- the shape ------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [0.0, 1.0, 7.5, 15.0, 29.0])
def test_shape_at_chosen_instants(offset):
    assert topoff.shape(RECORD, T0 + offset) == pytest.approx(math.exp(-offset / TAU), rel=1e-12)


def test_the_channel_reads_the_ceiling_times_the_shape():
    model = _model()
    for offset in (0.0, 3.0, 12.25, 29.5):
        expected = CEILING * math.exp(-offset / TAU)
        assert _read(model, CURRENT, T0 + offset) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("period", range(5))
def test_refill_at_each_period_boundary(period):
    model = _model()
    boundary = T0 + period * PERIOD_S
    before = _read(model, CURRENT, boundary - 0.001)
    at = _read(model, CURRENT, boundary)
    assert at == pytest.approx(CEILING, rel=1e-12)
    assert before == pytest.approx(CEILING * math.exp(-(PERIOD_S - 0.001) / TAU), rel=1e-9)
    assert at > before


def test_tooth_depth_is_the_larger_swing_from_the_held_level():
    floor = CEILING * math.exp(-PERIOD_S / TAU)
    depth = topoff.tooth_depth(RECORD)
    assert depth == pytest.approx(max(CEILING - LEVEL, LEVEL - floor), rel=1e-12)
    # The full tooth near these numbers is about 0.52 mA peak to peak.
    assert CEILING - floor == pytest.approx(0.52, abs=0.01)
    model = _model()
    times = np.linspace(T0, T0 + PERIOD_S, 3001, endpoint=False)
    swing = model.motion(CURRENT, times, base=LEVEL)
    assert float(np.max(np.abs(swing))) <= depth + 1e-9
    assert float(np.max(np.abs(swing))) == pytest.approx(depth, rel=1e-3)


def test_held_level_equals_the_period_mean():
    assert topoff.period_mean(RECORD) == pytest.approx(MEAN, rel=1e-12)
    assert topoff.held_value(SEEDS[CURRENT]) == pytest.approx(LEVEL, rel=1e-12)
    model = _model()
    assert model.held([CURRENT])[CURRENT] == pytest.approx(LEVEL, rel=1e-12)
    times = np.linspace(T0, T0 + PERIOD_S, 300_001, endpoint=False)
    reads = LEVEL + model.motion(CURRENT, times, base=LEVEL)
    assert float(np.mean(reads)) == pytest.approx(LEVEL, rel=1e-8)


def test_shape_version_digests_the_constants(monkeypatch):
    assert topoff.shape_version() == topoff.SHAPE_VERSION
    for name in topoff.SHAPE_CONSTANT_NAMES:
        with monkeypatch.context() as patch:
            patch.setattr(topoff, name, f"{getattr(topoff, name)}-changed")
            assert topoff.shape_version() != topoff.SHAPE_VERSION


# --- motion, still and envelope -------------------------------------------------------


def test_a_stilled_channel_serves_its_held_level():
    model = _model()
    model.set_motion({}, {}, still=frozenset({CURRENT}))
    for offset in (0.0, 9.0, 29.0):
        assert _read(model, CURRENT, T0 + offset) == pytest.approx(LEVEL, rel=1e-12)
    assert not model.has_motion(CURRENT)
    assert envelope.motion_envelope(SEEDS[CURRENT], still=True) == 0.0
    stilled = envelope.active_envelopes(SEEDS, {"quiet": {"still": [CURRENT]}}, ["quiet"])
    assert stilled[CURRENT] == 0.0


def test_still_all_serves_the_held_level():
    model = _model()
    model.set_motion({}, {}, still_all=True)
    assert _read(model, CURRENT, T0 + 11.0) == pytest.approx(LEVEL, rel=1e-12)
    assert not model.has_motion(CURRENT)


def test_an_unstilled_channel_moves_and_has_an_envelope():
    model = _model()
    assert model.has_motion(CURRENT)
    assert envelope.declares_motion(SEEDS[CURRENT])
    assert envelope.motion_envelope(SEEDS[CURRENT]) == pytest.approx(
        topoff.tooth_depth(RECORD), rel=1e-12
    )
    assert envelope.active_envelopes(SEEDS, {}, [])[CURRENT] == pytest.approx(
        topoff.tooth_depth(RECORD), rel=1e-12
    )
    # A scenario noise replacement keeps the tooth moving.
    model.set_motion({}, {CURRENT: {"absolute": 0.0}})
    assert model.has_motion(CURRENT)


def test_the_envelope_adds_noise_taken_of_the_held_level():
    seed = {"topoff": RECORD, "noise": {"relative": 0.001}}
    expected = topoff.tooth_depth(RECORD) + envelope._MOTION_SIGMAS * 0.001 * LEVEL
    assert envelope.motion_envelope(seed) == pytest.approx(expected, rel=1e-12)
    model = _model(seeds={CURRENT: seed})
    times = np.linspace(T0, T0 + 600.0, 20_001)
    swing = model.motion(CURRENT, times, base=LEVEL)
    assert float(np.max(np.abs(swing))) <= expected


def test_a_level_moved_to_zero_reads_within_noise_of_zero():
    sigma = 0.01
    seed = {"topoff": RECORD, "noise": {"absolute": sigma}}
    model = _model(seeds={CURRENT: seed})
    model.set_active({CURRENT: 0.0})
    for offset in np.linspace(0.0, 3 * PERIOD_S, 61):
        value = _read(model, CURRENT, T0 + float(offset))
        assert abs(value) <= envelope._MOTION_SIGMAS * sigma
    noiseless = _model()
    noiseless.set_active({CURRENT: 0.0})
    assert _read(noiseless, CURRENT, T0 + 4.0) == 0.0


def test_a_moved_level_scales_the_tooth():
    model = _model()
    model.set_active({CURRENT: LEVEL / 2})
    offset = 6.0
    expected = (CEILING / 2) * math.exp(-offset / TAU)
    assert _read(model, CURRENT, T0 + offset) == pytest.approx(expected, rel=1e-12)


# --- topoff_lifetime ------------------------------------------------------------------


def test_lifetime_is_a_held_constant():
    model = _model()
    assert model.held([LIFETIME])[LIFETIME] == LIFETIME_H
    assert not model.has_motion(LIFETIME)
    assert not envelope.declares_motion(SEEDS[LIFETIME])
    assert envelope.motion_envelope(SEEDS[LIFETIME]) == 0.0
    for offset in (0.0, 10.0, 29.9, 300.0):
        assert _read(model, LIFETIME, T0 + offset) == LIFETIME_H


def test_lifetime_without_a_topoff_source_holds_nothing():
    assert topoff.held_value(SEEDS[LIFETIME]) is None
    assert topoff.held_value(SEEDS[LIFETIME], {}.get) is None
    assert topoff.held_value({"nominal": 3.0}) is None


# --- archive --------------------------------------------------------------------------


def test_the_fingerprint_names_the_shape_version():
    knobs = SeedKnobs(retention_days=1, hot_span_hours=1, hot_cadence_sec=10, tail_cadence_sec=60)
    fingerprint = seed_fingerprint(knobs, [CURRENT], compression="zstd")
    assert fingerprint["topoff_shape_version"] == topoff.SHAPE_VERSION
