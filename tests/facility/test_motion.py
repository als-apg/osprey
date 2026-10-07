"""A readback's settle band: how far its declared motion can carry it from the held value.

The band is ``|drift.amplitude| + MOTION_SIGMAS x |noise|`` off the channel's
own seed, and 0.0 for a seed that declares no motion.
"""

from __future__ import annotations

import pytest

from osprey.facility.motion import MOTION_SIGMAS, settle_band


def test_noise_alone_is_its_sigmas() -> None:
    assert settle_band({"noise": 0.001}) == pytest.approx(MOTION_SIGMAS * 0.001)


def test_drift_alone_is_its_amplitude() -> None:
    assert settle_band({"drift": {"amplitude": 0.005, "period_s": 600.0}}) == pytest.approx(0.005)


def test_noise_and_drift_add() -> None:
    seed = {"nominal": -3.0, "noise": 0.001, "drift": {"amplitude": 0.005, "period_s": 600.0}}

    assert settle_band(seed) == pytest.approx(0.005 + 6.0 * 0.001)


@pytest.mark.parametrize("seed", [None, {}, {"nominal": 1.0}, {"drift": None, "noise": None}])
def test_a_seed_without_motion_has_no_band(seed: dict | None) -> None:
    assert settle_band(seed) == 0.0


def test_a_negative_sign_counts_by_magnitude() -> None:
    seed = {"noise": -0.002, "drift": {"amplitude": -0.01, "period_s": 60.0}}

    assert settle_band(seed) == pytest.approx(0.01 + 6.0 * 0.002)


def test_the_factor_is_six() -> None:
    assert MOTION_SIGMAS == 6.0
