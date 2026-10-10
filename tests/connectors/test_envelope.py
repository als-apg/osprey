"""A channel's motion envelope: how far its declared motion can carry it from the held value.

The envelope is ``|drift.amplitude| + sum |gain| x (1 + |gain_wander.amplitude|)
x |drive.amplitude| + MOTION_SIGMAS x noise sigma``, a relative noise sigma
taken at the seed's own ``nominal``, and 0.0 for a channel that declares no
motion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.facility.views import view_bytes
from osprey_connectors.simulation.envelope import (
    MOTION_SIGMAS,
    active_envelopes,
    declares_motion,
    motion_envelope,
    noise_sigma,
)
from osprey_connectors.simulation.view import SCHEMAS, SEEDS_FILE, SimulatorView
from tests._simulator_view import write_scenarios_view

_DRIFT = {"amplitude": 0.005, "period_s": 600.0}
_DRIVE = {"kind": "sine", "amplitude": 2.0, "period_s": 300.0}


def test_noise_alone_is_its_sigmas() -> None:
    assert motion_envelope({"noise": {"absolute": 0.001}}) == pytest.approx(MOTION_SIGMAS * 0.001)


def test_drift_alone_is_its_amplitude() -> None:
    assert motion_envelope({"drift": _DRIFT}) == pytest.approx(0.005)


def test_noise_and_drift_add() -> None:
    seed = {"nominal": -3.0, "noise": {"absolute": 0.001}, "drift": _DRIFT}

    assert motion_envelope(seed) == pytest.approx(0.005 + 6.0 * 0.001)


@pytest.mark.parametrize("seed", [None, {}, {"nominal": 1.0}, {"drift": None, "noise": None}])
def test_a_seed_without_motion_has_no_envelope(seed: dict | None) -> None:
    assert motion_envelope(seed) == 0.0


def test_a_negative_drift_counts_by_magnitude() -> None:
    seed = {"noise": {"absolute": 0.002}, "drift": {"amplitude": -0.01, "period_s": 60.0}}

    assert motion_envelope(seed) == pytest.approx(0.01 + 6.0 * 0.002)


def test_the_factor_is_six() -> None:
    assert MOTION_SIGMAS == 6.0


def test_relative_noise_is_taken_of_the_seeds_nominal() -> None:
    seed = {"nominal": -200.0, "noise": {"relative": 1e-4}}

    assert noise_sigma(seed["noise"], seed["nominal"]) == pytest.approx(0.02)
    assert motion_envelope(seed) == pytest.approx(6.0 * 0.02)


def test_a_relative_noise_without_a_nominal_is_refused() -> None:
    with pytest.raises(ValueError, match="numeric nominal"):
        noise_sigma({"relative": 0.01}, None)


def test_a_scenario_noise_replaces_the_seeds() -> None:
    seed = {"nominal": 10.0, "noise": {"absolute": 0.5}, "drift": _DRIFT}

    assert motion_envelope(seed, noise={"absolute": 0.1}) == pytest.approx(0.005 + 0.6)
    assert motion_envelope(seed, noise={"relative": 0.01}) == pytest.approx(0.005 + 0.6)
    assert motion_envelope(seed, noise={"absolute": 0.0}) == pytest.approx(0.005)


def test_couplings_add_gain_times_wander_times_driver_amplitude() -> None:
    couplings = [
        {"driver": "d1", "gain": -0.5, "drive": _DRIVE},
        {"driver": "d2", "gain": 2.0, "gain_wander": {"amplitude": 0.25}, "drive": _DRIVE},
    ]

    assert motion_envelope(None, couplings=couplings) == pytest.approx(0.5 * 2.0 + 2.0 * 1.25 * 2.0)


def test_active_envelopes_merge_like_the_composite() -> None:
    seeds = {
        "T:A": {"nominal": 1.0, "noise": {"absolute": 0.1}},
        "T:B": {"nominal": 4.0, "drift": _DRIFT},
        "T:SP": {"nominal": 3.0},
    }
    scenarios = {
        "warm": {
            "name": "warm",
            "drivers": {"d1": _DRIVE},
            "couple": {"T:B": [{"driver": "d1", "gain": 0.5}]},
            "noise": {"T:A": {"absolute": 0.3}},
        },
        "quiet": {"name": "quiet", "noise": {"T:A": {"absolute": 0.0}}},
        "orphan": {"name": "orphan", "couple": {"T:C": [{"driver": "nowhere", "gain": 9.0}]}},
    }

    assert active_envelopes(seeds, scenarios, []) == pytest.approx(
        {"T:A": 0.6, "T:B": 0.005, "T:SP": 0.0}
    )
    assert active_envelopes(seeds, scenarios, ["warm"]) == pytest.approx(
        {"T:A": 1.8, "T:B": 0.005 + 1.0, "T:SP": 0.0}
    )
    assert active_envelopes(seeds, scenarios, ["warm", "quiet"])["T:A"] == 0.0
    assert active_envelopes(seeds, scenarios, ["orphan", "unknown"])["T:C"] == 0.0
    assert list(active_envelopes(seeds, scenarios, ["orphan"])) == ["T:A", "T:B", "T:C", "T:SP"]


def test_a_stilled_reading_has_no_envelope() -> None:
    seed = {"nominal": 1.0, "noise": {"absolute": 0.1}, "drift": _DRIFT}
    couplings = [{"driver": "d1", "gain": 0.5, "drive": _DRIVE}]

    assert motion_envelope(seed, noise={"absolute": 0.3}, couplings=couplings, still=True) == 0.0


def test_active_envelopes_honour_still_all_and_a_list() -> None:
    seeds = {
        "T:A": {"nominal": 1.0, "noise": {"absolute": 0.1}},
        "T:B": {"nominal": 4.0, "drift": _DRIFT},
    }
    scenarios = {
        "warm": {
            "name": "warm",
            "drivers": {"d1": _DRIVE},
            "couple": {"T:C": [{"driver": "d1", "gain": 0.5}]},
        },
        "one": {"name": "one", "still": ["T:A"]},
        "every": {"name": "every", "still": "all"},
    }

    assert active_envelopes(seeds, scenarios, ["one"]) == pytest.approx({"T:A": 0.0, "T:B": 0.005})
    assert active_envelopes(seeds, scenarios, ["warm", "one"]) == pytest.approx(
        {"T:A": 0.0, "T:B": 0.005, "T:C": 1.0}
    )
    assert set(active_envelopes(seeds, scenarios, ["warm", "every"]).values()) == {0.0}


def test_nominal_counts_in_the_readers_envelope(tmp_path: Path) -> None:
    view = tmp_path / "data" / "simulator"
    view.mkdir(parents=True)
    seeds = {"T:A": {"nominal": 1.0, "noise": {"absolute": 0.1}}}
    (view / SEEDS_FILE).write_bytes(view_bytes({"schema": SCHEMAS[SEEDS_FILE], "seeds": seeds}))
    write_scenarios_view(tmp_path, {"nominal": {"still": ["T:A"]}, "loud": {}})
    reader = SimulatorView.open(view)

    assert reader.motion_envelope("T:A") == 0.0
    assert reader.motion_envelope("T:A", active=("loud",)) == 0.0


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        (None, False),
        ({"nominal": 1.0}, False),
        ({"noise": {"absolute": 0.0}}, False),
        ({"noise": {"relative": 0.0}}, False),
        ({"noise": {"absolute": 0.1}}, True),
        ({"nominal": 2.0, "noise": {"relative": 0.01}}, True),
        ({"drift": _DRIFT}, True),
    ],
)
def test_declares_motion_ignores_a_zero_term(seed: dict | None, expected: bool) -> None:
    assert declares_motion(seed) is expected
