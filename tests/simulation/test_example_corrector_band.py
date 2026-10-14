"""The example SR correctors over their whole current band, on the clean deck.

A corrector current is served as a kick through the linear calibration its
wiring entry carries, so the band a limits record allows in amperes is a band
of kicks. That band is wide enough to steer a first turn — its ends are
milliradian kicks — and the clean deck still has a closed orbit at both ends,
for the corrector the limits record bounds as the model serves it and for
every other corrector of the model as pyAT finds it.

Every case reads the committed demo as the build renders it.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

at = pytest.importorskip("at")

import numpy as np  # noqa: E402

from osprey.facility.views.simulator import simulator_wiring  # noqa: E402
from osprey.simulation.engines import pyat as engine  # noqa: E402

MODEL = "SR"
#: Correctors per plane in the example deck.
CORRECTORS_PER_PLANE = 72
#: The smallest kick at a band end that still counts as steering a first turn.
STEERING_KICK_RAD = 1.0e-3
#: Fractions of a band end the orbit search walks through, each solve seeded
#: with the orbit before it, so the search follows the orbit out to the end.
RAMP = (0.25, 0.5, 0.75, 1.0)


@pytest.fixture(scope="module")
def demo(built_control_assistant: Any) -> dict[str, Any]:
    """The SR model of the built demo: its wiring entries, deck and settings."""
    facility = built_control_assistant.facility
    (model,) = [entry for entry in facility["models"] if entry["name"] == MODEL]
    return {
        "wiring": simulator_wiring(facility, MODEL),
        "deck": built_control_assistant.facility_dir / model["deck"],
        "settings": model.get("settings"),
    }


def _is_kick(record: dict[str, Any]) -> bool:
    return record["engine"].get("attribute") == "KickAngle"


def _corrector_setpoints(demo: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        record for record in demo["wiring"] if _is_kick(record) and record["direction"] == "write"
    ]


def _banded(demo: dict[str, Any]) -> list[dict[str, Any]]:
    """The corrector setpoints a limits record bounds."""
    return [record for record in _corrector_setpoints(demo) if record.get("value_range")]


def _gain(record: dict[str, Any]) -> float:
    return record["calibration"]["curve"]["linear"]["gain"]


def _monitors(demo: dict[str, Any]) -> list[str]:
    return [
        record["address"]
        for record in demo["wiring"]
        if "axis" in record["engine"]
        and "attribute" not in record["engine"]
        and "series" not in record["engine"]
    ]


def _closed_orbit_at(ring: Any, element: Any, plane: int, kick: float, refpts: Any) -> Any:
    """The closed orbit at the monitors with one corrector at ``kick`` radians."""
    guess = None
    orbit = None
    for fraction in RAMP:
        angle = [0.0, 0.0]
        angle[plane] = fraction * kick
        element.KickAngle = angle
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            guess, orbit = at.find_orbit(ring, refpts=refpts, guess=guess)
        if not np.all(np.isfinite(guess)):
            break
    element.KickAngle = [0.0, 0.0]
    return orbit


class TestBand:
    def test_a_limits_record_bounds_a_corrector(self, demo):
        assert _banded(demo)

    def test_every_corrector_and_its_readback_share_one_gain(self, demo):
        kicks = [record for record in demo["wiring"] if _is_kick(record)]

        assert len(kicks) == 4 * CORRECTORS_PER_PLANE
        assert len({_gain(record) for record in kicks}) == 1

    def test_the_band_ends_are_steering_kicks(self, demo):
        for record in _banded(demo):
            low, high = record["value_range"]

            assert abs(_gain(record) * low) >= STEERING_KICK_RAD
            assert abs(_gain(record) * high) >= STEERING_KICK_RAD


class TestClosedOrbit:
    def test_the_model_serves_an_orbit_at_both_ends_of_the_limits_band(self, demo):
        readings = _monitors(demo)
        for record in _banded(demo):
            address = record["address"]
            rest = engine.build(MODEL, demo["wiring"], demo["deck"], demo["settings"], {})
            at_rest = rest.get(readings)
            for end in record["value_range"]:
                model = engine.build(MODEL, demo["wiring"], demo["deck"], demo["settings"], {})

                model.set({address: end})
                orbit = model.get(readings)

                assert model.get([address])[address] == end
                assert all(np.isfinite(value) for value in orbit.values()), (address, end)
                assert max(abs(orbit[name] - at_rest[name]) for name in readings) > 0.0

    def test_every_corrector_keeps_a_closed_orbit_at_both_ends_of_the_band(self, demo):
        (banded, *_) = _banded(demo)
        ring = at.load_lattice(str(demo["deck"]))
        refpts = ring.get_uint32_index(at.Monitor)
        setpoints = _corrector_setpoints(demo)
        assert len(setpoints) == 2 * CORRECTORS_PER_PLANE

        lost: list[tuple[str, float]] = []
        for record in setpoints:
            (position,) = ring.get_uint32_index(record["element"])
            plane = record["engine"]["index"]
            for end in banded["value_range"]:
                orbit = _closed_orbit_at(ring, ring[position], plane, _gain(record) * end, refpts)
                if not np.all(np.isfinite(orbit)):
                    lost.append((record["address"], end))

        assert lost == []
