"""The pyat engine's ``injection`` setting: where a tracked particle is launched and from what.

``prepare`` checks the block against the deck and normalises it onto
``Prepared``. A periodic model states it; a single-pass model refuses it and
launches from ``twiss_in.closed_orbit``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

at = pytest.importorskip("at")

from osprey.facility.errors import FacilityBuildError  # noqa: E402
from osprey.simulation.engines import pyat as engine  # noqa: E402

MODEL = "SR"

TWISS = {"beta": [5.0, 3.0], "alpha": [0.1, -0.2]}

#: The launch vector in the engine's coordinate order.
LAUNCH = {"x": 1e-3, "px": 2e-4, "y": -5e-4, "py": 1e-4, "delta": 1e-3, "ct": -2e-3}


@pytest.fixture
def deck(tmp_path: Path) -> Path:
    """A line whose marker ``INJ`` is unique and whose marker ``TWICE`` repeats."""
    lattice = at.Lattice(
        [
            at.Marker("TWICE"),
            at.Drift("D0", 1.0),
            at.Quadrupole("QF", 0.3, 1.2),
            at.Marker("INJ"),
            at.Drift("D1", 1.0),
            at.Monitor("BPM1"),
            at.Quadrupole("QD", 0.3, -1.1),
            at.Marker("TWICE"),
            at.Drift("D2", 1.0),
            at.Monitor("BPM2"),
        ],
        name="line",
        energy=3e9,
        particle="electron",
        periodicity=1,
    )
    path = tmp_path / "line.json"
    at.save_lattice(lattice, str(path))
    return path


def _periodic(deck: Path, injection: Any) -> engine.Prepared:
    return engine.prepare(deck, {"pyat": {"injection": injection}}, model=MODEL)


def _refusal(deck: Path, block: dict[str, Any]) -> FacilityBuildError:
    with pytest.raises(FacilityBuildError) as caught:
        engine.prepare(deck, {"pyat": block}, model=MODEL)
    assert caught.value.kind == "engine-invalid"
    assert caught.value.record_id == MODEL
    return caught.value


class TestNormalisedBlock:
    def test_injection_is_a_settings_key(self):
        assert "injection" in engine.SETTINGS_KEYS

    def test_the_marker_alone_launches_on_axis_with_ten_pre_trigger_samples(self, deck: Path):
        injection = _periodic(deck, {"marker": "INJ"}).injection
        assert injection.marker == "INJ"
        assert injection.index == 3
        assert injection.launch.dtype == float
        np.testing.assert_array_equal(injection.launch, np.zeros(6))
        assert injection.pre_trigger_samples == 10

    def test_the_launch_vector_is_in_the_order_pyat_tracks(self, deck: Path):
        injection = _periodic(deck, {"marker": "INJ", **LAUNCH}).injection
        np.testing.assert_array_equal(injection.launch, [1e-3, 2e-4, -5e-4, 1e-4, 1e-3, -2e-3])

    def test_an_unstated_coordinate_is_zero(self, deck: Path):
        injection = _periodic(deck, {"marker": "INJ", "x": 2, "delta": 1e-3}).injection
        np.testing.assert_array_equal(injection.launch, [2.0, 0.0, 0.0, 0.0, 1e-3, 0.0])

    def test_pre_trigger_samples_is_taken_as_stated(self, deck: Path):
        none = _periodic(deck, {"marker": "INJ", "pre_trigger_samples": 0}).injection
        assert none.pre_trigger_samples == 0
        injection = _periodic(deck, {"marker": "INJ", "pre_trigger_samples": 25}).injection
        assert injection.pre_trigger_samples == 25
        assert type(injection.pre_trigger_samples) is int

    def test_a_periodic_model_stating_no_injection_prepares_without_one(self, deck: Path):
        assert engine.prepare(deck, None, model=MODEL).injection is None
        assert engine.prepare(deck, {"pyat": {"injection": None}}, model=MODEL).injection is None

    def test_the_other_settings_are_untouched_by_the_block(self, deck: Path):
        prepared = _periodic(deck, {"marker": "INJ"})
        assert prepared.solve == "periodic"
        assert prepared.twiss_in is None

    def test_prepared_without_an_injection_defaults_to_none(self):
        prepared = engine.Prepared(
            solve="periodic", twiss_in=None, rest_mass_gev=0.000511, length_m=8.0
        )
        assert prepared.injection is None


class TestRefusals:
    def test_a_block_that_is_not_a_map(self, deck: Path):
        stop = _refusal(deck, {"injection": "INJ"})
        assert stop.detail == "settings key pyat.injection is not a map"

    def test_an_unknown_key(self, deck: Path):
        stop = _refusal(deck, {"injection": {"marker": "INJ", "turns": 64, "dp": 0.0}})
        assert stop.detail == "settings key pyat.injection carries unknown keys dp, turns"

    def test_tracked_length_is_not_a_setting(self, deck: Path):
        stop = _refusal(deck, {"injection": {"marker": "INJ", "shape": 64}})
        assert "unknown keys shape" in stop.detail

    def test_a_block_without_a_marker(self, deck: Path):
        stop = _refusal(deck, {"injection": {"x": 1e-3}})
        assert stop.detail == "settings key pyat.injection lacks marker"

    @pytest.mark.parametrize("marker", [3, None, "", ["INJ"]])
    def test_a_marker_that_is_not_a_name(self, deck: Path, marker: Any):
        stop = _refusal(deck, {"injection": {"marker": marker}})
        assert stop.detail == f"settings key pyat.injection.marker is {marker!r}"

    def test_a_marker_the_deck_does_not_hold(self, deck: Path):
        stop = _refusal(deck, {"injection": {"marker": "SEPTUM"}})
        assert isinstance(stop, engine.ElementStop)
        assert (stop.element, stop.count) == ("SEPTUM", 0)
        assert stop.detail == "element SEPTUM is not in the deck"

    def test_a_marker_the_deck_holds_twice(self, deck: Path):
        stop = _refusal(deck, {"injection": {"marker": "TWICE"}})
        assert isinstance(stop, engine.ElementStop)
        assert (stop.element, stop.count) == ("TWICE", 2)

    @pytest.mark.parametrize("key", ["x", "px", "y", "py", "delta", "ct"])
    @pytest.mark.parametrize("value", ["1e-3", True, float("nan"), float("inf"), [1e-3]])
    def test_a_coordinate_that_is_not_a_finite_number(self, deck: Path, key: str, value: Any):
        stop = _refusal(deck, {"injection": {"marker": "INJ", key: value}})
        assert stop.detail == f"settings key pyat.injection.{key} is {value!r}"

    @pytest.mark.parametrize("value", [-1, 2.0, 2.5, True, "10", float("nan")])
    def test_pre_trigger_samples_that_is_not_a_count(self, deck: Path, value: Any):
        stop = _refusal(deck, {"injection": {"marker": "INJ", "pre_trigger_samples": value}})
        assert stop.detail == f"settings key pyat.injection.pre_trigger_samples is {value!r}"

    def test_an_unknown_settings_key_still_stops_and_the_remedy_names_injection(self, deck: Path):
        stop = _refusal(deck, {"injection": {"marker": "INJ"}, "kicker": "K1"})
        assert stop.detail == "settings key pyat carries unknown keys kicker"
        assert "injection" in stop.remedy


class TestSinglePass:
    def test_the_block_is_refused(self, deck: Path):
        stop = _refusal(
            deck, {"solve": "single_pass", "twiss_in": TWISS, "injection": {"marker": "INJ"}}
        )
        assert stop.detail == "settings key pyat.injection is set on a single_pass model"

    def test_the_refusal_comes_before_the_block_is_read(self, deck: Path):
        stop = _refusal(
            deck, {"solve": "single_pass", "twiss_in": TWISS, "injection": {"marker": "SEPTUM"}}
        )
        assert stop.detail == "settings key pyat.injection is set on a single_pass model"

    def test_it_launches_from_the_closed_orbit_of_twiss_in_at_the_deck_start(self, deck: Path):
        orbit = [1e-3, 2e-4, -5e-4, 1e-4]
        prepared = engine.prepare(
            deck,
            {"pyat": {"solve": "single_pass", "twiss_in": {**TWISS, "closed_orbit": orbit}}},
            model=MODEL,
        )
        injection = prepared.injection
        assert injection.marker is None
        assert injection.index == 0
        np.testing.assert_array_equal(injection.launch, [*orbit, 0.0, 0.0])
        assert injection.pre_trigger_samples == 10

    def test_an_unstated_closed_orbit_launches_on_axis(self, deck: Path):
        prepared = engine.prepare(
            deck, {"pyat": {"solve": "single_pass", "twiss_in": TWISS}}, model=MODEL
        )
        np.testing.assert_array_equal(prepared.injection.launch, np.zeros(6))

    def test_the_launch_vector_is_its_own_array(self, deck: Path):
        prepared = engine.prepare(
            deck, {"pyat": {"solve": "single_pass", "twiss_in": TWISS}}, model=MODEL
        )
        assert prepared.injection.launch is not prepared.twiss_in["closed_orbit"]
