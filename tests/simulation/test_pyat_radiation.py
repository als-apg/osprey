"""The pyat engine's ``radiation`` setting: a deck made to radiate.

A deck radiates through its elements' pass methods. Where the deck a facility
has does not state them, ``pyat.radiation: true`` gives every bend,
quadrupole and higher multipole its radiating pass method on the model's own
copy, for the closed orbit, the response matrix and the turn-by-turn tracking
alike. The beam then loses energy every turn and the cavity makes it up: the
closed orbit sits at the synchronous phase, and with the RF off there is no
closed orbit and a tracked bunch spirals inwards until it is lost.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import at
import numpy as np
import pytest
from lume_pyat.exceptions import OrbitSolveError

from osprey.facility.errors import FacilityBuildError
from osprey.facility.records import FacilityRecords
from osprey.facility.views.simulator import simulator_wiring
from osprey.simulation.engines import pyat as engine
from osprey.simulation.engines import pyat_turns
from tests.simulation._sr_deck import load_sr_deck

MODEL = "SR"
INJECTION = "INJ"
SWITCH = "SR:RF:SWITCH"
VOLTAGE = "SR:RF:VOLTAGE:SP"
CORRECTOR = "SR:MAG:HCM:01:CURRENT:SP"

#: Sands' radiation constant, m / GeV^3.
C_GAMMA = 8.846e-5
#: The half aperture the loss test declares, metres.
APERTURE = 4.0e-3


def _radiating(lattice: Any) -> set[str]:
    return {type(element).__name__ for element in lattice if element.PassMethod.endswith("RadPass")}


@pytest.fixture(scope="module")
def demo(built_demo: Any, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The demo SR model over a copy of its deck that states no radiation."""
    facility = built_demo.facility
    (model,) = [entry for entry in facility["models"] if entry["name"] == MODEL]
    ring = load_sr_deck()
    ring.disable_6d(cavity_pass=None)
    assert not _radiating(ring) and ring.is_6d
    deck = tmp_path_factory.mktemp("plain") / "SR.json"
    ring.save(str(deck))
    return {
        "wiring": simulator_wiring(FacilityRecords(facility), MODEL),
        "deck": deck,
        "settings": model.get("settings"),
    }


def _settings(demo: dict[str, Any], *, radiation: Any = None, turns: bool = False) -> Any:
    settings = copy.deepcopy(demo["settings"]) or {}
    block = settings.setdefault("pyat", {})
    if radiation is not None:
        block["radiation"] = radiation
    if turns:
        block["turn_by_turn"] = {
            "injection_element": INJECTION,
            "aperture": {"x": APERTURE, "y": APERTURE},
        }
    return settings


def _build(demo: dict[str, Any], *, radiation: Any = None, turns: bool = False) -> Any:
    wiring = list(demo["wiring"])
    if turns:
        wiring.append(
            {
                "address": SWITCH,
                "engine": {"signal": "rf_switch"},
                "direction": "write",
                "default": 1.0,
            }
        )
    return engine.build(
        MODEL, wiring, demo["deck"], _settings(demo, radiation=radiation, turns=turns), {}
    )


def _energy_loss_ev(lattice: Any) -> float:
    """The energy an on-axis particle loses in one turn with every cavity off, eV."""
    ring = lattice.deepcopy()
    for element in ring:
        if isinstance(element, at.RFCavity):
            element.Voltage = 0.0
    out = ring.track(np.zeros((6, 1)), nturns=1)[0]
    return -float(out[4, 0, 0, 0]) * float(ring.energy)


def _radiation_integral_2(lattice: Any) -> float:
    return sum(
        float(element.BendingAngle) ** 2 / float(element.Length)
        for element in lattice
        if isinstance(element, at.Dipole)
    )


class TestTheSetting:
    @pytest.mark.parametrize("radiation", [None, False])
    def test_without_it_the_deck_stands_as_written(
        self, demo: dict[str, Any], radiation: Any
    ) -> None:
        model = _build(demo, radiation=radiation)
        assert not _radiating(model.lattice)
        assert _energy_loss_ev(model.lattice) == pytest.approx(0.0, abs=1e-3)

    def test_with_it_every_magnet_radiates_and_the_cavity_is_untouched(
        self, demo: dict[str, Any]
    ) -> None:
        plain, model = _build(demo), _build(demo, radiation=True)
        assert _radiating(model.lattice) == {"Dipole", "Quadrupole", "Sextupole"}
        cavities = [e for e in model.lattice if isinstance(e, at.RFCavity)]
        assert [e.PassMethod for e in cavities] == [
            e.PassMethod for e in plain.lattice if isinstance(e, at.RFCavity)
        ]

    def test_the_deck_on_disk_is_not_made_to_radiate(self, demo: dict[str, Any]) -> None:
        _build(demo, radiation=True)
        assert not _radiating(_build(demo).lattice)

    @pytest.mark.parametrize("radiation", ["yes", 1, 0.5])
    def test_anything_but_true_or_false_stops(self, demo: dict[str, Any], radiation: Any) -> None:
        with pytest.raises(FacilityBuildError, match="pyat.radiation"):
            engine.prepare(demo["deck"], _settings(demo, radiation=radiation), model=MODEL)

    def test_a_single_pass_model_refuses_it(self, demo: dict[str, Any]) -> None:
        settings = {
            "pyat": {
                "solve": "single_pass",
                "twiss_in": {"beta": [5.0, 5.0], "alpha": [0.0, 0.0]},
                "radiation": True,
            }
        }
        with pytest.raises(FacilityBuildError, match="pyat.radiation"):
            engine.prepare(demo["deck"], settings, model=MODEL)


class TestTheRadiatingRing:
    def test_a_turn_costs_the_energy_the_bends_radiate(self, demo: dict[str, Any]) -> None:
        lattice = _build(demo, radiation=True).lattice
        energy_gev = float(lattice.energy) / 1.0e9
        expected_ev = (
            C_GAMMA * energy_gev**4 * _radiation_integral_2(lattice) / (2.0 * math.pi) * 1.0e9
        )
        assert expected_ev > 1.0e3
        assert _energy_loss_ev(lattice) == pytest.approx(expected_ev, rel=0.01)

    def test_the_closed_orbit_sits_at_the_synchronous_phase(self, demo: dict[str, Any]) -> None:
        model = _build(demo, radiation=True)
        assert not model.no_closed_orbit
        lattice = model.lattice
        (cavity,) = [e for e in lattice if isinstance(e, at.RFCavity)]
        wavelength = at.constants.clight / float(cavity.Frequency)
        lag = (
            math.asin(_energy_loss_ev(lattice) / float(cavity.Voltage)) * wavelength / (2 * math.pi)
        )
        orbit, _ = lattice.find_orbit6()
        assert abs(lag) > 1.0e-3
        assert -orbit[5] == pytest.approx(lag, rel=0.02)

    def test_the_orbit_response_is_the_radiating_decks(self, demo: dict[str, Any]) -> None:
        kwargs = {"correctors": [CORRECTOR], "step": {CORRECTOR: 0.1}}
        plain = engine.response_matrix(demo["deck"], demo["wiring"], _settings(demo), **kwargs)
        radiating = engine.response_matrix(
            demo["deck"], demo["wiring"], _settings(demo, radiation=True), **kwargs
        )
        own = engine.response_matrix(
            _radiating_deck(demo), demo["wiring"], _settings(demo), **kwargs
        )
        assert not radiating.unsolved
        assert np.asarray(radiating.matrix) == pytest.approx(np.asarray(own.matrix), rel=1e-9)
        assert np.abs(np.asarray(radiating.matrix) - np.asarray(plain.matrix)).max() > 0.0

    def test_a_voltage_below_the_loss_is_refused_by_a_ring_that_is_not_tolerant(
        self, demo: dict[str, Any]
    ) -> None:
        wiring = [
            *demo["wiring"],
            {
                "address": VOLTAGE,
                "element": "CAV",
                "engine": {"attribute": "Voltage"},
                "direction": "write",
                "default": 1.0e6,
            },
        ]
        model = engine.build(MODEL, wiring, demo["deck"], _settings(demo, radiation=True), {})
        with pytest.raises(OrbitSolveError):
            model.set({VOLTAGE: 0.5 * _energy_loss_ev(model.lattice)})
        assert model.get([VOLTAGE])[VOLTAGE] == pytest.approx(1.0e6)


def _radiating_deck(demo: dict[str, Any]) -> Path:
    """The plain deck with the radiating pass methods written into the file."""
    ring = at.load_lattice(str(demo["deck"]))
    ring.enable_6d(cavity_pass=None, sextupole_pass="auto")
    deck = Path(demo["deck"]).with_name("SR-radiating.json")
    ring.save(str(deck))
    return deck


class TestTheRFOff:
    def test_there_is_no_closed_orbit_and_the_ring_stays_served(self, demo: dict[str, Any]) -> None:
        model = _build(demo, radiation=True, turns=True)
        assert not model.no_closed_orbit
        model.set({SWITCH: 0.0})
        assert model.no_closed_orbit

    def test_a_tracked_bunch_loses_energy_until_the_aperture_takes_it(
        self, demo: dict[str, Any]
    ) -> None:
        model = _build(demo, radiation=True, turns=True)
        lattice = model.lattice
        loss = _energy_loss_ev(lattice) / float(lattice.energy)
        monitors = [i for i, e in enumerate(lattice) if isinstance(e, at.Monitor)]
        flat = lattice.deepcopy()
        flat.disable_6d()
        _, _, optics = flat.get_optics(refpts=monitors)
        dispersion = optics.dispersion[:, 0]
        model.set({SWITCH: 0.0})

        tracked = model.lattice.deepcopy()
        pyat_turns.write_apertures(tracked, elliptical=(APERTURE, APERTURE))
        injection = next(i for i, e in enumerate(tracked) if e.FamName == INJECTION)
        reading = pyat_turns.track_shot(tracked, injection, monitors, np.zeros(6), 2000)

        alive = reading.alive.min(axis=0) > 0
        lost_at = int(np.argmin(alive))
        assert not alive[-1], "the bunch must be lost within the tracked turns"
        # the aperture is reached where the largest dispersion carries the energy lost so far
        estimate = APERTURE / (np.abs(dispersion).max() * loss)
        assert 0.3 * estimate < lost_at < 1.5 * estimate
        # before that the bunch moves inwards in step with the energy it has lost
        turn = lost_at // 4
        drift = reading.x[:, turn]
        assert drift == pytest.approx(-dispersion * loss * (turn + 0.5), rel=0.25, abs=2.0e-5)
