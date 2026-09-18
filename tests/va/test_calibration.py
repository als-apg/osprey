"""The virtual accelerator's hardware<->physics conversions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from osprey.services.virtual_accelerator.bindings import Linear, Table
from osprey.services.virtual_accelerator.lattice import calibration

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mml"

#: A storage ring's beam energy in GeV, and the rigidity the control system's
#: own expression gives for it: ``(10/2.99792458)*sqrt((E+E0)^2 - E0^2)`` with
#: the electron rest mass ``E0 = 0.51099906e-3`` GeV.
RING_ENERGY_GEV = 3.0
RING_RIGIDITY = 10.008627220193127


@pytest.fixture(scope="module")
def skew_quadrupole() -> dict:
    """One family's exported conversion pair, in the hardware units it is set in.

    The two gains are sampled independently -- each along the control system's
    own path -- which is why they are not exact reciprocals of one another.
    """
    document = json.loads(
        (FIXTURES / "nsls2" / "nsls2.storagering.ao.json").read_text(encoding="utf-8")
    )
    setpoint = document["SQ"]["Setpoint"]
    return {
        "to_physics": float(setpoint["HW2PhysicsParams"][0]),
        "to_hardware": float(setpoint["Physics2HWParams"][0]),
        "range": tuple(float(edge) for edge in setpoint["Range"][0]),
    }


class TestLinearConversion:
    """A straight-line curve, the shape a proportional conversion exports as."""

    def test_applies_gain_and_offset_per_device(self) -> None:
        """Every device's value goes through the curve, and the shape survives."""
        currents = np.array([-2.0, 0.0, 4.5])

        physics = calibration.to_physics(Linear(gain=0.25, offset=-1.0), currents)

        assert physics.shape == currents.shape
        assert physics == pytest.approx([-1.5, -1.0, 0.125])

    def test_inverse_carries_its_own_gain_and_offset(self) -> None:
        """The readback path reads the inverse's parameters, not the calibration's."""
        readings = np.array([1.0, 3.0])

        hardware = calibration.to_hardware(Linear(gain=4.0, offset=0.5), readings)

        assert hardware == pytest.approx([4.5, 12.5])


class TestTableConversion:
    """A sampled curve: piecewise linear inside, along its end segments outside."""

    CURVE = Table(grid=(-2.0, 0.0, 1.0, 3.0), values=(-8.0, 0.0, 1.0, 27.0))

    def test_reads_the_sampled_points_exactly(self) -> None:
        """A value that was sampled comes back as the sample."""
        read = calibration.to_physics(self.CURVE, np.array(self.CURVE.grid))

        assert read == pytest.approx(self.CURVE.values)

    def test_interpolates_between_the_sampled_points(self) -> None:
        """Between two points the curve is the straight line joining them."""
        read = calibration.to_physics(self.CURVE, np.array([-1.0, 0.5, 2.0]))

        assert read == pytest.approx([-4.0, 0.5, 14.0])

    def test_continues_the_first_segment_below_the_grid(self) -> None:
        """Beyond the low end the curve keeps the slope of its first segment."""
        read = calibration.to_physics(self.CURVE, np.array([-4.0, -3.0]))

        assert read == pytest.approx([-16.0, -12.0])

    def test_continues_the_last_segment_above_the_grid(self) -> None:
        """Beyond the high end the curve keeps the slope of its last segment."""
        read = calibration.to_physics(self.CURVE, np.array([4.0, 5.0]))

        assert read == pytest.approx([40.0, 53.0])

    def test_extrapolation_stays_straight(self) -> None:
        """Equal steps beyond the grid change the value by equal amounts."""
        read = calibration.to_physics(self.CURVE, np.array([3.0, 13.0, 23.0]))

        assert read[1] - read[0] == pytest.approx(read[2] - read[1])

    def test_a_falling_grid_reads_the_same_curve(self) -> None:
        """A negative-gain conversion samples onto a falling grid, same curve."""
        falling = Table(grid=self.CURVE.grid[::-1], values=self.CURVE.values[::-1])
        points = np.array([-4.0, -2.0, 0.5, 3.0, 5.0])

        assert calibration.to_physics(falling, points) == pytest.approx(
            calibration.to_physics(self.CURVE, points)
        )

    def test_a_falling_grid_extrapolates_from_its_own_ends(self) -> None:
        """A falling grid's end segments are its first and last sampled pairs."""
        falling = Table(grid=(4.0, 2.0, 0.0), values=(1.0, 5.0, 9.0))

        read = calibration.to_physics(falling, np.array([6.0, -2.0]))

        assert read == pytest.approx([-3.0, 13.0])


class TestRoundTrip:
    """What a written hardware value reads back as, through the exported pair."""

    def test_returns_the_written_value(self, skew_quadrupole: dict) -> None:
        """Across the family's whole range the readback is the setpoint to 1e-9."""
        low, high = skew_quadrupole["range"]
        currents = np.linspace(low, high, 33)
        forward = Linear(gain=skew_quadrupole["to_physics"], offset=0.0)
        backward = Linear(gain=skew_quadrupole["to_hardware"], offset=0.0)

        read = calibration.to_hardware(backward, calibration.to_physics(forward, currents))

        assert np.max(np.abs(read - currents)) < 1e-9

    def test_the_sampled_table_returns_the_written_value(self, skew_quadrupole: dict) -> None:
        """The same pair sampled as tables round-trips as closely."""
        low, high = skew_quadrupole["range"]
        grid = np.linspace(low, high, 33)
        physics_grid = grid * skew_quadrupole["to_physics"]
        forward = Table(grid=tuple(grid), values=tuple(physics_grid))
        backward = Table(
            grid=tuple(physics_grid),
            values=tuple(physics_grid * skew_quadrupole["to_hardware"]),
        )
        currents = np.linspace(low, high, 101)

        read = calibration.to_hardware(backward, calibration.to_physics(forward, currents))

        assert np.max(np.abs(read - currents)) < 1e-9

    def test_the_two_directions_are_independent_data(self, skew_quadrupole: dict) -> None:
        """The pair is not an exact reciprocal, so neither direction is derivable."""
        product = skew_quadrupole["to_physics"] * skew_quadrupole["to_hardware"]

        assert product != 1.0
        assert product == pytest.approx(1.0, abs=1e-12)


class TestRigidity:
    """The beam rigidity, and what it does to a calibration at another energy."""

    def test_matches_the_control_systems_expression(self) -> None:
        """The rigidity is the control system's own, to a part in 1e12."""
        assert calibration.brho(RING_ENERGY_GEV) == pytest.approx(RING_RIGIDITY, abs=1e-12)

    def test_the_rest_mass_moves_the_energy_factor(self) -> None:
        """Dropping the rest mass shifts the factor past the agreement it is judged by.

        A family is called rigidity-scaled when its physics value times the
        rigidity holds to a part in a million across an energy step. The
        massless ``E / c`` line misses the factor over a two-percent step by
        several times that, which is enough to decide the question wrongly.
        """
        raised = RING_ENERGY_GEV * 1.02

        exact = calibration.energy_factor(raised, RING_ENERGY_GEV)
        massless = RING_ENERGY_GEV / raised

        assert abs(exact - massless) / massless > 1e-6

    def test_reads_an_array_of_energies(self) -> None:
        """Rigidity rises with energy, one entry per energy asked for."""
        rigidity = calibration.brho(np.array([0.1, 1.0, RING_ENERGY_GEV]))

        assert rigidity.shape == (3,)
        assert np.all(np.diff(rigidity) > 0.0)

    def test_the_deck_energy_costs_nothing(self) -> None:
        """At the energy its calibrations were sampled at the factor is exactly one."""
        assert calibration.energy_factor(RING_ENERGY_GEV, RING_ENERGY_GEV) == 1.0

    def test_undoes_the_change_in_rigidity(self) -> None:
        """The factor is exactly what turns one rigidity into the other."""
        raised = RING_ENERGY_GEV * 1.02

        factor = calibration.energy_factor(raised, RING_ENERGY_GEV)

        assert factor * calibration.brho(raised) == pytest.approx(
            calibration.brho(RING_ENERGY_GEV), rel=1e-15
        )

    def test_a_stiffer_beam_takes_a_smaller_factor(self) -> None:
        """Above the deck energy the same hardware value is worth less physics."""
        assert calibration.energy_factor(RING_ENERGY_GEV * 1.02, RING_ENERGY_GEV) < 1.0
        assert calibration.energy_factor(RING_ENERGY_GEV * 0.98, RING_ENERGY_GEV) > 1.0


class TestSurface:
    """What the module offers, and what it refuses to offer."""

    def test_offers_both_directions_and_the_rigidity(self) -> None:
        """The two conversions and the rigidity pair are the module's contract."""
        contract = {"brho", "energy_factor", "to_hardware", "to_physics"}

        assert contract <= set(calibration.__all__)
        assert all(callable(getattr(calibration, name)) for name in contract)
