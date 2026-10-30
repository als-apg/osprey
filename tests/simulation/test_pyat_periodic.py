"""The pyat engine's periodic solve: lume-pyat's solve with the orbit searched once.

``solve_orbit_once`` stands in for :func:`lume_pyat.solve.solve_orbit` and has
to be it in everything but the tracking it costs. Each case here solves one
ring both ways and asks for the same bits or the same refusal, word for word.
The rings are a small synthetic FODO ring, 4D and with a cavity enabled.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

at = pytest.importorskip("at")

import numpy as np  # noqa: E402
from lume_pyat.exceptions import OrbitSolveError  # noqa: E402
from lume_pyat.simulator import PyATSimulator  # noqa: E402
from lume_pyat.solve import solve_orbit  # noqa: E402

from osprey.simulation.engines.pyat_periodic import (  # noqa: E402
    PeriodicSimulator,
    solve_orbit_once,
)

CELLS = 8


def _ring(*, six_d: bool) -> Any:
    """Eight FODO cells with a monitor and two correctors each, 4D or 6D."""
    elements: list[Any] = []
    for cell in range(1, CELLS + 1):
        tag = f"{cell:02d}"
        elements += [
            at.Monitor(f"BPM_{tag}"),
            at.Drift("DRIFT", 0.4),
            at.Quadrupole(f"QUAD_F_{tag}", 0.3, 1.0),
            at.Drift("DRIFT", 0.4),
            at.Sextupole(f"SEXT_F_{tag}", 0.15, 1.0),
            at.Drift("DRIFT", 0.4),
            at.Corrector(f"COR_H_{tag}", 0.0, [0.0, 0.0]),
            at.Dipole(f"BEND_{tag}", 1.0, 2 * np.pi / CELLS),
            at.Corrector(f"COR_V_{tag}", 0.0, [0.0, 0.0]),
            at.Drift("DRIFT", 0.4),
            at.Sextupole(f"SEXT_D_{tag}", 0.15, -1.0),
            at.Drift("DRIFT", 0.4),
            at.Quadrupole(f"QUAD_D_{tag}", 0.3, -1.0),
            at.Drift("DRIFT", 0.4),
        ]
    ring = at.Lattice(elements, name="RING", energy=1.0e9, periodicity=1)
    if not six_d:
        ring.disable_6d()
        return ring
    harmonic = 32
    frequency = harmonic * ring.get_revolution_frequency()
    ring.insert(0, at.RFCavity("RF", 0.0, 1.0e6, frequency, harmonic, ring.energy))
    ring.enable_6d(at.RFCavity)
    return ring


def _kicked(ring: Any) -> None:
    ring[ring.get_uint32_index("COR_H_01")[0]].KickAngle = np.array([2.0e-4, 0.0])
    ring[ring.get_uint32_index("COR_V_03")[0]].KickAngle = np.array([0.0, -1.0e-4])


def _non_finite_quadrupole(ring: Any) -> None:
    ring[ring.get_uint32_index("QUAD_F_01")[0]].PolynomB[1] = np.nan


def _vertically_unstable(ring: Any) -> None:
    for index in ring.get_uint32_index("QUAD_F_*"):
        ring[index].K = 3.0


def _unstable_in_both_planes(ring: Any) -> None:
    for index in ring.get_uint32_index("QUAD_F_*"):
        ring[index].K = -3.0


CHANGES: dict[str, Callable[[Any], None]] = {
    "nominal": lambda ring: None,
    "kicked": _kicked,
    "non-finite quadrupole": _non_finite_quadrupole,
    "vertically unstable": _vertically_unstable,
    "unstable in both planes": _unstable_in_both_planes,
}


def _outcome(solve: Callable[[Any], np.ndarray], ring: Any) -> np.ndarray | str:
    try:
        return solve(ring)
    except OrbitSolveError as exc:
        return f"OrbitSolveError: {exc}"


@pytest.mark.parametrize("six_d", [False, True], ids=["4d", "6d"])
@pytest.mark.parametrize("change", list(CHANGES), ids=list(CHANGES))
def test_the_solve_is_lume_pyats_bit_for_bit_or_refuses_in_its_words(six_d, change):
    stock, once = _ring(six_d=six_d), _ring(six_d=six_d)
    CHANGES[change](stock)
    CHANGES[change](once)

    expected = _outcome(solve_orbit, stock)
    observed = _outcome(solve_orbit_once, once)

    if isinstance(expected, str):
        assert observed == expected
    else:
        assert isinstance(observed, np.ndarray)
        np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("six_d", [False, True], ids=["4d", "6d"])
def test_a_non_finite_orbit_trips_the_last_guard_as_lume_pyats_does(six_d, monkeypatch):
    # Only a patched search reaches this guard: find_m66 differentiates about
    # the orbit, so a lattice whose orbit is non-finite trips the first guard.
    name = "find_orbit6" if six_d else "find_orbit4"

    def non_finite(*args: Any, **kwargs: Any) -> Any:
        return np.zeros(6), np.full((CELLS, 6), np.nan)

    non_finite.__name__ = name
    monkeypatch.setattr(at, name, non_finite)

    expected = _outcome(solve_orbit, _ring(six_d=six_d))
    observed = _outcome(solve_orbit_once, _ring(six_d=six_d))

    assert observed == expected == f"OrbitSolveError: {name} returned a non-finite closed orbit"


@pytest.mark.parametrize("six_d", [False, True], ids=["4d", "6d"])
def test_find_m66_is_handed_the_one_orbit_searched_for(six_d, monkeypatch):
    name = "find_orbit6" if six_d else "find_orbit4"
    search, differentiate = getattr(at, name), at.find_m66
    searched: list[Any] = []
    handed: list[Any] = []

    def counting_search(*args: Any, **kwargs: Any) -> Any:
        result = search(*args, **kwargs)
        searched.append(result[0])
        return result

    def recording_differentiate(*args: Any, **kwargs: Any) -> Any:
        handed.append(kwargs.get("orbit"))
        return differentiate(*args, **kwargs)

    monkeypatch.setattr(at, name, counting_search)
    monkeypatch.setattr(at, "find_m66", recording_differentiate)
    ring = _ring(six_d=six_d)
    _kicked(ring)

    solve_orbit_once(ring)

    assert len(searched) == 1
    assert len(handed) == 1
    assert handed[0] is searched[0]


def test_the_simulator_reads_what_lume_pyats_simulator_reads():
    stock, once = _ring(six_d=True), _ring(six_d=True)
    _kicked(stock)
    _kicked(once)

    assert PeriodicSimulator(once).solve() == PyATSimulator(stock).solve()


def test_a_failed_solve_folds_into_orbit_solve_error_and_keeps_the_last_solution(monkeypatch):
    simulator = PeriodicSimulator(_ring(six_d=True))
    solved = simulator.solve()

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise at.AtError("no synchronous phase")

    monkeypatch.setattr(at, "find_orbit6", refuse)

    with pytest.raises(OrbitSolveError, match="closed orbit solve raised AtError"):
        simulator.solve()
    assert simulator.last_solution == solved
