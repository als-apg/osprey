"""What holds about the LUME releases OSPREY pins.

lume-base states no exception for a request that has no solve, so lume-pyat's
``OrbitSolveError`` derives from ``Exception`` alone and the pyat plug-in names
it for the engine-neutral callers. The test fails on the first pinned pair of
releases where a lume-base class stands behind it.

lume-pyat's ``solve_orbit`` calls ``at.find_m66`` without the orbit it searches
for, so the matrix's own search repeats it and the response check solves with
``pyat_periodic`` instead. That test fails on the first pinned lume-pyat that
hands the orbit over.
"""

from __future__ import annotations


def test_no_lume_base_class_is_behind_the_orbit_solve_error() -> None:
    from lume_pyat.exceptions import OrbitSolveError

    shared = [
        f"{base.__module__}.{base.__qualname__}"
        for base in OrbitSolveError.__mro__
        if base.__module__.partition(".")[0] == "lume"
    ]
    assert not shared, (
        f"lume-pyat's OrbitSolveError now derives from {shared}: the engine-neutral callers "
        "(simulation/response.py, facility/layers/mml/response_check.py) catch that class, and "
        "SOLVE_ERRORS in osprey.simulation.engines.pyat goes with its __all__ entry, its "
        "EnginePlugin row, test_pyat_declares_the_orbit_solve_error, this test and its row in "
        "tests/infrastructure/test_stand_in_canaries.py"
    )


def test_lume_pyat_still_hands_find_m66_no_orbit(monkeypatch) -> None:
    import at
    import numpy as np
    from lume_pyat import solve

    lattice = at.Lattice(
        [
            at.Monitor("BPM"),
            at.Quadrupole("QF", 0.3, 1.0),
            at.Drift("D", 1.0),
            at.Quadrupole("QD", 0.3, -1.0),
            at.Drift("D", 1.0),
        ],
        energy=1.0e9,
        periodicity=1,
    )
    lattice.disable_6d()
    differentiate = at.find_m66
    handed: list[bool] = []

    def recording(*args: object, **kwargs: object) -> object:
        handed.append(kwargs.get("orbit") is not None)
        return differentiate(*args, **kwargs)

    monkeypatch.setattr(at, "find_m66", recording)

    assert np.all(np.isfinite(solve.solve_orbit(lattice)))
    assert handed == [False], (
        "lume-pyat's solve_orbit now hands find_m66 the orbit it searched for: "
        "src/osprey/simulation/engines/pyat_periodic.py goes, with its import and use in "
        "osprey.simulation.engines.pyat.response_matrix, tests/simulation/test_pyat_periodic.py, "
        "the PeriodicSimulator patches in tests/facility/test_response_check.py and "
        "tests/simulation/test_pyat_response_matrix.py, this test and its row in "
        "tests/infrastructure/test_stand_in_canaries.py"
    )
