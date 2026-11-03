"""What holds about the LUME releases OSPREY pins.

lume-base states no exception for a request that has no solve, so lume-pyat's
``OrbitSolveError`` derives from ``Exception`` alone and the pyat plug-in names
it for the engine-neutral callers. The test fails on the first pinned pair of
releases where a lume-base class stands behind it. Nor does lume-base state an
exception for a request an engine does not answer, so lume-pyat's
``UnknownElementError`` derives from ``ValueError`` alone and the pyat plug-in
names its own ``RequestRefused``; that test fails the same way.
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


def test_no_lume_base_class_is_behind_lume_pyats_request_refusals() -> None:
    from lume_pyat.exceptions import UnknownElementError

    shared = [
        f"{base.__module__}.{base.__qualname__}"
        for base in UnknownElementError.__mro__
        if base.__module__.partition(".")[0] == "lume"
    ]
    assert not shared, (
        f"lume-pyat's UnknownElementError now derives from {shared}: REFUSAL_ERRORS and "
        "RequestRefused in osprey.simulation.engines.pyat adopt that base, and the stand-in "
        "comment, their __all__ entries, their EnginePlugin row, "
        "test_pyat_declares_its_request_refusal, this test and its row in "
        "tests/infrastructure/test_stand_in_canaries.py go with it"
    )
