"""Independent lattice-fidelity check for the example facility's SR deck.

This compares the committed pyAT deck against optics computed **offline in
MATLAB AT** from the original source lattice (frozen in
:mod:`tests.simulation.matlab_reference`), an oracle independent of the deck.

Apples-to-apples recipe (4D, radiation OFF, cavity OFF)
-------------------------------------------------------
The MATLAB reference was computed with radiation stripped from the magnet pass
methods and the RF cavity switched off (``atlinopt`` on the 4D ring). To match
that exactly on the pyAT side we call :meth:`at.Lattice.disable_6d`, which turns
radiation off and the cavity off, leaving a 4D lattice (``is_6d`` is ``False``).
``at.get_optics(..., get_chrom=True, dp=1e-6)`` then yields the linear tunes and
chromaticity from the one-turn map at the same small momentum offset used for the
MATLAB finite-difference chromaticity.

Tolerances
----------
* Fractional tune:  ``|Δν| < 1e-3`` (compare only the fractional part; the
  integer tune is not carried by the reference).
* Chromaticity:     ``|Δξ| < 0.1`` absolute.

The deck's own fractional tunes are also pinned at ``1e-9``, so any edit to the
committed deck that moves the optics fails here rather than downstream.
"""

import at
import pytest

from tests.simulation import matlab_reference as ref
from tests.simulation._sr_deck import load_sr_deck_4d

#: The deck's fractional tunes (4D, ``dp=1e-6``), as it is committed.
DECK_TUNES = (0.22072485145318438, 0.3281225019273429)


def test_lattice_fidelity_against_matlab_reference():
    r = load_sr_deck_4d()
    assert r.is_6d is False

    res = at.get_optics(r, get_chrom=True, dp=1e-6)  # (elemdata0, ringdata, elemdata)
    ringdata = res[1]
    nu = ringdata["tune"]
    xi = ringdata["chromaticity"]

    dnu_x = abs((nu[0] % 1.0) - ref.NU_X)
    dnu_y = abs((nu[1] % 1.0) - ref.NU_Y)
    dxi_x = abs(xi[0] - ref.XI_X)
    dxi_y = abs(xi[1] - ref.XI_Y)

    assert dnu_x < 1e-3, f"nu_x delta {dnu_x} (got {nu[0] % 1.0}, ref {ref.NU_X})"
    assert dnu_y < 1e-3, f"nu_y delta {dnu_y} (got {nu[1] % 1.0}, ref {ref.NU_Y})"
    assert dxi_x < 0.1, f"xi_x delta {dxi_x} (got {xi[0]}, ref {ref.XI_X})"
    assert dxi_y < 0.1, f"xi_y delta {dxi_y} (got {xi[1]}, ref {ref.XI_Y})"


def test_deck_tunes_are_pinned():
    res = at.get_optics(load_sr_deck_4d(), get_chrom=True, dp=1e-6)
    nu = res[1]["tune"]

    assert nu[0] % 1.0 == pytest.approx(DECK_TUNES[0], abs=1e-9)
    assert nu[1] % 1.0 == pytest.approx(DECK_TUNES[1], abs=1e-9)
