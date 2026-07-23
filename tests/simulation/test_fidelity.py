"""Independent lattice-fidelity check for the hand-ported ALS-U AR ring (SC5).

This is the *independent* correctness oracle for the pyAT hand-port. The
ring-vs-spec consistency test is circular (same author, same source assumptions);
this one compares the built pyAT ring against optics computed **offline in
MATLAB AT** from the original source lattice (frozen in
:mod:`tests.simulation.matlab_reference`), so it is not.

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
"""

import at

from osprey.simulation.lattice import build_ring
from tests.simulation import matlab_reference as ref


def test_lattice_fidelity_against_matlab_reference():
    r = build_ring().deepcopy()
    r.disable_6d()  # radiation off + cavity off -> 4D (is_6d is False)
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
