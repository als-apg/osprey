"""Closed-orbit finding sanity checks for the 4D canonical ALS-U AR ring.

This is throwaway physics insurance: it locks down that ``at.find_orbit4``
actually converges on the shared ring topology (not a given for a hand-ported
lattice) and that a single corrector kick produces a bounded, sign-odd orbit
response, before later phases build orbit-response-matrix scan plans on top
of it.

Elements are selected by pyAT element TYPE (``at.Corrector``, ``at.Monitor``),
never by ``FamName`` string, so this file stays valid across the concurrent
FamName-rename work happening elsewhere in this phase -- orbit-finding is a
topology property, not a naming one.
"""

from __future__ import annotations

import at
import numpy as np

from osprey.simulation.lattice import build_ring


def _fresh_4d_ring():
    """A fresh, independent 4D copy of the shared ring (radiation+cavity off)."""
    ring = build_ring().deepcopy()
    ring.disable_6d()  # mutates in place; returns None -- do not chain/assign
    return ring


def _monitor_refpts(ring):
    return np.array(
        [i for i, e in enumerate(ring) if isinstance(e, at.Monitor)]
    )


def test_orbit4_closes_at_monitors():
    """find_orbit4 converges on the ideal 4D ring and closes at every monitor."""
    ring = _fresh_4d_ring()
    refpts = _monitor_refpts(ring)
    assert len(refpts) == 72  # SC5 AR ring: 72 BPMs, one per unit cell

    orbit0, orbit_at_refpts = at.find_orbit4(ring, refpts=refpts)

    assert np.all(np.isfinite(orbit0))
    assert orbit_at_refpts.shape == (72, 6)
    assert np.all(np.isfinite(orbit_at_refpts))

    x, y = orbit_at_refpts[:, 0], orbit_at_refpts[:, 2]
    max_abs_x, max_abs_y = np.max(np.abs(x)), np.max(np.abs(y))
    print(f"closed orbit at monitors: max|x| = {max_abs_x:.3e} m, "
          f"max|y| = {max_abs_y:.3e} m")
    # Evidence (ideal ring, no errors -> orbit is machine-precision zero):
    # max|x| = 0.000e+00 m, max|y| = 0.000e+00 m.
    assert max_abs_x < 1e-6
    assert max_abs_y < 1e-6


def test_corrector_kick_bounded_and_odd():
    """A small kick from one corrector gives a bounded, sign-odd orbit shift.

    Uses ``+theta``/``-theta`` on independent fresh 4D copies (deepcopy per
    kick so no element state leaks between the two evaluations), and checks
    the linear-response prediction ``orbit(+theta) ~= -orbit(-theta)`` at the
    monitors.

    theta = 1e-5 rad (rather than the more extreme 1e-4 rad) is chosen
    deliberately: at 1e-4 rad the measured antisymmetry residual is already
    ~4% of the signal (a genuine second-order effect -- sextupole feed-down
    at the small but nonzero displaced orbit, confirmed to scale as theta**2
    while the orbit itself scales as theta, by direct measurement over
    theta in {1e-4, 1e-5, 1e-6, 1e-7}). At theta = 1e-5 the same residual is
    already two orders of magnitude down (~0.4% of signal), which is a
    comfortably "linear-regime" kick to assert oddness against.
    """
    theta = 1e-5  # rad; small-kick linear-response regime (see docstring)

    ring0 = _fresh_4d_ring()
    refpts = _monitor_refpts(ring0)

    ring_plus = _fresh_4d_ring()
    cor_plus = next(e for e in ring_plus if isinstance(e, at.Corrector))
    cor_plus.KickAngle = [theta, 0.0]
    _, orbit_plus = at.find_orbit4(ring_plus, refpts=refpts)

    ring_minus = _fresh_4d_ring()
    cor_minus = next(e for e in ring_minus if isinstance(e, at.Corrector))
    cor_minus.KickAngle = [-theta, 0.0]
    _, orbit_minus = at.find_orbit4(ring_minus, refpts=refpts)

    assert np.all(np.isfinite(orbit_plus))
    assert np.all(np.isfinite(orbit_minus))

    x_plus, x_minus = orbit_plus[:, 0], orbit_minus[:, 0]
    y_plus, y_minus = orbit_plus[:, 2], orbit_minus[:, 2]

    # Generous physical sanity ceiling: a single corrector at a 1e-5 rad kick
    # should never produce a meter-scale orbit on a real accelerator lattice;
    # 0.1 m is many orders of magnitude above the observed ~1e-4 m response.
    max_abs_x = max(np.max(np.abs(x_plus)), np.max(np.abs(x_minus)))
    print(f"corrector kick response: max|x| = {max_abs_x:.3e} m at "
          f"theta = {theta:.1e} rad "
          f"({max_abs_x / theta:.3e} m/rad per unit kick)")
    # Evidence: max|x| ~= 1.176e-04 m at theta = 1.0e-05 rad
    # (~1.176e+01 m/rad per unit kick).
    assert max_abs_x < 0.1

    # Sign-odd linear response: orbit(+theta) ~= -orbit(-theta) at monitors.
    # atol = 2e-6 m gives ~4x margin over the measured second-order residual
    # (~4.6e-7 m at this theta; see docstring).
    assert np.allclose(x_plus, -x_minus, atol=2e-6)
    assert np.allclose(y_plus, -y_minus, atol=2e-6)
