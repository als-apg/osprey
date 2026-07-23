"""Shared current->kick calibration for the ALS-U AR virtual accelerator.

Leaf module: both :mod:`.strengths` (which applies the formula) and
:mod:`.response` (whose model oracle rides on it) depend on this constant,
so it lives below both of them -- keeping it in either would force an import
cycle between the two.
"""

from __future__ import annotations

# Current-to-kick calibration applied by strengths.StrengthMap for the
# HCM/VCM corrector families: KickAngle[plane] = I / AMPS_PER_RADIAN_KICK.
#
# This is still a deterministic, sign-correct calibration constant, not a
# claim of physical realism -- but its magnitude matters now in a way it
# didn't on the toy ring: the real AR lattice carries much stronger
# sextupoles (low-emittance MBA design), so a corrector kick large enough to
# move the closed orbit by several mm picks up measurable *nonlinear*
# sextupole feed-down (the +I/-I response stops being antisymmetric). Kept
# at 1e6 (rather than the toy ring's 1e4) so a corrector's typical +-10 A
# range stays in the small-signal/quasi-linear regime (tens of microns of
# orbit shift, not millimeters) -- see test_lattice.py's antisymmetry check.
AMPS_PER_RADIAN_KICK = 1_000_000.0
