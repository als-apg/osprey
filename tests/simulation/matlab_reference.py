"""Frozen MATLAB AT reference optics for the ALS-U AR source lattice.

Computed **once, offline** from the real MATLAB source lattice, and committed as
constants so the fidelity test (``test_fidelity.py``) is CI-runnable without a
live MATLAB. This is the *independent* correctness oracle for the hand-port — the
ring-vs-spec consistency test is circular (same author/source), this is not
(PROPOSAL.md FR7 / SC5).

Provenance
----------
* Source:  ``/Users/thellert/LBL/matlab/sc/applications/ALSU_AR_work/lattices/ALS_U_AR_v6.m``
* Toolbox: MATLAB AT (``atmat`` at ``/Users/thellert/LBL/matlab/at``), MATLAB R2023b
* Date:    2026-07-15
* Recipe (4D — radiation OFF and cavity OFF, apples-to-apples with pyAT
  ``ring.disable_6d()``)::

      ring = ALS_U_AR_v6;
      % strip radiation from magnet pass methods and switch the cavity off:
      %   '*RadPass' -> '*Pass',  'CavityPass' -> 'IdentityPass'
      [~, nu, xi] = atlinopt(ring4, 1e-6, 1);   % fractional tunes + chromaticity

  (``atradoff`` / ``atdisable_6d`` are avoided — this atmat build errors on the
  36-bend index array; the manual pass-method rewrite is byte-for-byte the same
  4D reduction and version-independent.)
"""

# Fractional betatron tunes (νx, νy).
NU_X = 0.2207248515
NU_Y = 0.3281225019

# Linear chromaticity (ξx, ξy) — the sextupole-corrected values (design point).
XI_X = 1.0162577989
XI_Y = 1.0026450020
