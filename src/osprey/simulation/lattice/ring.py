"""Hand-ported ALS-U Accumulator Ring lattice (pyAT ``at.Lattice``).

This is a by-hand port of the real MATLAB Accelerator Toolbox design
``ALS_U_AR_v6.m`` (E0 = 2.0 GeV, harmonic 304, 12 superperiods, summed
circumference ≈ 182.12195088 m). ``at.load_m`` cannot read a real MATLAB source
(it parses only pyAT's own ``save_m`` output — no variables, no ``SUP()``
closure), so each element is re-authored in Python and the superperiod / full-ring
assembly is replicated as list building.

Conversions from the source constructors (see PROPOSAL.md FR1):

* ``quadrupole(name, L, K)`` → :class:`at.Quadrupole` with ``k = K`` (sign identical).
* ``sextupole(name, L, S)`` → :class:`at.Sextupole` with ``h = S`` **1:1** (no ×2;
  the source ``PolynomB(3)`` maps directly onto pyAT ``PolynomB[2]``).
* ``sbend(name, L, A, E1, E2, K)`` with ``E1 = E2 = A/2`` → :class:`at.Dipole`
  with ``bending_angle = A``, ``EntranceAngle = E1``, ``ExitAngle = E2``, ``k = K``
  (kept a sector bend with explicit edge angles; **not** ``atrbend``).
* Source ``*RadPass`` pass methods are preserved so radiation is re-enablable in
  phase 2; energy is set at the :class:`at.Lattice` level (no per-element Energy).

Divergences from a verbatim import, all intended (PROPOSAL.md FR2/FR3/FR4):

* Instrumentation simplification — ``GIRDER`` / ``SECTIONSTART`` / ``SECTIONEND``
  markers stripped; the length-bearing injection kickers ``DK`` / ``Vkick``
  collapsed to equivalent-length :class:`at.Drift`; the zero-length ``NLK`` marker
  dropped; ``SEPTUM`` kept as a drift; ``SECT1..12`` / ``INJ`` markers and the
  ``CAV`` kept.
* BPMs promoted from bare ``IdentityPass`` markers to :class:`at.Monitor` and
  **named** ``BPM{id:02d}`` (the source's 72 BPMs all share one FamName).
* Synthetic orbit correctors added — one horizontal (HCM) + one vertical (VCM)
  :class:`at.Corrector` co-located with each BPM (thin, zero-kick), named
  ``HCM{id:02d}`` / ``VCM{id:02d}``. The source ships no steerers; a
  control-room ring must be steerable.

Element construction uses the source's per-superperiod ``AR:{sup}:{fam}:{id}``
tokens internally (see :func:`superperiod`), then :func:`build_ring` renames
every scheme element to the flat, family-scoped ``{fam}{id:02d}`` naming
declared by :data:`osprey.simulation.facility_spec.ALS_U_AR` — ids ascend by
s-position within each family, ring-wide (``BEND`` renamed to ``DIPOLE``).

Family names / counts are declared by
:data:`osprey.simulation.facility_spec.ALS_U_AR`; a drift-guard test binds the
built ring to that spec.
"""

from __future__ import annotations

from at import (
    Corrector,
    Dipole,
    Drift,
    Lattice,
    Marker,
    Monitor,
    Quadrupole,
    RFCavity,
    Sextupole,
)

from osprey.simulation.facility_spec import ALS_U_AR, FacilitySpec

__all__ = ["build_ring", "superperiod", "FacilitySpec", "ALS_U_AR"]

# ── Physical constants ────────────────────────────────────────────────────────
_C = 299792458.0  # speed of light [m/s]
_ENERGY_EV = 2.0e9  # GLOBVAL.E0 in the source
_HARMONIC = 304  # HarmNumber in the source

# ── Source pass methods (radiation ON; preserved so phase 2 can re-enable 6D) ──
_QUAD_PASS = "StrMPoleSymplectic4RadPass"
_SEXT_PASS = "StrMPoleSymplectic4RadPass"
_BEND_PASS = "BndMPoleSymplectic4RadPass"

# ── Magnet parameters, transcribed verbatim from ALS_U_AR_v6.m ────────────────
# Quadrupoles: (length, k)
_QF = (0.34400000, 2.13879800)
_QD = (0.18700000, -0.39507430)
_QFA = (0.44800000, 3.60765700)
# Dipole: (length, bending_angle, edge_angle, k); source E1 = E2 = A/2.
_BEND = (0.86514000, 0.17453293, 0.08726646, -1.20965300)
# Sextupoles: (length, h) — h is the source PolynomB(3), ported 1:1.
_SF = (0.20300000, 92.2549)
_SD = (0.20300000, -57.9465)
_SHF = (0.20300000, 21.54638000)
_SHD = (0.20300000, -21.74546000)


def _quad(sup: str, fam: str, ident: int, params: tuple[float, float]) -> Quadrupole:
    length, k = params
    return Quadrupole(f"AR:{sup}:{fam}:{ident}", length, k, PassMethod=_QUAD_PASS)


def _sext(sup: str, fam: str, ident: int, params: tuple[float, float]) -> Sextupole:
    length, h = params
    return Sextupole(f"AR:{sup}:{fam}:{ident}", length, h, PassMethod=_SEXT_PASS)


def _bend(sup: str, ident: int) -> Dipole:
    length, angle, edge, k = _BEND
    return Dipole(
        f"AR:{sup}:BEND:{ident}",
        length,
        bending_angle=angle,
        k=k,
        EntranceAngle=edge,
        ExitAngle=edge,
        PassMethod=_BEND_PASS,
    )


class _Bpm:
    """Per-superperiod BPM factory: assigns ``AR:{sup}:BPM:{id}`` names 1..6."""

    def __init__(self, sup: str) -> None:
        self._sup = sup
        self._n = 0

    def next(self) -> Monitor:
        self._n += 1
        return Monitor(f"AR:{self._sup}:BPM:{self._n}")


def superperiod(sup: str) -> list:
    """Build one ALS-U AR superperiod cell as a list of pyAT elements.

    Replicates the source ``SUP(subName)`` closure (a half-mirror-symmetric cell)
    element-for-element in the original order, with ``GIRDER`` /
    ``SECTIONSTART`` / ``SECTIONEND`` markers stripped and the 6 BPMs promoted to
    named :class:`at.Monitor` s. No correctors / injection / RF here — those are
    added during full-ring assembly (see :func:`build_ring`).

    Args:
        sup: Superperiod token used in element names (e.g. ``"01C"``).

    Returns:
        The ordered element list for one superperiod (fresh element instances).
    """
    b = _Bpm(sup)
    # Half-mirror-symmetric cell, transcribed from ALS_U_AR_v6.m:67-111.
    # (SECSTART/GIRDER/SECEND markers removed; drift lengths inlined.)
    return [
        b.next(),
        Drift("L1end", 0.11960922),
        _sext(sup, "SHF", 1, _SHF),
        Drift("L2a", 0.35548000),
        Drift("L2b", 0.08902000),
        _quad(sup, "QF", 1, _QF),
        Drift("L3", 0.11400000),
        _sext(sup, "SHD", 1, _SHD),
        Drift("L4a", 0.11750000 / 2),
        b.next(),
        Drift("L4a", 0.11750000 / 2),
        _quad(sup, "QD", 1, _QD),
        Drift("L5a", 0.14298200),
        Drift("L5b", 0.10571600),
        _bend(sup, 1),
        Drift("L6", 0.21569930 + 15e-3),
        _sext(sup, "SD", 1, _SD),
        Drift("L7", 0.118784 + 0.105716 - 15e-3),
        _quad(sup, "QFA", 1, _QFA),
        Drift("L8a", 0.12450000 / 2),
        b.next(),
        Drift("L8a", 0.12450000 / 2),
        _sext(sup, "SF", 1, _SF),
        Drift("L9a", 0.31184400),
        Drift("L9b", 0.17885410),
        _bend(sup, 2),
        Drift("L9b", 0.17885410),
        Drift("L9a", 0.31184400),
        _sext(sup, "SF", 2, _SF),
        Drift("L8a", 0.12450000 / 2),
        b.next(),
        Drift("L8a", 0.12450000 / 2),
        _quad(sup, "QFA", 2, _QFA),
        Drift("L7", 0.118784 + 0.105716 - 15e-3),
        _sext(sup, "SD", 2, _SD),
        Drift("L6", 0.21569930 + 15e-3),
        _bend(sup, 3),
        Drift("L5b", 0.10571600),
        Drift("L5a", 0.14298200),
        _quad(sup, "QD", 2, _QD),
        Drift("L4a", 0.11750000 / 2),
        b.next(),
        Drift("L4a", 0.11750000 / 2),
        _sext(sup, "SHD", 2, _SHD),
        Drift("L3", 0.11400000),
        _quad(sup, "QF", 2, _QF),
        Drift("L2b", 0.08902000),
        b.next(),
        Drift("L2a", 0.35548000),
        _sext(sup, "SHF", 2, _SHF),
        Drift("L1end", 0.11960922),
    ]


def _corrector(sup: str, fam: str, ident: str) -> Corrector:
    """A thin, zero-kick orbit corrector named ``AR:{sup}:{fam}:{id}``."""
    return Corrector(f"AR:{sup}:{fam}:{ident}", 0.0, [0.0, 0.0])


def _with_correctors(elements: list) -> list:
    """Insert an HCM + VCM :class:`at.Corrector` right after each Monitor.

    Correctors are co-located with the BPM (zero length → same s-position) and
    inherit the BPM's superperiod + index for their names.
    """
    out: list = []
    for el in elements:
        out.append(el)
        if isinstance(el, Monitor):
            # el.FamName == "AR:{sup}:BPM:{id}"
            _, sup, _, ident = el.FamName.split(":")
            out.append(_corrector(sup, "HCM", ident))
            out.append(_corrector(sup, "VCM", ident))
    return out


def _injection_drift(name: str, length: float) -> Drift:
    return Drift(name, length)


# Source token -> flat naming-contract family token (see facility_spec.py).
_FAM_RENAME = {"BEND": "DIPOLE"}


def _flatten_names(elements: list) -> None:
    """Rename every ``AR:{sup}:{fam}:{id}`` element to the flat naming scheme, in place.

    Walks ``elements`` in order (= ascending s-position) and assigns a
    per-family, ring-wide counter — ``BEND`` mapped to ``DIPOLE`` — so ids are
    zero-padded and ascend by s-position within each family, matching
    :data:`osprey.simulation.facility_spec.ALS_U_AR`. Non-scheme elements
    (Drift, Marker, RFCavity) are left untouched.
    """
    counters: dict[str, int] = {}
    for el in elements:
        name = el.FamName
        if not name.startswith("AR:"):
            continue
        _, _, fam, _ = name.split(":")
        fam = _FAM_RENAME.get(fam, fam)
        counters[fam] = counters.get(fam, 0) + 1
        el.FamName = ALS_U_AR.device_name("", fam, counters[fam])


def build_ring() -> Lattice:
    """Build the full ALS-U AR ring as an :class:`at.Lattice`.

    Concatenates the 12 superperiods with the injection-region elements from the
    source ``ELIST`` (``ALS_U_AR_v6.m:114-125``), collapsing the length-bearing
    injection kickers to drifts and dropping the zero-length ``NLK``; inserts the
    synthetic HCM/VCM correctors one-per-BPM; adds the RF cavity with a frequency
    derived from the summed circumference; and wraps everything at 2.0 GeV with
    ``periodicity=1`` (the injection region + single cavity break the 12-fold
    symmetry).

    Returns:
        The assembled ring (radiation-on magnets + cavity retained).
    """
    cav = RFCavity("CAV", 0.0, 1.0e6, 1.0, _HARMONIC, _ENERGY_EV)  # frequency set below

    elements: list = []

    # Section 1: SECT1 L1(2.1) DK(0.3) L1(0.21960922) SEPTUM(1.8) INJ Lround(0.38039078)
    elements += [
        Marker("SECT1"),
        _injection_drift("L1", 2.1),
        _injection_drift("DK", 0.3),
        _injection_drift("L1", 0.21960922),
        _injection_drift("SEPTUM", 1.8),
        Marker("INJ"),
        _injection_drift("Lround", 0.38039078),
    ]
    elements += superperiod("01C")

    # Section 2: SECT2 L1(3.81960922) DK(0.3) L1(0.1) [NLK dropped] Vkick(0.58039078)
    elements += [
        Marker("SECT2"),
        _injection_drift("L1", 3.81960922),
        _injection_drift("DK", 0.3),
        _injection_drift("L1", 0.1),
        _injection_drift("Vkick", 0.58039078),
    ]
    elements += superperiod("02C")

    # Sections 3-6: SECT(i) L1(2.4) L1(2.4) SUP
    for i in range(3, 7):
        elements += [
            Marker(f"SECT{i}"),
            _injection_drift("L1", 2.4),
            _injection_drift("L1", 2.4),
        ]
        elements += superperiod(f"{i:02d}C")

    # Section 7: SECT7 L1(4.11960922) DK(0.3) Lround(0.38039078)
    elements += [
        Marker("SECT7"),
        _injection_drift("L1", 4.11960922),
        _injection_drift("DK", 0.3),
        _injection_drift("Lround", 0.38039078),
    ]
    elements += superperiod("07C")

    # Sections 8-10: SECT(i) L1(2.4) L1(2.4) SUP
    for i in range(8, 11):
        elements += [
            Marker(f"SECT{i}"),
            _injection_drift("L1", 2.4),
            _injection_drift("L1", 2.4),
        ]
        elements += superperiod(f"{i:02d}C")

    # Section 11: SECT11 L1(2.4) CAV L1(2.4)
    elements += [
        Marker("SECT11"),
        _injection_drift("L1", 2.4),
        cav,
        _injection_drift("L1", 2.4),
    ]
    elements += superperiod("11C")

    # Section 12: SECT12 Vkick(0.58039078) L1(0.1) DK(0.3) L1(3.81960922)
    elements += [
        Marker("SECT12"),
        _injection_drift("Vkick", 0.58039078),
        _injection_drift("L1", 0.1),
        _injection_drift("DK", 0.3),
        _injection_drift("L1", 3.81960922),
    ]
    elements += superperiod("12C")

    # Insert synthetic orbit correctors co-located with every BPM.
    elements = _with_correctors(elements)

    # Rename every scheme element from the per-superperiod source token to the
    # flat, family-scoped naming contract declared by facility_spec.ALS_U_AR.
    _flatten_names(elements)

    # RF frequency from the *ported* summed circumference (RF/circumference
    # self-consistent for later 6D optics). Correctors/markers are zero-length.
    circumference = sum(float(getattr(el, "Length", 0.0)) for el in elements)
    cav.Frequency = _HARMONIC * _C / circumference

    return Lattice(elements, energy=_ENERGY_EV, periodicity=1)
