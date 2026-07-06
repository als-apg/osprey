"""Builds the generic SR PyAT lattice matching manifest partition (a) exactly.

Device inventory (family -> device-id list) is derived from the namespace-union
manifest's pyat-coupled partition via :mod:`inventory`, never hardcoded: this
lattice's element counts move automatically if the DB namespace changes.

Layout: one arc cell per SR:MAG:DIPOLE device (24 cells total). Each cell is a
short FODO-like doublet: [QF?] [SF?] dipole [SD?] [QD?] [HCM?] [VCM?] [BPM?],
where a bracketed element is only present in the leading N cells for its family
(N = that family's device count). So e.g. the 20 BPM/HCM/VCM devices occupy
cells 1-20 and the 12 SF/SD devices occupy cells 1-12. Sextupole strengths are
zero (no chromaticity model yet -- SF/SD exist to match the manifest's device
inventory; a later task can drive their strength from the SF/SD CURRENT
channels without changing this layout).

Each cell places its HCM and VCM correctors immediately upstream of its own BPM
in the same straight section, so "the BPM co-located with corrector HCM<nn>" is
always BPM<nn> -- an unambiguous, name-derived pairing for orbit_response's
"downstream BPM" contract.

The quadrupole gradients (QF_K, QD_K below) were chosen empirically (grid
search over both signs) to give a stable one-turn transfer matrix in both
transverse planes with a comfortable margin from the |trace| = 2 stability
boundary -- this is a toy demo lattice, not a physical machine design.
"""

from __future__ import annotations

import at
import numpy as np

from . import inventory

RING_ENERGY_EV = 1.9e9  # matches the DB's "1.9 GeV" storage-ring description
N_ARC_CELLS = 24  # one per SR:MAG:DIPOLE device -- fixes the ring's total bend

# Empirically-verified stable operating point (see module docstring). Signs
# match the DB's own family descriptions: QF has positive (horizontally
# focusing) gradient, QD negative (horizontally defocusing).
QF_K = 0.25
QD_K = -0.30

_CELL_LENGTH_DRIFT = 0.3
_QUAD_LENGTH = 0.25
_SEXTUPOLE_LENGTH = 0.15
_DIPOLE_LENGTH = 1.0
_CORRECTOR_LENGTH = 0.1
_BPM_APPROACH_DRIFT = 0.05
_CELL_CLOSING_DRIFT = 0.2


def _require(inv: dict[str, list[str]], family: str) -> list[str]:
    ids = inv.get(family, [])
    if not ids:
        raise ValueError(
            f"manifest pyat-coupled partition has no '{family}' devices -- "
            "cannot build the SR lattice"
        )
    return ids


def build_ring() -> at.Lattice:
    """Build the SR lattice.

    Element FamNames are "<FAMILY><device>" (e.g. "HCM01", "BPM20", "DIPOLE24"),
    matching the manifest's zero-padded device ids directly.

    Returns:
        An `at.Lattice` with a stable closed orbit at nominal (all-zero
        corrector) settings.

    Raises:
        ValueError: if the manifest's pyat-coupled partition is missing a
            required family, or its SR:MAG:DIPOLE count doesn't match
            N_ARC_CELLS (the DB fixes the lattice, not vice versa -- a mismatch
            here means this module's layout assumption is stale).
    """
    inv = inventory.pyat_coupled_device_ids()
    dipole_ids = _require(inv, "DIPOLE")
    qf_ids = _require(inv, "QF")
    qd_ids = _require(inv, "QD")
    sf_ids = _require(inv, "SF")
    sd_ids = _require(inv, "SD")
    hcm_ids = _require(inv, "HCM")
    vcm_ids = _require(inv, "VCM")
    bpm_ids = _require(inv, "BPM")

    if len(dipole_ids) != N_ARC_CELLS:
        raise ValueError(
            f"expected {N_ARC_CELLS} SR:MAG:DIPOLE devices (one per arc cell), "
            f"manifest has {len(dipole_ids)}; update N_ARC_CELLS or the cell layout"
        )

    bend_angle = 2 * np.pi / N_ARC_CELLS

    elements: list[at.elements.Element] = []
    for i in range(N_ARC_CELLS):
        elements.append(at.Drift(f"DR{i:02d}A", _CELL_LENGTH_DRIFT))

        if i < len(qf_ids):
            elements.append(at.Quadrupole(f"QF{qf_ids[i]}", _QUAD_LENGTH, k=QF_K))
        if i < len(sf_ids):
            elements.append(at.Sextupole(f"SF{sf_ids[i]}", _SEXTUPOLE_LENGTH, h=0.0))

        elements.append(at.Drift(f"DR{i:02d}B", _CELL_LENGTH_DRIFT))
        elements.append(
            at.Dipole(f"DIPOLE{dipole_ids[i]}", _DIPOLE_LENGTH, bending_angle=bend_angle)
        )
        elements.append(at.Drift(f"DR{i:02d}C", _CELL_LENGTH_DRIFT))

        if i < len(sd_ids):
            elements.append(at.Sextupole(f"SD{sd_ids[i]}", _SEXTUPOLE_LENGTH, h=0.0))
        if i < len(qd_ids):
            elements.append(at.Quadrupole(f"QD{qd_ids[i]}", _QUAD_LENGTH, k=QD_K))

        elements.append(at.Drift(f"DR{i:02d}D", _CELL_LENGTH_DRIFT))

        if i < len(hcm_ids):
            elements.append(
                at.Corrector(f"HCM{hcm_ids[i]}", _CORRECTOR_LENGTH, kick_angle=[0.0, 0.0])
            )
        if i < len(vcm_ids):
            elements.append(
                at.Corrector(f"VCM{vcm_ids[i]}", _CORRECTOR_LENGTH, kick_angle=[0.0, 0.0])
            )
        if i < len(bpm_ids):
            elements.append(at.Drift(f"DR{i:02d}E", _BPM_APPROACH_DRIFT))
            elements.append(at.Monitor(f"BPM{bpm_ids[i]}"))

        elements.append(at.Drift(f"DR{i:02d}F", _CELL_CLOSING_DRIFT))

    return at.Lattice(elements, energy=RING_ENERGY_EV, periodicity=1, name="SR-generic")
