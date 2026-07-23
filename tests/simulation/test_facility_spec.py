"""Tests for :mod:`osprey.simulation.facility_spec` and its ring consumer.

Two concerns are covered:

* **Accessor tests** — the declared families / counts / kinds / naming of the
  frozen :data:`ALS_U_AR` spec (task 1.1 gate).
* **Ring↔spec consistency** — a drift-guard binding the hand-ported
  :func:`osprey.simulation.lattice.build_ring` to the spec's declared per-family
  counts and BPM naming (task 3.2 gate, selected by ``-k consistency``).

Deterministic and offline: imports only ``at`` + ``numpy`` via the shared
lattice subpackage; no EPICS / softioc / MATLAB / network.
"""

from __future__ import annotations

import dataclasses
import re
from collections import Counter

import at
import pytest

from osprey.simulation.facility_spec import ALS_U_AR
from osprey.simulation.lattice import build_ring

# Expected per-family counts, mirrored verbatim from the spec declaration.
EXPECTED_COUNTS = {
    "QF": 24,
    "QD": 24,
    "QFA": 24,
    "DIPOLE": 36,
    "SF": 24,
    "SD": 24,
    "SHF": 24,
    "SHD": 24,
    "BPM": 72,
    "HCM": 72,
    "VCM": 72,
}

# Matches the flat ``{fam}{id:02d}`` device-naming scheme.
_NAME_RE = re.compile(r"^(HCM|VCM|QF|QD|QFA|DIPOLE|SF|SD|SHF|SHD|BPM)(\d{2,})$")


# ── Accessor tests (task 1.1: pytest tests/simulation/test_facility_spec.py) ──


def test_counts():
    assert ALS_U_AR.counts() == EXPECTED_COUNTS


def test_family_kinds():
    assert ALS_U_AR.family("BPM").kind == "monitor"
    assert ALS_U_AR.family("HCM").kind == "corrector"
    assert ALS_U_AR.family("QF").kind == "magnet"


def test_device_name():
    assert ALS_U_AR.device_name("01C", "BPM", 3) == "BPM03"


def test_machine_constants():
    assert ALS_U_AR.name == "ALS-U-AR"
    assert ALS_U_AR.energy_ev == 2.0e9
    assert ALS_U_AR.harmonic == 304
    assert ALS_U_AR.naming == "{fam}{id:02d}"


def test_family_missing_raises_keyerror():
    with pytest.raises(KeyError):
        ALS_U_AR.family("NOPE")


def test_spec_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ALS_U_AR.name = "mutated"  # type: ignore[misc]


# ── Ring↔spec consistency (task 3.2: pytest ... -k consistency) ──────────────


def test_ring_spec_consistency():
    """The built ring's scheme-named elements tally to the declared counts,
    and every ``at.Monitor`` is a properly named BPM."""
    ring = build_ring()

    tally: Counter[str] = Counter()
    monitors = 0
    for element in ring:
        match = _NAME_RE.match(getattr(element, "FamName", ""))
        if match is not None:
            tally[match.group(1)] += 1
        if isinstance(element, at.Monitor):
            monitors += 1
            assert match is not None, f"Monitor {element.FamName!r} is not scheme-named"
            assert match.group(1) == "BPM", f"Monitor {element.FamName!r} is not a BPM"

    assert dict(tally) == ALS_U_AR.counts()
    assert monitors == 72
