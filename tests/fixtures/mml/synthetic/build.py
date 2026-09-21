"""Build the synthetic 2.0 MML export fixture with pyAT and plain Python.

The committed files beside this script are its output. They are what
``mml_export.m`` 2.0 writes for one sub-machine -- a saved ``THERING``, the
Accelerator Objects and Data, the per-family calibration document and the orbit
response matrix -- for an invented machine small enough to read by eye:

    quokka.sr.lattice.mat    the ring, saved the way ``at.save_mat`` saves it
    quokka.sr.ao.json        the Accelerator Objects
    quokka.sr.ad.json        the Accelerator Data
    quokka.sr.va.json        per-family calibration and nominals
    quokka.sr.response.json  the orbit response matrix
    mismatched.lattice.mat   the same ring with one magnet renamed

Nothing here runs MATLAB. The export pipeline of ``mml_export.m`` is ported
into this file function for function -- the hardware grid and its anchor, the
line-or-table test, the rigidity probe, the energy table, the nominal read and
the refusal joining -- and the facility's own conversions, which a real export
calls out to, are the small Python functions in ``CONVERSIONS`` below. The ring
is the one every number is sampled against: the nominal settings drive the
element strengths, and the nominals the export records are read back off them,
so the document and the lattice describe one machine.

Run from the repository root to rebuild the fixture, or to check that the
committed files are still exactly what this script writes::

    uv run python tests/fixtures/mml/synthetic/build.py
    uv run python tests/fixtures/mml/synthetic/build.py --check

``--check`` is the fixture's determinism gate: it rebuilds into a temporary
directory and compares every byte. Everything a clock or a machine would
otherwise decide is pinned -- the ``_export`` timestamp, the MAT-file's header
text -- so a rebuild on another day on another machine writes the same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import at
import numpy as np

# ---------------------------------------------------------------------------
# What the export says about itself.
# ---------------------------------------------------------------------------

#: The exporter version this fixture is written against.
EXPORTER = "mml_export 2.0.0"

#: The MATLAB a real export would name. Pinned, like the timestamp: this
#: fixture is written by Python, and a version read off this machine would make
#: the committed files depend on where they were built.
MATLAB = "25.1.0.2943329 (R2025a)"

#: The ``_export`` timestamp, in the spelling ``datestr(now, 'yyyy-mm-ddTHH:MM:SS')``
#: gives. Pinned so a rebuild is byte-identical.
TIMESTAMP = "2026-09-17T09:00:00"

MACHINE = "Quokka"
SUBMACHINE = "SR"
STEM = "quokka.sr"

# ---------------------------------------------------------------------------
# The ring.
# ---------------------------------------------------------------------------

#: Metres per second.
C_LIGHT = 299792458.0

#: The electron rest energy in GeV, the mass ``getbrho`` carries.
ELECTRON_GEV = 0.51099895069e-3

#: The deck energy every sample in the document is taken at.
DECK_ENERGY_GEV = 2.0

#: Four cells, one dipole each, so the bends close the ring.
CELLS = 4

#: The geometry of one cell, in metres, and the focusing it is built with.
DIPOLE_LENGTH = 3.0
QUAD_LENGTH = 0.3
SEXT_LENGTH = 0.2
CELL_DRIFT = 0.75
TAIL_DRIFT = 0.5
QUAD_K = 1.2
SEXT_H = 2.5
SKEW_KS = 0.02

#: The cavity's harmonic number.
HARMONIC = 40

#: The bending angle of one dipole.
DIPOLE_ANGLE = 2.0 * math.pi / CELLS


def brho(energy_gev: float) -> float:
    """The beam rigidity in T*m, rest mass included, the way ``getbrho`` gives it."""
    momentum = math.sqrt(energy_gev**2 - ELECTRON_GEV**2)
    return momentum * 1.0e9 / C_LIGHT


def energy_of_brho(rigidity: float) -> float:
    """The energy in GeV a rigidity belongs to -- the inverse of :func:`brho`."""
    momentum = rigidity * C_LIGHT / 1.0e9
    return math.sqrt(momentum**2 + ELECTRON_GEV**2)


#: The dipole field the deck sits at, in T: the field that bends
#: :data:`DIPOLE_ANGLE` over :data:`DIPOLE_LENGTH` at the deck energy.
DIPOLE_FIELD = DIPOLE_ANGLE * brho(DECK_ENERGY_GEV) / DIPOLE_LENGTH

#: The invented facility's bend supply: a measured excitation ramp that
#: saturates, and that stops at 500 A because that is where the measurement
#: stopped. Above it the conversion answers with no number, which is what puts
#: a ``"NaN"`` tail and a ``finite_span`` into the committed table.
RAMP_CURRENT_LIMIT = 500.0
RAMP_TAU = 600.0
BEND_NOMINAL_AMPS = 420.0
RAMP_GAIN = DIPOLE_FIELD / (1.0 - math.exp(-BEND_NOMINAL_AMPS / RAMP_TAU))


def _ring_elements() -> list[Any]:
    """The lattice elements of one turn, in ring order."""
    elements: list[Any] = []
    for cell in range(1, CELLS + 1):
        elements += [
            at.Quadrupole(f"QF{cell}", QUAD_LENGTH, QUAD_K),
            at.Drift("DR", CELL_DRIFT),
            at.Monitor(f"BPM{cell}"),
            at.Dipole(f"BD{cell}", DIPOLE_LENGTH, DIPOLE_ANGLE),
            at.Sextupole(f"SF{cell}", SEXT_LENGTH, SEXT_H),
            at.Quadrupole(f"QD{cell}", QUAD_LENGTH, -QUAD_K),
            at.Drift("DR", CELL_DRIFT),
            at.Corrector(f"HC{cell}A", 0.0, [0.0, 0.0]),
        ]
        # The fourth cell carries one corrector where the others carry two, so
        # the corrector family's index list is ragged and one of its rows is
        # padded. That padding is the fixture's one "NaN" slice.
        if cell != CELLS:
            elements.append(at.Corrector(f"HC{cell}B", 0.0, [0.0, 0.0]))
        elements.append(at.Drift("DR", TAIL_DRIFT))
    return elements


def build_ring() -> at.Lattice:
    """The ring the document describes, in its nominal state.

    Returned 4D, the way a ring saved out of a facility's simulator model
    arrives: the cavity is an element of the lattice and longitudinal motion is
    off. What each step of the export asks of it is what that step needs. A
    nominal is read off this ring as it stands, which is the 4D closed orbit;
    the response matrix is measured about the 6D one, which is the orbit the
    served model runs on. The two differ at the monitors by about 4e-5 m in x,
    and a real export can hold the same split: every nominal is read before any
    matrix is measured, and the model need not be in the same state for both.
    """
    ring = at.Lattice(
        _ring_elements(), name="quokka_sr", energy=DECK_ENERGY_GEV * 1.0e9, periodicity=1
    )
    frequency = HARMONIC * C_LIGHT / ring.circumference
    ring.append(at.RFCavity("RFC", 0.0, 1.0e6, frequency, HARMONIC, ring.energy))
    ring.disable_6d()
    _apply_nominal_skew(ring)
    _apply_nominal_kicks(ring)
    return ring


def _apply_nominal_skew(ring: at.Lattice) -> None:
    """Put the skew-quadrupole family's nominal onto the sextupole elements."""
    for index in _named(ring, "SF"):
        polynom_a = np.array(ring[index].PolynomA, dtype=float)
        polynom_a[1] = SKEW_KS
        ring[index].PolynomA = polynom_a


def _apply_nominal_kicks(ring: at.Lattice) -> None:
    """Put the corrector families' nominal settings onto the corrector elements.

    A kick is additive along a split magnet, so a device's setting is shared
    out over the elements it is made of -- the same rule the Middle Layer's own
    write path applies, and the rule the emitted slice weights come from.
    """
    for plane, family in ((0, "HC"), (1, "VC")):
        gain = CONVERSIONS[family]["gain"]
        for device, current in enumerate(NOMINAL_AMPS[family]):
            indices = _corrector_elements(ring, device)
            kick = gain * current / len(indices)
            for index in indices:
                angle = np.array(ring[index].KickAngle, dtype=float)
                angle[plane] = kick
                ring[index].KickAngle = angle


def _named(ring: at.Lattice, prefix: str) -> list[int]:
    """The 0-based ring positions of every element whose name starts with ``prefix``."""
    return [index for index, element in enumerate(ring) if element.FamName.startswith(prefix)]


def _corrector_elements(ring: at.Lattice, device: int) -> list[int]:
    """The 0-based ring positions of the elements one corrector device is made of."""
    cell = device + 1
    return [
        index
        for index, element in enumerate(ring)
        if element.FamName in (f"HC{cell}A", f"HC{cell}B")
    ]


# ---------------------------------------------------------------------------
# The MATLAB file.
# ---------------------------------------------------------------------------

#: The 116-byte descriptive header ``scipy.io.savemat`` writes carries the
#: build clock, so it is overwritten with this. Nothing reads it -- the format
#: version and the endian marker live in the 12 bytes after it -- and pinning
#: it is what lets ``--check`` compare the lattice byte for byte like every
#: other committed file.
MAT_HEADER = b"MATLAB 5.0 MAT-file, written by tests/fixtures/mml/synthetic/build.py"


def save_lattice(ring: at.Lattice, path: Path) -> None:
    """Save ``ring`` as ``THERING``, with a header that does not carry a clock."""
    at.save_mat(ring, str(path), use="THERING")
    raw = bytearray(path.read_bytes())
    raw[0:116] = MAT_HEADER.ljust(116, b"\x00")
    path.write_bytes(bytes(raw))


def saved_ring(ring: at.Lattice) -> list[Any]:
    """``THERING`` as the saved file holds it: the ring behind its parameter element.

    ``at.save_mat`` writes the lattice's own properties as a leading
    ``RingParam`` element, so the file holds one element more than the lattice
    and every index the Accelerator Objects carry is one further along.
    """
    return [_RingParam(ring)] + list(ring)


class _RingParam:
    """The parameter element ``at.save_mat`` puts at the head of ``THERING``."""

    def __init__(self, ring: at.Lattice) -> None:
        self.FamName = ring.name
        self.Class = "RingParam"


def at_index(ring: at.Lattice, positions: list[int]) -> list[int]:
    """The 1-based ``THERING`` indices of the given 0-based ring positions."""
    return [position + 2 for position in positions]


# ---------------------------------------------------------------------------
# The invented facility's own conversions.
# ---------------------------------------------------------------------------
#
# A real export never reads a stored conversion parameter: it runs the
# facility's hw2physics, physics2hw, bend2gev and gev2bend and records what
# they answered. These are this facility's, and each one is chosen for the
# shape it puts into the committed document.


def _linear_brho(gain: float) -> Callable[[np.ndarray, float], np.ndarray]:
    """A conversion that divides by the rigidity, so the family follows the energy knob."""

    def convert(hardware: np.ndarray, energy: float) -> np.ndarray:
        return gain * hardware * brho(DECK_ENERGY_GEV) / brho(energy)

    return convert


def _linear_flat(gain: float) -> Callable[[np.ndarray, float], np.ndarray]:
    """A conversion that ignores the energy it is handed, the gain-and-offset branch."""

    def convert(hardware: np.ndarray, energy: float) -> np.ndarray:
        return gain * hardware

    return convert


def _bend_hw2physics(hardware: np.ndarray, energy: float) -> np.ndarray:
    """The bend supply's measured ramp, in radians of bend, ending where it ends."""
    field = np.where(
        hardware <= RAMP_CURRENT_LIMIT,
        RAMP_GAIN * (1.0 - np.exp(-np.abs(hardware) / RAMP_TAU)),
        np.nan,
    )
    return field * DIPOLE_LENGTH / brho(energy)


def _bend_physics2hw(physics: np.ndarray, energy: float) -> np.ndarray:
    """The ramp read the other way: the current that bends this angle."""
    field = np.asarray(physics, dtype=float) * brho(energy) / DIPOLE_LENGTH
    ratio = np.clip(1.0 - field / RAMP_GAIN, 1.0e-12, None)
    current = -RAMP_TAU * np.log(ratio)
    return np.where(current <= RAMP_CURRENT_LIMIT, current, np.nan)


def _bend2gev(hardware: np.ndarray) -> np.ndarray:
    """What the bend current says the ring's energy is, in GeV."""
    values = np.atleast_1d(np.asarray(hardware, dtype=float))
    out = np.full(values.shape, np.nan)
    for index, current in enumerate(values.flat):
        if current > RAMP_CURRENT_LIMIT:
            continue
        field = RAMP_GAIN * (1.0 - math.exp(-abs(current) / RAMP_TAU))
        out.flat[index] = energy_of_brho(field * DIPOLE_LENGTH / DIPOLE_ANGLE)
    return out


def _gev2bend(energy: float) -> float:
    """The current the ring's energy sits at, the facility's own inverse."""
    field = brho(energy) * DIPOLE_ANGLE / DIPOLE_LENGTH
    return _round_significant(-RAMP_TAU * math.log(1.0 - field / RAMP_GAIN), 12)


def _quad_monitor_inverse(physics: np.ndarray, energy: float) -> np.ndarray:
    """The quadrupole readback's own way back to amps.

    Deliberately not the inverse of the calibration beside it: a facility's two
    conversion parameter sets are independent data, and a consumer that derives
    one from the other rather than reading the sampled inverse gets this family
    wrong by five per cent.
    """
    return 95.0 * np.asarray(physics, dtype=float)


def _qd_monitor_inverse(physics: np.ndarray, energy: float) -> np.ndarray:
    """A readback conversion with a curve in it, so the inverse is written as a table."""
    strength = np.asarray(physics, dtype=float)
    return -82.0 * strength + 3.0e3 * strength**3


def _identity_inverse(gain: float) -> Callable[[np.ndarray, float], np.ndarray]:
    """The exact inverse of a flat linear conversion."""

    def convert(physics: np.ndarray, energy: float) -> np.ndarray:
        return np.asarray(physics, dtype=float) / gain

    return convert


def _brho_inverse(gain: float) -> Callable[[np.ndarray, float], np.ndarray]:
    """The exact inverse of a rigidity-scaled conversion."""

    def convert(physics: np.ndarray, energy: float) -> np.ndarray:
        return np.asarray(physics, dtype=float) * brho(energy) / (gain * brho(DECK_ENERGY_GEV))

    return convert


#: What each family is set to, per device, in hardware units. This is the seam
#: the ring is built from and the answer the model read gives back, so the
#: lattice and the document agree by construction.
NOMINAL_AMPS: dict[str, list[float]] = {
    "QF": [120.0, 120.0, 120.0, 120.0],
    "QD": [100.0, 100.0, 100.0, 100.0],
    "SF": [50.0, 50.0, 50.0, 50.0],
    "SQ": [5.0, 5.0, 5.0, 5.0],
    "HC": [1.5, -0.8, 0.4, 0.0],
    "VC": [0.2, -0.1, 0.0, 0.3],
    "BEND": [BEND_NOMINAL_AMPS] * 4,
    "BDM": [0.0, 0.0],
    "BSOFT": [60.0, 60.0],
    "IDGAP": [float("nan"), float("nan")],
    "SEPTUM": [48.0],
    "DCCT": [300.0],
}

#: The per-family conversions, as the export calls them: one hardware-to-physics
#: function, one back, and the gain the ring is built with where there is one.
CONVERSIONS: dict[str, dict[str, Any]] = {
    "QF": {
        "gain": 0.01,
        "hw2physics": _linear_brho(0.01),
        "physics2hw": _quad_monitor_inverse,
        "fcn": "amp2k",
        "inverse_fcn": "k2amp",
    },
    "QD": {
        "gain": -0.012,
        "hw2physics": _linear_brho(-0.012),
        "physics2hw": _qd_monitor_inverse,
        "fcn": "amp2k",
        "inverse_fcn": "k2amp_meas",
    },
    "SF": {
        "gain": 0.05,
        "hw2physics": _linear_brho(0.05),
        "physics2hw": _brho_inverse(0.05),
        "fcn": "amp2k2",
        "inverse_fcn": "k22amp",
    },
    "SQ": {
        "gain": 0.004,
        "hw2physics": _linear_brho(0.004),
        "physics2hw": _brho_inverse(0.004),
        "fcn": "amp2ks",
        "inverse_fcn": "ks2amp",
    },
    "HC": {
        "gain": 1.0e-4,
        "hw2physics": _linear_brho(1.0e-4),
        "physics2hw": _brho_inverse(1.0e-4),
        "fcn": "amp2rad",
        "inverse_fcn": "rad2amp",
    },
    "VC": {
        "gain": 9.0e-5,
        "hw2physics": _linear_brho(9.0e-5),
        "physics2hw": _brho_inverse(9.0e-5),
        "fcn": "amp2rad",
        "inverse_fcn": "rad2amp",
    },
    "BPMx": {
        "gain": 1.0e-3,
        "hw2physics": _linear_flat(1.0e-3),
        "physics2hw": _identity_inverse(1.0e-3),
        "fcn": "mm2m",
        "inverse_fcn": "m2mm",
    },
    "BPMy": {
        "gain": 1.0e-3,
        "hw2physics": _linear_flat(1.0e-3),
        "physics2hw": _identity_inverse(1.0e-3),
        "fcn": "mm2m",
        "inverse_fcn": "m2mm",
    },
    "BEND": {
        "hw2physics": _bend_hw2physics,
        "physics2hw": _bend_physics2hw,
        "fcn": "bend2rad",
        "inverse_fcn": "rad2bend",
    },
    "BDM": {
        "gain": 2.0e-6,
        "hw2physics": _linear_brho(2.0e-6),
        "physics2hw": _brho_inverse(2.0e-6),
        "fcn": "amp2rad",
        "inverse_fcn": "rad2amp",
    },
    "BSOFT": {
        "gain": 3.0e-4,
        "hw2physics": _linear_brho(3.0e-4),
        "physics2hw": _brho_inverse(3.0e-4),
        "fcn": "amp2rad",
        "inverse_fcn": "rad2amp",
    },
    "RF": {
        "gain": 1.0e6,
        "hw2physics": _linear_flat(1.0e6),
        "physics2hw": _identity_inverse(1.0e6),
        "fcn": "mhz2hz",
        "inverse_fcn": "hz2mhz",
    },
    "IDGAP": {
        "gain": 1.0e-3,
        "hw2physics": _linear_flat(1.0e-3),
        "physics2hw": _identity_inverse(1.0e-3),
        "fcn": "mm2m",
        "inverse_fcn": "m2mm",
    },
    "SEPTUM": {
        "gain": 1.0e3,
        "hw2physics": _linear_flat(1.0e3),
        "physics2hw": _identity_inverse(1.0e3),
        "fcn": "kv2v",
        "inverse_fcn": "v2kv",
    },
    "DCCT": {
        "gain": 1.0e-3,
        "hw2physics": _linear_flat(1.0e-3),
        "physics2hw": _identity_inverse(1.0e-3),
        "fcn": "ma2a",
        "inverse_fcn": "a2ma",
    },
    "TUNE": {
        "gain": 1.0,
        "hw2physics": _linear_flat(1.0),
        "physics2hw": _identity_inverse(1.0),
        "fcn": "identity",
        "inverse_fcn": "identity",
    },
}

#: What the Middle Layer corrects each beam monitor's own reading by, one value
#: per device. ``Gain`` and ``Offset`` are the pair the hardware-to-physics
#: conversion already applies before its own, the offset in the monitor's
#: hardware units; ``Roll`` is the angle in radians and ``Crunch`` the shear
#: that carry the two planes a monitor reads in into the model's, and neither
#: is in any conversion. Both families of a pair state the rotation, and it is
#: one rotation of the pair rather than one each.
#:
#: The two families are written differently on purpose, because both shapes are
#: what a facility writes. The horizontal one states four numbers per key, one
#: per device. The vertical one states a single number for ``Roll`` and
#: ``Crunch``, which the Middle Layer hands back for every device in the list
#: and the export therefore writes out per device; and it states no ``Offset``
#: at all, which stays absent rather than becoming a zero.
READOUT: dict[str, dict[str, Any]] = {
    "BPMx": {
        "Gain": np.array([1.02, 0.98, 1.01, 0.995]),
        "Offset": np.array([0.12, -0.05, 0.31, 0.0]),
        "Roll": np.array([1.0e-3, -2.0e-3, 0.0, 5.0e-4]),
        "Crunch": np.array([2.0e-3, 0.0, -1.0e-3, 4.0e-3]),
    },
    "BPMy": {
        "Gain": np.array([0.99, 1.03, 1.0, 0.97]),
        "Roll": 7.5e-4,
        "Crunch": -1.5e-3,
    },
}

#: Which families the ring's energy may be read from, and what their own
#: current-to-energy conversion answers. ``BEND`` rides its measured ramp;
#: ``BSOFT`` hands back the deck energy at every current, which is a fact about
#: this facility rather than a ring with no energy knob.
ENERGY_CONVERSIONS: dict[str, dict[str, Any]] = {
    "BEND": {"bend2gev": _bend2gev, "gev2bend": _gev2bend},
    "BSOFT": {
        "bend2gev": lambda hardware: np.full(np.atleast_1d(hardware).shape, DECK_ENERGY_GEV),
        "gev2bend": lambda energy: 60.0,
    },
}


# ---------------------------------------------------------------------------
# MATLAB values and the spelling jsonencode gives them.
# ---------------------------------------------------------------------------


class Fn:
    """A MATLAB function handle, as the export writes one."""

    def __init__(self, name: str, file: str = "") -> None:
        self.name = name
        self.file = file


def _number(value: float) -> str:
    """One finite double, at the fifteen significant digits ``jsonencode`` writes."""
    if value == 0:
        return "0"
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{float(value):.15g}"


def _nonfinite(value: float) -> str:
    """The word the export spells a value that is not a number with."""
    if math.isnan(value):
        return "NaN"
    return "Inf" if value > 0 else "-Inf"


def _entry(value: float) -> str:
    """One entry of a numeric row: a number, or the string a non-finite one becomes."""
    if math.isfinite(value):
        return _number(value)
    return json.dumps(_nonfinite(value))


def _encode_numeric(value: Any) -> str:
    """One numeric value, keeping 1-row and N-row shape the way jsonencode does."""
    array = np.atleast_2d(np.asarray(value, dtype=float))
    if array.size == 0:
        return "[]"
    if array.size == 1:
        return _entry(float(array.flat[0]))
    if array.shape[0] == 1 or array.shape[1] == 1:
        return "[" + ",".join(_entry(float(item)) for item in array.flatten()) + "]"
    return (
        "["
        + ",".join("[" + ",".join(_entry(float(item)) for item in row) + "]" for row in array)
        + "]"
    )


def encode(value: Any) -> str:
    """One MATLAB value in the spelling the export writes it in.

    ``Handles`` is dropped, function handles become their two-key record, char
    matrices become one deblanked string per row, and matrix shape is kept: a
    1-row value is a flat array, an N-row value an array of rows.
    """
    if isinstance(value, Fn):
        return f'{{"$fn":{json.dumps(value.name)},"file":{json.dumps(value.file)}}}'
    if isinstance(value, dict):
        items = [(name, item) for name, item in value.items() if name != "Handles"]
        return "{" + ",".join(f"{json.dumps(name)}:{encode(item)}" for name, item in items) + "}"
    if isinstance(value, str):
        return json.dumps(value.rstrip())
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, list):
        return "[" + ",".join(encode(item) for item in value) + "]"
    return _encode_numeric(value)


def document(body: dict[str, Any]) -> str:
    """One JSON document: the ``_export`` block first, then the body's own keys."""
    export = {
        "exporter": EXPORTER,
        "matlab": MATLAB,
        "machine": MACHINE,
        "submachine": SUBMACHINE,
        "timestamp": TIMESTAMP,
    }
    head = '{"_export":' + encode(export)
    encoded = encode(body)
    return head + "}" if encoded == "{}" else head + "," + encoded[1:]


# ---------------------------------------------------------------------------
# The Accelerator Objects and Data.
# ---------------------------------------------------------------------------


def _devices(count: int) -> np.ndarray:
    """A device list of ``count`` devices, one per cell, in the export's N-by-2 shape."""
    return np.array([[cell, 1] for cell in range(1, count + 1)], dtype=float)


def _field(
    member_of: list[str],
    channels: list[str],
    hw_units: Any,
    physics_units: str,
    family: str,
    **extra: Any,
) -> dict[str, Any]:
    """One AO field, with the conversion names and the metadata every field carries."""
    conversion = CONVERSIONS[family]
    body: dict[str, Any] = {
        "MemberOf": member_of,
        "Mode": "Simulator",
        "DataType": "Scalar",
        "ChannelNames": channels,
        "HWUnits": hw_units,
        "PhysicsUnits": physics_units,
        "Units": "Hardware",
        "HW2PhysicsFcn": Fn(conversion["fcn"]),
        "Physics2HWFcn": Fn(conversion["inverse_fcn"]),
    }
    body.update(extra)
    return body


def _channels(family: str, field: str, count: int) -> list[str]:
    """Invented channel names, one per device."""
    suffix = {"Setpoint": "CUR:SP", "Monitor": "CUR:RB"}.get(field, "VAL")
    return [f"QK:{family}:{device}:{suffix}" for device in range(1, count + 1)]


def build_ao(ring: at.Lattice) -> dict[str, Any]:
    """The Accelerator Objects of the invented sub-machine."""
    quad_f = at_index(ring, _named(ring, "QF"))
    quad_d = at_index(ring, _named(ring, "QD"))
    sext = at_index(ring, _named(ring, "SF"))
    monitors = at_index(ring, _named(ring, "BPM"))
    dipoles = at_index(ring, _named(ring, "BD"))
    cavity = at_index(ring, _named(ring, "RFC"))[0]
    septum = at_index(ring, [len(ring) - 2])[0]

    correctors = []
    for device in range(CELLS):
        row = at_index(ring, _corrector_elements(ring, device))
        correctors.append(row + [float("nan")] * (2 - len(row)))
    corrector_index = np.array(correctors, dtype=float)

    ao: dict[str, Any] = {}

    ao["QF"] = _magnet(
        "QF",
        ["QUAD", "Magnet"],
        quad_f,
        "K",
        "1/m^2",
        4,
        setpoint_range=[0.0, 200.0],
        monitor_range=[0.0, 200.0],
    )
    ao["QD"] = _magnet(
        "QD",
        ["QUAD", "Magnet"],
        quad_d,
        "K",
        "1/m^2",
        4,
        setpoint_range=[0.0, 200.0],
        monitor_range=[0.0, 200.0],
    )
    ao["SF"] = _magnet(
        "SF",
        ["SEXT", "Magnet"],
        sext,
        "K2",
        "1/m^3",
        4,
        setpoint_range=[0.0, 150.0],
        monitor_range=None,
        monitor=False,
    )
    ao["SQ"] = _magnet(
        "SQ",
        ["SKEWQUAD", "Magnet"],
        sext,
        "KS",
        "1/m^2",
        4,
        setpoint_range=[-20.0, 20.0],
        monitor_range=None,
        monitor=False,
    )
    ao["HC"] = _magnet(
        "HC",
        ["HCM", "COR", "Magnet"],
        corrector_index,
        "HCM",
        "Radian",
        4,
        setpoint_range=[-1.0, 1.0],
        monitor_range=[-1.0, 1.0],
        hw_units="Ampere",
    )
    ao["VC"] = _magnet(
        "VC",
        ["VCM", "COR", "Magnet"],
        corrector_index,
        "VCM",
        "Radian",
        4,
        setpoint_range=[-5.0, 5.0],
        monitor_range=[-5.0, 5.0],
        hw_units="Ampere",
    )

    ao["BPMx"] = _monitor_family(
        "BPMx",
        ["BPM", "Diagnostics"],
        monitors,
        "BPMx",
        "mm",
        "Meter",
        4,
        monitor_range=[-10.0, 10.0],
    )
    ao["BPMy"] = _monitor_family(
        "BPMy", ["BPM", "Diagnostics"], monitors, "BPMy", "mm", "Meter", 4, monitor_range=None
    )
    # What the Middle Layer corrects each monitor's own reading by, kept on the
    # family itself rather than under its Monitor field, which is the second of
    # the three places the Middle Layer's own readers look.
    ao["BPMx"].update(READOUT["BPMx"])
    ao["BPMy"].update(READOUT["BPMy"])

    ao["BEND"] = _magnet(
        "BEND",
        ["BEND", "Magnet"],
        dipoles,
        "BEND",
        "Radian",
        4,
        setpoint_range=[0.0, 600.0],
        monitor_range=[0.0, 600.0],
        hw_units="Ampere",
    )
    ao["BDM"] = _magnet(
        "BDM",
        ["BEND", "COR", "Magnet"],
        dipoles[:2],
        "BEND",
        "Radian",
        2,
        setpoint_range=[-2.0, 2.0],
        monitor_range=None,
        monitor=False,
        hw_units="Ampere",
    )
    ao["BSOFT"] = _magnet(
        "BSOFT",
        ["BEND", "Magnet"],
        None,
        None,
        "Radian",
        2,
        setpoint_range=[0.0, 120.0],
        monitor_range=None,
        monitor=False,
        hw_units="Ampere",
    )

    ao["RF"] = _magnet(
        "RF",
        ["RF"],
        cavity,
        "RF Cavity",
        "Hertz",
        1,
        setpoint_range=[510.0, 520.0],
        monitor_range=[510.0, 520.0],
        hw_units="MHz",
    )
    ao["IDGAP"] = _magnet(
        "IDGAP",
        ["ID", "Insertion"],
        [],
        "GAP",
        "Meter",
        2,
        setpoint_range=[4.0, 60.0],
        monitor_range=[4.0, 60.0],
        hw_units="mm",
    )
    ao["IDGAP"]["AT"]["SpecialFunctionSet"] = Fn("qk_setidgap")
    ao["IDGAP"]["AT"]["ATParameterGroup"] = "BendingAngle"

    ao["SEPTUM"] = _monitor_family(
        "SEPTUM", ["SEPTUM", "Injection"], septum, "Septum", "kV", "Volt", 1, monitor_range=None
    )
    ao["DCCT"] = _monitor_family(
        "DCCT", ["DCCT", "Diagnostics"], None, None, "mA", "Ampere", 1, monitor_range=[0.0, 500.0]
    )

    tune = _monitor_family(
        "TUNE", ["TUNE", "Diagnostics"], None, None, "", "", 0, monitor_range=None
    )
    tune["DeviceList"] = np.zeros((0, 0))
    tune["ElementList"] = np.zeros((0, 0))
    tune["Monitor"]["ChannelNames"] = ["QK:TUNE:1:X", "QK:TUNE:1:Y"]
    ao["TUNE"] = tune

    # Not a family at all: a stray text entry of the kind a facility's init
    # leaves in its Accelerator Objects. The export walks every field of the
    # struct, so this one gets a block too -- the one whose block is a refusal
    # and nothing else.
    ao["Version"] = "quokka 4.2"
    return ao


def _at_block(at_type: str | None, index: Any) -> dict[str, Any] | None:
    """One family's AT block, or nothing for a family the Middle Layer has none for."""
    if at_type is None and index is None:
        return None
    block: dict[str, Any] = {}
    if at_type is not None:
        block["ATType"] = at_type
    block["ATIndex"] = np.zeros((0, 0)) if index is None else np.asarray(index, dtype=float)
    return block


def _common(
    family: str, member_of: list[str], at_type: str | None, index: Any, count: int
) -> dict[str, Any]:
    """The family-level keys every family carries."""
    body: dict[str, Any] = {
        "FamilyName": family,
        "MemberOf": member_of,
        "DeviceList": _devices(count),
        "ElementList": np.arange(1, count + 1, dtype=float),
        "Status": np.ones(count),
        "CommonNames": [f"qk-{family.lower()}-{device}" for device in range(1, count + 1)],
    }
    block = _at_block(at_type, index)
    if block is not None:
        body["AT"] = block
    return body


def _magnet(
    family: str,
    member_of: list[str],
    index: Any,
    at_type: str | None,
    physics_units: str,
    count: int,
    *,
    setpoint_range: list[float],
    monitor_range: list[float] | None,
    monitor: bool = True,
    hw_units: Any = "Ampere",
) -> dict[str, Any]:
    """A family with a setting: a Setpoint, and a Monitor where it reads back."""
    body = _common(family, member_of, at_type, index, count)
    body["Setpoint"] = _field(
        member_of + ["Setpoint"],
        _channels(family, "Setpoint", count),
        hw_units,
        physics_units,
        family,
        Range=np.array(setpoint_range, dtype=float),
        Tolerance=0.01,
    )
    if monitor:
        body["Monitor"] = _field(
            member_of + ["Monitor"],
            _channels(family, "Monitor", count),
            hw_units,
            physics_units,
            family,
            **({} if monitor_range is None else {"Range": np.array(monitor_range, dtype=float)}),
        )
    return body


def _monitor_family(
    family: str,
    member_of: list[str],
    index: Any,
    at_type: str | None,
    hw_units: Any,
    physics_units: str,
    count: int,
    *,
    monitor_range: list[float] | None,
) -> dict[str, Any]:
    """A family that only reads: its reading is its setting."""
    body = _common(family, member_of, at_type, index, count)
    body["Monitor"] = _field(
        member_of + ["Monitor"],
        _channels(family, "Monitor", count),
        hw_units,
        physics_units,
        family,
        **({} if monitor_range is None else {"Range": np.array(monitor_range, dtype=float)}),
    )
    return body


def build_ad(ring: at.Lattice) -> dict[str, Any]:
    """The Accelerator Data of the invented sub-machine."""
    return {
        "Machine": MACHINE,
        "SubMachine": SUBMACHINE,
        "OperationalMode": "Simulated user beam",
        "Energy": DECK_ENERGY_GEV,
        "InjectionEnergy": DECK_ENERGY_GEV,
        "Circumference": _round_significant(ring.circumference, 12),
        "HarmonicNumber": float(HARMONIC),
        "MCF": _round_significant(float(at.get_mcf(ring.radiation_off(copy=True))), 6),
        "ATModel": "quokka_sr_lattice",
        "OpsData": {
            "LatticeFile": "quokka_sr_deck",
            "RespFiles": "quokka_sr_respmat",
            "PhysDataFile": "quokka_sr_physdata",
        },
    }


# ---------------------------------------------------------------------------
# The export pipeline, ported from mml_export.m.
# ---------------------------------------------------------------------------

#: How many points every calibration is sampled at. Odd, so the middle of a
#: symmetric grid is a sample rather than a gap between two.
GRID_POINTS = 33

#: How far a sample may sit off the line through two of its own before the
#: conversion is written as a table, relative to that device's largest sample.
LINEAR_TOLERANCE = 1e-9

#: How far the energy is moved to see whether a conversion follows it, and how
#: far k*Brho may move over that step before the conversion reads as not
#: carrying the rigidity.
ENERGY_STEP = 1.02
ENERGY_TOLERANCE = 1e-6

#: The physics span a monitor-only family with no Range is sampled over: 10 mm
#: of beam position either side of the axis, in metres.
MONITOR_SPAN = 0.010

#: How many significant digits of the response matrix are written.
RESPONSE_DIGITS = 6

#: How many significant digits of a rigidity deviation are written. Every
#: reader weighs the deviation against a tolerance of 1e-6, so the digits past
#: these are the arithmetic's noise and nobody's fact.
DEVIATION_DIGITS = 6


def _round_significant(value: float, digits: int) -> float:
    """One value at ``digits`` significant digits, the way MATLAB's round does it."""
    if value == 0 or not math.isfinite(value):
        return value
    exponent = math.floor(math.log10(abs(value)))
    factor = 10.0 ** (digits - 1 - exponent)
    return round(value * factor) / factor


def _finite_band(band: np.ndarray | None, devices: int) -> np.ndarray:
    """Which devices carry a band a grid could be laid over."""
    if band is None or band.size == 0 or band.shape[0] != devices:
        return np.zeros(devices, dtype=bool)
    return np.all(np.isfinite(band), axis=1) & (band[:, 0] < band[:, 1])


def _anchor(
    nominal: np.ndarray | None, band: np.ndarray | None, devices: int
) -> tuple[np.ndarray, str]:
    """The nominal a grid is sized by, and the word for where that number came from."""
    anchored = (
        np.full(devices, np.nan)
        if nominal is None or nominal.size != devices
        else np.asarray(nominal, dtype=float).flatten().copy()
    )
    missing = ~np.isfinite(anchored)
    if not missing.any():
        return anchored, "nominal"
    usable = missing & _finite_band(band, devices)
    if band is not None and band.size:
        anchored[usable] = (band[usable, 0] + band[usable, 1]) / 2.0
    anchored[missing & ~usable] = 0.0
    return anchored, ("zero" if (missing & ~usable).any() else "range_midpoint")


def _hardware_grid(
    band: np.ndarray | None, nominal: np.ndarray | None, devices: int
) -> tuple[np.ndarray, str, str]:
    """The hardware values a field is sampled at, one row per device.

    A device is sampled over its own band, stretched just far enough to hold
    its anchor where the band the facility states does not reach it, and a
    device with no band at all over the wide symmetric span. The word names
    the weakest grid the field used.
    """
    anchored, anchor = _anchor(nominal, band, devices)
    banded = _finite_band(band, devices)
    low = -np.maximum(2.0 * np.abs(anchored), 1.0)
    high = -low
    if band is not None and banded.any():
        low[banded] = np.minimum(band[banded, 0], anchored[banded])
        high[banded] = np.maximum(band[banded, 1], anchored[banded])
    source = "range" if banded.all() else "fallback"
    steps = np.arange(GRID_POINTS, dtype=float) / (GRID_POINTS - 1)
    return low[:, None] + (high - low)[:, None] * steps[None, :], source, anchor


def _family_range(ao: dict[str, Any], family: str, field: str, devices: int) -> np.ndarray | None:
    """A field's Range as one row per device, or nothing when it is no band at all."""
    body = ao[family].get(field, {}) if isinstance(ao[family], dict) else {}
    band = body.get("Range")
    if band is None:
        return None
    band = np.atleast_2d(np.asarray(band, dtype=float))
    if band.shape[1] != 2:
        return None
    if band.shape[0] == 1:
        return np.repeat(band, devices, axis=0)
    return band if band.shape[0] == devices else None


def _finite_span(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The hardware span each device's table actually answers over."""
    span = np.zeros((grid.shape[0], 2))
    for row in range(grid.shape[0]):
        finite = np.flatnonzero(np.isfinite(values[row]))
        if finite.size == 0:
            raise Refusal(f"Device row {row + 1} converted to no number anywhere on its grid.")
        span[row] = [grid[row, finite[0]], grid[row, finite[-1]]]
    return span


def _calibration(grid: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    """A sampled conversion as the least the consumer needs to reproduce it."""
    gain = (values[:, -1] - values[:, 0]) / (grid[:, -1] - grid[:, 0])
    offset = values[:, 0] - gain * grid[:, 0]

    with np.errstate(invalid="ignore"):
        scale = (
            np.nanmax(np.abs(values), axis=1)
            if np.isfinite(values).any()
            else np.full(values.shape[0], np.nan)
        )
    scale = np.where(~np.isfinite(scale) | (scale == 0), 1.0, scale)
    residual = np.abs(values - (gain[:, None] * grid + offset[:, None]))

    straight = (
        np.all(np.isfinite(gain))
        and np.all(np.isfinite(offset))
        and bool(np.all(residual <= LINEAR_TOLERANCE * scale[:, None]))
    )
    if straight:
        return {"kind": "linear", "gain": gain, "offset": offset}
    return {
        "kind": "table",
        "grid": grid,
        "values": values,
        "finite_span": _finite_span(grid, values),
    }


class Refusal(Exception):
    """What a conversion the facility refused leaves behind."""


def _require_finite(
    family: str, field: str, fcn: str, grid: np.ndarray, values: np.ndarray
) -> None:
    """Refuse a field whose conversion answered with nothing a consumer can use."""
    bad = np.flatnonzero(np.sum(np.isfinite(values), axis=1) < 2)
    if bad.size:
        rows = " ".join(str(index + 1) for index in bad)
        raise Refusal(
            f"{family}.{field}: {fcn} answered with nothing usable over the grid "
            f"{np.min(grid[bad]):g} to {np.max(grid[bad]):g} for device row(s) {rows}."
        )


def _sample(family: str, direction: str, grid: np.ndarray, energy: float) -> np.ndarray:
    """The image of a grid under one of the family's conversions, column by column."""
    convert = CONVERSIONS[family][direction]
    values = np.zeros(grid.shape)
    for column in range(grid.shape[1]):
        values[:, column] = np.asarray(convert(grid[:, column], energy), dtype=float).flatten()
    return values


def _sample_calibration(
    ao: dict[str, Any],
    family: str,
    field: str,
    nominal: np.ndarray | None,
    energy: float,
    devices: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """One field's hardware-to-physics calibration, sampled through the conversion."""
    if devices == 0:
        raise Refusal(f"Family {family} lists no devices to sample.")
    grid, source, anchor = _hardware_grid(
        _family_range(ao, family, field, devices), nominal, devices
    )
    values = _sample(family, "hw2physics", grid, energy)
    _require_finite(family, field, CONVERSIONS[family]["fcn"], grid, values)

    calibration = _calibration(grid, values)
    calibration["grid_source"] = source
    calibration["anchor"] = anchor
    calibration["fcn"] = CONVERSIONS[family]["fcn"]
    return calibration, grid, values


def _sample_inverse(
    family: str, sampled: dict[str, Any], devices: int, energy: float
) -> dict[str, Any]:
    """The Monitor field's physics-to-hardware conversion, sampled the way a monitor reads back."""
    if "Setpoint" in sampled:
        grid, source = sampled["Setpoint"]["values"], "setpoint"
    elif sampled["Monitor"]["calibration"]["grid_source"] == "range":
        grid, source = sampled["Monitor"]["values"], "range"
    else:
        grid = np.ones((devices, 1)) * np.linspace(-MONITOR_SPAN, MONITOR_SPAN, GRID_POINTS)
        source = "fallback"

    values = _sample(family, "physics2hw", grid, energy)
    _require_finite(family, "Monitor", CONVERSIONS[family]["inverse_fcn"], grid, values)
    inverse = _calibration(grid, values)
    inverse["grid_source"] = source
    inverse["fcn"] = CONVERSIONS[family]["inverse_fcn"]
    return inverse


def _energy_scaling(
    family: str, field: str, grid: np.ndarray, values: np.ndarray, energy: float
) -> tuple[str, float]:
    """Whether a family's conversion carries the beam's rigidity, measured rather than assumed."""
    shifted = _sample(family, "hw2physics", grid, energy * ENERGY_STEP)
    _require_finite(family, field, CONVERSIONS[family]["fcn"], grid, shifted)

    reference = values * brho(energy)
    moved = shifted * brho(energy * ENERGY_STEP)
    both = np.isfinite(reference) & np.isfinite(moved)
    if not both.any():
        raise Refusal(
            f"{family}.{field}: no grid point converted at both {energy:g} and "
            f"{energy * ENERGY_STEP:g} GeV."
        )

    scale = np.maximum(np.abs(reference), np.abs(moved))
    with np.errstate(invalid="ignore"):
        deviations = np.abs(moved - reference) / scale
    deviations[(scale == 0) | ~both] = 0.0
    deviation = float(np.max(deviations))
    # The word follows the measured deviation; the number recorded beside it is
    # rounded, because a deviation this small is arithmetic noise whose last
    # digits belong to the machine that computed them and to no reader.
    word = "brho" if deviation <= ENERGY_TOLERANCE else "none"
    return word, _round_significant(deviation, DEVIATION_DIGITS)


def _energy_candidate(ao: dict[str, Any], family: str) -> bool:
    """Whether the ring's energy is read from this family."""
    body = ao[family]
    members = [str(name).upper() for name in body.get("MemberOf", [])]
    at_type = str(body.get("AT", {}).get("ATType", "")).upper()
    return (at_type == "BEND" or "BEND" in members) and "COR" not in members


def _energy_table(
    family: str,
    devices: np.ndarray,
    sampled: dict[str, Any],
    nominals: dict[str, np.ndarray],
    energy: float,
) -> tuple[dict[str, Any], str]:
    """What the facility's conversion says the ring's energy is over this family's grid."""
    table: dict[str, Any] = {}
    try:
        if devices.size == 0 or "Setpoint" not in sampled:
            raise Refusal(
                f"{family}: the energy table is sampled over the Setpoint grid, which this "
                "family has none of."
            )
        conversion = ENERGY_CONVERSIONS[family]
        row = devices[0]
        grid = sampled["Setpoint"]["grid"][0]
        values = np.array(
            [float(np.atleast_1d(conversion["bend2gev"](np.array([point])))[0]) for point in grid]
        )
        if not np.isfinite(values).any():
            raise Refusal(
                f"{family}: bend2gev answered with no number over the grid "
                f"{np.min(grid):g} to {np.max(grid):g}."
            )

        table["device_row"] = row
        table["grid"] = grid
        table["values"] = values
        table["finite_span"] = _finite_span(grid[None, :], values[None, :])[0]

        current = float(conversion["gev2bend"](energy))
        if not math.isfinite(current):
            raise Refusal(
                f"{family}.Setpoint: gev2bend is NaN for the deck energy of "
                f"{energy:g} GeV, so the table has no nominal."
            )
        table["I_nom"] = current
        at_nominal = float(np.atleast_1d(conversion["bend2gev"](np.array([current])))[0])
        if not math.isfinite(at_nominal):
            raise Refusal(f"{family}.Setpoint: bend2gev is NaN at the nominal current {current:g}.")
        table["energy_at_nominal"] = at_nominal
    except Refusal as refusal:
        return table, str(refusal)
    return table, ""


def _model_read(ring: at.Lattice, family: str, devices: int) -> tuple[np.ndarray, str]:
    """What the Middle Layer's model read answers for one family, and its units.

    Every answer but the beam monitors' is the seam the ring was built from,
    so the document and the lattice describe one operating point. The monitors
    are read off the ring itself, the way the model read reads them.
    """
    if family in ("BPMx", "BPMy"):
        plane = 0 if family == "BPMx" else 2
        orbit = at.find_orbit4(ring, refpts=_named(ring, "BPM"))[1]
        return (
            np.array([_round_significant(float(row[plane]) * 1.0e3, 12) for row in orbit]),
            "Hardware",
        )
    if family == "RF":
        frequency = float(ring[_named(ring, "RFC")[0]].Frequency) / 1.0e6
        return np.array([_round_significant(frequency, 12)]), "Hardware"
    if family == "SEPTUM":
        # The septum's model read has no hardware conversion and answers in
        # physics units, which is a hardware grid this export cannot lay.
        return np.array(NOMINAL_AMPS[family]) * CONVERSIONS[family]["gain"], "Physics"
    return np.array(NOMINAL_AMPS[family], dtype=float), "Hardware"


def _synthetic(ring: at.Lattice, ao: dict[str, Any], family: str, values: np.ndarray) -> bool:
    """Whether the model read answers this family with a number it makes up."""
    block = ao[family].get("AT")
    at_type = str(block.get("ATType", "")) if block else ""
    return bool(
        family.upper() == "DCCT"
        or at_type in ("Septum", "null", "Photon BPM")
        # An element of the family's own name, matched exactly the way the
        # export's findcells matches it: a longer name that merely opens with
        # the family's is a different element.
        or (block is None and not any(element.FamName == family for element in ring))
        or not np.isfinite(values).any()
    )


def _nominals(
    ring: at.Lattice, ao: dict[str, Any], family: str, devices: int
) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    """What the Middle Layer says one family is set to, per device, in hardware units."""
    recorded: dict[str, Any] = {}
    seam: dict[str, np.ndarray] = {}
    body = ao[family]
    field = next((name for name in ("Setpoint", "Monitor") if isinstance(body.get(name), dict)), "")
    if not field:
        return recorded, seam, ""

    try:
        if devices == 0:
            raise Refusal(f"Family {family} lists no devices to read a nominal for.")
        block = body.get("Setpoint" if field == "Setpoint" else "Monitor", {}).get(
            "AT"
        ) or body.get("AT")
        values, units = _model_read(ring, family, devices)

        recorded[field] = {
            "values": values,
            "units": units,
            "at_type": str(block.get("ATType", "")) if block else "",
            "at_index": block.get("ATIndex", np.zeros((0, 0))) if block else "",
            "synthetic": _synthetic(ring, ao, family, values),
        }
        if units and units.lower() != "hardware":
            raise Refusal(
                f"{family}.{field}: getpvmodel answered the nominal in {units} "
                "units, not the hardware units it was asked in."
            )
        seam[field] = values
    except Refusal as refusal:
        return recorded, seam, str(refusal)
    return recorded, seam, ""


def _field_names(ao: dict[str, Any], family: str) -> list[str]:
    """Every field the family answers to, in the order the Accelerator Objects hold them."""
    return [name for name, body in ao[family].items() if isinstance(body, dict) and "Mode" in body]


def _readout(ao: dict[str, Any], family: str, devices: int) -> dict[str, Any]:
    """What one family's readings are corrected by, as the block carries it.

    Each number is looked up under the family's Monitor field first and on the
    family itself second, the way the Middle Layer's own readers look it up.
    The exporter has a third place to look, the facility's physics data file,
    which this fixture has no counterpart for. A key none of them carries is
    left out rather than defaulted, so the block states what the facility
    states and no more.

    A single number is what the whole family holds, and the Middle Layer hands
    it back once per device, so it is written out once per device here too. Any
    other shape is no shape for one number per device and is refused by name.
    """
    body = ao[family]
    field = body.get("Monitor") if isinstance(body.get("Monitor"), dict) else {}
    readout: dict[str, Any] = {}
    for stored in ("Gain", "Offset", "Roll", "Crunch"):
        value = field.get(stored, body.get(stored))
        if value is None:
            continue
        column = np.atleast_1d(np.asarray(value, dtype=float))
        if column.size == 1:
            column = np.full(devices, float(column[0]))
        elif column.size != devices:
            raise ValueError(f"{family}.{stored} holds {column.size} values for {devices} devices.")
        readout[stored.lower()] = column
    return readout


def _va_family(ring: at.Lattice, ao: dict[str, Any], family: str, energy: float) -> dict[str, Any]:
    """One family's VA block, filled fact by fact."""
    block: dict[str, Any] = {}
    device_list = np.atleast_2d(
        np.asarray(ao[family].get("DeviceList", np.zeros((0, 0))), dtype=float)
    )
    devices = 0 if device_list.size == 0 else device_list.shape[0]
    block["device_list"] = device_list if devices else np.zeros((0, 0))
    block["fields"] = _field_names(ao, family)

    recorded, seam, refused_nominal = _nominals(ring, ao, family, devices)
    if recorded:
        block["nominals"] = recorded

    sampled: dict[str, Any] = {}
    refused_sample = ""
    try:
        for name in ("Setpoint", "Monitor"):
            if not isinstance(ao[family].get(name), dict):
                continue
            calibration, grid, values = _sample_calibration(
                ao, family, name, seam.get(name), energy, devices
            )
            sampled[name] = {"calibration": calibration, "grid": grid, "values": values}
        if "Setpoint" in sampled:
            sampled["Setpoint"]["energy_scaling"], sampled["Setpoint"]["energy_deviation"] = (
                _energy_scaling(
                    family,
                    "Setpoint",
                    sampled["Setpoint"]["grid"],
                    sampled["Setpoint"]["values"],
                    energy,
                )
            )
    except Refusal as refusal:
        refused_sample = str(refusal)

    if not refused_sample and "Monitor" in sampled:
        try:
            sampled["Monitor"]["monitor_inverse"] = _sample_inverse(
                family, sampled, devices, energy
            )
        except Refusal as refusal:
            refused_sample = str(refusal)

    if "Setpoint" in sampled:
        setpoint = {"calibration": sampled["Setpoint"]["calibration"]}
        if "energy_scaling" in sampled["Setpoint"]:
            setpoint["energy_scaling"] = sampled["Setpoint"]["energy_scaling"]
            setpoint["energy_deviation"] = sampled["Setpoint"]["energy_deviation"]
        block["Setpoint"] = setpoint
    if "Monitor" in sampled:
        monitor = {"calibration": sampled["Monitor"]["calibration"]}
        if "monitor_inverse" in sampled["Monitor"]:
            monitor["monitor_inverse"] = sampled["Monitor"]["monitor_inverse"]
        readout = _readout(ao, family, devices)
        if readout:
            monitor["readout"] = readout
        block["Monitor"] = monitor

    refused_table = ""
    block["energy_candidate"] = _energy_candidate(ao, family)
    if block["energy_candidate"]:
        table, refused_table = _energy_table(family, device_list, sampled, seam, energy)
        if table:
            block["energy_table"] = table

    # Refusals are joined in the order they were met, and a sentence met twice
    # is recorded once: two steps that fail for the same reason say so once.
    kept: list[str] = []
    for message in (refused_nominal, refused_sample, refused_table):
        if message and message not in kept:
            kept.append(message)
    refused = "; ".join(kept)
    if refused:
        block["refused"] = refused
    return block


def _famnames(saved: list[Any]) -> list[str]:
    """The family name of every element of ``THERING``, in ring order, as the ring carries it."""
    return [element.FamName for element in saved]


def build_va(ring: at.Lattice, ao: dict[str, Any]) -> dict[str, Any]:
    """The per-family calibration document."""
    saved = saved_ring(ring)
    digest = hashlib.sha256("\n".join(_famnames(saved)).encode("utf-8")).hexdigest()
    ringparams = [
        index + 1
        for index, element in enumerate(saved)
        if getattr(element, "Class", None) == "RingParam"
    ]

    va: dict[str, Any] = {
        "lattice": {
            "elements": float(len(saved)),
            "famname_sha256": digest,
            "energy_gev": DECK_ENERGY_GEV,
            "ringparam_indices": np.array(ringparams, dtype=float),
        },
        "families": {},
    }
    for family, body in ao.items():
        if not isinstance(body, dict):
            # The export walks every field of the Accelerator Objects, and one
            # that is not a struct is a family it could not read at all.
            va["families"][family] = {
                "refused": "Invalid input argument of type 'char'. Input must be a structure "
                "array or an object."
            }
            continue
        va["families"][family] = _va_family(ring, ao, family, DECK_ENERGY_GEV)
    return va


# ---------------------------------------------------------------------------
# The response matrix.
# ---------------------------------------------------------------------------

#: The kick each corrector is stepped by while the matrix is measured, in radians.
ACTUATOR_DELTA = 1.0e-5

#: The monitor device whose measurement this file does not count as good.
BAD_MONITOR = 2


def _orbit(ring: at.Lattice) -> np.ndarray:
    """The closed orbit at the beam monitors, one row per monitor, in metres.

    The orbit is solved about the RF bucket, with longitudinal motion on for the
    cavity and nothing else, which is the state the served model runs in. A
    corrector's kick lengthens the path it steers the beam along, and a ring
    that cannot pay for that in energy pays for it in one constant offset at
    every monitor of the column instead -- an offset that belongs to the solve
    and not to the machine. Measuring the matrix about the same orbit the model
    is asked about afterwards is what leaves the two comparable.
    """
    return at.find_orbit6(ring.enable_6d(at.RFCavity, copy=True), refpts=_named(ring, "BPM"))[1]


def _response_column(ring: at.Lattice, plane: int, device: int) -> np.ndarray:
    """One corrector's orbit response, measured the way a model measurement is: both ways."""
    indices = _corrector_elements(ring, device)
    step = ACTUATOR_DELTA / len(indices)
    saved = [np.array(ring[index].KickAngle, dtype=float) for index in indices]

    def kick(sign: float) -> np.ndarray:
        for index, angle in zip(indices, saved, strict=True):
            moved = angle.copy()
            moved[plane] += sign * step
            ring[index].KickAngle = moved
        return _orbit(ring)

    plus, minus = kick(+1.0), kick(-1.0)
    for index, angle in zip(indices, saved, strict=True):
        ring[index].KickAngle = angle
    return (plus - minus) / (2.0 * ACTUATOR_DELTA)


def _rounded(value: Any) -> np.ndarray:
    """One measured value at the digits a measurement carries."""
    array = np.asarray(value, dtype=float)
    flat = np.array([_round_significant(float(item), RESPONSE_DIGITS) for item in array.flatten()])
    return flat.reshape(array.shape)


def build_response(ring: at.Lattice, ao: dict[str, Any]) -> dict[str, Any]:
    """The orbit response matrix, measured on the model, block by block."""
    orbit = _orbit(ring)
    columns = {
        family: [_response_column(ring, plane, device) for device in range(CELLS)]
        for plane, family in ((0, "HC"), (1, "VC"))
    }

    blocks = []
    for monitor_family, row in (("BPMx", 0), ("BPMy", 2)):
        for actuator_family in ("HC", "VC"):
            data = np.column_stack([column[:, row] for column in columns[actuator_family]])
            status = np.ones(CELLS)
            point = np.array([position[row] for position in orbit])
            if monitor_family == "BPMy":
                # A device the facility's own file does not hold: its flag is
                # down and its row of the matrix is no number at all.
                status[BAD_MONITOR] = 0.0
                data[BAD_MONITOR, :] = np.nan
                point[BAD_MONITOR] = np.nan
            blocks.append(
                {
                    "monitor": {
                        "family": monitor_family,
                        "device_list": ao[monitor_family]["DeviceList"],
                        "mode": "Simulator",
                        "status": status,
                        "data": _rounded(point),
                    },
                    "actuator": {
                        "family": actuator_family,
                        "device_list": ao[actuator_family]["DeviceList"],
                        "mode": "Simulator",
                        "status": np.ones(CELLS),
                        # The horizontal correctors' settings were kept when this
                        # matrix was taken; the vertical file predates the field.
                        "data": (
                            _rounded(
                                np.array(NOMINAL_AMPS[actuator_family])
                                * CONVERSIONS[actuator_family]["gain"]
                            )
                            if actuator_family == "HC"
                            else np.nan
                        ),
                    },
                    "origin": "model",
                    "timestamp": TIMESTAMP,
                    "gev": DECK_ENERGY_GEV,
                    "units": "Physics",
                    "units_string": "meter/radian",
                    "modulation_method": "bipolar",
                    "actuator_delta": ACTUATOR_DELTA,
                    "data": _rounded(data),
                }
            )
    return {"file": "", "blocks": blocks}


# ---------------------------------------------------------------------------
# Writing, and checking what was written.
# ---------------------------------------------------------------------------

#: The directory the committed fixture lives in.
HERE = Path(__file__).resolve().parent

#: Every file this script writes, in the order it writes them.
FILES = (
    f"{STEM}.lattice.mat",
    "mismatched.lattice.mat",
    f"{STEM}.ao.json",
    f"{STEM}.ad.json",
    f"{STEM}.va.json",
    f"{STEM}.response.json",
)


def build(outdir: Path) -> list[Path]:
    """Write the whole fixture into ``outdir`` and return the paths, in order."""
    outdir.mkdir(parents=True, exist_ok=True)
    ring = build_ring()

    save_lattice(ring, outdir / FILES[0])

    # The pairing counter-example: the same ring under one other name, so a
    # consumer holding it recomputes a digest the document does not carry while
    # the element count still matches.
    mismatched = build_ring()
    mismatched[_named(mismatched, "QF")[0]].FamName = "QFX"
    save_lattice(mismatched, outdir / FILES[1])

    # A committed text file ends in a newline, the exporter's own text does not;
    # the newline is the file's, so the documents stay the exporter's spelling.
    ao = build_ao(ring)
    bodies = (ao, build_ad(ring), build_va(ring, ao), build_response(ring, ao))
    for name, body in zip(FILES[2:], bodies, strict=True):
        (outdir / name).write_text(document(body) + "\n", encoding="utf-8")
    return [outdir / name for name in FILES]


def check() -> int:
    """Rebuild into a temporary directory and compare every committed byte."""
    with tempfile.TemporaryDirectory() as workdir:
        rebuilt = build(Path(workdir))
        failures = []
        for path in rebuilt:
            committed = HERE / path.name
            if not committed.exists():
                failures.append(f"{path.name}: not committed")
            elif committed.read_bytes() != path.read_bytes():
                failures.append(f"{path.name}: differs from the committed file")
        for line in failures:
            print(line, file=sys.stderr)
        if failures:
            return 1
        print(f"{len(rebuilt)} files regenerate byte-identically")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild into a temporary directory and compare the bytes",
    )
    parser.add_argument(
        "outdir",
        nargs="?",
        type=Path,
        default=HERE,
        help="where to write the fixture (default: beside this script)",
    )
    args = parser.parse_args(argv)
    if args.check:
        return check()
    for path in build(args.outdir):
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
