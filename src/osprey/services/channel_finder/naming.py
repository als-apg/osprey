"""Canonical channel-naming vocabulary for ALS-U channel databases.

Single source of truth for the token (PascalCase channel-name component) and
phrase (lowercase natural-language component) spellings of every ring, family,
FIELD, and SUBFIELD used when synthesizing ``in_context``-style channel names
and descriptions. Two generators consume it:

* :mod:`osprey.services.channel_finder.tools.generate_from_spec` -- grows the
  shipped tier-3/tier-1 databases; looks tokens up strictly (a missing key is
  a bug in the spec/schema wiring).
* :mod:`osprey.services.channel_finder.benchmarks.generator` -- builds the
  benchmark databases; falls back to the raw component for unmapped keys.

Spellings are sourced verbatim from the shipped SR ``in_context`` entries
where a shipped precedent exists (e.g. ``"QuadFocus"`` / ``"focusing
quadrupole"`` for QF). The three spec families with no shipped precedent
(QFA/SHF/SHD) extend the convention established by their structural analog
(QF, SF, SD respectively). QFA's ``A`` is the ALS convention for the family
placed inside the triple-bend achromat arc (between the dipoles, vs QF/QD at
the straights) — a position label, not an achromaticity claim: the AR runs
with distributed dispersion, so the phrase says "achromat-arc". This module
is a leaf: it must not import anything from ``channel_finder``, so both
generators can depend on it without cycles.
"""

from __future__ import annotations

RING_TOKENS: dict[str, str] = {
    "SR": "StorageRing",
    "BR": "BoosterRing",
    "BTS": "BoosterToStorageRing",
}

RING_PHRASES: dict[str, str] = {
    "SR": "storage ring",
    "BR": "booster ring",
    "BTS": "booster-to-storage transfer line",
}

FAMILY_TOKENS: dict[str, str] = {
    "DIPOLE": "Dipole",
    "QF": "QuadFocus",
    "QD": "QuadDefocus",
    "QFA": "QuadFocusAchromat",
    "SF": "SextFocus",
    "SD": "SextDefocus",
    "SHF": "SextHarmFocus",
    "SHD": "SextHarmDefocus",
    "HCM": "HorizCorr",
    "VCM": "VertCorr",
    "BPM": "BPM",
    "DCCT": "DCCT",
    "ION-PUMP": "IonPump",
    "GAUGE": "VacGauge",
    "VALVE": "GateValve",
    "CAVITY": "Cavity",
    "KLYSTRON": "Klystron",
    "NEUTRON": "NeutronDet",
    "GAMMA": "GammaDet",
}

FAMILY_PHRASES: dict[str, str] = {
    "DIPOLE": "dipole bending magnet",
    "QF": "focusing quadrupole",
    "QD": "defocusing quadrupole",
    "QFA": "achromat-arc focusing quadrupole",
    "SF": "sextupole (focusing)",
    "SD": "sextupole (defocusing)",
    "SHF": "harmonic sextupole (focusing)",
    "SHD": "harmonic sextupole (defocusing)",
    "HCM": "horizontal corrector",
    "VCM": "vertical corrector",
    "BPM": "beam position monitor",
    "DCCT": "DC current transformer",
    "ION-PUMP": "ion pump",
    "GAUGE": "vacuum gauge",
    "VALVE": "gate valve",
    "CAVITY": "RF cavity",
    "KLYSTRON": "klystron",
    "NEUTRON": "neutron detector",
    "GAMMA": "gamma detector",
}

# "GOLDEN" as a FIELD is the BPM top-level golden orbit, distinct from
# magnets' CURRENT:GOLDEN *subfield* (see SUBFIELD_TOKENS below).
FIELD_TOKENS: dict[str, str] = {
    "CURRENT": "Current",
    "STATUS": "Status",
    "POSITION": "Position",
    "PRESSURE": "Pressure",
    "VOLTAGE": "Voltage",
    "POWER": "Power",
    "FREQUENCY": "Frequency",
    "TEMPERATURE": "Temperature",
    "TUNER": "Tuner",
    "LIFETIME": "Lifetime",
    "SIGNAL": "Signal",
    "GOLDEN": "GoldenOrbit",
    "OFFSET": "Offset",
    "DOSE_RATE": "DoseRate",
    "CONTROL": "Control",
}

FIELD_PHRASES: dict[str, str] = {
    "CURRENT": "current",
    "STATUS": "status",
    "POSITION": "position",
    "PRESSURE": "pressure",
    "VOLTAGE": "voltage",
    "POWER": "power",
    "FREQUENCY": "frequency",
    "TEMPERATURE": "temperature",
    "TUNER": "tuner",
    "LIFETIME": "lifetime",
    "SIGNAL": "signal",
    "GOLDEN": "golden orbit",
    "OFFSET": "offset",
    "DOSE_RATE": "dose rate",
    "CONTROL": "control",
}

SUBFIELD_TOKENS: dict[str, str] = {
    "SP": "Setpoint",
    "RB": "Readback",
    "GOLDEN": "Golden",
    "X": "X",
    "Y": "Y",
    "SUM": "Sum",
    "FWD": "Forward",
    "REV": "Reverse",
    "NET": "Net",
    "INST": "Instantaneous",
    "AVG_1MIN": "Avg1Min",
    "AVG_1HR": "Avg1Hr",
    "READY": "Ready",
    "ON": "On",
    "FAULT": "Fault",
    "VALID": "Valid",
    "CONNECTED": "Connected",
    "INTERLOCK": "Interlock",
    "ALARM": "Alarm",
    "OPEN": "Open",
    "CLOSED": "Closed",
    "CLOSE": "Close",
    "MAIN": "Main",
}

SUBFIELD_PHRASES: dict[str, str] = {
    "SP": "setpoint",
    "RB": "readback",
    "GOLDEN": "golden setpoint",
    "X": "horizontal",
    "Y": "vertical",
    "SUM": "sum",
    "FWD": "forward",
    "REV": "reverse",
    "NET": "net",
    "INST": "instantaneous",
    "AVG_1MIN": "1-minute average",
    "AVG_1HR": "1-hour average",
    "READY": "ready",
    "ON": "on",
    "FAULT": "fault",
    "VALID": "valid",
    "CONNECTED": "connected",
    "INTERLOCK": "interlock",
    "ALARM": "alarm",
    "OPEN": "open",
    "CLOSED": "closed",
    "CLOSE": "close",
    "MAIN": "main",
}
