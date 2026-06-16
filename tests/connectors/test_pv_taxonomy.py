"""Tests for the shared procedural-PV taxonomy used by the mock connectors."""

import pytest

from osprey.connectors.pv_taxonomy import classify_pv


@pytest.mark.parametrize(
    ("pv_name", "kind", "base_value", "units"),
    [
        ("SR:BEAM:CURRENT", "beam_current", 500.0, "mA"),
        ("SR:DCCT", "beam_current", 500.0, "mA"),  # DCCT is a beam-current monitor
        ("PS:CURRENT", "current", 150.0, "A"),
        ("RF:VOLTAGE", "voltage", 5000.0, "V"),
        ("RF:POWER", "power", 50.0, "kW"),
        ("VAC:PRESSURE", "pressure", 1e-9, "Torr"),
        ("CRYO:TEMP", "temperature", 25.0, "°C"),
        ("SR:LIFETIME", "lifetime", 10.0, "hours"),
        ("BPM:POSITION:X", "position", 0.0, "mm"),
        ("BPM:POS:Y", "position", 0.0, "mm"),
        ("SR:ENERGY", "energy", 1900.0, "MeV"),
        ("SOME:RANDOM:PV", "default", 100.0, ""),
    ],
)
def test_classify_pv(pv_name, kind, base_value, units):
    result = classify_pv(pv_name)
    assert result.name == kind
    assert result.base_value == base_value
    assert result.units == units


def test_beam_current_takes_priority_over_generic_current():
    """A PV matching both 'beam' and 'current' classifies as beam_current."""
    assert classify_pv("BEAM:CURRENT:MONITOR").name == "beam_current"
    assert classify_pv("CORRECTOR:CURRENT").name == "current"
