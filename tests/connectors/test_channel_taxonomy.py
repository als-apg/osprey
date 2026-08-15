"""Tests for the shared procedural-PV taxonomy used by the mock connectors."""

import pytest

from osprey.connectors.channel_taxonomy import classify_channel


@pytest.mark.parametrize(
    ("channel", "kind", "base_value", "units", "noise_scale"),
    [
        ("SR:BEAM:CURRENT", "beam_current", 500.0, "mA", 0.0),
        ("SR:DCCT", "beam_current", 500.0, "mA", 0.0),  # DCCT is a beam-current monitor
        ("PS:CURRENT", "current", 150.0, "A", 0.0),
        ("RF:VOLTAGE", "voltage", 5000.0, "V", 0.0),
        ("RF:POWER", "power", 50.0, "kW", 0.0),
        ("VAC:PRESSURE", "pressure", 1e-9, "Torr", 0.0),
        ("CRYO:TEMP", "temperature", 25.0, "°C", 0.0),
        ("SR:LIFETIME", "lifetime", 10.0, "hours", 0.0),
        ("BPM:POSITION:X", "position", 0.0, "mm", 0.005),
        ("BPM:POS:Y", "position", 0.0, "mm", 0.005),
        ("SR:ENERGY", "energy", 1900.0, "MeV", 0.0),
        ("SOME:RANDOM:PV", "default", 100.0, "", 0.0),
    ],
)
def test_classify_channel(channel, kind, base_value, units, noise_scale):
    result = classify_channel(channel)
    assert result.name == kind
    assert result.base_value == base_value
    assert result.units == units
    assert result.noise_scale == noise_scale


def test_beam_current_takes_priority_over_generic_current():
    """A PV matching both 'beam' and 'current' classifies as beam_current."""
    assert classify_channel("BEAM:CURRENT:MONITOR").name == "beam_current"
    assert classify_channel("CORRECTOR:CURRENT").name == "current"


def test_position_is_the_only_kind_carrying_an_absolute_noise_floor():
    """``noise_scale`` exists only for kinds whose base can legitimately be 0.

    A single global absolute floor would be wrong-scale across bases spanning
    1e-9 Torr to 5000 V, so every kind with a non-zero base keeps ``0.0`` and
    its fallback sigma stays purely relative.
    """
    probes = {
        "SR:BEAM:CURRENT": 0.0,
        "PS:CURRENT": 0.0,
        "RF:VOLTAGE": 0.0,
        "RF:POWER": 0.0,
        "VAC:PRESSURE": 0.0,
        "CRYO:TEMP": 0.0,
        "SR:LIFETIME": 0.0,
        "SR:ENERGY": 0.0,
        "SOME:RANDOM:PV": 0.0,
        "BPM:POSITION:X": 0.005,
    }
    for channel, expected in probes.items():
        kind = classify_channel(channel)
        assert kind.noise_scale == expected, channel
        # The floor is only ever needed where the relative sigma is dead.
        assert (kind.noise_scale > 0.0) == (kind.base_value == 0.0), channel
