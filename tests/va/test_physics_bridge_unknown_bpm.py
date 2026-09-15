"""Tests for how a BPM error seed naming a BPM the ring lacks is refused.

Split out of test_physics_bridge.py because this is the one path where a seed
never reaches a reading at all. BPM readout errors are model state: they are
seeded through ``PyATRingModel(bpm_errors=...)``, and the model refuses a
fam_name its ring has no BPM for before it builds anything. A typo is
therefore a construction error naming the offender, never a seed that
silently perturbs nothing -- on the live stand-in (whose whole difference
from the sandbox VA is a shipped BPM offset) a silent drop would make the two
targets identical while looking configured.

Like test_physics_bridge.py, this targets the real ALS-U AR ring
(`PyATRingModel()`), not a toy lattice: the "known" ids asserted here are the
ones that ring actually carries.
"""

from __future__ import annotations

import logging

import pytest

from osprey.services.virtual_accelerator.ioc.physics_bridge import (
    PhysicsBridge,
    UnknownDeviceError,
)
from osprey.services.virtual_accelerator.model.pyat import PyATRingModel

_BRIDGE_LOGGER = "osprey.services.virtual_accelerator.ioc.physics_bridge"


class FakeRecord:
    """Minimal duck-typed stand-in for a softioc In record: just `.set()`."""

    def __init__(self) -> None:
        self.value: float | None = None

    def set(self, value: float) -> None:
        self.value = value


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The bridge's own WARNING messages, rendered, in emission order.

    Filtered by logger name: PyAT and numpy warn on this path too, and the
    assertions here are about what *this module* emitted.
    """
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == _BRIDGE_LOGGER and record.levelno == logging.WARNING
    ]


class TestUnknownBpmErrorId:
    """FR10: a `bpm_errors` id the lattice has no BPM for is refused at
    construction, by name, instead of silently applying to nothing."""

    def test_unknown_bpm_id_is_refused_naming_the_id(self):
        with pytest.raises(UnknownDeviceError, match="BPM99"):
            PyATRingModel(bpm_errors={"BPM99": {"offset_x": 1e-4}})

    def test_two_unknown_bpm_ids_are_named_in_one_sorted_refusal(self):
        # Sorted, so the refusal reads the same on every start whatever the
        # seed's own key order.
        with pytest.raises(UnknownDeviceError, match=r"\['BPM98', 'BPM99'\]"):
            PyATRingModel(bpm_errors={"BPM99": {"offset_x": 1e-4}, "BPM98": {"gain_x": 1.5}})

    def test_a_known_offset_beside_an_unknown_id_is_refused_as_a_whole(self):
        # The known half of a mixed seed is not applied on its own: the whole
        # seed is refused, and the refusal names only the offender.
        with pytest.raises(UnknownDeviceError) as refusal:
            PyATRingModel(bpm_errors={"BPM01": {"offset_x": 50e-6}, "BPM99": {"offset_x": 1e-4}})

        assert "BPM99" in str(refusal.value)
        assert "BPM01" not in str(refusal.value)

    def test_the_refusal_is_the_error_the_bridge_exports(self):
        # Callers that catch the bridge's `UnknownDeviceError` for a bad
        # address catch a bad seed with the same clause.
        with pytest.raises(UnknownDeviceError) as refusal:
            PyATRingModel(bpm_errors={"BPM99": {"offset_x": 1e-4}})

        assert isinstance(refusal.value, ValueError)

    def test_known_id_seeds_the_served_reading_without_a_warning(self, caplog):
        rec = FakeRecord()
        with caplog.at_level(logging.WARNING, logger=_BRIDGE_LOGGER):
            bridge = PhysicsBridge(model=PyATRingModel(bpm_errors={"BPM01": {"offset_x": 50e-6}}))
            bridge.bind({"SR:DIAG:BPM:01:POSITION:X": rec})
            bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 5.0)

        assert _warnings(caplog) == []
        true_position = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]
        assert rec.value == pytest.approx(true_position - 50e-6, abs=1e-12)

    def test_an_unseeded_bridge_emits_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger=_BRIDGE_LOGGER):
            PhysicsBridge(model=PyATRingModel())

        assert _warnings(caplog) == []
