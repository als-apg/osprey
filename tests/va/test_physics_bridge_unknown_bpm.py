"""Tests for PhysicsBridge's boot warning about seeded BPM ids the ring lacks.

Split out of test_physics_bridge.py because this is the one behaviour there
that is observed through the logging system rather than through readings:
`_push_bpm_readbacks` merges seeded errors per *served* device, so a
`bpm_errors` key naming a BPM the lattice does not carry perturbs nothing at
all. On the live stand-in (whose whole difference from the sandbox VA is a
shipped BPM offset) that silent drop would make the two targets identical
while looking configured, so the bridge says so once, at construction.

Like test_physics_bridge.py, this targets the real ALS-U AR ring
(`PyATRingModel()`), not a toy lattice: the "known" ids asserted here are the
ones that ring actually carries.
"""

from __future__ import annotations

import logging

import pytest

from osprey.services.virtual_accelerator.ioc.physics_bridge import PhysicsBridge
from osprey.services.virtual_accelerator.model.pyat import PyATRingModel

_BRIDGE_LOGGER = "osprey.services.virtual_accelerator.ioc.physics_bridge"


class FakeRecord:
    """Minimal duck-typed stand-in for a softioc In record: just `.set()`."""

    def __init__(self) -> None:
        self.value: float | None = None

    def set(self, value: float) -> None:
        self.value = value


@pytest.fixture
def model() -> PyATRingModel:
    """The backend the bridges under test serve.

    Built here rather than left to `PhysicsBridge()`'s own default so every
    test in this module serves exactly one ring, and function-scoped so a
    test that writes a setpoint cannot leave the lattice off-nominal for the
    next one.
    """
    return PyATRingModel()


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The bridge's own WARNING messages, rendered, in emission order.

    Filtered by logger name: PyAT and numpy warn on this path too, and the
    assertions here are about how many warnings *this module* emitted.
    """
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == _BRIDGE_LOGGER and record.levelno == logging.WARNING
    ]


class TestUnknownBpmErrorId:
    """FR10: a `bpm_errors` id the lattice has no BPM for is diagnosable in
    the container log instead of silently applying to nothing."""

    def test_unknown_bpm_id_warns_once_naming_the_id_and_the_env_var(self, model, caplog):
        with caplog.at_level(logging.WARNING, logger=_BRIDGE_LOGGER):
            PhysicsBridge(model=model, bpm_errors={"BPM99": {"offset_x": 1e-4}})

        messages = _warnings(caplog)
        assert len(messages) == 1
        assert "BPM99" in messages[0]
        assert "VA_BPM_ERRORS" in messages[0]

    def test_known_id_emits_no_unknown_bpm_warning(self, model, caplog):
        with caplog.at_level(logging.WARNING, logger=_BRIDGE_LOGGER):
            PhysicsBridge(model=model, bpm_errors={"BPM01": {"offset_x": 1e-4}})

        assert _warnings(caplog) == []

    def test_no_seeded_errors_emit_no_unknown_bpm_warning(self, model, caplog):
        with caplog.at_level(logging.WARNING, logger=_BRIDGE_LOGGER):
            PhysicsBridge(model=model)

        assert _warnings(caplog) == []

    def test_two_unknown_bpm_ids_warn_once_each(self, model, caplog):
        with caplog.at_level(logging.WARNING, logger=_BRIDGE_LOGGER):
            PhysicsBridge(
                model=model,
                bpm_errors={"BPM98": {"gain_x": 1.5}, "BPM99": {"offset_x": 1e-4}},
            )

        messages = _warnings(caplog)
        assert len(messages) == 2
        # Sorted emission, so the pairing is positional, not a search.
        assert "BPM98" in messages[0]
        assert "BPM99" in messages[1]

    def test_known_offset_still_applies_beside_an_unknown_bpm_id(self, model, caplog):
        # The known half of a mixed seed must behave exactly as it does
        # without the typo: the warning is diagnostics, not a fallback.
        rec = FakeRecord()
        with caplog.at_level(logging.WARNING, logger=_BRIDGE_LOGGER):
            bridge = PhysicsBridge(
                model=model,
                bpm_errors={"BPM01": {"offset_x": 50e-6}, "BPM99": {"offset_x": 1e-4}},
            )
        bridge.bind({"SR:DIAG:BPM:01:POSITION:X": rec})
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 5.0)

        messages = _warnings(caplog)
        assert len(messages) == 1
        assert "BPM99" in messages[0]

        true_position = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]
        assert rec.value == pytest.approx(true_position - 50e-6, abs=1e-12)
