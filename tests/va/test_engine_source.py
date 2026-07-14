"""Tests for the partition-(c) engine-source poller (task 3.5).

No container and no live softIOC needed: ``EngineSource`` only needs a real
``SimulationEngine`` (built from a small fake ``data/simulation/`` directory)
and plain fake record objects recording ``.set()`` calls.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from osprey.services.virtual_accelerator.ioc.engine_source import EngineSource
from osprey.services.virtual_accelerator.manifest import (
    PARTITION_STATIC_NOISY,
    RECORD_TYPE_ANALOG,
    RECORD_TYPE_BINARY,
)
from osprey.simulation.engine import SimulationEngine

VAC_RB = "ZZTEST:VAC:PRESSURE:01:RB"
FAULT_BI = "ZZTEST:STATUS:01:FAULT"
NOT_IN_MACHINE = "ZZTEST:UNKNOWN:01:LEVEL:RB"

TEST_MACHINE = {
    "name": "EngineSourceTestRig",
    "description": "Tiny inline machine for engine-source polling tests.",
    "channels": {
        VAC_RB: {"value": 1e-8, "units": "Torr", "noise": 0.0, "description": "vac gauge"},
        FAULT_BI: {"value": 0, "noise": 0.0, "description": "fault flag"},
    },
    "scenarios": {
        "nominal": {"description": "All systems nominal."},
        "leak": {
            "description": "Vacuum leak.",
            "overrides": {VAC_RB: 5e-6},
        },
    },
}


def _channel(address: str, *, record_type: str, noise: bool) -> dict:
    """One synthetic manifest-entry dict, matching what build_manifest() emits."""
    return {
        "address": address,
        "ring": "ZZTEST",
        "system": "VAC",
        "family": "GAUGE",
        "device": "01",
        "field": "PRESSURE",
        "subfield": "RB",
        "partition": PARTITION_STATIC_NOISY,
        "record_type": record_type,
        "noise": noise,
    }


CHANNELS = [
    _channel(VAC_RB, record_type=RECORD_TYPE_ANALOG, noise=False),
    _channel(FAULT_BI, record_type=RECORD_TYPE_BINARY, noise=False),
    _channel(NOT_IN_MACHINE, record_type=RECORD_TYPE_ANALOG, noise=False),
]


class FakeRecord:
    """Stand-in for a softioc In-type record: just records ``.set()`` calls."""

    def __init__(self) -> None:
        self.value = None
        self.set_calls: list[object] = []

    def set(self, value):
        self.value = value
        self.set_calls.append(value)


@pytest.fixture()
def data_dir(tmp_path) -> Path:
    (tmp_path / "machine.json").write_text(json.dumps(TEST_MACHINE))
    return tmp_path


@pytest.fixture()
def engine(data_dir) -> SimulationEngine:
    return SimulationEngine.from_file(data_dir / "machine.json")


@pytest.fixture()
def records() -> dict[str, FakeRecord]:
    return {VAC_RB: FakeRecord(), FAULT_BI: FakeRecord(), NOT_IN_MACHINE: FakeRecord()}


@pytest.fixture()
def source(engine, records, data_dir) -> EngineSource:
    return EngineSource(engine, CHANNELS, records, data_dir)


class TestBasicDrive:
    def test_first_poll_is_not_reported_as_a_switch(self, source):
        assert source.poll_once() is False

    def test_first_poll_pushes_nominal_baseline(self, source, records):
        source.poll_once()
        assert records[VAC_RB].value == pytest.approx(1e-8)

    def test_repeated_poll_without_change_is_not_a_switch(self, source):
        source.poll_once()
        assert source.poll_once() is False


class TestScenarioSwitchChangesComposedValues:
    def test_switching_to_leak_updates_the_pushed_value(self, source, records, engine):
        source.poll_once()  # baseline tick
        engine.set_active_scenarios(["leak"])
        switched = source.poll_once()
        assert switched is True
        assert records[VAC_RB].value == pytest.approx(5e-6)

    def test_switching_back_to_nominal_restores_baseline(self, source, records, engine):
        source.poll_once()
        engine.set_active_scenarios(["leak"])
        source.poll_once()
        engine.set_active_scenarios([])
        source.poll_once()
        assert records[VAC_RB].value == pytest.approx(1e-8)


class TestSessionWrittenRecordsResetOnSwitch:
    def test_session_write_is_reflected_before_any_switch(self, source, records, engine):
        source.poll_once()
        engine.write(VAC_RB, 9.9e-3)
        source.poll_once()
        assert records[VAC_RB].value == pytest.approx(9.9e-3)

    def test_session_write_is_cleared_by_a_scenario_switch(self, source, records, engine):
        source.poll_once()
        engine.write(VAC_RB, 9.9e-3)
        source.poll_once()
        assert records[VAC_RB].value == pytest.approx(9.9e-3)

        # Switching (even to the same nominal set) clears session writes --
        # SimulationEngine's own contract (engine.py's _refresh_scenario).
        engine.set_active_scenarios(["leak"])
        source.poll_once()
        assert records[VAC_RB].value == pytest.approx(5e-6)  # leak override, NOT the stale write

    def test_re_asserting_the_same_set_also_clears_session_writes(self, source, records, engine):
        source.poll_once()
        engine.write(VAC_RB, 9.9e-3)
        source.poll_once()

        engine.set_active_scenarios([])  # re-assert nominal explicitly
        source.poll_once()
        assert records[VAC_RB].value == pytest.approx(1e-8)


class TestAtomicRenameStillDetected:
    """The bind-mounted unit is the DIRECTORY, precisely so an atomic-rename
    swap of active_scenarios survives the mount. Prove this module notices
    such a swap even when the replacement file's mtime is forced to collide
    with the old one (the edge case content-hash comparison exists for)."""

    def test_inode_swap_with_colliding_mtime_is_still_detected(self, source, records, data_dir):
        state_path = data_dir / "active_scenarios"

        # Establish a baseline signature identical to what set_active_scenarios
        # would have written for the (implicit) nominal set.
        state_path.write_text("nominal\n")
        source.poll_once()
        original_ino = state_path.stat().st_ino
        original_mtime_ns = state_path.stat().st_mtime_ns

        # Atomic-rename swap (temp file + os.replace, the same technique
        # apply.py/set_active_scenarios uses) to a DIFFERENT scenario, with
        # its mtime forced to collide with the original -- this is the exact
        # scenario mtime-only detection can miss.
        tmp_path = data_dir / "active_scenarios.tmp"
        tmp_path.write_text("leak\n")
        os.utime(tmp_path, ns=(original_mtime_ns, original_mtime_ns))
        os.replace(tmp_path, state_path)

        assert state_path.stat().st_ino != original_ino  # the swap really happened
        assert state_path.stat().st_mtime_ns == original_mtime_ns  # ...and mtime really collides

        switched = source.poll_once()
        assert switched is True
        assert records[VAC_RB].value == pytest.approx(5e-6)  # leak override picked up


class TestLegacyFallbackForUnmodeledChannels:
    """Addresses partition (c) covers but machine.json doesn't define fall
    back to the same generic PV-taxonomy synthesis MockConnector itself uses,
    so mock and VA never disagree about an address neither has real data
    for."""

    def test_unmodeled_channel_gets_a_plausible_nonzero_value(self, source, records):
        source.poll_once()
        assert records[NOT_IN_MACHINE].value == pytest.approx(100.0)  # classify_pv's default kind

    def test_unmodeled_channel_value_is_stable_when_not_noisy(self, source, records):
        source.poll_once()
        first = records[NOT_IN_MACHINE].value
        source.poll_once()
        second = records[NOT_IN_MACHINE].value
        assert first == second

    def test_unmodeled_noisy_channel_varies_across_polls(self, engine, data_dir):
        noisy_channels = [
            *CHANNELS,
        ]
        noisy_channels[-1] = _channel(NOT_IN_MACHINE, record_type=RECORD_TYPE_ANALOG, noise=True)
        recs = {VAC_RB: FakeRecord(), FAULT_BI: FakeRecord(), NOT_IN_MACHINE: FakeRecord()}
        noisy_source = EngineSource(engine, noisy_channels, recs, data_dir, noise_level=0.05)

        values = set()
        for _ in range(20):
            noisy_source.poll_once()
            values.add(recs[NOT_IN_MACHINE].value)
        assert len(values) > 1, "expected noise to vary the value across polls"

    def test_unmodeled_binary_channel_falls_back_to_a_bool(self, engine, data_dir):
        """FAULT_BI IS defined in machine.json in this fixture, so exercise the
        fallback path directly with a fresh binary-typed unmodeled address."""
        bi_addr = "ZZTEST:UNKNOWN:02:STATUS:FAULT"
        channels = [*CHANNELS, _channel(bi_addr, record_type=RECORD_TYPE_BINARY, noise=False)]
        recs = {addr: FakeRecord() for addr in [VAC_RB, FAULT_BI, NOT_IN_MACHINE, bi_addr]}
        src = EngineSource(engine, channels, recs, data_dir)
        src.poll_once()
        assert isinstance(recs[bi_addr].value, bool)
