"""``settle`` waits for each setpoint's readback to come within the facility's stated tolerance.

The facility file is planted at ``build/facility.json`` of a throwaway render
(``guarded_repo``); the clock and the pause between polls are replaced, so a
deadline passes without waiting for it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import osprey.runtime
import pyaml_cs_osprey.device as device_mod
from osprey.errors import ChannelReadFailedError
from pyaml_cs_osprey.device import SETTLE_TIMEOUT_S, settle
from pyaml_cs_osprey.errors import OspreyReadFailed, OspreyWriteFailed


def plant_facility(root: Path, channels: dict[str, tuple[str, float]]) -> Path:
    """Write ``build/facility.json`` stating ``setpoint -> (readback, absolute tolerance)``."""
    path = root / "build" / "facility.json"
    records = [
        {"id": setpoint, "role": "setpoint", "pair": pair, "tolerance": {"absolute": bound}}
        for setpoint, (pair, bound) in channels.items()
    ]
    path.write_text(json.dumps({"channels": records}), encoding="utf-8")
    return path


class Clock:
    """A clock that moves only when the settle loop pauses; every pause is recorded."""

    def __init__(self) -> None:
        self.now = 0.0
        self.pauses: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.pauses.append(seconds)
        self.now += seconds

    @classmethod
    def install(cls, monkeypatch: pytest.MonkeyPatch) -> Clock:
        """A fresh clock serving the settle loop for the test."""
        fake = cls()
        monkeypatch.setattr(device_mod, "_clock", fake)
        monkeypatch.setattr(device_mod, "_sleep", fake.sleep)
        return fake


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    return Clock.install(monkeypatch)


class LaggingReadbacks:
    """``read_channels`` over readbacks that reach their target after ``lag`` reads each."""

    def __init__(self, start: dict[str, float], target: dict[str, float], lag: int) -> None:
        self.start = start
        self.target = target
        self.lag = lag
        self.batches: list[list[str]] = []

    def __call__(self, addresses: Sequence[str], *, timeout: float | None = None) -> list[Any]:
        del timeout
        self.batches.append(list(addresses))
        arrived = len(self.batches) > self.lag
        return [(self.target if arrived else self.start)[a] for a in addresses]


def test_settle_polls_the_readbacks_until_each_is_inside_its_tolerance(
    guarded_repo: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    plant_facility(guarded_repo, {"Q1:SP": ("Q1:RB", 0.05), "Q2:SP": ("Q2:RB", 0.05)})
    reads = LaggingReadbacks(
        start={"Q1:RB": 1.0, "Q2:RB": 1.0}, target={"Q1:RB": 2.04, "Q2:RB": 2.96}, lag=3
    )
    monkeypatch.setattr(osprey.runtime, "read_channels", reads)

    settle({"Q1:SP": 2.0, "Q2:SP": 3.0})

    assert reads.batches == [["Q1:RB", "Q2:RB"]] * 4, "one batch read per poll"
    assert clock.pauses == [device_mod._SETTLE_POLL_S] * 3


def test_settle_past_its_timeout_fails_naming_the_unsettled_setpoint(
    guarded_repo: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    plant_facility(guarded_repo, {"Q1:SP": ("Q1:RB", 0.05), "Q2:SP": ("Q2:RB", 0.05)})
    reads = LaggingReadbacks(
        start={"Q1:RB": 2.0, "Q2:RB": 1.5}, target={"Q1:RB": 2.0, "Q2:RB": 1.5}, lag=0
    )
    monkeypatch.setattr(osprey.runtime, "read_channels", reads)

    with pytest.raises(OspreyWriteFailed) as info:
        settle({"Q1:SP": 2.0, "Q2:SP": 3.0})

    failure = info.value
    assert failure.channel_address == "Q2:SP"
    assert "Q2:RB" in failure.reason
    assert "0.05" in failure.reason
    assert "3" in failure.reason
    assert "1.5" in failure.reason
    assert clock.now >= SETTLE_TIMEOUT_S


def test_settle_reads_nothing_for_setpoints_with_no_stated_tolerance(
    guarded_repo: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    plant_facility(guarded_repo, {"Q1:SP": ("Q1:RB", 0.05)})
    reads = LaggingReadbacks(start={"Q1:RB": 2.0}, target={"Q1:RB": 2.0}, lag=0)
    monkeypatch.setattr(osprey.runtime, "read_channels", reads)

    settle({"Q9:SP": 7.0})

    assert reads.batches == []
    assert clock.pauses == []


def test_settle_without_a_facility_file_reads_nothing(
    guarded_repo: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not (guarded_repo / "build" / "facility.json").exists()
    reads = LaggingReadbacks(start={}, target={}, lag=0)
    monkeypatch.setattr(osprey.runtime, "read_channels", reads)

    settle({"Q1:SP": 2.0})

    assert reads.batches == []
    assert clock.pauses == []


def test_a_failed_readback_read_is_a_read_failure(
    guarded_repo: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    del clock
    plant_facility(guarded_repo, {"Q1:SP": ("Q1:RB", 0.05)})

    def unreadable(addresses: Sequence[str], *, timeout: float | None = None) -> list[Any]:
        del timeout
        raise ChannelReadFailedError(list(addresses))

    monkeypatch.setattr(osprey.runtime, "read_channels", unreadable)

    with pytest.raises(OspreyReadFailed) as info:
        settle({"Q1:SP": 2.0})
    assert info.value.addresses == ["Q1:RB"]
