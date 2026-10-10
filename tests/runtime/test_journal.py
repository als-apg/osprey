"""The guarded-run write journal: first-seen setpoints, the durable file, the write guard."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

import osprey.runtime
from osprey.errors import ChannelLimitsViolationError, ChannelReadFailedError
from osprey.runtime.journal import (
    DurableJournal,
    Journal,
    OspreyRestoreIncomplete,
    OspreyStaleJournal,
    OspreyWriteFailed,
    OspreyWriteRefused,
    PendingJournal,
    active_journals,
    guarded_write,
    journaled_write,
    pop_journal,
    push_journal,
    read_pending_journal,
)
from osprey_connectors.control_system.limits_validator import (
    ChannelLimitsConfig,
    LimitsValidator,
)
from osprey_connectors.types import LIMITS_MODE_OPTIONAL


class _Channels:
    """A dict of channel values standing in for ``osprey.runtime``'s reads and writes."""

    def __init__(self, values: dict[str, Any]) -> None:
        self.values = dict(values)
        self.reads: list[list[str]] = []
        self.writes: list[tuple[str, Any]] = []

    def read_channels(self, addresses: list[str], **_kwargs: Any) -> list[Any]:
        self.reads.append(list(addresses))
        return [self.values[a] for a in addresses]

    def write(self, address: str, value: Any) -> None:
        self.writes.append((address, value))
        self.values[address] = value


@pytest.fixture
def channels(monkeypatch: pytest.MonkeyPatch) -> _Channels:
    fake = _Channels({"A": 1.0, "B": 2.0})
    monkeypatch.setattr(osprey.runtime, "read_channels", fake.read_channels)
    return fake


def _header(**overrides: Any) -> dict[str, Any]:
    header = {
        "target": "live",
        "generation": 4,
        "identity": "alice",
        "pid": 4242,
        "started": "2026-01-01T00:00:00Z",
    }
    header.update(overrides)
    return {"header": header}


def _write_lines(path: Path, *lines: dict[str, Any], tail: str = "") -> None:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines) + tail, encoding="utf-8")


def test_a_journal_keeps_the_first_value_per_address() -> None:
    journal = Journal()
    journal.record(["A", "B"], [1.0, 2.0])
    journal.record(["A"], [9.0])

    assert journal.values == {"A": 1.0, "B": 2.0}
    assert journal.addresses == ("A", "B")


def test_journaled_write_outside_a_level_only_writes(channels: _Channels) -> None:
    journaled_write(["A"], lambda: channels.write("A", 5.0))

    assert channels.reads == []
    assert channels.writes == [("A", 5.0)]


def test_journaled_write_journals_into_every_level(channels: _Channels) -> None:
    outer = push_journal()
    inner = push_journal()
    try:
        journaled_write(["A"], lambda: channels.write("A", 5.0))
        journaled_write(["A"], lambda: channels.write("A", 6.0))
    finally:
        pop_journal(inner)
        pop_journal(outer)

    assert outer.values == inner.values == {"A": 1.0}
    assert channels.reads == [["A"]]
    assert active_journals() == ()


def test_guarded_write_outside_a_journaled_run_refuses(channels: _Channels) -> None:
    with pytest.raises(OspreyWriteRefused, match="journaled guarded run"):
        guarded_write(["A"], lambda: channels.write("A", 5.0), lambda exc: exc)

    assert channels.reads == []
    assert channels.writes == []


def test_a_pushed_journal_does_not_open_guarded_write(channels: _Channels) -> None:
    level = push_journal()
    try:
        with pytest.raises(OspreyWriteRefused):
            guarded_write(["A"], lambda: channels.write("A", 5.0), lambda exc: exc)
    finally:
        pop_journal(level)

    assert channels.reads == []
    assert channels.writes == []
    assert level.values == {}


def test_the_stack_functions_stay_out_of_all() -> None:
    import osprey.runtime.journal as journal

    assert "push_journal" not in journal.__all__
    assert "pop_journal" not in journal.__all__
    assert {"guarded_write", "journaled_write", "active_journals", "Journal"} <= set(
        journal.__all__
    )


def test_the_write_errors_carry_reason_and_channel() -> None:
    refused = OspreyWriteRefused("above max", "A")
    failed = OspreyWriteFailed("unconfirmed", None)

    assert isinstance(refused, Exception) and isinstance(failed, Exception)
    assert (refused.reason, refused.channel_address) == ("above max", "A")
    assert str(refused) == "OSPREY refused the write to 'A': above max"
    assert str(failed) == "OSPREY write failed: unconfirmed"


def test_start_writes_one_header_into_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "journal"
    durable = DurableJournal(path)
    try:
        durable.start(target="live", generation=4, identity="alice", pid=1, started="t")
        durable.record("A", 1.0)
    finally:
        durable.close()

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines == [
        {
            "header": {
                "target": "live",
                "generation": 4,
                "identity": "alice",
                "pid": 1,
                "started": "t",
            }
        },
        {"address": "A", "value": 1.0},
    ]


def test_start_never_writes_behind_a_dead_runs_records(tmp_path: Path) -> None:
    path = tmp_path / "journal"
    _write_lines(path, _header(), {"address": "A", "value": 1.0})
    before = path.read_bytes()

    durable = DurableJournal(path)
    try:
        with pytest.raises(RuntimeError, match="not empty"):
            durable.start(target="live", generation=4, identity="bob", pid=2, started="t")
    finally:
        durable.close()

    assert path.read_bytes() == before


def test_a_created_journal_is_group_writable_under_umask_022(tmp_path: Path) -> None:
    path = tmp_path / "journal"
    previous = os.umask(0o022)
    try:
        DurableJournal(path).close()
    finally:
        os.umask(previous)

    assert path.stat().st_mode & 0o060 == 0o060


def test_clear_empties_the_file(tmp_path: Path) -> None:
    path = tmp_path / "journal"
    _write_lines(path, _header(), {"address": "A", "value": 1.0})

    durable = DurableJournal(path)
    try:
        durable.clear()
        durable.start(target="live", generation=4, identity="bob", pid=2, started="t")
    finally:
        durable.close()

    assert read_pending_journal(path) is None
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_a_pending_journal_reads_header_and_first_values(tmp_path: Path) -> None:
    path = tmp_path / "journal"
    _write_lines(
        path,
        _header(),
        {"address": "A", "value": 1.0},
        {"address": "B", "value": 2.0},
        {"address": "A", "value": 7.0},
        tail='{"address": "C", "val',
    )

    pending = read_pending_journal(path)

    assert pending == PendingJournal(
        target="live",
        generation=4,
        identity="alice",
        pid=4242,
        started="2026-01-01T00:00:00Z",
        values={"A": 1.0, "B": 2.0},
    )


@pytest.mark.parametrize(
    "content",
    ["", json.dumps(_header()) + "\n", json.dumps(_header()) + '\n{"addr'],
    ids=["empty", "header-only", "torn-first-record"],
)
def test_no_record_is_no_pending_journal(tmp_path: Path, content: str) -> None:
    path = tmp_path / "journal"
    path.write_text(content, encoding="utf-8")

    assert read_pending_journal(path) is None
    assert read_pending_journal(tmp_path / "missing") is None


def test_a_corrupt_middle_line_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "journal"
    path.write_text(
        json.dumps(_header()) + "\nnot json\n" + json.dumps({"address": "A", "value": 1}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2 is unreadable"):
        read_pending_journal(path)


def test_the_stale_journal_names_the_dead_run_and_the_remedy(tmp_path: Path) -> None:
    pending = PendingJournal("live", 4, "alice", 4242, None, {"A": 1.0, "B": 2.0})
    path = tmp_path / "journal"

    exc = OspreyStaleJournal(pending, path, target="live", generation=5)

    assert (exc.target, exc.generation, exc.addresses) == ("live", 4, ("A", "B"))
    assert exc.path == str(path)
    text = str(exc)
    assert "pid 4242" in text
    assert "target 'live' generation 4" in text
    assert "A, B" in text
    assert text.endswith(
        "call any guarded tool under approval: its prompt lists and restores these setpoints."
    )
    assert "switch back" not in text


def test_the_incomplete_restore_names_each_address_reason_and_value(tmp_path: Path) -> None:
    path = tmp_path / "journal"

    exc = OspreyRestoreIncomplete(
        path, refused=[("A", "step too large", 3.5)], failed=[("B", "no readback")]
    )

    assert exc.entries == (("A", "step too large", 3.5), ("B", "no readback", None))
    assert exc.path == str(path)
    text = str(exc)
    assert "A: step too large (left at 3.5)" in text
    assert "B: no readback (left at unknown)" in text
    assert "did not start" in text


class _LimitedChannels:
    """Channels whose every write is first checked by a real ``LimitsValidator``."""

    def __init__(self, values: dict[str, Any], max_step: dict[str, float]) -> None:
        self.values = dict(values)
        self.writes: list[tuple[str, Any]] = []
        self.validator = LimitsValidator(
            {a: ChannelLimitsConfig(a, max_step=s, writable=True) for a, s in max_step.items()},
            {"mode": LIMITS_MODE_OPTIONAL},
        )

    def read_channels(self, addresses: list[str], **_kwargs: Any) -> list[Any]:
        return [self.values[a] for a in addresses]

    def write_channel(self, address: str, value: Any, **_kwargs: Any) -> None:
        self.validator.validate(address, value, read_current=self.values.__getitem__)
        self.writes.append((address, value))
        self.values[address] = value

    def channel_limits(self, address: str) -> ChannelLimitsConfig | None:
        return self.validator.limits.get(address)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> _LimitedChannels:
        for name in ("read_channels", "write_channel", "channel_limits"):
            monkeypatch.setattr(osprey.runtime, name, getattr(self, name))
        return self


def _journal_of(values: dict[str, Any]) -> Journal:
    journal = Journal()
    journal.record(list(values), list(values.values()))
    return journal


def test_a_restore_walks_back_in_max_step_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    from osprey.runtime.guarded_run import _restore

    fake = _LimitedChannels({"Q": 3.0, "S": 5.0}, {"Q": 1.0}).install(monkeypatch)

    report = _restore(_journal_of({"Q": 0.0, "S": 5.0}), aborted=True)

    assert fake.writes == [("Q", 2.0), ("Q", 1.0), ("Q", 0.0)]
    assert report.restored == ["Q"]
    assert report.unchanged == ["S"]
    assert report.refused == report.failed == []
    assert report.aborted is True


def test_a_refused_restore_names_the_value_it_left(monkeypatch: pytest.MonkeyPatch) -> None:
    from osprey.runtime.guarded_run import _restore

    fake = _LimitedChannels({"Q": 3.0}, {"Q": 1.0}).install(monkeypatch)

    def refuse_below_two(address: str, value: Any, **kwargs: Any) -> None:
        if value < 2.0:
            raise ChannelLimitsViolationError(address, value, "MIN_VALUE", "below 2.0")
        _LimitedChannels.write_channel(fake, address, value, **kwargs)

    monkeypatch.setattr(osprey.runtime, "write_channel", refuse_below_two)

    report = _restore(_journal_of({"Q": 0.0}), aborted=True)

    assert report.refused == [("Q", "below 2.0", 2.0)]
    assert report.restored == []


def test_an_unreadable_address_fails_and_the_rest_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osprey.runtime.guarded_run import _restore

    fake = _LimitedChannels({"Q": 3.0, "R": 1.0}, {}).install(monkeypatch)

    def read_channels(addresses: list[str], **_kwargs: Any) -> list[Any]:
        if "R" in addresses:
            raise ChannelReadFailedError(["R"])
        return [fake.values[a] for a in addresses]

    monkeypatch.setattr(osprey.runtime, "read_channels", read_channels)

    report = _restore(_journal_of({"Q": 0.0, "R": 0.0}), aborted=False)

    assert report.restored == ["Q"]
    assert [address for address, _reason in report.failed] == ["R"]
    assert fake.writes == [("Q", 0.0)]
