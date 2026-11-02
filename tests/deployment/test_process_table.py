"""The process-table reader, on the host's real table and on a faked psutil."""

from __future__ import annotations

import os
import subprocess
import sys

import psutil
import pytest

from osprey.deployment import process_table

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="no process groups on Windows"),
]

# An epoch second the faked processes start from.
_SECOND = 1_700_000_000.0


def _failing_reads(exc: BaseException):
    """A ``process_iter`` that fails with *exc* once iterated."""

    def process_iter(*args, **kwargs):
        raise exc
        yield  # a generator, as ``psutil.process_iter`` is

    return process_iter


@pytest.mark.parametrize(
    ("failure", "retryable"),
    [
        (psutil.Error("x"), True),
        (OSError("x"), True),
        (psutil.AccessDenied(), False),
        (FileNotFoundError("x"), False),
    ],
)
def test_an_unreadable_table_raises_naming_the_cause(monkeypatch, failure, retryable):
    monkeypatch.setattr(process_table.psutil, "process_iter", _failing_reads(failure))

    with pytest.raises(process_table.ProcessTableUnreadable) as caught:
        process_table.snapshot()

    assert caught.value.cause is failure
    assert caught.value.retryable is retryable


class _FakeProcess:
    def __init__(
        self, pid: int, ppid: int, command: list[str], fails: BaseException | None = None
    ) -> None:
        self.fails = fails
        self.info = {
            "pid": pid,
            "ppid": ppid,
            "status": psutil.STATUS_SLEEPING,
            "create_time": _SECOND + pid,
            "cmdline": command,
            "name": command[0],
        }

    def as_dict(self, attrs=None, ad_value=None) -> dict:  # noqa: ARG002 - psutil's signature
        if self.fails is not None:
            raise self.fails
        return dict(self.info)


@pytest.mark.parametrize("refusal", [ProcessLookupError, PermissionError])
def test_a_process_vanishing_mid_read_is_absent_not_fatal(monkeypatch, refusal):
    processes = [
        _FakeProcess(4201, 1, ["python", "magnet_scan.py"]),
        _FakeProcess(4202, 1, ["sleep", "60"]),
        _FakeProcess(4203, 1, ["python", "orbit_poll.py"]),
    ]

    def getpgid(pid: int) -> int:
        if pid == 4202:
            raise refusal(pid)
        return pid

    monkeypatch.setattr(process_table.psutil, "process_iter", lambda *a, **k: iter(processes))
    monkeypatch.setattr(process_table.os, "getpgid", getpgid)

    rows = process_table.snapshot()

    assert set(rows) == {4201, 4203}
    assert rows[4201].command == "python magnet_scan.py"
    assert rows[4201].pgid == 4201


def _iterating(processes: list[_FakeProcess]):
    """A ``process_iter`` that reads each row inside the generator, as psutil 7.2.2 does."""

    def process_iter(attrs=None, ad_value=None):
        for process in processes:
            if attrs is not None:
                process.info = process.as_dict(attrs, ad_value)
            yield process

    return process_iter


def test_a_row_whose_command_line_raises_is_absent_not_fatal(monkeypatch):
    processes = [
        _FakeProcess(4201, 1, ["python", "magnet_scan.py"]),
        _FakeProcess(4202, 1, ["sleep", "60"], fails=SystemError("proc_cmdline")),
        _FakeProcess(4203, 1, ["python", "orbit_poll.py"]),
    ]
    monkeypatch.setattr(process_table.psutil, "process_iter", _iterating(processes))
    monkeypatch.setattr(process_table.os, "getpgid", lambda pid: pid)

    rows = process_table.snapshot()

    assert set(rows) == {4201, 4203}


def test_an_enumeration_that_raises_is_unreadable(monkeypatch):
    failure = SystemError("x")
    monkeypatch.setattr(process_table.psutil, "process_iter", _failing_reads(failure))

    with pytest.raises(process_table.ProcessTableUnreadable) as caught:
        process_table.snapshot()

    assert caught.value.cause is failure


def test_reading_the_table_never_forks(monkeypatch):
    def no_fork(*args, **kwargs):
        raise AssertionError("reading the process table forked")

    monkeypatch.setattr(subprocess, "Popen", no_fork)
    monkeypatch.setattr(os, "fork", no_fork)

    rows = process_table.snapshot()

    assert rows[os.getpid()].pgid == os.getpgrp()


def test_a_process_keeps_its_start_across_reads():
    first = process_table.snapshot()[os.getpid()]
    second = process_table.snapshot()[os.getpid()]

    assert first.started == second.started


def _failing_then(monkeypatch, failures: list[Exception]) -> list[None]:
    """Patch ``snapshot`` to raise each of *failures* in turn, then answer; return the calls."""
    calls: list[None] = []

    def flaky():
        calls.append(None)
        if failures:
            raise process_table.ProcessTableUnreadable(failures.pop(0))
        return {}

    monkeypatch.setattr(process_table, "snapshot", flaky)
    monkeypatch.setattr(process_table, "_READ_RETRY_S", 0.0)
    return calls


def test_snapshot_within_retries_a_failed_read(monkeypatch):
    calls = _failing_then(monkeypatch, [psutil.Error("busy")])

    assert process_table.snapshot_within(5.0) == {}
    assert len(calls) == 2


@pytest.mark.parametrize(
    "missing", [FileNotFoundError("proc"), PermissionError("proc"), psutil.AccessDenied()]
)
def test_snapshot_within_does_not_retry_an_unreadable_table(monkeypatch, missing):
    calls = _failing_then(monkeypatch, [missing, missing])

    with pytest.raises(process_table.ProcessTableUnreadable) as caught:
        process_table.snapshot_within(5.0)

    assert caught.value.cause is missing
    assert len(calls) == 1


def test_snapshot_within_raises_once_its_deadline_passes(monkeypatch):
    calls = _failing_then(monkeypatch, [psutil.Error("busy"), psutil.Error("busy")])

    with pytest.raises(process_table.ProcessTableUnreadable):
        process_table.snapshot_within(0.0)

    assert len(calls) == 1
