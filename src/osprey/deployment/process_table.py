"""The host's process table, read in-process.

The table is read with psutil, so a read never forks and has no timeout: a
busy host makes it slower, never unknown. A read that fails raises
:class:`ProcessTableUnreadable`; a caller that decides to end or keep
something never takes an unreadable table for an empty one.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import psutil

# What a read asks psutil for, per process; the group comes from ``os.getpgid``.
_ATTRS = ["pid", "ppid", "status", "create_time", "cmdline", "name"]
# The status of an exited process nobody has reaped.
_ZOMBIE: str = psutil.STATUS_ZOMBIE
# Interval between attempts to read an unreadable table.
_READ_RETRY_S = 0.25


@dataclass(frozen=True)
class ProcessRow:
    """One process as the process table reported it."""

    pid: int
    ppid: int
    pgid: int
    started: float  # epoch seconds
    command: str
    state: str  # the psutil status; ``psutil.STATUS_ZOMBIE`` is an exited process nobody has reaped

    @property
    def exited(self) -> bool:
        return self.state == _ZOMBIE


class ProcessTableUnreadable(RuntimeError):
    """The process table could not be read, so what is running is unknown.

    ``cause`` is the failure. ``retryable`` is false when another read cannot
    answer either: the host has no process table, or reading it is refused.
    """

    def __init__(self, cause: BaseException) -> None:
        super().__init__(f"cannot read the process table: {cause}")
        self.cause = cause
        self.retryable = not isinstance(
            cause, FileNotFoundError | PermissionError | psutil.AccessDenied
        )


def snapshot() -> dict[int, ProcessRow]:
    """Return every process on the host by pid.

    A row that cannot be read because its process is exiting, or whose group
    or start can no longer be read, is absent. Only a failure to enumerate the
    table itself raises.

    Raises:
        ProcessTableUnreadable: The process table could not be enumerated.
    """
    rows: dict[int, ProcessRow] = {}
    try:
        for process in psutil.process_iter():
            try:
                info = process.as_dict(_ATTRS, ad_value=None)
            # SystemError covers a psutil defect on macOS, where reading the command
            # line of an exiting process raises it from the C extension; the catch
            # is dropped once psutil fixes it.
            except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied, SystemError):
                continue
            pid, ppid, started = info["pid"], info["ppid"], info["create_time"]
            if ppid is None or started is None:
                continue
            try:
                pgid = os.getpgid(pid)
            except OSError:
                continue
            cmdline = info["cmdline"]
            command = " ".join(cmdline) if cmdline else info["name"] or ""
            rows[pid] = ProcessRow(pid, ppid, pgid, started, command, info["status"] or "")
    except (psutil.Error, OSError, SystemError) as exc:
        raise ProcessTableUnreadable(exc) from exc
    return rows


def snapshot_within(within: float) -> dict[int, ProcessRow]:
    """Return :func:`snapshot`, retrying a retryable failure for up to *within* seconds.

    Attempts are :data:`_READ_RETRY_S` apart and none starts once *within*
    seconds have passed since the first began, so the call blocks for at most
    *within* plus one read.

    Raises:
        ProcessTableUnreadable: The last failure, once no attempt is left or
            at once when it is not retryable.
    """
    deadline = time.monotonic() + within
    while True:
        try:
            return snapshot()
        except ProcessTableUnreadable as exc:
            if not exc.retryable or time.monotonic() + _READ_RETRY_S > deadline:
                raise
        time.sleep(_READ_RETRY_S)


__all__ = ["ProcessRow", "ProcessTableUnreadable", "snapshot", "snapshot_within"]
