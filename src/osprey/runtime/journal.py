"""Write journal: the setpoint each address held before its first guarded write.

A guarded multi-write run pushes a :class:`Journal` onto a stack held in a
context variable. Device ``set`` paths go through :func:`guarded_write`, which
journals into every active level, so a nested level journals into the outer run
as well as its own. :func:`journaled_write` is the same journaling without the
guard: outside any level it only writes.

:func:`guarded_write` writes only inside
:func:`osprey.runtime.guarded_run.journaled_run`, which holds the control
target's run lock and keeps the run's :class:`DurableJournal` open. Anywhere
else it raises :class:`OspreyWriteRefused` and touches no channel. A level
pushed with :func:`push_journal` alone does not open that context; the two
stack functions stay out of ``__all__`` for that reason.

A journal keeps, per address, only the first value it sees; later writes to the
same address leave it untouched. It also keeps the longest write latency and the
longest gap between callback invocations, which a deadline guard uses to budget
the time a restore needs.

The :class:`DurableJournal` is a file beside the run lock that survives the
process. It starts with one header line (control target, generation, identity,
pid, started) and gains one line per address, written and fsync'd before that
address is first written, so a run killed part-way leaves on disk every
setpoint it displaced. A run starts only on an empty file: a dead run's records
are restored and cleared by the lock before a new header is written, never
overwritten and never followed by a second header. :func:`read_pending_journal`
reads what a killed run left; its last record may be torn by the kill and is
then ignored.

This module is stdlib-only at import time.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, ItemsView, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

__all__ = [
    "DurableJournal",
    "Journal",
    "OspreyRestoreIncomplete",
    "OspreyStaleJournal",
    "OspreyWriteFailed",
    "OspreyWriteRefused",
    "PendingJournal",
    "active_journals",
    "guarded_write",
    "journaled_write",
    "read_map",
    "read_pending_journal",
]

#: Why :func:`guarded_write` refuses outside a journaled guarded run.
_NOT_JOURNALED_REASON = "guarded writes run only inside a journaled guarded run"

_T = TypeVar("_T")

_clock: Callable[[], float] = time.monotonic

_ACTIVE: ContextVar[tuple[Journal, ...]] = ContextVar("osprey_runtime_journals", default=())

#: The outermost level of the journaled guarded run open in this context, or
#: ``None``. Set only by :func:`_journaled_level`, which only
#: :func:`osprey.runtime.guarded_run.journaled_run` enters, with the run lock
#: held and the durable journal open.
_JOURNALED: ContextVar[Journal | None] = ContextVar("osprey_runtime_journaled", default=None)


class OspreyWriteRefused(Exception):
    """A write was refused and no value of the call was written.

    Treat it as a stop, not a retry.

    Attributes:
        reason: One-line reason taken from the refusal.
        channel_address: The refused channel, when the refusal names one.
    """

    def __init__(self, reason: str, channel_address: str | None = None) -> None:
        self.reason = reason
        self.channel_address = channel_address
        where = f" to '{channel_address}'" if channel_address else ""
        super().__init__(f"OSPREY refused the write{where}: {reason}")


class OspreyWriteFailed(Exception):
    """A write was attempted but its outcome was not confirmed.

    Attributes:
        reason: One-line reason taken from the connector's failure.
        channel_address: The failed channel, when the failure names one.
    """

    def __init__(self, reason: str, channel_address: str | None = None) -> None:
        self.reason = reason
        self.channel_address = channel_address
        where = f" to '{channel_address}'" if channel_address else ""
        super().__init__(f"OSPREY write{where} failed: {reason}")


class Journal:
    """First-seen setpoints, and the write and callback timing, of one guard.

    Attributes:
        max_write_latency: The longest write latency noted, ``None`` before the first.
        last_callback: When the latest callback invocation happened, ``None`` before one.
        max_callback_gap: The longest gap between consecutive callback invocations,
            ``None`` before two.
    """

    def __init__(self, sink: Callable[[str, Any], None] | None = None) -> None:
        """Create an empty journal.

        Args:
            sink: Called with ``(address, value)`` for each address this journal
                records, before the value is kept; an exception it raises leaves
                the address unrecorded and propagates.
        """
        self._values: dict[str, Any] = {}
        self._sink = sink
        self.max_write_latency: float | None = None
        self.last_callback: float | None = None
        self.max_callback_gap: float | None = None

    @property
    def values(self) -> dict[str, Any]:
        """Journaled value per address, in first-write order (a copy)."""
        return dict(self._values)

    @property
    def addresses(self) -> tuple[str, ...]:
        """Journaled addresses in first-write order."""
        return tuple(self._values)

    def items(self) -> ItemsView[str, Any]:
        """Journaled ``(address, value)`` pairs in first-write order (a live view)."""
        return self._values.items()

    def __contains__(self, address: object) -> bool:
        return address in self._values

    def record(self, addresses: Sequence[str], values_before: Sequence[Any]) -> None:
        """Journal each address's value unless it is already journaled.

        Args:
            addresses: Addresses about to be (or just) written.
            values_before: The value each address held before the write, paired
                one to one with ``addresses``.

        Raises:
            ValueError: If ``addresses`` and ``values_before`` differ in length.
        """
        for address, value in zip(addresses, values_before, strict=True):
            if address in self._values:
                continue
            if self._sink is not None:
                self._sink(address, value)
            self._values[address] = value

    def note_latency(self, latency_s: float) -> None:
        """Note the duration of one write."""
        if self.max_write_latency is None or latency_s > self.max_write_latency:
            self.max_write_latency = latency_s

    def note_callback(self, timestamp: float | None = None) -> None:
        """Timestamp one callback invocation (the module clock when not given)."""
        now = _clock() if timestamp is None else timestamp
        if self.last_callback is not None:
            gap = now - self.last_callback
            if self.max_callback_gap is None or gap > self.max_callback_gap:
                self.max_callback_gap = gap
        self.last_callback = now


def active_journals() -> tuple[Journal, ...]:
    """The active journals, outermost first; empty outside a guard."""
    return _ACTIVE.get()


def push_journal(journal: Journal | None = None) -> Journal:
    """Activate ``journal`` (a new one when not given) as the innermost level.

    A pushed level journals writes; it does not let :func:`guarded_write` write.
    """
    j = Journal() if journal is None else journal
    _ACTIVE.set((*_ACTIVE.get(), j))
    return j


def pop_journal(journal: Journal | None = None) -> Journal:
    """Deactivate the innermost journal and return it.

    Args:
        journal: When given, must be the innermost journal.

    Raises:
        RuntimeError: If no journal is active, or ``journal`` is not the innermost.
    """
    stack = _ACTIVE.get()
    if not stack:
        raise RuntimeError("pop_journal called with no active journal")
    if journal is not None and stack[-1] is not journal:
        raise RuntimeError("pop_journal: journal is not the innermost active level")
    _ACTIVE.set(stack[:-1])
    return stack[-1]


def _journaled() -> Journal | None:
    """The outermost level of the journaled guarded run open here, ``None`` outside one."""
    return _JOURNALED.get()


@contextmanager
def _journaled_level(journal: Journal) -> Iterator[Journal]:
    """Push ``journal`` as the outermost level of a journaled guarded run.

    Entered only by :func:`osprey.runtime.guarded_run.journaled_run`, with the
    target's run lock held and ``journal`` sinking into the open durable journal.

    Raises:
        RuntimeError: A level is already active, so ``journal`` would not be the
            outermost.
    """
    if _ACTIVE.get():
        raise RuntimeError("a journaled guarded run must be the outermost journal level")
    push_journal(journal)
    token = _JOURNALED.set(journal)
    try:
        yield journal
    finally:
        _JOURNALED.reset(token)
        pop_journal(journal)


def read_map(addresses: Sequence[str]) -> dict[str, Any]:
    """Read ``addresses`` in one ``read_channels`` call, keyed by address.

    Raises:
        Exception: Whatever ``osprey.runtime.read_channels`` raised.
    """
    import osprey.runtime

    return dict(zip(addresses, osprey.runtime.read_channels(addresses), strict=True))


def journaled_write(addresses: Sequence[str], write: Callable[[], _T]) -> _T:
    """Run ``write``, journaling ``addresses`` into every active level first.

    Outside a guard this only calls ``write``. Inside one, the addresses some
    active journal does not hold yet are read in one ``read_channels`` call before
    the write, and each level records those it lacks. The write's latency is
    noted in every level whether or not it raises, since a failed write may
    already have moved a channel.

    Args:
        addresses: Every address the write touches.
        write: Performs the write; its result is returned.

    Raises:
        ChannelReadFailedError: If the pre-write read fails; nothing is written.
    """
    journals = _ACTIVE.get()
    if not journals:
        return write()
    missing = [a for a in dict.fromkeys(addresses) if any(a not in j for j in journals)]
    if missing:
        before = read_map(missing)
        for j in journals:
            lacking = [a for a in missing if a not in j]
            j.record(lacking, [before[a] for a in lacking])
    start = _clock()
    try:
        return write()
    finally:
        latency = _clock() - start
        for j in journals:
            j.note_latency(latency)


def guarded_write(
    addresses: Sequence[str],
    write: Callable[..., object],
    map_write: Callable[[Exception], Exception],
    *,
    map_read: Callable[[Exception, list[str]], Exception] | None = None,
    confirm: bool | None = None,
) -> None:
    """Journal ``addresses``, then write, inside a journaled guarded run only.

    The one check that keeps a device write out of every context but
    :func:`osprey.runtime.guarded_run.journaled_run`: anywhere else, including
    under a level pushed with :func:`push_journal` or under the run lock alone,
    nothing is read or written and :class:`OspreyWriteRefused` is raised.

    Args:
        addresses: Every address the write touches.
        write: The runtime write. It is called with ``confirm=confirm`` when
            ``confirm`` is given and with no keyword otherwise, so each channel
            resolves its own confirm default.
        map_write: Maps an exception ``write`` raised onto the exception raised
            in its place.
        map_read: Maps an exception the pre-write read raised, with the
            addresses asked for, onto the exception raised in its place; the
            read's own exception propagates when not given.
        confirm: Passed to ``write`` when not ``None``.

    Raises:
        OspreyWriteRefused: No journaled guarded run is open; nothing was read
            or written.
        Exception: ``map_write``'s mapping of a write failure, or the pre-write
            read's failure (mapped by ``map_read`` when given); nothing is
            written after a read failure.
    """
    if _journaled() is None:
        raise OspreyWriteRefused(_NOT_JOURNALED_REASON)
    kwargs: dict[str, Any] = {} if confirm is None else {"confirm": confirm}
    attempted = False

    def mapped_write() -> None:
        nonlocal attempted
        attempted = True
        try:
            write(**kwargs)
        except Exception as exc:
            raise map_write(exc) from exc

    try:
        journaled_write(addresses, mapped_write)
    except Exception as exc:
        if attempted or map_read is None:
            raise
        raise map_read(exc, list(addresses)) from exc


@dataclass(frozen=True)
class PendingJournal:
    """What a killed guarded run left in its durable journal.

    Attributes:
        target: The control target the run was stamped with, ``None`` unstamped.
        generation: The generation it was stamped with, ``None`` when unknown.
        identity: The identity the run acted as; recorded, never compared.
        pid: The run's process id; informative only, since pids are recycled.
        started: When the run started (ISO-8601), ``None`` when unknown.
        values: The setpoint each address held before the run first wrote it,
            in record order.
    """

    target: str | None
    generation: int | None
    identity: str | None
    pid: int | None
    started: str | None
    values: dict[str, Any]


class OspreyStaleJournal(Exception):
    """A dead run's journal is for another control target; this run did not start.

    The journal is left in place.

    Attributes:
        path: The journal file.
        target: The control target the dead run was stamped with.
        generation: The generation it was stamped with.
        addresses: The journaled addresses, in record order.
    """

    def __init__(
        self,
        pending: PendingJournal,
        path: Path,
        *,
        target: str | None,
        generation: int | None,
    ) -> None:
        self.path = str(path)
        self.target = pending.target
        self.generation = pending.generation
        self.addresses = tuple(pending.values)
        who = "unknown" if pending.pid is None else str(pending.pid)
        super().__init__(
            f"a dead run (pid {who}) left the journal {path} for target {pending.target!r} "
            f"generation {pending.generation}, which this run (target {target!r} generation "
            f"{generation}) cannot replay; it holds {len(self.addresses)} addresses: "
            f"{', '.join(self.addresses)}; call any guarded tool under approval: its prompt "
            "lists and restores these setpoints."
        )


class OspreyRestoreIncomplete(Exception):
    """A dead run's journal could not be fully restored; this run did not start.

    The journal is left byte-unchanged, so the next run lists and restores it
    again.

    Attributes:
        path: The journal file.
        entries: ``(address, reason, value left at)`` for every address the
            restore refused or failed, refused first; the value is ``None``
            when nothing confirms where the channel stands.
    """

    def __init__(
        self,
        path: Path,
        refused: Sequence[tuple[str, str, Any]],
        failed: Sequence[tuple[str, str]],
    ) -> None:
        self.path = str(path)
        self.entries: tuple[tuple[str, str, Any], ...] = (
            *((address, reason, value) for address, reason, value in refused),
            *((address, reason, None) for address, reason in failed),
        )
        named = "; ".join(
            f"{address}: {reason} (left at {'unknown' if value is None else repr(value)})"
            for address, reason, value in self.entries
        )
        super().__init__(
            f"the restore of {path} left {len(self.entries)} setpoints displaced, so this "
            f"run did not start: {named}"
        )


def _jsonable(value: Any) -> Any:
    """``value`` as plain JSON data: numpy scalars and arrays become Python values."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return value


def _optional(value: Any, kind: type) -> Any:
    """``value`` when it is a ``kind`` (never a bool), else ``None``."""
    if isinstance(value, bool) or not isinstance(value, kind):
        return None
    return value


def read_pending_journal(path: Path) -> PendingJournal | None:
    """Read the journal at ``path``; ``None`` when it holds no record.

    A missing, empty or header-only file holds no record. A last line that is not
    newline-terminated or not a record is taken as torn by a kill and ignored.

    Raises:
        OSError: The file exists but could not be read.
        ValueError: A line other than the last is not a header or record.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    complete = raw.split(b"\n")[:-1]  # anything after the last newline is torn
    parsed: list[Any] = []
    for index, line in enumerate(complete):
        try:
            parsed.append(json.loads(line))
        except ValueError:
            if index == len(complete) - 1:
                break
            raise ValueError(f"journal {path}: line {index + 1} is unreadable") from None
    if (
        not parsed
        or not isinstance(parsed[0], dict)
        or not isinstance(parsed[0].get("header"), dict)
    ):
        return None
    header = parsed[0]["header"]
    values: dict[str, Any] = {}
    for index, record in enumerate(parsed[1:], start=2):
        address = record.get("address") if isinstance(record, dict) else None
        if not isinstance(address, str) or "value" not in record:
            if index == len(parsed):
                break
            raise ValueError(f"journal {path}: line {index} is not a record")
        values.setdefault(address, record["value"])
    if not values:
        return None
    return PendingJournal(
        target=_optional(header.get("target"), str),
        generation=_optional(header.get("generation"), int),
        identity=_optional(header.get("identity"), str),
        pid=_optional(header.get("pid"), int),
        started=_optional(header.get("started"), str),
        values=values,
    )


class DurableJournal:
    """The on-disk journal of the outermost guarded run.

    Every write is followed by ``fsync``, so each line is on disk before the call
    returns. Use :meth:`record` as a :class:`Journal` sink.
    """

    def __init__(self, path: Path) -> None:
        """Open (creating when missing) the journal at ``path``; its content is kept."""
        self.path = path
        created = not path.exists()
        self._fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o664)
        if created:
            os.fchmod(self._fd, 0o664)
            _fsync_directory(path.parent)

    def _append(self, payload: dict[str, Any]) -> None:
        os.write(self._fd, (json.dumps(payload, allow_nan=True) + "\n").encode("utf-8"))
        os.fsync(self._fd)

    def start(
        self,
        *,
        target: str | None,
        generation: int | None,
        identity: str,
        pid: int,
        started: str,
    ) -> None:
        """Write the header of a new run into the empty file.

        Raises:
            RuntimeError: The file is not empty; nothing is written. A dead
                run's records are restored and cleared by the run lock, never
                followed by a second header.
        """
        if os.fstat(self._fd).st_size:
            raise RuntimeError(f"journal {self.path} is not empty; a run starts on an empty one")
        self._append(
            {
                "header": {
                    "target": target,
                    "generation": generation,
                    "identity": identity,
                    "pid": pid,
                    "started": started,
                }
            }
        )

    def record(self, address: str, value: Any) -> None:
        """Append and fsync the setpoint ``address`` held before its first write.

        Raises:
            TypeError: ``value`` cannot be written as JSON.
            OSError: The line could not be written or synced.
        """
        self._append({"address": address, "value": _jsonable(value)})

    def clear(self) -> None:
        """Truncate the file to empty and fsync it."""
        os.ftruncate(self._fd, 0)
        os.fsync(self._fd)

    def close(self) -> None:
        """Close the file descriptor."""
        os.close(self._fd)


def _fsync_directory(directory: Path) -> None:
    """Sync ``directory`` so a newly created entry in it survives a crash."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
