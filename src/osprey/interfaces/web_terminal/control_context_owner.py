"""The web terminal's one way to change the control-context record.

The record at ``<agent-data root>/control_target/control_context.json`` has a
single writer — its owner — and when a web terminal is running, that owner is
the web terminal. Three parts of the server need to change the record: the
posture route, the target route, and the owner task that claims the record and
consumes switch requests. All three go through :meth:`ControlContextOwner.mutate_record`,
and this module exists so that there is exactly one implementation of what
"changing the record" means.

**Why a primitive at all.** Every change is a read-modify-write: what gets
written depends on what is already there. Two of those interleaving is a lost
update — the second writer computes its candidate from the record as it was
before the first one wrote, and ``os.replace`` silently discards the first
change. The posture store's predecessor, ``persist_or_raise``, avoided that by
doing load, write and memory update synchronously on the event loop with no
``await`` between them: the loop itself was the lock. That worked, and it cost
the loop a disk round-trip on every toggle — acceptable for a few hundred
bytes, not acceptable for this record, whose owner also stats a directory of
per-server reports and probes liveness on the same path.

So the two properties are separated. An :class:`asyncio.Lock` provides the
serialisation the loop used to provide, and the read, the caller's change and
the write happen together inside **one** ``asyncio.to_thread`` job so none of
it runs on the loop. Both halves are load-bearing: a lock around the write
alone would still lose updates, because the losing read has already happened by
then; a thread hop without the lock would lose them for the same reason.

**Who may write.** The record names its owner, and a process that is no longer
the owner must not write it — a second web terminal may have started, or this
one's claim may have been taken over while it was idle. The check is made
inside the job, against the record just read, rather than against anything held
in memory: that is the point of re-reading. A caller that is *establishing*
ownership rather than exercising it (the claim) opts the check out with
``verify_owner=False``, and every write through this primitive stamps
``owner`` back to this process either way, so a caller cannot give ownership
away by handing back a record it read before the check.

The ownership *rules* — when a terminal may claim, when it must follow — are
not here. They belong to the owner task, which is the only place that knows
whether the current owner is alive. This module knows only "am I the owner the
record names", which is the question a write has to answer.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Generic, TypeAlias, TypeVar

from osprey_connectors.control_context import (
    ControlContext,
    Owner,
    read_record,
    record_path,
    write_record,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ContextOwnedElsewhere",
    "ContextOwnerError",
    "ContextStoreUnavailable",
    "ControlContextOwner",
    "Mutation",
    "Mutator",
]

T = TypeVar("T")


@dataclass(frozen=True)
class Mutation(Generic[T]):
    """What a mutation callable hands back: what to store, and what to return.

    Attributes:
        record: The record to write, or ``None`` to write nothing. ``None`` is
            an ordinary outcome rather than a failure — a switch the gate
            refuses on facts read inside the job, a posture toggle that asks
            for the posture already stored, an owner tick with no request to
            consume. Nothing changed, so nothing is written, and the file's
            signature does not move for the readers watching it.
        result: Returned to the caller of :meth:`ControlContextOwner.mutate_record`
            once the write has landed. This is how a route gets its response
            body — the stored posture, the gate's verdict — out of the worker
            thread.
    """

    record: ControlContext | None
    result: T

    @classmethod
    def unchanged(cls, result: T) -> Mutation[T]:
        """A mutation that writes nothing and answers *result*."""
        return cls(record=None, result=result)


#: A mutation: given the record as it is on disk (``None`` when there is no
#: readable record), decide what to store. Runs in a worker thread, so it must
#: be synchronous and must not touch the event loop or anything guarded by it.
Mutator: TypeAlias = Callable[[ControlContext | None], Mutation[T]]


class ContextOwnerError(RuntimeError):
    """A mutation did not happen. Nothing was written.

    Carries an *error* code alongside the message because the routes answer
    these as HTTP failures and the code is what the response body and the log
    line agree on — the same shape the posture store's
    ``PostureStoreUnavailable`` established.
    """

    def __init__(self, error: str, message: str) -> None:
        super().__init__(message)
        self.error = error
        self.message = message


class ContextStoreUnavailable(ContextOwnerError):
    """There is nowhere to write, or the write itself failed.

    Both spellings of "the record could not be stored" share one class because
    the operator-facing answer is the same either way: the gesture was refused
    and nothing changed. :attr:`ContextOwnerError.error` tells them apart —
    ``store_unavailable`` for an agent-data root that does not resolve,
    ``store_write_failed`` for a write that raised.
    """


class ContextOwnedElsewhere(ContextOwnerError):
    """This process is not the owner the record names, so it may not write.

    Attributes:
        owner: The owner the record names, so a refusal can say where the
            operator should go. ``None`` when there is no readable record at
            all — this process's claim is simply gone, and the owner task will
            try to take it again on its next tick.
    """

    def __init__(self, owner: Owner | None, message: str) -> None:
        super().__init__("context_owned_elsewhere", message)
        self.owner = owner


class ControlContextOwner:
    """One process's claim on the record, and the only way it changes it.

    Built once per server (at lifespan start, by the owner task) and held on
    ``app.state``. The lock is an instance attribute rather than a module
    global so that it belongs to the same event loop as the app that owns it —
    an ``asyncio.Lock`` binds to the loop it is first awaited on, and a
    module-level one would be shared across every app in a process.

    Args:
        identity: Who this process is in the record's ``owner`` field. Every
            write stamps it.
        path: Write this file instead of the one :func:`record_path` resolves.
            For a caller that has already resolved a root, and for tests. When
            omitted the path is resolved on **each** mutation, so an owner
            built before the agent-data root existed starts working as soon as
            it does.
    """

    def __init__(self, identity: Owner, *, path: Path | None = None) -> None:
        self._identity = identity
        self._path = path
        self._lock = asyncio.Lock()

    @property
    def identity(self) -> Owner:
        """Who this process is in the record it writes."""
        return self._identity

    async def mutate_record(self, fn: Mutator[T], *, verify_owner: bool = True) -> T:
        """Read the record, apply *fn* to it, and store what *fn* returns.

        The read, the call and the write are one unit: serialised against every
        other mutation on this owner, and run together in a worker thread. A
        mutation therefore sees the result of the mutation before it, and no
        part of the sequence touches the event loop.

        Args:
            fn: The change to make. Receives the record as it is on disk —
                ``None`` when there is none, or none that parses — and returns
                a :class:`Mutation`. Runs in a worker thread.
            verify_owner: Refuse unless the record already names this process
                as its owner. ``False`` for a claim, which is how a process
                *becomes* the owner. The write stamps this owner either way.

        Returns:
            The mutation's ``result``, once any write has landed.

        Raises:
            ContextStoreUnavailable: The agent-data root does not resolve, or
                the write failed. Nothing changed.
            ContextOwnedElsewhere: *verify_owner* was set and the record names
                a different owner, or no record could be read. *fn* was not
                called and nothing changed.
            Exception: Whatever *fn* raises, unchanged. Nothing is written.
        """
        async with self._lock:
            return await asyncio.to_thread(self._apply, fn, verify_owner)

    def _apply(self, fn: Mutator[T], verify_owner: bool) -> T:
        """The whole read-modify-write, in one worker thread. Never on the loop."""
        path = self._resolve_path()
        record = read_record(path=path)

        if verify_owner:
            self._require_ownership(record)

        mutation = fn(record)
        if mutation.record is None:
            return mutation.result

        try:
            write_record(replace(mutation.record, owner=self._identity), path=path)
        except Exception as exc:  # noqa: BLE001 — reported to the operator as a 503
            logger.warning(
                "Could not write the control-context record to %s; nothing was changed",
                path,
                exc_info=True,
            )
            raise ContextStoreUnavailable(
                "store_write_failed",
                "The control-context record could not be written, so the change was not "
                "applied. Check the server's write access to the agent-data root and try "
                "again.",
            ) from exc
        return mutation.result

    def _resolve_path(self) -> Path:
        path = self._path if self._path is not None else record_path()
        if path is None:
            raise ContextStoreUnavailable(
                "store_unavailable",
                "This deployment's agent-data root does not resolve, so there is nowhere to "
                "record a control context that the agent would read back. Nothing was changed.",
            )
        return path

    def _require_ownership(self, record: ControlContext | None) -> None:
        owner = None if record is None else record.owner
        if owner == self._identity:
            return
        if owner is None:
            raise ContextOwnedElsewhere(
                None,
                "The control-context record no longer names an owner this process can write "
                "as, so the change was not applied. It will be claimed again on the next "
                "owner tick.",
            )
        where = f" on port {owner.port}" if owner.port else ""
        raise ContextOwnedElsewhere(
            owner,
            f"The control context is owned by {owner.kind} pid {owner.pid}{where}, not by this "
            "process, so the change was not applied. Make the change there.",
        )
