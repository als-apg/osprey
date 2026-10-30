"""What a supervisor does about a connector call that failed: one policy, one table.

A supervisor owns a connector's lifetime: the connector-host pool owns its
children, a server owns its connectors. A supervisor that calls the policy —
the connector-host pool does — hands the error and the supervised connector to
:meth:`RecoveryPolicy.recover` when a call fails, and the policy answers with
what it did, as a :class:`Recovery` record, or ``None`` when the failure
restarts nothing.

The decision is one table, failure class and origin in, one step out:

* a connector bug (:class:`~osprey_connectors.errors.ConnectorInternalError`),
  a failure about one channel (:class:`~osprey_connectors.errors.ChannelError`),
  a batch read that failed
  (:class:`~osprey_connectors.errors.ChannelReadFailedError`) and a refused
  client library (:class:`~osprey_connectors.errors.ClientLibraryRefusedError`),
  from anywhere: none;
* a :class:`TimeoutError` on the link: replace for a write when writes replace,
  probe otherwise;
* a lost connector (:class:`~osprey_connectors.errors.ConnectorUnavailableError`)
  and a bare :class:`ConnectionError`, from anywhere: replace;
* anything else, a :class:`TimeoutError` from the child or from this process
  included: none.

A failure under one of the three contract roots of
:mod:`osprey_connectors.errors` (a channel failure, a lost connector, a
connector bug) is decided by its root alone; the builtin rows decide only a
failure under no root. So no failure reaches two steps, and no decision depends
on the order of the rows.

*Probe* pings the incarnation: one that answers is kept (``child_kept``), one
that does not is replaced. *Replace* retires the incarnation under
:attr:`Timing.LAZY` (the next call starts a fresh one: ``child_replaced`` for a
child, ``connector_dropped`` for an in-process connector) and respawns it under
:attr:`Timing.EAGER` (``child_replaced``, or ``respawn_refused`` when no new
child could be brought up).

Invariants:

* Callers hand over the error; the policy decides where it came from. An error
  the connector-host child sent is :attr:`Origin.CHILD`, one the link to the
  child made (a missed deadline, a closed pipe) is :attr:`Origin.LINK`, and
  anything from a connector in this process is :attr:`Origin.IN_PROCESS`.
* Recovery happens at most once per incarnation. Every failure of one
  incarnation awaits the one act the first started, and an incarnation that has
  been replaced or dropped is answered with that record and never acted on
  again.
* One attempt per failure episode. A refused respawn answers every call that
  began before the refusal; a call that begins later may try once more. A kept
  child is no episode's end: its next missed deadline pings it again.
* The policy never retries a call, owns no deadline, and holds no incarnation
  alive.
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum, StrEnum
from functools import partial
from typing import Protocol

from osprey_connectors.errors import (
    ChannelError,
    ChannelReadFailedError,
    ClientLibraryRefusedError,
    ConnectorInternalError,
    ConnectorUnavailableError,
)

__all__ = [
    "Origin",
    "Recovery",
    "RecoveryAction",
    "RecoveryPolicy",
    "Supervised",
    "Timing",
]

#: The attribute the connector-host proxy sets, true, on an error the child sent.
_RAISED_BY_CHILD = "_osprey_raised_by_child"


class Origin(StrEnum):
    """Where a failure was raised."""

    #: In the connector-host child, which answered with it.
    CHILD = "child"
    #: On the link to the child: a missed deadline, a closed pipe.
    LINK = "link"
    #: By a connector running in this process.
    IN_PROCESS = "in_process"


class Timing(StrEnum):
    """When a replaced incarnation's successor is brought up."""

    #: By the next call.
    LAZY = "lazy"
    #: At once, by the recovery itself.
    EAGER = "eager"


class RecoveryAction(StrEnum):
    """What a supervisor did about a failure."""

    CONNECTOR_DROPPED = "connector_dropped"
    CHILD_REPLACED = "child_replaced"
    CHILD_KEPT = "child_kept"
    RESPAWN_REFUSED = "respawn_refused"


@dataclass(frozen=True)
class Recovery:
    """One thing a supervisor did about a failure, in one sentence."""

    action: RecoveryAction
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", RecoveryAction(self.action))
        if not self.text:
            raise ValueError("a recovery record states what was done; its text is empty")

    def as_dict(self) -> dict[str, str]:
        """The record as a failure's ``recovery`` carries it."""
        return {"action": str(self.action), "text": self.text}


class Supervised(Protocol):
    """A connector as the policy acts on it.

    ``incarnation`` is the one running instance the failure belongs to: the
    connector-host proxy of a child, or the in-process connector itself. It is
    hashable and weakly referable, and a successor is a different object.
    ``respawn`` is the eager-only member, required only under
    :attr:`Timing.EAGER` and never called under :attr:`Timing.LAZY`, so a lazy
    supervisor's target leaves it out: ``async respawn() -> bool`` brings up a
    successor in the incarnation's place, retires the incarnation once the
    successor serves, and answers whether it did.
    """

    @property
    def incarnation(self) -> object: ...

    @property
    def in_child(self) -> bool: ...

    async def ping(self) -> bool: ...

    async def retire(self) -> None: ...


class _Step(Enum):
    NONE = "none"
    PROBE = "probe"
    REPLACE = "replace"


class _Failure(Enum):
    """The failure class a sentence names: a lost connector, or a missed deadline and ping."""

    UNAVAILABLE = "unavailable"
    UNRESPONSIVE = "unresponsive"


@dataclass(frozen=True)
class _Row:
    failure: type[BaseException]
    origin: Origin | None
    step: _Step
    #: ``True``: only a write whose timeout the policy replaces for; ``False``: only any
    #: other call; ``None``: either.
    write_timeout: bool | None = None


_ROOTS: tuple[type[BaseException], ...] = (
    ChannelError,
    ConnectorUnavailableError,
    ConnectorInternalError,
)

_STEPS: tuple[_Row, ...] = (
    _Row(ConnectorInternalError, None, _Step.NONE),
    _Row(ChannelError, None, _Step.NONE),
    _Row(ChannelReadFailedError, None, _Step.NONE),
    _Row(ClientLibraryRefusedError, None, _Step.NONE),
    _Row(TimeoutError, Origin.LINK, _Step.REPLACE, write_timeout=True),
    _Row(TimeoutError, Origin.LINK, _Step.PROBE, write_timeout=False),
    _Row(ConnectorUnavailableError, None, _Step.REPLACE),
    _Row(ConnectionError, None, _Step.REPLACE),
    _Row(BaseException, None, _Step.NONE),
)

_KEPT = (
    "The connector-host child still answers a ping, so it was kept; the call "
    "may yet complete in it and was not retried."
)
_LAZY_REPLACED = (
    "The connector-host child was discarded; the next call starts a fresh one. "
    "The call was not retried."
)
_REFUSED = "Replacing the connector-host child was refused; the call was not retried."

_SENTENCES: dict[tuple[RecoveryAction, Timing, _Failure], str] = {
    (RecoveryAction.CHILD_KEPT, Timing.LAZY, _Failure.UNRESPONSIVE): _KEPT,
    (RecoveryAction.CHILD_KEPT, Timing.EAGER, _Failure.UNRESPONSIVE): _KEPT,
    (RecoveryAction.CHILD_REPLACED, Timing.LAZY, _Failure.UNAVAILABLE): _LAZY_REPLACED,
    (RecoveryAction.CHILD_REPLACED, Timing.LAZY, _Failure.UNRESPONSIVE): _LAZY_REPLACED,
    (RecoveryAction.CHILD_REPLACED, Timing.EAGER, _Failure.UNAVAILABLE): (
        "The connector-host child that served this call has been replaced; "
        "the call was not retried."
    ),
    (RecoveryAction.CHILD_REPLACED, Timing.EAGER, _Failure.UNRESPONSIVE): (
        "The connector-host child did not answer a ping and has been replaced; "
        "the call was not retried."
    ),
    (RecoveryAction.RESPAWN_REFUSED, Timing.EAGER, _Failure.UNAVAILABLE): _REFUSED,
    (RecoveryAction.RESPAWN_REFUSED, Timing.EAGER, _Failure.UNRESPONSIVE): _REFUSED,
    (RecoveryAction.CONNECTOR_DROPPED, Timing.LAZY, _Failure.UNAVAILABLE): (
        "The connector was dropped; the next call builds a new one."
    ),
    (RecoveryAction.CONNECTOR_DROPPED, Timing.LAZY, _Failure.UNRESPONSIVE): (
        "The connector was dropped; the next call builds a new one."
    ),
}


def _origin(exc: BaseException, target: Supervised) -> Origin:
    if not target.in_child:
        return Origin.IN_PROCESS
    return Origin.CHILD if getattr(exc, _RAISED_BY_CHILD, False) else Origin.LINK


def _matches(row: _Row, exc: BaseException, origin: Origin, *, write_timeout: bool) -> bool:
    if not isinstance(exc, row.failure):
        return False
    if row.origin is not None and row.origin is not origin:
        return False
    if row.write_timeout is not None and row.write_timeout is not write_timeout:
        return False
    if row.failure is BaseException:
        return True
    # A failure under a contract root is decided by that root's rows alone.
    root = next((root for root in _ROOTS if isinstance(exc, root)), None)
    return root is None or issubclass(row.failure, root)


def _step_of(exc: BaseException, origin: Origin, *, write_timeout: bool) -> _Step:
    return next(
        row.step for row in _STEPS if _matches(row, exc, origin, write_timeout=write_timeout)
    )


class _Episode:
    """One incarnation's one recovery, and the tickets its record answers."""

    __slots__ = ("answers_through", "task")

    def __init__(self, task: asyncio.Future[Recovery]) -> None:
        self.task = task
        #: Set when a respawn is refused: the last ticket issued before the refusal.
        self.answers_through: int | None = None

    def answers(self, ticket: int) -> bool:
        return self.answers_through is None or ticket <= self.answers_through


class RecoveryPolicy:
    """One supervisor's recovery: the table above, at one timing, with one memo.

    Args:
        timing: When a replaced incarnation's successor is brought up.
            An in-process connector always recovers lazily.
        replace_on_write_timeout: Replace the child of a write that missed its
            deadline without pinging it, so nothing left in it can send that
            write after a newer one.
    """

    def __init__(self, *, timing: Timing, replace_on_write_timeout: bool = False) -> None:
        self._timing = Timing(timing)
        self._replace_on_write_timeout = replace_on_write_timeout
        self._issued = 0
        self._memo: weakref.WeakKeyDictionary[object, _Episode] = weakref.WeakKeyDictionary()

    def begin(self) -> int:
        """A ticket for a call that is starting; later calls get larger tickets."""
        self._issued += 1
        return self._issued

    def _step(self, exc: BaseException, target: Supervised, *, write: bool) -> _Step:
        write_timeout = write and self._replace_on_write_timeout
        return _step_of(exc, _origin(exc, target), write_timeout=write_timeout)

    async def recover(
        self, exc: BaseException, target: Supervised, *, ticket: int, write: bool = False
    ) -> Recovery | None:
        """Act on *target* for *exc*, at most once per incarnation; ``None`` when nothing is done.

        *ticket* is the one :meth:`begin` gave the failed call when it started;
        *write* says the call could move the machine.

        Raises:
            ValueError: An eager policy was handed a target that is not in a
                child, or that cannot respawn.
        """
        respawn: Callable[[], Awaitable[bool]] | None = getattr(target, "respawn", None)
        if self._timing is Timing.EAGER:
            if not target.in_child:
                raise ValueError("an in-process connector recovers lazily; this policy is eager")
            if respawn is None:
                raise ValueError("an eager policy needs a target that can respawn")
        step = self._step(exc, target, write=write)
        if step is _Step.NONE:
            return None
        incarnation = target.incarnation
        episode = self._memo.get(incarnation)
        if episode is None or not episode.answers(ticket):
            episode = _Episode(asyncio.ensure_future(self._act(step, target, respawn)))
            self._memo[incarnation] = episode
            episode.task.add_done_callback(partial(self._settle, weakref.ref(incarnation), episode))
        return await asyncio.shield(episode.task)

    def _settle(
        self, incarnation: weakref.ref[object], episode: _Episode, task: asyncio.Future[Recovery]
    ) -> None:
        if not task.cancelled() and task.exception() is None:
            action = task.result().action
            if action is RecoveryAction.RESPAWN_REFUSED:
                episode.answers_through = self._issued
                return
            if action is not RecoveryAction.CHILD_KEPT:
                return
        alive = incarnation()
        if alive is not None and self._memo.get(alive) is episode:
            del self._memo[alive]

    async def _act(
        self, step: _Step, target: Supervised, respawn: Callable[[], Awaitable[bool]] | None
    ) -> Recovery:
        failure = _Failure.UNAVAILABLE
        if step is _Step.PROBE:
            if await target.ping():
                return self._record(RecoveryAction.CHILD_KEPT, _Failure.UNRESPONSIVE)
            failure = _Failure.UNRESPONSIVE
        if respawn is not None and self._timing is Timing.EAGER:
            respawned = await respawn()
            action = RecoveryAction.CHILD_REPLACED if respawned else RecoveryAction.RESPAWN_REFUSED
        else:
            await target.retire()
            action = (
                RecoveryAction.CHILD_REPLACED
                if target.in_child
                else RecoveryAction.CONNECTOR_DROPPED
            )
        return self._record(action, failure)

    def _record(self, action: RecoveryAction, failure: _Failure) -> Recovery:
        return Recovery(action, _SENTENCES[(action, self._timing, failure)])
