"""CANOPY's PV backend on the deployment's control session.

CANOPY loads :class:`OspreyPVConnector` by its ``module:Class`` path
(``CANOPY_EPICS__BACKEND=osprey.integrations.canopy.backend:OspreyPVConnector``)
and from then on every PV a screen shows comes from the one
:class:`~osprey.runtime.session.ControlSession` this process opens — the
connector the deployment's control-context record names, never a second EPICS
client beside it.

The process is bound for life, and four things end it rather than being
repaired in place. Each is a process exit, because the server that hosts the
backend would turn a raised exception into its own generic failure status and
the supervisor reads the status to decide what to do next:

* An environment the session refuses, or one that does not say which record it
  was rendered for, exits with
  :data:`~osprey.runtime.session.EXIT_START_REFUSED` before anything is served.
  The same environment would be refused again, so it is not respawned.
* An environment rendered for another record than the one this process read
  exits with :data:`~osprey.runtime.session.EXIT_BINDING_MOVED` before anything
  is served: everything else in the environment (the archive block above all)
  describes that other record.
* A binding that moves while serving stops writes, lets the admitted ones
  finish within their bound, and exits with
  :data:`~osprey.runtime.session.EXIT_BINDING_MOVED`; the next process opens
  against the new record. A write still running when the bound passes does not
  hold the exit: a named status is logged and the process ends anyway.
* A transport the session reports stale exits with
  :data:`~osprey.runtime.session.EXIT_TRANSPORT_STALE` at once, with no session
  teardown: closing a stale client can block the loop, and the server sees the
  sockets close when the process is gone. The next process judges its own
  transport only on channels it connected itself. Where staleness cannot occur — no connector, or one that
  cannot tell — the process waits on. A write in flight then stands in the
  audit as its ``attempt`` record alone: its outcome is unknown.

Writes go through the session, as CANOPY's write contract asks. A panel write
belongs to the terminal's sessions: the session asks the hold about every key
the terminal names in :data:`SESSIONS_FILENAME`, and is held while any one of
them is. :meth:`OspreyPVConnector.put` refuses a write itself when the binding
the client quotes is not this process's; every other write is
:meth:`~osprey.runtime.session.ControlSession.write` with the client's
``confirmed``, judged on the execution verdict read at its admission — at
``asked`` a write no person confirmed is refused ``approval_missing``. Each
outcome but ``confirmed`` becomes CANOPY's exception for it, whose detail is
the toast the client shows. Every write, refused or not, is audited by the
session. The raw-put block is armed at :meth:`~OspreyPVConnector.start`, so
nothing in this process writes around the connector.

What :attr:`~OspreyPVConnector.writes` claims is recomputed from the session's
verdict on every heartbeat tick, so a switch or a hold reaches the open panel
in place; only a move of the record ends the process.

What :attr:`~OspreyPVConnector.reachable` states outlives the process on the
same record. At each change between ``yes`` and ``no`` the process rewrites
:data:`STATEMENT_FILENAME` in its working directory (the sidecar's key
directory) with the record it serves, the state and its ``since``; a stale
transport writes ``no`` there from the stale watch, once the connector reports
it and before the exit. The sidecar hands that statement to the next process on an unchanged
record as :data:`REACH_SINCE_ENV`, which starts in it and holds it until its
own channels say otherwise, so a restart during an outage keeps the outage's
``since``.

A channel that goes down is said at once, as a not-connected value that keeps
the kind the channel last had — an enum stays an enum while it is down — and is
a scalar only when the channel never delivered anything. What the binding is
(:attr:`~OspreyPVConnector.label`, :attr:`~OspreyPVConnector.notice`,
:attr:`~OspreyPVConnector.binding`) is readable while the session is open.

A PV is keyed by the name exactly as the client spelled it — ``X``, ``ca://X``
and ``pva://X`` are three keys, and every value and metadata answer carries the
spelling it was asked under — while the session is always given the bare name:
which protocol reaches a channel is the deployment connector's decision.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import sys
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from canopy.contracts.pv import PVMeta, PVValue, PVWriteValue, ValueKind
from canopy.contracts.sample import AlarmSeverity
from canopy.core.errors import CanopyError
from canopy.epics.base import (
    PVConnectionFailed,
    PVReadTimeout,
    normalize_severity,
)
from canopy.epics.pv_protocol import strip_protocol_prefix

from osprey.integrations.canopy import (
    NO_RECORD,
    REACH_SINCE_ENV,
    RENDERED_FOR_ENV,
    SESSIONS_FILENAME,
    STATEMENT_FILENAME,
    WRITE_BUDGET_ENV,
)
from osprey.runtime.budget import WRITE_MARGIN_S, WriteBudget, write_budget
from osprey.runtime.session import (
    EXIT_BINDING_MOVED,
    EXIT_START_REFUSED,
    EXIT_TRANSPORT_STALE,
    OUTCOME_CONFIRMED,
    OUTCOME_MISMATCH,
    OUTCOME_REFUSED,
    OUTCOME_SWITCH_IN_PROGRESS,
    OUTCOME_TARGET_CHANGED,
    OUTCOME_UNCONFIRMED,
    OUTCOME_UNKNOWN,
    STATE_BOUND,
    STATE_UNSERVABLE,
    Binding,
    ControlSession,
    SessionEnvironmentError,
    SessionUnservableError,
    SessionWriteResult,
    Subscription,
)
from osprey.runtime.session_outcomes import (
    MESSAGE_NO_RECORD,
    MESSAGE_UNSERVABLE,
    REASON_CONTEXT_UNAVAILABLE,
    REASON_SWITCHING,
    refusal_message,
)
from osprey_connectors import control_context, control_system
from osprey_connectors.logger import get_logger

logger = get_logger("canopy_backend")

__all__ = [
    "NO_RECORD",
    "PVSwitchInProgress",
    "PVTargetChanged",
    "PVWriteMismatch",
    "PVWriteRefused",
    "PVWriteUnconfirmed",
    "PVWriteUnknown",
    "REACH_NO",
    "REACH_UNKNOWN",
    "REACH_YES",
    "RENDERED_FOR_ENV",
    "STATEMENT_PUSH_WAIT_S",
    "CLIENT_LABEL",
    "SURFACE",
    "WRITE_BUDGET_ENV",
    "WRITE_DETAIL_MAX_CHARS",
    "OspreyPVConnector",
    "process_write_budget",
    "read_session_keys",
    "record_identity",
    "writes_claim",
]

#: The surface name this backend opens its session under; audit readers match
#: CANOPY panel writes on it.
SURFACE = "canopy"

#: How a refused target switch names this backend's in-flight write.
CLIENT_LABEL = "a CANOPY panel write"

#: The audit ``subject`` of every write record this backend's session files.
WRITE_SUBJECT = "canopy_write"

#: The longest detail a write answer carries, CANOPY's ``detail_max_chars``.
WRITE_DETAIL_MAX_CHARS = 512
_WRITE_DETAIL_ELLIPSIS = "\u2026"


def _cut_write_detail(text: str, limit: int = WRITE_DETAIL_MAX_CHARS) -> str:
    """``text`` when it fits in ``limit`` characters, else cut at a word boundary with an ellipsis.

    A leading token longer than the room left has no boundary to keep and is
    cut inside it, still under ``limit``. Whitespace before the ellipsis goes.
    """
    if len(text) <= limit:
        return text
    head = text[: limit - len(_WRITE_DETAIL_ELLIPSIS)]
    if not text[len(head)].isspace():
        words = head.rsplit(None, 1)
        if len(words) == 2:
            head = words[0]
    return head.rstrip() + _WRITE_DETAIL_ELLIPSIS


class _WriteCode(StrEnum):
    """The envelope codes of the connector rows of CANOPY's write-outcome table."""

    REFUSED = "pv_write_refused"
    TARGET_CHANGED = "pv_target_changed"
    SWITCH_IN_PROGRESS = "pv_switch_in_progress"
    MISMATCH = "pv_write_mismatch"
    UNCONFIRMED = "pv_write_unconfirmed"
    UNKNOWN = "pv_write_unknown"


class _PVWriteOutcome(CanopyError):
    """A write that did not end ``confirmed``, as CANOPY's error handler answers it.

    The status, the code and the default detail are the row's in CANOPY's
    write-outcome table; a test holds them to its ``write_outcome_vectors.json``.
    The detail is the toast the client shows, stripped and capped.
    """

    default_detail = ""

    def __init__(self, detail: str = "") -> None:
        super().__init__(_cut_write_detail(detail.strip()) or self.default_detail)


class PVWriteRefused(_PVWriteOutcome):
    """The write was refused before anything reached the channel."""

    status_code = 403
    error_code = _WriteCode.REFUSED
    default_detail = "the write was refused; nothing written"


class PVTargetChanged(_PVWriteOutcome):
    """The write's binding is not this process's control target."""

    status_code = 409
    error_code = _WriteCode.TARGET_CHANGED
    default_detail = "the control target changed; nothing written"


class PVSwitchInProgress(_PVWriteOutcome):
    """A control-target switch is under way."""

    status_code = 409
    error_code = _WriteCode.SWITCH_IN_PROGRESS
    default_detail = "a target switch is in progress; nothing written"


class PVWriteMismatch(_PVWriteOutcome):
    """Written, but the confirming read saw another value."""

    status_code = 409
    error_code = _WriteCode.MISMATCH
    default_detail = "written; the channel holds a different value"


class PVWriteUnconfirmed(_PVWriteOutcome):
    """Sent, but the put was not acknowledged or the confirming read failed."""

    status_code = 502
    error_code = _WriteCode.UNCONFIRMED
    default_detail = "sent, not verified"


class PVWriteUnknown(_PVWriteOutcome):
    """The write's outcome is unknown."""

    status_code = 502
    error_code = _WriteCode.UNKNOWN
    default_detail = "outcome unknown \u2014 check the readback"


#: CANOPY's exception for each outcome but ``confirmed``. Each class's
#: default detail is the vector file's default for its dynamic rows.
_OUTCOME_ERRORS: dict[str, type[CanopyError]] = {
    OUTCOME_REFUSED: PVWriteRefused,
    OUTCOME_TARGET_CHANGED: PVTargetChanged,
    OUTCOME_SWITCH_IN_PROGRESS: PVSwitchInProgress,
    OUTCOME_MISMATCH: PVWriteMismatch,
    OUTCOME_UNCONFIRMED: PVWriteUnconfirmed,
    OUTCOME_UNKNOWN: PVWriteUnknown,
}

#: Used when the configuration handed to :meth:`OspreyPVConnector.from_config`
#: does not carry the setting.
_DEFAULT_READ_TIMEOUT_S = 5.0
_DEFAULT_CONNECT_REPORT_S = 1.5
_DEFAULT_MAX_ARRAY_ELEMENTS = 100_000

#: How long a move watcher waits, after a move, for the broker to push the
#: changed statement before it drains: a process with nothing to drain must not
#: end before its flip frame has left, and a broker that never pushes must not
#: hold the exit.
STATEMENT_PUSH_WAIT_S = 1.0

#: The three words :attr:`OspreyPVConnector.reachable` states.
REACH_YES = "yes"
REACH_NO = "no"
REACH_UNKNOWN = "unknown"


def _hold_down_s() -> float:
    """How long the connected count stays at zero after a drop before it reads ``no``.

    The connector's stale hold-down, read when used: one bounce length for the
    stale watch and for this statement.
    """
    return control_system.STALE_HOLD_DOWN_S


def _reach(
    state: str, *, since: float | str | None = None, reason: str | None = None
) -> dict[str, Any]:
    """One ``reachable`` statement, the heartbeat's shape."""
    return {"state": state, "since": since, "reason": reason}


def _carried_reach(value: str | None) -> dict[str, Any] | None:
    """The statement :data:`REACH_SINCE_ENV` carries, or ``None`` for no usable carry."""
    text = (value or "").strip()
    state, _, since_text = text.partition(",")
    if state == REACH_YES and not since_text:
        return _reach(REACH_YES)
    if state == REACH_NO:
        try:
            since = float(since_text)
        except ValueError:
            return None
        if math.isfinite(since):
            return _reach(REACH_NO, since=since)
    return None


def record_identity(binding: Binding) -> str:
    """The record a session read at open, as :data:`RENDERED_FOR_ENV` spells it.

    ``<target>@<generation>`` of the record, or :data:`NO_RECORD` when there
    was none. This is deliberately not ``binding.id``: that names the *state*
    for a fallback or unservable binding, so a process serving either would
    never match the record its environment was rendered from.
    """
    if binding.generation is None:
        return NO_RECORD
    return f"{binding.target}@{binding.generation}"


def _exit_process(status: int) -> None:
    """End the process now with *status*, after flushing what was logged.

    ``os._exit`` skips buffers and exit handlers, and nothing else gets the
    status out: an exception raised during the hosting server's startup becomes
    the server's own failure status.
    """
    loggers = [logging.getLogger()]
    loggers += [
        candidate
        for candidate in logging.Logger.manager.loggerDict.values()
        if isinstance(candidate, logging.Logger)
    ]
    for candidate in loggers:
        for handler in list(candidate.handlers):
            with contextlib.suppress(Exception):
                handler.flush()
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    os._exit(status)


def _arm_raw_put_block() -> None:
    """Refuse every raw client put in this process that no connector let through.

    The executor's own armed contract, as the notebook kernel arms it: rpc and
    Tango commands stay allowed, since nothing here bounds them.
    """
    from osprey.runtime import raw_put_block
    from osprey.services.python_executor.execution.wrapper import armed_contract

    raw_put_block.install("armed", **armed_contract(refuse_rpc=False))


def _control_system_section() -> dict[str, Any]:
    """The deployment's ``control_system:`` section, or an empty one."""
    from osprey_connectors.config import get_config_value

    try:
        section = get_config_value("control_system", {})
    except Exception:
        logger.warning("The control_system config could not be read", exc_info=True)
        return {}
    return dict(section) if isinstance(section, Mapping) else {}


def _parse_budget(text: str) -> tuple[float, float, float]:
    """``T,S,D`` as three finite positive numbers.

    Raises:
        ValueError: The text is not three such numbers.
    """
    parts = text.split(",")
    if len(parts) != 3:
        raise ValueError(f"expected T,S,D, got {len(parts)} field(s)")
    call, step, bound = (float(part) for part in parts)
    if not all(math.isfinite(v) and v > 0 for v in (call, step, bound)):
        raise ValueError("every field must be a finite positive number")
    return call, step, bound


def process_write_budget(
    configured: WriteBudget, environ: Mapping[str, str] | None = None
) -> WriteBudget:
    """The write budget this process runs with.

    *configured* (the deployment config's budget), lowered by
    :data:`WRITE_BUDGET_ENV` where that is smaller: ``T`` and ``S`` are the
    componentwise minima and ``D = S + 2T`` is recomputed from them; the stop
    grace and the relay timeout stay *configured*'s, which only ever bound a
    smaller ``D`` from above. An absent variable leaves *configured* as it is;
    a malformed one is logged and ignored.
    """
    environ = os.environ if environ is None else environ
    text = (environ.get(WRITE_BUDGET_ENV) or "").strip()
    if not text:
        return configured
    try:
        env_t, env_s, _env_d = _parse_budget(text)
    except ValueError as exc:
        logger.warning(
            "%s=%r is malformed (%s); using the config's write budget", WRITE_BUDGET_ENV, text, exc
        )
        return configured
    call = min(env_t, configured.T)
    step = min(env_s, configured.S)
    return WriteBudget(
        T=call,
        S=step,
        D=step + 2 * call,
        grace_s=configured.grace_s,
        relay_timeout_s=configured.relay_timeout_s,
    )


def read_session_keys(path: Path) -> list[str]:
    """The session keys the terminal names in *path*; ``[]`` when there is no file.

    Raises:
        OSError, ValueError: The file is there but cannot be read as a JSON list
            of strings. The session then refuses writes rather than read it as
            naming no session.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    keys = json.loads(text)
    if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
        raise ValueError(f"{path.name} is not a list of session keys")
    return keys


#: The claim's ``state`` for each refusal reason that has a word of its own;
#: every other refusal is ``locked``.
_REFUSED_STATES: dict[str, str] = {
    "session_held": "held",
    "switch_off": "off",
    "stopped_since": "off",
    "switching": "applying",
}


def writes_claim(session: ControlSession) -> dict[str, Any]:
    """What a panel's writes are now, as the heartbeat's ``{state, reason, cause}``.

    Read from the session's cached verdict and its own states; reads no file.

    * allowed at ``allowed``: ``on``;
    * allowed at ``asked``: ``asked``, cause ``asked`` (the viewer confirms
      each run and sends ``confirmed``);
    * refused ``session_held``: ``held``, with the phrase and each open flag's
      source and reason;
    * refused ``switch_off`` or ``stopped_since``: ``off``;
    * refused ``switching``, or the record moving: ``applying``;
    * no record, or no connector: ``locked``, with the session's own text;
    * any other refusal: ``locked``.

    A refusal's ``reason`` is the verdict's phrase for its kind of machine and
    its ``cause`` the verdict's reason.
    A viewer that predates ``asked`` and ``held`` reads either as inert.
    """
    if session.target_moving:
        return _claim("applying", _phrase_for(session, REASON_SWITCHING), REASON_SWITCHING)
    state = session.binding.state
    if state != STATE_BOUND:
        text = MESSAGE_NO_RECORD if state == "fallback" else MESSAGE_UNSERVABLE
        return _claim("locked", text, REASON_CONTEXT_UNAVAILABLE)
    verdict = session.verdict
    if verdict.allowed:
        if verdict.position == "asked":
            return _claim("asked", None, "asked")
        return _claim("on", None, None)
    reason = str(verdict.reason)
    return _claim(_REFUSED_STATES.get(reason, "locked"), refusal_message(verdict), reason)


def _phrase_for(session: ControlSession, reason: str) -> str:
    """The refusal phrase for *reason*, naming the session's kind of machine."""
    from osprey_connectors.refusals import refusal_phrase

    return refusal_phrase(reason, session.verdict.kind)


def _claim(state: str, reason: str | None, cause: str | None) -> dict[str, Any]:
    return {"state": state, "reason": reason, "cause": cause}


def _raise_for(pv: str, result: SessionWriteResult) -> None:
    """Return for ``confirmed``; raise CANOPY's exception for any other outcome.

    The detail is the session's message, stripped; CANOPY's exception class cuts
    it to its limit, and takes its default (the vector file's) when it is empty.
    """
    if result.outcome == OUTCOME_CONFIRMED:
        return
    error = _OUTCOME_ERRORS.get(result.outcome)
    if error is None:
        logger.error("Write to %s ended in an unknown outcome %r", pv, result.outcome)
        error = PVWriteUnknown
    raise error((result.message or "").strip())


def _timestamp_ns(timestamp: Any) -> int:
    try:
        return int(timestamp.timestamp() * 1_000_000_000)
    except Exception:
        return time.time_ns()


def _disconnected(pv: str, error: str, kind: ValueKind = ValueKind.SCALAR) -> PVValue:
    """A not-connected value of *pv*, of the *kind* last known for it."""
    return PVValue(
        pv=pv,
        value=None,
        value_kind=kind,
        severity=AlarmSeverity.INVALID,
        ts_ns=time.time_ns(),
        connected=False,
        error=error,
    )


class _Channel:
    """One session subscription, shared by everything that asks for one PV key."""

    def __init__(self, pv: str, address: str) -> None:
        self.pv = pv
        self.address = address
        self.holders = 0
        self.listeners: list[Callable[[PVValue], None]] = []
        self.handle: Subscription | None = None
        #: The last event this channel produced, connected or not.
        self.current: PVValue | None = None
        self.meta: PVMeta | None = None
        #: Set by the first value; what ``get`` and ``describe`` wait on.
        self.first_value = asyncio.Event()
        self.report_timer: asyncio.TimerHandle | None = None
        self.closed = False

    @property
    def kind(self) -> ValueKind:
        """The kind this channel last had, ``SCALAR`` when it never delivered one."""
        if self.meta is not None:
            return self.meta.value_kind
        if self.current is not None:
            return self.current.value_kind
        return ValueKind.SCALAR

    def cancel_report(self) -> None:
        if self.report_timer is not None:
            self.report_timer.cancel()
            self.report_timer = None


class _Monitor:
    """The handle :meth:`OspreyPVConnector.subscribe` returns."""

    def __init__(
        self,
        connector: OspreyPVConnector,
        channel: _Channel,
        on_event: Callable[[PVValue], None],
    ) -> None:
        self._connector = connector
        self._channel = channel
        self._on_event = on_event
        self._closed = False

    async def close(self) -> None:
        """Stop delivering to this monitor. Idempotent."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(ValueError):
            self._channel.listeners.remove(self._on_event)
        await self._connector._release(self._channel)


class OspreyPVConnector:
    """CANOPY's ``PVConnector`` contract, served by one control session.

    Build it with :meth:`from_config`; nothing is opened until :meth:`start`.
    """

    backend_name = "osprey"

    def __init__(
        self,
        *,
        read_timeout_s: float = _DEFAULT_READ_TIMEOUT_S,
        connect_report_s: float = _DEFAULT_CONNECT_REPORT_S,
        max_array_elements: int = _DEFAULT_MAX_ARRAY_ELEMENTS,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._read_timeout = read_timeout_s
        self._connect_report = connect_report_s
        self._max_array_elements = max_array_elements
        self._session: ControlSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._watcher: asyncio.Task[None] | None = None
        self._stale_watcher: asyncio.Task[None] | None = None
        #: The watchers that have left ``wait_moved`` / ``wait_stale`` and are
        #: acting on what they saw; :meth:`close` cancels only the others.
        self._acting: set[asyncio.Task[Any]] = set()
        self._budget: WriteBudget | None = None
        self._channels: dict[str, _Channel] = {}
        self._channel_lock = asyncio.Lock()
        #: What :attr:`reachable` measures the hold-down with, and stamps ``since`` from.
        self._clock = clock
        self._wall_clock = wall_clock
        #: Channels connected now.
        self._up: set[_Channel] = set()
        #: Subscribed channels that connected in this process and have dropped since.
        self._dropped: set[_Channel] = set()
        #: When a drop took the connected count to zero (``clock``), and the same
        #: moment in epoch seconds: the ``since`` of the ``no`` it becomes.
        #: Both None while there is no such drop to account for.
        self._drop_at: float | None = None
        self._drop_since: float | None = None
        #: The ``since`` of the last ``no``, kept across a release until a channel connects.
        self._last_no_since: float | None = None
        #: The predecessor's statement this process holds until its own evidence.
        self._carried: dict[str, Any] | None = None
        #: The ``since`` of the ``no`` a stale transport set; the process is leaving.
        self._stale_since: float | None = None
        #: The last definite ``(state, since)`` written to :data:`STATEMENT_FILENAME`.
        self._stated: tuple[str, float | None] | None = None
        #: Where that statement is written; fixed when the session opens.
        self._statement_path: Path | None = None
        #: Set by :meth:`statement_sent` each time the broker has pushed a changed
        #: statement; whoever waits on a push clears it first.
        self._statement_sent = asyncio.Event()

    @classmethod
    def from_config(cls, cfg: Any) -> OspreyPVConnector:
        """Build an unstarted connector from CANOPY's EPICS settings.

        Only the settings that apply to every backend are read: the read
        timeout, how long a monitor waits before reporting a channel not
        connected, and the longest waveform passed on.
        """
        return cls(
            read_timeout_s=float(getattr(cfg, "ca_timeout_s", _DEFAULT_READ_TIMEOUT_S)),
            connect_report_s=float(getattr(cfg, "connect_report_s", _DEFAULT_CONNECT_REPORT_S)),
            max_array_elements=int(getattr(cfg, "max_array_elements", _DEFAULT_MAX_ARRAY_ELEMENTS)),
        )

    # -- identity -----------------------------------------------------------

    @property
    def label(self) -> str | None:
        """What a person calls the bound target; ``None`` while no session is open."""
        session = self._session
        return None if session is None else session.binding.display_name

    @property
    def notice(self) -> str | None:
        """``None`` while bound, else the state (``fallback`` or ``unservable``).

        Also ``None`` while no session is open.
        """
        session = self._session
        if session is None or session.binding.state == STATE_BOUND:
            return None
        return session.binding.state

    @property
    def binding(self) -> str | None:
        """The session's :attr:`~osprey.runtime.session.Binding.id`; ``None`` while none is open."""
        session = self._session
        return None if session is None else session.binding.id

    # -- writes -------------------------------------------------------------

    @property
    def writes(self) -> dict[str, Any] | None:
        """What this panel's writes are, as ``{state, reason, cause}`` (:func:`writes_claim`).

        ``None`` while no session is open. Read on every heartbeat and every
        flush tick, from the verdict the session last read: the move watch
        refreshes it when the record, the switch or a session's hold changes.
        """
        session = self._session
        if session is None:
            return None
        return writes_claim(session)

    def statement_sent(self) -> None:
        """The broker has pushed a changed statement; wake whoever waits on it.

        Called once per changed statement, whether or not any socket is
        connected — so at most once per flush tick, from the same compare that
        feeds the heartbeat. Rewrites this process's display-only report with
        the statement just sent. Must not block: it runs inside the broker's
        flush loop. A report the file system refuses does not raise here
        (:meth:`_write_statement_report`).
        """
        self._write_statement_report()
        self._statement_sent.set()

    def _write_statement_report(self) -> None:
        """Rewrite this process's display-only server report.

        The DATA process holds a connector to show the machine, not to control
        it, so its report says nothing about sessions, switches, children,
        targets or reachability: only the binding it is on and its machine
        statement ``{reachable, writes}`` on the fleet's terms — ``writes`` the
        session's verdict (:attr:`~osprey.runtime.session.ControlSession.writes`),
        ``reachable`` ``no`` once the transport is lost and ``yes`` until then. The ``statement`` key is what marks
        it display-only, so the fleet readers leave it out at the source
        (:func:`osprey_connectors.control_context.live_reports`). Written by
        itself rather than through the controls server's writer: it sweeps
        nobody's report and answers nobody's request. No agent-data root, no
        report.

        A write the file system refuses (:class:`OSError`) is logged and the
        broker's flush goes on: the next changed statement writes the file
        again. Anything else is a defect in building the report and raises, so
        a report that is never written cannot pass for a written one.
        """
        session = self._session
        if session is None:
            return
        path = control_context.report_path(os.getpid())
        if path is None:
            return
        binding = session.binding
        report = control_context.ServerReport(
            server_pid=os.getpid(),
            applied_target=binding.target,
            applied_generation=binding.generation,
            updated_at=datetime.fromtimestamp(self._wall_clock(), UTC).isoformat(),
            statement={"reachable": self._report_reach(), "writes": session.writes},
        )
        try:
            control_context.write_json_atomic(path, report.to_payload())
        except OSError:
            logger.warning("Could not write the display-only server report", exc_info=True)

    def _report_reach(self) -> dict[str, Any]:
        """The report's ``reachable``: ``no`` since the transport went stale, else ``yes``.

        A channel that does not answer is that channel's failure and never
        makes the machine unreachable (:data:`~osprey_connectors.control_context.REACHABLE_STATES`).
        """
        if self._stale_since is None:
            return _reach(REACH_YES)
        since = datetime.fromtimestamp(self._stale_since, UTC).isoformat()
        return _reach(REACH_NO, since=since)

    # -- reachability -------------------------------------------------------

    @property
    def reachable(self) -> dict[str, Any] | None:
        """Whether the control system answers, from this process's own channels.

        ``{"state": "yes" | "no" | "unknown", "since": epoch s | None, "reason":
        sentence | None}``, or ``None`` while no session is open. Read on every
        heartbeat and every flush tick.

        * ``yes`` while any channel is connected.
        * ``no`` when a drop took the connected count to zero and it has stayed
          there for the hold-down; ``since`` is that drop, stamped once. A bounce
          shorter than the hold-down changes nothing.
        * Otherwise ``unknown``: nothing connected yet, or a release took the
          count to zero.

        A ``no`` reached after a release, with no channel connected in between,
        keeps the ``since`` of the ``no`` before it. A process started with the
        statement its predecessor on the same record left (:data:`REACH_SINCE_ENV`)
        states that where it would say ``unknown``, until a channel connects, its
        evidence says ``no`` (keeping a carried ``no``'s ``since``) or the last
        subscription is released. Once the transport is stale it is ``no``.

        Each change between ``yes`` and ``no`` is written to
        :data:`STATEMENT_FILENAME` for the next process.
        """
        session = self._session
        if session is None:
            return None
        statement = self._reach_now()
        self._state_definite(statement)
        return statement

    def _reach_now(self) -> dict[str, Any]:
        """What :attr:`reachable` says now, without writing it down."""
        if self._stale_since is not None:
            return _reach(REACH_NO, since=self._stale_since)
        if self._up:
            return _reach(REACH_YES)
        if self._drop_at is not None:
            if not self._drop_held():
                # Still inside the hold-down: the statement before the drop stands.
                return _reach(REACH_YES)
            self._last_no_since = self._drop_since
            return _reach(REACH_NO, since=self._drop_since)
        if self._carried is not None:
            return dict(self._carried)
        return _reach(REACH_UNKNOWN)

    def _state_definite(self, statement: dict[str, Any], *, force: bool = False) -> None:
        """Write *statement* to :data:`STATEMENT_FILENAME` when it is a new ``yes`` / ``no``.

        Never raises, makes no client-library call and is safe right before the
        process ends: it is one atomic file replace.
        """
        state, since = statement["state"], statement["since"]
        if state not in (REACH_YES, REACH_NO):
            return
        if not force and self._stated == (state, since):
            return
        session, path = self._session, self._statement_path
        if session is None or path is None:
            return
        self._stated = (state, since)
        try:
            control_context.write_json_atomic(
                path,
                {
                    "binding_key": record_identity(session.binding),
                    "state": state,
                    "since": since,
                    "written_at": self._wall_clock(),
                },
            )
        except OSError:
            logger.warning("Could not write the reachability statement", exc_info=True)

    def _state_stale(self) -> None:
        """The connector's transport went stale: state ``no`` and write it down now.

        Called by the stale watch on the event loop, just before the process
        ends. A ``no`` already stated keeps its ``since``; otherwise the
        ``since`` is this probe.
        """
        session = self._session
        if session is None:
            return
        current = self._reach_now()
        since = current["since"] if current["state"] == REACH_NO else self._wall_clock()
        logger.warning("Stale transport: stating no since %s", since)
        self._stale_since = since
        self._state_definite(_reach(REACH_NO, since=since), force=True)

    def _drop_held(self) -> bool:
        return self._drop_at is not None and self._clock() - self._drop_at >= _hold_down_s()

    def _channel_up(self, channel: _Channel) -> None:
        if channel in self._up:
            return
        self._up.add(channel)
        self._dropped.discard(channel)
        self._forget_outage()

    def _forget_outage(self) -> None:
        """Drop every account of an outage: its drop, its evidence, its ``since`` and the carry."""
        self._drop_at = self._drop_since = None
        self._last_no_since = None
        self._carried = None

    def _channel_dropped(self, channel: _Channel) -> None:
        if channel not in self._up:
            return
        self._up.discard(channel)
        self._dropped.add(channel)
        if not self._up:
            self._drop_at = self._clock()
            self._drop_since = self._wall_clock()

    def _channel_released(self, channel: _Channel) -> None:
        """A release that leaves nothing connected and no dropped channel held is no evidence.

        Releasing the last subscription also ends a carried statement; its ``no``
        keeps its ``since`` for the next ``no``, as a ``no`` of this process would.
        """
        was_up = channel in self._up
        self._up.discard(channel)
        self._dropped.discard(channel)
        if self._carried is not None and all(c is channel for c in self._channels.values()):
            self._carried = None
        if self._up or self._drop_at is None:
            return
        if was_up or not self._dropped:
            if self._drop_held():
                self._last_no_since = self._drop_since
            self._drop_at = self._drop_since = None

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Arm the raw-put block and open the control session, or end the process.

        Runs during the hosting application's startup, so nothing has been
        served when it exits. On return raw client puts are refused process-wide,
        the session is open with this process's write budget
        (:func:`process_write_budget`), matches the record the environment was
        rendered for, and is being watched for a move.
        """
        if self._session is not None:
            return
        self._loop = asyncio.get_running_loop()

        try:
            _arm_raw_put_block()
        except Exception:
            # A process that could not arm the block must not serve: anything
            # in it could put around the connector unchecked.
            logger.error(
                "CANOPY backend start refused: the raw-put block could not be armed", exc_info=True
            )
            _exit_process(EXIT_START_REFUSED)
            raise

        rendered_for = (os.environ.get(RENDERED_FOR_ENV) or "").strip()
        try:
            if not rendered_for:
                raise SessionEnvironmentError([f"{RENDERED_FOR_ENV} is not set"])
            budget = process_write_budget(
                write_budget({"control_system": _control_system_section()})
            )
            sessions_path = Path.cwd() / SESSIONS_FILENAME
            session = await ControlSession.open(
                surface=SURFACE,
                subject=WRITE_SUBJECT,
                client_label=CLIENT_LABEL,
                budget=budget,
                session_keys=lambda: read_session_keys(sessions_path),
            )
        except SessionEnvironmentError as exc:
            logger.error("CANOPY backend start refused: %s", "; ".join(exc.problems))
            _exit_process(EXIT_START_REFUSED)
            raise

        read = record_identity(session.binding)
        if read != rendered_for:
            logger.warning(
                "This environment was rendered for control record %s and the record read "
                "is %s; leaving so the next process is rendered for it",
                rendered_for,
                read,
            )
            await self._close_session(session)
            _exit_process(EXIT_BINDING_MOVED)
            raise PVConnectionFailed(
                f"environment rendered for {rendered_for}, control record is {read}"
            )

        self._session = session
        self._budget = budget
        self._statement_path = Path.cwd() / STATEMENT_FILENAME
        self._carried = _carried_reach(os.environ.get(REACH_SINCE_ENV))
        if self._carried is not None and self._carried["state"] == REACH_NO:
            self._last_no_since = self._carried["since"]
        # The broker's first compare only seeds its last statement, so the
        # report a view matches against exists from the moment the session does.
        self._write_statement_report()
        self._watcher = asyncio.create_task(self._watch(session), name="canopy-binding-watch")
        self._stale_watcher = asyncio.create_task(
            self._watch_stale(session), name="canopy-stale-watch"
        )

    def _leave_parking(self) -> None:
        """Mark the current watcher as acting: :meth:`close` no longer cancels it."""
        task = asyncio.current_task()
        if task is not None:
            self._acting.add(task)

    async def _watch(self, session: ControlSession) -> None:
        """Wait for the binding to move, then stop writes, push, drain and end the process.

        A watch that fails is treated as a move: a process that can no longer
        tell whether its binding stands must not go on serving it.

        When ``wait_moved`` returns the statement already says what the move
        is (``off`` for writes turned off, ``applying`` for writes turned on,
        ``applying`` with cause ``switching`` for another target); writes stop
        before anything is awaited, and the watcher waits for the broker to push
        that statement (bounded by :data:`STATEMENT_PUSH_WAIT_S`) before draining,
        so the outgoing process states the flip before it leaves.

        Writes already admitted get the drain bound (``D`` + M of this
        process's budget; the session never waits past a write's own bound).
        Drained, the session is closed and the connector disconnected; not
        drained, the process ends anyway with a named status logged, and the
        transport is left to the exit — a write still in the connector must not
        have it pulled from under it.
        """
        try:
            await session.wait_moved()
            self._leave_parking()
            logger.warning(
                "Control binding %s moved; leaving so the next process opens the new one",
                session.binding.id,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            self._leave_parking()
            logger.error("Watching the control record failed; leaving", exc_info=True)
        # The statement changed inside ``wait_moved``; nothing has awaited since,
        # so no flush tick has seen it. Writes stop before the first await, and
        # the push of the changed statement is waited for before the drain.
        self._statement_sent.clear()
        session.stop_writes()
        await self._await_statement_push()
        budget = self._budget or write_budget({})
        try:
            drained = await session.drain(budget.D + WRITE_MARGIN_S)
        except Exception:
            logger.error("Draining writes before the move exit failed", exc_info=True)
            drained = False
        if drained:
            await self._close_session(session)
        else:
            logger.error(
                "move_drain_incomplete: writes still in flight at the move exit; "
                "their outcome records may be missing — exit %d",
                EXIT_BINDING_MOVED,
            )
        # What the next process on an unchanged record starts in.
        with contextlib.suppress(Exception):
            self._state_definite(self._reach_now())
        _exit_process(EXIT_BINDING_MOVED)

    async def _await_statement_push(self) -> None:
        """Wait for the broker's next pushed statement, at most :data:`STATEMENT_PUSH_WAIT_S`.

        A statement that did not change is never pushed; the bound then expires
        and the move goes on.
        """
        try:
            await asyncio.wait_for(self._statement_sent.wait(), STATEMENT_PUSH_WAIT_S)
        except TimeoutError:
            logger.info(
                "No changed statement was pushed within %.1f s of the move; draining anyway",
                STATEMENT_PUSH_WAIT_S,
            )

    async def _watch_stale(self, session: ControlSession) -> None:
        """Wait for the transport to go stale, then end the process with no teardown.

        The session is deliberately left open and nothing is drained: a stale
        client's disconnect can block the loop, and ending the process releases
        every socket anyway. A write in flight stands as its ``attempt`` record.
        A stale probe that fails is logged and ends this task only — unlike a
        failed move watch, it says nothing about what is being served.
        """
        try:
            await session.wait_stale()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("Watching the transport for staleness failed", exc_info=True)
            return
        self._state_stale()
        self._leave_parking()
        logger.error("stale transport — exit %d", EXIT_TRANSPORT_STALE)
        _exit_process(EXIT_TRANSPORT_STALE)

    @staticmethod
    async def _close_session(session: ControlSession) -> None:
        try:
            await session.close()
        except Exception:
            logger.warning("Closing the control session failed", exc_info=True)

    async def close(self) -> None:
        """Stop watching, drop every subscription and close the session. Idempotent.

        Only a watcher still parked in ``wait_moved`` / ``wait_stale`` is
        cancelled: a move watcher already draining runs on to its exit. The
        session's close stops writes, drains them and disconnects; it never
        ends the process.
        """
        watchers = (self._watcher, self._stale_watcher)
        self._watcher = self._stale_watcher = None
        parked = [w for w in watchers if w is not None and w not in self._acting]
        for watcher in parked:
            watcher.cancel()
        await asyncio.gather(*parked, return_exceptions=True)
        channels, self._channels = self._channels, {}
        for channel in channels.values():
            channel.closed = True
            channel.cancel_report()
            channel.listeners.clear()
        self._up.clear()
        self._dropped.clear()
        self._forget_outage()
        self._stale_since = self._stated = None
        session, self._session = self._session, None
        if session is not None:
            await self._close_session(session)

    def _require_session(self) -> ControlSession:
        if self._session is None:
            raise PVConnectionFailed("the control session is not open")
        return self._session

    # -- channels -----------------------------------------------------------

    def _on_loop(self, call: Callable[..., None], *args: Any) -> None:
        """Run *call* on the event loop thread, wherever this is called from.

        An in-process connector delivers on the loop thread, synchronously
        inside ``subscribe``; a real client library delivers from its own thread.
        """
        loop = self._loop
        if loop is None:
            return
        try:
            running: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            call(*args)
        else:
            loop.call_soon_threadsafe(call, *args)

    async def _acquire(self, pv: str) -> _Channel:
        """The channel for *pv*, subscribed on first use. Pair with :meth:`_release`."""
        session = self._require_session()
        async with self._channel_lock:
            channel = self._channels.get(pv)
            if channel is not None:
                channel.holders += 1
                return channel
            channel = _Channel(pv, strip_protocol_prefix(pv))
            channel.holders = 1
            self._channels[pv] = channel
            try:
                channel.handle = await session.subscribe(
                    channel.address,
                    lambda reading: self._on_loop(self._on_value, channel, reading),
                    on_connection=lambda up: self._on_loop(self._on_connection, channel, up),
                )
            except Exception:
                channel.closed = True
                self._channels.pop(pv, None)
                self._channel_released(channel)
                raise
            if channel.current is None:
                # Not connected yet. A session with no connector never will be;
                # anything else gets the usual grace before it is said aloud.
                delay = 0.0 if session.binding.state == STATE_UNSERVABLE else self._connect_report
                loop = asyncio.get_running_loop()
                channel.report_timer = loop.call_later(delay, self._report_not_connected, channel)
            return channel

    async def _release(self, channel: _Channel) -> None:
        async with self._channel_lock:
            channel.holders -= 1
            if channel.holders > 0 or channel.closed:
                return
            channel.closed = True
            channel.cancel_report()
            self._channel_released(channel)
            if self._channels.get(channel.pv) is channel:
                del self._channels[channel.pv]
            session, handle = self._session, channel.handle
            if session is not None and handle is not None:
                await session.unsubscribe(handle)

    def _emit(self, channel: _Channel, event: PVValue) -> None:
        channel.current = event
        for listener in list(channel.listeners):
            try:
                listener(event)
            except Exception:
                logger.warning("A monitor callback failed for %s", channel.pv, exc_info=True)

    def _report_not_connected(self, channel: _Channel) -> None:
        channel.report_timer = None
        if channel.closed or channel.current is not None:
            return
        if self._session is not None and self._session.binding.state == STATE_UNSERVABLE:
            error = "no control system is reachable for this control target"
        else:
            error = f"not connected within {self._connect_report:g} s"
        self._emit(channel, _disconnected(channel.pv, error, channel.kind))

    def _on_value(self, channel: _Channel, reading: Any) -> None:
        if channel.closed:
            return
        channel.cancel_report()
        self._channel_up(channel)
        event = self._to_value(channel.pv, reading)
        channel.meta = self._to_meta(channel.pv, reading.metadata, event.value_kind)
        self._emit(channel, event)
        channel.first_value.set()

    def _on_connection(self, channel: _Channel, up: bool) -> None:
        """A drop after the channel was up is said at once.

        The ``False`` every subscription starts with is not: it is the
        :meth:`_report_not_connected` timer that turns a channel which stays
        down into one event. Either way the connected count follows.
        """
        if channel.closed:
            return
        if up:
            self._channel_up(channel)
            return
        self._channel_dropped(channel)
        if channel.current is not None and channel.current.connected:
            self._emit(channel, _disconnected(channel.pv, "disconnected", channel.kind))

    # -- conversion ---------------------------------------------------------

    def _to_value(self, pv: str, reading: Any) -> PVValue:
        metadata = reading.metadata
        raw = reading.value
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        error: str | None = None
        value: Any
        if getattr(metadata, "enum_labels", None) and not isinstance(raw, (str, list, tuple)):
            kind = ValueKind.ENUM
            value = raw if raw is None else int(raw)
        elif isinstance(raw, (list, tuple)):
            kind = ValueKind.WAVEFORM
            value = list(raw)
            if len(value) > self._max_array_elements:
                error = f"array truncated to {self._max_array_elements} of {len(value)} elements"
                value = value[: self._max_array_elements]
        elif isinstance(raw, bytes):
            kind, value = ValueKind.STRING, raw.decode("utf-8", errors="replace")
        elif isinstance(raw, str):
            kind, value = ValueKind.STRING, raw
        elif raw is None or isinstance(raw, (bool, int, float)):
            kind, value = ValueKind.SCALAR, raw
        else:
            kind, value = ValueKind.UNSUPPORTED, None
            error = f"a {type(raw).__name__} value cannot be shown"

        severity = getattr(metadata, "alarm_severity", None)
        return PVValue(
            pv=pv,
            value=value,
            value_kind=kind,
            # No severity reported is not an alarm; an unknown one is.
            severity=AlarmSeverity.NO_ALARM if severity is None else normalize_severity(severity),
            ts_ns=_timestamp_ns(reading.timestamp),
            connected=True,
            error=error,
        )

    @staticmethod
    def _to_meta(pv: str, metadata: Any, kind: ValueKind) -> PVMeta:
        labels = getattr(metadata, "enum_labels", None)
        return PVMeta(
            pv=pv,
            value_kind=kind,
            units=getattr(metadata, "units", None) or None,
            precision=getattr(metadata, "precision", None),
            description=getattr(metadata, "description", None),
            enum_choices=list(labels) if labels else None,
            display_low=getattr(metadata, "display_low", None),
            display_high=getattr(metadata, "display_high", None),
            control_low=getattr(metadata, "control_low", None),
            control_high=getattr(metadata, "control_high", None),
            warning_low=getattr(metadata, "warning_low", None),
            warning_high=getattr(metadata, "warning_high", None),
            alarm_low=getattr(metadata, "alarm_low", None),
            alarm_high=getattr(metadata, "alarm_high", None),
            writable=getattr(metadata, "writable", None),
        )

    # -- the PVConnector contract ---------------------------------------------

    async def get(self, pv: str, *, timeout: float | None = None) -> PVValue:
        """The value of *pv* now.

        A PV that is being monitored is answered from its subscription — the
        value it holds, or its first one, awaited up to the read timeout — so a
        screen's read never opens a second channel. Anything else is one read
        through the session. A PV that does not answer is a not-connected
        value, never an exception.
        """
        session = self._require_session()
        bound = self._read_timeout if timeout is None else timeout
        channel = self._channels.get(pv)
        if channel is not None:
            if channel.current is None:
                try:
                    await asyncio.wait_for(channel.first_value.wait(), bound)
                except TimeoutError:
                    return _disconnected(pv, f"not connected within {bound:g} s", channel.kind)
            return channel.current or _disconnected(pv, "not connected", channel.kind)
        try:
            async with asyncio.timeout(bound) as deadline:
                reading = await session.read(strip_protocol_prefix(pv))
        except SessionUnservableError:
            return _disconnected(pv, "no control system is reachable for this control target")
        except Exception as exc:
            if deadline.expired():
                return _disconnected(pv, f"no answer within {bound:g} s")
            return _disconnected(pv, f"read failed: {exc}")
        return self._to_value(pv, reading)

    async def put(
        self,
        pv: str,
        value: PVWriteValue,
        *,
        timeout: float | None = None,  # noqa: ARG002 - the session's own bounds apply
        wait: bool = True,  # noqa: ARG002 - never weakens the connector's confirm policy
        origin: Mapping[str, Any] | None = None,
        binding: str | None = None,
        confirmed: bool = False,
    ) -> None:
        """Write *value* to *pv* through the session, in the order CANOPY's write contract sets.

        A *binding* that is absent, null or not this process's ``binding.id``
        is refused as a changed target; the refusal is the session's, so it is
        audited and nothing reaches the connector. Otherwise the session
        writes, judged on the execution verdict at its admission: at ``asked``
        a write without *confirmed* is refused ``approval_missing``, at
        ``allowed`` *confirmed* is not read.

        The client's *wait* and *timeout* are ignored: the session's budget
        bounds the write, and its confirm policy is the connector's.

        Args:
            pv: The PV as the client spelled it; the session gets the bare name.
            value: The value to write.
            origin: The client's ``{screen, rev}``, for the audit.
            binding: The ``binding.id`` the client composed the write against.
            confirmed: Whether a person confirmed the write in the viewer's
                dialog; an older CANOPY never passes it, so an ``asked`` panel
                refuses its writes.

        Raises:
            PVWriteRefused, PVTargetChanged, PVSwitchInProgress, PVWriteMismatch,
            PVWriteUnconfirmed, PVWriteUnknown: The write did not end
                ``confirmed``; the detail is the text the client shows.
            PVConnectionFailed: No session is open.
        """
        session = self._require_session()
        address = strip_protocol_prefix(pv)
        if binding is None or binding != session.binding.id:
            result = session.refuse(
                address, value, REASON_SWITCHING, origin=origin, confirmed=confirmed
            )
        else:
            result = await session.write(address, value, origin=origin, confirmed=confirmed)
        _raise_for(pv, result)

    async def subscribe(self, pv: str, on_event: Callable[[PVValue], None]) -> _Monitor:
        """Deliver every value of *pv* to *on_event* until the handle is closed.

        A monitor gets the value the channel holds as soon as there is one. A
        channel that does not connect yields one not-connected event and then
        nothing until it does.
        """
        channel = await self._acquire(pv)
        channel.listeners.append(on_event)
        if channel.current is not None:
            try:
                on_event(channel.current)
            except Exception:
                logger.warning("A monitor callback failed for %s", pv, exc_info=True)
        return _Monitor(self, channel, on_event)

    async def describe(self, pv: str) -> PVMeta:
        """The static metadata of *pv*, as its subscription delivered it.

        Units, precision, description, enum choices, the display, control,
        warning and alarm limits, and whether the channel is writable.

        Waits up to the read timeout for the channel's first value and raises
        when none came, so a caller never caches an empty answer for a channel
        that was merely slow. A PV nobody monitors is subscribed for the length
        of the call.

        Raises:
            PVConnectionFailed: The session holds no connector.
            PVReadTimeout: The channel delivered nothing within the read timeout.
        """
        session = self._require_session()
        if session.binding.state == STATE_UNSERVABLE:
            raise PVConnectionFailed(
                f"{pv}: no control system is reachable for this control target"
            )
        channel = await self._acquire(pv)
        try:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(channel.first_value.wait(), self._read_timeout)
            if channel.meta is None:
                raise PVReadTimeout(f"{pv}: no metadata within {self._read_timeout:g} s")
            return channel.meta
        finally:
            await self._release(channel)
