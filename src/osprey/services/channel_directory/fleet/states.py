"""The server fleet's state model: one IOC's inputs in, one state, a reason and findings out.

Pure: no I/O, no clock, no connector. The refresher gathers each IOC's inputs for
a cycle and calls :func:`classify_fleet`; every surface orders rows with
:func:`order_key`.

States (one per IOC):

* ``alive`` -- the liveness PV answered and, where it is a counter, moved since
  the previous reading.
* ``stalled`` -- it answered and the counter has not moved.
* ``silent`` -- it gave no answer now and has answered before through this read
  path.
* ``unreached`` -- it has never answered through this read path.
* ``unknown`` -- the read failed, the read path is down, the IOC has no liveness
  PV, a counter has only one reading so far, or the read path is unverified and
  every IOC that answered last cycle went quiet at once.

Read outcomes (:func:`classify_read`): a read that ends in
``ChannelUnreachableError`` or a timeout is *no answer* and feeds ``silent`` or
``unreached``; ``ConnectorUnavailableError`` is *connector loss*; any other
exception is *read failed*. Both of those feed ``unknown``, never a quiet state.

Findings: ``degraded`` (alive, a health reading out of range; an unread health
PV is not one) and, only with a roster source configured,
``expected_running_not_alive`` (``stalled``, ``silent`` or ``unreached``),
``expected_stopped_alive`` and ``alive_unexpected`` (alive, no source expects it
to run or stop). ``unknown`` never carries a finding.

Two counter readings closer together than the minimum interval are never judged
moved or not moved: the row keeps its previous ``alive`` or ``stalled`` state
(``unknown`` when it had neither), and the verdict says not to replace the
stored reading, so the next comparison spans a full interval.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from osprey_connectors.errors import ChannelUnreachableError, ConnectorUnavailableError

__all__ = [
    "EXPECTATIONS",
    "MIN_COUNTER_INTERVAL_S",
    "Finding",
    "HealthReading",
    "HealthStatus",
    "ServerInputs",
    "LivenessKind",
    "ReadOutcome",
    "ReadPath",
    "Reading",
    "State",
    "Verdict",
    "classify",
    "classify_fleet",
    "classify_read",
    "fleet_went_quiet",
    "order_key",
]

MIN_COUNTER_INTERVAL_S = 2.5
"""Counter readings closer together than this keep the row's previous state."""

EXPECTATIONS = ("running", "stopped", None)
"""What a roster source may expect of an IOC; ``None`` is no expectation."""


class State(StrEnum):
    ALIVE = "alive"
    STALLED = "stalled"
    SILENT = "silent"
    UNREACHED = "unreached"
    UNKNOWN = "unknown"


class Finding(StrEnum):
    DEGRADED = "degraded"
    EXPECTED_RUNNING_NOT_ALIVE = "expected_running_not_alive"
    EXPECTED_STOPPED_ALIVE = "expected_stopped_alive"
    ALIVE_UNEXPECTED = "alive_unexpected"


class ReadOutcome(StrEnum):
    """How one liveness read ended (C7), plus ``no_channel`` for an IOC with nothing to read."""

    ANSWERED = "answered"
    NO_ANSWER = "no_answer"
    READ_FAILED = "read_failed"
    CONNECTOR_LOSS = "connector_loss"
    NO_CHANNEL = "no_channel"


class ReadPath(StrEnum):
    """The target's read-path PV this cycle: answered, unanswered, or none configured."""

    OK = "ok"
    DOWN = "down"
    UNVERIFIED = "unverified"


class LivenessKind(StrEnum):
    """``counter`` must move between readings; ``value`` is alive whenever it answers."""

    COUNTER = "counter"
    VALUE = "value"


class HealthStatus(StrEnum):
    OK = "ok"
    OUT_OF_RANGE = "out_of_range"
    UNREAD = "unread"


@dataclass(frozen=True, kw_only=True)
class Reading:
    """A liveness value and when it was read (seconds, any monotonic-enough epoch)."""

    value: Any
    read_at: float


@dataclass(frozen=True, kw_only=True)
class HealthReading:
    channel: str
    value: Any
    status: HealthStatus


@dataclass(frozen=True, kw_only=True)
class ServerInputs:
    """Everything the model needs about one IOC for one cycle.

    ``answer`` is this cycle's reading (required when ``outcome`` is
    ``answered``); ``previous`` the stored reading; ``previous_state`` the
    state last published; ``seen`` whether it has ever answered through this
    read path; ``answered_last_cycle`` whether it answered in the previous
    cycle. ``expected`` is the merged roster expectation and
    ``roster_configured`` whether any roster source applies to the target.
    ``detail`` is the failure's text, carried into the reason.
    """

    name: str
    outcome: ReadOutcome
    kind: LivenessKind = LivenessKind.COUNTER
    answer: Reading | None = None
    previous: Reading | None = None
    previous_state: State | None = None
    seen: bool = False
    answered_last_cycle: bool = False
    health: tuple[HealthReading, ...] = ()
    expected: str | None = None
    roster_configured: bool = False
    detail: str = ""

    def __post_init__(self) -> None:
        if self.outcome is ReadOutcome.ANSWERED and self.answer is None:
            raise ValueError(f"{self.name}: an answered read needs its answer")
        if self.expected not in EXPECTATIONS:
            raise ValueError(f"{self.name}: expected must be one of {EXPECTATIONS}")


@dataclass(frozen=True, kw_only=True)
class Verdict:
    """One IOC's state for a cycle.

    ``advance_reading`` tells the caller whether this cycle's answer replaces
    the stored reading: false when nothing was read, or when the readings were
    too close to judge.
    """

    state: State
    reason: str
    findings: tuple[Finding, ...] = ()
    advance_reading: bool = False


def classify_read(exc: BaseException | None) -> ReadOutcome:
    """The C7 class of a liveness read that raised ``exc`` (``None``: it answered).

    A cancelled read is not an outcome; the refresher never cancels one, so
    being handed one is a caller error.
    """
    if exc is None:
        return ReadOutcome.ANSWERED
    if not isinstance(exc, Exception):
        raise TypeError(f"not a read outcome: {type(exc).__name__}")
    if isinstance(exc, ConnectorUnavailableError):
        return ReadOutcome.CONNECTOR_LOSS
    if isinstance(exc, ChannelUnreachableError | TimeoutError):
        return ReadOutcome.NO_ANSWER
    return ReadOutcome.READ_FAILED


def fleet_went_quiet(iocs: Iterable[ServerInputs], *, read_path: ReadPath) -> bool:
    """True when, with no read-path PV, every IOC that answered last cycle gave no answer now."""
    if read_path is not ReadPath.UNVERIFIED:
        return False
    previous = [i for i in iocs if i.answered_last_cycle]
    return bool(previous) and all(i.outcome is ReadOutcome.NO_ANSWER for i in previous)


def classify_fleet(
    iocs: Sequence[ServerInputs],
    *,
    read_path: ReadPath,
    min_interval_s: float = MIN_COUNTER_INTERVAL_S,
) -> list[Verdict]:
    """One verdict per IOC, in the given order, applying the whole-cycle rule."""
    quiet = fleet_went_quiet(iocs, read_path=read_path)
    return [
        classify(i, read_path=read_path, fleet_quiet=quiet, min_interval_s=min_interval_s)
        for i in iocs
    ]


def classify(
    ioc: ServerInputs,
    *,
    read_path: ReadPath,
    fleet_quiet: bool = False,
    min_interval_s: float = MIN_COUNTER_INTERVAL_S,
) -> Verdict:
    """One IOC's state, reason and findings for this cycle.

    ``fleet_quiet`` is :func:`fleet_went_quiet` over the whole cycle;
    :func:`classify_fleet` computes it.
    """
    if read_path is ReadPath.DOWN:
        return Verdict(state=State.UNKNOWN, reason="read path down; nothing was read")
    outcome = ioc.outcome
    if outcome is ReadOutcome.NO_CHANNEL:
        return Verdict(state=State.UNKNOWN, reason="no liveness PV")
    if outcome is ReadOutcome.READ_FAILED:
        return Verdict(state=State.UNKNOWN, reason=_with_detail("read failed", ioc.detail))
    if outcome is ReadOutcome.CONNECTOR_LOSS:
        return Verdict(state=State.UNKNOWN, reason=_with_detail("connector lost", ioc.detail))
    if outcome is ReadOutcome.NO_ANSWER:
        if fleet_quiet and ioc.answered_last_cycle:
            return Verdict(
                state=State.UNKNOWN,
                reason="read path unverified and every answering IOC went quiet at once",
            )
        if ioc.seen:
            return _judged(ioc, State.SILENT, "no answer; answered before")
        return _judged(ioc, State.UNREACHED, "never answered on this read path")
    return _answered(ioc, min_interval_s)


def _answered(ioc: ServerInputs, min_interval_s: float) -> Verdict:
    answer = ioc.answer
    assert answer is not None  # guaranteed by ServerInputs
    if ioc.kind is LivenessKind.VALUE:
        return _judged(ioc, State.ALIVE, "answered", advance=True)
    previous = ioc.previous
    if previous is None:
        return Verdict(
            state=State.UNKNOWN, reason="one reading of the counter so far", advance_reading=True
        )
    if answer.read_at - previous.read_at < min_interval_s:
        reason = "counter readings too close to judge; state kept"
        if ioc.previous_state in (State.ALIVE, State.STALLED):
            return _judged(ioc, ioc.previous_state, reason)
        return Verdict(state=State.UNKNOWN, reason=reason)
    if answer.value != previous.value:
        return _judged(ioc, State.ALIVE, "counter moved", advance=True)
    return _judged(ioc, State.STALLED, "answers; counter not moved", advance=True)


def _judged(ioc: ServerInputs, state: State, reason: str, *, advance: bool = False) -> Verdict:
    return Verdict(
        state=state, reason=reason, findings=_findings(ioc, state), advance_reading=advance
    )


def _findings(ioc: ServerInputs, state: State) -> tuple[Finding, ...]:
    found: list[Finding] = []
    if state is State.ALIVE:
        if any(h.status is HealthStatus.OUT_OF_RANGE for h in ioc.health):
            found.append(Finding.DEGRADED)
        if ioc.roster_configured:
            if ioc.expected == "stopped":
                found.append(Finding.EXPECTED_STOPPED_ALIVE)
            elif ioc.expected is None:
                found.append(Finding.ALIVE_UNEXPECTED)
    elif ioc.roster_configured and ioc.expected == "running":
        found.append(Finding.EXPECTED_RUNNING_NOT_ALIVE)
    return tuple(found)


def _with_detail(reason: str, detail: str) -> str:
    return f"{reason}: {detail}" if detail else reason


_RANK_PLAIN = {
    State.STALLED: 0,
    State.SILENT: 1,
    State.UNKNOWN: 4,
    State.UNREACHED: 5,
    State.ALIVE: 6,
}
_RANK_WITH_FINDING = {State.UNREACHED: 2, State.ALIVE: 3}


def order_key(
    state: State | str, findings: Sequence[Finding | str], name: str
) -> tuple[int, int, str]:
    """C4 order: stalled, silent, unreached with a finding, alive with a finding,
    unknown, unreached, alive; within a state, rows with a finding first, then by name."""
    state = State(state)
    rank = _RANK_WITH_FINDING.get(state, _RANK_PLAIN[state]) if findings else _RANK_PLAIN[state]
    return (rank, 0 if findings else 1, name)
