"""The dedup marker: the innermost recording layer owns the decision.

Two of Osprey's safety layers can see the same operation. The MCP audit
middleware wraps every ``tools/call``; the tool's own gates run inside it. The
HTTP mutation layer wraps every state-changing request; a route's own recorder
would run inside it. Without coordination the outer layer either files a second
record for a decision the inner one already filed, or — the case that actually
loses information — files ``allowed`` for a call an inner guard *refused* while
still returning a successful result. The executor's runtime guard is exactly
that shape: it fires inside the subprocess, the tool reports the refusal and
hands back whatever the script legitimately produced, and nothing the
middleware can see about the result says a write was stopped.

**The rule.** The innermost layer that records a decision owns it. It sets a
marker; every layer outside it defers and files nothing.

**The marker carries the decision, not a flag.** ``(decision, reason)`` rather
than "handled", so the outer layer defers to a specific answer and can log a
disagreement — a tool that raised while its inner recorder said ``allowed`` is
a bug worth a line in the log — instead of skipping blindly.

**Same process, same call, and nothing wider.** The carrier is a
:class:`~contextvars.ContextVar`, which is per-task in asyncio and *copied* into
each new task. That single property is what bounds the mechanism, and it cuts
both ways — the two hazards below are the mirror image of each other, and only
one of them is loud.

*Too wide — a marker seen where it should not be (false suppression).* A
``fork`` copies the whole context, so a child would otherwise inherit its
parent's marker and silently suppress its own first record. Two guards:
:func:`decision_scope`, which the outer layer wraps one call in so a marker can
neither arrive from before the call nor survive past it, exception or not; and
the recorded pid, which :func:`recorded_decision` checks so that a marker is
believed only from the process that set it.

*Too narrow — a marker set where the outer layer cannot see it (false
``allowed``).* **The inner recorder must run on the task the outer layer is
awaiting.** A copied context is not a shared one: a mark set on a *child* task
is invisible to the parent that started it. So an inner recorder reached
through ``asyncio.create_task``/``gather``, through ``to_thread`` /
``run_in_threadpool``, or inside a *synchronous* tool body that the server runs
on a worker thread (FastMCP hands sync tools to ``anyio.to_thread.run_sync``;
Starlette does the same for a ``def`` route) records its refusal and the outer
layer, seeing no marker, files ``allowed`` on top of it. Nothing raises; the
ledger simply grows a second, wrong line. A recorder in that position must
either be moved onto the awaiting task (make the tool or route ``async def``
and call it inline) or record through a seam the outer layer does not wrap.
:mod:`tests.audit.test_dedup_contract` pins this limitation rather than papering
over it — a mark behind ``create_task`` is *not* seen, on purpose, because
believing it would mean reaching across tasks that may be different calls.

The subprocess hook emitters need none of this: they run in their own process
against ``hook_<name>`` surfaces no in-process layer writes to, so there is
nothing for them to be deduplicated against.

**Marking is paired with recording.** :func:`record_and_mark` is the entry
point an inner recorder uses, and it is the only *supported* way the two happen
together — a layer that marks without recording silences the outer layer and
leaves the ledger with nothing at all. (:func:`mark_recorded` is reachable on
its own for the two callers that legitimately need it: a layer that already
wrote its record by another route, and an outer layer re-asserting a marker it
deferred to so that a layer above *it* defers to the same answer.) It marks
even when the durable write fails: the decision was still made and owned here,
and letting a failed write hand the call back to an outer layer would produce
an ``allowed`` record for a refused operation, which is the one outcome this
module exists to prevent. The marker carries ``stored`` so an outer layer can
tell that case apart and, where its own record would not contradict the inner
decision, file one rather than leave the refusal with no line at all.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.audit import writer

logger = logging.getLogger(__name__)

__all__ = [
    "RecordedDecision",
    "clear_recorded",
    "decision_scope",
    "mark_recorded",
    "record_and_mark",
    "recorded_decision",
]


@dataclass(frozen=True, slots=True)
class RecordedDecision:
    """What an inner layer recorded, and which process recorded it.

    Frozen for the same reason the envelope is: the marker is read by a layer
    that must not be able to edit the decision it is deferring to.

    :param decision: The decision the inner layer filed — see
        :data:`~osprey.audit.envelope.DECISIONS`.
    :param reason: The short machine-ish reason it filed with.
    :param pid: The process that set the marker. A forked child inherits the
        context but not the authority to speak for it.
    :param stored: Whether the inner layer's record was durably written. The
        decision is owned here either way — that is why the marker is set at
        all — but a ``False`` tells an outer layer that deferring silently
        would leave the ledger with no line, so it may file its own record when
        that record would not contradict :attr:`decision`.
    """

    decision: str
    reason: str
    pid: int
    stored: bool = True


def _current_pid() -> int:
    """This process's id — the one seam the fork guard is tested through.

    A function rather than an inline ``os.getpid()`` so a test can simulate the
    only situation the guard exists for: a context that crossed a ``fork``.
    Nothing else calls it.
    """
    return os.getpid()


#: The per-call carrier. Module-level and private: every read goes through
#: :func:`recorded_decision`, which is where the pid guard lives, and a caller
#: reaching around it would get a marker this module has already decided not to
#: believe.
_MARKER: ContextVar[RecordedDecision | None] = ContextVar(
    "osprey_audit_recorded_decision", default=None
)


def mark_recorded(decision: str, reason: str, *, stored: bool = True) -> RecordedDecision | None:
    """Claim this call's decision for the current layer. Returns the marker, or ``None``.

    ``None`` means nothing was marked, which happens only when *decision* or
    *reason* is empty — an unusable marker would tell the outer layer to stay
    quiet without telling it what it is staying quiet about, so the outer
    layer records instead. Both fields are required for exactly that reason.

    **Marking is a claim that this decision is already in the ledger.** Nothing
    here checks that, and a caller that marks without recording silences every
    layer outside it — the worst outcome in this module's vocabulary. Use
    :func:`record_and_mark`, which cannot mark without also recording, unless
    you are one of the two callers this seam exists for: a layer that already
    wrote its record by another route, or an outer layer re-asserting a marker
    it deferred to (pass the inner marker's *stored* through, so the layer
    above learns the same thing this one did).

    :param stored: Whether the record this marker speaks for was durably
        written. Leave it ``True`` for a caller that wrote its own record; pass
        the inner value through when re-asserting someone else's.
    """
    if not decision or not reason:
        logger.warning(
            "Refusing to set an audit dedup marker with decision=%r reason=%r; "
            "the outer layer will record this call itself",
            decision,
            reason,
        )
        return None
    marker = RecordedDecision(decision=decision, reason=reason, pid=_current_pid(), stored=stored)
    _MARKER.set(marker)
    return marker


def recorded_decision() -> RecordedDecision | None:
    """The decision an inner layer already recorded for this call, or ``None``.

    ``None`` is the outer layer's cue to record the call itself. A marker set
    by a *different* process is treated as absent: it can only have arrived by
    a ``fork`` copying the context, and the child's own first decision would
    otherwise go unrecorded.
    """
    marker = _MARKER.get()
    if marker is None:
        return None
    if marker.pid != _current_pid():
        logger.debug(
            "Ignoring an audit dedup marker inherited from pid %d; this process records "
            "its own decisions",
            marker.pid,
        )
        return None
    return marker


def clear_recorded() -> None:
    """Forget any marker in the current context.

    Used by :func:`decision_scope` and by tests. Not an invalidation anyone on
    the running path has to remember to perform — the scope is.
    """
    _MARKER.set(None)


@contextlib.contextmanager
def decision_scope() -> Iterator[None]:
    """Bound one call's marker to this block.

    Entering clears whatever the context carried in, so a marker left by an
    earlier call — or by a caller that is not a recording layer at all — cannot
    silence this one. Leaving restores what was there, on the way out of an
    exception too, so a refusal that propagates as a ``ToolError`` does not
    leave its marker behind for the next call on the same task.

    The outer layer wraps its ``call_next`` in this; inner recorders do not
    need it.
    """
    token = _MARKER.set(None)
    try:
        yield
    finally:
        _MARKER.reset(token)


def record_and_mark(*, decision: str, reason: str, **fields: Any) -> Path | None:
    """File one record as the innermost layer and claim the decision.

    The entry point for a layer that decides *inside* another one. Everything
    but *decision* and *reason* is passed straight to
    :func:`~osprey.audit.writer.record`, which fills ``actor`` from the
    identity ladder and swallows its own errors; those two are named
    explicitly because the marker carries them.

    The marker is set whether or not the write landed. A record that could not
    be stored is still a decision this layer made and owns, and handing the
    call back to the outer layer at that point would file ``allowed`` for an
    operation that was refused — a wrong record where the failure had already
    cost us the right one. The marker says which of the two happened
    (:attr:`RecordedDecision.stored`), so an outer layer whose own record
    *agrees* with this one can still file it and keep the refusal from vanishing
    entirely.

    Returns whatever :func:`~osprey.audit.writer.record` returned: the ledger
    path, or ``None`` if the record was not durably stored.
    """
    # Called through the module, never as a name imported at module scope:
    # a from-import snapshots whatever `osprey.audit.writer.record` was at
    # the moment this module happened to be first imported, which in a test
    # process can be a stub some other test installed. Resolving the
    # attribute per call is also what lets a test redirect the writer for
    # the inner recorder and the outer layer alike.
    path = writer.record(decision=decision, reason=reason, **fields)
    mark_recorded(decision, reason, stored=path is not None)
    return path
