"""Per-conversation history so follow-up questions carry their context.

Each dispatch run is stateless — without this, "now plot it over 24h" arrives
at the agent with no idea what "it" is. The bridge is the only component that
sees the whole conversation (it handles every question and posts every answer),
so it ships the recent turns along with each new question in the webhook
payload. The agent itself is unchanged: it simply receives the conversation so
far, the same context it would naturally have in an interactive session.

**Where a turn lives.** Every settled turn is one ``bridge_turn`` entry in the
agent record, appended through the spool writer
(:func:`osprey.agent_record.writer.write`) into the bridge's own spool
directory. The entry body is ``{conversation, run_id, question, artifacts,
status}``, plus ``asked_by`` = ``{id, name}`` when the bridge knew who asked.
``status`` is the bridge's own outcome: :data:`TURN_ANSWERED` when the answer
reached the conversation, :data:`TURN_FAILED` when the request produced no
answer. The answer text is not copied into the turn: it is the run's own final
answer, which the record holds once, in the run's session. An answered turn
names its run as the entry's ``run_id``, so the read joins the two; a failed
turn names no run, so its replay needs no run lookup. Keying is channel-defined
(a Chat thread/space, an email sender). The record keeps entries for
``agent_record.retention_days``, which bounds how far back a conversation can
be replayed.

**Reading it back.** :meth:`ConversationHistory.read` asks the record's
``bridge_history`` for one conversation with a drain, so a turn the bridge
appended a moment ago, and the run entries and transcript it names, are
committed before the read answers. The bridge's login reads under its own
``Source`` scope: it sees its own turns and the runs they name, never another
bridge's. A read the record could not answer, or answered incomplete, yields
no turns and says which (:data:`RECORD_UNAVAILABLE` / :data:`RECORD_INCOMPLETE`),
so the pipeline proceeds without history and tells the agent so.

A replayed turn is ``{question, answer, ts, run_id, artifacts}`` (``ts`` in
epoch seconds, the retry queue's ``queued_at`` clock), with ``asked_by`` when
recorded. A failed turn replays its question with :data:`FAILED_TURN_ANSWER`
so a follow-up ("try again") still resolves its referent. :func:`replay_window`
applies the turn and character budgets and can replay a long answer shortened:
its opening and a note naming the run that answered it, so the agent can read
the whole answer back when a follow-up needs it.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, TypeGuard

from osprey.agent_record import writer
from osprey.agent_record.client import ClientAnswer, RecordClient
from osprey.agent_record.contract import KIND_BRIDGE_TURN, Drain

logger = logging.getLogger(__name__)

__all__ = [
    "FAILED_TURN_ANSWER",
    "MAX_ARTIFACTS_PER_TURN",
    "MAX_CHARS",
    "MAX_TURNS",
    "RECORD_INCOMPLETE",
    "RECORD_UNAVAILABLE",
    "SHORTENED_ANSWER_NOTE",
    "TURN_ANSWERED",
    "TURN_FAILED",
    "ConversationHistory",
    "HistoryRead",
    "replay_window",
    "shorten_answer",
    "turn_body",
]

# Bounds on what we replay into the payload. The CHAR budget is the primary
# knob; the turn count is a runaway backstop, NOT the intended limiter.
#
# Why bound this at all — the window is replayed as fresh input tokens on EVERY
# follow-up (each dispatch run is stateless, so there is no cross-run prompt
# cache to amortize it), and a DM's history key is its space, which never
# rotates: a DM is one unbounded, ever-growing conversation.
#
# Why the char budget leads, and the turn count does not. Turn count is only a
# proxy for size, and a poor one: observed turns span ~370-4,100 chars, so "N
# turns" can differ by an order of magnitude in what actually rides the prompt.
# The char budget measures the thing we care about directly. The turn cap is
# also the most a read asks the record for; if it is ever the binding
# constraint, that is a bug in these numbers, not the design working.
#
# Sized against a measured deployment rather than intuition: a real 30-turn DM
# serializes to ~19k chars (~5k tokens), the largest single conversation to
# ~24k chars — a few percent of a >=200k-token context. The budget below is
# therefore set so ordinary working conversations are never trimmed by size at
# all, while one pathological thread still cannot dominate the worker's prompt.
MAX_TURNS = 100
MAX_CHARS = 100000

# Per-turn cap on carried artifact descriptors. One run can emit many plots; we
# keep the most recent handful so the payload (and the char budget) stays
# bounded. We retain the TAIL — descriptors arrive oldest-first, so the last N
# are the run's most recent outputs, the ones a "show me that again" follow-up
# most likely means.
MAX_ARTIFACTS_PER_TURN = 10

# Placeholder replayed in the "answer" slot for a turn that did not produce an
# answer (it timed out, errored, or returned no text). We keep the QUESTION in
# history so a follow-up ("do that but take 30 min", "try again") can resolve its
# referent — without a marker turn the failed request leaves no trace at all and
# the agent is honestly told the thread is empty. The marker is a NON-answer: it
# must never read as a fabricated result the agent could cite.
FAILED_TURN_ANSWER = "[This request did not produce an answer — it timed out or errored.]"

# The note that ends a long answer replayed shortened. Prose for a reader, not a
# token: nothing matches on the wording. The payload it rides in is documented
# at docs/source/reference/contracts/bridge-dispatch.rst.
SHORTENED_ANSWER_NOTE = (
    "[Answer shortened; it is {chars:,} characters in full. "
    'prior_answer_read(run_id="{run_id}") returns the whole answer.]'
)

#: A turn whose answer reached the conversation.
TURN_ANSWERED = "answered"
#: A turn whose request produced no answer.
TURN_FAILED = "failed"

#: The record did not answer the history read.
RECORD_UNAVAILABLE = "unavailable"
#: The record answered, but something the history depends on was missing.
RECORD_INCOMPLETE = "incomplete"


class HistoryRecord(Protocol):
    """The one read :class:`ConversationHistory` makes of the record."""

    def bridge_history(
        self,
        *,
        conversation: str | None = None,
        limit: int | None = None,
        wait: Drain | None = None,
    ) -> ClientAnswer: ...


Writer = Callable[..., str | None]


def shorten_answer(answer: str, run_id: str, limit: int) -> str | None:
    """Return ``answer`` as its opening and a note naming ``run_id``, or ``None``.

    ``None`` when ``limit`` is not positive, when the answer is at or under it,
    or when the shortened form would not be shorter. One key sets both numbers:
    the opening is half the threshold. A line break in the second half of the
    opening ends it there, so a table keeps whole rows.
    """
    if limit <= 0 or len(answer) <= limit:
        return None
    keep = limit // 2
    cut = answer.rfind("\n", 0, keep + 1)
    if cut < keep // 2:
        cut = keep
    shortened = (
        answer[:cut].rstrip()
        + "\n"
        + SHORTENED_ANSWER_NOTE.format(chars=len(answer), run_id=run_id)
    )
    return shortened if len(shortened) < len(answer) else None


def replay_window(
    turns: Sequence[Mapping[str, Any]],
    *,
    max_turns: int = MAX_TURNS,
    max_chars: int = MAX_CHARS,
    shorten_over: int = 0,
) -> list[dict[str, Any]]:
    """The turns to replay, oldest first, within the turn and character budgets.

    Applies the turn cap and then the char budget (dropping oldest first), so
    the newest exchanges always survive. The char budget counts the FULL
    serialized turn — descriptors included — because the whole turn is what
    rides the payload, so an answer with many/large artifact descriptors must
    be charged for them, not just its question+answer text.

    ``shorten_over`` > 0 replays each answer longer than that many characters
    as :func:`shorten_answer` builds it, with ``answer_chars`` set to the full
    length. Only a turn with a ``run_id`` is shortened, since nothing else could
    be read back. Shortening comes before the char budget, so one long answer
    costs fewer whole turns. It works on per-call copies: *turns* is never
    changed.
    """
    window = [dict(t) for t in turns[-max_turns:]] if max_turns > 0 else []
    if shorten_over > 0:
        for turn in window:
            run_id = turn.get("run_id")
            answer = turn.get("answer")
            if not (isinstance(run_id, str) and run_id and isinstance(answer, str)):
                continue
            shortened = shorten_answer(answer, run_id, shorten_over)
            if shortened is not None:
                turn["answer"] = shortened
                turn["answer_chars"] = len(answer)
    kept: list[dict[str, Any]] = []
    used = 0
    for turn in reversed(window):
        size = len(json.dumps(turn, ensure_ascii=False, default=str))
        if kept and used + size > max_chars:
            break
        kept.append(turn)
        used += size
    kept.reverse()
    return kept


def turn_body(
    key: str,
    question: str,
    *,
    status: str,
    run_id: str | None = None,
    artifacts: Sequence[Mapping[str, Any]] | None = None,
    asked_by: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``bridge_turn`` body for one settled turn (see the module docstring).

    ``artifacts`` is capped to :data:`MAX_ARTIFACTS_PER_TURN` (tail kept);
    ``asked_by`` is recorded only when it names someone.
    """
    body: dict[str, Any] = {
        "conversation": key,
        "run_id": run_id,
        "question": question,
        "artifacts": _cap_artifacts([dict(a) for a in artifacts or ()]),
        "status": status,
    }
    if _valid_asker(asked_by):
        body["asked_by"] = _asker_record(asked_by)
    return body


@dataclass(frozen=True)
class HistoryRead:
    """One conversation's turns as the record answered them, oldest first.

    ``record`` is ``None`` when the read was complete, and otherwise
    :data:`RECORD_UNAVAILABLE` or :data:`RECORD_INCOMPLETE`; ``turns`` is then
    empty. ``reason`` is the record's own explanation, for the log.
    """

    turns: list[dict[str, Any]] = field(default_factory=list)
    record: str | None = None
    reason: str = ""


class ConversationHistory:
    """A bridge's conversation history, kept in the agent record.

    Args:
        client: The bridge's record client; :meth:`RecordClient.from_env` over
            the process environment when ``None``.
        write: The spool writer; :func:`osprey.agent_record.writer.write` when
            ``None``.
        max_turns: The most turns one read asks for and replays.
        max_chars: The character budget of one replay.
    """

    def __init__(
        self,
        client: HistoryRecord | None = None,
        *,
        write: Writer | None = None,
        max_turns: int = MAX_TURNS,
        max_chars: int = MAX_CHARS,
    ) -> None:
        self.client: HistoryRecord = (
            client if client is not None else RecordClient.from_env(os.environ)
        )
        self._write = write
        self.max_turns = max_turns
        self.max_chars = max_chars

    # --- read ----------------------------------------------------------------
    def read(self, key: str) -> HistoryRead:
        """The conversation's turns, oldest first, drained and complete — or none.

        Never raises: a read the record could not answer, or answered
        incomplete, is logged and answered with no turns and the reason.
        """
        if not key:
            return HistoryRead()
        try:
            answer = self.client.bridge_history(
                conversation=key, limit=self.max_turns, wait=Drain()
            )
        except Exception as exc:  # the client never raises; a double might
            answer = ClientAnswer.failed(f"{type(exc).__name__}: {exc}")
        if not answer.available:
            logger.warning("conversation history unavailable for %s: %s", key, answer.unavailable)
            return HistoryRead(record=RECORD_UNAVAILABLE, reason=answer.unavailable or "")
        if not answer.complete:
            logger.warning("conversation history incomplete for %s: %s", key, answer.reason)
            return HistoryRead(record=RECORD_INCOMPLETE, reason=answer.reason)
        turns = [t for t in map(_replayed_turn, reversed(list(answer.data or []))) if t]
        return HistoryRead(turns=turns)

    def cut(
        self, turns: Sequence[Mapping[str, Any]], *, shorten_over: int = 0
    ) -> list[dict[str, Any]]:
        """:func:`replay_window` under this history's budgets."""
        return replay_window(
            turns, max_turns=self.max_turns, max_chars=self.max_chars, shorten_over=shorten_over
        )

    # --- write ---------------------------------------------------------------
    def append(
        self,
        key: str,
        question: str,
        *,
        run_id: str | None,
        artifacts: Sequence[Mapping[str, Any]] | None = None,
        asked_by: Mapping[str, Any] | None = None,
    ) -> str | None:
        """Record one answered turn, naming the run whose answer it was.

        Returns the entry id, or ``None`` when nothing was recorded (no key, or
        the writer could not store the line).
        """
        if not key:
            return None
        body = turn_body(
            key,
            question,
            status=TURN_ANSWERED,
            run_id=run_id,
            artifacts=artifacts,
            asked_by=asked_by,
        )
        return self._emit(body, run_id=run_id) if run_id else self._emit(body)

    def append_failed(
        self,
        key: str,
        question: str,
        asked_by: Mapping[str, Any] | None = None,
        *,
        run_id: str | None = None,
    ) -> str | None:
        """Record a turn that produced no answer.

        Its replay is the question with :data:`FAILED_TURN_ANSWER`, so the next
        question in this conversation can still resolve its referent. The body
        keeps the run it tried, if any, but the entry names no run, so the
        replay needs no run lookup. An empty question carries no referent, so it
        is not recorded.
        """
        if not key or not question:
            return None
        body = turn_body(key, question, status=TURN_FAILED, run_id=run_id, asked_by=asked_by)
        return self._emit(body)

    def _emit(self, body: Mapping[str, Any], **ids: Any) -> str | None:
        write = self._write if self._write is not None else writer.write
        return write(KIND_BRIDGE_TURN, body, **ids)


# --- turn assembly -------------------------------------------------------------------


def _replayed_turn(row: Any) -> dict[str, Any] | None:
    """One ``bridge_history`` row as the turn the payload replays, or ``None``."""
    if not isinstance(row, Mapping):
        return None
    turn = row.get("turn")
    if not isinstance(turn, Mapping):
        return None
    body = turn.get("body")
    if not isinstance(body, Mapping):
        return None
    question = body.get("question")
    answer = FAILED_TURN_ANSWER
    run_id: str | None = None
    artifacts: list[dict[str, Any]] = []
    if body.get("status") == TURN_ANSWERED:
        run = row.get("run")
        text = run.get("answer") if isinstance(run, Mapping) else None
        answer = text if isinstance(text, str) else ""
        rid = turn.get("run_id")
        run_id = rid if isinstance(rid, str) and rid else None
        raw = body.get("artifacts")
        artifacts = (
            _cap_artifacts([dict(a) for a in raw if isinstance(a, Mapping)])
            if (isinstance(raw, list))
            else []
        )
    replayed: dict[str, Any] = {
        "question": question if isinstance(question, str) else "",
        "answer": answer,
        "ts": _epoch(turn.get("ts")),
        "run_id": run_id,
        "artifacts": artifacts,
    }
    asked_by = body.get("asked_by")
    if _valid_asker(asked_by):
        replayed["asked_by"] = _asker_record(asked_by)
    return replayed


def _epoch(value: Any) -> float | None:
    """An ISO-8601 entry timestamp as epoch seconds; ``None`` when unreadable."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _valid_asker(value: Any) -> TypeGuard[Mapping[str, Any]]:
    """A mapping whose ``id`` and ``name`` are each ``str`` or ``None``, with at least
    one a non-empty ``str``."""
    if not isinstance(value, Mapping):
        return False
    ident, name = value.get("id"), value.get("name")
    if not all(v is None or isinstance(v, str) for v in (ident, name)):
        return False
    return bool(ident) or bool(name)


def _asker_record(value: Mapping[str, Any]) -> dict[str, str | None]:
    """Normalise a valid asker to exactly ``{"id", "name"}``."""
    return {"id": value.get("id"), "name": value.get("name")}


def _cap_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep at most :data:`MAX_ARTIFACTS_PER_TURN` descriptors, retaining the
    tail (most recent). Descriptors are round-tripped opaquely."""
    if len(artifacts) > MAX_ARTIFACTS_PER_TURN:
        return list(artifacts[-MAX_ARTIFACTS_PER_TURN:])
    return list(artifacts)
