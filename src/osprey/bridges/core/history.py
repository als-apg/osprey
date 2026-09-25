"""Per-conversation history so follow-up questions carry their context.

Each dispatch run is stateless — without this, "now plot it over 24h" arrives
at the agent with no idea what "it" is. The bridge is the only component that
sees the whole conversation (it handles every question and posts every answer),
so it owns the transcript and ships the recent turns along with each new
question in the webhook payload. The agent itself is unchanged: it simply
receives the conversation so far, the same context it would naturally have in
an interactive session.

Keying is channel-defined (a Chat thread/space, an email sender). Persistence
lives in :class:`~osprey.bridges.core.store.JsonFileStore` (same volume and
durability guarantees as the dedup store) so a bridge restart keeps
conversations.

A turn is ``{question, answer, ts, run_id, artifacts}``: the ``ts`` (epoch
seconds, matching the retry queue's ``queued_at`` clock) anchors the 180-day
age-prune; ``run_id`` ties the turn back to the run that produced it; and
``artifacts`` carries up to :data:`MAX_ARTIFACTS_PER_TURN` opaque descriptor
dicts (produced elsewhere — this store round-trips them without inspecting
their shape) so a follow-up can refer to a plot/file the agent already made.
When the bridge knew who asked, the turn also carries ``asked_by`` =
``{id, name}``. A turn with no ``asked_by`` means the asker is unknown: that
covers turns written before the field existed and channels that parse no
sender.

:meth:`HistoryStore.recent` can replay a long answer shortened: its opening and
a note naming the run that answered it, so the agent can read the whole answer
back when a follow-up needs it. The shortened copy is what is sent; the stored
turn is never changed.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from typing import Any, TypeGuard

from .store import JsonFileStore

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
# The char budget measures the thing we care about directly. The turn cap earns
# its place only by stopping a conversation of thousands of tiny turns from
# growing the persisted store without limit — if it is ever the binding
# constraint, that is a bug in these numbers, not the design working.
#
# Sized against a measured deployment rather than intuition: a real 30-turn DM
# serializes to ~19k chars (~5k tokens), the largest single conversation to
# ~24k chars, and an ENTIRE multi-conversation store to ~39k tokens — less than
# a bare agent-harness system prompt, and a few percent of a >=200k-token
# context. The budget below is therefore set so ordinary working conversations
# are never trimmed by size at all, while one pathological thread still cannot
# dominate the worker's prompt.
MAX_TURNS = 100
MAX_CHARS = 100000

# Hard age ceiling: a turn older than this is dropped on the next append.
#
# This is a RELEVANCE bound, not a cost one — the char budget already bounds
# cost. Its load-bearing job is that an abandoned conversation eventually
# shrinks: nothing else ever removes turns from a thread that stopped being
# used, and the store rewrites its whole file on every append, so unbounded
# retention would tax every future message in every OTHER conversation too.
#
# Six months is deliberately generous. Picking a months-old thread back up is a
# legitimate thing to do — an operator returning to a prior investigation — and
# each replayed turn carries its ``ts``, so the agent can weigh an old exchange
# for itself rather than having it silently deleted underneath it. Prefer that
# over a short ceiling: destroying context is not reversible, discounting it is.
#
# Pre-feature turns (no ``ts``) are grandfathered to the load time (see
# :meth:`HistoryStore._coerce`), so this prune can never mistake them for
# ancient and wipe a conversation's whole history on first append.
MAX_AGE_SECONDS = 180 * 24 * 60 * 60

# Per-turn cap on carried artifact descriptors. One run can emit many plots; we
# keep the most recent handful so the payload (and the char budget) stays
# bounded. We retain the TAIL — descriptors arrive oldest-first, so the last N
# are the run's most recent outputs, the ones a "show me that again" follow-up
# most likely means.
MAX_ARTIFACTS_PER_TURN = 10

# Placeholder recorded in the "answer" slot when a run did not produce one (it
# timed out, errored, or returned no text). We keep the QUESTION in history so a
# follow-up ("do that but take 30 min", "try again") can resolve its referent —
# without a marker turn the failed request leaves no trace at all and the agent
# is honestly told the thread is empty. The marker is a NON-answer: it must never
# read as a fabricated result the agent could cite.
FAILED_TURN_ANSWER = "[This request did not produce an answer — it timed out or errored.]"

# The note that ends a long answer replayed shortened. Prose for a reader, not a
# token: nothing matches on the wording. The payload it rides in is documented
# at docs/source/reference/contracts/bridge-dispatch.rst.
SHORTENED_ANSWER_NOTE = (
    "[Answer shortened; it is {chars:,} characters in full. "
    'prior_answer_read(run_id="{run_id}") returns the whole answer.]'
)


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


class HistoryStore(JsonFileStore):
    def __init__(
        self,
        path: str,
        *,
        max_turns: int = MAX_TURNS,
        max_chars: int = MAX_CHARS,
        max_age_seconds: float = MAX_AGE_SECONDS,
        now: Callable[[], float] = time.time,
    ):
        self.max_turns = max_turns
        self.max_chars = max_chars
        self.max_age_seconds = max_age_seconds
        # Clock seam: tests inject a fake ``now`` to exercise grandfathering and
        # the age-prune without real sleeps. Read once at load (grandfathering)
        # and once per append (timestamp + prune cutoff).
        self._now = now
        # Set by _coerce (which runs inside super().__init__ -> _load) when it
        # had to fill in any missing field; drives the persist-on-load below so
        # a grandfathered ``ts`` survives a restart.
        self._coerce_dirty = False
        super().__init__(path)
        # Persist the grandfathered turns exactly once, so the assigned ts is
        # durable and the age-prune reads a stable load-time anchor rather than
        # re-grandfathering (to an ever-later "now") on every restart.
        if self._coerce_dirty:
            with self._lock:
                self._flush_locked()

    def _coerce(self, loaded: dict) -> dict:
        """Normalize the loaded mapping into the current turn shape.

        Every conversation must be a list of turns; each turn is upgraded to
        ``{question, answer, ts, run_id, artifacts}``. Pre-feature turns lack
        ``ts``/``run_id``/``artifacts`` — a missing ``ts`` is stamped with the
        load time (so the age-prune can't treat it as ancient), a missing
        ``run_id`` becomes ``None``, and missing/oversized ``artifacts`` become
        a capped list. Any fill-in marks the store dirty so __init__ persists it.
        """
        now = self._now()
        result: dict[str, list[dict[str, Any]]] = {}
        for key, turns in loaded.items():
            if not isinstance(turns, list):
                continue  # drop malformed conversations
            coerced: list[dict[str, Any]] = []
            for turn in turns:
                if not isinstance(turn, dict):
                    self._coerce_dirty = True
                    continue  # drop malformed turns
                normalized = self._coerce_turn(turn, now)
                if normalized != turn:
                    self._coerce_dirty = True
                coerced.append(normalized)
            result[key] = coerced
        return result

    def _coerce_turn(self, turn: dict, now: float) -> dict[str, Any]:
        """Upgrade one loaded turn to the canonical shape (see :meth:`_coerce`)."""
        question = turn.get("question")
        answer = turn.get("answer")
        ts = turn.get("ts")
        # bool is an int subclass — reject it so a stray True/False ts is
        # re-stamped rather than read as 1.0/0.0 (epoch 1970 → instantly pruned).
        if not isinstance(ts, (int, float)) or isinstance(ts, bool):
            ts = now
        run_id = turn.get("run_id")
        if not isinstance(run_id, str):
            run_id = None
        artifacts = turn.get("artifacts")
        if not isinstance(artifacts, list):
            artifacts = []
        coerced: dict[str, Any] = {
            "question": question if isinstance(question, str) else "",
            "answer": answer if isinstance(answer, str) else "",
            "ts": ts,
            "run_id": run_id,
            "artifacts": _cap_artifacts(artifacts),
        }
        # A turn with no asker stays in the five-field shape, so a current file is
        # not rewritten on load; a malformed asker is dropped (and marks the store
        # dirty through the caller's ``normalized != turn`` rule).
        asked_by = turn.get("asked_by")
        if _valid_asker(asked_by):
            coerced["asked_by"] = _asker_record(asked_by)
        return coerced

    # --- API ---------------------------------------------------------------
    def recent(self, key: str, *, shorten_over: int = 0) -> list[dict[str, Any]]:
        """Return the replayable turns for a conversation, oldest first.

        Applies the turn cap and then the char budget (dropping oldest first),
        so the newest exchanges always survive. The char budget counts the FULL
        serialized turn — descriptors included — because the whole turn is what
        rides the payload, so an answer with many/large artifact descriptors
        must be charged for them, not just its question+answer text.

        ``shorten_over`` > 0 replays each answer longer than that many
        characters as :func:`shorten_answer` builds it, with ``answer_chars``
        set to the full length. Only a turn with a ``run_id`` is shortened,
        since nothing else could be read back. Shortening comes before the char
        budget, so one long answer costs fewer whole turns. It works on per-call
        copies: ``recent(key)`` is always the stored history unchanged.
        """
        with self._lock:
            turns = [dict(t) for t in self._data.get(key, [])]
        turns = turns[-self.max_turns :]
        if shorten_over > 0:
            for turn in turns:
                run_id = turn.get("run_id")
                answer = turn.get("answer")
                if not (isinstance(run_id, str) and run_id and isinstance(answer, str)):
                    continue
                shortened = shorten_answer(answer, run_id, shorten_over)
                if shortened is not None:
                    turn["answer"] = shortened
                    turn["answer_chars"] = len(answer)
        # Enforce the char budget from the tail (newest) backwards.
        kept: list[dict[str, Any]] = []
        used = 0
        for turn in reversed(turns):
            size = len(json.dumps(turn, ensure_ascii=False, default=str))
            if kept and used + size > self.max_chars:
                break
            kept.append(turn)
            used += size
        kept.reverse()
        return kept

    def append(
        self,
        key: str,
        question: str,
        answer: str,
        run_id: str | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        asked_by: Mapping[str, Any] | None = None,
    ) -> None:
        """Record one completed exchange for a conversation.

        ``run_id``, ``artifacts`` and ``asked_by`` are optional so pre-existing
        3-arg callers keep working unchanged. ``asked_by`` (``{id, name}``) is
        recorded only when it names someone; otherwise the turn has no
        ``asked_by`` key at all. ``artifacts`` is capped to
        :data:`MAX_ARTIFACTS_PER_TURN` (tail kept). Appending also age-prunes the
        conversation, dropping turns whose ``ts`` is older than
        ``max_age_seconds`` relative to now.
        """
        if not key:
            return
        now = self._now()
        turn: dict[str, Any] = {
            "question": question,
            "answer": answer,
            "ts": now,
            "run_id": run_id,
            "artifacts": _cap_artifacts(list(artifacts) if artifacts else []),
        }
        if _valid_asker(asked_by):
            turn["asked_by"] = _asker_record(asked_by)
        cutoff = now - self.max_age_seconds
        with self._lock:
            turns = self._data.setdefault(key, [])
            turns.append(turn)
            # Age-prune: drop turns older than the ceiling. A turn missing ts is
            # treated as "now" (kept) — coercion always stamps one, so this only
            # guards a turn appended in the current process.
            turns[:] = [t for t in turns if t.get("ts", now) >= cutoff]
            # Persist a bounded window; recent() re-applies the cap on read.
            if len(turns) > self.max_turns:
                del turns[: len(turns) - self.max_turns]
            self._flush_locked()

    def append_failed(
        self, key: str, question: str, asked_by: Mapping[str, Any] | None = None
    ) -> None:
        """Record a failed turn: the question with the :data:`FAILED_TURN_ANSWER`
        marker, so the next question in this conversation can still resolve its
        referent. An empty question carries no referent, so it is not recorded.
        ``asked_by`` is recorded as :meth:`append` records it."""
        if not question:
            return
        self.append(key, question, FAILED_TURN_ANSWER, asked_by=asked_by)


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
