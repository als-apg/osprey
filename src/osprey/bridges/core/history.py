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
seconds, matching the retry queue's ``queued_at`` clock) anchors the 90-day
age-prune; ``run_id`` ties the turn back to the run that produced it; and
``artifacts`` carries up to :data:`MAX_ARTIFACTS_PER_TURN` opaque descriptor
dicts (produced elsewhere — this store round-trips them without inspecting
their shape) so a follow-up can refer to a plot/file the agent already made.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from .store import JsonFileStore

# Bounds on what we replay into the payload. Turn count is the primary knob;
# the char budget guards against a few huge answers bloating every follow-up
# (the worker's prompt is finite and channel answers can be long).
#
# Why bound this at all — the window is replayed as fresh input tokens on EVERY
# follow-up (each dispatch run is stateless, so there is no cross-run prompt
# cache to amortize it), and a DM's history key is its space, which never
# rotates: a DM is one unbounded, ever-growing conversation. Without a cap an
# old DM would eventually replay months of stale, unrelated turns into every
# new question. The caps below are sized generously (~10k tokens of history)
# so normal working sessions carry full context, while still fencing the
# eternal-DM case.
MAX_TURNS = 30
MAX_CHARS = 40000

# Hard age ceiling: a turn older than this is dropped on the next append, so a
# long-lived DM never replays months-stale exchanges even if it stays under the
# turn/char caps. Pre-feature turns (no ``ts``) are grandfathered to the load
# time (see :meth:`HistoryStore._coerce`), so this prune can never mistake them
# for ancient and wipe a conversation's whole history on first append.
MAX_AGE_SECONDS = 90 * 24 * 60 * 60

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
        return {
            "question": question if isinstance(question, str) else "",
            "answer": answer if isinstance(answer, str) else "",
            "ts": ts,
            "run_id": run_id,
            "artifacts": _cap_artifacts(artifacts),
        }

    # --- API ---------------------------------------------------------------
    def recent(self, key: str) -> list[dict[str, Any]]:
        """Return the replayable turns for a conversation, oldest first.

        Applies the turn cap and then the char budget (dropping oldest first),
        so the newest exchanges always survive. The char budget counts the FULL
        serialized turn — descriptors included — because the whole turn is what
        rides the payload, so an answer with many/large artifact descriptors
        must be charged for them, not just its question+answer text.
        """
        with self._lock:
            turns = [dict(t) for t in self._data.get(key, [])]
        turns = turns[-self.max_turns :]
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
    ) -> None:
        """Record one completed exchange for a conversation.

        ``run_id`` and ``artifacts`` are optional so pre-existing 3-arg callers
        keep working unchanged. ``artifacts`` is capped to
        :data:`MAX_ARTIFACTS_PER_TURN` (tail kept). Appending also age-prunes the
        conversation, dropping turns whose ``ts`` is older than
        ``max_age_seconds`` relative to now.
        """
        if not key:
            return
        now = self._now()
        turn = {
            "question": question,
            "answer": answer,
            "ts": now,
            "run_id": run_id,
            "artifacts": _cap_artifacts(list(artifacts) if artifacts else []),
        }
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

    def append_failed(self, key: str, question: str) -> None:
        """Record a failed turn: the question with the :data:`FAILED_TURN_ANSWER`
        marker, so the next question in this conversation can still resolve its
        referent. An empty question carries no referent, so it is not recorded."""
        if not question:
            return
        self.append(key, question, FAILED_TURN_ANSWER)


def _cap_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep at most :data:`MAX_ARTIFACTS_PER_TURN` descriptors, retaining the
    tail (most recent). Descriptors are round-tripped opaquely."""
    if len(artifacts) > MAX_ARTIFACTS_PER_TURN:
        return list(artifacts[-MAX_ARTIFACTS_PER_TURN:])
    return list(artifacts)
