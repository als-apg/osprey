"""Whether the expert view's Claude Code process is mid-turn, per session key.

The store is one mapping on ``app.state.turn_state``, keyed by session key::

    app.state.turn_state[key] = {"state": ..., "ts": ..., "transcript_id": ...}

Its writers are the ``POST /api/agent-turn`` route in
:mod:`osprey.interfaces.web_terminal.routes.agent_turn`, which records what the
process's own hooks report, and :func:`reset_turn_state`, which a PTY spawn or
teardown calls: a freshly spawned process has no turn in flight, and a
torn-down one has none either. Its reader is the hand-off door in
:mod:`osprey.interfaces.web_terminal.session_handoff`, which weighs the entry
against the transcript's tail before deciding a terminal is idle.

The store lives here rather than in the route module so that the door — a
core module every route depends on — never imports the ``routes`` package.
"""

from __future__ import annotations

import time
from typing import Any

from osprey.interfaces.web_terminal import transcript_map

#: The two states a key can be in. ``idle`` is also what a spawn or teardown
#: resets to, so it is the value :func:`reset_turn_state` writes.
BUSY = "busy"
IDLE = "idle"


def _store(app) -> dict[str, dict[str, Any]]:
    """Return ``app.state.turn_state``, creating it when absent.

    Apps that mount the agent-turn router standalone (tests, embedders) never
    ran the lifespan that installs the store, and a report they receive is
    still worth keeping — the reader helpers below find it either way.
    """
    store = getattr(app.state, "turn_state", None)
    if store is None:
        store = {}
        app.state.turn_state = store
    return store


def get_turn_state(app, key: str) -> dict[str, Any] | None:
    """Return the last turn reported for a session key, or ``None``.

    ``None`` means nothing has been reported for the key — a process running
    without the hook installed, or one that has not reached its first turn
    edge yet. It is not a claim that the key is idle, so a caller waiting for
    idleness must treat it as "unknown" and fall back to its own evidence.

    Args:
        app: The FastAPI application carrying ``state.turn_state``.
        key: The session key (the PTY pool key), not the transcript id.

    Returns:
        The stored ``{"state", "ts", "transcript_id"}`` mapping, or ``None``.
    """
    if not key:
        return None
    return _store(app).get(key)


def record_turn_state(app, key: str, *, state: str, ts: float, transcript_id: str) -> None:
    """Store a reported turn edge for a session key.

    Args:
        app: The FastAPI application carrying ``state.turn_state``.
        key: The session key the report is about.
        state: :data:`BUSY` or :data:`IDLE`.
        ts: When the edge happened, as POSIX seconds on the server's clock.
        transcript_id: The transcript the key's conversation lives in.
    """
    _store(app)[key] = {"state": state, "ts": ts, "transcript_id": transcript_id}


def reset_turn_state(app, key: str) -> None:
    """Record a session key as idle as of now.

    Called on both edges of a PTY's life. A process that has just spawned has
    no turn in flight, and a process that has just been torn down has none
    either, so both leave the key idle — and both do so *explicitly*, because
    the alternative (leaving the previous process's last report in place)
    would let a key that died mid-turn read as busy forever.

    The transcript id is carried over from the map rather than dropped, so the
    entry keeps the shape every reader expects. The map defaults an unmapped
    key to itself, which is the right answer for a session that has never
    cleared.

    Args:
        app: The FastAPI application carrying ``state.turn_state``.
        key: The session key to reset. An empty key is a no-op.
    """
    if not key:
        return
    record_turn_state(
        app, key, state=IDLE, ts=time.time(), transcript_id=transcript_map.get(app, key)
    )
