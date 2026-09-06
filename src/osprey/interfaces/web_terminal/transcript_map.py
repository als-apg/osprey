"""Which transcript each session key currently points at.

A session key is the one identity a browser tab keeps for its whole life: the
same string names the PTY pool entry, the chat pool entry and the audit
session. The *transcript* is a different thing — the Claude
Code session id whose ``.jsonl`` holds the conversation — and the two are equal
only until the conversation is replaced. A ``/clear`` in the terminal starts a
fresh transcript under a new id while the key stays exactly where it was, so
after one ``/clear`` the key and the transcript have diverged permanently.

This module holds that divergence, and nothing else:

* :func:`get` answers the transcript a key currently points at, **defaulting to
  the key itself**. A session that has never cleared has no entry here and
  needs none, which is why the default is not ``None``: a caller resuming a
  conversation always has an id to resume, and the absent-entry case is the
  ordinary one rather than an error to branch on.
* :func:`set` records a move. It is a no-op when the mapping already says what
  it is being told — a hook that reports the same transcript on every turn must
  not fsync a file each time — and an id equal to the key *removes* the entry
  rather than storing an identity mapping, so the file grows by one line per
  conversation that actually moved and by nothing per session.

**Where the file lives.** Beside the control-context record, under the same one
path rule (:func:`osprey_connectors.posture_store.state_dir`): a deployment that
resolves one of these two files resolves both, and an operator looking for
either finds them in one directory. A root that does not resolve is a map with
no location — every :func:`get` still answers, the process keeps its in-memory
mapping, and only durability is lost. That is the right degrade: a transcript
id that survives in memory but not a restart costs the operator the *contents*
of a cleared-and-continued conversation after a container recreation, whereas
raising here would take down a view flip over a file nobody can repair from the
browser.

**Durability is best-effort by design.** Unlike the control-context record,
whose write is a commit point (a narrowing that is not on disk is a lie the
badge tells),
a lost mapping degrades to resuming the key itself — a conversation that reads
as fresh, not a session that writes when it promised not to. So the write is
attempted, its failure is logged, and memory keeps the answer.
"""

from __future__ import annotations

import logging
from pathlib import Path

from osprey.interfaces.web_terminal._json_store import read_json_object, write_json_atomic
from osprey_connectors import posture_store

logger = logging.getLogger(__name__)

__all__ = ["STORE_FILENAME", "get", "load", "set", "store_path"]

#: The document's name, beside ``control_context.json`` in the same directory.
STORE_FILENAME = "session-transcripts.json"


def store_path() -> Path | None:
    """Where the map lives, or ``None`` when it has no location.

    Delegates to :func:`osprey_connectors.posture_store.state_dir` — the same
    resolution the posture store uses (env ``OSPREY_AGENT_DATA_ROOT``, else the
    config derivation), so the two files are co-sited by construction rather
    than by two rules that agree today. That resolution can raise as well as
    answer ``None``; both mean the same thing here, a map that is memory-only.
    """
    try:
        directory = posture_store.state_dir()
    except Exception:  # noqa: BLE001 — an unresolvable root is "no file", not a crash
        logger.warning("The transcript map's location does not resolve", exc_info=True)
        return None
    return None if directory is None else directory / STORE_FILENAME


def _read(path: Path) -> dict[str, str]:
    """The persisted mapping at *path*, tolerating every absence.

    :func:`~osprey.interfaces.web_terminal._json_store.read_json_object` already
    answers ``None`` for a file that is missing, unreadable, not JSON, or not an
    object. What is added here is the value filter: only string-to-non-empty-
    string pairs survive, because an entry of any other shape cannot name a
    transcript to resume and honouring it would send ``--resume`` an argument
    Claude would reject. A hand-edited or half-migrated file therefore loses the
    entries nobody could have used and keeps the rest.
    """
    document = read_json_object(path) or {}
    return {
        key: value
        for key, value in document.items()
        if isinstance(key, str) and isinstance(value, str) and value.strip()
    }


def load(app) -> dict[str, str]:
    """Read the map into ``app.state.transcript_map`` and return it.

    Called once from the lifespan so a restarted container knows where its keys
    point before it serves anything, and again — lazily, through
    :func:`_mapping` — only when an earlier load found no location.

    A load taken while the map has **no location** is kept provisional and
    retried on the next access, and entries recorded during that outage win the
    merge. Caching a location-less load would let one transient config failure
    outlive itself: every later read would answer the key itself and silently
    resume the wrong transcript. The dict is mutated in place because callers
    hold it.
    """
    path = store_path()
    loaded = _read(path) if path is not None else {}

    mapping: dict[str, str] | None = getattr(app.state, "transcript_map", None)
    if mapping is None:
        mapping = loaded
    else:
        merged = {**loaded, **mapping}
        mapping.clear()
        mapping.update(merged)
    app.state.transcript_map = mapping
    app.state.transcript_map_provisional = path is None
    return mapping


def _mapping(app) -> dict[str, str]:
    """``app.state.transcript_map``, loading it on first use or after an outage."""
    mapping: dict[str, str] | None = getattr(app.state, "transcript_map", None)
    if mapping is not None and not getattr(app.state, "transcript_map_provisional", False):
        return mapping
    return load(app)


def get(app, key: str) -> str:
    """The transcript *key* currently points at — *key* itself when it has moved nowhere."""
    return _mapping(app).get(key) or key


def set(app, key: str, transcript_id: str) -> None:
    """Point *key* at *transcript_id*, persisting the change.

    A blank id is ignored: there is no such transcript, and storing it would
    replace a usable answer with one that cannot be resumed. An id equal to
    *key* clears the entry, which is the same statement said in the form the
    file is meant to hold — :func:`get` already answers the key by default.

    Writes only when the mapping actually changes, then only best-effort: a
    failed write is logged and the in-memory answer stands for the rest of the
    process's life.
    """
    transcript_id = (transcript_id or "").strip()
    if not transcript_id:
        return

    mapping = _mapping(app)
    if transcript_id == key:
        if mapping.pop(key, None) is None:
            return
    else:
        if mapping.get(key) == transcript_id:
            return
        mapping[key] = transcript_id

    path = store_path()
    if path is None:
        app.state.transcript_map_provisional = True
        logger.warning(
            "This deployment's agent-data root does not resolve, so the transcript for "
            "session %s is held in memory only and will not survive a restart.",
            key,
        )
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, dict(mapping))
    except OSError:
        logger.warning("Could not persist the transcript map to %s", path, exc_info=True)
