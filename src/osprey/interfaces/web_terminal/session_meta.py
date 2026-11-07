"""The hub's per-operator session stores: session meta, and the hub side of the ledger.

**Session meta** is one JSON document, ``session-meta.json``, beside the transcript
map and the ledger (:func:`osprey_connectors.posture_store.state_dir`, the rule
:mod:`~osprey.interfaces.web_terminal.transcript_map` uses). It holds what the
operator and the naming service say about each *transcript* — a session in the
sidebar is a transcript, not a key::

    {
      "sessions": {transcript_id: {"title", "title_source": "user"|"auto",
                                   "auto_title", "naming": "done"|"failed",
                                   "last_artifact_id", "layout", "pinned",
                                   "deleted"}},
      "last_active": {"session_key", "at"},
      "naming_since": ISO time of the document's first creation
    }

``title`` is what the sidebar shows: the operator's own title when they set one,
else the auto title. Clearing a user title (``set_title(tid, "")``) reverts to the
auto title rather than to nothing. ``layout`` is the web layout the session was
last left in (see :func:`clean_layout`). ``naming_since`` bounds auto-naming to sessions
that began after the feature existed, so turning it on does not spend a model call
per historical conversation. ``pinned`` (an ISO time) is when the operator pinned
the session in the sidebar; the pinned sessions are listed in that order (the
agent is not told of pins). ``deleted`` (an ISO time) is a deleted session's
tombstone, which drops its pin too: the agent record keeps its rows, so the listing, the namer, ``enter``,
the ledger's write point and the backfill all read :func:`deleted_ids` to keep it
from coming back. A naming that is in flight lives in memory only
(``app.state.session_naming_pending``): a restart must retry it, not believe it.

Like the transcript map the document is best-effort durable: no location keeps it
in memory, a failed write is logged, and a damaged file loads as defaults.

**The hub side of the ledger.** :mod:`osprey.stores.session_ledger` is a pure
reader/appender the artifacts app shares. The hub adds two things here, because
both need hub state: :func:`record_transcript_move` (append, then send the
``session_transcript`` frame) for every write point, and :func:`backfill_ledger`,
the one-time seeding from pre-feature state when no ledger file exists yet.

**One identity per hub.** Every store here sits under the acting identity's state
directory. A hub whose ``terminal_user`` differs from that identity would file one
operator's sessions under another's name, so :func:`check_identity` refuses the
start instead.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osprey.interfaces.web_terminal._json_store import read_json_object, write_json_atomic
from osprey.interfaces.web_terminal.session_key import is_posture_key
from osprey.stores import session_ledger
from osprey_connectors import posture_store
from osprey_connectors.identity import acting_identity

logger = logging.getLogger(__name__)

__all__ = [
    "LAYOUT_MAX_BYTES",
    "LayoutTooLarge",
    "SESSION_DELETED_FRAME",
    "SESSION_PINNED_FRAME",
    "SESSION_TRANSCRIPT_FRAME",
    "STORE_FILENAME",
    "backfill_ledger",
    "backfill_rows",
    "check_identity",
    "clean_layout",
    "deleted_ids",
    "get",
    "get_last_active",
    "ledger_state_dir",
    "load",
    "mark_deleted",
    "naming_since",
    "pinned_ids",
    "record_transcript_move",
    "set_auto_title",
    "set_last_active",
    "set_last_artifact",
    "set_layout",
    "set_pinned",
    "set_title",
    "store_path",
]

#: The document's name, beside ``session-transcripts.json``.
STORE_FILENAME = "session-meta.json"

#: The hub page frame sent on every ledger append.
SESSION_TRANSCRIPT_FRAME = "session_transcript"

#: The hub page frame sent when the operator deletes a session.
SESSION_DELETED_FRAME = "session_deleted"

#: The hub frame type that tells every tab a session was pinned or unpinned.
SESSION_PINNED_FRAME = "session_pinned"

_TITLE_SOURCES = ("user", "auto")
_NAMING_STATES = ("done", "failed")

#: The largest session layout kept, in bytes of its compact JSON.
LAYOUT_MAX_BYTES = 64 * 1024

#: The stored-dock version a session layout carries (dock-workspace.js's STORE_VERSION).
_LAYOUT_VERSION = 2


class LayoutTooLarge(ValueError):
    """A session layout over :data:`LAYOUT_MAX_BYTES`."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---- location --------------------------------------------------------------


def ledger_state_dir() -> Path | None:
    """The state directory the ledger and the meta live in, or ``None``.

    :func:`osprey_connectors.posture_store.state_dir` can raise as well as answer
    ``None``; both mean "no location" here.
    """
    try:
        return posture_store.state_dir()
    except Exception:  # an unresolvable root is "no file", not a crash
        logger.warning("The session stores' location does not resolve", exc_info=True)
        return None


def store_path() -> Path | None:
    """Where the meta document lives, or ``None`` when it has no location."""
    directory = ledger_state_dir()
    return None if directory is None else directory / STORE_FILENAME


# ---- identity guard ---------------------------------------------------------


def check_identity(terminal_user: str) -> None:
    """Refuse a hub whose ``terminal_user`` is not the acting identity.

    Raises:
        RuntimeError: the lifespan's refusal, naming both identities.
    """
    terminal_user = (terminal_user or "").strip()
    if not terminal_user:
        return
    identity = acting_identity()
    if terminal_user != identity:
        raise RuntimeError(
            "Refusing to start the Web Terminal.\n"
            f"This hub serves the terminal user {terminal_user!r}, but its per-operator "
            f"stores resolve to the acting identity {identity!r}. A hub serves one "
            "identity: set OSPREY_TERMINAL_USER to a usable account name (no path "
            "separators) that matches the identity the stores are filed under."
        )


# ---- the meta document --------------------------------------------------------


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _root_relative(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("/")
        and not value.startswith("//")
        and "\\" not in value
    )


def clean_layout(raw: Any) -> dict[str, Any]:
    """A session layout as stored: the page's stored dock plus each panel's page.

    The shape is dock-workspace.js's stored dock, ``{version: 2, layout, panels}``,
    with ``pages`` beside it: panel id to the root-relative address the panel was
    on. The dock tree is the page's to repair, so only its container is checked
    here. Unknown top-level fields are dropped.

    Raises:
        LayoutTooLarge: the cleaned layout is over :data:`LAYOUT_MAX_BYTES`.
        ValueError: anything else that is not a session layout.
    """
    if not isinstance(raw, dict) or raw.get("version") != _LAYOUT_VERSION:
        raise ValueError(f"a session layout is an object with version {_LAYOUT_VERSION}")
    dock = raw.get("layout")
    if dock is not None and not isinstance(dock, dict):
        raise ValueError("layout must be an object or null")
    panels = raw.get("panels", {})
    if not isinstance(panels, dict) or not all(
        isinstance(k, str) and k and isinstance(v, dict) for k, v in panels.items()
    ):
        raise ValueError("panels must map panel ids to records")
    pages = raw.get("pages", {})
    if not isinstance(pages, dict) or not all(
        isinstance(k, str) and k and _root_relative(v) for k, v in pages.items()
    ):
        raise ValueError("pages must map panel ids to root-relative addresses")
    cleaned = {"version": _LAYOUT_VERSION, "layout": dock, "panels": panels, "pages": pages}
    if len(json.dumps(cleaned, separators=(",", ":"))) > LAYOUT_MAX_BYTES:
        raise LayoutTooLarge(f"a session layout is at most {LAYOUT_MAX_BYTES} bytes")
    return cleaned


def _layout_or_none(raw: Any) -> dict[str, Any] | None:
    try:
        return None if raw is None else clean_layout(raw)
    except ValueError:
        return None


def _clean_session(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    source = raw.get("title_source")
    naming = raw.get("naming")
    return {
        "title": _str_or_none(raw.get("title")),
        "title_source": source if source in _TITLE_SOURCES else None,
        "auto_title": _str_or_none(raw.get("auto_title")),
        "naming": naming if naming in _NAMING_STATES else None,
        "last_artifact_id": _str_or_none(raw.get("last_artifact_id")),
        "layout": _layout_or_none(raw.get("layout")),
        "pinned": _str_or_none(raw.get("pinned")),
        "deleted": _str_or_none(raw.get("deleted")),
    }


def _blank() -> dict[str, Any]:
    return {
        "title": None,
        "title_source": None,
        "auto_title": None,
        "naming": None,
        "last_artifact_id": None,
        "layout": None,
        "pinned": None,
        "deleted": None,
    }


def _clean(document: Mapping[str, Any] | None) -> dict[str, Any]:
    document = document or {}
    sessions_raw = document.get("sessions")
    sessions: dict[str, dict[str, Any]] = {}
    if isinstance(sessions_raw, dict):
        for tid, raw in sessions_raw.items():
            entry = _clean_session(raw)
            if isinstance(tid, str) and tid and entry is not None:
                sessions[tid] = entry
    last_raw = document.get("last_active")
    last_active = None
    if isinstance(last_raw, dict) and _str_or_none(last_raw.get("session_key")):
        last_active = {
            "session_key": last_raw["session_key"],
            "at": _str_or_none(last_raw.get("at")),
        }
    return {
        "sessions": sessions,
        "last_active": last_active,
        "naming_since": _str_or_none(document.get("naming_since")),
    }


def _persist(app, document: dict[str, Any]) -> None:
    path = store_path()
    if path is None:
        app.state.session_meta_provisional = True
        logger.warning("The session meta has no location; it is held in memory only.")
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, document)
    except OSError:
        logger.warning("Could not persist the session meta to %s", path, exc_info=True)


def load(app) -> dict[str, Any]:
    """Read the meta into ``app.state.session_meta`` and return it.

    A missing or damaged document loads as defaults. ``naming_since`` is stamped
    (and the document written) on the first load that finds none, which is the
    store's first creation.
    """
    path = store_path()
    raw = read_json_object(path) if path is not None else None
    if path is not None and raw is None and path.exists():
        logger.warning("The session meta at %s is damaged; loading defaults", path)
    document = _clean(raw)
    held = getattr(app.state, "session_meta", None)
    if held is not None and getattr(app.state, "session_meta_provisional", False):
        # A reload after a location-less load: what was recorded in memory
        # during the outage wins the merge, as the transcript map does.
        document["sessions"].update(held["sessions"])
        document["last_active"] = held["last_active"] or document["last_active"]
        document["naming_since"] = document["naming_since"] or held["naming_since"]
    app.state.session_meta = document
    app.state.session_meta_provisional = path is None
    if not hasattr(app.state, "session_naming_pending"):
        app.state.session_naming_pending = set()
    if document["naming_since"] is None:
        document["naming_since"] = _now()
        _persist(app, document)
    return document


def _document(app) -> dict[str, Any]:
    document: dict[str, Any] | None = getattr(app.state, "session_meta", None)
    if document is None or getattr(app.state, "session_meta_provisional", False):
        return load(app)
    return document


def _entry(document: dict[str, Any], tid: str) -> dict[str, Any]:
    sessions: dict[str, dict[str, Any]] = document["sessions"]
    return sessions.setdefault(tid, _blank())


def get(app, tid: str) -> dict[str, Any]:
    """A copy of *tid*'s record (every field ``None`` for an unknown transcript)."""
    entry = _document(app)["sessions"].get(tid)
    return _blank() if entry is None else dict(entry)


def deleted_ids(app) -> set[str]:
    """The transcripts the operator deleted (their tombstones)."""
    sessions = list(_document(app)["sessions"].items())
    return {tid for tid, entry in sessions if entry.get("deleted")}


def mark_deleted(app, tid: str) -> dict[str, Any]:
    """Replace *tid*'s record with a tombstone: no title, artifact, layout or pin, ``deleted`` now."""
    document = _document(app)
    document["sessions"][tid] = {**_blank(), "deleted": _now()}
    _persist(app, document)
    return dict(document["sessions"][tid])


def set_title(app, tid: str, title: str) -> dict[str, Any]:
    """Set the operator's title for *tid*; an empty title reverts to the auto title."""
    document = _document(app)
    entry = _entry(document, tid)
    title = (title or "").strip()
    if title:
        entry["title"], entry["title_source"] = title, "user"
    else:
        entry["title"] = entry["auto_title"]
        entry["title_source"] = "auto" if entry["auto_title"] else None
    _persist(app, document)
    return dict(entry)


def set_pinned(app, tid: str, pinned: bool) -> dict[str, Any]:
    """Pin *tid* (``pinned`` now, kept when already pinned) or unpin it.

    Raises:
        KeyError: *tid* is deleted; a tombstone takes no pin.
    """
    document = _document(app)
    if (document["sessions"].get(tid) or {}).get("deleted"):
        raise KeyError(tid)
    entry = _entry(document, tid)
    value = (entry["pinned"] or _now()) if pinned else None
    if entry["pinned"] != value:
        entry["pinned"] = value
        _persist(app, document)
    return dict(entry)


def pinned_ids(app) -> list[str]:
    """The pinned transcripts that are not deleted, first pinned first."""
    pins = [
        (entry["pinned"], tid)
        for tid, entry in list(_document(app)["sessions"].items())
        if entry.get("pinned") and not entry.get("deleted")
    ]
    return [tid for _, tid in sorted(pins)]


def set_auto_title(app, tid: str, title: str | None, state: str) -> dict[str, Any]:
    """Record a naming outcome for *tid*: its auto title and ``done`` / ``failed``.

    A user title is never overwritten; the auto title is kept beside it so that
    clearing the user title reverts to it.
    """
    if state not in _NAMING_STATES:
        raise ValueError(f"naming state must be one of {_NAMING_STATES}: {state!r}")
    document = _document(app)
    entry = _entry(document, tid)
    entry["auto_title"] = _str_or_none((title or "").strip())
    entry["naming"] = state
    if entry["title_source"] != "user":
        entry["title"] = entry["auto_title"]
        entry["title_source"] = "auto" if entry["auto_title"] else None
    _persist(app, document)
    pending: set[str] = getattr(app.state, "session_naming_pending", set())
    pending.discard(tid)
    return dict(entry)


def set_last_artifact(app, tid: str, aid: str) -> dict[str, Any]:
    """Record the artifact last shown for *tid*."""
    document = _document(app)
    entry = _entry(document, tid)
    if entry["last_artifact_id"] != aid:
        entry["last_artifact_id"] = _str_or_none(aid)
        _persist(app, document)
    return dict(entry)


def set_layout(app, tid: str, layout: Any) -> dict[str, Any]:
    """Record the web layout *tid* was last left in; ``None`` forgets it.

    Raises:
        ValueError: *layout* is not a session layout (see :func:`clean_layout`).
    """
    cleaned = None if layout is None else clean_layout(layout)
    document = _document(app)
    entry = _entry(document, tid)
    if entry["layout"] != cleaned:
        entry["layout"] = cleaned
        _persist(app, document)
    return dict(entry)


def get_last_active(app) -> dict[str, Any] | None:
    """``{session_key, at}`` of the key the operator was last on, or ``None``."""
    last = _document(app)["last_active"]
    return None if last is None else dict(last)


def set_last_active(app, key: str) -> dict[str, Any]:
    """Record *key* as the operator's last-active session key, at now."""
    document = _document(app)
    document["last_active"] = {"session_key": key, "at": _now()}
    _persist(app, document)
    return dict(document["last_active"])


def naming_since(app) -> str:
    """The ISO time of the store's first creation."""
    return str(_document(app)["naming_since"])


# ---- the hub side of the ledger ------------------------------------------------


def record_transcript_move(
    app, key: str, transcript_id: str, from_ts: str | datetime | None = None
) -> bool:
    """Append ``(key, transcript_id, from_ts)`` and, when a row was written, send the frame.

    The one routine every hub write point calls, so each sends exactly one
    ``session_transcript {session_key}`` per row actually appended and none for a
    repeat. A hub without a broadcaster (a bare test app) still appends. A
    deleted transcript is never written back.
    """
    if transcript_id in deleted_ids(app):
        return False
    if not session_ledger.append(ledger_state_dir(), key, transcript_id, from_ts):
        return False
    broadcaster = getattr(app.state, "broadcaster", None)
    if broadcaster is not None:
        try:
            broadcaster.broadcast({"type": SESSION_TRANSCRIPT_FRAME, "session_key": key})
        except Exception:  # a frame is a hint; the row is already durable
            logger.warning("Could not send the session_transcript frame", exc_info=True)
    return True


def _first_timestamp(path: Path) -> str | None:
    """The first aware ``timestamp`` in a transcript ``.jsonl``, else its mtime."""
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                stamp = record.get("timestamp") if isinstance(record, dict) else None
                if not isinstance(stamp, str):
                    continue
                try:
                    parsed = datetime.fromisoformat(stamp)
                except ValueError:
                    continue
                if parsed.tzinfo is not None:
                    return parsed.isoformat()
        return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
    except OSError:
        return None


def backfill_rows(
    transcript_map: Mapping[str, str],
    sessions_dir: Path | None,
    artifact_session_ids: Iterable[str],
) -> list[tuple[str, str, str]]:
    """The rows the one-time backfill appends, in order.

    For every map key K: ``(K, K, epoch)``, plus ``(K, map[K], ts)`` when
    ``map[K] != K`` (ts = first timestamp in ``map[K].jsonl``; that row is left out
    when the file is gone). Then ``(K, K, epoch)`` for each distinct artifact
    ``session_id`` whose ``K.jsonl`` is on disk. Only canonical UUID keys get
    ``(K, K)``: a non-UUID key's first transcript is never the key itself.
    """
    rows: list[tuple[str, str, str]] = []
    for key, target in transcript_map.items():
        if not is_posture_key(key):
            continue
        rows.append((key, key, session_ledger.EPOCH))
        if target and target != key and is_posture_key(target) and sessions_dir is not None:
            stamp = _first_timestamp(sessions_dir / f"{target}.jsonl")
            if stamp is not None:
                rows.append((key, target, stamp))
    seen = set(transcript_map)
    for key in artifact_session_ids:
        if key in seen or not is_posture_key(key):
            continue
        seen.add(key)
        if sessions_dir is not None and (sessions_dir / f"{key}.jsonl").is_file():
            rows.append((key, key, session_ledger.EPOCH))
    return rows


def backfill_ledger(app) -> int:
    """Seed the ledger from pre-feature state, once: only when no ledger file exists.

    Sources are this identity's files only: the transcript map document, the
    project's transcript directory, and the artifact store at the hub's shared
    data root. Writes the file even when there is nothing to seed, so the
    backfill never runs twice. Returns the number of rows appended. Never raises.
    """
    state_dir = ledger_state_dir()
    if state_dir is None or session_ledger.exists(state_dir):
        return 0
    try:
        from osprey.interfaces.web_terminal import transcript_map as tmap

        map_path = tmap.store_path()
        mapping: dict[str, str] = {}
        if map_path is not None:
            raw = read_json_object(map_path) or {}
            mapping = {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}

        sessions_dir: Path | None = None
        cwd = getattr(app.state, "project_cwd", None)
        if cwd:
            from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery

            sessions_dir = SessionDiscovery(cwd)._resolve_sessions_dir()

        artifact_ids: list[str] = []
        root = getattr(app.state, "panel_status_root", None)
        if root is not None:
            from osprey.stores.artifact_store import ArtifactStore

            store = ArtifactStore(workspace_root=Path(root), auto_launch=False)
            artifact_ids = list(
                dict.fromkeys(e.session_id for e in store.list_entries() if e.session_id)
            )
        deleted = deleted_ids(app)
        rows = [
            r for r in backfill_rows(mapping, sessions_dir, artifact_ids) if r[1] not in deleted
        ]
    except Exception:
        logger.warning("The session ledger backfill could not read its sources", exc_info=True)
        return 0
    added = session_ledger.append_many(state_dir, rows)
    if not added:
        session_ledger.write_empty(state_dir)
    logger.info("Backfilled the session ledger with %d row(s)", len(added))
    return len(added)
