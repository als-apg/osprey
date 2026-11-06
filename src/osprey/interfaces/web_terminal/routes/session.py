"""Session diagnostics routes — agent hierarchy, logs, summaries, and chat.

Every session fact these routes answer is read from the agent record with this
terminal's login, which sees only its own person's sessions, and with a drain
naming the session, so a step the agent just took is in the answer. Each
answer carries a ``record`` field — ``complete``, ``incomplete`` or
``unavailable`` (with ``record_reason``) — and never shows an unread session
as an empty complete one. Only the lookup of which transcript is current
stays local.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import re
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from osprey.build.claude_code_telemetry import (
    DISPATCH_SESSION_PREFIX,
    HOST_SESSION_PREFIX,
    WATCHDOG_SESSION_PREFIX,
)
from osprey.interfaces.web_terminal import session_meta, transcript_map
from osprey.interfaces.web_terminal.session_key import is_posture_key
from osprey.mcp_server import session as session_record
from osprey.stores import session_ledger

logger = logging.getLogger(__name__)

router = APIRouter()


def record_client():
    """The record client this process reads with, configured from its environment.

    Resolved per request and reached through the module, so a test replaces it
    with ``monkeypatch.setattr(session_routes, "record_client", ...)``.
    """
    return session_record.record_client()


def _identity(request: Request) -> str:
    """This terminal's identity: its configured user, else the acting identity."""
    from osprey_connectors.identity import acting_identity

    terminal_user = getattr(request.app.state, "terminal_user", "")
    if isinstance(terminal_user, str) and terminal_user.strip():
        return terminal_user.strip()
    return acting_identity()


def _unavailable(reason: str) -> dict[str, str]:
    return session_record.record_fields(session_record.RECORD_UNAVAILABLE, reason)


#: What a session with no title and no readable first prompt is called.
UNTITLED_SESSION = "Untitled session"
#: The longest title a first prompt is shortened to.
PROMPT_TITLE_CHARS = 60
#: ``terminal_session`` prefixes of sessions that are not the operator's own
#: conversations: a dispatch run, a host ``osprey chat``, a watchdog judge.
_NOT_OPERATOR_PREFIXES = (DISPATCH_SESSION_PREFIX, HOST_SESSION_PREFIX, WATCHDOG_SESSION_PREFIX)


def _encode_cursor(before: str) -> str:
    raw = json.dumps({"before": before}, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str) -> datetime:
    """The page boundary *cursor* names. Raises ValueError for one this route did not issue."""
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        before = datetime.fromisoformat(json.loads(raw)["before"])
    except (ValueError, TypeError, KeyError, binascii.Error) as exc:
        raise ValueError("cursor is not one this listing issued") from exc
    if before.tzinfo is None:
        raise ValueError("cursor is not one this listing issued")
    return before


def _shortened(prompt: str) -> str:
    """A first prompt as a title: one line, at most :data:`PROMPT_TITLE_CHARS`."""
    text = " ".join(prompt.split())
    if len(text) <= PROMPT_TITLE_CHARS:
        return text
    cut = text[: PROMPT_TITLE_CHARS - 1]
    space = cut.rfind(" ")
    if space > PROMPT_TITLE_CHARS // 2:
        cut = cut[:space]
    return cut.rstrip() + "\u2026"


def _title_of(meta: Mapping[str, Any], prompt: str) -> tuple[str, str | None]:
    """A session's title and its source: the meta's, else its first prompt shortened."""
    if meta["title"]:
        return meta["title"], meta["title_source"]
    if prompt.strip():
        return _shortened(prompt), "prompt"
    return UNTITLED_SESSION, None


def _operator_row(row: Mapping[str, Any]) -> bool:
    """Decision 6: whether a record row is one of the operator's own conversations.

    A dispatch run (any ``run_ids``), a host, dispatch or watchdog terminal
    session and a promptless row the read was not cut short of are not.
    """
    if row.get("run_ids"):
        return False
    terminal = row.get("terminal_session")
    if isinstance(terminal, str) and terminal.startswith(_NOT_OPERATOR_PREFIXES):
        return False
    prompt = row.get("prompt")
    return (isinstance(prompt, str) and bool(prompt.strip())) or bool(row.get("truncated"))


def _live_keys(app) -> set[str]:
    """The session keys whose child runs now: a live PTY or a chat child."""
    live: set[str] = set()
    registry = getattr(app.state, "pty_registry", None)
    if registry is not None:
        for key in registry.session_keys():
            pty = registry.get_session(key)
            if pty is not None and pty.is_alive:
                live.add(key)
    operators = getattr(app.state, "operator_registry", None)
    chats = getattr(operators, "chats", None) if operators is not None else None
    if chats is not None:
        live.update(chats.keys())
    return live


def _session_row(
    app,
    transcript_id: str,
    record_row: Mapping[str, Any] | None,
    ledger: session_ledger.Ledger,
    live: set[str],
) -> dict[str, Any]:
    """One ``GET /api/sessions`` row: the record's facts, the ledger's key, the meta's title."""
    record_row = record_row or {}
    key = ledger.key_of(transcript_id)
    current = key is not None and ledger.current(key) == transcript_id
    meta = session_meta.get(app, transcript_id)
    prompt = record_row.get("prompt")
    prompt = prompt if isinstance(prompt, str) and prompt.strip() else ""
    truncated = bool(record_row.get("truncated"))
    title, title_source = _title_of(meta, prompt)
    pending: set[str] = getattr(app.state, "session_naming_pending", set())
    naming = "pending" if transcript_id in pending else meta["naming"]
    message_count = record_row.get("message_count")
    return {
        "transcript_id": transcript_id,
        "session_key": key,
        "current_for_key": current,
        "title": title,
        "title_source": title_source,
        "naming": naming,
        "last_artifact_id": meta["last_artifact_id"],
        "state": "live" if current and key in live else "idle",
        "message_count": message_count if isinstance(message_count, int) else 0,
        "first_ts": record_row.get("first_ts"),
        "last_ts": record_row.get("last_ts"),
        "truncated": truncated,
        "prompt": prompt,
    }


def _ledger_only(
    ledger: session_ledger.Ledger,
    listed: set[str],
    newest: datetime | None,
    oldest: datetime | None,
) -> list[tuple[datetime, str]]:
    """The ledger's transcripts not in *listed*, placed inside ``[oldest, newest)``.

    A transcript's place is its first ledger row's ``from_ts``: when the hub
    first saw it, which for a transcript the record has not ingested is as close
    to its activity as anything known. ``None`` leaves that side open. These are
    candidates only: one the record knows is listed by the record, on its own
    page (:func:`_unknown_to_record`).
    """
    first: dict[str, datetime] = {}
    for row in ledger.rows:
        if row.transcript_id in listed or row.transcript_id in first:
            continue
        at = session_ledger.aware_time(row.from_ts)
        if at is not None:
            first[row.transcript_id] = at
    return [
        (at, tid)
        for tid, at in first.items()
        if (newest is None or at < newest) and (oldest is None or at >= oldest)
    ]


def _unknown_to_record(
    candidates: list[tuple[datetime, str]], client: Any
) -> tuple[list[tuple[datetime, str]] | None, str]:
    """*candidates* without the transcripts the record has any row of.

    ``None`` (and the reason) when the record could not say which it knows.
    """
    if not candidates:
        return candidates, ""
    known, answer = session_record.known_sessions([tid for _, tid in candidates], client=client)
    if known is None:
        why = (answer.reason or answer.unavailable or "") if answer is not None else ""
        return None, why or "the record could not say which ledger transcripts it holds"
    return [c for c in candidates if c[1] not in known], ""


def _session_listing(
    app,
    identity: str,
    client: Any,
    *,
    before: datetime | None,
    drain: bool,
) -> dict[str, Any]:
    """The body of ``GET /api/sessions``: one page, newest activity first.

    Record rows (Decision 6 drops applied before anything is named) and the
    ledger's transcripts the record has not ingested (it holds no row of
    them), merged by activity time. Each transcript is listed once across
    the pages.
    """
    rows, answer = session_record.own_sessions(identity, client=client, drain=drain, before=before)
    ledger = session_ledger.load(session_meta.ledger_state_dir())
    live = _live_keys(app)
    # The record keeps a deleted session's rows; its tombstone keeps it unlisted.
    deleted = session_meta.deleted_ids(app)
    more = False
    next_cursor = None
    record_rows: list[Mapping[str, Any]] = []
    extra: list[tuple[datetime, str]] = []
    if answer.available:
        record = (
            session_record.RECORD_COMPLETE if answer.complete else session_record.RECORD_INCOMPLETE
        )
        reason = answer.reason
        raw = answer.data if isinstance(answer.data, list) else []
        listed = {r["session_id"] for r in raw if isinstance(r, Mapping) and r.get("session_id")}
        more = bool(answer.more)
        oldest = None
        if raw and isinstance(raw[-1], Mapping):
            oldest = session_ledger.aware_time(raw[-1].get("last_ts"))
        if more and oldest is not None:
            next_cursor = _encode_cursor(oldest.isoformat())
        record_rows = [r for r in rows if _operator_row(r) and r["session_id"] not in deleted]
        candidates = _ledger_only(ledger, listed, before, oldest if more else None)
        unknown, why = _unknown_to_record(candidates, client)
        if unknown is None:
            # Listing them could repeat a transcript the record lists on
            # another page; the page says it is incomplete instead.
            record = session_record.RECORD_INCOMPLETE
            reason = reason or why
        else:
            extra = unknown
    else:
        # A refused read (an older service without ``before``) answered nothing
        # for this page; the record is not down, the page is incomplete.
        refused = bool(getattr(answer, "refused", False))
        record = session_record.RECORD_INCOMPLETE if refused else session_record.RECORD_UNAVAILABLE
        reason = answer.unavailable or ""
        # With no record rows and no next page, the first page lists every
        # ledger transcript; a later page lists none, as an earlier one did.
        if before is None:
            extra = _ledger_only(ledger, set(), None, None)

    placed: list[tuple[datetime, str, dict[str, Any]]] = []
    epoch = datetime.min.replace(tzinfo=UTC)
    for row in record_rows:
        at = session_ledger.aware_time(row.get("last_ts")) or epoch
        tid = row["session_id"]
        placed.append((at, tid, _session_row(app, tid, row, ledger, live)))
    for at, tid in extra:
        if tid in deleted:
            continue
        placed.append((at, tid, _session_row(app, tid, None, ledger, live)))
    placed.sort(key=lambda p: (p[0], p[1]), reverse=True)
    return {
        "sessions": [row for _, _, row in placed],
        "more": more,
        "next_cursor": next_cursor,
        **session_record.record_fields(record, reason),
    }


@router.get("/api/sessions")
async def list_sessions(request: Request, cursor: str | None = None, drain: str | None = None):
    """This operator's sessions (one page, newest activity first).

    Only ``drain=1`` waits for ingest to pass what is pending; ``cursor`` is the
    ``next_cursor`` of the previous page.
    """
    before = None
    if cursor:
        try:
            before = _decode_cursor(cursor)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
    identity = _identity(request)
    client = record_client()
    body = await asyncio.to_thread(
        _session_listing,
        request.app,
        identity,
        client,
        before=before,
        drain=drain == "1",
    )
    _schedule_naming(request.app, body["sessions"])
    return body


def _schedule_naming(app, rows: list[dict[str, Any]]) -> None:
    """Hand the listed rows to the hub's namer; the scheduled ones read ``pending``.

    Scheduling only queues: the listing never waits on the model.
    """
    namer = getattr(app.state, "session_namer", None)
    if namer is None:
        return
    try:
        scheduled = set(namer.schedule(rows))
    except Exception:  # naming is a nicety; the listing stands without it
        logger.warning("Could not schedule session naming", exc_info=True)
        return
    for row in rows:
        if row["transcript_id"] in scheduled:
            row["naming"] = "pending"


# ---- per-session meta ------------------------------------------------------------


class _TitleBody(BaseModel):
    title: str = ""


class _LastArtifactBody(BaseModel):
    artifact_id: str = ""


class _LastActiveBody(BaseModel):
    session_key: str


def _uuid_id(value: str, name: str) -> str:
    """*value* when it is a canonical UUID; a 422 naming *name* otherwise."""
    if not is_posture_key(value):
        raise HTTPException(status_code=422, detail=f"{name} must be a session UUID")
    return value


@router.get("/api/sessions/last-active")
async def get_last_active(request: Request):
    """The key the operator was last on: ``{session_key, at}``, or nulls."""
    last = await asyncio.to_thread(session_meta.get_last_active, request.app)
    return last or {"session_key": None, "at": None}


@router.put("/api/sessions/last-active")
async def put_last_active(request: Request, body: _LastActiveBody):
    key = _uuid_id(body.session_key, "session_key")
    return await asyncio.to_thread(session_meta.set_last_active, request.app, key)


@router.put("/api/sessions/{transcript_id}/title")
async def put_title(request: Request, transcript_id: str, body: _TitleBody):
    """Set the operator's title; an empty one reverts to the auto title."""
    tid = _uuid_id(transcript_id, "transcript_id")
    entry = await asyncio.to_thread(session_meta.set_title, request.app, tid, body.title)
    return {"transcript_id": tid, **entry}


@router.put("/api/sessions/{transcript_id}/last-artifact")
async def put_last_artifact(request: Request, transcript_id: str, body: _LastArtifactBody):
    tid = _uuid_id(transcript_id, "transcript_id")
    entry = await asyncio.to_thread(
        session_meta.set_last_artifact, request.app, tid, body.artifact_id
    )
    return {"transcript_id": tid, **entry}


def _is_mine(app, tid: str) -> bool:
    """Whether *tid* is one of this operator's sessions and not deleted."""
    if tid in session_meta.deleted_ids(app):
        return False
    ledger = session_ledger.load(session_meta.ledger_state_dir())
    return _owned(app, tid, ledger, _sessions_dir(app))


@router.get("/api/sessions/{transcript_id}/layout")
async def get_layout(request: Request, transcript_id: str):
    """The web layout the session was last left in, or ``null``; 404 for a session not mine."""
    tid = _uuid_id(transcript_id, "transcript_id")
    if not await asyncio.to_thread(_is_mine, request.app, tid):
        raise HTTPException(status_code=404, detail="No such session.")
    meta = await asyncio.to_thread(session_meta.get, request.app, tid)
    return {"transcript_id": tid, "layout": meta["layout"]}


class _LayoutBody(BaseModel):
    layout: dict[str, Any] | None = None


@router.put("/api/sessions/{transcript_id}/layout")
async def put_layout(request: Request, transcript_id: str, body: _LayoutBody):
    """Keep the web layout the session is in; ``null`` forgets it.

    404 for a session not mine (or deleted), 413 over the size cap, 422 for
    anything that is not a session layout.
    """
    tid = _uuid_id(transcript_id, "transcript_id")
    app = request.app
    if not await asyncio.to_thread(_is_mine, app, tid):
        raise HTTPException(status_code=404, detail="No such session.")
    try:
        await asyncio.to_thread(session_meta.set_layout, app, tid, body.layout)
    except session_meta.LayoutTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"transcript_id": tid, "saved": body.layout is not None}


def _enter(app, transcript_id: str) -> str:
    key = str(uuid.uuid4())
    transcript_map.set(app, key, transcript_id)
    session_meta.record_transcript_move(app, key, transcript_id, datetime.now(UTC))
    return key


@router.post("/api/sessions/{transcript_id}/enter")
async def enter_session(request: Request, transcript_id: str):
    """Enter a transcript that is no key's current one: a new key K\u2032 pointing at it.

    The spawn of K\u2032 then resumes the transcript, because it resolves
    ``transcript_map.get(K\u2032)``.
    """
    tid = _uuid_id(transcript_id, "transcript_id")
    if tid in await asyncio.to_thread(session_meta.deleted_ids, request.app):
        raise HTTPException(status_code=404, detail="This session was deleted.")
    key = await asyncio.to_thread(_enter, request.app, tid)
    return {"session_key": key}


# ---- delete ----------------------------------------------------------------------


def _sessions_dir(app) -> Path | None:
    """This hub's Claude Code transcript directory, or ``None`` without a project."""
    cwd = getattr(app.state, "project_cwd", None)
    if not cwd:
        return None
    from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery

    return SessionDiscovery(cwd)._resolve_sessions_dir()


def _owned(app, tid: str, ledger: session_ledger.Ledger, sessions_dir: Path | None) -> bool:
    """Whether *tid* is this operator's: in their ledger, their meta or their transcripts.

    All three are filed under this hub's one identity, so another operator's
    session is in none of them.
    """
    if tid in ledger.transcripts() or any(session_meta.get(app, tid).values()):
        return True
    return sessions_dir is not None and (sessions_dir / f"{tid}.jsonl").is_file()


def _keys_on(app, tid: str, ledger: session_ledger.Ledger) -> set[str]:
    """The session keys whose current transcript is *tid*."""
    keys = {row.session_key for row in ledger.rows if ledger.current(row.session_key) == tid}
    keys.update(transcript_map.keys_on(app, tid))
    if ledger.current(tid) in (None, tid):
        keys.add(tid)  # a key's first transcript is the key itself
    return keys


def _refusal(app, keys: set[str]) -> str | None:
    """Why the session on *keys* cannot be deleted now (the Agent is on it, or it runs), else ``None``."""
    last = session_meta.get_last_active(app)
    if last is not None and last["session_key"] in keys:
        return "The Agent is on this session. Switch to another session to delete this one."
    registry = getattr(app.state, "pty_registry", None)
    operators = getattr(app.state, "operator_registry", None)
    chats = getattr(operators, "chats", None) if operators is not None else None
    for key in keys:
        pty = registry.get_session(key) if registry is not None else None
        if pty is not None and pty.is_alive:
            return "This session is open in a terminal. Close it there to delete it."
        chat = chats.get(key) if chats is not None else None
        if chat is not None and chat.is_busy:
            return "A turn is running in this session. Wait for it to end to delete it."
    return None


def _delete_artifacts(app, tid: str, ledger: session_ledger.Ledger) -> list[str]:
    """Delete the artifacts the tree lists under *tid*; the ids removed.

    The same set ``session-list`` shows (operator-visible, owned by *tid* over
    the ledger as it was before the delete), each through the store's own
    delete, so a run's head takes its run and every removal is announced.
    """
    root = getattr(app.state, "panel_status_root", None)
    if root is None:
        return []
    from osprey.stores.artifact_store import (
        ArtifactStore,
        artifact_mutation_actor,
        operator_visible,
        owner_of,
    )

    store = ArtifactStore(workspace_root=Path(root), auto_launch=False)
    owned = [
        e.id for e in store.list_entries() if operator_visible(e) and owner_of(e, ledger) == tid
    ]
    removed: list[str] = []
    with artifact_mutation_actor("human"):
        for artifact_id in owned:
            removed.extend(e.id for e in store.delete_with_group(artifact_id))
    return removed


def _remove_transcript_files(tid: str, sessions_dir: Path | None) -> int:
    """Remove ``<tid>.jsonl`` and its sibling ``<tid>/`` (subagents) from this hub's
    transcript directory; how many were there."""
    if sessions_dir is None:
        return 0
    import shutil

    removed = 0
    jsonl = sessions_dir / f"{tid}.jsonl"
    if jsonl.is_file():
        jsonl.unlink()
        removed += 1
    folder = sessions_dir / tid
    if folder.is_dir() and not folder.is_symlink():
        shutil.rmtree(folder)
        removed += 1
    return removed


def _delete_session(
    app, tid: str, ledger: session_ledger.Ledger, sessions_dir: Path | None
) -> dict[str, Any]:
    """Tombstone *tid*, then delete its artifacts, ledger rows, map entries and files.

    The tombstone goes first, so a delete cut short leaves the session hidden
    rather than half-there. The agent record is not touched.
    """
    session_meta.mark_deleted(app, tid)
    pending: set[str] = getattr(app.state, "session_naming_pending", set())
    pending.discard(tid)
    artifacts = _delete_artifacts(app, tid, ledger)
    state_dir = session_meta.ledger_state_dir()
    session_ledger.remove_transcript(state_dir, tid)
    after = session_ledger.load(state_dir)
    for key in transcript_map.keys_on(app, tid):
        transcript_map.set(app, key, after.current(key) or key)
    files = _remove_transcript_files(tid, sessions_dir)
    return {"transcript_id": tid, "deleted_artifacts": artifacts, "transcript_files": files}


@router.delete("/api/sessions/{transcript_id}")
async def delete_session(request: Request, transcript_id: str):
    """Delete one of the operator's past sessions with its artifacts.

    404 for a session that is not this operator's (or is already deleted);
    409 while the Agent is on it, a terminal has it open or a turn runs in it.
    An idle chat still holding it is stopped. Every tab drops the row on the
    ``session_deleted`` frame.
    """
    tid = _uuid_id(transcript_id, "transcript_id")
    app = request.app
    ledger = await asyncio.to_thread(session_ledger.load, session_meta.ledger_state_dir())
    sessions_dir = _sessions_dir(app)
    deleted = await asyncio.to_thread(session_meta.deleted_ids, app)
    if tid in deleted or not await asyncio.to_thread(_owned, app, tid, ledger, sessions_dir):
        raise HTTPException(status_code=404, detail="No such session.")
    keys = _keys_on(app, tid, ledger)
    refusal = _refusal(app, keys)
    if refusal:
        raise HTTPException(status_code=409, detail=refusal)
    operators = getattr(app.state, "operator_registry", None)
    chats = getattr(operators, "chats", None) if operators is not None else None
    if chats is not None:
        for key in keys:
            if chats.get(key) is not None:
                await chats.terminate(key)
    body = await asyncio.to_thread(_delete_session, app, tid, ledger, sessions_dir)
    broadcaster = getattr(app.state, "broadcaster", None)
    if broadcaster is not None:
        try:
            broadcaster.broadcast(
                {"type": session_meta.SESSION_DELETED_FRAME, "transcript_id": tid}
            )
        except Exception:  # a frame is a hint; the next listing omits the row anyway
            logger.warning("Could not send the session_deleted frame", exc_info=True)
    return body


def facility_timezone():
    """The zone an export's timestamps are written in: the facility's, else UTC.

    Reached through the module, so a test replaces it like :func:`record_client`.
    """
    from osprey_connectors.config import get_facility_timezone

    return get_facility_timezone()


#: The line a transcript export carries when the record answered only part of it.
INCOMPLETE_TRANSCRIPT_NOTICE = (
    "**Incomplete transcript:** the record answered only part of this session"
)
#: What a transcript export answers when the record cannot be read.
TRANSCRIPT_UNAVAILABLE = "The session record is unavailable; try again shortly."
#: The longest stem an export's file name is cut to.
_EXPORT_NAME_CHARS = 80


def _local_stamp(value: Any, zone) -> str:
    """A record timestamp in *zone*, ``YYYY-MM-DD HH:MM:SS TZ``; ``""`` when unreadable."""
    stamp = _ts(value) if value else None
    if stamp is None:
        return ""
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp.astimezone(zone).strftime("%Y-%m-%d %H:%M:%S %Z")


def _transcript_markdown(title: str, turns: list[dict[str, Any]], read, zone) -> str:
    """The export: ``# title``, an incomplete notice when due, then each turn in order."""
    out = [f"# {title}", ""]
    if read.record == session_record.RECORD_INCOMPLETE:
        reason = f" ({read.reason})" if read.reason else ""
        out += [f"> {INCOMPLETE_TRANSCRIPT_NOTICE}{reason}.", ""]
    for turn in turns:
        who = "Operator" if turn.get("role") == "user" else "Agent"
        stamp = _local_stamp(turn.get("timestamp"), zone)
        out += [f"## {who} · {stamp}" if stamp else f"## {who}", "", turn["content"], ""]
    return "\n".join(out).rstrip() + "\n"


def _export_name(title: str) -> str:
    """An ASCII file name for an export of the session titled *title*."""
    import unicodedata

    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    stem = re.sub(r"[^A-Za-z0-9._]+", "-", ascii_title).strip("-._")[:_EXPORT_NAME_CHARS]
    return f"{stem.rstrip('-._') or 'session'}.md"


@router.get("/api/sessions/{transcript_id}/transcript.md")
async def export_transcript(request: Request, transcript_id: str):
    """The session as Markdown, served as an attachment named from its title.

    Read through the record read the chat replay uses: every operator prompt and
    agent reply in order, each under its facility-local timestamp. A session the
    record holds nothing of is a 404; an unreadable record a 503.
    """
    from urllib.parse import quote

    from fastapi.responses import Response

    tid = _uuid_id(transcript_id, "transcript_id")
    read = await asyncio.to_thread(session_record.read_timeline, tid, client=record_client())
    if read.record == session_record.RECORD_UNAVAILABLE:
        logger.info("transcript export: record unavailable: %s", read.reason)
        raise HTTPException(status_code=503, detail=TRANSCRIPT_UNAVAILABLE)
    if not read.nodes and read.record == session_record.RECORD_COMPLETE:
        raise HTTPException(status_code=404, detail="No such session")
    turns = session_record.chat_history(read.nodes, max_chars=None)
    meta = await asyncio.to_thread(session_meta.get, request.app, tid)
    first_prompt = next((t["content"] for t in turns if t.get("role") == "user"), "")
    title, _source = _title_of(meta, first_prompt)
    zone = await asyncio.to_thread(facility_timezone)
    name = _export_name(title)
    return Response(
        content=_transcript_markdown(title, turns, read, zone),
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": (
                f"attachment; filename=\"{name}\"; filename*=UTF-8''{quote(name)}"
            )
        },
    )


def _current_session(request: Request) -> str | None:
    """The session a route reads: the ``session_id`` key mapped to the transcript
    it names now, else this terminal's current transcript."""
    session_id = request.query_params.get("session_id")
    if session_id:
        return transcript_map.get(request.app, session_id)
    from osprey.mcp_server.workspace.transcript_reader import TranscriptReader

    path = TranscriptReader(request.app.state.project_cwd).find_current_transcript()
    return path.stem if path is not None else None


async def _read_session(request: Request, *, agent_id: str | None = None):
    """The session's timeline read (or its subagent's), or ``None`` when no session is known.

    An id that is not a record token names no session, so it reads as a
    missing one rather than as a record outage.
    """
    from osprey.agent_record.contract import is_token

    session_id = _current_session(request)
    if not session_id or not is_token(session_id):
        return None
    if agent_id is not None and not is_token(agent_id):
        return None
    client = record_client()
    return await asyncio.to_thread(
        session_record.read_timeline, session_id, agent_id=agent_id, client=client
    )


async def _read_session_events(request: Request) -> tuple[list[dict], dict[str, str]]:
    """The session's events and the ``record`` fields of the read behind them."""
    read = await _read_session(request)
    if read is None:
        return [], session_record.record_fields(session_record.RECORD_COMPLETE, "")
    events = session_record.session_events(read.nodes, read.session_id)
    return events, session_record.record_fields(read.record, read.reason)


@router.get("/api/session-agents")
async def session_agents(request: Request):
    """Return agent hierarchy and tool call breakdown for the current session."""
    try:
        from osprey.mcp_server.workspace.tools.session_log import _build_agent_summary

        events, record = await _read_session_events(request)
        agents = _build_agent_summary(events)

        # Group tool_call events by agent_id
        tool_calls_by_agent: dict[str, list[dict]] = {}
        for ev in events:
            if ev.get("type") != "tool_call":
                continue
            aid = ev.get("agent_id") or "main"
            tool_calls_by_agent.setdefault(aid, []).append(ev)

        return {
            "agents": agents,
            "tool_calls_by_agent": tool_calls_by_agent,
            "total_events": len(events),
            **record,
        }
    except Exception as exc:
        logger.debug("session-agents: read failed", exc_info=True)
        return {
            "agents": [],
            "tool_calls_by_agent": {},
            "total_events": 0,
            **_unavailable(str(exc)),
        }


@router.get("/api/session-agent-timeline")
async def session_agent_timeline(request: Request):
    """Return the full internal timeline of a subagent."""
    agent_id = request.query_params.get("agent_id", "")
    if not agent_id:
        return {"timeline": [], "error": "agent_id required"}
    try:
        read = await _read_session(request, agent_id=agent_id)
        if read is None:
            return {
                "agent_id": agent_id,
                "timeline": [],
                "count": 0,
                "record": session_record.RECORD_COMPLETE,
            }
        timeline = session_record.agent_steps(read.nodes)
        return {
            "agent_id": agent_id,
            "timeline": timeline,
            "count": len(timeline),
            **session_record.record_fields(read.record, read.reason),
        }
    except Exception as exc:
        logger.debug("session-agent-timeline: failed", exc_info=True)
        return {"agent_id": agent_id, "timeline": [], "count": 0, **_unavailable(str(exc))}


@router.get("/api/session-log")
async def session_log(request: Request):
    """Return filtered tool call events from the current session."""
    try:
        events, record = await _read_session_events(request)

        # Parse query filters
        agent_filter = request.query_params.get("agent")
        agent_id_filter = request.query_params.get("agent_id")
        tool_filter = request.query_params.get("tool")
        errors_only = request.query_params.get("errors_only", "").lower() == "true"
        since = request.query_params.get("since")
        before = request.query_params.get("before")
        last_n = min(int(request.query_params.get("last_n", "100")), 500)

        # Filter to tool_call events
        filtered = [e for e in events if e.get("type") == "tool_call"]

        if agent_filter:
            filtered = [e for e in filtered if e.get("agent_type") == agent_filter]
        if agent_id_filter:
            filtered = [e for e in filtered if (e.get("agent_id") or "main") == agent_id_filter]
        if tool_filter:
            filtered = [
                e
                for e in filtered
                if tool_filter.lower() in (e.get("tool_name") or e.get("tool") or "").lower()
            ]
        if errors_only:
            filtered = [e for e in filtered if e.get("is_error")]
        if since or before:
            from osprey.mcp_server.workspace.tools.session_log import _parse_iso_timestamp

            since_at = _parse_iso_timestamp(since) if since else None
            before_at = _parse_iso_timestamp(before) if before else None

            def inside(event: dict) -> bool:
                at = _parse_iso_timestamp(event.get("timestamp", ""))
                if at is None:
                    return False
                if since_at is not None and at < since_at:
                    return False
                return before_at is None or at < before_at

            filtered = [e for e in filtered if inside(e)]

        total = len(filtered)
        filtered = filtered[-last_n:]

        return {"events": filtered, "total_events": total, "showing": len(filtered), **record}
    except Exception as exc:
        logger.debug("session-log: read failed", exc_info=True)
        return {"events": [], "total_events": 0, "showing": 0, **_unavailable(str(exc))}


@router.get("/api/session-summary")
async def session_summary(request: Request):
    """Return artifact inventory with channel extraction for the current session."""
    try:
        from osprey.mcp_server.workspace.tools.session_summary import (
            _extract_channels,
            entry_forms,
            form_totals,
        )
        from osprey.stores.artifact_store import ArtifactStore

        store = ArtifactStore(request.app.state.workspace_dir)
        session_id = request.query_params.get("session_id")
        entries = store.list_entries(
            session_filter=session_id or None,
            # Session material, not a shipped example.
            exclude_examples=True,
        )

        total_bytes = 0
        forms = entry_forms(store, entries)
        tools_used: set[str] = set()
        artifact_types: dict[str, int] = {}
        entry_dicts = []

        for entry, (form, label) in zip(entries, forms, strict=True):
            total_bytes += entry.size_bytes
            if entry.tool_source:
                tools_used.add(entry.tool_source)
            artifact_types[entry.artifact_type] = artifact_types.get(entry.artifact_type, 0) + 1
            d = entry.to_dict()
            d["channels"] = _extract_channels(entry)
            d["form"] = form
            d["form_label"] = label
            entry_dicts.append(d)

        return {
            "entries": entry_dicts,
            "totals": {
                "entry_count": len(entries),
                "total_bytes": total_bytes,
                "forms": form_totals(forms),
                "tools_used": sorted(tools_used),
                "artifact_types": artifact_types,
            },
        }
    except Exception:
        logger.debug("session-summary: no artifacts available", exc_info=True)
        return {
            "entries": [],
            "totals": {
                "entry_count": 0,
                "total_bytes": 0,
                "forms": [],
                "tools_used": [],
                "artifact_types": {},
            },
        }


#: Values of the ``full`` flag that lift the per-message cap. The browser sends
#: ``1``; the other spellings are accepted so an operator calling the endpoint
#: by hand is not handed a silently shortened conversation.
_FULL_HISTORY_VALUES = frozenset({"1", "true", "yes", "on"})

#: How the CLI opens the user message it injects when a background agent reports.
_TASK_NOTIFICATION_MARK = "<task-notification>"


def _notification_field(text: str, tag: str) -> str:
    """The contents of ``<tag>…</tag>`` in a task notification, or an empty string."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _agent_done_event(text: str) -> dict[str, Any]:
    """The ``agent_done`` frame the live chat drew for this background agent's report."""
    status = _notification_field(text, "status")
    return {
        "type": "system",
        "subtype": "agent_done",
        "tool_use_id": _notification_field(text, "tool-use-id"),
        "status": "stopped" if status == "killed" else status,
        "summary": _notification_field(text, "summary"),
    }


def _wants_full_messages(request: Request) -> bool:
    """Whether the caller asked for whole messages via the ``full`` flag."""
    return request.query_params.get("full", "").strip().lower() in _FULL_HISTORY_VALUES


def _wants_events(request: Request) -> bool:
    """Whether the caller asked for agent turns as replayable chat events."""
    return request.query_params.get("events", "").strip().lower() in _FULL_HISTORY_VALUES


def _ts(value: Any) -> datetime | None:
    """A transcript timestamp as a datetime, or ``None`` when it is unreadable."""
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


#: The note the approval hook gives the model when a person approved a call and the call
#: then ran, or ran and failed. A copy of ``APPROVED_CALL_NOTE`` in the hook at
#: ``templates/claude_code/claude/hooks/osprey_approval.py``; a test pins the two together.
_APPROVED_CALL_NOTE = "The operator approved this call at the approval prompt before it ran."

#: The hook events that run once a call has executed. The approval note arrives on one.
_OUTCOME_HOOK_EVENTS = frozenset({"PostToolUse", "PostToolUseFailure"})

#: How the result of a declined call opens: the chat's answer
#: (``OperatorSession.resolve_permission``), and the CLI's rejection at its own prompt.
_DECLINED_RESULTS = (
    "The operator declined this tool call.",
    "The user doesn't want to proceed with this tool use.",
)


def _approved_call_ids(lines: list[str]) -> set[str]:
    """The ``tool_use_id`` of every call a person approved and the harness then ran.

    The approval hook answers the outcome event with the note, and the CLI records
    that answer as an ``attachment`` entry naming the call. Lines without the note's
    text are skipped before they are parsed.
    """
    approved: set[str] = set()
    for line in lines:
        if _APPROVED_CALL_NOTE not in line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        attachment = entry.get("attachment") if isinstance(entry, dict) else None
        if (
            not isinstance(attachment, dict)
            or attachment.get("hookEvent") not in _OUTCOME_HOOK_EVENTS
        ):
            continue
        tool_use_id = attachment.get("toolUseID")
        # The note sits in the hook's stdout and in the context the CLI made of it.
        carries_note = any(_APPROVED_CALL_NOTE in str(value) for value in attachment.values())
        if carries_note and isinstance(tool_use_id, str) and tool_use_id:
            approved.add(tool_use_id)
    return approved


def _replayed_decision(
    tool_use_id: str, text: str, is_error: bool, approved: set[str]
) -> str | None:
    """The decision a replayed call's row shows, or ``None`` when it shows none.

    ``allowed`` for a call the approval note names: a person allowed it, and it ran.
    The transcript cannot tell "Allowed for this session" from a one-off allow, so a
    session allow replays as ``allowed`` too. ``denied`` for an error result that is an
    operator's decline, from either surface. A call a safety hook refused never reached the
    operator; its result opens with ``PreToolUse:<tool> hook error:``, so it shows no pill.
    """
    if tool_use_id in approved:
        return "allowed"
    if is_error and text.startswith(_DECLINED_RESULTS):
        return "denied"
    return None


def _operator_text(blocks: list[Any]) -> str:
    """The operator's own words in a user entry: its text blocks, less the CLI's markers.

    The CLI writes its bookkeeping as user text — the marker of an interrupted or
    rejected call, a slash command's echo and its captured output. Each is dropped
    block by block, so a marker beside a tool result or beside the operator's words
    costs neither, and the marker alone never makes a bubble.
    """
    from osprey.mcp_server.workspace.transcript_reader import (
        COMMAND_ECHO_PREFIXES,
        INTERRUPT_MARKER,
        LOCAL_COMMAND_PREFIXES,
    )

    markers = (INTERRUPT_MARKER, *COMMAND_ECHO_PREFIXES, *LOCAL_COMMAND_PREFIXES)
    return "\n".join(
        block["text"]
        for block in blocks
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and not block["text"].lstrip().startswith(markers)
    ).strip()


#: A slash command's echo, as the CLI writes it for a skill or a command file.
_COMMAND_NAME_RE = re.compile(r"<command-name>/?([^<\s]+)</command-name>")
_COMMAND_ARGS_RE = re.compile(r"<command-args>(.*?)</command-args>", re.DOTALL)

#: How the CLI opens the instructions it injects when a skill is loaded.
_SKILL_BODY_MARK = "Base directory for this skill"


def _operator_command(blocks: list[Any]) -> tuple[str, str] | None:
    """The command and arguments an operator ran, from the CLI's echo of it.

    The CLI echoes a skill or a command file the operator ran as a user entry
    opening with ``<command-message>``, naming the command in ``<command-name>``
    and its arguments in ``<command-args>``. A built-in's echo opens with
    ``<command-name>`` and is not read here: it stays bookkeeping.

    Returns:
        ``(name, args)``, or ``None`` for an entry that is no such echo. ``args``
        is the echo's text as the CLI wrote it, whitespace and all, so the
        attached-files block the composer appends still reads as one.
    """
    from osprey.mcp_server.workspace.transcript_reader import COMMAND_MESSAGE_PREFIX

    for block in blocks:
        text = block.get("text") if isinstance(block, dict) else None
        if not isinstance(text, str) or not text.lstrip().startswith(COMMAND_MESSAGE_PREFIX):
            continue
        name = _COMMAND_NAME_RE.search(text)
        if name is None:
            return None
        args = _COMMAND_ARGS_RE.search(text)
        return name.group(1), (args.group(1) if args else "")
    return None


#: The stop reasons of a message the agent goes on from: it waits on its tool calls,
#: or the API paused a long turn for the CLI to resume. Any other stop ends the response.
_CONTINUING_STOPS = frozenset({"tool_use", "pause_turn"})


def _stopped_by_operator(blocks: list[Any]) -> bool:
    """Whether a user entry carries the marker the CLI writes when the operator stops a turn."""
    from osprey.mcp_server.workspace.transcript_reader import INTERRUPT_MARKER

    return any(
        isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and block["text"].lstrip().startswith(INTERRUPT_MARKER)
        for block in blocks
    )


def _api_error_event(error: Any) -> dict[str, Any] | None:
    """The ``error`` frame the live chat drew for an answer the model API failed.

    *error* is the kind of failure the CLI wrote on the entry (``authentication_failed``,
    ``rate_limit``, …), which the record keeps on the answer as ``api_error``. The SDK
    reads the same field, and the live stream draws it as this frame
    (``operator_session._event_to_wire``); a test pins the two together.
    Returns ``None`` for an answer that did not fail.
    """
    if not error:
        return None
    return {
        "type": "error",
        "message": f"API error: {error}",
        "error_type": "AssistantMessageError",
    }


#: The tools that start a subagent; the chat draws each call as that agent's chip.
_SUBAGENT_TOOLS = frozenset({"Agent", "Task"})


def _transcript_nodes(path: Path) -> list[dict[str, Any]]:
    """The timeline nodes the agent record makes of the session transcript at *path*.

    The record's own transcript parser and timeline builder, applied to one local
    session: its file and every subagent file beside it
    (``<session>/subagents/agent-<id>.jsonl``), at the paths the record reads them
    under, so a transcript can be replayed exactly as the record would serve it.
    """
    from dataclasses import replace
    from datetime import UTC, timedelta

    from osprey.agent_record.ingest.transcripts import parse_line
    from osprey.agent_record.reader import RecordItem, build_timeline

    files = [(f"local/{path.name}", path)]
    subagents = path.parent / path.stem / "subagents"
    files += [
        (f"local/{path.stem}/subagents/{sub.name}", sub)
        for sub in sorted(subagents.glob("*.jsonl"))
    ]
    arrived = datetime.now(UTC)
    items: list[RecordItem] = []
    index = 0
    for relpath, file in files:
        for line in file.read_bytes().splitlines():
            for entry in parse_line("local", relpath, line):
                # Arrival in reading order keeps entries stamped alike in file order.
                arrival = arrived + timedelta(microseconds=index)
                items.append(RecordItem(replace(entry, inserted_at=arrival), None))
            index += 1
    items.sort(key=lambda i: (i.entry.ts, i.entry.inserted_at))
    return [node.to_json() for node in build_timeline(items)]


def _node_ts(node: Any) -> str:
    ts = node.get("ts") if isinstance(node, dict) else None
    return ts if isinstance(ts, str) else ""


def _node_body(node: Any) -> dict[str, Any]:
    body = node.get("body") if isinstance(node, dict) else None
    return body if isinstance(body, dict) else {}


def _approved_in(nodes: list[dict[str, Any]]) -> set[str]:
    """The calls a person approved, from the hook attachments among *nodes*."""
    return _approved_call_ids(
        [
            raw
            for n in nodes
            if n.get("kind") == "debug" and isinstance(raw := _node_body(n).get("raw"), str)
        ]
    )


def _tool_use_event(node: Mapping[str, Any]) -> dict[str, Any]:
    """The ``tool_use`` frame of a ``tool_call`` node."""
    from osprey.interfaces.web_terminal.operator_session import format_tool_name
    from osprey.interfaces.web_terminal.routes.chat import clip_input

    body = _node_body(node)
    name = body.get("name", "")
    return {
        "type": "tool_use",
        "tool_name": format_tool_name(name),
        "tool_name_raw": name,
        "tool_use_id": body.get("id", "") or node.get("tool_use_id") or "",
        "input": clip_input(body.get("input", {})),
    }


def _tool_result_event(result: Mapping[str, Any], tool_use_id: str, approved: set[str]) -> dict:
    """The ``tool_result`` frame of a result node, read as the live stream reads it."""
    from osprey.interfaces.web_terminal.tool_preview import (
        tool_result_preview,
        tool_result_refusal,
        tool_result_text,
    )

    body = _node_body(result)
    text = tool_result_text(body.get("content"))
    is_error = bool(body.get("is_error"))
    # Read before the preview cut, as the live stream does.
    refusal, shown = tool_result_refusal(text, is_error)
    preview, truncated = tool_result_preview(shown)
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "is_error": is_error,
        "preview": preview,
        "truncated": truncated,
        "refusal": refusal,
        "decision": _replayed_decision(tool_use_id, text, is_error, approved),
    }


def _subagent_events(timeline: list[dict[str, Any]], parent: str) -> list[dict[str, Any]]:
    """A subagent's text, calls and results as the frames the live chat drew in its chip.

    *timeline* is the subagent's own stream as the record nests it under the call
    that started it. Each frame carries ``parent_tool_use_id``, the way the live
    stream marks a subagent's own records, so the replay files it under that
    agent's chip. Its reasoning, and the prompt it was started with, are left
    out, as the chip never shows them.
    """
    approved = _approved_in(timeline)
    events: list[dict[str, Any]] = []
    for node in timeline:
        kind = node.get("kind")
        body = _node_body(node)
        if kind == "answer" and str(body.get("text") or "").strip():
            events.append({"type": "text", "content": body["text"]})
        elif kind == "tool_call":
            call = _tool_use_event(node)
            events.append(call)
            result = node.get("result")
            if isinstance(result, dict):
                events.append(_tool_result_event(result, call["tool_use_id"], approved))
        elif kind in ("tool_result", "subagent") and not node.get("orphan"):
            tool_use_id = node.get("tool_use_id") or body.get("tool_use_id", "")
            events.append(_tool_result_event(node, tool_use_id, approved))
    return [{**event, "parent_tool_use_id": parent} for event in events]


def _subagent_timelines(nodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Each subagent call's own stream, by the ``tool_use_id`` of the call that started it.

    The record nests a subagent's stream under its call once the call's result
    names the agent. A subagent still at work when the session is read has no
    such result yet, so the record serves its stream as a group of its own; each
    such group is the stream of the earliest subagent call before it that has no
    result and no stream yet, the call that is waiting on it.
    """
    timelines: dict[str, list[dict[str, Any]]] = {}
    waiting: list[str] = []
    for node in nodes:
        sub = node.get("subagent")
        if node.get("orphan"):
            timeline = sub.get("timeline") if isinstance(sub, dict) else None
            if waiting and isinstance(timeline, list):
                timelines[waiting.pop(0)] = timeline
            continue
        if node.get("agent_id") or node.get("kind") != "tool_call":
            continue
        call = _tool_use_event(node)
        if call["tool_name_raw"] not in _SUBAGENT_TOOLS:
            continue
        if isinstance(sub, dict) and isinstance(sub.get("timeline"), list):
            timelines[call["tool_use_id"]] = sub["timeline"]
        elif not isinstance(node.get("result"), dict):
            waiting.append(call["tool_use_id"])
    return timelines


def _chat_turns(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A session's main stream as the chat's turns, agent turns carrying their events.

    A user turn is ``{role, content, timestamp}``. Everything the agent did up
    to the next user prompt is one assistant turn whose ``events`` are the
    frames a live stream would have carried — ``text``, ``thinking``,
    ``tool_use``, ``tool_result`` — closed by a ``result`` whose duration is
    read off the timestamps. ``content`` holds the turn's joined text, so a
    client that ignores ``events`` still replays the words. The results of one
    transcript line's calls follow that line, as they did live.

    The session's last turn gets its ``result`` only when the record shows that
    it ended: its last answer stopped (``end_turn``, or any stop other than one
    for tool calls) with no background agent of the turn still to report, or
    the operator stopped the turn. A turn that waits on a call, reads a result,
    waits on a background agent or answers one's report is still running, as a
    reload mid-turn finds it, so it is left open for the live stream to
    continue. An answer that records no stop reason gives no sign either way and
    is read as the end.

    An answer the model API failed (its ``api_error``) replays as the ``error``
    frame the live stream drew for it, ahead of its text. A turn whose last
    answer failed closes with ``is_error`` set.

    A ``thinking`` event carries ``duration_ms``: the time from the entry before
    it to the entry that holds the reasoning, which is how long the agent spent
    on it. It is left out when either timestamp is missing or unreadable.

    A ``tool_result`` event carries its call's ``decision`` when the session
    shows one; see :func:`_replayed_decision`. It carries ``refusal`` exactly as
    the live stream does (:func:`~osprey.interfaces.web_terminal.tool_preview.tool_result_refusal`).

    A subagent's work follows the call that started it, read from the stream the
    record holds for it (see :func:`_subagent_timelines`), so its chip replays
    with the activity it showed live. Those frames stay out of ``content``.

    A skill or a command file the operator ran by its slash command replays as
    the operator's turn, ``/name args`` as typed rather than the CLI's echo of
    it. A skill then opens the agent's turn with the row the live stream drew
    for it (:func:`~osprey.interfaces.web_terminal.slash_commands.operator_skill_frames`).
    """
    from osprey.interfaces.web_terminal.operator_session import _ASYNC_LAUNCH_MARK

    main = [n for n in nodes if isinstance(n, dict) and not n.get("agent_id")]
    approved = _approved_in(main)
    subagents = _subagent_timelines([n for n in nodes if isinstance(n, dict)])

    turns: list[dict[str, Any]] = []
    agent: dict[str, Any] | None = None
    started: datetime | None = None
    last: datetime | None = None
    #: The timestamp of the conversation entry before the one being read.
    previous: datetime | None = None
    #: Whether the record shows the current agent turn over, as its latest entry reads.
    ended = True
    #: Whether the current agent turn's latest answer is one the model API failed.
    failed = False
    #: The current turn's subagent calls, and those the CLI sent to the
    #: background that have not reported yet: the live turn stays open for them.
    agent_calls: set[str] = set()
    background: set[str] = set()

    def close_agent(*, at_end: bool = False) -> None:
        nonlocal agent, failed
        if agent is None:
            return
        # A turn the operator's next prompt follows is over. The session's last
        # turn is over only when its own entries say so.
        if not at_end or (ended and not background):
            duration = (last - started).total_seconds() * 1000 if started and last else None
            agent["events"].append({"type": "result", "is_error": failed, "duration_ms": duration})
        agent["content"] = "\n\n".join(
            e["content"]
            for e in agent["events"]
            if e["type"] == "text" and not e.get("parent_tool_use_id")
        )
        turns.append(agent)
        agent = None
        failed = False
        agent_calls.clear()
        background.clear()

    def open_agent(stamp_text: str) -> dict[str, Any]:
        nonlocal agent
        if agent is None:
            agent = {"role": "assistant", "content": "", "events": [], "timestamp": stamp_text}
        return agent

    def add_result(current: dict[str, Any], event: dict[str, Any]) -> None:
        current["events"].append(event)
        if event["tool_use_id"] in agent_calls and _ASYNC_LAUNCH_MARK in event["preview"]:
            # The CLI sent the subagent to the background: the turn waits on its report.
            background.add(event["tool_use_id"])

    #: The skill an operator's slash command named, until the CLI's entry that
    #: loads it shows it was a skill (a command file loads no such entry).
    ran: tuple[str, str] | None = None
    skills_run = 0
    for group in session_record.lines(main):
        kinds = {n.get("kind") for n in group}
        if kinds <= {"debug"}:
            continue  # the CLI's own records: nothing of the conversation
        stamp_text = _node_ts(group[0])
        prompts = [n for n in group if n.get("kind") == "prompt"]
        bodies = [_node_body(n) for n in prompts]
        blocks = [{"type": "text", "text": session_record.prompt_text(b)} for b in bodies]
        if any(b.get("is_meta") for b in bodies):
            # The instructions a skill injects are not shown; the skill's row is.
            if ran is not None and session_record.prompt_text(bodies[0]).startswith(
                _SKILL_BODY_MARK
            ):
                from osprey.interfaces.web_terminal.slash_commands import operator_skill_frames

                skills_run += 1
                open_agent(stamp_text)["events"].extend(
                    operator_skill_frames(*ran, f"operator-skill-{skills_run}")
                )
                # The agent has the skill's instructions and answers them next.
                ended = False
            ran = None
            continue
        ran = None
        stamp = _ts(stamp_text)
        since, previous = previous, stamp if stamp is not None else previous

        if prompts:
            command = _operator_command(blocks)
            if command is not None:
                from osprey.interfaces.web_terminal.slash_commands import strip_attachment_block

                # What the operator typed, as the live chat showed it.
                name, args = command
                close_agent()
                turns.append(
                    {
                        "role": "user",
                        "content": f"/{name} {args.strip()}".rstrip(),
                        "timestamp": stamp_text,
                    }
                )
                started = stamp
                # A skill's row, as the live one, leaves out the files attached to it.
                ran = (name, strip_attachment_block(args).strip())
                continue
            text = _operator_text(blocks)
            # A background agent's report is injected as a user message; it
            # continues the agent's turn rather than starting an operator one.
            if text and not text.startswith(_TASK_NOTIFICATION_MARK):
                close_agent()
                turns.append({"role": "user", "content": text, "timestamp": stamp_text})
                started = stamp
                continue
            if text.startswith(_TASK_NOTIFICATION_MARK):
                current = open_agent(stamp_text)
                if stamp is not None:
                    last = stamp
                # The agent answers the report next: the turn goes on.
                ended = False
                done = _agent_done_event(text)
                background.discard(done["tool_use_id"])
                current["events"].append(done)
                continue
            if _stopped_by_operator(blocks):
                # The operator stopped the turn, and its background agents with it.
                ended = True
                background.clear()
        conversation = [
            n
            for n in group
            if n.get("kind") in ("answer", "thinking", "tool_call", "tool_result", "subagent")
            and not (n.get("kind") == "answer" and not str(_node_body(n).get("text") or "").strip())
        ]
        if not conversation:
            continue
        current = open_agent(stamp_text)
        if stamp is not None:
            last = stamp
        answers = [n for n in conversation if n.get("kind") == "answer"]
        if answers or any(n.get("kind") in ("thinking", "tool_call") for n in conversation):
            stop = next(
                (b["stop_reason"] for n in answers if (b := _node_body(n)).get("stop_reason")),
                None,
            )
            calls = any(n.get("kind") == "tool_call" for n in conversation)
            # The agent reads its calls' results next; else its answer's stop decides.
            ended = not calls and stop not in _CONTINUING_STOPS
            error = _api_error_event(next((_node_body(n).get("api_error") for n in answers), None))
            failed = error is not None
            if error is not None:
                current["events"].append(error)
        else:
            # The agent reads the results next.
            ended = False
        pending: list[dict[str, Any]] = []
        for node in conversation:
            kind = node.get("kind")
            body = _node_body(node)
            if kind == "answer":
                current["events"].append({"type": "text", "content": body["text"]})
            elif kind == "thinking":
                event: dict[str, Any] = {"type": "thinking", "content": body.get("text", "")}
                if since is not None and stamp is not None:
                    event["duration_ms"] = max(0, round((stamp - since).total_seconds() * 1000))
                current["events"].append(event)
            elif kind == "tool_call":
                call = _tool_use_event(node)
                current["events"].append(call)
                tool_use_id = call["tool_use_id"]
                if call["tool_name_raw"] in _SUBAGENT_TOOLS:
                    agent_calls.add(tool_use_id)
                    if tool_use_id in subagents:
                        current["events"].extend(
                            _subagent_events(subagents[tool_use_id], tool_use_id)
                        )
                result = node.get("result")
                if isinstance(result, dict):
                    pending.append(_tool_result_event(result, tool_use_id, approved))
                    if (at := _ts(_node_ts(result))) is not None:
                        last = max(last, at) if last else at
                        # The entry that carried the result is the one before the next.
                        previous = max(previous, at) if previous else at
            else:
                tool_use_id = node.get("tool_use_id") or body.get("tool_use_id", "")
                add_result(current, _tool_result_event(node, tool_use_id, approved))
        for event in pending:
            add_result(current, event)
    close_agent(at_end=True)
    return turns


@router.get("/api/session-chat")
async def session_chat(request: Request):
    """Return conversation turns of a session, read from the agent record.

    ``session_id`` names a session key, and a key stops naming its transcript
    the moment a ``/clear`` replaces the conversation, so the key is mapped
    through :mod:`osprey.interfaces.web_terminal.transcript_map` first: a caller
    that knows only its key still reads the conversation that key is on.

    ``events=1`` answers the chat's turns with their replayable events
    (:func:`_chat_turns`). ``full=1`` leaves each message whole, which is what
    replaying a conversation onto another surface needs. Without either, each
    message is cut at the diagnostics view's cap.
    """
    from osprey.mcp_server.workspace.transcript_reader import MAX_CHAT_MESSAGE_LENGTH

    try:
        read = await _read_session(request)
        if read is None:
            return {"turns": [], "count": 0, "record": session_record.RECORD_COMPLETE}
        if request.query_params.get("session_id") and _wants_events(request):
            turns = _chat_turns(read.nodes)
        elif request.query_params.get("session_id") and _wants_full_messages(request):
            turns = session_record.chat_history(read.nodes, max_chars=None)
        else:
            turns = session_record.chat_history(read.nodes, max_chars=MAX_CHAT_MESSAGE_LENGTH)
        return {
            "turns": turns,
            "count": len(turns),
            **session_record.record_fields(read.record, read.reason),
        }
    except Exception as exc:
        logger.debug("session-chat: read failed", exc_info=True)
        return {"turns": [], "count": 0, **_unavailable(str(exc))}
