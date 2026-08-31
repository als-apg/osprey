"""WebSocket routes for terminal PTY and operator (Agent SDK) sessions."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import tempfile
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from osprey.audit.posture import OSPREY_AGENT_DATA_ROOT
from osprey.interfaces.common_middleware import (
    HTTP_MUTATION_POSTURE,
    HTTP_MUTATION_SURFACE,
    read_cookie_candidates,
    session_cookie_name,
)
from osprey.interfaces.web_auth import PANEL_TOKEN_ENV, get_web_credentials
from osprey.interfaces.web_terminal.operator_session import (
    POSTURE_SESSION_ENV,
    POSTURE_SOURCE_ENV,
    POSTURE_SOURCE_LIVE,
    POSTURE_SOURCE_SPAWN,
    build_operator_child_env,
    resolve_agent_data_root,
)
from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery
from osprey_connectors import session_store

logger = logging.getLogger(__name__)

router = APIRouter()

_UUID_RE = re.compile(r"^[a-f0-9-]{36}$")

# The posture surface's *closed* key grammar: a canonical lowercase UUID, and
# nothing else. ``_UUID_RE`` above is the loose shape check the resume path
# (``switch_session``) applies to ids Claude itself wrote; it admits any 36
# characters drawn from ``[a-f0-9-]``, which is fine for "does this look like a
# session file stem" and much too wide for a key that is written to a store on
# disk and later decides a child process's execution mode. Both identities the
# posture route can legitimately name — a discovered PTY session (a Claude
# session-file stem) and a live chat-pool session (``crypto.randomUUID()`` in
# ``static/js/chat.js``) — are canonical UUIDs, so nothing shipped loses reach
# by closing the grammar here. The ``/ws/operator`` pool's minted
# ``operator-<hex8>`` keys stay unaddressable by design.
_POSTURE_KEY_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# ── Per-(session, target) runtime posture ────────────────────────────────────
#
# The posture is the operator's per-target sandbox toggle for one session:
# narrow one control target to ``sandbox`` and leave the others alone. It is
# deliberately *not* a config edit — config is a build-time input that reaches
# the agent only through a re-render, and it is deployment-wide, whereas this
# is one session's view of one machine.
#
# The store is a single JSON file that this server writes and three readers
# answer from; its path, its shapes and its lookup rule live in
# :mod:`osprey_connectors.session_store`, which is the canonical statement of
# all three. Nothing here re-implements any of them: the values below are that
# module's, the path comes from :func:`session_store.store_path`, and every
# decode goes through :func:`session_store.parse_store`. A store this server
# filtered differently from the connector chain would be a narrowing the
# operator can see and the machine cannot.
POSTURE_SANDBOX = session_store.POSTURE_SANDBOX
POSTURE_WRITES = session_store.POSTURE_WRITES


class PostureRequest(BaseModel):
    """Body of ``POST /api/terminal/posture``.

    ``posture`` is a ``Literal`` so an unknown value is rejected by request
    validation with a 422 naming the field, before any handler code runs — the
    value decides whether a session's writes are refused, and a silent coercion
    to some default would be the worst possible failure here.

    ``target`` names one configured control target, or the literal
    :data:`ALL_TARGETS` for the popover's ``[ Sandbox everything ]`` gesture.
    It is a plain string rather than a ``Literal`` for the reason given on
    :class:`TargetRequest`: which targets exist is a property of the rendered
    deployment, not of this build's vocabulary.
    """

    session_id: str
    target: str
    posture: Literal["sandbox", "writes"]


class TargetRequest(BaseModel):
    """Body of ``POST /api/terminal/target``.

    ``target`` is a plain string rather than a ``Literal``: which targets exist
    is a property of the *rendered deployment*, not of this build's vocabulary,
    and a name pinned here would either admit a machine the deployment never
    described or refuse one a later render adds. The handler checks it against
    :func:`~osprey_connectors.types.configured_targets` instead, which is the
    same list the roster, the prober and the popover enumerate.
    """

    session_id: str
    target: str


def is_posture_key(session_id: str) -> bool:
    """Whether *session_id* matches the posture surface's closed key grammar.

    The public half of :data:`_POSTURE_KEY_RE`, for the one caller outside this
    module that has to agree with the posture route on what it can address:
    ``routes/chat.py`` labels a chat child's ``posture_source`` by it, because
    a key this answers ``False`` for is a key no posture store will ever answer
    for. Kept as a function rather than an exported pattern so the grammar
    itself stays private and there is one place to change it.
    """
    return bool(_POSTURE_KEY_RE.match(session_id))


def _require_session_uuid(session_id: str) -> None:
    """Refuse *session_id* unless it is a canonical, bare session UUID.

    One implementation for both posture routes, so the two cannot drift on the
    status, the error slug or the sentence. An arbitrary string can never
    become a store key that is then written to disk.

    The grammar is closed (:data:`_POSTURE_KEY_RE`): eight-four-four-four-twelve
    lowercase hex, no prefix, no suffix. Every key the posture surface can
    legitimately name is minted that way — a Claude session-file stem or a chat
    id from ``crypto.randomUUID()`` — so the closed form costs no reach and
    keeps decorated keys (``operator-<hex8>``) and near-miss strings out of a
    store that decides a child's execution mode.

    Raises:
        HTTPException: 400 ``invalid_session_id`` when the shape does not match.
    """
    if not is_posture_key(session_id):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_session_id",
                "message": "session_id must be a Claude session UUID.",
            },
        )


def _holds_a_chat_pool_entry(app, session_id: str) -> bool:
    """Whether the chat pool holds an entry under *session_id* right now.

    Deliberately **not** a liveness check: ``get_chat_session`` reads the
    pool's session map and a dead-but-unreaped entry answers ``True``. That is
    the right answer for both callers — such a key still names a chat the
    operator can address, and terminating it evicts the corpse, which is what
    wants to happen anyway.

    It is also the *narrower* of this module's two chat probes. The map it
    reads is one of two places a chat can live: a creation still inside
    ``start()`` sits in the pool's ``_pending`` and is invisible here, which on
    the first prompt of a chat is the ordinary state rather than a corner case.
    :func:`_chat_pool_answers_to` is the one that sees both, and it is what the
    addressability gate asks.

    The Simple-mode chat surface (``POST /api/chat``) keys its pool on the
    caller-supplied ``chat_id`` and spawns the child under that key, so the
    pool key and the posture-store key are the same string. Membership is read
    through the registry's own read-only accessor — never the pool's internals
    — so a probe cannot refresh an entry's idle clock or evict anything.

    Absent or unfamiliar registries answer ``False`` rather than raising: the
    caller is an existence gate, and a registry that cannot be asked simply has
    no chat session to offer.
    """
    registry = getattr(app.state, "operator_registry", None)
    getter = getattr(registry, "get_chat_session", None)
    if not callable(getter):
        return False
    return getter(session_id) is not None


def _chat_pool_answers_to(app, session_id: str) -> bool:
    """Whether the chat pool would answer to *session_id* at all.

    The *addressability* probe, and deliberately a wider question than
    :func:`_holds_a_chat_pool_entry`: it also says ``True`` while a creation is
    still inside ``start()``. That window is not a corner case on this surface
    — it is the first prompt of a chat, the moment the child is being armed
    with tools — and answering ``False`` there refuses the operator's toggle
    with a 409 that stores nothing, on a session that is starting in front of
    them. The narrowing they were refused would have been read live by that
    child's very first write.

    Reached through the registry's own read-only facade
    (:meth:`~osprey.interfaces.web_terminal.operator_session.OperatorRegistry.has_chat_key`),
    so a probe disturbs no LRU order and creates nothing. A registry that
    predates the facade — a hand-rolled double, say — falls back to the
    narrower session-map probe rather than raising, the same tolerance the rest
    of this surface grants an unfamiliar registry.
    """
    registry = getattr(app.state, "operator_registry", None)
    prober = getattr(registry, "has_chat_key", None)
    if callable(prober):
        return bool(prober(session_id))
    return _holds_a_chat_pool_entry(app, session_id)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Serialize *data* to *path* as JSON, atomically.

    Mirrors :func:`osprey.interfaces.web_terminal.feedback_store._atomic_write`
    (the same pattern recurs in ``stores/base_store.py`` and
    ``deployment/compose_merge.py``): a temporary file in the destination
    directory, flushed and fsynced, then ``os.replace``d over the target, so a
    crash mid-write can never leave a half-written store that the next startup
    would read as "no session is sandboxed". ``path.parent`` must exist.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2)
            handle.flush()
            with suppress(OSError):
                os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_name)
        raise


class PostureStoreUnavailable(RuntimeError):
    """A narrowing could not be recorded, so it did not happen.

    Raised by :func:`persist_or_raise` for both ways the commit point can fail
    — nowhere to write (the agent-data root does not resolve) and the write
    itself failing — because the operator-facing answer is the same 503 either
    way: the toggle was refused and nothing changed. :attr:`error` names which
    one for the response body and the log.
    """

    def __init__(self, error: str, message: str) -> None:
        super().__init__(message)
        self.error = error
        self.message = message


def _posture_store_path() -> Path | None:
    """Where the posture store lives, or ``None`` when it has no location.

    Delegates to :func:`osprey_connectors.session_store.store_path` — the ONE
    path rule (env ``OSPREY_AGENT_DATA_ROOT``, else
    ``resolve_shared_data_root()``) that `session_store` owns.
    """
    return session_store.store_path()


#: Prefix of the ``/ws/operator`` session keys (``operator-<hex8>``, minted per
#: accepted websocket). Postures under these keys are deliberately
#: **non-durable**: see :func:`_load_postures`.
_NON_DURABLE_KEY_PREFIX = "operator-"


def _load_postures(path: Path) -> dict[str, dict[str, str]]:
    """Read the persisted narrowings from *path*, tolerating every absence.

    The decode itself is :func:`session_store.parse_store` — the shared filter
    every reader of this file runs, which is where the legacy bare ``"sandbox"``
    becomes a narrowing of every target, a bare ``"writes"`` disappears (the
    writes posture is the *absence* of an entry, never a stored assertion), and
    anything unrecognised is dropped rather than honoured. A missing or corrupt
    file yields an empty store — the operator can set the postures again, which
    is a far better outcome than every toggle failing on a file nobody can
    repair from the browser.

    What this function adds is the one rule that belongs to the *web server's
    startup load* and to no other reader: ``operator-`` keys are dropped.

    **Operator keys do not survive a restart.** ``operator-<hex8>`` keys name a
    ``/ws/operator`` connection, and that registry is per *process*: the key is
    minted when the websocket is accepted and is addressable by nothing else,
    so a key restored from disk can never name a live session. Keeping such an
    entry would grow the store without bound with keys nothing will ever spawn
    under, and would let a future key collision hand a fresh connection a
    stranger's posture. Durability of the operator half stays out of scope
    until an operator client exists to define its reconnect protocol.

    The other two key shapes survive the filter, and one of them has to. A PTY
    session's Claude UUID names a session file that outlives the process, so
    its posture is durable in the full sense: the key comes back and the
    restored entry governs the respawn.

    A chat ``chat_id`` is weaker, and the honest version is worth stating: the
    shipped client mints a fresh one per page load (``crypto.randomUUID()`` in
    ``static/js/chat.js``), so a restored chat posture is *speculative* — no
    shipped client will ever address that key again, and it would be reachable
    only by a future client that persists its id. Chat keys are nonetheless
    kept, because they are bare canonical UUIDs and so indistinguishable at
    load time from the PTY stems that must survive; filtering them would need
    a key registry this store does not have. The unbounded-growth objection
    that justifies dropping ``operator-`` keys does apply here in miniature —
    it is bounded by one entry per chat the operator actually sandboxed, not
    by one per connection, which is why it is tolerated rather than solved.

    This load-side filter is the single enforcement point.
    :func:`persist_or_raise` still writes whatever the in-memory store holds,
    operator keys included — the in-memory entries are live and load-bearing
    for the rest of the process's life, and dropping them on the way *out*
    would only add a second place for the rule to drift.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        logger.warning("Could not read the session-posture store at %s", path, exc_info=True)
        return {}
    return {
        key: entry
        for key, entry in session_store.parse_store(raw).items()
        if not key.startswith(_NON_DURABLE_KEY_PREFIX)
    }


def _write_store(path: Path, store: dict[str, dict[str, str]]) -> None:
    """Put *store* on disk at *path*, atomically. Raises on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, store)


def _session_postures(app) -> dict[str, dict[str, str]]:
    """Return ``app.state.session_postures``, loading it from disk on first use.

    Lazily initialised rather than wired into the lifespan so the whole feature
    lives in this module. First access is whichever comes first after a
    restart — a spawn or a posture route — and both go through here, so a
    recreated container never serves a session whose persisted narrowing it has
    not read.

    **The one-shot migration.** The session-wide posture kept its store
    directly under the agent-data root; this one lives in the ``control_target``
    directory beside the state file. When the new path holds nothing, the old
    one is read through once, migrated into memory and written to the new path
    straight away — the old file is left where it is, because a deployment that
    rolls back must still find it. Persisting immediately is what ends the
    read-through: :func:`session_store.legacy_store_path` answers only while
    the new file does not exist, so every other reader is on the new path from
    that moment. Nothing is migrated when the old store narrowed nothing —
    there is no such thing as an empty entry worth writing.

    A load taken while the store has **no location** is kept provisional and
    retried on the next access. Caching it would let one transient config
    failure at first access outlive itself: every later read would serve an
    empty store and report a sandboxed session as unnarrowed — a silent revert
    to writes, which is precisely what persisting the store exists to prevent.
    """
    store = getattr(app.state, "session_postures", None)
    if store is not None and not getattr(app.state, "session_postures_provisional", False):
        return store

    path = _posture_store_path()
    migrated = False
    if path is None:
        loaded: dict[str, dict[str, str]] = {}
    elif path.exists():
        loaded = _load_postures(path)
    else:
        legacy = session_store.legacy_store_path()
        loaded = _load_postures(legacy) if legacy is not None else {}
        migrated = bool(loaded)

    if store is None:
        store = loaded
    else:
        # Recovering from an earlier location-less load. The persisted store is
        # authoritative for everything memory has not been told about, but a
        # narrowing the operator set *during* the outage exists only in memory,
        # so it wins on overlap — otherwise the recovery read would quietly
        # undo it. Mutated in place because callers hold this dict.
        merged = {**loaded, **store}
        store.clear()
        store.update(merged)
    app.state.session_postures = store
    app.state.session_postures_provisional = path is None

    if migrated and path is not None:
        try:
            _write_store(path, store)
        except Exception:  # noqa: BLE001 — the old file still answers until this lands
            logger.warning(
                "Could not write the migrated session-posture store to %s; the previous "
                "store still governs and the migration is retried on the next restart",
                path,
                exc_info=True,
            )
    return store


def _spawn_posture_key(app, session_key: str) -> str:
    """The key the live child under *session_key* was spawned with.

    ``PtyRegistry.audit_session_key`` resolves a current pool key back to the
    telemetry id a rekeyed session's child exported as
    ``OSPREY_POSTURE_SESSION`` — the key that child's own store reads use, and
    which cannot be rewritten without killing it. Identical to *session_key*
    for everything else, which is the overwhelmingly common case.
    """
    registry = getattr(getattr(app, "state", None), "pty_registry", None)
    resolve = getattr(registry, "audit_session_key", None)
    if resolve is None:
        return session_key
    spawn_key = resolve(session_key)
    return spawn_key if isinstance(spawn_key, str) and spawn_key else session_key


def _posture_entry(app, session_key: str | None) -> dict[str, str]:
    """The narrowings this server holds for one session — ``{target: posture}``.

    Current key first, spawn key second, which is the read half of the
    dual-write in :func:`persist_or_raise`. The two differ for exactly as long
    as one live child outlives a rekey, and each answers for a different
    reader: a session reattached after a server restart is addressed by its
    Claude UUID (the alias map is memory-only and died with the process), while
    the child still running from before the rekey reads the telemetry id it was
    spawned with. Taking the current key first means the entry a route just
    wrote is the one it reads back.

    The returned mapping is the stored one — treat it as read-only; go through
    :func:`persist_or_raise` to change it.
    """
    if not session_key:
        return {}
    store = _session_postures(app)
    entry = store.get(session_key)
    if entry is not None:
        return entry
    spawn_key = _spawn_posture_key(app, session_key)
    if spawn_key != session_key:
        return store.get(spawn_key) or {}
    return {}


def persist_or_raise(app, session_key: str, entry: Any) -> dict[str, str]:
    """Record *entry* as this session's narrowings, or refuse the change.

    **The write is the commit point.** The file lands first and the in-memory
    store is updated only once it has; a failure raises and leaves memory
    exactly as it was. That ordering is the whole point of this function.
    Enforcement now reads the store rather than an environment variable frozen
    at spawn, so memory and disk disagreeing is not a lost convenience — it is
    a session the badge shows as sandboxed whose next write is still permitted,
    or the reverse. Refusing the operator's toggle with a 503 they can retry is
    the honest answer; applying it to half the system is not.

    *entry* is normalised through :func:`session_store.parse_store`, so a route
    may hand over a ``{target: posture}`` mapping or one of the legacy bare
    strings and get the same reading every other consumer of this file gets.
    An entry that narrows nothing **removes** the key: absence is how this
    store spells ``writes``, and a stored ``{}`` would be a second spelling.

    Both the current key and the spawn key are written (identically, and they
    are usually the same key), so the running child and a later reattach find
    the same narrowing — see :func:`_posture_entry` for the read order.

    **Synchronous on the event loop, deliberately.** Load, write and the memory
    update run with no ``await`` between them, so two posture POSTs arriving
    together are serialised by the loop itself and the second one's file is
    written from a store that already holds the first one's entry. Moving this
    to ``run_in_threadpool`` for the disk write would reopen exactly that
    read-modify-write race: both requests would compute their candidate from
    the same pre-write store and the later ``os.replace`` would silently drop
    the other's narrowing. The write is a few hundred bytes to the agent-data
    root; it does not earn a thread hop.

    Args:
        app: The web app holding the in-memory store.
        session_key: The session's **current** pool key — the only legal write
            target. The spawn key is a read alias only (:func:`_posture_entry`
            falls back to it): a clear addressed to the spawn key would delete
            that key and the one *it* resolves to, leaving the session's
            current-key entry in place and still narrowing.
        entry: The narrowings to record — ``{target: "sandbox"}``, a bare
            legacy posture string, or anything falsy to clear the session.

    Returns:
        The narrowings as they were stored, ``{}`` when the session was cleared.

    Raises:
        PostureStoreUnavailable: The store has no location, or the write
            failed. Nothing changed in memory or on disk.
    """
    path = _posture_store_path()
    if path is None:
        raise PostureStoreUnavailable(
            "store_unavailable",
            "This deployment's agent-data root does not resolve, so there is nowhere to "
            "record a posture that the agent would read back. No posture was changed.",
        )

    store = _session_postures(app)
    narrowed = session_store.parse_store({session_key: entry}).get(session_key, {})
    candidate = dict(store)
    for key in {session_key, _spawn_posture_key(app, session_key)}:
        if narrowed:
            candidate[key] = dict(narrowed)
        else:
            candidate.pop(key, None)

    try:
        _write_store(path, candidate)
    except Exception as exc:  # noqa: BLE001 — reported to the operator as a 503
        logger.warning(
            "Could not persist the session-posture store to %s; the posture was NOT applied",
            path,
            exc_info=True,
        )
        raise PostureStoreUnavailable(
            "store_write_failed",
            "The posture store could not be written, so the change was not applied. "
            "Check the server's write access to the agent-data root and try again.",
        ) from exc

    # Committed. Mutated in place because callers hold this dict.
    store.clear()
    store.update(candidate)
    return narrowed


#: Sentinel telling "the render could not be read" apart from "the render has
#: no ``control_system:`` block". Both answer writes-off, but only the first is
#: a failure worth logging, and the connector helpers have their own opinion
#: about a ``None`` section that the failure case must not borrow.
_UNREADABLE_SECTION = object()


#: The last render this module parsed: ``(path, signature, config)``. One entry
#: is the whole cache, because one server serves one render — a second path
#: simply replaces it rather than growing a map of stale files.
_CONFIG_MEMO: tuple[Path, tuple[int, int, int], Any] | None = None


def _reset_rendered_config_memo() -> None:
    """Forget the parsed render. For tests, and for anything that rewrites it."""
    global _CONFIG_MEMO
    _CONFIG_MEMO = None


def _rendered_config(config_path: Path | None) -> Any:
    """The WHOLE parsed render, or :data:`_UNREADABLE_SECTION`.

    The whole file rather than the ``control_system:`` block alone because the
    target labels are derived from more than that block — a deployment running a
    stand-in for its live machine says so under ``services:`` — and one caller
    then gets to read the file exactly once for every question it asks.
    Blocking; never call it from the event loop.

    **Memoized by the file's signature.** ``config.yml`` is a build artefact: it
    changes when a deployment is re-rendered and not otherwise, while the chip
    polls the roster per open card and every row it draws asks this render three
    or four questions. Re-parsing a few hundred lines of YAML for each of them
    is the cost this removes. The key is
    ``(st_mtime_ns, st_size, st_ino)`` — the same signature the state-file memo
    uses, and the inode is in it because a re-render lands through a rename and
    can carry the same size and, on a coarse clock, the same mtime.

    The memo is invisible to callers: what comes back is a deep copy, so a
    caller may keep or mutate the render without the next request inheriting it.
    An unreadable render is never memoized — the file may be mid-write — so the
    next call looks again rather than serving a failure until the mtime moves.
    """
    global _CONFIG_MEMO
    try:
        if not config_path:
            return _UNREADABLE_SECTION
        path = Path(config_path)
        signature = _file_signature(path)
        if signature is None:
            return _UNREADABLE_SECTION
        memo = _CONFIG_MEMO
        if memo is not None and memo[0] == path and memo[1] == signature:
            return copy.deepcopy(memo[2])
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — an unreadable config is not a writes-on render
        logger.warning("Could not read the rendered config at %s", config_path)
        return _UNREADABLE_SECTION
    _CONFIG_MEMO = (path, signature, config)
    return copy.deepcopy(config)


def _section_of(config: Any) -> Any:
    """The ``control_system:`` block of an ALREADY-READ render.

    A render that could not be read stays :data:`_UNREADABLE_SECTION`; one that
    parsed to something that is not a mapping has no section, which is a
    different (and non-failing) answer the predicates below already handle.
    """
    if config is _UNREADABLE_SECTION:
        return _UNREADABLE_SECTION
    return config.get("control_system") if isinstance(config, dict) else None


def _control_system_section(config_path: Path | None) -> Any:
    """The rendered ``control_system:`` section, or :data:`_UNREADABLE_SECTION`.

    For the one caller that asks the render a single question. Anything that
    asks it several — the roster, the posture ladder — takes
    :func:`_rendered_config` and passes the parse down through
    :func:`_section_of`, because the target labels are derived from the whole
    file and not from this block. Blocking; never call it from the event loop.
    """
    return _section_of(_rendered_config(config_path))


# ── One state-record resolution per request ──────────────────────────────────
#
# The control-target surfaces — the header chip, its target rows, the switch
# gesture's answer — all read facts the controls server publishes into ONE
# record: the target, its label, the last switch's outcome, the endpoint
# prober's reachability, a pending posture realignment. Resolving which record
# belongs to a session is the expensive part: a scan of the state directory
# plus an ancestor walk that, on a platform without ``/proc``, forks ``ps``.
#
# So a request resolves once and every renderer reads that one record. The memo
# below carries that resolution ACROSS requests as well, because the chip polls
# every few seconds per open card, and re-walking the process table for an
# answer that has not moved is the cost this exists to remove.
#
# What the memo caches is the MATCH — "this session's record is that file" —
# never the file's contents. The contents change constantly and on someone
# else's schedule (the prober republishes reachability every sweep), so a hit
# still re-reads the file whenever its signature moved. That is a file read, not
# a process walk, and it is what keeps a memoized answer fresh rather than
# merely cheap.


@dataclass(frozen=True)
class _RecordMemo:
    """One session's remembered state-record match.

    Attributes:
        pty_pid: The PTY pid the match was made against. A respawned session
            gets a new pid, and the old match must not answer for it.
        path: The state file the match landed on.
        signature: ``(st_mtime_ns, st_size, st_ino)`` of that file when it was
            last read, or ``None`` when it could not be stat'd.
        directory: The state directory's file names at match time. A changed
            set means a file appeared or disappeared, which can change WHICH
            record matches — including into the ambiguous "two matches, no
            answer" case — so the match is re-made rather than trusted.
        record: The record as last read.
    """

    pty_pid: int
    path: Path
    signature: tuple[int, int, int] | None
    directory: tuple[str, ...]
    record: dict[str, Any]


#: Remembered matches, keyed by session key. Bounded (see
#: :data:`_RECORD_MEMO_MAX`); entries are dropped the moment their session
#: resolves no PTY pid or no record.
_SESSION_RECORD_MEMO: dict[str, _RecordMemo] = {}

#: How many sessions' matches to remember. The pools this server runs are far
#: smaller than this, so the cap is a leak stop rather than a policy: keys of
#: sessions that ended without passing back through here would otherwise
#: accumulate for the life of the process. Oldest insertion is evicted first.
_RECORD_MEMO_MAX = 64


def _pid_or_none(value: object) -> int | None:
    """Coerce a record field to ``int``, or ``None`` when it is not a number."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _file_signature(path: Path) -> tuple[int, int, int] | None:
    """``(st_mtime_ns, st_size, st_ino)`` of *path*, or ``None`` when it is gone.

    One ``stat`` — the cheapest question that distinguishes "the writer has
    republished" from "nothing has moved", and the one that answers "the file
    disappeared" at the same time. The inode is in the tuple because both files
    this module memoizes — the control-target state record and the render — are
    replaced through a rename, and a replacement can land with the same size
    and, on a coarse clock, the same mtime.

    Shared by both memos deliberately: two signatures of the same shape, kept
    separately, is two chances to leave the inode out of one of them.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size, stat.st_ino)


def _state_dir_names(target_state: Any) -> tuple[str, ...]:
    """The state directory's file names, sorted. Empty when it cannot be read.

    One directory scan, no forks. It is what lets a memo hit keep the
    resolver's fail-closed ambiguity rule: a second server's record appearing
    inside this PTY's process tree turns one answer into none, and a memo that
    only watched its own file would go on happily naming a target while the
    live resolver had stopped being able to.
    """
    try:
        directory = target_state.state_dir()
        return tuple(sorted(p.name for p in directory.glob(target_state.STATE_FILE_GLOB)))
    except OSError:
        return ()


def _pty_pid_for(app: Any, session_key: str) -> int | None:
    """The pid of the PTY this session runs in, or ``None``.

    ``None`` for a key that names no PTY at all — a chat-pool session, or a
    terminal card whose session has not started yet. Both are ordinary states
    here, not errors: the caller falls back to the deployment render, exactly as
    the posture badge already does.

    Reached through the registry's own read-only accessor, and tolerant of a
    registry that does not have one, for the same reason the chat-pool probes
    above are: this surface grants nothing to an unfamiliar registry.
    """
    registry = getattr(app.state, "pty_registry", None)
    getter = getattr(registry, "get_session", None)
    if getter is None:
        return None
    try:
        session = getter(session_key)
    except Exception:  # noqa: BLE001 — a registry that cannot be asked has no pid
        logger.warning("Could not ask the PTY registry about session %s", session_key)
        return None
    pid = _pid_or_none(getattr(session, "pid", None))
    return pid if pid and pid > 0 else None


def _remember_session_record(
    session_key: str,
    pty_pid: int,
    path: Path,
    signature: tuple[int, int, int] | None,
    directory: tuple[str, ...],
    record: dict[str, Any],
) -> None:
    """Store one match, evicting the oldest when the memo is full."""
    if session_key not in _SESSION_RECORD_MEMO and len(_SESSION_RECORD_MEMO) >= _RECORD_MEMO_MAX:
        _SESSION_RECORD_MEMO.pop(next(iter(_SESSION_RECORD_MEMO)), None)
    _SESSION_RECORD_MEMO[session_key] = _RecordMemo(
        pty_pid=pty_pid,
        path=path,
        signature=signature,
        directory=directory,
        record=record,
    )


def _reset_session_record_memo() -> None:
    """Forget every remembered match. For tests, and for a pool teardown."""
    _SESSION_RECORD_MEMO.clear()


def _reread_matched_file(target_state: Any, entry: _RecordMemo) -> dict[str, Any] | None:
    """Re-read a remembered match's file, or ``None`` when it must be re-matched.

    The republish case: the file the memo matched is still the one this session
    owns, and its writer has simply written to it again. Re-reading is a file
    read; re-matching would be another walk of the process table.

    It is only the *contents* that are allowed to have moved. Anything that
    would change the MATCH sends the caller back to the full resolver: a record
    that no longer parses, one whose ``server_pid`` or ``owner_ppid`` moved (the
    file was recycled by a different server, so the ancestor walk that matched
    it no longer applies), a dead writer, or a target name this build does not
    know — the last two being exactly the checks the resolver itself makes, kept
    here so a memo hit is never weaker than a miss.
    """
    try:
        record = target_state.read_file(entry.path)
        if not isinstance(record, dict):
            return None
        server_pid = _pid_or_none(record.get("server_pid"))
        if server_pid is None or server_pid != _pid_or_none(entry.record.get("server_pid")):
            return None
        if _pid_or_none(record.get("owner_ppid")) != _pid_or_none(entry.record.get("owner_ppid")):
            return None
        if record.get("target") not in target_state.TARGET_NAMES:
            return None
        if not target_state.is_process_alive(server_pid):
            return None
    except Exception:  # noqa: BLE001 — an unreadable record is simply no memo hit
        logger.warning("Could not re-read the target-state file at %s", entry.path)
        return None
    return record


def _memo_writer_is_alive(target_state: Any, record: dict[str, Any]) -> bool:
    """Whether the controls server that published *record* is still running.

    The liveness half of the full resolver's filter (``target_banner``'s
    ``_live_records``), lifted out so the memo's cheapest path — a signature
    that has not moved — can make the same check the resolver and
    :func:`_reread_matched_file` make. A record with no readable ``server_pid``
    is not a live one: an answer this route cannot justify must not be the
    stronger one.
    """
    server_pid = _pid_or_none(record.get("server_pid"))
    if server_pid is None:
        return False
    try:
        return bool(target_state.is_process_alive(server_pid))
    except Exception:  # noqa: BLE001 — an unreadable process table is no memo hit
        logger.warning("Could not check whether the controls server %s is alive", server_pid)
        return False


def _session_record(app: Any, session_key: str) -> dict[str, Any] | None:
    """The control-target state record this session's controls server published.

    The one entry point every control-target surface on this router reads from,
    so a request resolves the process table at most once and every fact it
    renders — target, label, ``last_switch``, ``reachability``,
    ``last_posture_realign`` — comes from the same record. Facts drawn from two
    resolutions could straddle a switch and describe two different machines.

    Fields the writer publishes on its own schedule are optional: read them with
    ``record.get(...)`` and treat absence as "not recorded yet", never as "no".

    **Blocking, and deliberately so**: on a miss it walks the process table,
    which without ``/proc`` forks ``ps``. Call it from a worker thread
    (``run_in_threadpool``), never on the event loop — a wedged process table
    would otherwise stall every request this server is serving, the terminal
    websocket included.

    Returns:
        The record, freshly copied so a caller may keep or mutate it, or
        ``None`` — no PTY (a chat key, or a card whose session has not started),
        no matching record, an ambiguous one, or an unreadable process table.
        The caller renders the deployment baseline for all of them. Never
        raises.
    """
    from osprey.mcp_server.control_system import target_banner, target_state

    pty_pid = _pty_pid_for(app, session_key)
    if pty_pid is None:
        _SESSION_RECORD_MEMO.pop(session_key, None)
        return None

    directory = _state_dir_names(target_state)
    entry = _SESSION_RECORD_MEMO.get(session_key)
    if entry is not None and entry.pty_pid == pty_pid and entry.directory == directory:
        signature = _file_signature(entry.path)
        if signature is not None:
            if signature == entry.signature:
                # Liveness is checked even on the cheapest hit. A controls
                # server that died leaves its file behind untouched, so the
                # signature stops moving and never expires the memo — the chip
                # would go on reporting a target, a switch outcome and a
                # reachability sweep from a process that is gone. Both the full
                # resolver and the re-read path below make this check; a memo
                # hit must never be the weaker answer.
                if _memo_writer_is_alive(target_state, entry.record):
                    return copy.deepcopy(entry.record)
                _SESSION_RECORD_MEMO.pop(session_key, None)
            record = _reread_matched_file(target_state, entry)
            if record is not None:
                _remember_session_record(
                    session_key, pty_pid, entry.path, signature, directory, record
                )
                return copy.deepcopy(record)

    try:
        record = target_banner.session_record_for_pid(pty_pid)
    except Exception:  # noqa: BLE001 — the surface must render, not 500
        logger.warning("Could not resolve the session's control-target record", exc_info=True)
        record = None
    if not isinstance(record, dict):
        _SESSION_RECORD_MEMO.pop(session_key, None)
        return None

    server_pid = _pid_or_none(record.get("server_pid"))
    if server_pid is not None:
        path = target_state.state_file_path(server_pid)
        _remember_session_record(
            session_key, pty_pid, path, _file_signature(path), directory, record
        )
    else:  # pragma: no cover - the resolver only matches records with a live pid
        _SESSION_RECORD_MEMO.pop(session_key, None)
    return copy.deepcopy(record)


def _read_effort_level(config_path: Path | None) -> str | None:
    """Read claude_code.effort from config.yml."""
    if not config_path or not Path(config_path).exists():
        return None
    try:
        config = yaml.safe_load(Path(config_path).read_text()) or {}
        return config.get("claude_code", {}).get("effort")
    except Exception:
        return None


async def _run_output_loop(
    session,
    websocket: WebSocket,
    stop_event: asyncio.Event,
) -> None:
    """Forward PTY bytes to the WebSocket until stopped or process exits."""
    try:
        async for data in session.read_output():
            if stop_event.is_set():
                return
            await websocket.send_bytes(data)
    except Exception:
        pass
    finally:
        if not stop_event.is_set():
            code = session.exit_code
            try:
                await websocket.send_text(json.dumps({"type": "exit", "code": code}))
            except Exception:
                pass


async def _discover_and_notify(
    snapshot: set[str],
    discovery: SessionDiscovery,
    registry,
    current_key: str,
    websocket: WebSocket,
    timeout: float = 15.0,
) -> str | None:
    """Discover a newly-created Claude session UUID and notify the client.

    Returns the discovered UUID (or None). Also rekeys the registry entry,
    which renames the pooled session and records the audit alias back to the
    spawn key; the posture store is untouched, because every entry it holds is
    written under both keys — see
    :meth:`~osprey.interfaces.web_terminal.pty_manager.PtyRegistry.rekey_session`
    and :func:`persist_or_raise`.
    """
    loop = asyncio.get_event_loop()
    new_id = await loop.run_in_executor(None, discovery.discover_new_session, snapshot, timeout)
    if new_id:
        registry.rekey_session(current_key, new_id)
        try:
            await websocket.send_text(json.dumps({"type": "session_info", "session_id": new_id}))
        except Exception:
            pass
    return new_id


def _build_extra_env(
    websocket: WebSocket,
    claude_session_id: str | None,
    telemetry_session_id: str | None = None,
) -> dict[str, str]:
    """Build the extra environment dict for PTY sessions.

    ``telemetry_session_id`` is the session UUID this terminal's ``claude`` is
    forced onto (via ``--session-id``); it is handed to the workspace
    provenance_locator tool so a filed issue can point back to this session's
    telemetry. Kept separate from ``claude_session_id`` — which drives
    ``OSPREY_SESSION_ID`` (session-scoped agent-data relocation and artifact
    session tagging) and stays unset for new sessions — because the telemetry
    locator must not carry those side effects.

    The result also carries the **panel token**, and that is the one place the
    PTY child gets it. :func:`~osprey.interfaces.web_auth._populate` pops the
    token out of ``os.environ`` and
    :func:`~osprey.agent_runner.clean_env.build_base_child_env` strips it, so
    :func:`~osprey.interfaces.web_terminal.pty_manager.build_pty_env` — which
    hands its result to ``Popen(env=...)`` as the child's *complete*
    environment — would otherwise produce a child that holds no panel
    credential at all, leaving the MCP panel tools and the panel/approval hooks
    to send no bearer and be answered 401 in silence. ``extra_env`` is applied
    after the strip, which is what makes this the seam for a deliberate
    re-introduction. Only the panel token is re-introduced: it authorises the
    narrow panel tier (:data:`~osprey.interfaces.web_auth.PANEL_TIER_ROUTES`)
    and nothing else. The operator secret is never put back.
    """
    extra_env: dict[str, str] = {}
    # The PTY terminal IS the expert web surface — every session spawned here
    # serves it, whatever web.ui_mode the deployment defaults to (the operator
    # can flip modes live; the chat surface runs its own SDK sessions, marked
    # "simple" in operator_session.py). The panels-context SessionStart hook
    # reads this to tell the agent which UI the operator is looking at.
    extra_env["OSPREY_WEB_UX"] = "expert"
    if claude_session_id:
        extra_env["OSPREY_SESSION_ID"] = claude_session_id
    if telemetry_session_id:
        extra_env["OSPREY_TELEMETRY_SESSION_ID"] = telemetry_session_id
        extra_env["OSPREY_TELEMETRY_SESSION_START"] = datetime.now(UTC).isoformat()
    extra_env[PANEL_TOKEN_ENV] = get_web_credentials(websocket.app).panel_token
    hooks_env = getattr(websocket.app.state, "hooks_env", {})
    if hooks_env:
        extra_env.update(hooks_env)

    # The posture ANCHORS — never the posture itself. Keyed on the pool key:
    # ``terminal_ws`` computes ``current_key = claude_session_id or
    # telemetry_session_id`` and all three call sites reach here with that same
    # pair, so this expression is the pool key in every one of them (a
    # brand-new session, whose claude id is still None, included).
    #
    # **No execution mode is stamped here.** The session's write posture is
    # per target and lives in the store; every write-time gate reads it there,
    # which is what lets a narrowing land on a session already
    # mid-conversation. An ``OSPREY_EXECUTION_MODE=readonly`` stamped at spawn
    # could not express "the stand-in is read-only and the simulator is not" —
    # it sandboxes the whole session — and it could only be changed by killing
    # the child, which is the conversation this feature exists to keep. A
    # deployment-wide readonly marker still reaches the child, as it always
    # has, through ``hooks_env`` above or the inherited environment.
    #
    # What the child is handed is where to look and whose answer to read:
    # ``OSPREY_POSTURE_SESSION`` (the store key) and
    # :data:`~osprey.audit.posture.OSPREY_AGENT_DATA_ROOT` (the directory that
    # store — and the control-target state file beside it — lives in), plus
    # ``OSPREY_POSTURE_SOURCE`` for the audit envelope. The source is always
    # "live" here: a PTY pool key is exactly the id the posture route
    # addresses, so the store keeps answering for it after the child is up.
    #
    # The root is stamped as a PAIR with the key, on the same condition and in
    # the same block. Everything below this spawn re-derives that directory
    # today (the controls server from config, the stdlib-only hooks from a
    # repo-root guess and a literal ``var/agent_data``), and those derivations
    # part company as soon as a deployment moves ``agent_data.base_dir``.
    # Handing the child the root the server actually resolved makes one answer
    # authoritative for all of them; a child that held the key without it would
    # be told whose posture to read and left to guess where. Never one without
    # the other, and a test pins that.
    posture_key = claude_session_id or telemetry_session_id
    if posture_key:
        extra_env[POSTURE_SOURCE_ENV] = POSTURE_SOURCE_LIVE
        extra_env[POSTURE_SESSION_ENV] = posture_key
        extra_env[OSPREY_AGENT_DATA_ROOT] = resolve_agent_data_root(websocket.app)
    return extra_env


@router.websocket("/ws/terminal")
async def terminal_ws(websocket: WebSocket):
    """WebSocket bridge for terminal I/O with session pool support.

    Protocol:
    - Client -> Server text frames: raw terminal input (keystrokes)
    - Client -> Server JSON: {"type": "resize", "cols": N, "rows": N}
    - Client -> Server JSON: {"type": "switch_session", "session_id": UUID}
    - Server -> Client binary frames: raw PTY output
    - Server -> Client JSON: {"type": "exit", "code": N}
    - Server -> Client JSON: {"type": "session_switched", "session_id": UUID}
    - Server -> Client JSON: {"type": "session_info", "session_id": UUID}
    - Server -> Client JSON: {"type": "error", "message": str}
    """
    await websocket.accept()

    registry = websocket.app.state.pty_registry
    base_shell_command = websocket.app.state.shell_command
    discovery = SessionDiscovery(websocket.app.state.project_cwd)

    # Parse session params from query string
    req_session_id = websocket.query_params.get("session_id")
    mode = websocket.query_params.get("mode", "new")

    effort = _read_effort_level(websocket.app.state.config_path)

    # Build the command and determine the initial session key.
    # base_shell_command is list[str] (set by app.lifespan), so unpack with
    # [*base, ...] — nesting would break PtySession's exec (issue #218).
    if mode == "resume" and req_session_id:
        command: list[str] = [*base_shell_command, "--resume", req_session_id]
        claude_session_id: str | None = req_session_id
        telemetry_session_id: str = req_session_id
    else:
        # Force a known session UUID so the workspace provenance_locator tool can
        # hand it back (via OSPREY_TELEMETRY_SESSION_ID, injected below) and it
        # matches the value the OTEL emitter tags records with as session.id — a
        # filed issue's provenance pointer then resolves. (Not claude_session_id,
        # which would set OSPREY_SESSION_ID and relocate session-scoped agent
        # data — this is the CLI's session id, not an agent-data scope.)
        telemetry_session_id = str(uuid.uuid4())
        command = [*base_shell_command, "--session-id", telemetry_session_id]
        claude_session_id = None

    if effort:
        command.extend(["--effort", effort])

    # Pool key: the requested id for resumes, the forced id for new sessions.
    # A new session's id is dictated on the command line above, never guessed,
    # so the pool is keyed by the real session id from the first moment and
    # needs no later rekey.
    current_key = claude_session_id or telemetry_session_id

    # Wait for the client's initial resize message before spawning the PTY.
    initial_cols, initial_rows = 80, 24
    try:
        first = await asyncio.wait_for(websocket.receive(), timeout=5.0)
        if "text" in first:
            try:
                msg = json.loads(first["text"])
                if msg.get("type") == "resize":
                    initial_cols = msg["cols"]
                    initial_rows = msg["rows"]
            except (json.JSONDecodeError, KeyError):
                pass
    except TimeoutError:
        logger.warning("No initial resize from client within 5s, using defaults")

    # For resumes, snapshot the session files before spawning — a stale/absent
    # --resume-id can make the CLI silently start a fresh session instead of
    # resuming, and this is how we tell the two apart once the PTY is up.
    resume_snapshot: set[str] | None = None
    if mode == "resume" and req_session_id:
        resume_snapshot = discovery.snapshot_session_ids()

    extra_env = _build_extra_env(websocket, claude_session_id, telemetry_session_id)

    session, was_reused = registry.get_or_create_session(
        current_key,
        command,
        rows=initial_rows,
        cols=initial_cols,
        extra_env=extra_env if extra_env else None,
        cwd=websocket.app.state.project_cwd,
    )
    registry.attach_session(current_key)

    # New sessions: confirm the id immediately. It is the id this handler put
    # on the CLI's own command line a few lines up, so there is nothing to wait
    # for and nothing to race.
    if claude_session_id is None:
        try:
            await websocket.send_text(
                json.dumps({"type": "session_info", "session_id": current_key})
            )
        except Exception:
            pass

    # For resumes, confirm the actually-attached session id so the client can
    # tell a live resume from a silently-fresh PTY on a stale --resume-id.
    # Two cases are trusted immediately, with no discovery needed: a reused
    # warm session (same live PTY) and a cold resume whose session file was
    # already on disk before we spawned (the id was genuinely valid). Only a
    # request for an id with no file on disk — stale/absent — is ambiguous:
    # the CLI may fall back to creating a fresh session under an id of its own
    # choosing, and ``--resume`` gives this handler no way to dictate that id
    # the way the new-session path dictates one. So this branch alone still
    # discovers, and it keeps the generous window: it is racing the CLI's own
    # startup, and a fallback id that arrives late is still worth having. When
    # the window closes with nothing new, the requested id is confirmed rather
    # than left unanswered, so the client is never stranded without one.
    if resume_snapshot is not None:
        if was_reused or req_session_id in resume_snapshot:
            try:
                await websocket.send_text(
                    json.dumps({"type": "session_info", "session_id": current_key})
                )
            except Exception:
                pass
        else:

            async def _confirm_resume():
                nonlocal current_key
                found = await _discover_and_notify(
                    resume_snapshot, discovery, registry, current_key, websocket
                )
                if found:
                    current_key = found
                elif session.is_alive:
                    # Nothing new appeared and the PTY is still up, so the
                    # requested id resumed after all — confirm it. A PTY that
                    # has already exited says the opposite: ``--resume`` found
                    # no such conversation and the CLI quit. Confirming the id
                    # back in that case would tell the client to keep an id
                    # that resolves to nothing and re-resume it on the next
                    # reload, which is how a dead tab becomes a permanent one.
                    # Staying quiet leaves the client's own failover — driven
                    # by the ``exit`` frame it has already been sent — to
                    # discard the id and start clean.
                    try:
                        await websocket.send_text(
                            json.dumps({"type": "session_info", "session_id": current_key})
                        )
                    except Exception:
                        pass

            asyncio.create_task(_confirm_resume())

    # Start output forwarding
    stop_event = asyncio.Event()
    output_task = asyncio.create_task(_run_output_loop(session, websocket, stop_event))

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                text = message["text"]
                try:
                    msg = json.loads(text)
                except (json.JSONDecodeError, KeyError):
                    msg = None

                if isinstance(msg, dict):
                    msg_type = msg.get("type")

                    if msg_type == "resize":
                        logger.debug("PTY resize: %dx%d", msg["cols"], msg["rows"])
                        session.resize(msg["rows"], msg["cols"])
                        continue

                    if msg_type == "switch_session":
                        target_id = msg.get("session_id", "")
                        if not _UUID_RE.match(target_id):
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "error",
                                        "message": "Invalid session ID format",
                                    }
                                )
                            )
                            continue

                        if target_id == current_key:
                            # Already on this session — no-op
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "session_switched",
                                        "session_id": target_id,
                                    }
                                )
                            )
                            continue

                        try:
                            # 1. Stop current output loop
                            stop_event.set()
                            output_task.cancel()
                            try:
                                await output_task
                            except asyncio.CancelledError:
                                pass

                            # 2. Detach current session (stays alive in pool)
                            registry.detach_session(current_key)

                            # 3. Build command for target — unpack base_shell_command
                            #    (list[str]) so a pinned ["npx", "-y", "..."] prefix
                            #    flattens into target_cmd rather than nesting.
                            target_cmd: list[str] = [
                                *base_shell_command,
                                "--resume",
                                target_id,
                            ]
                            if effort:
                                target_cmd.extend(["--effort", effort])
                            target_env = _build_extra_env(websocket, target_id)

                            # 4. Get or create target session
                            session, was_reused = registry.get_or_create_session(
                                target_id,
                                target_cmd,
                                rows=initial_rows,
                                cols=initial_cols,
                                extra_env=target_env if target_env else None,
                                cwd=websocket.app.state.project_cwd,
                            )
                            registry.attach_session(target_id)

                            # 5. Notify client
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "session_switched",
                                        "session_id": target_id,
                                    }
                                )
                            )

                            # 6. Start new output loop
                            stop_event = asyncio.Event()
                            output_task = asyncio.create_task(
                                _run_output_loop(session, websocket, stop_event)
                            )

                            # 7. Update tracking
                            current_key = target_id

                            logger.info(
                                "Session switched to %s (reused=%s)",
                                target_id,
                                was_reused,
                            )
                        except Exception:
                            logger.exception("Session switch failed")
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "error",
                                        "message": "Session switch failed",
                                    }
                                )
                            )
                        continue

                # Not a recognized JSON control message — treat as terminal input
                session.write_input(text.encode("utf-8"))

            elif "bytes" in message:
                session.write_input(message["bytes"])

    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        stop_event.set()
        output_task.cancel()
        # Detach instead of terminate — keep session alive in the pool.
        # Only terminate if the process has already died.
        #
        # Both steps are guarded on this handler still OWNING the pool entry,
        # because the key alone no longer identifies it. Now that every new
        # session hands the client an id it stores and resumes, two handlers
        # meeting on one key is ordinary: a second tab (or a reload whose
        # disconnect the server sees late) can resume the id, find this PTY
        # dead, and spawn a replacement under the same key. An unguarded
        # teardown would then terminate the live replacement and clear the
        # attachment the newer handler holds, killing a terminal the operator
        # is looking at.
        if registry.get_session(current_key) is session:
            registry.detach_session(current_key)
        if not session.is_alive:
            registry.terminate_session_if_owner(current_key, session)


# ── Control-target gestures: audit, vocabulary, refusals ─────────────────────
#
# Both control POSTs below are *gestures an operator made*, not incidental
# state changes, so each files exactly one audit record naming the session it
# governs. ``HttpAuditMiddleware`` would otherwise file a bare
# ``http_mutation`` line with ``session: null`` — enough to know a request came
# in, useless for joining a toggle to the tool calls it governed. The dedup
# marker is what keeps the two from both writing: the innermost recorder owns
# the decision, and the middleware defers to it.

#: Audit subject for a gesture that moves the session's control target. The
#: same word the agent's own tool records under
#: (``osprey.mcp_server.http.TARGET_SWITCH_TOOL``), so an operator reading the
#: ledger sees one kind of event whichever surface asked for it.
AUDIT_SUBJECT_TARGET_SET = "control_target_set"

#: Audit subject for a gesture that narrows or widens one target's posture.
AUDIT_SUBJECT_POSTURE_SET = "session_posture_set"


def _record_control_gesture(
    app: Any,
    session_key: str,
    *,
    subject: str,
    decision: str,
    reason: str,
    detail: str | None = None,
) -> None:
    """File one ledger record for a control gesture, and claim the decision.

    Through :func:`~osprey.audit.dedup.record_and_mark` rather than the writer
    directly, because this recorder runs *inside* ``HttpAuditMiddleware``: the
    marker tells that outer layer a specific answer was already filed, so one
    POST leaves one line rather than two — and, on a refusal, the right one.

    **Must be called on the task the middleware is awaiting.** The marker is a
    ``ContextVar``, and a mark set behind ``run_in_threadpool`` or
    ``create_task`` is invisible to the layer outside; the middleware would
    then file ``allowed`` on top of a refusal. That is why both routes are
    ``async def`` and call this inline.

    ``session`` is the SPAWN key (:func:`_spawn_posture_key`): the running
    child cannot have its ``OSPREY_POSTURE_SESSION`` rewritten without being
    killed, so every record that child emits carries the key it was spawned
    with. A gesture filed under a rekeyed session's *current* key would split
    one session into two unrelated actors.

    Never raises. A gesture whose record could not be written is still a
    gesture that happened; the trail degrades, the operation does not.
    """
    try:
        from osprey.audit.dedup import record_and_mark
        from osprey.audit.envelope import POSTURE_SOURCE_APP

        record_and_mark(
            decision=decision,
            reason=reason,
            surface=HTTP_MUTATION_SURFACE,
            posture=HTTP_MUTATION_POSTURE,
            posture_source=POSTURE_SOURCE_APP,
            session=_spawn_posture_key(app, session_key),
            subject=subject,
            detail=detail,
        )
    except Exception:  # noqa: BLE001 — the audit trail degrades; the gesture does not
        logger.warning("Could not record the %s gesture for audit", subject, exc_info=True)


def _refuse_gesture(
    app: Any,
    session_key: str,
    *,
    subject: str,
    status_code: int,
    error: str,
    message: str,
    detail: str | None = None,
) -> HTTPException:
    """Record a refusal, then hand back the exception the caller raises.

    Returned rather than raised so the call site reads ``raise
    _refuse_gesture(...)`` and a reader can see the control flow leave at that
    line. The record is filed first: a refusal the operator sees and the ledger
    does not is exactly the gap these routes exist to close.
    """
    from osprey.audit.envelope import DECISION_REFUSED

    _record_control_gesture(
        app,
        session_key,
        subject=subject,
        decision=DECISION_REFUSED,
        reason=error,
        detail=detail,
    )
    return HTTPException(status_code=status_code, detail={"error": error, "message": message})


def _unknown_target_message(configured: tuple[str, ...]) -> str:
    """The 400's sentence when a gesture names a target this render does not have.

    Both control-gesture routes refuse that with the same words on purpose: an
    operator who mistypes a target name — or a client built against another
    deployment — reads one sentence whichever gesture they made, and it names
    the vocabulary that WOULD have worked. Spelled once because it is pinned by
    tests on both routes, and two copies of a pinned sentence are two chances
    for one of them to drift.
    """
    return (
        "This deployment configures no control target by that name. "
        f"It has: {', '.join(configured) or 'none'}."
    )


def _configured_target_names(section: Any) -> tuple[str, ...]:
    """The control targets this render describes, or ``()``.

    :func:`~osprey_connectors.types.configured_targets` and nothing else — the
    same list the roster, the endpoint prober and the popover enumerate, so a
    name these routes accept is a name some row exists for. Never
    :data:`~osprey_connectors.types.CONTROL_TARGETS`, which is the vocabulary
    of machines that *can* exist: accepting a request for a target the
    deployment never configured would hand the reconciler a switch nobody could
    have meant.

    An unreadable render answers ``()``, which refuses every target. That is
    the same direction every other predicate here takes on an unreadable
    config, and the honest one: a server that cannot read its own render does
    not know which machines exist.
    """
    if section is _UNREADABLE_SECTION:
        return ()
    try:
        from osprey_connectors.types import configured_targets

        return tuple(configured_targets(section))
    except Exception:  # noqa: BLE001 — an unreadable render configures no targets
        logger.warning("Could not read the configured control targets")
        return ()


@dataclass(frozen=True)
class _TargetRequestFacts:
    """Everything ``POST /api/terminal/target`` needs off the event loop.

    Gathered in ONE worker-thread hop: the render parse, the process-table walk
    behind :func:`_session_record`, and two reads of the state directory. Split
    across several hops they would straddle each other — a record resolved
    before a switch landed, beside a request file read after it.
    """

    #: Target names this render configures; the vocabulary the 400 is keyed on.
    #: Spelled the same as ``_PostureRequestFacts.configured``: it is the same
    #: fact from the same function, and the two routes refuse an unknown target
    #: with the same sentence.
    configured: tuple[str, ...]
    #: The state record this session's controls server published, or ``None``.
    record: dict[str, Any] | None
    #: Whether the state directory resolves at all. ``False`` is the 503.
    store_available: bool
    #: The request already addressed to that server, fresh or stale, or ``None``.
    pending: dict[str, Any] | None
    #: The pid of the controls server the record names, or ``None`` when there
    #: is no record or its ``server_pid`` is not a number. Carried rather than
    #: re-derived: the reader that decides the 409, the writer that addresses
    #: the request file and the ledger line must all name one pid.
    server_pid: int | None


def _target_request_facts(app: Any, session_key: str, config_path: Path | None):
    """Read the render, the session's record and any pending request. BLOCKING.

    Called through ``run_in_threadpool``: :func:`_session_record` walks the
    process table (forking ``ps`` where there is no ``/proc``) and the rest is
    file I/O. None of it may happen on the event loop.
    """
    from osprey.mcp_server.control_system import target_state

    targets = _configured_target_names(_control_system_section(config_path))

    # The directory is resolved FIRST because everything below reads it: the
    # record lives in it, and so does any pending request. A root that does not
    # resolve is therefore not "this session has not started" — it is a server
    # that cannot look, which is the 503 and not the 409.
    try:
        target_state.state_dir()
    except Exception:  # noqa: BLE001 — an unresolvable root is the 503, not a crash
        logger.warning("The control-target state directory does not resolve", exc_info=True)
        return _TargetRequestFacts(
            configured=targets,
            record=None,
            store_available=False,
            pending=None,
            server_pid=None,
        )

    record = _session_record(app, session_key)
    server_pid = _pid_or_none((record or {}).get("server_pid"))
    pending = target_state.read_request(server_pid) if server_pid is not None else None

    return _TargetRequestFacts(
        configured=targets,
        record=record,
        store_available=True,
        pending=pending,
        server_pid=server_pid,
    )


def _target_refusal(
    app: Any, session_id: str, body: TargetRequest, facts: _TargetRequestFacts
) -> HTTPException | None:
    """The switch ladder's refusals, in order — or ``None`` to write the request.

    Split out of :func:`request_terminal_target` the way
    :func:`_posture_refusal` is split out of its own route, so the ladder
    reads as the ordered list of reasons it is and the handler as what it
    does once nothing refuses: mint an id, write the request, record, answer.

    Only the rungs decided BEFORE the write live here. The second
    ``request_pending`` — the one that answers a ``RequestSuperseded`` — is
    not a rung: it is what the write itself came back with, and moving it
    here would mean refusing before finding out.

    The order differs from the posture route's and is load-bearing in its
    own right: 400 for a target this render does not configure, then the 503
    ahead of the 409, because the state directory is where a record would
    have been found and "cannot look" is not "never started".
    """
    subject = AUDIT_SUBJECT_TARGET_SET

    if body.target not in facts.configured:
        return _refuse_gesture(
            app,
            session_id,
            subject=subject,
            status_code=400,
            error="unknown_target",
            message=_unknown_target_message(facts.configured),
            detail=f"target={body.target}",
        )

    # 503 ahead of the 409, because the state directory is where a record would
    # have been found: a root that does not resolve cannot tell "this session
    # never started a controls server" from "this server cannot look".
    if not facts.store_available:
        return _refuse_gesture(
            app,
            session_id,
            subject=subject,
            status_code=503,
            error="store_unavailable",
            message=(
                "This deployment's agent-data root does not resolve, so there is "
                "nowhere to write a switch request the controls server would read. "
                "Nothing was requested."
            ),
            detail=f"target={body.target}",
        )

    if facts.record is None or facts.server_pid is None:
        return _refuse_gesture(
            app,
            session_id,
            subject=subject,
            status_code=409,
            error="session_not_started",
            message=(
                "This session has no running control-system server yet, so there is "
                "nothing to ask for a switch. Send one prompt first, then switch."
            ),
            detail=f"target={body.target}",
        )

    if facts.pending is not None:
        from osprey.mcp_server.control_system import target_state

        if target_state.is_request_fresh(facts.pending):
            return _refuse_gesture(
                app,
                session_id,
                subject=subject,
                status_code=409,
                error="request_pending",
                message=(
                    "A switch was already requested for this session and has not "
                    "been answered yet. Wait for its outcome, then switch again."
                ),
                detail=f"target={body.target} pending={facts.pending.get('request_id')}",
            )

    return None


@router.post("/api/terminal/target", status_code=202)
async def request_terminal_target(body: TargetRequest, request: Request):
    """Ask this session's controls server to switch control target.

    **This route does not switch anything.** The connector is owned by the
    controls MCP server — a stdio child of the Claude process inside the PTY,
    with no inbound channel and the sole right to write the target state file.
    So the web server writes *desired state*: one request file, named for and
    addressed to that server's pid, which the reconciler inside it consumes,
    gates and answers by publishing ``last_switch`` back into the state file.
    The operator's chip watches for that outcome; this route's job ends at
    ``202 Accepted``.

    **No availability pre-check, on purpose.** Eligibility, reachability and
    "already active" are re-evaluated by ``switch_gate`` immediately before the
    switch, inside the process that holds the connector. Answering them here
    would answer from a snapshot taken a moment earlier — and a route that
    disagrees with the gate that actually decides is worse than one that does
    not try.

    The ladder, in order:

    * **400** — an id outside the closed key grammar, or a target this render
      does not configure. Both are identifiers, checked before anything is read
      or written.
    * **409** ``session_not_started`` — no controls server has published a
      record this session's process tree owns. There is nothing to address: the
      request file's whole addressing scheme is that pid.
    * **503** ``store_unavailable`` / ``store_write_failed`` — the state
      directory does not resolve, or the write failed. The gesture did not
      happen and the operator is told so, rather than being shown a request id
      nothing will ever read.
    * **409** ``request_pending`` — a fresh request is already addressed to
      that server. One outstanding gesture at a time; the pending file is left
      exactly as it is, because overwriting it would strand the operator who is
      still watching for the first one's outcome.
    """
    session_id = body.session_id
    _require_session_uuid(session_id)

    app = request.app
    subject = AUDIT_SUBJECT_TARGET_SET
    facts: _TargetRequestFacts = await run_in_threadpool(
        _target_request_facts, app, session_id, app.state.config_path
    )

    refusal = _target_refusal(app, session_id, body, facts)
    if refusal is not None:
        raise refusal

    from osprey.mcp_server.control_system import target_state
    from osprey.utils.identity import acting_identity

    request_id = str(uuid.uuid4())
    payload = {
        "request_id": request_id,
        "target": body.target,
        "server_pid": facts.server_pid,
        "created_at": datetime.now(UTC).isoformat(),
        "requested_by": acting_identity(),
    }
    try:
        await run_in_threadpool(target_state.write_request, payload)
    except target_state.RequestSuperseded as exc:
        # Two operators clicked Switch in the same moment. The freshness read
        # above happens before this write, with an ``await`` between them, so
        # both passed it; the writer that did not end up in the slot is told
        # what that read would have told it a moment later.
        raise _refuse_gesture(
            app,
            session_id,
            subject=subject,
            status_code=409,
            error="request_pending",
            message=(
                "A switch was requested for this session a moment ago and has not "
                "been answered yet. Wait for its outcome, then switch again."
            ),
            detail=f"target={body.target} superseded={request_id}",
        ) from exc
    except Exception as exc:  # noqa: BLE001 — reported to the operator as a 503
        logger.warning(
            "Could not write the switch request for session %s; nothing was requested",
            session_id,
            exc_info=True,
        )
        raise _refuse_gesture(
            app,
            session_id,
            subject=subject,
            status_code=503,
            error="store_write_failed",
            message=(
                "The switch request could not be written, so nothing was requested. "
                "Check the server's write access to the agent-data root and try again."
            ),
            detail=f"target={body.target}",
        ) from exc

    from osprey.audit.envelope import DECISION_ALLOWED

    _record_control_gesture(
        app,
        session_id,
        subject=subject,
        decision=DECISION_ALLOWED,
        reason="target_switch_requested",
        detail=f"target={body.target} request_id={request_id} server_pid={facts.server_pid}",
    )
    logger.info(
        "Session %s requested control target %s (request %s addressed to pid %s)",
        session_id,
        body.target,
        request_id,
        facts.server_pid,
    )
    return {"session_id": session_id, "target": body.target, "request_id": request_id}


#: The popover's ``[ Sandbox everything ]`` gesture, spelled in the ``target``
#: field. Narrowing everything is one unambiguous act; there is deliberately no
#: matching "arm everything", because each target's ceiling is its own.
ALL_TARGETS = "all"


@dataclass(frozen=True)
class _PostureRequestFacts:
    """Everything ``POST /api/terminal/posture`` needs off the event loop.

    One worker-thread hop for the whole ladder: the render parse, the
    store-path resolution, the hypothetical narrowing derivation and — only
    when the request widens — the execution-marker sweep. Answers drawn from
    separate hops could straddle each other: a ceiling read from one render
    beside a narrowing verdict taken from the next.
    """

    #: Target names this render configures — the 400's vocabulary.
    configured: tuple[str, ...]
    #: The targets this request actually touches (``all`` expands here).
    wanted: tuple[str, ...]
    #: ``session_posture(section)``: the persona ceiling, per target.
    ceilings: dict[str, bool]
    #: The ``writes_enabled`` config key each wanted target's ceiling came from.
    writes_keys: dict[str, str]
    #: Whether the store has a location at all. ``False`` is the 503.
    store_available: bool
    #: ``{target: why}`` for every target this request would CHANGE that cannot
    #: be narrowed. Keyed per target rather than reduced to a first refusal,
    #: because ``all`` narrows around one and a single target refuses on it.
    narrowing_refusals: dict[str, str]
    #: The first live execution marker, when the request widens; else ``None``.
    in_flight: dict[str, Any] | None
    #: The narrowings already recorded for this session, read in the same hop.
    current: dict[str, str]


def _session_ceilings(section: Any) -> dict[str, bool]:
    """The persona ceiling per target: ``session_posture(section)``.

    The per-target map, never the union. The union is true as soon as ONE
    target is armed, so on a mixed render it would offer a writes toggle on the
    facility's own machine that every write through it then refuses. An
    unreadable render arms nothing, which is where every other predicate here
    lands too.
    """
    if section is _UNREADABLE_SECTION:
        return {}
    try:
        from osprey_connectors.types import session_posture

        return dict(session_posture(section))
    except Exception:  # noqa: BLE001 — an unreadable render arms no target
        logger.warning("Could not read the per-target write ceilings")
        return {}


def _target_writes_key(section: Any, target: str) -> str:
    """The config key that decided *target*'s ceiling, for the 403 to name.

    :func:`~osprey_connectors.types.target_writes_enabled_key`, so the key a
    refusal names is the key that answered it — the operator's next action is
    to go and look at that line.
    """
    from osprey_connectors.types import WRITES_ENABLED_KEY

    if section is _UNREADABLE_SECTION:
        return WRITES_ENABLED_KEY
    try:
        from osprey_connectors.types import target_writes_enabled_key

        return str(target_writes_enabled_key(section, target))
    except Exception:  # noqa: BLE001 — name the deployment-wide key rather than none
        logger.warning("Could not derive the writes key for control target %s", target)
        return WRITES_ENABLED_KEY


def _narrowing_refusals(config: Any, targets: tuple[str, ...]) -> dict[str, str]:
    """Which of *targets* cannot be narrowed, and why. ``{target: detail}``.

    Delegated to
    :func:`~osprey.mcp_server.control_system.target_eligibility.narrowing_refusal`,
    the same hypothetical derivation the roster and the tool consult, so the
    popover's locked toggle and this route's refusal carry one sentence.

    **Only pass targets the request would actually CHANGE.** That function is
    store-blind — it answers "what would narrowing this target cost?" from the
    config alone — so a target already sitting in ``sandbox`` answers the same
    refusal it answered when it was narrowed. Asking about one would let a
    target that is *already* read-only veto a gesture that does not touch it.

    Fails **open** on an unreadable render: the 400 above has already refused
    every target there, and a narrowing on a config nobody can read is the safe
    direction to let through.
    """
    if config is _UNREADABLE_SECTION:
        return {}
    refusals: dict[str, str] = {}
    try:
        from osprey.mcp_server.control_system.target_eligibility import narrowing_refusal

        for target in targets:
            verdict = narrowing_refusal(config, target)
            if verdict is not None:
                refusals[target] = verdict.detail
    except Exception:  # noqa: BLE001 — an underivable narrowing is not a refusal
        logger.warning("Could not evaluate what narrowing would cost", exc_info=True)
    return refusals


def _first_live_execution() -> dict[str, Any] | None:
    """The first live execution marker, or ``None``. Never raises."""
    try:
        from osprey.mcp_server.control_system.target_state import in_flight_executions

        running = in_flight_executions()
    except Exception:  # noqa: BLE001 — an unreadable marker directory reports none
        logger.warning("Could not read the execution markers", exc_info=True)
        return None
    return running[0] if running else None


def _posture_post_facts(
    app: Any, session_key: str, config_path: Path | None, target: str, posture: str
) -> _PostureRequestFacts:
    """Read everything the posture ladder decides on. BLOCKING.

    Called through ``run_in_threadpool``: it parses ``config.yml`` and globs
    the state directory. None of that may happen on the event loop, where one
    slow shared volume would stall every request this server is serving, the
    terminal websocket included.
    """
    config = _rendered_config(config_path)
    section = _section_of(config)
    configured = _configured_target_names(section)
    wanted = configured if target == ALL_TARGETS else (target,)
    widening = posture == POSTURE_WRITES

    # The narrowing question is asked only about targets this request MOVES. A
    # target already in ``sandbox`` is not being narrowed by this gesture, and
    # ``narrowing_refusal`` — which reads the config, never the store — would
    # otherwise let it refuse on behalf of a change nobody requested.
    current = dict(_posture_entry(app, session_key))
    changing = tuple(t for t in wanted if current.get(t) != POSTURE_SANDBOX)

    return _PostureRequestFacts(
        configured=configured,
        wanted=wanted,
        ceilings=_session_ceilings(section),
        writes_keys={t: _target_writes_key(section, t) for t in wanted},
        store_available=_posture_store_path() is not None,
        narrowing_refusals={} if widening else _narrowing_refusals(config, changing),
        in_flight=_first_live_execution() if widening else None,
        current=current,
    )


def _in_flight_message(record: dict[str, Any]) -> str:
    """The refusal sentence the switch tool already says for a running run.

    Borrowed rather than reworded: an operator who has read it once in the
    agent's answer should read the same words in the popover.

    **The attribution clause is dropped here.** The tool's suggestions open by
    saying whose run it is, decided by comparing the marker's ``owner_ppid`` to
    ``os.getppid()`` — true in the agent's own process tree, meaningless in the
    web server's, where it would always read "another session sharing this
    deployment". Telling the operator whose own session is running the
    execution that it belongs to somebody else is worse than not saying, so
    this surface keeps only the sentences that hold wherever they are read:
    what is in flight, on which target, and what to do about it.
    """
    try:
        from osprey.mcp_server.control_system.tools.control_target import _in_flight_detail

        message, suggestions, _details = _in_flight_detail(record, "")
        remedy = [suggestions[-1]] if suggestions else []
        return " ".join([message[:1].upper() + message[1:], *remedy])
    except Exception:  # noqa: BLE001 — the refusal stands even without the tool's words
        logger.warning("Could not build the in-flight refusal message", exc_info=True)
        running_on = str(record.get("target") or "unknown")
        return (
            f"An execution in flight on target {running_on!r}; wait or stop it, then widen again."
        )


def _posture_refusal(
    app: Any, session_id: str, body: PostureRequest, facts: _PostureRequestFacts, gesture: str
) -> HTTPException | None:
    """The posture ladder's refusals, in order — or ``None`` to let the gesture through.

    Split out of :func:`set_terminal_posture` so the ladder reads as the
    ordered list of reasons it is, and the handler as the steps it performs
    around it: refuse, apply, persist, record, answer. Everything here is
    decided from :class:`_PostureRequestFacts`; the one refusal that fires
    BEFORE that threadpool hop stays in the handler, where it can be reached
    without paying for the facts.

    The order is load-bearing and is documented on the route: 400 before 503
    before 403, because each rung answers a question the next one assumes.

    Returns the exception rather than raising it, for the same reason
    :func:`_refuse_gesture` does: the audit record is filed here, and the
    control flow leaves at the caller's ``raise``.
    """
    subject = AUDIT_SUBJECT_POSTURE_SET
    widening = body.posture == POSTURE_WRITES

    if not facts.wanted or (body.target != ALL_TARGETS and body.target not in facts.configured):
        return _refuse_gesture(
            app,
            session_id,
            subject=subject,
            status_code=400,
            error="unknown_target",
            message=_unknown_target_message(facts.configured),
            detail=gesture,
        )

    if not facts.store_available:
        return _refuse_gesture(
            app,
            session_id,
            subject=subject,
            status_code=503,
            error="store_unavailable",
            message=(
                "This deployment's agent-data root does not resolve, so there is "
                "nowhere to record a posture the agent would read back. No posture "
                "was changed."
            ),
            detail=gesture,
        )

    if widening:
        unarmed = [t for t in facts.wanted if not facts.ceilings.get(t, False)]
        if unarmed:
            blocked = unarmed[0]
            return _refuse_gesture(
                app,
                session_id,
                subject=subject,
                status_code=403,
                error="writes_disabled",
                message=(
                    f"This deployment does not arm writes for target {blocked!r}: "
                    f"{facts.writes_keys.get(blocked, '')} is off. A session posture "
                    "narrows what the render permits and never widens it."
                ),
                detail=gesture,
            )

    # A named target that cannot go read-only is a refusal: the operator asked
    # for exactly that one and must be told it would strand the target.
    # ``all`` is a different gesture — "narrow whatever can be narrowed" — so a
    # single unnarrowable target must not veto the rest. Refusing wholesale
    # there would leave "Sandbox everything" doing nothing at all on a
    # deployment with one write_access-only target, which is the opposite of
    # what the operator reached for.
    if body.target != ALL_TARGETS and facts.narrowing_refusals:
        blocked, why = next(iter(facts.narrowing_refusals.items()))
        return _refuse_gesture(
            app,
            session_id,
            subject=subject,
            status_code=409,
            error="selected_role_missing",
            message=why,
            detail=f"{gesture} blocked_by={blocked}",
        )

    if facts.in_flight is not None:
        return _refuse_gesture(
            app,
            session_id,
            subject=subject,
            status_code=409,
            error="execution_in_flight",
            message=_in_flight_message(facts.in_flight),
            detail=f"{gesture} executor_pid={facts.in_flight.get('pid')}",
        )

    return None


@router.post("/api/terminal/posture")
async def set_terminal_posture(body: PostureRequest, request: Request):
    """Narrow or widen one control target's write posture for one session.

    **Nothing is respawned.** The posture is read live from the store by every
    write-time gate — the connector's reference monitor, the executor's clamp,
    the write hook — so the running agent obeys a narrowing on its very next
    write. The route that used to terminate the child to re-stamp an
    environment variable would now be throwing away a conversation for a
    toggle.

    **The persist is the commit point** (:func:`persist_or_raise`): the file
    lands first and memory follows only once it has. Enforcement reads the
    store, so memory and disk disagreeing is a session the popover shows as
    narrowed whose next write is still permitted — or the reverse.

    **Any well-formed session id is accepted, spoken-to or not.** The posture
    must never depend on whether the operator has talked to the agent: both
    spawn paths read the store at spawn (``_acquire_chat_turn`` for a chat,
    ``build_operator_child_env`` for a PTY), so a narrowing recorded before
    the first prompt binds that session's very first write. An entry under a
    key nothing ever spawns is inert — the store only narrows, so the worst a
    stale or mistyped key can do is restrict a session that does not exist —
    and the ``operator-`` filter in :func:`_load_postures` is the hygiene
    boundary for keys that cannot come back. An earlier gate here refused
    fresh sessions with "send one prompt first"; it guarded nothing.

    The ladder, in order:

    * **400** — an id outside the closed key grammar; a target this render does
      not configure; or :data:`ALL_TARGETS` with ``writes``.
    * **503** — the store has no location, or the write failed. The toggle was
      refused and nothing changed.
    * **403** ``writes_disabled`` — ``writes`` on a target this render does not
      arm, naming that target's OWN ``writes_enabled`` key. Per target, never
      the union: a deployment that arms only its simulator must not offer a
      writes toggle on the facility's machine.
    * **409** ``selected_role_missing`` — narrowing a NAMED target would move
      the selected gateway role to one this deployment does not configure,
      leaving the target unusable. The operator is owed that sentence *before*
      they act. :data:`ALL_TARGETS` does not refuse on it: that gesture means
      "narrow whatever can be narrowed", so it narrows the rest and reports the
      others in ``skipped``. Only targets the request would actually MOVE are
      asked — a target already in ``sandbox`` must not veto a gesture that does
      not touch it.
    * **409** ``execution_in_flight`` — ``writes`` while any execution marker is
      live. A run is pinned to the posture it launched under, so a widening
      cannot reach it; refusing here turns a silent no-op into an answer.
    """
    session_id = body.session_id
    _require_session_uuid(session_id)

    app = request.app
    subject = AUDIT_SUBJECT_POSTURE_SET
    widening = body.posture == POSTURE_WRITES
    gesture = f"target={body.target} posture={body.posture}"

    if widening and body.target == ALL_TARGETS:
        raise _refuse_gesture(
            app,
            session_id,
            subject=subject,
            status_code=400,
            error="writes_requires_one_target",
            message=(
                "Writes are armed one target at a time. Every target's ceiling is "
                "its own, so name the target to widen."
            ),
            detail=gesture,
        )

    facts: _PostureRequestFacts = await run_in_threadpool(
        _posture_post_facts, app, session_id, app.state.config_path, body.target, body.posture
    )

    refusal = _posture_refusal(app, session_id, body, facts, gesture)
    if refusal is not None:
        raise refusal

    # Everything ``all`` could not narrow is reported rather than silently
    # dropped: the popover has to be able to say which machine stayed writable
    # and why, or "Sandbox everything" would be a claim the store does not back.
    skipped = [
        {"target": target, "reason": "selected_role_missing", "detail": why}
        for target, why in sorted(facts.narrowing_refusals.items())
    ]
    applying = [t for t in facts.wanted if t not in facts.narrowing_refusals]

    entry = dict(facts.current)
    for target in applying:
        if widening:
            entry.pop(target, None)
        else:
            entry[target] = POSTURE_SANDBOX

    try:
        stored = persist_or_raise(app, session_id, entry)
    except PostureStoreUnavailable as exc:
        raise _refuse_gesture(
            app,
            session_id,
            subject=subject,
            status_code=503,
            error=exc.error,
            message=exc.message,
            detail=gesture,
        ) from exc

    from osprey.audit.envelope import DECISION_ALLOWED

    _record_control_gesture(
        app,
        session_id,
        subject=subject,
        decision=DECISION_ALLOWED,
        reason="posture_set",
        detail=(
            f"{gesture} narrowed={','.join(sorted(stored)) or 'none'}"
            f" skipped={','.join(row['target'] for row in skipped) or 'none'}"
        ),
    )
    logger.info(
        "Session %s set posture %s on %s; narrowed targets: %s; skipped: %s",
        session_id,
        body.posture,
        body.target,
        ", ".join(sorted(stored)) or "none",
        ", ".join(row["target"] for row in skipped) or "none",
    )

    return {
        "session_id": session_id,
        "target": body.target,
        "posture": body.posture,
        "entry": stored,
        "skipped": skipped,
    }


# ── The control-target roster the header chip renders ────────────────────────
#
# ``GET /api/terminal/posture`` answers one question in many columns: if the
# agent writes now, where does it land and will it be refused? The chip shows
# that answer for the target the session is on; its popover shows one row per
# configured target, and every row carries enough for the operator to act on it
# — the machine's name, where it points, whether anything is reaching it, the
# ceiling the persona rendered, this session's own narrowing, and what a switch
# would do.
#
# Every one of those facts is derived HERE and not in the browser, for the same
# three reasons each time. The words a refusal uses are the switch tool's and
# must not be re-invented in JavaScript, or the popover and the agent would
# disagree about the same machine. ``age_s`` and ``stale`` are clock questions,
# and the browser's clock is not the one the stamps were written on. And the
# collapse from "two gateway roles were probed" to "this row is reachable" is a
# policy decision that has to match the role the connector will actually select.

#: Short chip labels, by what a row IS rather than by its target name. The name
#: is the key the rest of the system uses (``live``, ``va``, ``standin``); what
#: an operator has to read off the chip is the consequence, and on a stand-in
#: deployment the name and the consequence disagree by design.
SHORT_LIVE = "LIVE"
SHORT_STANDIN = "STAND-IN"
SHORT_VIRTUAL = "VIRTUAL"
SHORT_SIMULATED = "SIMULATED"

#: The same four in plain language, for the popover's simple density: one word
#: per row, chosen so an operator who has never met the word "target" can still
#: tell the facility's machine from a simulation of it.
KIND_LIVE = "live machine"
KIND_STANDIN = "stand-in"
KIND_VIRTUAL = "virtual accelerator"
KIND_SIMULATED = "simulated"

#: The two reachability states the prober never publishes, because both are
#: read-time verdicts: ``unknown`` is the absence of a row (no sweep has landed,
#: or this session owns no record at all) and ``stale`` is a row whose
#: ``probed_at`` has aged past the prober's own interval. The published three —
#: ``reached``, ``down``, ``not_applicable`` — pass through as measured.
REACH_UNKNOWN = "unknown"
REACH_STALE = "stale"

#: ``available_now`` reason for a chat session's rows. A chat has no PTY and so
#: no controls server of its own to address a switch request to; no row it
#: renders can offer one. Its toggles are untouched — the posture store is keyed
#: on the session, not on the topology, and a chat's writes meet the same
#: ceiling and the same narrowing a terminal's do.
REASON_CHAT_SESSION = "chat_session"

#: ``enforceable_reason`` for the one case that is not enforceable: a PTY
#: session that has started and still resolves no state record of its own. Its
#: controls server is outside this process tree — another session's, or none —
#: so a narrowing recorded here would be read by nobody, and the popover says so
#: instead of offering toggles that govern nothing.
ENFORCEABLE_REASON_NO_RECORD = "no_session_record"


def _age_seconds(stamp: Any) -> float | None:
    """Seconds since the wall-clock ISO-8601 *stamp*, or ``None``.

    Computed on the server for every published timestamp this route returns.
    The stamps are written by another process on its own schedule and the one
    thing a reader needs from them is how long ago; a browser subtracting them
    from its own clock would report the skew between two machines as the age of
    a probe, and call a live gateway stale on the strength of it.

    Never negative — a stamp from a clock running slightly ahead is reported as
    ``0.0``, which is what a reader has a rendering for. Anything unparseable
    answers ``None``, "not aged", which every consumer treats as absence rather
    than as freshness.
    """
    if not isinstance(stamp, str) or not stamp:
        return None
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return max(0.0, round((datetime.now(UTC) - moment).total_seconds(), 1))


def _aged(block: Any) -> dict[str, Any] | None:
    """A published block with ``age_s`` added, from its own ``at`` stamp.

    For ``last_switch`` above all: the chip shows a switch's outcome while it is
    news and stops when it becomes history, so the age is the whole of what
    decides which of the two an operator is looking at.

    The block is passed through whole and nothing is dropped: the chip matches
    an outcome by ``request_id`` and the popover puts it on the row named by
    ``target``, and both keys are the publisher's to choose, not this route's.
    """
    if not isinstance(block, dict):
        return None
    return {**block, "age_s": _age_seconds(block.get("at"))}


def _short_label_and_kind(label: Any, real_machine: bool) -> tuple[str, str]:
    """The chip's short word and the popover's plain word for one row.

    Derived from the row's ``real_machine`` flag and the SHAPE of the label the
    controls server minted — never from the target name. ``live`` is a name in a
    config file; whether writing there moves hardware is a property of the
    connector behind it, and reading the name would put ``LIVE`` on a simulator.

    ``real_machine`` decides the loud half — it is true for the facility's own
    machine AND for a stand-in, both of which get every strict limit and
    approval prompt hardware gets — and the label's shape then separates those
    two. An unrecognised label falls back to its half's louder answer, which is
    the direction this stack must fail in.
    """
    text = str(label or "").strip().lower()
    if real_machine:
        return (SHORT_STANDIN, KIND_STANDIN) if "(stand-in)" in text else (SHORT_LIVE, KIND_LIVE)
    if text.startswith("virtual accelerator"):
        return SHORT_VIRTUAL, KIND_VIRTUAL
    return SHORT_SIMULATED, KIND_SIMULATED


def _probe_staleness_threshold_s(config: Any) -> float | None:
    """How old a reachability row may be before it reads ``stale``.

    The prober's own interval times its own
    :data:`~osprey.mcp_server.control_system.endpoint_prober.STALENESS_INTERVALS`,
    read from the config key the prober reads, so the moment this route calls a
    row stale is the moment the prober would have replaced it three sweeps ago.
    A threshold invented here — a round number, say — would call a row on a
    slowly-probed deployment stale while the prober still considered it current.

    ``None`` when the interval cannot be derived, and that is not the same as a
    very large threshold: with no interval nothing is aged out at all, so a row
    reports what was measured rather than a staleness verdict this route could
    not justify.
    """
    try:
        from osprey.mcp_server.control_system.endpoint_prober import (
            STALENESS_INTERVALS,
            _configured_interval,
        )

        interval = _configured_interval(config if isinstance(config, dict) else {})
        return float(interval) * STALENESS_INTERVALS
    except Exception:  # noqa: BLE001 — an underivable interval ages nothing out
        logger.warning("Could not derive the endpoint prober's staleness threshold")
        return None


def _reach_state(row: Any, threshold_s: float | None) -> tuple[str, str | None, float | None]:
    """One published probe row as ``(state, probed_at, age_s)``.

    Two of the five states this surface renders are produced here rather than
    read: ``unknown`` for a row that is missing or carries none, and ``stale``
    for one whose age has passed *threshold_s*. ``not_applicable`` passes
    through untouched — the prober *decided* not to probe that gateway (Channel
    Access search runs over UDP, so a TCP connection would prove nothing), and
    ageing a decision out would report a working deployment as broken.
    """
    if not isinstance(row, dict):
        return REACH_UNKNOWN, None, None
    state = row.get("state")
    if not isinstance(state, str) or not state:
        return REACH_UNKNOWN, None, None
    stamp = row.get("probed_at")
    probed_at = stamp if isinstance(stamp, str) and stamp else None
    age_s = _age_seconds(probed_at)

    from osprey.mcp_server.control_system.endpoint_prober import STATUS_NOT_APPLICABLE

    if state == STATUS_NOT_APPLICABLE:
        return state, probed_at, age_s
    if threshold_s is not None and age_s is not None and age_s > threshold_s:
        return REACH_STALE, probed_at, age_s
    return state, probed_at, age_s


def _collapse_reachability(
    roles: Any, selected_role: str | None, threshold_s: float | None
) -> dict[str, Any]:
    """One row's reachability: the SELECTED role's state, the rest named beside.

    The collapse rule, in one place, because it is a judgement and not a lookup.
    A target has one probe row per gateway role the deployment configures and a
    connector uses exactly one of them — EPICS keeps a single process-wide
    context. So the row reports the state of the role ``derive_endpoints``
    selects under this session's EFFECTIVE posture: a target narrowed to
    read-only is reachable if its READ gateway answers, and a write gateway that
    is down says nothing about it.

    The other roles are not merged in. An OR would call a target reachable
    through a gateway it will not use; an AND would call it unreachable for the
    same reason. They are named separately in ``role_detail`` so a tooltip can
    say what else was measured without either verdict pretending to be the row's.
    """
    rows = roles if isinstance(roles, dict) else {}
    state, probed_at, age_s = _reach_state(
        rows.get(selected_role) if selected_role else None, threshold_s
    )
    return {
        "state": state,
        "role": selected_role,
        "probed_at": probed_at,
        "age_s": age_s,
        "role_detail": {
            str(role): _reach_state(row, threshold_s)[0]
            for role, row in rows.items()
            if role != selected_role
        },
    }


def _posture_lookup_key(app: Any, session_key: str) -> str:
    """The key this session's narrowings are actually recorded under.

    The read order :func:`_posture_entry` applies, as a key rather than as an
    entry, because :func:`~osprey_connectors.session_store.effective_writes`
    indexes the store itself and takes one key. Resolving it here means the
    popover's ``effective`` column and the connector's own refusal are reading
    the same entry — a route that handed over the current key while the running
    child answers to the spawn key would show an operator a narrowing their
    agent is not under.
    """
    store = _session_postures(app)
    if store.get(session_key) is not None:
        return session_key
    spawn_key = _spawn_posture_key(app, session_key)
    return spawn_key if store.get(spawn_key) is not None else session_key


def _effective_writes(section: Any, store_key: str, target: str) -> bool:
    """Whether *target* may be written on THIS session, right now.

    ``ceiling ∧ not is_readonly_run() ∧ store entry ≠ sandbox`` — rule 3 of the
    posture-store contract, and delegated to
    :func:`~osprey_connectors.session_store.effective_writes` rather than
    restated. The contract has exactly two implementations (that one and the
    stdlib restatement the hooks carry); a third spelled out in a route is how a
    popover comes to show ``writes`` on a machine the connector refuses.

    The session's own store key is passed rather than this process's
    ``OSPREY_POSTURE_SESSION``: the web server carries no such stamp, so the
    in-agent wrapper (``effective_writes_for_target``) would read no narrowing
    at all here and report the persona ceiling as the effective posture.

    An unreadable render answers ``False``, where every other predicate on this
    surface lands.
    """
    if section is _UNREADABLE_SECTION:
        return False
    try:
        return bool(session_store.effective_writes(section, store_key, target))
    except Exception:  # noqa: BLE001 — an underivable posture is not a writable one
        logger.warning("Could not resolve the effective write posture for target %s", target)
        return False


def _row_selected_role(config: Any, target: str, writes_enabled: bool) -> str | None:
    """The gateway role a connector would select for *target* under this posture.

    The reachability collapse keys on it, so it is derived through
    :func:`~osprey.mcp_server.control_system.target_eligibility.derive_endpoints`
    — the function the connector-host child's own selection is verified against
    — with the session's effective posture rather than the configured one. A
    role derived from config alone would name the write gateway for a target the
    operator has just narrowed, and the row would report the reachability of a
    gateway this session will never open.

    ``None`` for a target this render cannot derive at all, which the collapse
    renders as ``unknown``.
    """
    if config is _UNREADABLE_SECTION:
        return None
    try:
        from osprey.mcp_server.control_system.target_eligibility import derive_endpoints

        return derive_endpoints(config, target, writes_enabled=writes_enabled).selected_role
    except Exception:  # noqa: BLE001 — an underivable target selects no role
        logger.warning("Could not derive the selected gateway role for target %s", target)
        return None


def _row_narrowing_refusal(config: Any, target: str) -> str | None:
    """What narrowing *target* would cost, as a reason word, or ``None``.

    :func:`~osprey.mcp_server.control_system.target_eligibility.narrowing_refusal`
    — the same hypothetical derivation the POST answers its 409 with, so the
    toggle the popover LOCKS and the toggle the route would refuse are decided
    by one function. A deployment whose block configures ``write_access`` alone
    has nothing for a narrowed session to select: it would land on ``read_only``,
    find no such gateway, and the target would stop being usable at all. The
    operator is owed that before they click, not after.

    Reported as the reason word rather than the sentence: the sentence is the
    POST's to say when the gesture is actually made, and the row needs a token
    to key a lock label on. ``selected_role_missing`` is the case this exists
    for; ``target_unresolvable`` can arrive from the same call and is passed
    through rather than flattened, because a row that cannot be derived at all
    is not a row whose toggle should look live.

    Fails **open** (``None``) on an unreadable render, which is where the POST's
    own narrowing check lands: a narrowing on a config nobody can read is the
    safe direction to let through.
    """
    if config is _UNREADABLE_SECTION:
        return None
    try:
        from osprey.mcp_server.control_system.target_eligibility import narrowing_refusal

        verdict = narrowing_refusal(config, target)
    except Exception:  # noqa: BLE001 — an underivable narrowing is not a refusal
        logger.warning("Could not evaluate what narrowing target %s would cost", target)
        return None
    return verdict.reason if verdict is not None else None


def _row_availability(
    config: Any, target: str, session_target: str, baseline: str, writes_enabled: bool
) -> tuple[bool, str | None, str | None]:
    """Whether a switch to *target* is offered now, plus the reason and its sentence.

    :func:`~osprey.mcp_server.control_system.target_eligibility.target_availability`
    and nothing else, so the reason under a missing Switch button is the reason
    the reconciler would refuse with, character for character. Re-deriving it
    here would be a second opinion about the same machine, phrased differently.
    The verdict's two voices travel together: ``reason`` is the machine code
    the popover and the switch tool key on, and the ``detail`` sentence is what
    the popover puts on the operator's tooltip.

    An unreadable render offers no switch and names
    :data:`~osprey.mcp_server.control_system.target_eligibility.REASON_TARGET_UNRESOLVABLE`:
    a server that cannot read its own config cannot say a target is reachable.
    """
    try:
        from osprey.mcp_server.control_system.target_eligibility import (
            REASON_TARGET_UNRESOLVABLE,
            target_availability,
        )
    except Exception:  # noqa: BLE001 — no eligibility module, no switch offered
        logger.warning("Could not import the target eligibility rules", exc_info=True)
        return False, None, None

    if config is _UNREADABLE_SECTION:
        return False, REASON_TARGET_UNRESOLVABLE, None
    try:
        verdict = target_availability(
            config, target, session_target, baseline, writes_enabled=writes_enabled
        )
    except Exception:  # noqa: BLE001 — an unjudgeable target is not an available one
        logger.warning("Could not judge availability for control target %s", target)
        return False, REASON_TARGET_UNRESOLVABLE, None
    # The verdict narrates an offered target too ("Target 'live' is
    # configured…"), but this field explains a REASON: with no refusal there
    # is nothing for a tooltip to explain, and publishing the happy sentence
    # would put prose on rows whose whole answer is the Switch button.
    detail = (verdict.detail or None) if verdict.reason else None
    return bool(verdict.available_now), verdict.reason, detail


def _target_display(config: Any, record: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Per-target ``label`` / ``endpoint`` / ``real_machine``, published first.

    The controls server mints these once, for the target it is actually running,
    and every reader renders what it was handed. So a live record wins wherever
    it carries a label: it describes the process that owns the connector, while
    the render describes what a *new* server would do with the config as it
    stands now.

    The render is the fallback, through the same
    :func:`~osprey.mcp_server.control_system.connector_host_manager.target_display_metadata`
    the writer uses — a card whose session has not started a controls server yet
    still has to name its targets, and naming them with a second derivation of
    this module's own is how a badge comes to disagree with a prompt.
    """
    published = record.get("targets") if isinstance(record, dict) else None
    derived: dict[str, Any] = {}
    if config is not _UNREADABLE_SECTION and isinstance(config, dict):
        try:
            from osprey.mcp_server.control_system.connector_host_manager import (
                target_display_metadata,
            )

            derived = dict(target_display_metadata(config))
        except Exception:  # noqa: BLE001 — the roster must render, not 500
            logger.warning("Could not derive the control targets' display metadata")

    meta: dict[str, dict[str, Any]] = {}
    for target, row in derived.items():
        meta[str(target)] = dict(row) if isinstance(row, dict) else {}
    if isinstance(published, dict):
        for target, row in published.items():
            if isinstance(row, dict) and row.get("label"):
                # MERGED over the derivation, never substituted for it. A writer
                # from an older build publishes a label and no ``real_machine``,
                # and that key decides which half of
                # :func:`_short_label_and_kind` answers — losing it renders a
                # stand-in as a muted simulator on the one surface whose job is
                # write safety. Published fields still win where they exist.
                meta[str(target)] = {**(meta.get(str(target)) or {}), **row}
    return meta


def _posture_view(app: Any, session_key: str, config_path: Path | None) -> dict[str, Any]:
    """Everything ``GET /api/terminal/posture`` reports. BLOCKING.

    Called through ``run_in_threadpool``: it parses ``config.yml``, globs the
    state directory and — through :func:`_session_record` — walks the process
    table, forking ``ps`` where there is no ``/proc``. On the event loop one
    wedged process table would stall every request this server is serving, the
    terminal websocket included, and the chip polls this route per open card.

    **One resolution per request.** The record is resolved once and every fact
    drawn from it — the session's target, ``last_switch``, ``reachability``,
    ``last_posture_realign`` — comes from that one read. Two resolutions could
    straddle a switch and describe two different machines in one payload. The
    render is read once for the same reason and memoized by ``config.yml``'s
    signature besides (:func:`_rendered_config`).
    """
    config = _rendered_config(config_path)
    section = _section_of(config)
    record = _session_record(app, session_key)
    pty_pid = _pty_pid_for(app, session_key)

    # A chat key has no PTY, so it has no controls server of its own and no
    # record will ever match it. That is not a failure — its rows still carry
    # the ceiling, its own narrowing and the effective answer, which is
    # everything the toggles need. Only the switch, and the reachability a
    # prober publishes into a record, are out of its reach.
    is_chat = pty_pid is None and _chat_pool_answers_to(app, session_key)

    baseline = "live"
    if section is not _UNREADABLE_SECTION:
        try:
            from osprey_connectors.types import baseline_target

            baseline = baseline_target(section)
        except Exception:  # noqa: BLE001 — a render we cannot classify is `live`
            logger.warning("Could not resolve the deployment's baseline control target")

    published_target = (record or {}).get("target")
    session_target = (
        published_target if isinstance(published_target, str) and published_target else baseline
    )

    store_key = _posture_lookup_key(app, session_key)
    entry = _posture_entry(app, session_key)
    ceilings = _session_ceilings(section)
    display = _target_display(config, record)
    threshold_s = _probe_staleness_threshold_s(config)
    reachability = (record or {}).get("reachability")
    probed = reachability.get("targets") if isinstance(reachability, dict) else {}

    rows: list[dict[str, Any]] = []
    for target in _configured_target_names(section):
        meta = display.get(target) or {}
        real_machine = bool(meta.get("real_machine"))
        label = str(meta.get("label") or "") or target
        short_label, kind = _short_label_and_kind(label, real_machine)
        # ONE ceiling per row. ``ceiling_writes`` reports ``session_posture``'s
        # per-target map; ``effective`` runs through the store, whose own
        # ceiling is ``target_writes_enabled`` for every configured target. On a
        # NON-switch-capable render those two disagree — ``session_posture``
        # answers for the baseline alone, while ``configured_targets`` still
        # lists the others — so an unarmed row would render ceiling-off beside
        # effective-on: a filled dot for a target no connector here is ever
        # built for. Gating on the ceiling the route already holds keeps the row
        # internally consistent; where the two agree (every switch-capable
        # deployment) this changes nothing.
        ceiling_writes = bool(ceilings.get(target, False))
        effective = ceiling_writes and _effective_writes(section, store_key, target)
        posture = entry.get(target, POSTURE_WRITES)
        # Only for a row a narrowing would actually CHANGE, mirroring the POST's
        # "targets the request would touch" rule: a target already narrowed
        # cannot be stranded by narrowing it, and reporting the refusal there
        # would lock the toggle that brings it back.
        narrowing = None if posture == POSTURE_SANDBOX else _row_narrowing_refusal(config, target)
        if is_chat:
            available_now, reason, reason_detail = False, REASON_CHAT_SESSION, None
        else:
            available_now, reason, reason_detail = _row_availability(
                config, target, session_target, baseline, effective
            )
        rows.append(
            {
                "target": target,
                "label": label,
                "display_name": str(meta.get("display_name") or ""),
                "short_label": short_label,
                "kind": kind,
                "endpoint": str(meta.get("endpoint") or ""),
                "real_machine": real_machine,
                "active": target == session_target,
                "is_baseline": target == baseline,
                "available_now": available_now,
                "reason": reason,
                "reason_detail": reason_detail,
                "ceiling_writes": ceiling_writes,
                "posture": posture,
                "effective": effective,
                "narrowing_refusal": narrowing,
                "reachability": _collapse_reachability(
                    (probed or {}).get(target) if not is_chat else None,
                    _row_selected_role(config, target, effective),
                    threshold_s,
                ),
            }
        )

    # Enforceability is a question about the ANCHOR, not about the store: every
    # surface that spawns a session stamps ``OSPREY_POSTURE_SESSION`` and the
    # agent-data root beside it, so a narrowing recorded under that key is one
    # the child will read. The single exception is a PTY session that has
    # started and still resolves no record of its own — its controls server is
    # outside this process tree, and the toggles would govern nothing.
    enforceable = not (pty_pid is not None and record is None)

    return {
        "session_target": session_target,
        "store_available": _posture_store_path() is not None,
        "enforceable": enforceable,
        "enforceable_reason": None if enforceable else ENFORCEABLE_REASON_NO_RECORD,
        "execution_in_flight": _first_live_execution() is not None,
        "last_switch": _aged((record or {}).get("last_switch")),
        "last_posture_realign": (record or {}).get("last_posture_realign") or None,
        "targets": rows,
    }


@router.get("/api/terminal/posture")
async def get_terminal_posture(session_id: str, request: Request):
    """Report one session's control-target roster and its write posture.

    The single truth the header chip and its popover read. One row per
    configured control target, each answering the whole of what the operator
    needs about that machine:

    * ``label`` / ``display_name`` / ``short_label`` / ``kind`` / ``endpoint``
      / ``real_machine`` — what to call it and what it is. The label is the one
      the controls server published for the target it is running, or the one
      this render derives for a session that has not started one;
      ``display_name`` is the operator-facing name minted beside it
      (``control_system.target_display_names`` renames it per deployment);
      ``short_label`` and ``kind`` come from ``real_machine`` and the label's
      shape, never from the target name, so a stand-in never renders as the
      facility's own machine.
    * ``ceiling_writes`` / ``posture`` / ``effective`` — the three terms of the
      write decision, kept separate on purpose. The ceiling is the deployment's
      (``session_posture``); the posture is this session's own narrowing; the
      effective answer is the whole rule the connector applies, which also folds
      in a read-only run. A popover that showed only the last of them could not
      say whether a locked toggle is the persona's doing or the operator's.
    * ``narrowing_refusal`` — ``null``, or the reason word narrowing this target
      would earn (``selected_role_missing`` on a deployment whose block
      configures ``write_access`` alone). It is the one lock reason the payload
      could not otherwise be derived from, and it is the same verdict the POST
      answers its 409 with, so the toggle the popover locks and the toggle the
      route would refuse are decided by one function. ``null`` on a row already
      narrowed: that toggle brings the target BACK, and nothing about it can
      strand anything.
    * ``active`` / ``is_baseline`` / ``available_now`` / ``reason`` /
      ``reason_detail`` — where the session is standing and whether a switch is
      offered. ``reason`` is the switch tool's own machine code, so the popover
      and the agent keep agreeing about the same refusal; ``reason_detail`` is
      the eligibility verdict's operator sentence, which the popover renders as
      the tooltip behind its short phrase.
    * ``reachability`` — the state of the gateway role this target would
      actually select under its effective posture, aged server-side, with the
      other roles named beside it (see :func:`_collapse_reachability`).

    And, once for the session: ``session_target``, ``store_available``,
    ``enforceable`` (+``enforceable_reason``), ``execution_in_flight``,
    ``last_switch`` (the publisher's block — ``request_id``, ``target``,
    ``status``, ``reason``, ``detail``, ``at`` — passed through whole with
    ``age_s`` added) and ``last_posture_realign``.

    Everything costs a config read, a state-directory glob and a walk of the
    process table, so it is computed in a worker thread (:func:`_posture_view`
    via ``run_in_threadpool``).

    Unlike POST, an id that names no session on disk is **not** a 409. The chip
    renders with the page, which can be before the first prompt has written a
    session file, and refusing there would blank the one surface that tells the
    operator what the deployment permits. Answering costs nothing: a read grants
    nothing, stores nothing, and reports exactly the posture that session will
    run under — for a chat key just as for a PTY one. The id is still
    shape-checked with the closed grammar POST uses, so the two routes keep one
    error contract.
    """
    _require_session_uuid(session_id)

    view = await run_in_threadpool(
        _posture_view, request.app, session_id, request.app.state.config_path
    )
    return {"session_id": session_id, **view}


@router.post("/api/terminal/logout")
async def logout_terminal(request: Request, response: Response):
    """Revoke the browser session, then terminate the warm PTY (and operator) pools.

    **Revocation first, because it is the only part that is a guarantee.**
    A cookie value that has already left this process cannot be un-sent —
    it is sitting in a browser jar, possibly in a proxy log, possibly in a
    second tab — so clearing it in the browser is a courtesy the client is
    free to ignore and an attacker certainly will. What makes logout real
    is the server no longer holding the session: ``revoke_session`` drops
    the digest from the in-memory map *and* rewrites the on-disk store, so
    the credential is refused from the next request onward and stays
    refused across a restart. That has to happen before the pools are
    emptied, so that a request racing this one cannot re-attach to a
    session on its way out with a cookie this handler has not yet dropped.

    **Every candidate cookie is revoked, not just the first — and the
    header is read the way the gate reads it.** A browser can be made to
    send two cookies of the same name: a page on a sibling host under the
    same registrable domain sets a ``Domain``-scoped one and the browser
    then sends it alongside the app's own host-scoped cookie, in an order
    this app does not control (see ``read_cookie_candidates`` in
    ``common_middleware``). The gate accepts *any* of them, so logging out
    only the one that happened to come first would leave a live session
    behind, and the operator would have no way to tell. That primitive is
    also what rejoins the repeated ``Cookie`` *headers*: HTTP/2 permits a
    client to split the cookie header, so a session offered only in the
    second one is a credential the gate honours. Reading the header any
    differently from the gate is precisely how a credential ends up
    admitted but never revoked, which is why neither side spells the rule
    itself and both go through one reader. The count is
    reported back in the body so the client — and a test — can see how many
    were actually live rather than how many were offered.

    **The delete cookie carries no ``Secure``.** A browser matches a cookie
    for deletion by name, domain and path — never by its other attributes —
    so an expiry that omits ``Secure`` still clears a cookie that was set
    with it. The reverse is not true: a ``Secure`` delete sent over plain
    ``http`` is discarded before it can match anything, which is exactly
    the single-user loopback shape. The exchange derives ``Secure`` from
    the browser-facing origin (``WebAuthMiddleware._session_cookie`` /
    ``_cookie_is_secure``) because it is handing out a credential that must
    not travel in the clear; mirroring that derivation here would buy
    nothing and would silently strand the delete on the one shape where it
    is wrong. Set through ``response.headers`` so the ordinary dict body
    below is still serialised by FastAPI, with this header merged onto it.

    Each Web Terminal container serves a single user (the multi-user
    topology puts one container behind each ``/u/<user>/`` path), so — like
    ``/api/terminal/restart`` — there is no per-caller session to pick out;
    the whole pool is this user's. Unlike restart, which the client
    immediately reconnects to (respawning a fresh PTY under the same
    flow), logout must not leave anything resumable behind: this empties
    both pools — the PTY registry and the operator-mode (Agent SDK)
    registry, the latter a live agent with tool access and therefore the
    more sensitive of the two — via their existing ``cleanup_all``
    primitives, mirroring ``restart_terminal`` (routes/panels.py), so the
    next visitor at a shared browser inherits no live session of either
    kind (closes the M2 warm-session-inheritance hazard). The client
    clears its stored session id and navigates to the landing page
    afterward — it does not reconnect.
    """
    # The name is re-derived rather than read back from the gate because no
    # app pins one: every interface installs ``WebAuthMiddleware`` with no
    # ``cookie_name`` (``_app_setup.py``), so both sides resolve the same
    # ``session_cookie_name()`` from ``OSPREY_WEB_PORT``. A deployment that
    # ever does pin the middleware's name has to route the settled name here
    # too — logout would otherwise revoke nothing and expire a cookie the
    # browser does not hold, a total failure reported as a cheerful 200.
    cookie_name = session_cookie_name()
    credentials = get_web_credentials(request.app)
    cookie_headers = request.headers.getlist("cookie")

    def _revoke_candidates() -> int:
        """Revoke every offered candidate, returning how many were live.

        Off the event loop: a revocation that hits writes the session store
        through a full atomic replace — temp file, ``json.dump``, ``fsync``,
        rename — and there can be one per candidate. That is a handful of
        milliseconds of blocking disk I/O in the best case and unbounded on a
        stalled filesystem, and every other connection this process is serving
        would wait it out.
        """
        live = 0
        for candidate in read_cookie_candidates(cookie_headers, cookie_name):
            if credentials.revoke_session(candidate):
                live += 1
        return live

    # Revoke before the pools are torn down: see the docstring.
    sessions_revoked = await run_in_threadpool(_revoke_candidates)
    if sessions_revoked:
        logger.info("Browser session(s) revoked for logout: %d", sessions_revoked)

    # No ``Secure``: a delete must be able to land on the plain-http shape too.
    response.headers.append(
        "set-cookie",
        f"{cookie_name}=; Max-Age=0; Path=/; HttpOnly; SameSite=Lax",
    )

    pty_registry = request.app.state.pty_registry
    operator_registry = request.app.state.operator_registry

    # Terminate all PTY sessions (single-user model)
    pty_registry.cleanup_all()
    logger.info("PTY session(s) terminated for logout")

    # Terminate all operator sessions if active
    try:
        await operator_registry.cleanup_all()
    except Exception:
        pass  # May not have active operator sessions

    return {
        "status": "ok",
        "message": "Logged out — terminal session terminated",
        "sessions_revoked": sessions_revoked,
    }


@router.websocket("/ws/operator")
async def operator_ws(websocket: WebSocket):
    """WebSocket bridge for operator-mode (Claude Agent SDK).

    Protocol:
    - Client -> Server JSON: {"type": "prompt", "text": "..."}
    - Client -> Server JSON: {"type": "cancel"}
    - Server -> Client JSON: structured events (text, thinking, tool_use, etc.)
    """
    await websocket.accept()

    registry = websocket.app.state.operator_registry
    cwd = websocket.app.state.project_cwd
    operator_key = f"operator-{uuid.uuid4().hex[:8]}"
    session = None
    forward_task = None

    try:
        # operator_key is this connection's whole identity — the operator
        # websocket resumes no Claude session — so it is the key the runtime
        # posture is looked up under.
        # "spawn": operator_key is minted here and addressable by nothing else
        # (the posture route only takes a session UUID), so whatever the store
        # holds for it at spawn is the whole story this child's audit records
        # can tell about where its posture came from.
        env = build_operator_child_env(
            project_cwd=cwd,
            session_key=operator_key,
            app=websocket.app,
            posture_source=POSTURE_SOURCE_SPAWN,
        )
        session = await registry.create_session(operator_key, cwd=cwd, env=env)
    except Exception as exc:
        logger.error("Failed to create operator session: %s", exc)
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"Failed to start operator session: {exc}",
                    "error_type": type(exc).__name__,
                }
            )
        except Exception:
            pass
        await websocket.close()
        return

    async def forward_events():
        """Drain the session queue and send events to the WebSocket."""
        try:
            while True:
                event = await session._queue.get()
                if event.get("type") == "keepalive":
                    continue
                await websocket.send_json(event)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    forward_task = asyncio.create_task(forward_events())

    try:
        # Notify client that operator session is ready
        await websocket.send_json({"type": "system", "subtype": "init"})

        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")
            if msg_type == "prompt":
                text = msg.get("text", "").strip()
                if text:
                    await session.send_prompt(text)
            elif msg_type == "cancel":
                await session.cancel()

    except WebSocketDisconnect:
        pass
    finally:
        if forward_task is not None:
            forward_task.cancel()
            try:
                await forward_task
            except asyncio.CancelledError:
                pass
        if session is not None:
            await registry.terminate_session_if_owner(operator_key, session)
