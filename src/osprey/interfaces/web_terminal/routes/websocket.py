"""WebSocket routes for terminal PTY and operator (Agent SDK) sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from osprey.interfaces.web_auth import PANEL_TOKEN_ENV, get_web_credentials
from osprey.interfaces.web_terminal.operator_session import (
    POSTURE_SESSION_ENV,
    POSTURE_SOURCE_ENV,
    POSTURE_SOURCE_LIVE,
    POSTURE_SOURCE_SPAWN,
    build_operator_child_env,
)
from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery

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

# ── Per-session runtime posture ──────────────────────────────────────────────
#
# The posture is the operator's per-session sandbox toggle: step a live session
# into ``sandbox`` and back out to ``writes``. ``sandbox`` spawns the session's
# agent with ``OSPREY_EXECUTION_MODE=readonly``; ``writes`` is the render's own
# baseline and adds nothing. It is deliberately *not* a config edit — config is
# a build-time input that reaches the agent only through a re-render, and it is
# deployment-wide, whereas this is one session, applied by respawning that
# session's child.
POSTURE_SANDBOX = "sandbox"
POSTURE_WRITES = "writes"
_VALID_POSTURES = frozenset({POSTURE_SANDBOX, POSTURE_WRITES})

# Store filename, sited on the shared agent-data root (see _posture_store_path).
_POSTURE_STORE_NAME = "session-postures.json"


class PostureRequest(BaseModel):
    """Body of ``POST /api/terminal/posture``.

    ``posture`` is a ``Literal`` so an unknown value is rejected by request
    validation with a 422 naming the field, before any handler code runs — the
    value decides a child process's execution mode, and a silent coercion to
    some default would be the worst possible failure here.
    """

    session_id: str
    posture: Literal["sandbox", "writes"]


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

    What it is emphatically not is a gate on whether there is anything to
    terminate. The map it reads is one of *two* places a chat can live: a
    creation still inside ``start()`` sits in the pool's ``_pending`` and is
    invisible here, which is exactly the child a posture flip most needs to
    catch. :func:`_terminate_for_respawn` therefore terminates unconditionally
    and uses this only to describe what happened.

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
    with tools — and answering ``False`` there is what made a flip during chat
    creation a 409 that wrote nothing and terminated nothing, while the
    pre-flip child went on to register itself into the pool with the
    environment the operator had just stepped out of.

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


async def _terminate_for_respawn(app, session_id: str) -> tuple[str, ...]:
    """Terminate every live child answering to *session_id*, via its owning pool.

    A posture reaches an agent only through a child process's environment, so
    applying one means killing the child that carries the old one and letting
    it come back. Which pool can do that is a property of the topology, not of
    the key's shape:

    * the **PTY registry** owns terminal sessions keyed on a Claude UUID; and
    * the **chat pool** behind ``operator_registry`` owns Simple-mode SDK
      sessions keyed on the caller's ``chat_id``.

    The PTY registry is asked whether it holds a session under this key,
    because ``terminate_session`` there is only meaningful for one it holds.
    The chat pool is **not** asked: it is told to terminate whenever the
    registry exposes the call. The route used to blind-call the PTY registry
    for every key; for a chat key that pops nothing, so the operator got a 200
    and the SDK child kept running with the posture they had just left.

    Gating the chat terminate on a membership probe would reintroduce that in a
    narrower window. The probe reads the pool's session map, and a creation
    still inside ``start()`` lives in the pool's ``_pending`` instead — so a
    flip arriving while a chat is being created would answer "nothing here",
    skip the terminate, and let the pre-flip child register itself into the
    pool *after* the operator was told 200. ``ChatSessionPool.terminate`` is
    documented idempotent and busy-safe: for a key it does not hold it pops
    nothing and supersedes nothing, so the unconditional call costs one no-op
    on a PTY-only key and closes the window on a chat one.

    ``ChatSessionPool.has_key`` *can* see a starting creation, and the
    addressability gate uses it — but gating the terminate on it would still
    be a probe and an act with a gap between them, and the pool's own lock is
    the only thing that closes that gap. The unconditional call keeps the
    decision inside the pool, where it belongs.

    Both pools can own one key at once — ``chat_id`` is caller-chosen and
    nothing stops an embedder from picking a live PTY session's UUID — so both
    are terminated in that case rather than one being guessed at.

    Returns the names of the pools that had a *pooled* entry under the key, for
    the log line. It is a description, not the set of calls made: an in-flight
    chat creation is superseded without appearing here, because the supersede
    is deliberately fire-and-forget (a hung ``start()`` must not hang the
    operator's toggle) and the pool has no answer to give back yet. An empty
    tuple means the key was addressable but nothing was pooled under it, which
    is the normal state of a discovered PTY session nobody has attached to.
    """
    terminated: list[str] = []

    pty_registry = getattr(app.state, "pty_registry", None)
    if pty_registry is not None and pty_registry.get_session(session_id) is not None:
        pty_registry.terminate_session(session_id)
        terminated.append("pty")

    registry = getattr(app.state, "operator_registry", None)
    terminate_chat = getattr(registry, "terminate_chat_session", None)
    held_a_chat_entry = _holds_a_chat_pool_entry(app, session_id)
    if callable(terminate_chat):
        await terminate_chat(session_id)
        if held_a_chat_entry:
            terminated.append("chat")
    elif held_a_chat_entry:
        # Same tolerance the membership probe grants an unfamiliar registry
        # (a test double, say): it can answer the probe but cannot be told to
        # act. The store is still written, so the next spawn under this key
        # carries the posture — only the live child outlives the change.
        logger.warning(
            "Session %s names a chat-pool entry on a registry with no "
            "terminate_chat_session; its running child keeps the old posture "
            "until it is restarted",
            session_id,
        )

    return tuple(terminated)


def _posture_key_is_addressable(app, session_id: str) -> bool:
    """Whether some real session answers to *session_id* on either topology.

    The posture is per session, so a key that names no session is a key nothing
    will ever spawn under. Three answers count, cheapest first:

    * a **chat session the pool would answer to** — pooled, or still inside
      ``start()`` (:func:`_chat_pool_answers_to`). The SDK topology's key
      exists from the moment the pool accepts the creation and never appears
      in the JSONL stems the discovery walks.
    * a **key the posture store already holds an entry for**. The chat pool is
      LRU-capped, idle-reaped, and evicted by the flip itself, so pool
      membership is a fact with an end — and without this clause a chat
      sandboxed once could never be brought back out, because the successful
      flip is what removed it from the pool. An entry in the store is the
      operator's own earlier, accepted decision about this key; letting them
      revise it grants nothing new (the store only ever *narrows* a spawn, and
      a key nothing spawns under is inert), while refusing it would strand a
      session in the sandbox.
    * a **PTY session discovered on disk** — a Claude session-file stem, which
      only exists once the operator has sent a prompt. Checked last: it walks
      the session directory, while the other two are in-memory reads.

    Checking only the stems is what made a chat session's posture unsettable:
    the chat spawn already honours the store (``_acquire_chat_turn`` passes the
    key to ``build_operator_child_env``), so the store was readable on that
    surface and not writable — a badge the operator could not act on.
    """
    if _chat_pool_answers_to(app, session_id):
        return True
    if session_id in _session_postures(app):
        return True
    return session_id in SessionDiscovery(app.state.project_cwd).snapshot_session_ids()


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


def _resolve_posture_store_path(app) -> tuple[Path, bool]:
    """Return the posture store's location and whether it is the fallback one.

    Sited on the CONFIGURED agent-data root, like the feedback store
    (``app.py``): with ``web_terminal.watch_dir`` set the watched tree is
    somewhere else entirely, and a store written there would land outside the
    ``{user}-agent-data`` volume — the one directory that survives a container
    recreation, which is the whole point of persisting this. Never
    ``resolve_agent_data_root()``: that appends ``sessions/<OSPREY_SESSION_ID>``
    and this store spans sessions by definition.

    The second element says the primary root could not be resolved and the
    workspace dir stood in for it. Callers need it because a config load can
    fail *transiently*: a load off the fallback path must not be cached as
    though it were the real store (see :func:`_session_postures`).
    """
    try:
        from osprey.utils.workspace import resolve_shared_data_root

        root = resolve_shared_data_root()
    except Exception:  # noqa: BLE001 — never let a config load break the toggle
        root = Path(getattr(app.state, "workspace_dir", Path.cwd()))
        logger.warning(
            "Could not resolve the shared data root for the session-posture store; "
            "falling back to %s",
            root,
            exc_info=True,
        )
        return Path(root) / _POSTURE_STORE_NAME, True
    return Path(root) / _POSTURE_STORE_NAME, False


def _posture_store_path(app) -> Path:
    """Where the posture store lives, primary root or fallback."""
    return _resolve_posture_store_path(app)[0]


#: Prefix of the ``/ws/operator`` session keys (``operator-<hex8>``, minted per
#: accepted websocket). Postures under these keys are deliberately
#: **non-durable**: see :func:`_load_postures`.
_NON_DURABLE_KEY_PREFIX = "operator-"


def _load_postures(path: Path) -> dict[str, str]:
    """Read the persisted postures, tolerating every kind of absence.

    Unknown posture values are dropped rather than honored: whatever survives
    this filter flows straight into :func:`_build_extra_env` and decides a
    child's execution mode, so a hand-edited or future-version entry must not
    reach it. A missing or corrupt file yields an empty store — the operator
    can set the postures again, which is a far better outcome than every spawn
    and every toggle failing on a file nobody can repair from the browser.

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
    :func:`_persist_postures` still writes whatever the in-memory store holds,
    operator keys included — the in-memory entries are live and load-bearing
    for the rest of the process's life, and dropping them on the way *out*
    would only add a second place for the rule to drift.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:  # noqa: BLE001 — corrupt/unreadable store must not wedge startup
        logger.warning("Could not read the session-posture store at %s", path, exc_info=True)
        return {}
    if not isinstance(raw, dict):
        logger.warning("Session-posture store at %s is not a JSON object; ignoring", path)
        return {}
    return {
        key: value
        for key, value in raw.items()
        if isinstance(key, str)
        and value in _VALID_POSTURES
        and not key.startswith(_NON_DURABLE_KEY_PREFIX)
    }


def _session_postures(app) -> dict[str, str]:
    """Return ``app.state.session_postures``, loading it from disk on first use.

    Lazily initialised rather than wired into the lifespan so the whole feature
    lives in this module. First access is whichever comes first after a
    restart — a spawn or a posture route — and both go through here, so a
    recreated container never serves a session its persisted posture has not
    been applied to.

    A load off the *fallback* path is kept provisional and retried on the next
    access. Caching it would let one transient config failure at first access
    outlive itself: every later read would serve the empty fallback store and
    spawn a persisted-sandbox session without the marker — a silent revert to
    writes, which is precisely what persisting the store exists to prevent.
    """
    store = getattr(app.state, "session_postures", None)
    if store is not None and not getattr(app.state, "session_postures_provisional", False):
        return store

    path, used_fallback = _resolve_posture_store_path(app)
    loaded = _load_postures(path)
    if store is None:
        store = loaded
    else:
        # Recovering from an earlier fallback load. The persisted store is
        # authoritative for everything memory has not been told about, but a
        # posture the operator set *during* the outage exists only in memory
        # (and in the fallback file nothing reads again), so it wins on
        # overlap — otherwise the recovery read would quietly undo it. Mutated
        # in place because callers hold this dict.
        merged = {**loaded, **store}
        store.clear()
        store.update(merged)
    app.state.session_postures = store
    app.state.session_postures_provisional = used_fallback
    return store


def _persist_postures(app, store: dict[str, str]) -> None:
    """Write the store through to disk, best-effort.

    Fail-open relative to operator intent: the in-memory store already carries
    the new posture and the session is terminated either way, so the toggle
    takes effect now even when the write fails. Only its durability across a
    restart is lost, and refusing the operator's request over that would be the
    worse trade — it would leave a live session in the posture they asked to
    leave.
    """
    path = _posture_store_path(app)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, store)
    except Exception:  # noqa: BLE001 — durability is best-effort, the toggle is not
        logger.warning(
            "Could not persist the session-posture store to %s; the posture is applied "
            "for this process but will not survive a restart",
            path,
            exc_info=True,
        )


#: Sentinel telling "the render could not be read" apart from "the render has
#: no ``control_system:`` block". Both answer writes-off, but only the first is
#: a failure worth logging, and the connector helpers have their own opinion
#: about a ``None`` section that the failure case must not borrow.
_UNREADABLE_SECTION = object()


def _control_system_section(config_path: Path | None) -> Any:
    """The rendered ``control_system:`` section, or :data:`_UNREADABLE_SECTION`.

    Reads and parses the file, so a caller that needs the section for more than
    one question reads it ONCE and passes the result down —
    :func:`_posture_render_facts` is the reason this is separate from the
    predicates below. Blocking; never call it from the event loop.
    """
    try:
        if not config_path or not Path(config_path).exists():
            return _UNREADABLE_SECTION
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — an unreadable config is not a writes-on render
        logger.warning("Could not read the control-system section from %s", config_path)
        return _UNREADABLE_SECTION
    return config.get("control_system") if isinstance(config, dict) else None


def _rendered_writes_enabled(config_path: Path | None) -> bool:
    """Whether the rendered config arms control-system writes for *any* target.

    Write posture is per connector type — each target's
    ``control_system.connector.<type>.writes_enabled``, inheriting
    ``control_system.writes_enabled`` where it is absent — so the render permits
    writes as soon as one target is armed. A deployment that arms only its
    virtual accelerator can still step a session out of the sandbox; what it
    cannot do is write to the live machine, and the connector layer refuses that
    on its own rather than this gate pre-empting it.

    ``any_target_writes_enabled`` rather than a loop over ``CONTROL_TARGETS``:
    the union has to run over the targets a session here can actually be pointed
    at, and an unresolvable one falls back to the deployment-wide key. On a
    virtual-accelerator deployment with no live block that fallback would answer
    for a machine no session reaches, and a global ``true`` over an explicitly
    disarmed simulator would offer the operator a button that arms nothing.

    The one predicate behind both the 403 gate and the badge's
    ``rendered_writes_enabled``, so the button an operator is offered and the
    answer they get when they press it can never disagree.

    Everything that can go wrong — no config, an unreadable one, a section the
    resolver cannot make sense of — answers ``False``: no target may write. That
    is the default ``cli/templates/claude_code.py`` bakes into
    ``permissions.deny`` and the ``osprey_writes_check`` hook applies per call,
    so an absent config stays a writes-off render here as everywhere else.
    """
    return _any_target_writes_enabled(_control_system_section(config_path))


def _any_target_writes_enabled(section: Any) -> bool:
    """:func:`_rendered_writes_enabled` over an ALREADY-READ section.

    Split out so the GET route can answer both of its render questions from a
    single read of ``config.yml`` while the POST gate keeps its one-argument
    predicate. Both paths end here, so the button an operator is offered and the
    answer they get on pressing it still cannot disagree.
    """
    if section is _UNREADABLE_SECTION:
        return False
    try:
        from osprey_connectors.types import any_target_writes_enabled

        return bool(any_target_writes_enabled(section))
    except Exception:  # noqa: BLE001 — an unreadable config is not a writes-on render
        logger.warning("Could not read the control-system write posture")
        return False


def _posture_render_facts(config_path: Path | None, pty_pid: int | None) -> dict[str, Any]:
    """Everything the badge needs from the render and the process table.

    **Blocking, and deliberately so**: it reads ``config.yml`` and — through
    :func:`~osprey.mcp_server.control_system.target_banner.session_target_for_pid`
    — walks the process table, which on a platform without ``/proc`` means
    forking ``ps``. The GET route hands the whole thing to a worker thread
    (``run_in_threadpool``) rather than doing any of it on the event loop, where
    a wedged process table would stall every other request, the terminal
    websocket included.

    One read of the config serves both render questions. The route used to call
    :func:`_rendered_writes_enabled` and the target lookup separately, which
    parsed the same YAML twice per request — now twice per five seconds per
    open card, thanks to the badge's refresh poll.
    """
    section = _control_system_section(config_path)
    return {
        "rendered_writes_enabled": _any_target_writes_enabled(section),
        **_session_target_posture(section, pty_pid),
    }


def _session_target_posture(section: Any, pty_pid: int | None) -> dict[str, Any]:
    """Which control target this session is on, and whether THAT target is armed.

    ``rendered_writes_enabled`` is a union over every target a session here can
    select — true as soon as one is armed — so on a mixed render (a live machine
    read-only beside an armed virtual accelerator) it says nothing about the
    machine the operator is actually pointed at. An operator on the live target
    would read it as "this session can write" and have every write refused, per
    call, by the connector layer. These three fields are what let the badge say
    the honest thing instead:

    * ``session_target`` — the target the controls MCP server published for this
      PTY's process tree, or the deployment baseline when there is no answer;
    * ``target_writes_enabled`` — the per-type posture for *that* target;
    * ``target_source`` — ``session`` when a live record matched, ``baseline``
      otherwise, so the badge can mark a fallback as one rather than presenting
      a guess as fact.

    Every unknowable case is the baseline: no controls server yet, a record
    another session owns, two ambiguous records, a dead server, an unreadable
    process table. That is the direction ``target_banner`` and the Claude Code
    hooks already take, and it is the one that cannot invent a target nobody is
    on. An absent or unreadable render answers "baseline, and no writes", the
    same posture the rest of this module gives it.

    Takes an already-read *section* (see :func:`_posture_render_facts`), not a
    path. Blocking; never call it from the event loop.
    """
    readable = section is not _UNREADABLE_SECTION

    baseline = "live"
    if readable:
        try:
            from osprey_connectors.types import baseline_target

            baseline = baseline_target(section)
        except Exception:  # noqa: BLE001 — a render we cannot classify is `live`
            logger.warning("Could not resolve the deployment's baseline control target")

    resolved: str | None = None
    if pty_pid:
        try:
            from osprey.mcp_server.control_system.target_banner import session_target_for_pid

            resolved = session_target_for_pid(pty_pid)
        except Exception:  # noqa: BLE001 — the badge must render, not 500
            logger.warning("Could not resolve the session's control target", exc_info=True)

    target = resolved or baseline
    armed = False
    if readable:
        try:
            from osprey_connectors.types import target_writes_enabled

            armed = bool(target_writes_enabled(section, target))
        except Exception:  # noqa: BLE001 — an unreadable posture is not an armed one
            logger.warning("Could not read the write posture for control target %s", target)

    return {
        "session_target": target,
        "target_writes_enabled": armed,
        "target_source": "session" if resolved else "baseline",
    }


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

    Returns the discovered UUID (or None). Also rekeys the registry entry —
    ``app`` is handed along so the session's posture-store entry moves with it
    and the audit alias back to the spawn key is recorded; see
    :meth:`~osprey.interfaces.web_terminal.pty_manager.PtyRegistry.rekey_session`.
    """
    loop = asyncio.get_event_loop()
    new_id = await loop.run_in_executor(None, discovery.discover_new_session, snapshot, timeout)
    if new_id:
        registry.rekey_session(current_key, new_id, app=websocket.app)
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

    # Per-session runtime posture. Keyed on the pool key — ``terminal_ws``
    # computes ``current_key = claude_session_id or telemetry_session_id`` and
    # all three call sites reach here with that same pair, so this expression
    # is the pool key in every one of them (a brand-new session, whose claude
    # id is still None, included).
    #
    # Applied AFTER the hooks_env merge, and only ever by *setting* the
    # sandbox marker — never by clearing one. A deployment that injects
    # ``OSPREY_EXECUTION_MODE=readonly`` through hooks_env has made a
    # deployment-wide decision, and a per-session ``writes`` posture must not
    # be able to lift it: this toggle narrows privilege, it never widens it.
    # The audit pair rides along whenever there is a key to name, ``writes``
    # sessions included: it records which key was consulted and that a live
    # store answered, which is a different fact from the posture it answered
    # with. Only the posture *value* below is narrowing-only. The source is
    # always "live" here — a PTY pool key is exactly the id the posture route
    # addresses, so the store keeps answering for it after the child is up.
    posture_key = claude_session_id or telemetry_session_id
    if posture_key:
        extra_env[POSTURE_SOURCE_ENV] = POSTURE_SOURCE_LIVE
        extra_env[POSTURE_SESSION_ENV] = posture_key
        if _session_postures(websocket.app).get(posture_key) == POSTURE_SANDBOX:
            extra_env["OSPREY_EXECUTION_MODE"] = "readonly"
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


@router.post("/api/terminal/posture")
async def set_terminal_posture(body: PostureRequest, request: Request):
    """Set one session's runtime posture and respawn it under the new one.

    The posture reaches the agent only through a child process's environment,
    so recording it is not enough: the live child is terminated through the
    pool that owns it (:func:`_terminate_for_respawn`), and the next attach or
    prompt brings the session back with the posture applied. The store is
    updated and persisted *before* the terminate, because the respawn reads it
    back — through :func:`_build_extra_env` on the PTY seam and
    ``build_operator_child_env`` on the chat one — and would otherwise race it.

    Three refusals, all before anything is written:

    * **409** — the id names no session at all: no chat the pool would answer
      to, no entry already in the store, no Claude session file on disk (see
      :func:`_posture_key_is_addressable`). A PTY session's file only appears
      once the operator has sent a prompt, and a chat the pool has dropped and
      never had a posture set on comes back on its next prompt; until then
      there is nothing to respawn, and the detail says so, because sending a
      prompt is the entire remedy and the alternative is a toggle that
      silently does nothing.
    * **403** — ``writes`` on a render that arms no control target at all:
      ``control_system.connector.<type>.writes_enabled`` off for every type,
      and off in the deployment-wide ``control_system.writes_enabled`` they
      inherit from. One armed target is enough, because the posture may narrow
      what the render permits and never widen it.
    * **400** — an id outside the closed key grammar
      (:func:`_require_session_uuid`), so an arbitrary string can never become
      a store key that is then written to disk.
    """
    session_id = body.session_id
    _require_session_uuid(session_id)

    if not _posture_key_is_addressable(request.app, session_id):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "session_not_started",
                "message": (
                    "This session has not started yet — send one prompt first, "
                    "then set its posture. A chat session becomes addressable "
                    "again on its next prompt."
                ),
            },
        )

    if body.posture == POSTURE_WRITES and not _rendered_writes_enabled(
        request.app.state.config_path
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "writes_disabled",
                "message": (
                    "This deployment arms writes for no control target: "
                    "control_system.connector.<type>.writes_enabled is off for every "
                    "connector type, as is the deployment-wide "
                    "control_system.writes_enabled they inherit from. No session can "
                    "step out of the sandbox until one target is armed."
                ),
            },
        )

    store = _session_postures(request.app)
    store[session_id] = body.posture
    _persist_postures(request.app, store)
    terminated = await _terminate_for_respawn(request.app, session_id)
    logger.info(
        "Session %s posture set to %s; terminated for respawn on: %s",
        session_id,
        body.posture,
        ", ".join(terminated) if terminated else "no live child",
    )

    return {"status": "ok", "session_id": session_id, "posture": body.posture}


@router.get("/api/terminal/posture")
async def get_terminal_posture(session_id: str, request: Request):
    """Report one session's posture and what the render permits.

    The single truth the terminal-card badge reads, and it answers two
    different questions because the badge has to show both:

    * ``posture`` — what the store holds for this session, defaulting to
      ``writes``. The store records only a *deviation*: ``_build_extra_env``
      adds ``OSPREY_EXECUTION_MODE`` for a sandboxed session and nothing at all
      otherwise, so "no entry" means the session runs the render's baseline and
      ``writes`` is the honest name for it.
    * ``rendered_writes_enabled`` — whether the render arms writes for *any*
      control target, over the per-type
      ``control_system.connector.<type>.writes_enabled`` and the
      deployment-wide ``control_system.writes_enabled`` it inherits from. This
      is what keeps the default reading honest on a writes-off deployment: the
      posture is ``writes`` and the effective write capability is still nil,
      because the render, not the toggle, is the binding constraint there. The
      badge says so rather than implying the session can write.
    * ``session_target`` / ``target_writes_enabled`` / ``target_source`` — the
      control target this session is pointed at and whether *that* target is
      armed (see :func:`_session_target_posture`). ``rendered_writes_enabled``
      is a union and cannot answer this: on a render that arms the virtual
      accelerator alone it is ``True`` for a session sitting on a live machine
      every write to which is refused. The POST contract is unchanged — stepping
      out of the sandbox stays keyed on the union, because an operator may
      legitimately leave the sandbox before switching to the target they intend
      to write on.

    The last two answers cost a config read and a walk of the process table, so
    they are computed in a worker thread (:func:`_posture_render_facts` via
    ``run_in_threadpool``). On a platform without ``/proc`` the walk forks
    ``ps``; doing that on the event loop would let one wedged process table
    stall every other request this server is serving, and the badge polls this
    route every few seconds per open card.

    Unlike POST, an id that names no session on disk is **not** a 409. The badge
    renders with the terminal card, which can be before the first prompt has
    written a session file, and refusing there would blank the one surface that
    tells the operator what the render permits. Answering costs nothing: a read
    grants nothing, stores nothing, and reports exactly the posture that
    session will spawn under — for a chat key just as for a PTY one. The id is
    still shape-checked with the same closed grammar POST uses, so the two
    routes keep one error contract.
    """
    _require_session_uuid(session_id)

    posture = _session_postures(request.app).get(session_id, POSTURE_WRITES)
    config_path = request.app.state.config_path
    # The PTY is how the session is found in the process table; a card that has
    # not started one yet (a chat key never does) simply gets the baseline.
    session = request.app.state.pty_registry.get_session(session_id)
    facts = await run_in_threadpool(
        _posture_render_facts, config_path, session.pid if session else None
    )
    return {"session_id": session_id, "posture": posture, **facts}


@router.post("/api/terminal/logout")
async def logout_terminal(request: Request):
    """Terminate the user's warm PTY (and operator) session(s) on logout.

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

    return {"status": "ok", "message": "Logged out — terminal session terminated"}


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
