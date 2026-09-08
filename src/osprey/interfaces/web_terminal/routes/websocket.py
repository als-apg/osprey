"""WebSocket routes for terminal PTY and operator (Agent SDK) sessions."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

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
from osprey.interfaces.web_terminal import session_handoff
from osprey.interfaces.web_terminal.control_context_owner import (
    ContextOwnedElsewhere,
    ContextOwnerError,
    Mutation,
    owned_elsewhere_message,
    terminal_identity,
)
from osprey.interfaces.web_terminal.operator_session import (
    POSTURE_SESSION_ENV,
    POSTURE_SOURCE_ENV,
    POSTURE_SOURCE_LIVE,
    POSTURE_SOURCE_SPAWN,
    build_operator_child_env,
    resolve_agent_data_root,
)
from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery
from osprey.interfaces.web_terminal.session_key import is_posture_key
from osprey_connectors import control_context, posture_store
from osprey_connectors.control_system.base import is_readonly_run

if TYPE_CHECKING:
    from osprey.interfaces.web_terminal.pty_manager import PtySession
    from osprey.interfaces.web_terminal.session_handoff import (
        AcquireResult,
        SpawnCallback,
        SpawnRequest,
    )

logger = logging.getLogger(__name__)

router = APIRouter()

# The loose shape check the resume path (``switch_session``) applies to ids
# Claude itself wrote: any 36 characters drawn from ``[a-f0-9-]``, which is
# fine for "does this look like a session file stem" and much too wide for a
# key that is written to a store on disk and later decides a child process's
# execution mode. The posture surface's *closed* key grammar is
# :func:`~osprey.interfaces.web_terminal.session_key.is_posture_key`.
_UUID_RE = re.compile(r"^[a-f0-9-]{36}$")

# ── Per-target runtime posture ───────────────────────────────────────────────
#
# The posture is the operator's per-target sandbox toggle: narrow one control
# target to ``sandbox`` and leave the others alone. It is deliberately *not* a
# config edit — config is a build-time input that reaches the agent only
# through a re-render — and it is deployment-wide, because so is the control
# target it narrows: one deployment, one control context, one posture.
#
# It lives in the ``posture`` field of the control-context record. Its grammar
# is :mod:`osprey_connectors.posture_store`'s, which is what every reader in
# the connector chain decodes with, and the values below are that module's. A
# narrowing this server spelled differently would be one the operator can see
# and the machine cannot.
POSTURE_SANDBOX = posture_store.POSTURE_SANDBOX
POSTURE_WRITES = posture_store.POSTURE_WRITES


class PostureRequest(BaseModel):
    """Body of ``POST /api/terminal/posture``.

    ``posture`` is a ``Literal`` so an unknown value is rejected by request
    validation with a 422 naming the field, before any handler code runs — the
    value decides whether writes to a target are refused, and a silent coercion
    to some default would be the worst possible failure here.

    ``target`` names one configured control target, or the literal
    :data:`ALL_TARGETS` for the popover's ``[ Sandbox everything ]`` gesture.
    It is a plain string rather than a ``Literal`` for the reason given on
    :class:`TargetRequest`: which targets exist is a property of the rendered
    deployment, not of this build's vocabulary.
    """

    session_id: str | None = None
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

    session_id: str | None = None
    target: str


def _require_session_uuid(session_id: str | None) -> None:
    """Refuse *session_id* unless it is a canonical, bare session UUID.

    One implementation for the three control-gesture routes, so they cannot
    drift on the status, the error slug or the sentence. The grammar, and what
    ``None`` means on this surface, are
    :func:`~osprey.interfaces.web_terminal.session_key.is_posture_key`'s: the
    id is optional here and is validated only when it is sent.

    Raises:
        HTTPException: 400 ``invalid_session_id`` when a session id was sent
            and its shape does not match.
    """
    if session_id is None:
        return
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
    pool key and the audit session id are the same string. Membership is read
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


def _record_available() -> bool:
    """Whether the control-context record has a location at all.

    :func:`~osprey_connectors.control_context.record_path` is the ONE path rule
    (env ``OSPREY_AGENT_DATA_ROOT``, else ``resolve_shared_data_root()``), and
    it answers ``None`` rather than raising when neither resolves. A deployment
    with no location for its record is ``store_available: false`` on the GET
    and the 503 on a gesture: there is nowhere to record a context the agent
    would read back.
    """
    return control_context.record_path() is not None


def _recorded_posture() -> dict[str, str]:
    """The deployment's per-target narrowings — ``{target: "sandbox"}``.

    The record's ``posture`` field, which is the whole of what this deployment
    has narrowed: there is one control context per deployment, and no
    per-session posture store behind it. A target that narrows nothing is
    ABSENT from the map — absence is how this field spells ``writes``, and a
    stored ``"writes"`` would be a second spelling of it.

    Every way of not knowing answers ``{}``, and that grants nothing: a
    narrowing can only refuse, so failing to read one leaves whatever the
    deployment ceiling already decided.
    """
    record = control_context.read_record()
    return {} if record is None else dict(record.posture)


@dataclass(frozen=True)
class _ContextState:
    """What ``app.state`` says this terminal may do with the control context.

    Published by the owner task each tick (see
    :mod:`osprey.interfaces.web_terminal.control_context_owner`) and read on
    the request task, because both attributes are plain values. It is a
    **render hint and a fast refusal, not the guard**: the snapshot can be one
    tick stale, and a write that races a takeover is refused inside the
    mutation primitive with :class:`ContextOwnedElsewhere`, which carries the
    owner the record actually names.

    Attributes:
        owner: The mutation primitive, or ``None`` when no tick has yet got far
            enough to know the record is reachable. ``None`` is the 503: this
            terminal owns nothing it could write.
        follows: The web terminal this one is behind, or ``None`` when this
            terminal is the owner.
    """

    owner: Any
    follows: Any


def _context_state(app: Any) -> _ContextState:
    """The owner task's two ``app.state`` attributes. Never raises."""
    return _ContextState(
        owner=getattr(app.state, "control_context_owner", None),
        follows=getattr(app.state, "control_context_follows", None),
    )


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


# ── Reading files another process wrote ──────────────────────────────────────
#
# Two coercions the surfaces on this router share, spelled once because both
# answer a question about a file this process does not own: what a pid field
# holds, and whether the file has been rewritten since it was last read.


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
    disappeared" at the same time. The inode is in the tuple because the render
    this module memoizes is replaced through a rename, and a replacement can
    land with the same size and, on a coarse clock, the same mtime.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size, stat.st_ino)


def _read_effort_level(config_path: Path | None) -> str | None:
    """Read claude_code.effort from config.yml."""
    if not config_path or not Path(config_path).exists():
        return None
    try:
        config = yaml.safe_load(Path(config_path).read_text()) or {}
        return config.get("claude_code", {}).get("effort")
    except Exception:
        return None


#: What ``claude --resume <id>`` prints before exiting 1 when no transcript
#: for that id exists. The pre-spawn check in :func:`_transcript_missing` is
#: meant to keep such a child from ever being spawned; this is the second net,
#: for a transcript that vanished after the check or one the CLI refuses to
#: load.
NO_CONVERSATION_MARKER = b"No conversation found with session ID"

#: How much of a ``--resume`` child's output is searched for the marker. The
#: verdict is the first thing the CLI prints, so anything beyond the first
#: few kilobytes is a session that resumed and is now doing real work.
_NO_CONVERSATION_SCAN_LIMIT = 16 * 1024


def _transcript_missing(app, registry, discovery: SessionDiscovery, session_id: str) -> bool:
    """Whether resuming *session_id* would spawn a child with nothing to resume.

    The question is whether the key names a session at all, and three things
    can answer yes. A pooled PTY that is still alive IS the session — a
    terminal that was opened and never prompted has one, and no ``.jsonl``
    yet, because Claude Code writes the transcript on the first prompt. A key
    the chat pool answers to (:func:`_chat_pool_answers_to`, so a creation
    still inside ``start()`` counts) is equally a session: both views are
    windows onto one key, and the conversation the operator started in Simple
    is the one they are asking for here. Only with none of those is the
    transcript directory the authority: ``--resume`` on an id with no file
    there prints :data:`NO_CONVERSATION_MARKER` and exits, which is the dead
    PTY the caller refuses to hand the operator.
    """
    existing = registry.get_session(session_id)
    if existing is not None and existing.is_alive:
        return False
    if _chat_pool_answers_to(app, session_id):
        return False
    return session_id not in discovery.snapshot_session_ids()


def _transcript_missing_frame(session_id: str, code: int | None = None) -> str:
    frame: dict[str, Any] = {"type": "transcript_missing", "session_id": session_id}
    if code is not None:
        frame["code"] = code
    return json.dumps(frame)


async def _run_output_loop(
    session,
    websocket: WebSocket,
    stop_event: asyncio.Event,
    *,
    resume_id: str | None = None,
) -> None:
    """Forward PTY bytes to the WebSocket until stopped or process exits.

    ``resume_id`` names the id a ``--resume`` child spawned by this handler
    was asked for. Its early output is then watched for
    :data:`NO_CONVERSATION_MARKER`: a child that prints it and exits is
    reported as ``transcript_missing`` rather than a bare ``exit``, so the
    client renders the same state the pre-spawn refusal produces instead of
    a process-exited line the operator cannot act on.
    """
    no_conversation = False
    scanned = bytearray()
    try:
        async for data in session.read_output():
            if stop_event.is_set():
                return
            if resume_id is not None and not no_conversation:
                if len(scanned) < _NO_CONVERSATION_SCAN_LIMIT:
                    scanned.extend(data)
                    no_conversation = NO_CONVERSATION_MARKER in scanned
            await websocket.send_bytes(data)
    except Exception:
        pass
    finally:
        if not stop_event.is_set():
            code = session.exit_code
            if no_conversation and resume_id is not None:
                frame = _transcript_missing_frame(resume_id, code)
            else:
                frame = json.dumps({"type": "exit", "code": code})
            try:
                await websocket.send_text(frame)
            except Exception:
                pass


def _build_extra_env(
    websocket: WebSocket,
    claude_session_id: str | None,
    telemetry_session_id: str | None = None,
) -> dict[str, str]:
    """Build the extra environment dict for PTY sessions.

    ``telemetry_session_id`` is the session UUID this terminal's ``claude`` is
    forced onto (via ``--session-id``); it is handed to the workspace
    provenance_locator tool so a filed issue can point back to this session's
    telemetry.

    ``OSPREY_SESSION_ID`` is stamped from the **pool key** — the same
    ``claude_session_id or telemetry_session_id`` the handler keys the pool on
    — so it is set on every spawn this function serves, the brand-new session
    included. It scopes the child's execution root
    (:func:`osprey_connectors.workspace.resolve_agent_data_root` appends
    ``sessions/<key>``) and tags the artifacts it files. One key, one root: a
    conversation is one session whichever view is showing it, so the chat
    child spawned under that key
    (:func:`~osprey.interfaces.web_terminal.operator_session.build_operator_child_env`)
    works out of the same directory as this one. The name is in
    :data:`~osprey.interfaces.web_terminal.pty_manager.POOL_FINGERPRINT_EXCLUDED_ENV`,
    so stamping it can never make a reattach respawn a live child.

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
    session_key = claude_session_id or telemetry_session_id
    if session_key:
        extra_env["OSPREY_SESSION_ID"] = session_key
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
    # **No execution mode is stamped here.** The write posture is per target and
    # lives in the control-context record; every write-time gate reads it there,
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
    if session_key:
        extra_env[POSTURE_SOURCE_ENV] = POSTURE_SOURCE_LIVE
        extra_env[POSTURE_SESSION_ENV] = session_key
        extra_env[OSPREY_AGENT_DATA_ROOT] = resolve_agent_data_root(websocket.app)
    return extra_env


#: Query values read as "yes" on ``/ws/terminal?interrupt=``.
_TRUTHY_QUERY_VALUES = frozenset({"1", "true", "yes", "on"})


def _query_flag(websocket: WebSocket, name: str) -> bool:
    """Whether query parameter *name* is set to a truthy value."""
    return websocket.query_params.get(name, "").strip().lower() in _TRUTHY_QUERY_VALUES


class _TerminalChannel:
    """One terminal socket as the hand-off door sees it, and its single reader.

    ``token`` is this handler's identity for everything that identifies a
    connection: the channel a pending acquire is registered under, the owner
    token the PTY is attached with, and the key its 4409 closer is filed
    under in :attr:`~...session_handoff.HandoffState.closers`. One per
    socket, kept across a ``switch_session``: the registry checks it before
    letting a detach release a key, so a handler that has been displaced
    cannot clear the attachment its successor holds.

    ``closed`` is set the moment the socket's ``websocket.disconnect`` is
    read — by :meth:`receive` in the main loop or by the side reader
    :meth:`acquire` runs — and by the closer when phase (c) displaces this
    handler. The token's closed probe reads it, so a wait inside the door
    ends as :class:`~...session_handoff.ChannelClosed` instead of running on
    for a client that is gone.

    Only one ``receive`` may be outstanding on a socket. During an acquire
    the handler is awaiting the door, so the side reader is that one reader:
    it records resizes — the size a spawn is started at, or a reused PTY is
    resized to — drops keystrokes, which have no PTY to go to yet, and keeps
    every other control message for the main loop, which reads
    :attr:`deferred` before it reads the socket again.

    The socket read itself is one task that outlives whichever reader awaits
    it. The side reader is cancelled the moment the door returns, and a
    frame the transport hands over in that same loop turn is consumed by the
    cancelled task on transports that deliver straight to a waiting receiver
    (Starlette's test client does; the ASGI contract promises nothing either
    way). Cancelling the awaiter leaves the read in flight, so the next
    :meth:`receive` collects that frame instead of losing it.
    """

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.closed = asyncio.Event()
        self.token = session_handoff.ChannelToken(self.closed.is_set)
        self.rows = 24
        self.cols = 80
        self.deferred: list[Mapping[str, Any]] = []
        self._read: asyncio.Task[Any] | None = None

    async def receive(self) -> Mapping[str, Any]:
        """The next message for the main loop: a deferred control message first."""
        if self.deferred:
            return self.deferred.pop(0)
        return await self._read_socket()

    async def _read_socket(self) -> Mapping[str, Any]:
        """The next message off the socket; a disconnect closes the channel."""
        if self._read is None:
            self._read = asyncio.ensure_future(self.websocket.receive())
        try:
            message: Mapping[str, Any] = await asyncio.shield(self._read)
        except asyncio.CancelledError:
            raise
        except BaseException:
            self._read = None
            raise
        self._read = None
        if message.get("type") == "websocket.disconnect":
            self.closed.set()
        return message

    def release(self) -> None:
        """Cancel a read left in flight; the handler is done with the socket."""
        if self._read is not None:
            self._read.cancel()
            self._read = None

    def note_resize(self, msg: Any) -> bool:
        """Record a ``resize`` control message; False for anything else.

        A malformed resize is recorded as nothing and still answered True:
        it was a control message, not keystrokes for the PTY.
        """
        if not isinstance(msg, dict) or msg.get("type") != "resize":
            return False
        try:
            rows, cols = int(msg["rows"]), int(msg["cols"])
        except (KeyError, TypeError, ValueError):
            return True
        self.rows, self.cols = rows, cols
        return True

    async def send_text(self, text: str) -> None:
        """Send a control frame, tolerating a socket that is already gone."""
        try:
            await self.websocket.send_text(text)
        except Exception:
            pass

    async def send_json(self, frame: dict[str, Any]) -> None:
        await self.send_text(json.dumps(frame))

    async def close(self, code: int | None = None) -> None:
        """Close the socket, tolerating one that is already closed."""
        try:
            if code is None:
                await self.websocket.close()
            else:
                await self.websocket.close(code=code)
        except Exception:
            pass

    async def acquire(
        self, app: Any, key: str, *, interrupt: bool, spawn: SpawnCallback
    ) -> AcquireResult:
        """Take *key* for the Expert surface while reading the socket.

        The acquire runs in the handler's own task — a cancellation delivered
        while phase (c) runs is then honoured only once the phase is done,
        which is what lets the handler's teardown detach unconditionally.
        The side reader is cancelled and reaped before this returns, so the
        main loop's own ``receive`` never overlaps it.
        """
        reader = asyncio.ensure_future(self._read_while_acquiring())
        try:
            return await session_handoff.acquire_surface(
                app,
                key,
                session_handoff.SURFACE_EXPERT,
                self.token,
                interrupt=interrupt,
                spawn=spawn,
            )
        finally:
            reader.cancel()
            with suppress(asyncio.CancelledError):
                await reader

    async def _read_while_acquiring(self) -> None:
        while True:
            try:
                message = await self._read_socket()
            except Exception:
                self.closed.set()
                return
            if message.get("type") == "websocket.disconnect":
                return
            text = message.get("text")
            if not text:
                # Keystrokes are dropped here; the one frame the surviving
                # socket read holds when the door returns is read by the main
                # loop instead and reaches the new PTY.
                continue
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and not self.note_resize(msg):
                self.deferred.append(message)


async def _open_surface(
    channel: _TerminalChannel,
    app: Any,
    key: str,
    *,
    interrupt: bool,
    spawn: SpawnCallback,
) -> AcquireResult | None:
    """Take *key* for this terminal, answering the client for every way that can end.

    Returns the result on success. Returns None once the client has been
    answered — a refusal closes the socket with the refusal's close code
    (4409 attached elsewhere, 4503 the outgoing process survived its kill,
    which the client offers a retry for); a hand-off error is an ``error``
    frame and a close; a channel that closed during the wait gets nothing.
    The ``handoff_pending`` frame goes out first whenever the chat surface
    holds the key, which is the one case the door waits on a foreign entry
    for a terminal; the client shows the transitional state until
    ``session_info`` (or ``error``) replaces it.
    """
    if _chat_pool_answers_to(app, key):
        await channel.send_json({"type": "handoff_pending"})
    try:
        return await channel.acquire(app, key, interrupt=interrupt, spawn=spawn)
    except session_handoff.HandoffRefused as refused:
        logger.info("Refusing the terminal session %s: %s", key, refused)
        await channel.close(refused.ws_close_code or session_handoff.WS_CLOSE_SESSION_ATTACHED)
        return None
    except session_handoff.ChannelClosed:
        logger.info("Terminal connection for session %s closed while waiting", key)
        return None
    except session_handoff.HandoffError as error:
        logger.warning("Hand-off to the terminal for session %s failed: %s", key, error)
        await channel.send_json({"type": "error", "message": str(error)})
        await channel.close()
        return None


def _write_to_pty(session: PtySession, data: bytes) -> None:
    """Forward keystrokes; a PTY whose child has gone swallows them."""
    try:
        session.write_input(data)
    except OSError:
        logger.debug("Dropped terminal input: the PTY is closed", exc_info=True)


def _resize_pty(session: PtySession, rows: int, cols: int) -> None:
    try:
        session.resize(rows, cols)
    except OSError:
        logger.debug("Could not resize the PTY", exc_info=True)


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
    - Server -> Client JSON: {"type": "handoff_pending"}
    - Server -> Client JSON: {"type": "transcript_missing", "session_id": UUID, "code"?: N}
    - Server -> Client JSON: {"type": "error", "message": str}

    Query: ``session_id`` and ``mode=resume`` name the session key to
    resume; without them a new key is minted. ``interrupt=1`` on a resume
    cuts short a turn the chat surface is running on that key instead of
    waiting for it.

    Every PTY this handler serves comes through
    :func:`~osprey.interfaces.web_terminal.session_handoff.acquire_surface`:
    the key is one session whichever view shows it, so taking it here hands
    the conversation off from a chat that holds it (``handoff_pending``
    while that finishes), takes it over from an older terminal on the same
    key (closed with 4409), reuses the pooled PTY, or spawns one resuming
    the key's current transcript. The spawn callback below is the only
    place the registry's blocking create path is called from.

    ``transcript_missing`` answers a resume — the ``mode=resume`` connect or a
    ``switch_session`` — of an id no surface holds and no transcript on disk
    names.
    Nothing is spawned for it: on the connect path the socket is then closed,
    on the switch path the current session stays attached. The same frame,
    with the exit ``code``, replaces ``exit`` when a ``--resume`` child prints
    that it found no such conversation and quits.
    """
    await websocket.accept()

    app = websocket.app
    registry = app.state.pty_registry
    base_shell_command = app.state.shell_command
    discovery = SessionDiscovery(app.state.project_cwd)

    # Parse session params from query string
    req_session_id = websocket.query_params.get("session_id")
    mode = websocket.query_params.get("mode", "new")
    interrupt = mode == "resume" and _query_flag(websocket, "interrupt")

    effort = _read_effort_level(app.state.config_path)

    # Pool key: the requested id for resumes, a forced id for new sessions.
    # A new session's id is dictated on the command line by the spawn below
    # (``--session-id``), never guessed, so the pool is keyed by the real
    # session id from the first moment and needs no later rekey. The forced
    # id is also what the workspace provenance_locator tool hands back (via
    # OSPREY_TELEMETRY_SESSION_ID) and what the OTEL emitter tags records
    # with as session.id, so a filed issue's provenance pointer resolves.
    if mode == "resume" and req_session_id:
        current_key: str = req_session_id
    else:
        current_key = str(uuid.uuid4())

    channel = _TerminalChannel(websocket)
    token = channel.token
    state = session_handoff.get_state(app)

    # The resume boundary. ``--resume`` on an id with no transcript exits at
    # once with "No conversation found", and a PTY that dies on attach is a
    # terminal the operator cannot use. The server knows before spawning, so
    # it says so and closes; the client renders the state and offers a fresh
    # start (terminal.js). Deliberately not a silent fresh session: the
    # operator picked a chat, and starting a different one under them hides
    # that it is gone.
    if (
        mode == "resume"
        and req_session_id
        and _transcript_missing(app, registry, discovery, req_session_id)
    ):
        logger.info(
            "Refusing to resume %s: no live PTY, no chat session and no transcript",
            req_session_id,
        )
        await channel.send_text(_transcript_missing_frame(req_session_id))
        await channel.close()
        return

    async def spawn(request: SpawnRequest) -> PtySession:
        # base_shell_command is list[str] (set by app.lifespan), so unpack
        # with [*base, ...] — nesting would break PtySession's exec (issue
        # #218). The door decided what to resume: the key's current
        # transcript when one is on disk, else a fresh session under the key.
        if request.resume_id:
            command: list[str] = [*base_shell_command, "--resume", request.resume_id]
        else:
            command = [*base_shell_command, "--session-id", request.key]
        if effort:
            command.extend(["--effort", effort])
        extra_env = _build_extra_env(websocket, request.key, request.key)
        # A full pool evicts its oldest background session first, and the
        # registry's own eviction kills it on the calling thread. This runs
        # on the event loop, under the key's hand-off lock, so the victim is
        # taken out here and killed off the loop instead.
        victim = registry.pop_lru_victim()
        if victim is not None:
            await asyncio.to_thread(victim.terminate)
        spawned: PtySession
        spawned, _ = registry.get_or_create_session(
            request.key,
            command,
            rows=channel.rows,
            cols=channel.cols,
            extra_env=extra_env if extra_env else None,
            cwd=app.state.project_cwd,
        )
        return spawned

    async def close_displaced() -> None:
        # A newer terminal on the same key took the PTY over. This handler's
        # output loop is stopped first: two readers on one PTY descriptor
        # split the child's output between them, and the close handshake the
        # main loop waits for is a round trip away (longer for a client that
        # is gone). The main loop then stops writing to a PTY that is no
        # longer this handler's. Reads the loop bindings as they are now.
        stop_event.set()
        if output_task is not None:
            output_task.cancel()
        channel.closed.set()
        await channel.close(session_handoff.WS_CLOSE_SESSION_ATTACHED)

    state.closers[token] = close_displaced

    # ``session`` is the PTY this handler holds and ``session_key`` the key it
    # holds it under; ``current_key`` is the key under acquisition, which
    # differs from ``session_key`` only while a switch is in flight.
    session: PtySession | None = None
    session_key = current_key
    stop_event = asyncio.Event()
    output_task: asyncio.Task[None] | None = None
    try:
        result = await _open_surface(channel, app, current_key, interrupt=interrupt, spawn=spawn)
        if result is None or channel.closed.is_set():
            return
        session = cast("PtySession", result.session)
        # Sized after the door whichever path filled the key: a resize the
        # side reader recorded after the spawn read the size is applied here
        # (an unchanged size is a no-op), and a reused PTY is not resized by
        # the door at all.
        _resize_pty(session, channel.rows, channel.cols)

        # Confirm the id, whichever path filled the key. A new session's id
        # is the one the spawn put on the CLI's command line; a resume's is
        # either a warm PTY (the session itself) or an id whose transcript
        # was on disk a moment ago — the boundary above refused everything
        # else. There is nothing to wait for and nothing to race.
        await channel.send_json({"type": "session_info", "session_id": current_key})

        # Start output forwarding. A ``--resume`` child spawned for this
        # handler is watched for the CLI's own "no such conversation" verdict.
        output_task = asyncio.create_task(
            _run_output_loop(session, websocket, stop_event, resume_id=result.resume_id)
        )

        while True:
            message = await channel.receive()
            if channel.closed.is_set():
                break

            if "text" in message:
                text = message["text"]
                try:
                    msg = json.loads(text)
                except (json.JSONDecodeError, KeyError):
                    msg = None

                if isinstance(msg, dict):
                    if channel.note_resize(msg):
                        logger.debug("PTY resize: %dx%d", channel.cols, channel.rows)
                        _resize_pty(session, channel.rows, channel.cols)
                        continue

                    if msg.get("type") == "switch_session":
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

                        if _transcript_missing(app, registry, discovery, target_id):
                            # Nothing to switch to; the operator stays on the
                            # session they are on. Same frame as the connect
                            # path, so the client renders one state.
                            logger.info(
                                "Refusing to switch to %s: no live PTY, no chat session "
                                "and no transcript",
                                target_id,
                            )
                            await websocket.send_text(_transcript_missing_frame(target_id))
                            continue

                        # 1. Stop the current output loop
                        stop_event.set()
                        output_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await output_task

                        # 2. Detach the current session (stays alive in the pool)
                        registry.detach_session(current_key, token)

                        # 3. Take the target through the door. From here the
                        #    handler holds nothing, so anything that stops it
                        #    short of the target ends the connection: the
                        #    client reconnects on its stored pointer. The
                        #    target is the key under acquisition from now on,
                        #    so the teardown detaches it even when a
                        #    cancellation lands after phase (c) attached it.
                        current_key = target_id
                        try:
                            result = await _open_surface(
                                channel, app, target_id, interrupt=False, spawn=spawn
                            )
                        except Exception:
                            logger.exception("Session switch to %s failed", target_id)
                            await channel.send_json(
                                {"type": "error", "message": "Session switch failed"}
                            )
                            await channel.close()
                            return
                        if result is None or channel.closed.is_set():
                            return
                        session = cast("PtySession", result.session)
                        session_key = target_id
                        _resize_pty(session, channel.rows, channel.cols)

                        # 4. Notify the client
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "session_switched",
                                    "session_id": target_id,
                                }
                            )
                        )

                        # 5. Start the new output loop
                        stop_event = asyncio.Event()
                        output_task = asyncio.create_task(
                            _run_output_loop(
                                session,
                                websocket,
                                stop_event,
                                resume_id=result.resume_id,
                            )
                        )
                        logger.info(
                            "Session switched to %s (spawned=%s)",
                            target_id,
                            result.spawned,
                        )
                        continue

                # Not a recognized JSON control message — treat as terminal input
                _write_to_pty(session, text.encode("utf-8"))

            elif "bytes" in message:
                _write_to_pty(session, message["bytes"])

    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        state.closers.pop(token, None)
        channel.release()
        stop_event.set()
        if output_task is not None:
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
        # is looking at. The detach carries this handler's attachment token
        # too, so the registry refuses it from its own side as well — the two
        # checks cover the same hazard through the two things that can be
        # compared: which session sits under the key, and who holds it.
        #
        # With no session under the key being acquired — the door refused,
        # or a cancellation landed after phase (c) had already attached this
        # token — the detach runs unconditionally; it is owner-checked and a
        # no-op for a token that never attached. A dead PTY is terminated
        # under the key it was held by.
        if (
            session is None
            or current_key != session_key
            or registry.get_session(current_key) is session
        ):
            registry.detach_session(current_key, token)
        if session is not None and not session.is_alive:
            registry.terminate_session_if_owner(session_key, session)


# ── Control-target gestures: audit, vocabulary, refusals ─────────────────────
#
# Both control POSTs below are *gestures an operator made*, not incidental
# state changes, so each files exactly one audit record naming the session it
# governs. ``HttpAuditMiddleware`` would otherwise file a bare
# ``http_mutation`` line with ``session: null`` — enough to know a request came
# in, useless for joining a toggle to the tool calls it governed. The dedup
# marker is what keeps the two from both writing: the innermost recorder owns
# the decision, and the middleware defers to it.

#: Audit subject for a gesture that moves the deployment's control target. The
#: same word the agent's own tool records under
#: (``osprey.mcp_server.http.TARGET_SWITCH_TOOL``), so an operator reading the
#: ledger sees one kind of event whichever surface asked for it.
AUDIT_SUBJECT_TARGET_SET = "control_target_set"

#: Audit subject for a gesture that narrows or widens one target's posture.
AUDIT_SUBJECT_POSTURE_SET = "session_posture_set"

#: Audit surface for a control gesture that names no session. The control
#: context is the deployment's, so a gesture against it needs no session to
#: act — and the surface that has none is the JupyterLab page's bar, which
#: posts the same bodies as the terminal chip from a page with no terminal in
#: it. Recorded as its own surface rather than as ``http_mutation`` with a null
#: session, so the ledger says where the gesture came from instead of only
#: that it could not be attributed.
LAB_MUTATION_SURFACE = "jupyter_lab"


def _record_control_gesture(
    session_key: str | None,
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

    ``session`` is the session key itself. A session is keyed on one string
    from the moment it is spawned — the same string its child exports as
    ``OSPREY_POSTURE_SESSION`` — so a gesture and the tool calls it governs
    join on one actor without anything having to be resolved back.

    A gesture that names **no** session is filed under
    :data:`LAB_MUTATION_SURFACE` with a null session. There is no key to guess
    at: the control context is the deployment's, and the surface that posts
    without one is the Lab page's bar. Recording the surface is what keeps the
    line joinable — to the deployment and to the moment, if not to a session.

    Never raises. A gesture whose record could not be written is still a
    gesture that happened; the trail degrades, the operation does not.
    """
    try:
        from osprey.audit.dedup import record_and_mark
        from osprey.audit.envelope import POSTURE_SOURCE_APP

        record_and_mark(
            decision=decision,
            reason=reason,
            surface=HTTP_MUTATION_SURFACE if session_key else LAB_MUTATION_SURFACE,
            posture=HTTP_MUTATION_POSTURE,
            posture_source=POSTURE_SOURCE_APP,
            session=session_key or None,
            subject=subject,
            detail=detail,
        )
    except Exception:  # noqa: BLE001 — the audit trail degrades; the gesture does not
        logger.warning("Could not record the %s gesture for audit", subject, exc_info=True)


def _refuse_gesture(
    session_key: str | None,
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
        session_key,
        subject=subject,
        decision=DECISION_REFUSED,
        reason=error,
        detail=detail,
    )
    return HTTPException(status_code=status_code, detail={"error": error, "message": message})


def _context_write_rung(
    app: Any,
    session_key: str | None,
    context: _ContextState,
    *,
    subject: str,
    detail: str,
) -> HTTPException | None:
    """The rung both gesture routes share: may this terminal write at all?

    Two refusals, in this order, and both are about *where* the gesture has to
    be made rather than about what it asks for:

    * **503** ``store_unavailable`` — no owner on ``app.state``. The record has
      no location, or no tick has yet reached one, so there is nowhere to
      record a control context the agent would read back.
    * **409** ``context_owned_elsewhere`` — this terminal is following another
      one. A deployment has a single control context and a single writer for
      it; the refusal names the owner's pid and port so the operator can open
      the terminal that does own it.

    Placed where the old per-session ``store_unavailable`` rung was, ahead of
    every judgement about the target: a terminal that may not write must not go
    on to tell the operator why their target was ineligible.
    """
    if context.owner is None:
        return _refuse_gesture(
            session_key,
            subject=subject,
            status_code=503,
            error="store_unavailable",
            # Two causes reach here — a record with no location, and a terminal
            # whose first owner tick has not landed yet — and this rung cannot
            # tell them apart. Naming the root would be a guess that is wrong
            # in the second case; the primitive's own ContextStoreUnavailable
            # does know, and says so.
            message=(
                "This terminal holds no control context to write yet, so nothing was changed. "
                "Try again in a moment."
            ),
            detail=detail,
        )
    if context.follows is not None:
        return _refuse_gesture(
            session_key,
            subject=subject,
            status_code=409,
            error="context_owned_elsewhere",
            message=owned_elsewhere_message(context.follows),
            detail=f"{detail} owner_pid={context.follows.pid} owner_port={context.follows.port}",
        )
    return None


def _context_error_refusal(
    app: Any,
    session_key: str | None,
    exc: ContextOwnerError,
    *,
    subject: str,
    detail: str,
) -> HTTPException:
    """The same two refusals, raised by the primitive instead of foreseen.

    ``app.state`` can be one tick stale, so a write may reach the record and
    find it owned by somebody else — or find that it cannot be written at all.
    The primitive raises then, carrying the owner the record actually names,
    and the operator reads the same words the fast rung would have given them.
    """
    owned = isinstance(exc, ContextOwnedElsewhere)
    return _refuse_gesture(
        session_key,
        subject=subject,
        status_code=409 if owned else 503,
        error=exc.error,
        message=exc.message,
        detail=detail,
    )


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


def _baseline_target(section: Any) -> str:
    """The control target this deployment's own config selects.

    ``live`` for a render nobody can classify, which is where every other
    predicate on this surface lands on an unreadable config: the baseline is
    what a switch is measured against, and guessing a simulator would make a
    deployment look further from its own machine than it is.
    """
    if section is _UNREADABLE_SECTION:
        return "live"
    try:
        from osprey_connectors.types import baseline_target

        return str(baseline_target(section))
    except Exception:  # noqa: BLE001 — a render we cannot classify is `live`
        logger.warning("Could not resolve the deployment's baseline control target")
        return "live"


def _kernel_name_resolver(app: Any) -> Any:
    """The sidecar query that turns a kernel id into its notebook path.

    ``None`` when the Jupyter panel is off or was retracted after its sidecar
    died — there is nothing to ask, and a fabricated URL would be a two-second
    timeout per refusal.

    What comes back is a **blocking HTTP call**. Its one caller is
    :func:`_notebook_names`, which runs it in the facts hop; nothing hands this
    callable onwards, because the only other place it could be invoked from is
    the mutation job, and that holds the record lock.
    """
    from osprey.interfaces.web_terminal.jupyter_sidecar import kernel_notebook_path
    from osprey.profiles.web_panels import JUPYTER_PANEL_ID
    from osprey.registry.web import panel_url_state_attr

    url = getattr(app.state, panel_url_state_attr(JUPYTER_PANEL_ID), None)
    if not url:
        return None
    headers = getattr(app.state, "panel_auth_headers", {}).get(JUPYTER_PANEL_ID)
    return partial(kernel_notebook_path, url, headers or {})


def _notebook_names(app: Any, markers: tuple[dict[str, Any], ...]) -> dict[str, str]:
    """``{kernel_id: notebook path}`` for the notebook kernels holding the target.

    BLOCKING, and resolved **here** rather than handed to the gate as a
    callable. The gate names a busy notebook kernel by the notebook it is
    running, and only this surface can answer that — the sidecar is the
    terminal's — but the gate is evaluated inside the record mutation, under
    the owner's lock. A two-second sidecar timeout taken there would stall the
    owner task's one-second tick, every posture toggle and every other switch
    queued behind it. So the network I/O happens in this hop and the gate is
    handed a pure lookup.

    A sidecar that cannot answer contributes no entry — no panel, a dead one, a
    403, an unknown kernel, a body in an unexpected shape, a resolver that
    raises — and the refusal falls back to ``notebook kernel <id[:8]>``, which
    is what the controls server's tool says for the same marker.
    """
    from osprey.mcp_server.control_system.target_eligibility import SURFACE_NOTEBOOK_KERNEL

    resolve = _kernel_name_resolver(app)
    if resolve is None:
        return {}

    names: dict[str, str] = {}
    for marker in markers:
        if marker.get("surface") != SURFACE_NOTEBOOK_KERNEL:
            continue
        kernel_id = marker.get("kernel_id")
        if not isinstance(kernel_id, str) or not kernel_id.strip() or kernel_id in names:
            continue
        try:
            path = resolve(kernel_id)
        except Exception:  # noqa: BLE001 — a resolver that raised answered nothing
            logger.warning("Could not resolve the notebook running kernel %s", kernel_id)
            continue
        if isinstance(path, str) and path.strip():
            names[kernel_id] = path
    return names


def _live_reports() -> tuple[Any, ...]:
    """Every running controls server's report. Never raises. BLOCKING.

    The gate resolves no pid of its own, so the liveness filter is the caller's
    job and it is done here: a dead server's stale ``reached`` row would
    otherwise allow a switch to a machine nobody has reached in hours.

    The predicate is passed rather than left to default: one request asks
    liveness through one binding, so the fleet and the execution markers beside
    it cannot answer from two different views of the process table.
    """
    from osprey.mcp_server.control_system import target_state

    try:
        return tuple(control_context.live_reports(is_alive=target_state.is_process_alive))
    except Exception:  # noqa: BLE001 — an unreadable report directory reports no fleet
        logger.warning("Could not read the controls servers' reports", exc_info=True)
        return ()


def _live_executions() -> tuple[dict[str, Any], ...]:
    """Every live in-flight execution marker, oldest first. Never raises."""
    try:
        from osprey.mcp_server.control_system.target_state import in_flight_executions

        return tuple(in_flight_executions())
    except Exception:  # noqa: BLE001 — an unreadable marker directory reports none
        logger.warning("Could not read the execution markers", exc_info=True)
        return ()


@dataclass(frozen=True)
class _TargetRequestFacts:
    """Everything ``POST /api/terminal/target`` needs off the event loop.

    Gathered in ONE worker-thread hop: the render parse, the fleet's reports,
    the execution markers. Split across several hops they would straddle each
    other — a marker read before a run ended, beside a report read after it.

    What is deliberately NOT here is the record. The gate's verdict is decided
    against the record the answer is written into, inside the mutation the
    write happens in, because a verdict taken against an earlier read is a
    verdict about a different deployment state.
    """

    #: Target names this render configures; the vocabulary the 400 is keyed on.
    #: Spelled the same as ``_PostureRequestFacts.configured``: it is the same
    #: fact from the same function, and the two routes refuse an unknown target
    #: with the same sentence.
    configured: tuple[str, ...]
    #: The whole rendered config, which is what the switch gate reads.
    config: Any
    #: Its ``control_system:`` section, for the ceilings the gate is handed.
    section: Any
    #: The target this deployment's own config selects.
    baseline: str
    #: The live execution markers, oldest first.
    in_flight: tuple[dict[str, Any], ...]
    #: The live controls servers' reports.
    reports: tuple[Any, ...]
    #: ``{kernel_id: notebook path}`` for the notebook kernels among
    #: ``in_flight``, already asked of the sidecar. Resolved here rather than
    #: in the gate so the mutation job does no network I/O under the record
    #: lock; the gate is handed ``.get``, which opens nothing.
    kernel_names: dict[str, str]


def _target_request_facts(app: Any, config_path: Path | None) -> _TargetRequestFacts:
    """Read the render, the fleet and the execution markers. BLOCKING.

    Called through ``run_in_threadpool``: it parses ``config.yml``, globs the
    control-target directory and may ask the Jupyter sidecar over HTTP. None of
    it may happen on the event loop, where one slow shared volume would stall
    every request this server is serving, the terminal websocket included.
    """
    config = _rendered_config(config_path)
    section = _section_of(config)
    in_flight = _live_executions()
    return _TargetRequestFacts(
        configured=_configured_target_names(section),
        config=config,
        section=section,
        baseline=_baseline_target(section),
        in_flight=in_flight,
        reports=_live_reports(),
        kernel_names=_notebook_names(app, in_flight),
    )


#: The switch was neither applied nor refused: the fleet has not settled, so no
#: answer can be written without overwriting the one it is about to produce.
#: Not a :data:`~osprey_connectors.control_context.SWITCH_REFUSED` terminus —
#: the record is not touched at all.
SWITCH_IN_PROGRESS = "switch_in_progress"


@dataclass(frozen=True)
class _SwitchOutcome:
    """What the mutation decided, carried back out of the worker thread.

    Attributes:
        status: :data:`~osprey_connectors.control_context.SWITCH_APPLIED`,
            :data:`~osprey_connectors.control_context.SWITCH_REFUSED`, or
            :data:`SWITCH_IN_PROGRESS`.
        generation: The generation the record carries after the answer — the
            one the fleet reconciles to, and what the chip resolves its pending
            switch against.
        reason: The gate's machine-readable reason; ``""`` when applied.
        detail: What ``last_switch.detail`` carries — the gate's ``detail`` and
            nothing else, because the owner task and the agent's own tool write
            that same string into that same field, and a record whose terminus
            read differently depending on which surface answered would be two
            vocabularies in one place.
        message: What the operator reads in the response, which is the whole
            refusal: the gate's headline plus the suggestions that name the
            busy client and the remedy. Equal to *detail* unless the gate had
            more to say.
        blocking: The controls servers holding the deployment mid-swap, for
            :data:`SWITCH_IN_PROGRESS` and empty otherwise.
    """

    status: str
    generation: int
    reason: str = ""
    detail: str = ""
    message: str = ""
    blocking: tuple[int, ...] = ()


def _switch_mutation(
    record: Any,
    *,
    facts: _TargetRequestFacts,
    wanted: str,
    request_id: str,
    requested_at: str,
    requested_by: str,
) -> Mutation[_SwitchOutcome]:
    """Judge the switch against *record* and return what to store beside it.

    Runs inside :meth:`ControlContextOwner.mutate_record`'s worker thread, so
    the verdict is taken from the record the write lands on rather than from a
    read a hop earlier. The order is the switch tool's, deliberately — the two
    surfaces answer one gesture and must not answer it differently:

    1. **Already there.** ``wanted == record.target`` is a success and not
       ``already_active``: nothing is written, no generation is minted, and the
       chip resolves against the generation the fleet is already on. A mint for
       a switch that did not happen would refuse every write pinned to the old
       one, for nothing.
    2. **Not converged.** A live server is still applying the record's
       generation, so a terminus written now would overwrite the answer
       somebody is waiting on. Nothing is written; the pids are the refusal.
    3. **The gate.** :func:`~osprey.mcp_server.control_system.target_eligibility.evaluate_switch`,
       with ``current_target`` taken from the record. A refusal is a record
       write that moves neither target nor generation; an allowance moves both
       and mints the next generation.
    """
    from osprey.mcp_server.control_system import target_eligibility

    if wanted == record.target:
        return Mutation.unchanged(
            _SwitchOutcome(
                status=control_context.SWITCH_APPLIED,
                generation=record.generation,
                detail=control_context.unchanged_detail(wanted, record.generation),
            )
        )

    blocking = control_context.blocking_pids(record, facts.reports, None)
    if blocking:
        return Mutation.unchanged(
            _SwitchOutcome(
                status=SWITCH_IN_PROGRESS,
                generation=record.generation,
                reason=SWITCH_IN_PROGRESS,
                # No ``detail``: nothing is written, so the record has no
                # terminus to carry one.
                message=_switch_in_progress_message(blocking),
                blocking=blocking,
            )
        )

    verdict = target_eligibility.evaluate_switch(
        facts.config,
        wanted,
        current_target=record.target,
        baseline=facts.baseline,
        in_flight=facts.in_flight,
        reports=facts.reports,
        writes_enabled=target_eligibility.effective_writes_for_target(facts.section, wanted),
        # A pure lookup: the sidecar was asked in the facts hop, because this
        # call happens under the record lock.
        kernel_name=facts.kernel_names.get,
    )
    applied = verdict.allowed
    if applied:
        generation: int | None = record.generation + 1
        detail = control_context.applied_detail(wanted, record.generation + 1)
    else:
        generation = None
        detail = verdict.detail
    return Mutation(
        record=control_context.terminus(
            record,
            request_id=request_id,
            target=wanted,
            requested_at=requested_at,
            requested_by=requested_by,
            status=control_context.SWITCH_APPLIED if applied else control_context.SWITCH_REFUSED,
            reason=None if applied else str(verdict.reason or ""),
            detail=detail,
            generation=generation,
        ),
        result=_SwitchOutcome(
            status=control_context.SWITCH_APPLIED if applied else control_context.SWITCH_REFUSED,
            generation=record.generation if generation is None else generation,
            reason="" if applied else str(verdict.reason or ""),
            detail=detail,
            message=detail if applied else _verdict_message(verdict),
        ),
    )


def _verdict_message(verdict: Any) -> str:
    """The gate's refusal as one fact and one action, for the operator.

    The gate writes for two readers at once: its ``detail`` is the headline and
    its ``suggestions`` carry the attribution and the remedy. The agent's tool
    renders all of them; this surface renders the two that an operator can act
    on, because a popover reading three sentences that restate each other is
    the filler the copy rule exists to keep out.

    Two shapes, and which one applies is decided by the reason rather than by
    reading the strings:

    * **An execution in flight** says everything in its suggestions — which
      client holds the target (a notebook by name, where the sidecar could
      answer) and what to do about it. Its headline repeats both, and its own
      "wait or stop it" would stand beside the sharper "interrupt that kernel",
      so the headline is dropped and the suggestions are the message.
    * **Everything else** says it in the headline and offers at most a remedy
      beside it, so the message is the headline plus the last suggestion.

    A roster suggestion is never shown either way: those sentences send the
    reader to the target roster, and this surface IS the roster. They are
    recognised by the gate's own
    :data:`~osprey.mcp_server.control_system.target_eligibility.ROSTER_SUGGESTION_OPENING`
    rather than by a copy of the words here, so re-wording one there cannot
    quietly put it back in an operator's refusal.
    """
    from osprey.mcp_server.control_system.target_eligibility import (
        REASON_EXECUTION_IN_FLIGHT,
        ROSTER_SUGGESTION_OPENING,
    )

    lines = [
        text
        for text in (str(line).strip() for line in verdict.suggestions or ())
        if text and not text.startswith(ROSTER_SUGGESTION_OPENING)
    ]

    if verdict.reason == REASON_EXECUTION_IN_FLIGHT and lines:
        parts = lines
    else:
        parts = [str(verdict.detail or "").strip(), *lines[-1:]]

    text = " ".join(part for part in parts if part)
    if not text:
        return "The switch was refused."
    return text[:1].upper() + text[1:]


def _switch_in_progress_message(pids: tuple[int, ...]) -> str:
    """The refusal a deployment mid-swap earns, naming who is holding it.

    The pids are the point: they are what an operator acts on, and they are the
    same list the agent's own tool prints for the same refusal.
    """
    named = ", ".join(str(pid) for pid in pids) or "an unnamed server"
    return (
        f"A control-target switch is already in flight on pid {named}, so this one was not "
        "made. Wait for the chip to settle, then switch again."
    )


def _target_refusal(
    app: Any,
    session_id: str | None,
    body: TargetRequest,
    facts: _TargetRequestFacts,
    context: _ContextState,
) -> HTTPException | None:
    """The rungs decided before the record is opened — or ``None`` to write.

    Two of them, and neither is a judgement about the switch: an identifier
    this render does not know, and a terminal that may not write the control
    context at all. Everything else — read-only run, execution in flight,
    eligibility, reachability — is the gate's, and the gate runs against the
    record inside the mutation, where its verdict cannot be stale.
    """
    subject = AUDIT_SUBJECT_TARGET_SET
    detail = f"target={body.target}"

    if body.target not in facts.configured:
        return _refuse_gesture(
            session_id,
            subject=subject,
            status_code=400,
            error="unknown_target",
            message=_unknown_target_message(facts.configured),
            detail=detail,
        )

    return _context_write_rung(app, session_id, context, subject=subject, detail=detail)


@router.post("/api/terminal/target", status_code=202)
async def request_terminal_target(body: TargetRequest, request: Request):
    """Move this deployment's control target.

    **One control context per deployment, and one writer for it.** The record
    holds the target, the generation the fleet coordinates on and the terminus
    of the last switch; a web terminal owns it while it is running. So this
    route does not file desired state for somebody else to apply — it takes the
    record, runs the switch gate against it and writes the answer, both halves
    inside one mutation, so no reader can ever see a target that moved without
    a generation or a generation minted for a switch that was refused.

    What it does **not** do is wait for the fleet. Each controls server sees the
    new generation on its next reconciler pass and swaps its connector host
    then; the chip resolves the pending switch when every live server reports
    that generation. This route's job ends when the record does.

    The ladder, in order:

    * **400** — a session id outside the closed key grammar (only when one is
      sent; the gesture needs none), or a target this render does not
      configure. Both are identifiers, checked before anything is opened.
    * **503** ``store_unavailable`` — this terminal holds no control context to
      write. There is nowhere to record a switch the agent would read back.
    * **409** ``context_owned_elsewhere`` — another web terminal owns the
      context. The refusal names its pid and port; the switch is made there.
    * **409** ``switch_in_progress`` — a live controls server is still applying
      the generation the deployment is on. Answering now would overwrite the
      terminus that swap is about to produce, so nothing is written and the
      refusal names the pids holding it.
    * **409** — the gate's own refusal, in the gate's words: a read-only run,
      an execution in flight (naming the busy client), a target this render
      cannot select, one the fleet reports unreachable. A gate refusal IS
      recorded — ``last_switch`` carries it, and target and generation do not
      move — so the roster and the agent read the same outcome the operator
      just saw.

    A switch to the target the deployment is already on succeeds without
    minting a generation, which is what the agent's own tool answers too.
    """
    session_id = body.session_id
    _require_session_uuid(session_id)

    app = request.app
    subject = AUDIT_SUBJECT_TARGET_SET
    context = _context_state(app)
    facts: _TargetRequestFacts = await run_in_threadpool(
        _target_request_facts, app, app.state.config_path
    )

    refusal = _target_refusal(app, session_id, body, facts, context)
    if refusal is not None:
        raise refusal

    request_id = str(uuid.uuid4())
    detail = f"target={body.target} request_id={request_id}"
    try:
        outcome: _SwitchOutcome = await context.owner.mutate_record(
            partial(
                _switch_mutation,
                facts=facts,
                wanted=body.target,
                request_id=request_id,
                requested_at=datetime.now(UTC).isoformat(),
                requested_by=session_id or f"pid:{os.getpid()}",
            )
        )
    except ContextOwnerError as exc:
        raise _context_error_refusal(app, session_id, exc, subject=subject, detail=detail) from exc

    if outcome.status != control_context.SWITCH_APPLIED:
        raise _refuse_gesture(
            session_id,
            subject=subject,
            status_code=409,
            error=outcome.reason,
            message=outcome.message,
            detail=f"{detail} blocked_by={','.join(str(p) for p in outcome.blocking) or 'gate'}",
        )

    from osprey.audit.envelope import DECISION_ALLOWED

    _record_control_gesture(
        session_id,
        subject=subject,
        decision=DECISION_ALLOWED,
        reason="target_switch_requested",
        detail=f"{detail} generation={outcome.generation}",
    )
    logger.info(
        "Control target set to %s at generation %s (request %s)",
        body.target,
        outcome.generation,
        request_id,
    )
    return {
        "session_id": session_id,
        "target": body.target,
        "request_id": request_id,
        "generation": outcome.generation,
        "detail": outcome.detail,
    }


#: The popover's ``[ Sandbox everything ]`` gesture, spelled in the ``target``
#: field. Narrowing everything is one unambiguous act; there is deliberately no
#: matching "arm everything", because each target's ceiling is its own.
ALL_TARGETS = "all"


@dataclass(frozen=True)
class _PostureRequestFacts:
    """Everything ``POST /api/terminal/posture`` needs off the event loop.

    One worker-thread hop for the whole ladder: the render parse, the record
    read, the hypothetical narrowing derivation and — only when the request
    widens — the execution-marker sweep. Answers drawn from separate hops could
    straddle each other: a ceiling read from one render beside a narrowing
    verdict taken from the next.
    """

    #: Target names this render configures — the 400's vocabulary.
    configured: tuple[str, ...]
    #: The targets this request actually touches (``all`` expands here).
    wanted: tuple[str, ...]
    #: ``session_posture(section)``: the persona ceiling, per target.
    ceilings: dict[str, bool]
    #: The ``writes_enabled`` config key each wanted target's ceiling came from.
    writes_keys: dict[str, str]
    #: ``{target: why}`` for every target this request would CHANGE that cannot
    #: be narrowed. Keyed per target rather than reduced to a first refusal,
    #: because ``all`` narrows around one and a single target refuses on it.
    narrowing_refusals: dict[str, str]
    #: The first live execution marker, when the request widens; else ``None``.
    in_flight: dict[str, Any] | None
    #: The deployment's narrowings as this hop read them. The ladder is decided
    #: against these; the write recomputes from the record it lands on.
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
    """The oldest live execution marker, or ``None``. Never raises.

    The one the refusal names, and the one the switch gate would name for the
    same deployment: both read :func:`_live_executions`, which hands markers
    back oldest first.
    """
    running = _live_executions()
    return running[0] if running else None


def _posture_post_facts(
    app: Any, config_path: Path | None, target: str, posture: str
) -> _PostureRequestFacts:
    """Read everything the posture ladder decides on. BLOCKING.

    Called through ``run_in_threadpool``: it parses ``config.yml`` and reads
    the control-context record. Neither may happen on the event loop, where one
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
    # ``narrowing_refusal`` — which reads the config, never the record — would
    # otherwise let it refuse on behalf of a change nobody requested.
    current = _recorded_posture()
    changing = tuple(t for t in wanted if current.get(t) != POSTURE_SANDBOX)

    return _PostureRequestFacts(
        configured=configured,
        wanted=wanted,
        ceilings=_session_ceilings(section),
        writes_keys={t: _target_writes_key(section, t) for t in wanted},
        narrowing_refusals={} if widening else _narrowing_refusals(config, changing),
        in_flight=_first_live_execution() if widening else None,
        current=current,
    )


def _in_flight_message(marker: dict[str, Any]) -> str:
    """The refusal sentence the switch gate already says for a running run.

    Borrowed rather than reworded: an operator who has read it once in the
    agent's answer should read the same words in the popover. Only the message
    and the remedy are kept — the middle suggestion names the busy client, and
    :func:`~osprey.mcp_server.control_system.target_eligibility.busy_client`
    resolves it from the marker's own session and surface, which is a fact
    wherever it is read.
    """
    try:
        from osprey.mcp_server.control_system.target_eligibility import in_flight_detail

        message, suggestions, _details = in_flight_detail(marker, "")
        remedy = [suggestions[-1]] if suggestions else []
        return " ".join([message[:1].upper() + message[1:], *remedy])
    except Exception:  # noqa: BLE001 — the refusal stands even without the gate's words
        logger.warning("Could not build the in-flight refusal message", exc_info=True)
        running_on = str(marker.get("target") or "unknown")
        return (
            f"An execution in flight on target {running_on!r}; wait or stop it, then widen again."
        )


def _posture_refusal(
    app: Any,
    session_id: str | None,
    body: PostureRequest,
    facts: _PostureRequestFacts,
    context: _ContextState,
    gesture: str,
) -> HTTPException | None:
    """The posture ladder's refusals, in order — or ``None`` to let it through.

    Split out of :func:`set_terminal_posture` so the ladder reads as the
    ordered list of reasons it is, and the handler as the steps it performs
    around it: refuse, apply, record, answer. Everything here is decided from
    :class:`_PostureRequestFacts` and :class:`_ContextState`; the one refusal
    that fires BEFORE the threadpool hop stays in the handler, where it can be
    reached without paying for the facts. The rungs, in order and with their
    reasons, are documented on the route.

    Returns the exception rather than raising it, for the same reason
    :func:`_refuse_gesture` does: the audit record is filed here, and the
    control flow leaves at the caller's ``raise``.
    """
    subject = AUDIT_SUBJECT_POSTURE_SET
    widening = body.posture == POSTURE_WRITES

    if not facts.wanted or (body.target != ALL_TARGETS and body.target not in facts.configured):
        return _refuse_gesture(
            session_id,
            subject=subject,
            status_code=400,
            error="unknown_target",
            message=_unknown_target_message(facts.configured),
            detail=gesture,
        )

    context_refusal = _context_write_rung(app, session_id, context, subject=subject, detail=gesture)
    if context_refusal is not None:
        return context_refusal

    if widening:
        unarmed = [t for t in facts.wanted if not facts.ceilings.get(t, False)]
        if unarmed:
            blocked = unarmed[0]
            return _refuse_gesture(
                session_id,
                subject=subject,
                status_code=403,
                error="writes_disabled",
                message=(
                    f"This deployment does not arm writes for target {blocked!r}: "
                    f"{facts.writes_keys.get(blocked, '')} is off. A posture "
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
            session_id,
            subject=subject,
            status_code=409,
            error="selected_role_missing",
            message=why,
            detail=f"{gesture} blocked_by={blocked}",
        )

    if facts.in_flight is not None:
        return _refuse_gesture(
            session_id,
            subject=subject,
            status_code=409,
            error="execution_in_flight",
            message=_in_flight_message(facts.in_flight),
            detail=f"{gesture} executor_pid={facts.in_flight.get('pid')}",
        )

    return None


def _posture_mutation(
    record: Any, *, applying: tuple[str, ...], widening: bool
) -> Mutation[dict[str, str]]:
    """Apply the gesture to *record*'s narrowings and return what is stored.

    Runs inside :meth:`ControlContextOwner.mutate_record`'s worker thread, so
    the entry is derived from the record the write lands on: two toggles
    arriving together are serialised by the mutation lock, and the second one
    computes from a record that already holds the first one's narrowing.

    An entry that narrows nothing REMOVES the key. Absence is how this field
    spells ``writes``, and a stored ``"writes"`` would be a second spelling of
    it — one the connector chain does not read.
    """
    posture = dict(record.posture)
    for target in applying:
        if widening:
            posture.pop(target, None)
        else:
            posture[target] = POSTURE_SANDBOX
    if posture == record.posture:
        return Mutation.unchanged(posture)
    return Mutation(record=replace(record, posture=posture), result=posture)


@router.post("/api/terminal/posture")
async def set_terminal_posture(body: PostureRequest, request: Request):
    """Narrow or widen one control target's write posture for this deployment.

    **Nothing is respawned.** The posture is read live from the control-context
    record by every write-time gate — the connector's reference monitor, the
    executor's clamp, the write hook — so a running agent obeys a narrowing on
    its very next write.

    **The record write is the commit point.** It happens inside the mutation
    primitive: the read, the derivation and the atomic replace are one unit,
    serialised against every other write to the record, so a toggle can never
    be computed from a record that a switch has since moved.

    **One posture per deployment, not one per session.** The narrowing is the
    operator's statement about a machine, and the machine is the deployment's.
    A ``session_id`` is therefore optional here: it names who made the gesture
    for the audit trail and decides nothing about what the gesture does.

    The ladder, in order:

    * **400** — a session id outside the closed key grammar (only when one is
      sent); a target this render does not configure; or :data:`ALL_TARGETS`
      with ``writes``.
    * **503** ``store_unavailable`` — this terminal holds no control context to
      write. The toggle was refused and nothing changed.
    * **409** ``context_owned_elsewhere`` — another web terminal owns the
      context. The refusal names its pid and port; the toggle is made there.
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

    context = _context_state(app)
    facts: _PostureRequestFacts = await run_in_threadpool(
        _posture_post_facts, app, app.state.config_path, body.target, body.posture
    )

    refusal = _posture_refusal(app, session_id, body, facts, context, gesture)
    if refusal is not None:
        raise refusal

    # Everything ``all`` could not narrow is reported rather than silently
    # dropped: the popover has to be able to say which machine stayed writable
    # and why, or "Sandbox everything" would be a claim the record does not back.
    skipped = [
        {"target": target, "reason": "selected_role_missing", "detail": why}
        for target, why in sorted(facts.narrowing_refusals.items())
    ]
    applying = tuple(t for t in facts.wanted if t not in facts.narrowing_refusals)

    try:
        stored = await context.owner.mutate_record(
            partial(_posture_mutation, applying=applying, widening=widening)
        )
    except ContextOwnerError as exc:
        raise _context_error_refusal(app, session_id, exc, subject=subject, detail=gesture) from exc

    from osprey.audit.envelope import DECISION_ALLOWED

    _record_control_gesture(
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
        "Posture %s on %s; narrowed targets: %s; skipped: %s",
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
# that answer for the target the deployment is on; its popover shows one row per
# configured target, and every row carries enough for the operator to act on it
# — the machine's name, where it points, whether anything is reaching it, the
# ceiling the persona rendered, the operator's own narrowing, and what a switch
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
#: read-time verdicts: ``unknown`` is the absence of a row (no live server has
#: probed that target) and ``stale`` is a row whose
#: ``probed_at`` has aged past the prober's own interval. The published three —
#: ``reached``, ``down``, ``not_applicable`` — pass through as measured.
REACH_UNKNOWN = "unknown"
REACH_STALE = "stale"


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
    selects under the target's EFFECTIVE posture: a target narrowed to
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


def _effective_writes(section: Any, target: str) -> bool:
    """Whether *target* may be written on this deployment, right now.

    ``ceiling ∧ not is_readonly_run() ∧ recorded posture ≠ sandbox`` — rule 3
    of the posture contract, and delegated to
    :func:`~osprey_connectors.posture_store.effective_writes` rather than
    restated. The contract has exactly two implementations (that one and the
    stdlib restatement the hooks carry); a third spelled out in a route is how a
    popover comes to show ``writes`` on a machine the connector refuses.

    No key is passed because there is none to pass: the narrowing lives in the
    deployment's control-context record, and that function reads it. A web
    server carries no ``OSPREY_POSTURE_SESSION`` stamp and needs none.

    An unreadable render answers ``False``, where every other predicate on this
    surface lands.
    """
    if section is _UNREADABLE_SECTION:
        return False
    try:
        return bool(posture_store.effective_writes(section, target))
    except Exception:  # noqa: BLE001 — an underivable posture is not a writable one
        logger.warning("Could not resolve the effective write posture for target %s", target)
        return False


def _row_selected_role(config: Any, target: str, writes_enabled: bool) -> str | None:
    """The gateway role a connector would select for *target* under this posture.

    The reachability collapse keys on it, so it is derived through
    :func:`~osprey.mcp_server.control_system.target_eligibility.derive_endpoints`
    — the function the connector-host child's own selection is verified against
    — with the recorded effective posture rather than the configured one. A
    role derived from config alone would name the write gateway for a target the
    operator has just narrowed, and the row would report the reachability of a
    gateway no connector on this deployment will open.

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
    config: Any, target: str, control_target: str, baseline: str, writes_enabled: bool
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
            config, target, control_target, baseline, writes_enabled=writes_enabled
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


def _stamp_epoch(stamp: Any) -> float:
    """*stamp* as a POSIX timestamp, or ``-inf`` when it cannot be read.

    Ordering ISO-8601 strings would sort them by their UTC offsets before their
    moments, and the servers on one deployment need not be in one zone. An
    unparseable or absent stamp loses every comparison rather than winning one
    by accident: "when this was written is unknown" must never outrank a
    measurement that carries its own time.
    """
    if not isinstance(stamp, str) or not stamp:
        return float("-inf")
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        return float("-inf")
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.timestamp()


def _owner_row(app: Any, record: Any) -> dict[str, Any] | None:
    """Who may write the control context, and whether that is this terminal.

    Read from ``app.state`` first, because the owner task publishes what this
    process decided on its last tick while the record only says what was last
    written. A terminal that has claimed and not yet read itself back would
    otherwise report the previous owner as somebody else.

    ``self`` is the point of the row: behind another live terminal the popover
    renders the roster read-only and the write routes answer ``409
    context_owned_elsewhere``, and the operator is owed the pid and the port to
    go to rather than a disabled control with no explanation.

    ``None`` when nothing owns the context — no tick has got far enough and the
    record on disk names nobody. A ``controls_server`` owner is an ordinary
    answer here and carries ``port: null``: it serves nothing to open.
    """
    context = _context_state(app)
    if context.owner is not None:
        identity = context.follows if context.follows is not None else terminal_identity()
        return {**identity.to_payload(), "self": context.follows is None}
    recorded = getattr(record, "owner", None)
    if recorded is None:
        return None
    return {**recorded.to_payload(), "self": control_context.owned_here(record)}


def _server_rows(reports: Sequence[Any]) -> list[dict[str, Any]]:
    """One row per running controls server, oldest report first.

    The fleet, published rather than collapsed. A deployment runs one controls
    server per agent session, each binding the record's generation on its own
    schedule, so "has the switch landed" is a question about all of them at
    once: the chip resolves its pending switch when every live server reports
    the generation it asked for, and names the pid of any that reported
    ``failed``. One collapsed verdict could not name that pid.

    ``applied_target`` and ``applied_generation`` are ``null`` until a server's
    first child has answered its init frame. Null is not "baseline" — it is
    "this server has not got there yet" — and a reader that treats the two the
    same reports a swap as complete before anything moved.

    ``children`` is the row's word for whether that ever has to happen:
    a server publishing an empty list serves nothing, so nothing it runs can
    still touch the target the deployment left, and the chip's convergence
    wait passes over it — the same stance ``converged()`` takes on a null
    binding. A row with children is a connector serving or mid-launch, and is
    owed to the wait.
    """
    rows: list[dict[str, Any]] = []
    for report in sorted(reports, key=lambda r: _stamp_epoch(r.updated_at)):
        rows.append(
            {
                "pid": report.server_pid,
                "session": report.session,
                "applied_target": report.applied_target,
                "applied_generation": report.applied_generation,
                "children": list(report.children),
                "last_switch": _aged(report.last_switch),
                "last_posture_realign": report.last_posture_realign,
                "updated_at": report.updated_at,
            }
        )
    return rows


def _published_targets(reports: Sequence[Any]) -> dict[str, Any]:
    """Per-target display metadata, the most recent publisher winning per target.

    Each server publishes the identity of the targets IT rendered, under the
    posture IT is running, so two servers on one deployment can name two
    gateways for one target — one narrowed, one not. The newer report is the
    answer, decided per target rather than per fleet: a server that has just
    started has published nothing for the targets it has not rendered, and
    letting its empty block win would blank a row another server has named.
    """
    merged: dict[str, Any] = {}
    for report in sorted(reports, key=lambda r: _stamp_epoch(r.updated_at)):
        merged.update({str(target): row for target, row in report.targets.items()})
    return merged


def _published_reachability(reports: Sequence[Any]) -> dict[str, Any]:
    """Per target, the probe rows of the server that looked at it most recently.

    Two servers probing one target are two measurements of the same gateways at
    two different moments, and the fresher one is the deployment's answer: an
    older ``reached`` would go on speaking for a gateway that has since stopped
    answering, and merging the two would report a state nobody measured.

    Newest is decided by the ``probed_at`` inside the row, never by the
    report's ``updated_at`` — a server rewrites its file to publish children or
    switch progress without going near a gateway, and reading freshness off the
    file would let that overwrite a real sweep. The sweep's own ``published_at``
    is not read either: it stamps the pass, and a pass can carry a row the
    prober decided not to re-measure.

    A server that has not probed publishes ``{}`` and contributes nothing, which
    is how a target with no live measurement reaches
    :func:`_collapse_reachability` as ``unknown``.
    """
    newest: dict[str, tuple[float, dict[str, Any]]] = {}
    for report in reports:
        rows = report.reachability.get("targets")
        if not isinstance(rows, dict):
            continue
        for target, roles in rows.items():
            if not isinstance(roles, dict):
                continue
            probed = max(
                (
                    _stamp_epoch(row.get("probed_at"))
                    for row in roles.values()
                    if isinstance(row, dict)
                ),
                default=float("-inf"),
            )
            key = str(target)
            if key not in newest or probed > newest[key][0]:
                newest[key] = (probed, roles)
    return {target: roles for target, (_, roles) in newest.items()}


#: The ``last_posture_realign`` state that means a narrowing has not reached the
#: agent yet. The connector is rebuilt only once the run in flight finishes, and
#: the popover says so rather than leaving a toggle that appears to have done
#: nothing.
REALIGN_PENDING = "pending"


def _fleet_realign(reports: Sequence[Any]) -> dict[str, Any] | None:
    """The deployment's posture realignment — any pending one wins.

    A narrowing lands on a server only when its connector is rebuilt, so one
    server still ``pending`` is the fact the operator needs even when a second
    server finished realigning afterwards. Ordering by time alone would let that
    newer ``done`` hide the toggle that has not taken effect.
    """
    blocks = [
        block
        for block in (report.last_posture_realign for report in reports)
        if isinstance(block, dict)
    ]
    if not blocks:
        return None
    pending = [block for block in blocks if block.get("state") == REALIGN_PENDING]
    return dict(max(pending or blocks, key=lambda block: _stamp_epoch(block.get("at"))))


def _execution_rows() -> list[dict[str, Any]]:
    """One row per live execution marker, oldest first.

    Named rather than counted. A boolean can say that something is running; it
    cannot say that it is somebody else's notebook, which is the difference
    between "wait a moment" and "go and interrupt that kernel". ``session``,
    ``surface`` and ``kernel_id`` are the marker's own fields — the three
    :func:`~osprey.mcp_server.control_system.target_eligibility.busy_client`
    names the holder from — so the popover and the switch refusal describe one
    run in the same terms.

    ``age_s`` is computed here for the reason every other stamp on this route
    is: the marker was written by another process on another clock.
    """
    rows: list[dict[str, Any]] = []
    for marker in _live_executions():
        started_at = marker.get("started_at")
        rows.append(
            {
                "pid": _pid_or_none(marker.get("pid")),
                "target": marker.get("target"),
                "session": marker.get("session"),
                "surface": marker.get("surface"),
                "kernel_id": marker.get("kernel_id"),
                "started_at": started_at if isinstance(started_at, str) and started_at else None,
                "age_s": _age_seconds(started_at),
            }
        )
    return rows


def _target_display(
    config: Any,
    published: Mapping[str, Any],
    effective_writes: Mapping[str, bool],
) -> dict[str, dict[str, Any]]:
    """Per-target ``label`` / ``endpoint`` / ``real_machine``, published first.

    The controls servers mint these for the targets they are actually running,
    and every reader renders what it was handed. So a published row wins
    wherever it carries a label: it describes a process that owns a connector,
    while the render describes what a *new* server would do with the config as
    it stands now. A labelled row therefore names the gateway that server last
    published, which after a narrowing is the gateway the previous posture
    selected until the reconciler's next pass republishes it.

    The render is the fallback, through the same
    :func:`~osprey.mcp_server.control_system.connector_host_manager.target_display_metadata`
    the writer uses — a deployment whose servers have not started yet still has
    to name its targets, and naming them with a second derivation of this
    module's own is how a badge comes to disagree with a prompt.

    That fallback is rendered under the deployment's effective posture, which is
    what *effective_writes* carries. The renderer's own default resolves the
    posture from ``OSPREY_POSTURE_SESSION``, a stamp the web server does not
    carry, so an uninjected render would find no narrowing at all and name the
    write gateway of a target the operator has already given up writes on. The
    caller resolves the column once and hands it over, so the endpoint on a row
    and the ``effective`` flag beside it are one answer rather than two.

    Args:
        config: The rendered config mapping, or ``_UNREADABLE_SECTION`` when
            ``config.yml`` could not be read.
        published: The fleet's per-target display metadata, collapsed by
            :func:`_published_targets`.
        effective_writes: Per-target effective write posture, keyed by target
            name. A target the mapping does not answer for is rendered from the
            deployment ceiling and this run's mode.
    """
    derived: dict[str, Any] = {}
    if config is not _UNREADABLE_SECTION and isinstance(config, dict):
        try:
            from osprey.mcp_server.control_system.connector_host_manager import (
                target_display_metadata,
            )

            derived = dict(target_display_metadata(config, effective_writes=effective_writes))
        except Exception:  # noqa: BLE001 — the roster must render, not 500
            logger.warning("Could not derive the control targets' display metadata")

    meta: dict[str, dict[str, Any]] = {}
    for target, row in derived.items():
        meta[str(target)] = dict(row) if isinstance(row, dict) else {}
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


def _posture_view(app: Any, config_path: Path | None) -> dict[str, Any]:
    """Everything ``GET /api/terminal/posture`` reports. BLOCKING.

    Called through ``run_in_threadpool``: it parses ``config.yml``, reads the
    control-context record, globs the state directory for the servers' reports
    and the execution markers, and asks the process table which of their writers
    are still running. On the event loop one wedged process table would stall
    every request this server is serving, the terminal websocket included, and
    the chip polls this route per open card.

    **One resolution for what this function reads itself.** The record is read
    once here and every fact this body draws from it — the target, the
    generation, the narrowings, ``last_switch``, the owner — comes from that one
    read; the fleet is listed once and the server rows, the reachability and the
    display metadata all come from that one listing. Two resolutions could
    straddle a switch and describe two different machines in one payload. The
    render is read once for the same reason and memoized by ``config.yml``'s
    signature besides (:func:`_rendered_config`).

    **``effective`` is the exception, and it is not one this route can close.**
    :func:`_effective_writes` delegates to
    :func:`~osprey_connectors.posture_store.effective_writes`, which resolves the
    narrowing by reading the record itself, once per target — that function is
    the single implementation of rule 3 and restating it here to reuse the
    record above would be the second implementation it exists to prevent. So a
    switch landing mid-render can move ``effective`` on a later row while the
    earlier rows and the top-level fields still describe the previous record.
    The window is one poll wide, the chip refetches on the ``control_context``
    frame, and closing it properly means letting that function take an
    already-read record rather than opening the file again.

    **No session is involved.** There is one control context per deployment, so
    every answer here is the deployment's: the caller's ``session_id`` names who
    is asking and decides nothing about what they are told.
    """
    config = _rendered_config(config_path)
    section = _section_of(config)
    record = control_context.read_record()
    reports = _live_reports()

    baseline = _baseline_target(section)
    control_target = record.target if record is not None else baseline

    entry = {} if record is None else dict(record.posture)
    ceilings = _session_ceilings(section)
    configured = list(_configured_target_names(section))
    # Resolved BEFORE the render, and exactly once. The display metadata names
    # one gateway per target, and WHICH gateway depends on the same effective
    # answer the rows below publish — the renderer's own default would resolve
    # it from a posture stamp this process does not carry, and name the write
    # gateway of a target this deployment has narrowed.
    #
    # ONE ceiling per row. ``ceiling_writes`` reports ``session_posture``'s
    # per-target map; ``effective`` runs through the record, whose own ceiling is
    # ``target_writes_enabled`` for every configured target. On a NON-switch-
    # capable render those two disagree — ``session_posture`` answers for the
    # baseline alone, while ``configured_targets`` still lists the others — so
    # an unarmed row would render ceiling-off beside effective-on: a filled dot
    # for a target no connector here is ever built for. Gating on the ceiling
    # the route already holds keeps the row internally consistent; where the two
    # agree (every switch-capable deployment) this changes nothing.
    effective_by_target = {
        target: bool(ceilings.get(target, False)) and _effective_writes(section, target)
        for target in configured
    }
    display = _target_display(config, _published_targets(reports), effective_by_target)
    threshold_s = _probe_staleness_threshold_s(config)
    probed = _published_reachability(reports)

    rows: list[dict[str, Any]] = []
    for target in configured:
        meta = display.get(target) or {}
        real_machine = bool(meta.get("real_machine"))
        label = str(meta.get("label") or "") or target
        short_label, kind = _short_label_and_kind(label, real_machine)
        ceiling_writes = bool(ceilings.get(target, False))
        effective = effective_by_target[target]
        posture = entry.get(target, POSTURE_WRITES)
        # Only for a row a narrowing would actually CHANGE, mirroring the POST's
        # "targets the request would touch" rule: a target already narrowed
        # cannot be stranded by narrowing it, and reporting the refusal there
        # would lock the toggle that brings it back.
        narrowing = None if posture == POSTURE_SANDBOX else _row_narrowing_refusal(config, target)
        available_now, reason, reason_detail = _row_availability(
            config, target, control_target, baseline, effective
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
                "active": target == control_target,
                "is_baseline": target == baseline,
                "available_now": available_now,
                "reason": reason,
                "reason_detail": reason_detail,
                "ceiling_writes": ceiling_writes,
                "posture": posture,
                "effective": effective,
                "narrowing_refusal": narrowing,
                "reachability": _collapse_reachability(
                    probed.get(target),
                    _row_selected_role(config, target, effective),
                    threshold_s,
                ),
            }
        )

    return {
        "control_target": control_target,
        "generation": None if record is None else record.generation,
        "store_available": _record_available(),
        # Published rather than left for the client to infer from
        # ``ceiling_writes ∧ posture ≠ sandbox ∧ ¬effective``: that signature
        # is also what a record that fails to resolve leaves behind, and an
        # operator must not be told the deployment is read-only when it is not.
        "readonly_run": is_readonly_run(),
        "owner": _owner_row(app, record),
        "servers": _server_rows(reports),
        "execution_in_flight": _execution_rows(),
        "last_switch": _aged(None if record is None else record.last_switch),
        "last_posture_realign": _fleet_realign(reports),
        "targets": rows,
    }


@router.get("/api/terminal/posture")
async def get_terminal_posture(request: Request, session_id: str | None = None):
    """Report the deployment's control-target roster and its write posture.

    The single truth the header chip and its popover read. There is one control
    context per deployment, so every answer here is the deployment's: the
    caller's ``session_id`` names who is asking and changes nothing about what
    they are told.

    One row per configured control target, each answering the whole of what the
    operator needs about that machine:

    * ``label`` / ``display_name`` / ``short_label`` / ``kind`` / ``endpoint``
      / ``real_machine`` — what to call it and what it is. The label is the one
      a running controls server published for the target it is on, or the one
      this render derives where no server has published one;
      ``display_name`` is the operator-facing name minted beside it
      (``control_system.target_display_names`` renames it per deployment);
      ``short_label`` and ``kind`` come from ``real_machine`` and the label's
      shape, never from the target name, so a stand-in never renders as the
      facility's own machine.
    * ``ceiling_writes`` / ``posture`` / ``effective`` — the three terms of the
      write decision, kept separate on purpose. The ceiling is the deployment's
      (``session_posture``); the posture is the operator's own narrowing; the
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
      ``reason_detail`` — where the deployment is standing and whether a switch
      is offered. ``reason`` is the switch tool's own machine code, so the
      popover and the agent keep agreeing about the same refusal;
      ``reason_detail`` is the eligibility verdict's operator sentence, which
      the popover renders as the tooltip behind its short phrase.
    * ``reachability`` — the state of the gateway role this target would
      actually select under its effective posture, aged server-side, with the
      other roles named beside it (see :func:`_collapse_reachability`). It is
      the sweep of whichever live server probed that target most recently.

    And, once for the deployment:

    * ``control_target`` and ``generation`` — the record's target and the
      generation it was minted at. Every stamped execution pins itself to that
      number, so a client watching a switch land watches this.
    * ``owner`` — ``{kind, pid, port, self}``, or ``null`` when nothing owns the
      context. ``self`` is false on a terminal that is following another one,
      which is exactly when the write routes answer ``409
      context_owned_elsewhere`` and the roster has to render read-only.
    * ``servers`` — one row per running controls server: ``pid``, ``session``,
      ``applied_target``, ``applied_generation``, ``children``, ``last_switch``
      (aged), ``last_posture_realign`` and ``updated_at``. A switch has landed
      when every one of them holding a connector (``children`` non-empty)
      reports the generation it was asked for; a row reporting ``failed`` names
      the pid an operator has to go and look at.
    * ``execution_in_flight`` — one row per live execution marker, carrying
      ``session``, ``surface`` and ``kernel_id`` so the surface can say WHOSE
      run is holding the target rather than only that something is.
    * ``store_available`` — whether there is anywhere to record a narrowing at
      all.
    * ``readonly_run`` — the whole deployment was started read-only, which is
      the one thing besides the ceiling and the narrowing that holds
      ``effective`` down. Stated outright, so a client never has to infer it
      from a row whose ``effective`` a failed record read zeroed.
    * ``last_switch`` — the record's terminus, passed through whole with
      ``age_s`` added, and ``last_posture_realign``, the fleet's, with any
      pending realignment winning over a settled one.

    Everything costs a config read, a record read, a state-directory glob and a
    walk of the process table, so it is computed in a worker thread
    (:func:`_posture_view` via ``run_in_threadpool``).

    Unlike POST, an id that names no session on disk is **not** a 409. The chip
    renders with the page, which can be before the first prompt has written a
    session file, and refusing there would blank the one surface that tells the
    operator what the deployment permits. Answering costs nothing: a read grants
    nothing and stores nothing. ``session_id`` is optional for the same reason
    it is optional on the two POSTs: the roster is the deployment's, and the Lab
    page's bar has no session to name. A string that IS sent is shape-checked
    with the closed grammar the POSTs use, so the three routes keep one error
    contract.
    """
    _require_session_uuid(session_id)

    view = await run_in_threadpool(_posture_view, request.app, request.app.state.config_path)
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
