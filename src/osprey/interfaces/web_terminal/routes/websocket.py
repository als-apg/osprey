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

from osprey.interfaces.web_auth import PANEL_TOKEN_ENV, get_web_credentials
from osprey.interfaces.web_terminal.operator_session import build_operator_child_env
from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery

logger = logging.getLogger(__name__)

router = APIRouter()

_UUID_RE = re.compile(r"^[a-f0-9-]{36}$")

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


def _require_session_uuid(session_id: str) -> None:
    """Refuse *session_id* unless it is shaped like a Claude session UUID.

    One implementation for both posture routes, so the two cannot drift on the
    status, the error slug or the sentence. An arbitrary string can never
    become a store key that is then written to disk.

    Raises:
        HTTPException: 400 ``invalid_session_id`` when the shape does not match.
    """
    if not _UUID_RE.match(session_id):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_session_id",
                "message": "session_id must be a Claude session UUID.",
            },
        )


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


def _load_postures(path: Path) -> dict[str, str]:
    """Read the persisted postures, tolerating every kind of absence.

    Unknown posture values are dropped rather than honored: whatever survives
    this filter flows straight into :func:`_build_extra_env` and decides a
    child's execution mode, so a hand-edited or future-version entry must not
    reach it. A missing or corrupt file yields an empty store — the operator
    can set the postures again, which is a far better outcome than every spawn
    and every toggle failing on a file nobody can repair from the browser.
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
        if isinstance(key, str) and value in _VALID_POSTURES
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
    try:
        from osprey_connectors.types import any_target_writes_enabled

        if not config_path or not Path(config_path).exists():
            return False
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        return any_target_writes_enabled(config.get("control_system"))
    except Exception:  # noqa: BLE001 — an unreadable config is not a writes-on render
        logger.warning("Could not read the control-system write posture from %s", config_path)
        return False


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

    Returns the discovered UUID (or None). Also rekeys the registry entry.
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
    posture_key = claude_session_id or telemetry_session_id
    if posture_key and _session_postures(websocket.app).get(posture_key) == POSTURE_SANDBOX:
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
    so recording it is not enough: the session's PTY is terminated, and the
    client's reconnect brings the session back with the posture applied. The
    store is updated and persisted *before* the terminate, because the respawn
    reads it back through :func:`_build_extra_env` and would otherwise race it.

    Three refusals, all before anything is written:

    * **409** — the id names no session on disk. A Claude session file only
      appears once the operator has sent a prompt, and until then there is
      nothing to respawn; the detail says so, because "send one prompt first"
      is the entire remedy and the alternative is a toggle that silently does
      nothing.
    * **403** — ``writes`` on a render that arms no control target at all:
      ``control_system.connector.<type>.writes_enabled`` off for every type,
      and off in the deployment-wide ``control_system.writes_enabled`` they
      inherit from. One armed target is enough, because the posture may narrow
      what the render permits and never widen it.
    * **400** — an id that is not a session UUID, checked with the same
      ``_UUID_RE`` ``switch_session`` uses, so an arbitrary string can never
      become a store key that is then written to disk.
    """
    session_id = body.session_id
    _require_session_uuid(session_id)

    discovery = SessionDiscovery(request.app.state.project_cwd)
    if session_id not in discovery.snapshot_session_ids():
        raise HTTPException(
            status_code=409,
            detail={
                "error": "session_not_started",
                "message": (
                    "This session has not started yet — send one prompt first, "
                    "then set its posture."
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
    request.app.state.pty_registry.terminate_session(session_id)
    logger.info(
        "Session %s posture set to %s; PTY terminated for respawn", session_id, body.posture
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

    Unlike POST, an id that names no session on disk is **not** a 409. The badge
    renders with the terminal card, which can be before the first prompt has
    written a session file, and refusing there would blank the one surface that
    tells the operator what the render permits. Answering costs nothing: a read
    grants nothing, stores nothing, and reports exactly the posture that
    session will spawn under. The id is still shape-checked with the same
    ``_UUID_RE`` POST uses, so the two routes keep one error contract.
    """
    _require_session_uuid(session_id)

    posture = _session_postures(request.app).get(session_id, POSTURE_WRITES)
    return {
        "session_id": session_id,
        "posture": posture,
        "rendered_writes_enabled": _rendered_writes_enabled(request.app.state.config_path),
    }


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
        env = build_operator_child_env(project_cwd=cwd, session_key=operator_key, app=websocket.app)
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
