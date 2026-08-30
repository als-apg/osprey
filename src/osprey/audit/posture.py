"""The session posture as the ledger records it — one reader for every surface.

Three in-process recorders — the MCP audit middleware, the executor's gates and
the protected-set funnel — file records that carry the same three facts about
the process they run in: which posture it is under, how that posture was
established, and which posture-store key it belongs to. All three read them
from the environment the Web Terminal's spawn sites stamp, and all three spell
them the same way, so the answers live here once and each recorder imports
them. This module is a leaf below all three: the middleware and the gates
already depend on the audit package, and nothing here depends on them.

The posture *value* has two sources, and :func:`posture` is the seam between
them. The environment answers for a process that belongs to no session — a
dispatch worker, an ``agent_runner`` child, a CLI run — and its answer is
returned before anything else is read. A process that carries a posture-store
key belongs to a Web Terminal session, whose operator narrows write posture per
CONTROL TARGET: "the live machine is read-only for me, leave the virtual
accelerator alone". No environment variable can carry that, because setting one
would sandbox both targets, so the answer for a session comes from the
per-(session, target) store — :mod:`osprey_connectors.session_store`, imported
lazily so that the leaf stays a leaf and the session-less paths never pay for
it. The store can only NARROW: an environment that already says sandbox is
returned unchanged, and no store entry has ever granted a write.

:func:`posture` never raises. Three refusal paths call it on every tool call,
and an exception from any of them would cost the call rather than answer it, so
every way of failing to read the store — a missing file, a corrupt one, an
unresolvable target, an import that fails — degrades to the environment answer.

The spellings are the wire contract with the spawn sites
(``interfaces/web_terminal``), which stamp the variables by these names and are
pinned against them by test rather than by import — the interfaces package must
not be dragged behind every MCP server.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from osprey.audit.envelope import POSTURE_SOURCE_PROCESS, POSTURE_SOURCES

logger = logging.getLogger("osprey.audit.posture")

__all__ = [
    "CONTROL_TARGET_ENV_VAR",
    "OSPREY_AGENT_DATA_ROOT",
    "POSTURE_ENV_VAR",
    "POSTURE_SANDBOX",
    "POSTURE_SESSION_ENV_VAR",
    "POSTURE_SOURCE_ENV_VAR",
    "POSTURE_WRITES",
    "SANDBOX_MODE",
    "invalidate_session_target_cache",
    "posture",
    "posture_session",
    "posture_source",
    "session_control_target",
]

#: The session posture, carried into every child of a Web Terminal session,
#: MCP servers included.
POSTURE_ENV_VAR = "OSPREY_EXECUTION_MODE"

#: How the posture in :data:`POSTURE_ENV_VAR` was established, and the
#: posture-store key it belongs to. Stamped by the Web Terminal spawn sites and
#: absent everywhere else (a dispatch worker, a CLI run, a container-level
#: execution mode), which is what
#: :data:`~osprey.audit.envelope.POSTURE_SOURCE_PROCESS` is for.
POSTURE_SOURCE_ENV_VAR = "OSPREY_POSTURE_SOURCE"
POSTURE_SESSION_ENV_VAR = "OSPREY_POSTURE_SESSION"

#: The agent-data root the spawning surface resolved, stamped as a PAIR with
#: :data:`POSTURE_SESSION_ENV_VAR` — never one without the other. A session
#: child, the MCP servers below it and the stdlib-only hooks all have to agree
#: on ONE directory for the control-target state file and the session-posture
#: store; each deriving it for itself means a deployment that moves
#: ``agent_data.base_dir``, or a hook that can only guess the repo root, reads a
#: different directory from the one the server writes. The spawn sites resolve
#: it once and say so here, so every reader below has an authoritative anchor
#: and only a process with no stamp at all falls back to its own derivation.
#:
#: The name is the value: this is an environment variable's spelling, and the
#: wire contract is the string, not the constant.
OSPREY_AGENT_DATA_ROOT = "OSPREY_AGENT_DATA_ROOT"

#: The control target a RUN was pinned to, stamped into the executor's sandbox
#: subprocess beside its generation and state PID (see
#: :data:`osprey.runtime.ENV_CONTROL_TARGET`, which spells the same string for
#: the reading side of the write pin). Where it is present it is the authority
#: on which target this process's writes are about: the session may have
#: switched since the run started, and the run's writes are still checked
#: against the target it was pinned to. Absent everywhere else — a stamp is
#: never inherited stale, because the executor pops all three variables when it
#: has no target to name.
CONTROL_TARGET_ENV_VAR = "OSPREY_CONTROL_TARGET"

#: The one value of :data:`POSTURE_ENV_VAR` that sandboxes a session. A *value*
#: comparison, never a presence check: the writes posture sets the same
#: variable.
SANDBOX_MODE = "readonly"

#: How the ledger spells the two postures — the Web Terminal's vocabulary, not
#: the environment variable's. Records from surfaces that never see
#: ``OSPREY_EXECUTION_MODE`` join on these.
POSTURE_SANDBOX = "sandbox"
POSTURE_WRITES = "writes"


def posture() -> str:
    """This process's posture, spelled the way the ledger spells it.

    The state at decision time, not the reason for any decision: the protected
    set and the static executor layers refuse in the writes posture too.

    Two sources, in this order:

    1. **The environment.** ``OSPREY_EXECUTION_MODE == "readonly"`` is a
       sandbox — a *value* comparison, never a presence check, because the
       writes posture and a readwrite run set the same variable. A process with
       no posture-store key (:func:`posture_session` is ``None``) is answered
       here and nothing else is read: a dispatch worker, an ``agent_runner``
       child and a CLI run belong to no session, so no store entry can address
       them and no file read may be charged to them.
    2. **The per-(session, target) store**, for a process that does carry a
       key. The entry that governs it is the one for the SESSION's current
       control target (:func:`session_control_target`); a sandbox entry there
       is a sandbox here. An entry for any OTHER target says nothing about this
       process — narrowing the live machine must leave a session working on the
       virtual accelerator alone, which is the entire point of the feature.

    An environment that already says sandbox short-circuits: the store holds
    narrowings only, so consulting it could not change the answer.

    Never raises. See :func:`_session_target_is_sandboxed` for the degradation
    rule: every failure to read the store answers the environment.
    """
    env_answer = (
        POSTURE_SANDBOX if os.environ.get(POSTURE_ENV_VAR) == SANDBOX_MODE else POSTURE_WRITES
    )
    session_key = posture_session()
    if session_key is None or env_answer == POSTURE_SANDBOX:
        return env_answer
    return POSTURE_SANDBOX if _session_target_is_sandboxed(session_key) else env_answer


def posture_source(declared: str | None = None) -> str:
    """The provenance of the posture a decision was made under.

    *declared* is the call site's own answer, for a surface that knows it —
    a web request belongs to no session and stamps ``app`` whatever the server
    process inherited. With no answer the environment ladder is read: a session
    child carries the marker its spawn stamped, and anything else is
    ``process``.

    An unrecognised value degrades rather than being carried through, from
    either source: :data:`~osprey.audit.envelope.POSTURE_SOURCES` is closed, a
    record whose provenance is unrecognised reads as authoritative while
    meaning nothing, and the envelope would reject it — which would cost the
    record entirely rather than one field of it.
    """
    stamped = declared if declared is not None else os.environ.get(POSTURE_SOURCE_ENV_VAR)
    return stamped if stamped in POSTURE_SOURCES else POSTURE_SOURCE_PROCESS


def posture_session() -> str | None:
    """The posture-store key this process's posture belongs to, if it was stamped."""
    return (os.environ.get(POSTURE_SESSION_ENV_VAR) or "").strip() or None


# -- the session's control target ------------------------------------------

#: The resolved session target, cached against the signature of the state
#: files that produced it. Same shape and same reason as the store's own cache
#: in :mod:`osprey_connectors.session_store`: :func:`posture` runs on every
#: tool call, and a glob plus a JSON parse per call is a cost the answer does
#: not need to pay twice for one unchanged directory.
_TARGET_CACHE_LOCK = threading.Lock()
_TARGET_CACHE: tuple[Any, str | None] | None = None


def invalidate_session_target_cache() -> None:
    """Forget the resolved session target. For tests and a process that moved roots."""
    global _TARGET_CACHE
    with _TARGET_CACHE_LOCK:
        _TARGET_CACHE = None


def _state_signature(path: Path) -> tuple[str, int, int, int] | None:
    """``(path, mtime_ns, size, inode)`` for one state file, or ``None``.

    The inode is part of it because the state file is replaced atomically
    (temp file plus ``os.replace``), so two switches inside one filesystem
    clock tick differ by inode when mtime and size do not — the rule
    :mod:`osprey_connectors.session_store` and :mod:`osprey.health.signatures`
    already follow. The path is part of it so that a reader whose agent-data
    root moved does not answer from the previous root's cache.
    """
    try:
        st = path.stat()
    except OSError:
        return None
    return (str(path), st.st_mtime_ns, st.st_size, st.st_ino)


def _match_session_record(target_state: Any, entries: list[Path], owner_ppid: int) -> str | None:
    """The target named by the one live record this session owns, or ``None``.

    Selection is *exact parent equality*: the controls MCP server that writes
    these files and the server this code is running in are both spawned by the
    same Claude Code process, so the record whose ``owner_ppid`` is our parent
    is the one describing our session. That is the rule
    ``executor._session_target_record`` and ``target_banner.resolve_session_target``
    already use, and it is deliberately narrower than the ancestor-chain walk
    the stdlib-only hooks do: a deployment that interposes a process breaks the
    equality and gets "no answer", never another session's target.

    Zero matches (no controls server, or a directory this deployment never
    created) and more than one (an ``owner_ppid`` collision after PID reuse)
    both answer ``None``, as does a record naming a target no reader knows. A
    record whose ``server_pid`` is gone is residue — its target describes a
    server nobody is talking to any more — and is skipped.

    That liveness skip is evaluated when the *signature* moves, not on every
    lookup: the caller caches this answer against the state files' signature,
    and a server dying does not touch its file. Deliberately so — a ``kill(pid,
    0)`` per tool call buys very little, because the record of a server that
    died is rewritten or swept by the next server to start, which moves the
    signature and re-runs this. Nothing rests on the freshness of the answer
    either: the guarantee that a write cannot land on a machine nobody selected
    is the runtime's generation pin, not this lookup.
    """
    matches: list[str] = []
    for entry in entries:
        record = target_state.read_file(entry)
        if not isinstance(record, dict) or record.get("owner_ppid") != owner_ppid:
            continue
        server_pid = record.get("server_pid")
        if not isinstance(server_pid, int) or not target_state.is_process_alive(server_pid):
            continue
        target = record.get("target")
        if target in target_state.TARGET_NAMES:
            matches.append(str(target))
    if len(matches) != 1:
        if matches:
            logger.debug(
                "%d control-target records share owner_ppid %s; the session target is unknown",
                len(matches),
                owner_ppid,
            )
        return None
    return matches[0]


def session_control_target() -> str | None:
    """The control target this process's writes are about, or ``None``.

    Public because the refusals are per target now: a gate that names the
    narrowed target in its message has to resolve the same target
    :func:`posture` resolved, and resolving it a second way would let the two
    disagree — the refusal would name a machine the clamp did not fire for.

    Two readers, in this order:

    1. :data:`CONTROL_TARGET_ENV_VAR`, when it names a target readers know.
       Only the executor's sandbox subprocess carries it, and there it is the
       authority: that run is pinned to that target and its writes are refused
       if the session moves off it, so its posture is the posture of the target
       it was pinned to, not of the one the session has since switched to. The
       value is checked against ``target_state.TARGET_NAMES`` exactly as a
       record's is — a stamp naming something no reader knows can only index
       the store to a key nothing writes, so it is dropped in favour of the
       state record rather than answered with.
    2. Otherwise the controls server's state record for this session — see
       :func:`_match_session_record`. This is the reader for the MCP servers,
       which is where :func:`posture` is called from on every tool call.

    ``None`` is an honest "not knowable here", and :func:`posture` answers the
    environment for it rather than clamping: this gate refuses EVERY write tool
    in the process, so firing it on a target nobody could name would refuse
    writes to machines the operator never narrowed. The fail-closed layer for
    one specific write is the connector's reference monitor, whose
    ``effective_writes`` takes the most restrictive entry when it cannot name a
    target.

    Never raises: an unimportable ``target_state`` (a deployment where the MCP
    servers are not installed) and an unreadable state directory are both "not
    knowable here" — for the stamp too, which is validated against that module's
    vocabulary and so cannot be answered without it.
    """
    stamped = (os.environ.get(CONTROL_TARGET_ENV_VAR) or "").strip()

    global _TARGET_CACHE
    owner_ppid = os.getppid()
    try:
        # Imported inside the function: this module is a leaf, and every MCP
        # server imports it while only a session child ever reaches this line.
        from osprey.mcp_server.control_system import target_state

        if stamped:
            if stamped in target_state.TARGET_NAMES:
                return stamped
            logger.debug(
                "%s names an unknown control target %r; reading the state record instead",
                CONTROL_TARGET_ENV_VAR,
                stamped,
            )

        entries = sorted(target_state.state_dir().glob(target_state.STATE_FILE_GLOB))
        signature: Any = (owner_ppid, tuple(_state_signature(entry) for entry in entries))
    except Exception:  # noqa: BLE001 — an unreadable state directory is "unknown"
        logger.debug(
            "Control-target state unavailable; the session target is unknown", exc_info=True
        )
        return None

    cached = _TARGET_CACHE
    if cached is not None and cached[0] == signature:
        return cached[1]

    target = _match_session_record(target_state, entries, owner_ppid)
    with _TARGET_CACHE_LOCK:
        _TARGET_CACHE = (signature, target)
    return target


def _session_target_is_sandboxed(session_key: str) -> bool:
    """Whether the store narrows *session_key*'s current target to the sandbox.

    The one place :mod:`osprey_connectors.session_store` is reached from the
    audit package, and the one place the degradation rule lives: ``False`` for
    every way of not knowing — no store, a corrupt or unreadable one, a target
    that cannot be resolved, a connector package that cannot be imported. The
    caller then answers the environment, which is what a session with nothing
    narrowed has always been answered.

    ``False`` is not a grant. A store entry can only narrow, so failing to read
    one leaves whatever the deployment and the environment already decided —
    including a deployment-wide read-only run, which no reader here can lift.
    """
    try:
        from osprey_connectors import session_store

        target = session_control_target()
        if target is None:
            return False
        return session_store.target_posture(session_key, target) == session_store.POSTURE_SANDBOX
    except Exception:  # noqa: BLE001 — posture() is called on every tool call
        logger.debug("Session-posture store unavailable; answering the environment", exc_info=True)
        return False
