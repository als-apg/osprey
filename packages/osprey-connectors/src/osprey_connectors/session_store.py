"""The per-(session, target) write-posture store — one file, three readers.

An operator narrows one control target for one session from the header chip:
"stand-in is read-only for me, leave the virtual accelerator alone". That
narrowing is recorded in a single JSON file under the agent-data root and is
enforced at *write* time rather than delivered by respawning the agent, which
is what lets a flip land on a session already mid-conversation.

This module is the canonical reader. Two others answer from the same file and
must agree with it byte for byte: the controls MCP server (which imports this
one) and the stdlib-only PreToolUse hook in
``osprey/templates/claude_code/claude/hooks/osprey_target_state.py``, which
cannot import anything and therefore *restates* the contract below. Treat the
following four rules as the contract; a change here is a change there.

**1. Where the file is.** :data:`STORE_FILENAME` inside
:data:`STATE_DIR_NAME` under the agent-data root, so the store is co-sited
with the control-target state file and one directory answers "session state
for the control targets". The root resolves by ONE rule, used by writer and
both readers:

* :data:`AGENT_DATA_ROOT_ENV_VAR` when it names a non-blank path — the stamp
  a session child carries, always beside ``OSPREY_POSTURE_SESSION``; a child
  that has one anchor and not the other would read a store nobody writes;
* otherwise :func:`~osprey_connectors.workspace.resolve_shared_data_root`,
  the config derivation the state file has always used.

There is no third path and no fallback. A root that resolves to nothing makes
:func:`store_path` ``None``, which a route surfaces as ``store_unavailable``
rather than quietly writing somewhere a reader will never look. For *reading*,
an unresolvable root is indistinguishable from an empty store: nothing has
been narrowed, so the deployment ceiling stays in charge.

**2. What the shapes mean.** The file is a JSON object keyed by session key.
A value is either a per-target object (``{"live": "sandbox"}``) or one of the
two bare legacy strings the session-wide posture wrote before targets existed:

* bare ``"sandbox"`` narrows EVERY target in :data:`CONTROL_TARGETS`;
* bare ``"writes"`` is dropped — the writes posture is the *absence* of an
  entry, never a stored assertion, so nothing in this file can ever widen.

Anything else — an unknown value, a non-string key, a per-target leaf nobody
recognises — is dropped rather than honoured: what survives this filter
decides whether a real machine is written to, so a hand-edited or
future-version entry must not reach the decision. Every surviving key is
returned, ``operator-`` keys included. Their drop-on-restore rule belongs to
the web server's startup load alone: an enforcement reader that dropped them
would ignore a narrowing that is live for the rest of that process's life.

**3. How a lookup combines.** :func:`effective_writes` is the whole rule:

    ceiling ∧ not is_readonly_run() ∧ (store entry ≠ sandbox)

The ceiling is the deployment's own posture, read through the existing
predicates and never re-derived here — by connector type when the caller is a
connector, by target when it follows the session, and the union across
configured targets when it holds neither. The store can only narrow it.
With a session key and no resolvable target the MOST RESTRICTIVE entry for
that key wins (any sandbox refuses), because a caller that cannot say which
machine it is about must not be granted the most permissive answer. With no
session key at all the store is not consulted: nothing addressed the session,
so nothing narrowed it.

Inside an executor sandbox one further term is ANDed in, and only there:
:data:`LAUNCH_POSTURE_ENV_VAR`, the posture the run was LAUNCHED under. It is
not part of the restated rule — no hook process ever carries the stamp, and no
process that lacks it can be narrowed by it — but it is part of
:func:`store_permits`, because the sandbox's own reference monitor asks through
that function. Its job is asymmetry: a narrowing that lands mid-run is honoured
by the store read, while a WIDENING never reaches a run that started narrow,
which would otherwise hand a running script write access to a machine the
operator took away from it. See :func:`launch_permits`.

The store clause alone is :func:`store_permits`, public for the one caller that
holds a ceiling this module cannot derive — the connector's reference monitor
reads a deployment posture keyed on connector TYPE. It delegates that clause
here rather than restating it, so rule 3 keeps exactly two implementations.

**4. When a change is seen.** Every read re-stats the file and re-parses on
``(st_mtime_ns, st_size, st_ino)`` — the precedent is
``osprey.health.signatures``, widened by the inode because two narrowings a
second apart arrive as atomic temp+rename and a coarse filesystem clock would
otherwise hide the second one. A missing file is an empty store, not an error.

This module reads no config for the store half, holds no config cache, and
imports nothing from ``osprey``: it runs inside the connector-host child and
the executor sandbox, where the dependency budget is the lean connector chain.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from osprey_connectors.types import (
    CONTROL_TARGETS,
    any_target_writes_enabled,
    target_writes_enabled,
    type_writes_enabled,
)
from osprey_connectors.workspace import resolve_shared_data_root

logger = logging.getLogger("osprey_connectors.session_store")

__all__ = [
    "AGENT_DATA_ROOT_ENV_VAR",
    "LAUNCH_POSTURE_ALL_TARGETS",
    "LAUNCH_POSTURE_ENV_VAR",
    "POSTURE_SANDBOX",
    "POSTURE_WRITES",
    "STATE_DIR_NAME",
    "STORE_FILENAME",
    "VALID_POSTURES",
    "agent_data_root",
    "effective_writes",
    "invalidate_cache",
    "launch_narrowed_target",
    "launch_permits",
    "launch_posture_stamp",
    "legacy_store_path",
    "load_store",
    "parse_launch_posture",
    "parse_store",
    "session_map",
    "stamped_agent_data_root",
    "state_dir",
    "store_path",
    "store_permits",
    "target_posture",
]

#: The anchor stamp. Read by NAME rather than imported from
#: ``osprey.audit.posture``, which declares it for the stamping side: this
#: package is the lean connector chain and must not grow an ``osprey`` import
#: to learn one string.
AGENT_DATA_ROOT_ENV_VAR = "OSPREY_AGENT_DATA_ROOT"

#: Subdirectory of the agent-data root holding the control-target session
#: state. The same directory ``target_state.state_dir()`` resolves, spelled
#: here because a store the writer puts in one directory and a reader looks
#: for in another is a narrowing that silently never applies.
STATE_DIR_NAME = "control_target"

#: The store's filename inside :func:`state_dir`.
STORE_FILENAME = "session-postures.json"

#: The narrowing value. The only one that ever refuses anything.
POSTURE_SANDBOX = "sandbox"

#: The un-narrowed value. Recorded by some writers, meaningful to none: it is
#: dropped on parse so that the store holds narrowings and nothing else.
POSTURE_WRITES = "writes"

#: The two values a store entry may spell. Everything else is dropped.
VALID_POSTURES = frozenset({POSTURE_SANDBOX, POSTURE_WRITES})

#: The launch-time pin, stamped into an executor sandbox's environment by
#: ``osprey.mcp_server.python_executor.executor._apply_target_stamp`` and read
#: back HERE, inside that sandbox, by :func:`launch_permits`.
#:
#: The wire format is one ``"<target>=<posture>"`` pair and nothing else:
#:
#: * ``<target>`` is the control target the run was stamped against — a member
#:   of :data:`~osprey_connectors.types.CONTROL_TARGETS` — or
#:   :data:`LAUNCH_POSTURE_ALL_TARGETS` when the executor could not identify
#:   one, in which case the pin covers every target, exactly as the store's
#:   legacy bare ``"sandbox"`` does;
#: * ``<posture>`` is :data:`POSTURE_SANDBOX` or :data:`POSTURE_WRITES`.
#:
#: ``sandbox`` is the only value that does anything: ``writes`` is recorded so
#: the in-flight marker states what the run launched under, and is inert here
#: for the same reason a stored ``"writes"`` is dropped — nothing may widen.
#: A missing, blank or unparseable stamp is inert too, which is what makes
#: every process that is not an executor sandbox unaffected by this term.
#:
#: Spelled here rather than imported because this is the reading side and the
#: reading side runs in the lean connector chain;
#: ``registry/mcp.py`` and the executor import the name from this module, and
#: it is stripped from every rendered ``.mcp.json`` env block
#: (``NON_PINNABLE_AUDIT_MARKERS``) — a spec that could pin it could hand a
#: launched-narrow run the writes posture it was denied.
LAUNCH_POSTURE_ENV_VAR = "OSPREY_LAUNCH_POSTURE"

#: The launch stamp's target when the executor could not name one. Covers every
#: target: a run that could not say which machine it is about must not be the
#: one run a narrowing fails to reach.
LAUNCH_POSTURE_ALL_TARGETS = "*"

#: Parsed store, cached against the signature that produced it.
_CACHE_LOCK = threading.Lock()
_CACHE: tuple[Any, dict[str, dict[str, str]]] | None = None


# -- path resolution --------------------------------------------------------


def stamped_agent_data_root() -> Path | None:
    """The :data:`AGENT_DATA_ROOT_ENV_VAR` stamp as a path, or ``None`` unstamped.

    The stamp half of rule 1 on its own, so that the two files under this root
    — this store and the control-target state file — cannot read one
    environment variable two different ways. A blank or whitespace-only stamp
    is no stamp, and a ``~`` in it is expanded: a reader that took either
    literally would look in a directory no writer ever creates, which is the
    silent-no-narrowing failure rule 1 exists to prevent.

    ``osprey.mcp_server.control_system.target_state.state_dir`` calls this
    rather than restating it. The CONFIG half below is deliberately not shared:
    that module raises where this one answers ``None``, and a route turns its
    raise into a ``store_unavailable`` 503.
    """
    stamped = (os.environ.get(AGENT_DATA_ROOT_ENV_VAR) or "").strip()
    return Path(stamped).expanduser() if stamped else None


def agent_data_root() -> Path | None:
    """The agent-data root the store and the state file both live under.

    Rule 1 of the module contract: the :data:`AGENT_DATA_ROOT_ENV_VAR` stamp
    when it names a non-blank path, else the config derivation. ``None`` when
    neither answers — a deployment whose project root cannot be resolved has
    no store, which is a different thing from having an empty one.
    """
    stamped = stamped_agent_data_root()
    if stamped is not None:
        return stamped
    try:
        return resolve_shared_data_root()
    except Exception:  # noqa: BLE001 — an unresolvable root is "no store", not a crash
        logger.debug("Could not resolve the shared data root for the posture store", exc_info=True)
        return None


def state_dir() -> Path | None:
    """:data:`STATE_DIR_NAME` under :func:`agent_data_root`, or ``None``."""
    root = agent_data_root()
    return None if root is None else root / STATE_DIR_NAME


def store_path() -> Path | None:
    """The posture store's path, or ``None`` when the root is unresolvable.

    A ``None`` here is what a route reports as ``store_unavailable``: there is
    nowhere to record a narrowing that any reader would find.
    """
    directory = state_dir()
    return None if directory is None else directory / STORE_FILENAME


def legacy_store_path() -> Path | None:
    """The pre-feature store path, read through once when the new one is absent.

    The session-wide posture kept its store directly under the agent-data root
    (``resolve_shared_data_root() / "session-postures.json"``). A deployment
    upgrading with a sandboxed session live must not have that narrowing
    silently lifted the moment the code lands, so readers fall back to it while
    the new file does not exist. Nothing here writes: the web server persists
    the migrated shape on its first load, and this fallback stops answering the
    moment it does.
    """
    root = agent_data_root()
    return None if root is None else root / STORE_FILENAME


# -- parsing ----------------------------------------------------------------


def _target_map(value: Any) -> dict[str, str]:
    """One entry's value as a ``{target: posture}`` map of narrowings only.

    Rule 2 of the module contract. Returns an empty map for anything that
    narrows nothing, which is what drops the key.
    """
    if isinstance(value, str):
        if value == POSTURE_SANDBOX:
            return dict.fromkeys(CONTROL_TARGETS, POSTURE_SANDBOX)
        # Bare "writes" (and every unknown string) narrows nothing.
        return {}
    if isinstance(value, dict):
        return {
            target: posture
            for target, posture in value.items()
            if isinstance(target, str) and posture == POSTURE_SANDBOX
        }
    return {}


def parse_store(raw: Any) -> dict[str, dict[str, str]]:
    """Decode a store into ``{session_key: {target: "sandbox"}}``.

    Accepts the decoded JSON object or the raw text/bytes of one, so a reader
    that has already read the file and one that has not share this filter
    rather than each writing half of it. Never raises: a corrupt, truncated or
    hand-edited store is an empty store, because the alternative — every write
    and every toggle failing on a file nobody can repair from the browser —
    is worse than losing narrowings an operator can set again.
    """
    if isinstance(raw, str | bytes | bytearray):
        try:
            raw = json.loads(raw)
        except Exception:  # noqa: BLE001 — corrupt store must not wedge a write path
            logger.warning("Session-posture store is not valid JSON; ignoring")
            return {}
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, dict[str, str]] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        narrowed = _target_map(value)
        if narrowed:
            parsed[key] = narrowed
    return parsed


# -- reading ----------------------------------------------------------------


def _signature(path: Path) -> tuple[int, int, int] | None:
    """``(mtime_ns, size, inode)`` for *path*, or ``None`` when it is absent.

    Rule 4 of the module contract. The inode is the third element on purpose:
    both writers replace the store atomically, so two narrowings inside one
    filesystem clock tick differ by inode even when mtime and size do not.
    """
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def _read(path: Path) -> dict[str, dict[str, str]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError):
        # ``UnicodeDecodeError`` is a ``ValueError``, not an ``OSError``, so it
        # needs naming: a store that is not UTF-8 is unreadable for exactly the
        # reason a corrupt one is, and :func:`parse_store` already answers that
        # with an empty store rather than an exception. Letting this one escape
        # would make a single mis-encoded file raise inside every write path
        # that consults the posture — and it would put this reader at odds with
        # the stdlib restatement in the hooks, which cannot raise at all.
        logger.warning("Could not read the session-posture store at %s", path, exc_info=True)
        return {}
    return parse_store(raw)


def load_store() -> dict[str, dict[str, str]]:
    """The parsed store, re-read only when the file's signature moved.

    The returned mapping is the cached one — readers treat it as immutable.
    """
    global _CACHE
    path = store_path()
    if path is None:
        return {}
    signature = _signature(path)
    if signature is not None:
        key: Any = (str(path), signature)
    else:
        legacy = legacy_store_path()
        key = (
            str(path),
            None,
            None if legacy is None else str(legacy),
            _signature(legacy) if legacy else None,
        )
    cached = _CACHE
    if cached is not None and cached[0] == key:
        return cached[1]
    if signature is not None:
        parsed = _read(path)
    else:
        legacy = legacy_store_path()
        parsed = _read(legacy) if legacy is not None else {}
    with _CACHE_LOCK:
        _CACHE = (key, parsed)
    return parsed


def invalidate_cache() -> None:
    """Forget the parsed store. For tests and for a process that moved roots."""
    global _CACHE
    with _CACHE_LOCK:
        _CACHE = None


def session_map(session_key: str | None) -> dict[str, str]:
    """The narrowings recorded for one session key — ``{target: "sandbox"}``."""
    if not session_key:
        return {}
    return load_store().get(session_key, {})


def target_posture(session_key: str | None, target: str | None) -> str | None:
    """The stored posture for one (session, target), or ``None`` when unnarrowed."""
    if not target:
        return None
    return session_map(session_key).get(target)


# -- the launch-time pin ----------------------------------------------------


def launch_posture_stamp(target: str | None, launch_posture: str) -> str:
    """Compose the :data:`LAUNCH_POSTURE_ENV_VAR` value for one launch.

    The one place the wire format is written, so the executor that stamps it
    and :func:`parse_launch_posture` which reads it cannot disagree about the
    separator or about how "no target" is spelled.

    Args:
        target: The control target the run is stamped against, or ``None`` when
            the executor could not identify one — spelled
            :data:`LAUNCH_POSTURE_ALL_TARGETS`, which covers every target.
        launch_posture: :data:`POSTURE_SANDBOX` or :data:`POSTURE_WRITES` — the
            store's answer for that target at the moment of launch.
    """
    return f"{target or LAUNCH_POSTURE_ALL_TARGETS}={launch_posture}"


def parse_launch_posture(raw: str | None) -> dict[str, str]:
    """Decode a launch stamp into ``{target: "sandbox"}`` — narrowings only.

    The same filter rule 2 applies to the store, on the environment's one-pair
    spelling: only :data:`POSTURE_SANDBOX` survives, so a stamp can refuse and
    can never grant. Anything that is not exactly one ``target=posture`` pair —
    absent, blank, no separator, an unknown posture — is an empty map, which is
    what leaves every process that is not an executor sandbox untouched by this
    term.
    """
    text = (raw or "").strip()
    if not text:
        return {}
    target, separator, launch_posture = text.partition("=")
    if not separator:
        return {}
    target = target.strip()
    if launch_posture.strip() != POSTURE_SANDBOX:
        return {}
    if target == LAUNCH_POSTURE_ALL_TARGETS:
        return dict.fromkeys(CONTROL_TARGETS, POSTURE_SANDBOX)
    if not target:
        return {}
    return {target: POSTURE_SANDBOX}


def _permits(narrowed: dict[str, str], target: str | None) -> bool:
    """Whether *narrowed* leaves *target* writable — the combining half of rule 3.

    Shared by the store clause and the launch clause so the "most restrictive
    entry wins when the caller cannot name a target" rule has one implementation
    rather than one per source of narrowings.
    """
    if not narrowed:
        return True
    if target:
        return narrowed.get(target) != POSTURE_SANDBOX
    return POSTURE_SANDBOX not in narrowed.values()


def launch_narrowed_target() -> str | None:
    """The target the launch stamp names when it narrows, else ``None``.

    :data:`LAUNCH_POSTURE_ALL_TARGETS` when the executor could not name one, so
    a caller composing a refusal can tell "the operator had this machine
    read-only when the run started" from "nothing could be resolved at launch,
    so the run was pinned everywhere". Those two send an operator to different
    places, and the second must not be reported as somebody's decision.
    """
    text = (os.environ.get(LAUNCH_POSTURE_ENV_VAR) or "").strip()
    if not parse_launch_posture(text):
        return None
    target = text.partition("=")[0].strip()
    return target or None


def launch_permits(target: str | None) -> bool:
    """The launch clause — whether the run this process belongs to started open.

    Read from :data:`LAUNCH_POSTURE_ENV_VAR` on every call, like every other
    term here. What makes it a PIN is not that the value is unforgeable but that
    nothing ever re-derives it: the executor computes the store's answer once,
    at launch, and no reader here consults the store again on its behalf. So a
    narrowing that lands mid-run is enforced by the store read beside this one,
    while a WIDENING has nothing to reach — the run keeps answering from the
    posture it started under until it ends.

    It is an environment marker with exactly the strength of
    ``OSPREY_EXECUTION_MODE``: agent code inside the sandbox can pop or
    overwrite either one, and neither was ever the barrier against a script that
    sets out to defeat its own sandbox. That barrier is the connector the
    sandbox has to go through, the deployment ceiling it cannot edit, and the
    gateway role the connector-host child was connected on. This term exists for
    the honest case — an operator moving a posture under a run that is already
    in flight — and it is exactly as trustworthy as the mode stamp beside it.

    ``True`` — permitted — for every process that carries no stamp, which is
    every process except an executor sandbox.
    """
    return _permits(parse_launch_posture(os.environ.get(LAUNCH_POSTURE_ENV_VAR)), target)


# -- the rule ---------------------------------------------------------------


def store_permits(session_key: str | None, target: str | None) -> bool:
    """The store clause of :func:`effective_writes` — rule 3, on its own.

    Public because one caller needs this clause WITHOUT the ceiling
    :func:`effective_writes` derives: the connector's reference monitor
    (``control_system.base``) reads a deployment ceiling keyed on the connector
    TYPE, which is not a ceiling this module can produce from a target. It
    therefore ANDs its own ceiling with this function rather than restating the
    four combining terms below — the contract's rule 3 has exactly two
    implementations, this one and the stdlib restatement in the hooks, and a
    third would be a third thing to keep in step.

    Answers ``True`` — permitted — for everything that is not an actual
    narrowing: no session key (nothing addressed this session), no entry for
    the key, or an entry that does not name this target. It can only refuse;
    nothing here widens a ceiling.

    Inside an executor sandbox the launch pin (:func:`launch_permits`) is ANDed
    in ahead of the store read, so a run that started narrow stays narrow even
    after the operator widens the store under it. Everywhere else that term is
    inert, because only the executor stamps it.

    Args:
        session_key: The posture-store key (``OSPREY_POSTURE_SESSION``), or
            ``None``/blank when this process carries none.
        target: The control target the write lands on, or ``None`` when the
            caller cannot name one — in which case the most restrictive entry
            recorded for *session_key* decides.
    """
    # The launch pin first, and independently of the session key: it is a fact
    # about THIS RUN rather than about the session, and it costs one environment
    # read, so a sandbox that launched narrow refuses without touching the disk.
    if not launch_permits(target):
        return False
    if not session_key:
        return True
    # No resolvable target: the most restrictive entry for this key wins.
    return _permits(session_map(session_key), target)


def effective_writes(
    section: Any,
    session_key: str | None,
    target: str | None = None,
    *,
    connector_type: str | None = None,
) -> bool:
    """Whether a write may proceed here and now.

    ``ceiling ∧ not is_readonly_run() ∧ (store entry ≠ sandbox)`` — rule 3 of
    the module contract, spelled once so that the connector's reference
    monitor, the executor's gate, the tool roster and the popover cannot
    answer it differently.

    Args:
        section: The ``control_system:`` config section the ceiling is read
            from, through the existing deployment predicates.
        session_key: The posture-store key (``OSPREY_POSTURE_SESSION``).
            ``None`` or blank means nothing addressed this session, and the
            store is not consulted.
        target: The session control target this write lands on, when the
            caller knows it. ``None`` takes the most restrictive entry.
        connector_type: The connector type, for a caller that IS a connector.
            Given, it decides the ceiling — the deployment half stays keyed by
            type — while *target* still indexes the store.

    Returns:
        ``True`` only when the deployment arms this machine, the process is
        not a read-only run, and the operator has not narrowed it.
    """
    if connector_type is not None:
        ceiling = type_writes_enabled(section, connector_type)
    elif target is not None:
        ceiling = target_writes_enabled(section, target)
    else:
        ceiling = any_target_writes_enabled(section)
    if not ceiling:
        return False
    # Imported here, not at module scope: ``control_system.base`` is the
    # reference monitor that calls back into this module, and the two must not
    # import each other at load time.
    from osprey_connectors.control_system.base import is_readonly_run

    if is_readonly_run():
        return False
    return store_permits(session_key, target)
