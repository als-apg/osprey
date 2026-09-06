"""The write-posture reader — one field of one record, three readers.

An operator narrows one control target from the header chip: "stand-in is
read-only, leave the virtual accelerator alone". That narrowing is one field
of the control-context record (:mod:`osprey_connectors.control_context`) and
is enforced at *write* time rather than delivered by respawning the agent,
which is what lets a flip land on a session already mid-conversation.

This module is the canonical reader of that field. Two others answer from the
same file and must agree with it byte for byte: the controls MCP server (which
imports this one) and the stdlib-only PreToolUse hook in
``osprey/templates/claude_code/claude/hooks/osprey_target_state.py``, which
cannot import anything and therefore *restates* the contract below. Treat the
following four rules as the contract; a change here is a change there.

**1. Where the file is.** :data:`STATE_DIR_NAME` under the agent-data root,
so one directory answers "control state for this deployment". The root
resolves by ONE rule, used by every writer and every reader:

* :data:`AGENT_DATA_ROOT_ENV_VAR` when it names a non-blank path — the stamp
  a session child carries;
* otherwise :func:`~osprey_connectors.workspace.resolve_shared_data_root`,
  the config derivation the state file has always used.

There is no third path and no fallback. A root that resolves to nothing makes
:func:`state_dir` ``None``, which a route surfaces as ``store_unavailable``
rather than quietly writing somewhere a reader will never look. For *reading*,
an unresolvable root is indistinguishable from a record that narrows nothing:
the deployment ceiling stays in charge.

**2. What the shapes mean.** The record's ``posture`` field is either a
per-target object (``{"live": "sandbox"}``) or the bare string ``"sandbox"``,
which narrows EVERY target in :data:`CONTROL_TARGETS`. A bare ``"writes"`` is
dropped — the writes posture is the *absence* of a narrowing, never a stored
assertion, so nothing in this field can ever widen. Anything else — an unknown
value, a non-string key, a per-target leaf nobody recognises — is dropped
rather than honoured: what survives this filter decides whether a real machine
is written to, so a hand-edited or future-version field must not reach the
decision. :func:`parse_posture_value` is that filter, applied to the record's field by
:func:`~osprey_connectors.control_context.parse_posture` rather than restated
there — two filters that disagree about which narrowings survive is a
narrowing that silently does not apply.

**3. How a lookup combines.** :func:`effective_writes` is the whole rule:

    ceiling ∧ not is_readonly_run() ∧ (recorded posture ≠ sandbox)

The ceiling is the deployment's own posture, read through the existing
predicates and never re-derived here — by connector type when the caller is a
connector, by target otherwise, and the union across configured targets when
the caller holds neither. The record can only narrow it. With no resolvable
target the MOST RESTRICTIVE narrowing wins (any sandbox refuses), because a
caller that cannot say which machine it is about must not be granted the most
permissive answer.

The posture is a property of the DEPLOYMENT, not of a process tree inside it:
no session key indexes it, and a process that belongs to no session — a
dispatch worker, a CLI run — is narrowed by the same field as one that does.
``OSPREY_POSTURE_SESSION`` remains the audit session id and nothing more.

Inside an executor sandbox one further term is ANDed in, and only there:
:data:`LAUNCH_POSTURE_ENV_VAR`, the posture the run was LAUNCHED under. It is
not part of the restated rule — no hook process ever carries the stamp, and no
process that lacks it can be narrowed by it — but it is part of
:func:`store_permits`, because the sandbox's own reference monitor asks through
that function. Its job is asymmetry: a narrowing that lands mid-run is honoured
by the record read, while a WIDENING never reaches a run that started narrow,
which would otherwise hand a running script write access to a machine the
operator took away from it. See :func:`launch_permits`.

The record clause alone is :func:`store_permits`, public for the one caller
that holds a ceiling this module cannot derive — the connector's reference
monitor reads a deployment posture keyed on connector TYPE. It delegates that
clause here rather than restating it, so rule 3 keeps exactly two
implementations.

**4. When a change is seen.** Every read re-stats the record and re-parses on
``(st_mtime_ns, st_size, st_ino)`` — the cache lives in
:mod:`~osprey_connectors.control_context`, and :func:`invalidate_cache` is the
hook a process that moved roots (or a test) drops it through. A missing record
is no narrowing, not an error.

This module reads no config for the posture half, holds no cache of its own,
and imports nothing from ``osprey``: it runs inside the connector-host child
and the executor sandbox, where the dependency budget is the lean connector
chain. The record reader is imported inside the functions that need it,
because the import direction between the two modules is one-way — the record
module imports THIS one at module scope for the root and the grammar.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from osprey_connectors.types import (
    CONTROL_TARGETS,
    any_target_writes_enabled,
    target_writes_enabled,
    type_writes_enabled,
)
from osprey_connectors.workspace import resolve_shared_data_root

logger = logging.getLogger("osprey_connectors.posture_store")

__all__ = [
    "AGENT_DATA_ROOT_ENV_VAR",
    "LAUNCH_POSTURE_ALL_TARGETS",
    "LAUNCH_POSTURE_ENV_VAR",
    "POSTURE_SANDBOX",
    "POSTURE_WRITES",
    "STATE_DIR_NAME",
    "VALID_POSTURES",
    "agent_data_root",
    "effective_writes",
    "invalidate_cache",
    "launch_narrowed_target",
    "launch_permits",
    "launch_posture_stamp",
    "parse_launch_posture",
    "parse_posture_value",
    "recorded_posture",
    "stamped_agent_data_root",
    "state_dir",
    "store_permits",
    "target_posture",
]

#: The anchor stamp. Read by NAME rather than imported from
#: ``osprey.audit.posture``, which declares it for the stamping side: this
#: package is the lean connector chain and must not grow an ``osprey`` import
#: to learn one string.
AGENT_DATA_ROOT_ENV_VAR = "OSPREY_AGENT_DATA_ROOT"

#: Subdirectory of the agent-data root holding the control-context record, the
#: per-server reports and the request files. The same directory
#: ``target_state.state_dir()`` resolves, spelled here because a record the
#: writer puts in one directory and a reader looks for in another is a
#: narrowing that silently never applies.
STATE_DIR_NAME = "control_target"

#: The narrowing value. The only one that ever refuses anything.
POSTURE_SANDBOX = "sandbox"

#: The un-narrowed value. Recorded by some writers, meaningful to none: it is
#: dropped on parse so that the record holds narrowings and nothing else.
POSTURE_WRITES = "writes"

#: The two values a posture entry may spell. Everything else is dropped.
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
#:   one, in which case the pin covers every target, exactly as a bare
#:   ``"sandbox"`` in the record does;
#: * ``<posture>`` is :data:`POSTURE_SANDBOX` or :data:`POSTURE_WRITES`.
#:
#: ``sandbox`` is the only value that does anything: ``writes`` is recorded so
#: the in-flight marker states what the run launched under, and is inert here
#: for the same reason a recorded ``"writes"`` is dropped — nothing may widen.
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


# -- path resolution --------------------------------------------------------


def stamped_agent_data_root() -> Path | None:
    """The :data:`AGENT_DATA_ROOT_ENV_VAR` stamp as a path, or ``None`` unstamped.

    The stamp half of rule 1 on its own, so that the files under this root
    cannot read one environment variable two different ways. A blank or
    whitespace-only stamp is no stamp, and a ``~`` in it is expanded: a reader
    that took either literally would look in a directory no writer ever
    creates, which is the silent-no-narrowing failure rule 1 exists to prevent.

    ``osprey.mcp_server.control_system.target_state.state_dir`` calls this
    rather than restating it. The CONFIG half below is deliberately not shared:
    that module raises where this one answers ``None``, and a route turns its
    raise into a ``store_unavailable`` 503.
    """
    stamped = (os.environ.get(AGENT_DATA_ROOT_ENV_VAR) or "").strip()
    return Path(stamped).expanduser() if stamped else None


def agent_data_root() -> Path | None:
    """The agent-data root the control state lives under.

    Rule 1 of the module contract: the :data:`AGENT_DATA_ROOT_ENV_VAR` stamp
    when it names a non-blank path, else the config derivation. ``None`` when
    neither answers — a deployment whose project root cannot be resolved has
    no record, which is a different thing from having one that narrows nothing.
    """
    stamped = stamped_agent_data_root()
    if stamped is not None:
        return stamped
    try:
        return resolve_shared_data_root()
    except Exception:  # noqa: BLE001 — an unresolvable root is "no record", not a crash
        logger.debug("Could not resolve the shared data root for the control state", exc_info=True)
        return None


def state_dir() -> Path | None:
    """:data:`STATE_DIR_NAME` under :func:`agent_data_root`, or ``None``.

    A ``None`` here is what a route reports as ``store_unavailable``: there is
    nowhere to record a narrowing that any reader would find.
    """
    root = agent_data_root()
    return None if root is None else root / STATE_DIR_NAME


# -- parsing ----------------------------------------------------------------


def parse_posture_value(value: Any) -> dict[str, str]:
    """One posture value as ``{target: "sandbox"}`` — narrowings only.

    Rule 2 of the module contract, in one place. The record carries exactly one
    posture value and :func:`~osprey_connectors.control_context.parse_posture`
    reaches this filter for it rather than restating the grammar: two filters
    that disagree about which narrowings survive is a narrowing that silently
    does not apply.

    Never raises. Anything that narrows nothing — a bare ``"writes"``, an
    unknown string, a non-string target key, a hand-edited or future-version
    shape — returns an empty map, because the alternative (every write and
    every toggle failing on a field nobody can repair from the browser) is
    worse than losing narrowings an operator can set again.
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


# -- reading ----------------------------------------------------------------


def recorded_posture() -> dict[str, str]:
    """The narrowings the control-context record carries — ``{target: "sandbox"}``.

    Rules 1, 2 and 4 together, and the one place this module reaches the record.
    An empty map for every way of not knowing: no agent-data root, no record, a
    corrupt or unreadable one, a field nobody recognises. That is not a grant —
    a narrowing can only refuse, so failing to read one leaves whatever the
    deployment ceiling and the run's own environment already decided.
    """
    try:
        # Imported here, not at module scope: ``control_context`` imports THIS
        # module for the root and the posture grammar, and the two must not
        # import each other at load time. Same reason as ``effective_writes``'s
        # local import of ``control_system.base``.
        from osprey_connectors import control_context

        record = control_context.read_record()
    except Exception:  # noqa: BLE001 — every reader here sits on a write path
        logger.debug("Control-context record unavailable; nothing is narrowed", exc_info=True)
        return {}
    return {} if record is None else record.posture


def target_posture(target: str | None) -> str | None:
    """The recorded posture for one target, or ``None`` when it is unnarrowed."""
    if not target:
        return None
    return recorded_posture().get(target)


def invalidate_cache() -> None:
    """Forget the parsed record. For tests and for a process that moved roots.

    The cache is the record reader's — this module holds none — so this is the
    hook, not a second cache. Kept here because the callers that drop it (a
    notebook kernel restamping its root, every fixture that writes a posture)
    are readers of this module and should not have to know where the parse is
    memoised.
    """
    try:
        from osprey_connectors import control_context

        control_context.invalidate_cache()
    except Exception:  # noqa: BLE001 — dropping a cache must not raise into a caller
        logger.debug("Could not drop the control-context cache", exc_info=True)


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
            recorded answer for that target at the moment of launch.
    """
    return f"{target or LAUNCH_POSTURE_ALL_TARGETS}={launch_posture}"


def parse_launch_posture(raw: str | None) -> dict[str, str]:
    """Decode a launch stamp into ``{target: "sandbox"}`` — narrowings only.

    The same filter rule 2 applies to the record, on the environment's one-pair
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

    Shared by the record clause and the launch clause so the "most restrictive
    narrowing wins when the caller cannot name a target" rule has one
    implementation rather than one per source of narrowings.
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
    nothing ever re-derives it: the executor computes the recorded answer once,
    at launch, and no reader here consults the record again on its behalf. So a
    narrowing that lands mid-run is enforced by the record read beside this one,
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


def store_permits(target: str | None) -> bool:
    """The recorded clause of :func:`effective_writes` — rule 3, on its own.

    Public because one caller needs this clause WITHOUT the ceiling
    :func:`effective_writes` derives: the connector's reference monitor
    (``control_system.base``) reads a deployment ceiling keyed on the connector
    TYPE, which is not a ceiling this module can produce from a target. It
    therefore ANDs its own ceiling with this function rather than restating the
    combining terms below — the contract's rule 3 has exactly two
    implementations, this one and the stdlib restatement in the hooks, and a
    third would be a third thing to keep in step.

    Answers ``True`` — permitted — for everything that is not an actual
    narrowing: no record, or a record whose posture does not name this target.
    It can only refuse; nothing here widens a ceiling.

    Inside an executor sandbox the launch pin (:func:`launch_permits`) is ANDed
    in ahead of the record read, so a run that started narrow stays narrow even
    after the operator widens the record under it. Everywhere else that term is
    inert, because only the executor stamps it.

    Args:
        target: The control target the write lands on, or ``None`` when the
            caller cannot name one — in which case the most restrictive
            narrowing in the record decides.
    """
    # The launch pin first: it is a fact about THIS RUN rather than about the
    # deployment, and it costs one environment read, so a sandbox that launched
    # narrow refuses without touching the disk.
    if not launch_permits(target):
        return False
    # No resolvable target: the most restrictive narrowing wins.
    return _permits(recorded_posture(), target)


def effective_writes(
    section: Any,
    target: str | None = None,
    *,
    connector_type: str | None = None,
) -> bool:
    """Whether a write may proceed here and now.

    ``ceiling ∧ not is_readonly_run() ∧ (recorded posture ≠ sandbox)`` — rule 3
    of the module contract, spelled once so that the connector's reference
    monitor, the executor's gate, the tool roster and the popover cannot
    answer it differently.

    Args:
        section: The ``control_system:`` config section the ceiling is read
            from, through the existing deployment predicates.
        target: The control target this write lands on, when the caller knows
            it. ``None`` takes the most restrictive narrowing.
        connector_type: The connector type, for a caller that IS a connector.
            Given, it decides the ceiling — the deployment half stays keyed by
            type — while *target* still indexes the posture.

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
    return store_permits(target)
