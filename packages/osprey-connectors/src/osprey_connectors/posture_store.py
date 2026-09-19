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
following five rules as the contract; a change here is a change there.

**1. Where the file is.** One directory per identity:
:func:`~osprey_connectors.identity.acting_identity` under
:data:`STATE_DIR_NAME` under the agent-data root, so one directory answers
"control state for whoever this process is acting as". The narrowing is one
operator's, not the deployment's, and a single shared directory would make one
user's chip decide every other user's writes. The directory resolves by ONE
ladder, used by every writer and every reader:

* :data:`CONTROL_CONTEXT_DIR_ENV_VAR` when it names a non-blank path — the
  container's own state directory, bound in by compose. It is already the
  per-identity directory, so nothing is appended to it: a container sees its
  own dir at a path that has nothing to do with the host's tree, and deriving
  the identity hop inside it would name a directory nobody binds;
* otherwise :data:`AGENT_DATA_ROOT_ENV_VAR` when it names a non-blank path —
  the stamp a session child carries — else
  :func:`~osprey_connectors.workspace.resolve_shared_data_root`, the config
  derivation the state file has always used; and then the two fixed hops,
  :data:`STATE_DIR_NAME` and the acting identity.

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

The record clause alone is :func:`store_verdict`, which answers a
:class:`StoreVerdict` — the reason a write may or may not proceed, not only
whether it may — with :func:`store_permits` its bool spelling for the callers
that have to decide whether to proceed and not what to say about it, and
:func:`store_verdict_detail` the spelling that also hands back the operator
sentence for the one verdict whose remedy the refusing process cannot compose
for itself. All three are one read of the record, so a caller pays nothing for
asking in the spelling that says the most. The clause is public for the one
caller that holds a ceiling this module cannot derive: the connector's
reference monitor reads a deployment posture keyed on connector TYPE. That
caller delegates the clause here rather than restating it, so rule 3 keeps
exactly two implementations.

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

**5. Who a narrowing is about.** This rule is not part of what the hook
restates, and not because a hook could never see its inputs: a container that
holds no chip of its own carries both the tree bind and the owner, and the
agent it spawns inherits them, hooks included. It is a division of labour. A
hook answers for the CONTAINER — the deployment ceiling it runs under, which
is the one question it can settle without a tree read that has to fail closed —
and the connector's reference monitor answers for the PERSON, delegating the
record clause to this module. Two answers to one question would be two places
to get it wrong, and the stricter of them is the one held closest to the write.
A chip is one person's, so a lookup needs a
name as well as a target. That name is the OWNER, and it is resolved by one
ladder here — :func:`current_owner`, with :data:`NO_OWNER` for work that belongs
to nobody — so that the write monitor and the queue never disagree about whose
chip a plan runs under. The owner of a queued plan rides on the item as
:data:`RESERVED_OWNER_KWARG` and is bound for the wrapper's body by
:func:`bind_owner`.

A container that holds no chip of its own — a lane's queueserver, the dispatch
worker — reads that owner's record out of a read-only bind of the WHOLE tree,
:data:`CONTROL_CONTEXT_TREE_ENV_VAR`, rather than out of a directory of its own.
That read fails CLOSED, which is the one place this module departs from rule 2:
on the host, the process reading the record is the one whose chip wrote it, so
every way of not knowing is fairly reported as "nothing is narrowed"; across the
bind it is not, because an unprovisioned mount, a group the container could not
join and a record written 0600 all look exactly like an operator who narrowed
nothing, while being in fact an operator whose narrowing nobody could read. So
the tree reader separates the one absence that is an answer — no record for this
owner — from every failure on a path that is THERE, and the latter refuses.
:data:`CONTROL_TREE_MARKER_NAME` is what makes the first question answerable at
all: a bind whose source does not exist is created empty by the container
runtime, and an empty directory says nothing about whether it was ever built.

The invariant that binding rests on: **the owner variable is read on the
connector's async path, never after a thread or executor hop.** A context is
copied into a task when the task is created, so the binding reaches everything
a RunEngine schedules from the wrapper's context. An offload is where that
stops being true, and which helper made it decides whether it does:
``asyncio.to_thread`` copies the context and carries the binding into the
thread, while ``loop.run_in_executor``, a raw executor submit and a bare
``threading.Thread`` do not. No gate may depend on the carrying one having been
chosen. A verdict taken behind such a hop could read :data:`NO_OWNER` and
quietly grant the deployment ceiling to a narrowed user; every verdict is
therefore taken before the offload.
"""

from __future__ import annotations

import contextvars
import errno
import logging
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, NamedTuple

from osprey_connectors.identity import acting_identity
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
    "CONTROL_CONTEXT_DIR_ENV_VAR",
    "CONTROL_CONTEXT_TREE_ENV_VAR",
    "CONTROL_OWNER_ENV_VAR",
    "CONTROL_TREE_MARKER_NAME",
    "LAUNCH_POSTURE_ALL_TARGETS",
    "LAUNCH_POSTURE_ENV_VAR",
    "NO_OWNER",
    "POSTURE_SANDBOX",
    "POSTURE_WRITES",
    "RESERVED_OWNER_KWARG",
    "STATE_DIR_NAME",
    "StoreVerdict",
    "StoreVerdictDetail",
    "VALID_POSTURES",
    "agent_data_root",
    "bind_owner",
    "bound_state_dir",
    "current_owner",
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
    "store_verdict",
    "store_verdict_detail",
    "target_posture",
]

#: The anchor stamp. Read by NAME rather than imported from
#: ``osprey.audit.posture``, which declares it for the stamping side: this
#: package is the lean connector chain and must not grow an ``osprey`` import
#: to learn one string.
AGENT_DATA_ROOT_ENV_VAR = "OSPREY_AGENT_DATA_ROOT"

#: Subdirectory of the agent-data root holding one directory per identity, each
#: with that identity's control-context record, per-server reports and request
#: files. The same tree ``target_state.state_dir()`` resolves, spelled here
#: because a record the writer puts in one directory and a reader looks for in
#: another is a narrowing that silently never applies.
STATE_DIR_NAME = "control_target"

#: The container's OWN state directory, bound in by compose and named in the
#: environment of every container that holds a chip: the terminal's
#: ``web-<user>``. Top rung of rule 1, and already the per-identity directory —
#: a container's bind target has nothing to do with the host tree's shape, so
#: the two fixed hops are the host derivation's business and never appended to
#: this. Distinct from :data:`CONTROL_CONTEXT_TREE_ENV_VAR`, the read-only bind
#: of every identity's directory, which names no single one of them.
#:
#: It belongs in ``NON_PINNABLE_AUDIT_MARKERS`` for the same reason the audit
#: identity does: a spec that could pin it could point one user's writes at
#: another user's narrowing.
CONTROL_CONTEXT_DIR_ENV_VAR = "OSPREY_CONTROL_CONTEXT_DIR"

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

#: The stamped owner of the work a process is doing, exported into a dispatch
#: job's environment by the dispatch worker's ``sdk_runner`` — the one export
#: site — and read back as the third rung of :func:`current_owner`. Always a
#: ``str`` when it is set at all: the absence of an owner is :data:`NO_OWNER`,
#: which has no spelling, so a blank value here is the unset case and not a
#: nameless owner.
#:
#: It belongs in ``NON_PINNABLE_AUDIT_MARKERS`` for the same reason the audit
#: identity does: a spec that could pin the owner could hand a plan another
#: user's narrowing — or take one away.
CONTROL_OWNER_ENV_VAR = "OSPREY_CONTROL_OWNER"

#: The read-only bind of the whole per-user control-state tree, named in the
#: environment of a container that holds no chip of its own: a lane's
#: queueserver, the dispatch worker. Only its PRESENCE is read here; the tree is
#: read by the verdict.
#:
#: Presence is what makes such a container owned-or-nothing. The account it runs
#: as names nobody whose chip anyone ever set, so falling through to
#: :func:`~osprey_connectors.identity.acting_identity` there would read a record
#: directory that does not exist and call the result "no narrowing" — a plan
#: would then run at the deployment ceiling however its owner set their chip.
CONTROL_CONTEXT_TREE_ENV_VAR = "OSPREY_CONTROL_CONTEXT_TREE"

#: The marker file at the root of the control-context tree, written by
#: ``osprey build``. Its PRESENCE is the tree reader's first question, because a
#: read-only bind whose host source does not exist is created empty by the
#: container runtime — after which an unprovisioned tree is indistinguishable
#: from a tree in which nobody has narrowed anything, and the chip fails OPEN on
#: exactly the deployment that never provisioned it.
#:
#: Restated here rather than imported from
#: ``osprey.deployment.compose_generator``, which declares it for the writing
#: side: this package is the lean connector chain and may not import the
#: deployment layer. The two spellings are pinned together by a test that
#: imports both — a drift between them is a tree that reads as unprovisioned in
#: every container.
CONTROL_TREE_MARKER_NAME = ".osprey-control-tree"

#: The plan kwarg an owner rides on into a queueserver worker, popped by
#: :func:`bind_owner` before the plan's own signature is validated against it.
#: Reserved: it is never a plan parameter and never appears in a relayed view of
#: a queue item.
#:
#: Spelled here rather than in the bridge because the import direction between
#: the two is one-way — the bridge imports this name; nothing in this package
#: may import the bridge — and a reserved key restated on the far side is a key
#: one end strips while the other leaks it into a plan's arguments.
RESERVED_OWNER_KWARG = "_osprey_owner"


# -- owner ------------------------------------------------------------------


class _NoOwner:
    """The absence of an owner: one instance, compared by identity and nothing else.

    ``None`` cannot carry this meaning, because :func:`current_owner` already
    uses ``None`` for "the caller named nobody, so walk the ladder" — a
    different question from "the ladder ran out". A string cannot carry it
    either: every string is a name some deployment may legitimately give an
    account, so an owner spelled ``""`` or ``"none"`` would be indistinguishable
    from a real one at the directory a verdict reads.

    There is deliberately no parse in the other direction: nothing turns
    ``"<no owner>"`` back into this value. The string form exists so that a
    warning line about an owner-less plan can print it — a raising ``__str__``
    would cost the one line that says the plan ran at the ceiling — and printing
    is all it is for. Callers ask ``owner is NO_OWNER``, never truthiness (a real
    name can be falsey only by being empty, which this module never produces)
    and never equality with the text.
    """

    __slots__ = ()

    def __str__(self) -> str:
        return "<no owner>"

    def __repr__(self) -> str:
        return "<no owner>"


#: The answer every owner-less path gives. See :class:`_NoOwner`.
NO_OWNER: Final = _NoOwner()

#: Who the work running in this context belongs to.
#:
#: Set by :func:`bind_owner` around a plan wrapper's body and read by the write
#: monitor through :func:`current_owner`. A context variable rather than a
#: module global because a queueserver worker runs one plan at a time while the
#: processes that share this module do not: an owner that leaked between
#: concurrent tasks would gate one user's write against another user's chip.
#:
#: **The owner is read on the connector's async path, never after a thread or
#: executor hop.** A context is copied when a task is created, so the value
#: reaches everything a RunEngine schedules from the wrapper's own context,
#: including the task a device's ``AsyncStatus`` wraps a write into. Off that
#: path it depends on the helper the caller reached for: ``asyncio.to_thread``
#: copies the context and carries the owner into the worker thread, while
#: ``loop.run_in_executor``, a raw executor ``submit`` and a bare
#: ``threading.Thread`` start from a fresh context and read the default. The
#: gate rests on neither — every store verdict is decided before any offload, so
#: the rule above holds whichever helper is used, and a gate moved behind a hop
#: would start reading :data:`NO_OWNER` for the hops that do not copy.
_owner_var: contextvars.ContextVar[str | _NoOwner] = contextvars.ContextVar(
    "osprey_control_owner", default=NO_OWNER
)


def current_owner(owner: str | _NoOwner | None = None) -> str | _NoOwner:
    """The owner a narrowing should be looked up for, by a four-rung ladder.

    Most specific first:

    1. *owner* when the caller named one, stripped — the same normalisation the
       two rungs below apply, so a padded name and a bare one address one
       person's directory rather than two. Passing :data:`NO_OWNER` explicitly
       is naming one — it says "this work belongs to nobody, read the ceiling
       only" — which is why the "unnamed" spelling is ``None`` and not the
       sentinel. A caller that names an owner with no name (``""``, or only
       spaces) is answered with that empty name rather than falling through to
       the rungs below: the fall-through is for a caller that named nobody, and
       a name no directory can hold is a lookup that fails rather than one that
       is not attempted.
    2. The bound context variable, set by :func:`bind_owner` from the reserved
       kwarg a queued plan carried. This is the rung that makes a lane's plan
       obey the chip of the person who queued it.
    3. :data:`CONTROL_OWNER_ENV_VAR`, the stamp a dispatch job's environment
       carries. Always answered as a ``str``; a blank or whitespace-only value is
       the unset case spelled differently — a rendered-but-empty ``environment:``
       entry — and falls through rather than naming an owner with no name.
    4. Otherwise: :data:`NO_OWNER` when a tree is bound (:func:`_tree_is_bound`,
       the one spelling of that question), else
       :func:`~osprey_connectors.identity.acting_identity`. A container holding
       the tree holds no chip of its own, so the account it runs as is not an
       answer; anywhere else the process account *is* the person whose chip the
       record was written from.

    Never raises: this is read on a write path, and a lookup that cannot name an
    owner must produce a verdict rather than an exception.
    """
    if isinstance(owner, str):
        return owner.strip()
    if owner is not None:
        return owner

    bound = _owner_var.get()
    if bound is not NO_OWNER:
        return bound

    stamped = os.environ.get(CONTROL_OWNER_ENV_VAR)
    if isinstance(stamped, str) and stamped.strip():
        return stamped.strip()

    if _tree_is_bound():
        return NO_OWNER

    return acting_identity()


@contextmanager
def bind_owner(kwargs: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Bind the owner riding on *kwargs* for the body, yielding the kwargs without it.

    The one way an owner enters a queueserver worker: the bridge puts it on the
    item as :data:`RESERVED_OWNER_KWARG`, the worker hands those kwargs to the
    plan wrapper, and the wrapper runs its body inside this block. The yielded
    mapping is what the plan's own signature is validated against — the reserved
    key is gone by then, so a plan never sees an argument it does not declare.

    *kwargs* itself is not mutated. The caller keeps whatever it was handed, and
    two wrappers reading the same mapping cannot race over who pops the key.

    A missing key, a blank one, or one carrying anything that is not a string
    binds :data:`NO_OWNER`: the plan reached the queue without a name, and the
    ceiling is all that governs it.

    The reset is an assignment to :data:`NO_OWNER`, not a token restore. Plan
    wrappers are generators, so the ``set`` and the ``finally`` can run in
    different contexts, and a token from one context cannot be reset in another.
    There is nothing to restore in any case: owners do not nest — a run belongs
    to one person or to nobody — so the floor is the only correct value to leave
    behind, and leaving it is what stops one run's owner from gating the next
    run's writes.
    """
    clean = {key: value for key, value in kwargs.items() if key != RESERVED_OWNER_KWARG}
    claimed = kwargs.get(RESERVED_OWNER_KWARG)
    owner: str | _NoOwner = NO_OWNER
    if isinstance(claimed, str) and claimed.strip():
        owner = claimed.strip()

    _owner_var.set(owner)
    try:
        yield clean
    finally:
        _owner_var.set(NO_OWNER)


# -- path resolution --------------------------------------------------------


def _absolute_bind(variable: str) -> Path | None:
    """The bind *variable* names as an absolute path, or ``None`` when it names none.

    The one normalisation every bind in rule 1 goes through, so the container's
    own state directory and the read-only tree cannot read their environment two
    different ways.

    Three answers are "no bind": unset, blank or whitespace-only, and a value
    that is not absolute after expansion. A bind names a path in the container's
    own filesystem — the compose file is where it is written — so a relative one
    would resolve against whatever directory the process happened to start in,
    which is a directory nothing binds and no writer creates. It is a deployment
    mistake rather than a degraded state, and it gets the one warning line that
    says which variable to fix; falling back to the derivation below it is what
    keeps a mistake here from raising into a write path.

    Expansion is :func:`os.path.expanduser` rather than
    :meth:`pathlib.Path.expanduser`, which raises ``RuntimeError`` for a ``~``
    naming a user this container has no passwd entry for — a container binds
    paths from another host's layout, so that is an ordinary value here and must
    degrade to "no bind" rather than to an exception. The stdlib hook restates
    this rule with the same call.
    """
    raw = (os.environ.get(variable) or "").strip()
    if not raw:
        return None
    expanded = os.path.expanduser(raw)
    if not os.path.isabs(expanded):
        logger.warning(
            "%s=%s is not an absolute path, so it names no bind: a bind is a path "
            "in this container's own filesystem. Set it to an absolute path or "
            "unset it.",
            variable,
            raw,
        )
        return None
    return Path(expanded)


def stamped_agent_data_root() -> Path | None:
    """The :data:`AGENT_DATA_ROOT_ENV_VAR` stamp as a path, or ``None`` unstamped.

    The stamp half of rule 1 on its own, so that the files under this root
    cannot read one environment variable two different ways. A blank or
    whitespace-only stamp is no stamp, and a ``~`` in it is expanded: a reader
    that took either literally would look in a directory no writer ever
    creates, which is the silent-no-narrowing failure rule 1 exists to prevent.
    The expansion is :func:`os.path.expanduser`, for the reason
    :func:`_absolute_bind` uses it: the ``Path`` method raises for a ``~``
    naming a user this process cannot resolve, and this is read on a write path.

    Unlike a bind, a stamp is not required to be absolute: it is written by a
    process that shares this one's working directory rather than by a compose
    file describing another host's layout.

    ``osprey.mcp_server.control_system.target_state.state_dir`` calls this
    rather than restating it. The CONFIG half below is deliberately not shared:
    that module raises where this one answers ``None``, and a route turns its
    raise into a ``store_unavailable`` 503.
    """
    stamped = (os.environ.get(AGENT_DATA_ROOT_ENV_VAR) or "").strip()
    return Path(os.path.expanduser(stamped)) if stamped else None


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


def bound_state_dir() -> Path | None:
    """The :data:`CONTROL_CONTEXT_DIR_ENV_VAR` bind as a path, or ``None`` unset.

    The top rung of rule 1 on its own, so that the three seams resolving this
    directory cannot read one environment variable three different ways. Blank,
    whitespace-only and non-absolute values all name no bind, and a ``~`` is
    expanded — :func:`_absolute_bind` is that rule, shared with the tree bind
    beside it.

    ``osprey.mcp_server.control_system.target_state.state_dir`` calls this
    rather than restating it, exactly as it calls
    :func:`stamped_agent_data_root`.
    """
    return _absolute_bind(CONTROL_CONTEXT_DIR_ENV_VAR)


def state_dir() -> Path | None:
    """This identity's control-state directory, or ``None``.

    Rule 1 of the module contract: the :func:`bound_state_dir` bind when a
    container carries one, else the acting identity under
    :data:`STATE_DIR_NAME` under :func:`agent_data_root`.

    The identity is resolved on every call rather than once at import, because
    :func:`~osprey_connectors.identity.acting_identity` reads an environment
    that compose and the entrypoint set per process; a value frozen at import
    would be whatever the first importer happened to see. It never fails —
    an unresolvable identity is :data:`~osprey_connectors.identity.UNKNOWN_IDENTITY`,
    a real directory holding the state of whoever could not be named, which is
    a narrowing that applies rather than one that silently does not.

    A ``None`` here is what a route reports as ``store_unavailable``: there is
    nowhere to record a narrowing that any reader would find.
    """
    bound = bound_state_dir()
    if bound is not None:
        return bound
    root = agent_data_root()
    return None if root is None else root / STATE_DIR_NAME / acting_identity()


def _tree_is_bound() -> bool:
    """Whether this process was handed the read-only tree at all — usable or not.

    The ONE spelling of that question, because two readings of it are a
    split-brain on the write path: :func:`current_owner` answers
    :data:`NO_OWNER` for a container that holds the tree (it holds no chip of
    its own), while :func:`store_verdict` picks the reader that governs the
    write. A variable that made one of them say "tree" and the other say "host"
    would send a container with no host record to the host reader, whose every
    way of not knowing is "nothing is narrowed" — the fail-open the tree reader
    exists to close.

    So this asks only whether a value is THERE. Whether the value is usable is
    :func:`_control_context_tree`'s question, and a bind that is set but
    unusable is a deployment mistake that refuses rather than falls back.
    """
    return bool((os.environ.get(CONTROL_CONTEXT_TREE_ENV_VAR) or "").strip())


def _control_context_tree() -> Path | None:
    """The :data:`CONTROL_CONTEXT_TREE_ENV_VAR` bind as a path, or ``None`` unset.

    The read-only bind of EVERY identity's directory, carried by a container
    that holds no chip of its own. It is not a rung of :func:`state_dir`: it
    names no single identity's directory, so nothing reads a record through it
    without first being told whose record to read — which is what
    :func:`_read_tree_record` is for.

    Normalised by :func:`_absolute_bind`, the same rule the container's own
    state directory goes through, so the two binds cannot disagree about what a
    ``~`` or a relative path means.
    """
    return _absolute_bind(CONTROL_CONTEXT_TREE_ENV_VAR)


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


# -- the read-only tree -----------------------------------------------------


#: Characters that cannot appear in the one path segment an owner names. The
#: same rule :func:`~osprey_connectors.identity.acting_identity` applies to an
#: identity, restated for the name a queue item carries rather than shared with
#: it, because that function ANSWERS a name while this one screens one somebody
#: else supplied.
_TREE_UNSAFE: Final = frozenset({"/", "\\", "\x00"})


class _TreeRead(NamedTuple):
    """One read of one owner's record out of the read-only tree.

    Attributes:
        narrowed: The narrowings that record carries, ``{target: "sandbox"}``,
            in the same grammar :func:`recorded_posture` answers in and empty
            when nothing is narrowed. Empty whenever *unavailable* is set: a
            read that did not reach the record knows of no narrowing and must
            not be combined as though it did.
        unavailable: ``None`` when the read reached an answer — a record, or a
            provisioned tree holding none for this owner — and otherwise the
            operator-facing sentence saying what could not be read and what to
            do about it. The caller turns a non-``None`` into its own
            ``control_context_unavailable`` verdict; the reason vocabulary is
            the verdict's, and the remedy is this reader's, because only the
            read knows which of the four ways of not knowing happened.
    """

    narrowed: dict[str, str]
    unavailable: str | None


def _tree_unprovisioned(tree: Path, why: str) -> str:
    """The remedy for a tree that was never built: *why* names how it shows."""
    return (
        f"the control-context tree {tree} {why}, so it is not provisioned — "
        "run `osprey build` on the deployment"
    )


def _tree_unreadable(tree: Path) -> str:
    """The remedy for a tree this container cannot get into."""
    return (
        f"the control-context tree {tree} is not readable by this container — "
        "check the group join; on Docker Desktop the bind's ownership is remapped, "
        "so share the tree through a group the container can join or run the "
        "worker on a Linux host"
    )


def _tree_record_unreadable(record_file: Path, detail: str) -> str:
    """The remedy for a record that is THERE and could not be used: *detail* says how."""
    return (
        f"the control-context record {record_file} is present but {detail} — "
        "check the file's mode and group"
    )


def _tree_record_oversized(record_file: Path) -> str:
    """The remedy for a record too large to be the one a chip wrote.

    Its own sentence rather than the unreadable one's: the mode and the group
    permitted this read, so an operator sent to check them finds both correct
    and learns nothing. A record is a few hundred bytes, and one past the bound
    is a file something other than the chip put at that name.
    """
    return (
        f"the control-context record {record_file} is larger than the "
        f"{_RECORD_READ_LIMIT} bytes a record may be, so it was not read — "
        "re-select the control target to write the record again"
    )


def _tree_bind_unusable() -> str:
    """The remedy for a bind that is set and names no directory to open.

    Its own wording rather than the unprovisioned one: nothing is missing on
    the deployment's disk, the variable simply does not name a path this
    container can resolve, and the operator repairs the compose file rather
    than building anything.
    """
    return (
        f"{CONTROL_CONTEXT_TREE_ENV_VAR} names no usable path, so the read-only "
        "control-context tree could not be opened — the bind must name an "
        "absolute path in this container"
    )


def _owner_is_not_a_name(owner: object) -> str:
    """The sentence for work whose owner arrived as something other than a name.

    Nothing an operator can flip, and the wording says so: a value of the wrong
    type comes off the wire (a queue item's reserved kwarg is JSON), so what
    repairs it is whatever stamped the owner, not a chip and not the bind.
    """
    return (
        f"the owner {owner!r} is not a name, so no narrowing could be looked up — "
        "check what stamped the owner onto this work"
    )


def _tree_planted_link(path: Path, what: str) -> str:
    """The remedy for a symbolic link where a real *what* belongs.

    Its own wording rather than a mode-and-group one, because the operator has
    to remove something rather than fix a permission, and because a link here is
    the one failure that may be somebody's doing.
    """
    return (
        f"the control-context {what} {path} is a symbolic link, which this reader "
        "will not follow: the tree is group-writable, so a link planted there can "
        "point one account's reader at a record its owner never wrote — remove it "
        "and check the directory's mode and group"
    )


#: The most a record may be and still be read. A record is a few hundred bytes,
#: so this is headroom rather than a limit anything real approaches. It exists
#: because the file lives in a group-writable tree and is re-read on every
#: gated write: without it, the account whose plan is running chooses how much
#: the container allocates per write, and the container is shared with other
#: people's plans.
_RECORD_READ_LIMIT: Final = 64 * 1024


def _read_no_follow(path: Path) -> str:
    """*path*'s text, refusing to follow a symbolic link at its last component.

    ``O_NOFOLLOW`` rather than a :meth:`~pathlib.Path.is_symlink` check before
    an ordinary open: the check and the open are two trips to the filesystem,
    and the directory holding this file is group-writable by parties other than
    its owner, so a link planted between the two would be followed by a reader
    that had just decided there was none. The flag makes the kernel decide it
    once, at the open, and a link raises ``ELOOP`` there.

    The read is bounded at :data:`_RECORD_READ_LIMIT` and an over-long file
    raises rather than being truncated into a parse — a truncated record either
    fails to parse or, worse, parses into a different narrowing than the one on
    disk. The bound is applied to the read itself rather than to a prior
    :func:`os.fstat`, so nothing is allocated for a file that grew between the
    two.

    Raises:
        OSError: The file is absent (``FileNotFoundError``), is a symbolic link
            (``ELOOP``), is longer than the limit (``EFBIG``), or could not be
            read.
        UnicodeDecodeError: The bytes are not UTF-8.
    """
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with os.fdopen(fd, encoding="utf-8", closefd=False) as handle:
            text = handle.read(_RECORD_READ_LIMIT + 1)
    finally:
        os.close(fd)
    if len(text) > _RECORD_READ_LIMIT:
        raise OSError(
            errno.EFBIG,
            f"the file is larger than the {_RECORD_READ_LIMIT} bytes a record may be",
        )
    return text


def _read_tree_record(tree: Path, owner: str) -> _TreeRead:
    """*owner*'s record out of the read-only *tree*, with the reason when there is none.

    The reader a container uses when it holds the whole tree and no chip of its
    own: a lane's queueserver, the dispatch worker. It fails CLOSED, and that is
    the whole difference from the host readers above. :func:`recorded_posture`
    answers "nothing is narrowed" for every way of not knowing, and
    :func:`~osprey_connectors.control_context.read_record` folds an unreadable
    file into ``None`` for the same reason — correct on the host, where the
    process that reads the record is the one whose chip wrote it, and wrong
    here, where an unprovisioned bind, a group the container could not join or a
    record written 0600 would all read as a grant and run somebody's owned plan
    at the deployment ceiling. So this reader goes through neither of them.

    Exactly one failure is not a refusal: an ENOENT on the record. A tree that
    is provisioned and readable and holds no record for this owner is an owner
    who has narrowed nothing, which is what the deployment ceiling is for. Every
    other failure on a PRESENT path — EACCES, EISDIR, a corrupt payload, a
    payload from another schema — is a narrowing that may exist and cannot be
    read, and is reported rather than guessed at.

    **Nothing here follows a symbolic link.** The tree root is group-writable
    (2770) and its group is shared with accounts other than the one reading, so
    any account in it can plant a link under another identity's name; followed,
    that link decides whether a real machine is written to, from a file its
    supposed owner never wrote. The marker and the ``<owner>/`` directory are
    therefore checked with :func:`os.lstat` and the record is opened
    ``O_NOFOLLOW`` — the open is what closes the window between a check and a
    read — and a link found at any of the three answers unavailable NAMING the
    path. Never ENOENT: an absent record means nobody narrowed anything, and a
    planted link is not an absence.

    There is no cache. The parse memo in
    :mod:`~osprey_connectors.control_context` is reached through the reader
    whose ``None`` this one exists to avoid, and a memo of its own would be a
    second thing to invalidate for a file that is a few hundred bytes and is
    read once per verdict.

    Args:
        tree: The tree bind, from :func:`_control_context_tree`.
        owner: The name whose record to read — one path segment, resolved by
            :func:`current_owner` before the call. :data:`NO_OWNER` never
            reaches here: work that belongs to nobody is governed by the
            ceiling alone and reads no record.

    Returns:
        A :class:`_TreeRead`. Never raises: this is read on a write path.
    """
    if not tree.is_dir():
        return _TreeRead({}, _tree_unprovisioned(tree, "does not exist"))
    if not os.access(tree, os.R_OK | os.X_OK):
        return _TreeRead({}, _tree_unreadable(tree))

    marker = tree / CONTROL_TREE_MARKER_NAME
    try:
        marker_mode = os.lstat(marker).st_mode
    except FileNotFoundError:
        return _TreeRead({}, _tree_unprovisioned(tree, f"carries no {CONTROL_TREE_MARKER_NAME}"))
    except OSError:
        return _TreeRead({}, _tree_unreadable(tree))
    if stat.S_ISLNK(marker_mode):
        # A link here would let any account in the tree's group declare an
        # unbuilt tree provisioned, which is the fail-open this marker exists
        # to close.
        return _TreeRead({}, _tree_planted_link(marker, "tree marker"))
    if not stat.S_ISREG(marker_mode):
        return _TreeRead(
            {},
            _tree_unprovisioned(tree, f"carries a {CONTROL_TREE_MARKER_NAME} that is not a file"),
        )

    # An owner that is not one path segment addresses no record in this tree,
    # and must not be answered as an owner who narrowed nothing: the name
    # arrived on a queue item, and a name nobody can look up is a lookup that
    # did not happen rather than a lookup that found nothing.
    if not isinstance(owner, str) or owner.strip() in {"", ".", ".."} or _TREE_UNSAFE & set(owner):
        return _TreeRead(
            {},
            f"the owner {owner!r} is not a name the control-context tree can hold, "
            "so no narrowing can be read for it — the owner a plan carries must be "
            "one directory name",
        )

    # Imported here, not at module scope, for the reason ``recorded_posture``
    # does it: ``control_context`` imports THIS module for the root and the
    # grammar, and the two must not import each other at load time.
    from osprey_connectors import control_context

    owner_dir = tree / owner
    try:
        owner_mode = os.lstat(owner_dir).st_mode
    except FileNotFoundError:
        # The one absence that is an answer: nobody narrowed anything for this
        # owner, and a missing ``<owner>/`` directory says exactly that.
        return _TreeRead({}, None)
    except OSError:
        logger.debug("Could not stat the control-context directory %s", owner_dir, exc_info=True)
        return _TreeRead({}, _tree_unreadable(tree))
    if stat.S_ISLNK(owner_mode):
        # ``O_NOFOLLOW`` on the record below guards its last component only, so
        # the directory above it is checked here: a link at this level redirects
        # the read just as effectively.
        return _TreeRead({}, _tree_planted_link(owner_dir, "directory"))

    record_file = owner_dir / control_context.RECORD_FILENAME
    try:
        record = control_context.parse_record(_read_no_follow(record_file))
    except FileNotFoundError:
        # No record for this owner: the same answer as no directory.
        return _TreeRead({}, None)
    except Exception as exc:  # noqa: BLE001 — every failure here REFUSES
        # Blanket on purpose, and the opposite of the one this reader avoids:
        # an unexpected failure on a present record leaves a narrowing that may
        # exist unread, so it answers unavailable rather than permitted.
        logger.debug("Could not read the control-context record at %s", record_file, exc_info=True)
        if getattr(exc, "errno", None) == errno.ELOOP:
            return _TreeRead({}, _tree_planted_link(record_file, "record"))
        if getattr(exc, "errno", None) == errno.EFBIG:
            # Asked by name for the same reason the planted link is: the size
            # bound refuses a file whose mode and group are beyond reproach, so
            # the generic remedy points at two things an operator then finds
            # correct.
            return _TreeRead({}, _tree_record_oversized(record_file))
        detail = getattr(exc, "strerror", None) or type(exc).__name__
        return _TreeRead({}, _tree_record_unreadable(record_file, f"could not be read ({detail})"))
    if record is None:
        return _TreeRead(
            {},
            _tree_record_unreadable(record_file, "is not a record this version can read"),
        )
    return _TreeRead(dict(record.posture), None)


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


class StoreVerdict(StrEnum):
    """What the recorded clause found — rule 3's answer with the reason for it.

    A ``bool`` says a write was refused; only a name says what to put in front
    of the operator. The three members are the three things this clause can
    find out, and a consumer branches on the member rather than on the wording
    it composes beside it, so the reason a refusal names and the reason the
    code took cannot drift apart.

    ``control_context_unavailable`` is the one that is neither a grant nor
    somebody's decision: a narrowing may exist and could not be read, which the
    tree reader refuses on and which a message must not report as a chip
    somebody set — the remedy for it is a deployment's, not an operator's.

    A :class:`~enum.StrEnum` because the value crosses the connector IPC as a
    bare ``str``: the far side compares the word, where an identity check
    against a member would fail.
    """

    PERMITTED = "permitted"
    NARROWING = "narrowing"
    CONTROL_CONTEXT_UNAVAILABLE = "control_context_unavailable"


#: The remedy for the launch-pin arm of ``control_context_unavailable``. Short,
#: and about the run rather than the deployment: the pin is a fact about one
#: run, nothing on disk is wrong, and a write state set since applies to the
#: next run rather than to one already in flight.
_LAUNCH_PIN_REASON: Final = "pinned everywhere at launch — re-run the script"


class StoreVerdictDetail(NamedTuple):
    """A verdict and, where there is one, the sentence an operator needs with it.

    Attributes:
        verdict: What the clause found. This is what code branches on; a
            consumer that has to act reads only this field.
        reason: ``None`` for every verdict a consumer can word on its own — a
            grant, and a narrowing, whose remedy is the target's chip and
            therefore known to whoever refused. Set only for
            :attr:`StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE`, whose four ways
            of happening have four different remedies and whose facts — the
            record that could not be read, the bind that named no path, the
            owner that was not a name — exist only in this reader. Without this
            field they reach the deployment's log and nowhere else, and the
            person who ran the refused write is often the only one who ever
            sees a message about it.
    """

    verdict: StoreVerdict
    reason: str | None


def store_verdict(target: str | None, owner: str | _NoOwner | None = None) -> StoreVerdict:
    """The recorded clause of :func:`effective_writes` — rule 3, by name alone.

    Exactly :func:`store_verdict_detail`'s verdict, for the callers that have
    to decide what to do and not what to say about it. One read either way: the
    detail is what this asks for, so a caller never pays for the sentence by
    reading the record a second time.

    Args:
        target: The control target the write lands on, or ``None`` when the
            caller cannot name one — in which case the most restrictive
            narrowing decides.
        owner: Whose narrowing governs this write; ``None`` asks
            :func:`current_owner`.

    Returns:
        A :class:`StoreVerdict`. Never raises: this is read on a write path,
        and a clause that cannot decide must produce a verdict.
    """
    return store_verdict_detail(target, owner).verdict


def store_verdict_detail(
    target: str | None, owner: str | _NoOwner | None = None
) -> StoreVerdictDetail:
    """The recorded clause of :func:`effective_writes` — rule 3, with its reason.

    The clause reads, in this order:

    1. The launch pin, before any record is read: it is a fact about THIS RUN
       rather than about the deployment and costs one environment read, so a
       sandbox that launched narrow refuses without touching the disk. Its two
       arms are different reasons, not one — a named target is an operator's
       decision (``narrowing``), while :data:`LAUNCH_POSTURE_ALL_TARGETS` says
       the executor could resolve neither a target nor the record at launch and
       pinned the run everywhere, which nobody decided
       (``control_context_unavailable``).
    2. *owner*, resolved through :func:`current_owner`. :data:`NO_OWNER` is an
       answer rather than a failure: work that belongs to nobody is governed by
       the deployment ceiling alone, and no record is read for it. Anything else
       that is not a name — a value a decoded payload could carry where a string
       was meant — is a lookup that did not happen, and refuses.
    3. That owner's record — out of the read-only tree through
       :func:`_read_tree_record` when a tree is bound, else out of this host's
       own record through :func:`recorded_posture`. A tree bound but unusable
       refuses rather than falling back to the host: the fall-back exists for a
       process that was given no tree, not for one whose tree cannot be opened.
       The two rungs differ in what a failed read means: the tree reader fails
       CLOSED, because a container that cannot read a record would otherwise run
       somebody's owned plan at the deployment ceiling, while a host reader is
       the process whose own chip wrote the record and keeps the fail-open it
       has always had.

    The comparison is target-only for every rung, through :func:`_permits`: a
    record narrowing another target narrows nothing here, so a chip pointed at
    one machine does not stop a plan running on another.

    Args:
        target: The control target the write lands on, or ``None`` when the
            caller cannot name one — in which case the most restrictive
            narrowing decides.
        owner: Whose narrowing governs this write. ``None`` — the usual case —
            asks :func:`current_owner`. An owner names WHO the work belongs to;
            it does not index the posture, which is filed one record per
            identity and holds one narrowing per person.

    Returns:
        A :class:`StoreVerdictDetail`. Never raises: this is read on a write
        path, and a clause that cannot decide must produce a verdict.
    """
    if not launch_permits(target):
        if launch_narrowed_target() == LAUNCH_POSTURE_ALL_TARGETS:
            return StoreVerdictDetail(StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE, _LAUNCH_PIN_REASON)
        return StoreVerdictDetail(StoreVerdict.NARROWING, None)

    resolved = current_owner(owner)
    if resolved is NO_OWNER:
        return StoreVerdictDetail(StoreVerdict.PERMITTED, None)
    if not isinstance(resolved, str):
        # An owner that is neither a name nor the sentinel is a lookup that did
        # not happen, which is the tree reader's classification too — and the
        # opposite of the rung above it. The distinction has to be two checks:
        # collapsed into one type test, a decoded payload's list or number would
        # take the ceiling-only path and run an owned plan at the deployment
        # ceiling, which is the one outcome the owner exists to prevent.
        reason = _owner_is_not_a_name(resolved)
        logger.warning("Write posture cannot be looked up: %s; refusing", reason)
        return StoreVerdictDetail(StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE, reason)

    tree = _control_context_tree()
    if tree is None:
        if _tree_is_bound():
            # Set but unusable — relative, or a ``~user`` this container has no
            # passwd entry for. From the operator's side that is the same class
            # as an unprovisioned mount, and falling back to the host reader
            # here would read a record this container does not have.
            reason = _tree_bind_unusable()
            logger.warning("Write posture unreadable for %r: %s", resolved, reason)
            return StoreVerdictDetail(StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE, reason)
        narrowed = recorded_posture()
    else:
        read = _read_tree_record(tree, resolved)
        if read.unavailable is not None:
            # The sentence goes two ways, and both matter: to the deployment's
            # log, which reaches whoever fixes the bind, and back to the caller,
            # whose refusal message may be the only one the person who ran the
            # write ever sees.
            logger.warning("Write posture unreadable for %r: %s", resolved, read.unavailable)
            return StoreVerdictDetail(StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE, read.unavailable)
        narrowed = read.narrowed

    if _permits(narrowed, target):
        return StoreVerdictDetail(StoreVerdict.PERMITTED, None)
    return StoreVerdictDetail(StoreVerdict.NARROWING, None)


def store_permits(target: str | None) -> bool:
    """The recorded clause of :func:`effective_writes` as a bool — rule 3, on its own.

    Public because one caller needs this clause WITHOUT the ceiling
    :func:`effective_writes` derives: the connector's reference monitor
    (``control_system.base``) reads a deployment ceiling keyed on the connector
    TYPE, which is not a ceiling this module can produce from a target. It
    therefore ANDs its own ceiling with this function rather than restating the
    combining terms — the contract's rule 3 has exactly two implementations,
    this one and the stdlib restatement in the hooks, and a third would be a
    third thing to keep in step.

    Exactly :func:`store_verdict` compared against
    :attr:`StoreVerdict.PERMITTED`, so the two cannot answer differently. A
    caller that must say WHY a write was refused asks for the verdict instead;
    this spelling is for the callers that only have to decide whether to
    proceed.

    Answers ``True`` — permitted — for everything that is not an actual
    narrowing and not an unreadable one: no record, or a record whose posture
    does not name this target. It can only refuse; nothing here widens a
    ceiling.

    Args:
        target: The control target the write lands on, or ``None`` when the
            caller cannot name one — in which case the most restrictive
            narrowing decides.
    """
    return store_verdict(target) is StoreVerdict.PERMITTED


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
