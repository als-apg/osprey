"""The write-switch reader — one field of one record, one verdict.

An operator sets one control target's writes to off, asked or allowed from the
header chip: "writes on for the stand-in, leave the virtual accelerator alone".
That switch is one field of the control-context record
(:mod:`osprey_connectors.control_context`) and is enforced at *write* time
rather than delivered by respawning the agent, which is what lets a flip land
on a session already mid-conversation.

This module is the canonical reader of that field. Two others answer from the
same file and must agree with it: the controls MCP server (which imports this
one) and the stdlib-only PreToolUse hook in
``osprey/templates/claude_code/claude/hooks/osprey_target_state.py``, which
cannot import anything and therefore *restates* the contract below. Treat the
following five rules as the contract; a change here is a change there.

**1. Where the file is.** One directory per identity:
:func:`~osprey_connectors.identity.acting_identity` under
:data:`STATE_DIR_NAME` under the agent-data root, so one directory answers
"control state for whoever this process is acting as". The switch is one
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
rather than quietly writing somewhere a reader will never look, and which
:func:`write_verdict` refuses as ``record_unavailable``.

**2. What the shapes mean.** The record's ``switch`` field is a per-target map
of positions, and :func:`~osprey_connectors.control_context.switch_position` is
its one reader: a machine is armed only at ``"asked"`` or ``"allowed"``, and a
stored ``"on"`` from an older release reads as ``"asked"``. Everything else — no
record, a record that does not name the machine, any other value — is that
machine's writes off. The record's ``posture`` field is a mirror of the switch
(:func:`~osprey_connectors.control_context.posture_mirror`) kept for readers
that predate it, and no verdict reads it. :func:`parse_posture_value` is the
grammar of that mirror, applied by
:func:`~osprey_connectors.control_context.parse_posture` rather than restated
there.

**3. How a write is decided.** :func:`write_verdict` is the whole rule: one
total order of terms — read-only run, tier, target, owner, owner tier, record,
switch, launch pin — and the first that refuses names the reason. Its order is
shared with the hook's stdlib twin, so a write is refused for the same reason
on both sides. An allowed verdict carries the effective position: the stored
one, capped by the ceiling the environment's verdict already evaluates — the
section's tier on a host, the owner's rung under a bound tree. A position above
the ceiling refuses nothing and stays on disk. :func:`backstop_verdict` is the
floor for a write no target or owner can be named for: its first two terms and
nothing else, so it reports no position.

The switch is a property of one OWNER's record, not of a process tree: no
session key indexes it, and a process that belongs to no session — a dispatch
worker, a CLI run — is decided by the same field as one that does.
``OSPREY_POSTURE_SESSION`` remains the audit session id and nothing more.

Inside an executor sandbox one further term is read, and only there:
:data:`LAUNCH_POSTURE_ENV_VAR`, the switch state the run was LAUNCHED under
(:func:`launch_posture_stamp` writes it, :func:`launch_pin_term` reads it). Its
job is asymmetry: a machine switched off mid-run is refused by the record read,
while a machine switched ON never reaches a run that started with it off, which
would otherwise hand a running script write access the operator had not given
it when the run began.

**4. When a change is seen.** Every read re-stats the record and re-parses on
``(st_mtime_ns, st_size, st_ino)`` — the cache lives in
:mod:`~osprey_connectors.control_context`, and :func:`invalidate_cache` is the
hook a process that moved roots (or a test) drops it through. A missing record
is every switch off, not an error.

This module reads no config for the switch, holds no cache of its own, and
imports nothing from ``osprey``: it runs inside the connector-host child and
the executor sandbox, where the dependency budget is the lean connector chain.
The record reader is imported inside the functions that need it, because the
import direction between the two modules is one-way — the record module
imports THIS one at module scope for the root and the grammar.

**5. Whose switch decides.** A chip is one person's, so a verdict needs a name
as well as a target. That name is the OWNER, and it is resolved by one ladder
here — :func:`current_owner`, with :data:`NO_OWNER` for work that belongs to
nobody — so that the write monitor and the queue never disagree about whose
switch a plan runs under. The owner of a queued plan rides on the item as
:data:`RESERVED_OWNER_KWARG` and is bound for the wrapper's body by
:func:`bind_owner`.

A container that holds no chip of its own — a lane's queueserver, the dispatch
worker — reads that owner's record out of a read-only bind of the WHOLE tree,
:data:`CONTROL_CONTEXT_TREE_ENV_VAR`, rather than out of a directory of its own.
That read fails CLOSED: an unprovisioned mount, a group the container could not
join and a record written 0600 must not be answered as an operator who switched
nothing on, nor as one whose switch nobody could read being on. So the tree
reader separates the one absence that is an answer — no record for this owner,
every switch off — from every failure on a path that is THERE, which refuses as
``record_unavailable``. :data:`CONTROL_TREE_MARKER_NAME` is what makes the first
question answerable at all: a bind whose source does not exist is created empty
by the container runtime, and an empty directory says nothing about whether it
was ever built.

The invariant that binding rests on: **the owner variable is read on the
connector's async path, never after a thread or executor hop.** A context is
copied into a task when the task is created, so the binding reaches everything
a RunEngine schedules from the wrapper's context. An offload is where that
stops being true, and which helper made it decides whether it does:
``asyncio.to_thread`` copies the context and carries the binding into the
thread, while ``loop.run_in_executor``, a raw executor submit and a bare
``threading.Thread`` do not. No gate may depend on the carrying one having been
chosen. A verdict taken behind such a hop would read :data:`NO_OWNER`; every
verdict is therefore taken before the offload.
"""

from __future__ import annotations

import contextvars
import errno
import logging
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple, cast

from osprey_connectors.identity import acting_identity
from osprey_connectors.posture_env import is_readonly_run
from osprey_connectors.types import (
    CONTROL_TARGETS,
    TIER_WORDS,
    machine_kind,
    published_machine_kind,
    resolve_target,
    tier_allows,
    tier_writes,
)
from osprey_connectors.workspace import resolve_shared_data_root

logger = logging.getLogger("osprey_connectors.posture_store")

__all__ = [
    "NoOwner",
    "AGENT_DATA_ROOT_ENV_VAR",
    "CONTROL_CONTEXT_DIR_ENV_VAR",
    "CONTROL_CONTEXT_TREE_ENV_VAR",
    "CONTROL_OWNER_ENV_VAR",
    "CONTROL_TREE_MARKER_NAME",
    "DISPATCH_RUN_ENV_VAR",
    "LAUNCH_POSTURE_ALL_TARGETS",
    "LAUNCH_POSTURE_ENV_VAR",
    "LOGIN_TIERS_KEY",
    "NO_OWNER",
    "POSTURE_SANDBOX",
    "POSTURE_WRITES",
    "RESERVED_OWNER_KWARG",
    "STATE_DIR_NAME",
    "VALID_POSTURES",
    "WriteVerdict",
    "agent_data_root",
    "backstop_verdict",
    "bind_owner",
    "bound_owner",
    "bound_state_dir",
    "current_owner",
    "invalidate_cache",
    "launch_pin_term",
    "launch_posture_stamp",
    "owner_tier_term",
    "parse_posture_value",
    "stamped_agent_data_root",
    "state_dir",
    "write_verdict",
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
#: another is a switch that silently never applies.
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
#: another user's switch.
CONTROL_CONTEXT_DIR_ENV_VAR = "OSPREY_CONTROL_CONTEXT_DIR"

#: The launch-pin value: the run started with the machine's writes off. The
#: only posture value that ever refuses anything.
POSTURE_SANDBOX = "sandbox"

#: The run started with the machine's writes on. Recorded so a stamp states
#: what the run launched under; it never grants anything.
POSTURE_WRITES = "writes"

#: The two values a posture entry may spell. Everything else is dropped.
VALID_POSTURES = frozenset({POSTURE_SANDBOX, POSTURE_WRITES})

#: The launch-time pin, stamped into an executor sandbox's environment by
#: ``osprey.mcp_server.python_executor.executor._apply_target_stamp`` and read
#: back HERE, inside that sandbox, by :func:`launch_pin_term`.
#:
#: The wire format is one ``"<target>=<posture>"`` pair and nothing else:
#:
#: * ``<target>`` is the control target the run was stamped against — a member
#:   of :data:`~osprey_connectors.types.CONTROL_TARGETS` — or
#:   :data:`LAUNCH_POSTURE_ALL_TARGETS` when the executor could not identify
#:   one, in which case the pin covers every target;
#: * ``<posture>`` is :data:`POSTURE_SANDBOX` or :data:`POSTURE_WRITES`.
#:
#: ``sandbox`` is the only value that does anything: ``writes`` is recorded so
#: the in-flight marker states what the run launched under, and is inert here:
#: a stamp can refuse and never grant. A missing or blank stamp is inert too,
#: which is what makes every process that is not an executor sandbox
#: unaffected by this term; an unparseable one refuses (:func:`launch_pin_term`).
#:
#: Spelled here rather than imported because this is the reading side and the
#: reading side runs in the lean connector chain;
#: ``registry/mcp.py`` and the executor import the name from this module, and
#: it is stripped from every rendered ``.mcp.json`` env block
#: (``NON_PINNABLE_AUDIT_MARKERS``) — a spec that could pin it could hand a
#: run launched with writes off the writes it was denied.
LAUNCH_POSTURE_ENV_VAR = "OSPREY_LAUNCH_POSTURE"

#: The launch stamp's target when the executor could not name one. Covers every
#: target: a run that could not say which machine it is about must not be the
#: one run a switched-off machine fails to reach.
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
#: user's switch — or take one away.
CONTROL_OWNER_ENV_VAR = "OSPREY_CONTROL_OWNER"

#: The mark the dispatch worker's ``sdk_runner`` sets on every job it runs: a
#: scheduled or triggered job, started by no login at a terminal. Only the value
#: ``"1"`` marks a dispatch run; anything else, ``"0"`` included, is no dispatch.
#: A dispatch run that names no owner resolves :data:`NO_OWNER` and is refused
#: ``owner_unscheduled`` (:func:`write_verdict`): the process account it runs as
#: is no login whose switch could cover it.
DISPATCH_RUN_ENV_VAR = "OSPREY_DISPATCH_RUN"

#: The read-only bind of the whole per-user control-state tree, named in the
#: environment of a container that holds no chip of its own: a lane's
#: queueserver, the dispatch worker. Only its PRESENCE is read here; the tree is
#: read by the verdict.
#:
#: Presence is what makes such a container owned-or-nothing. The account it runs
#: as names nobody whose chip anyone ever set, so falling through to
#: :func:`~osprey_connectors.identity.acting_identity` there would read a record
#: directory that does not exist, so a plan would be decided by nobody's
#: switch rather than its owner's.
CONTROL_CONTEXT_TREE_ENV_VAR = "OSPREY_CONTROL_CONTEXT_TREE"

#: The marker file at the root of the control-context tree, written by
#: ``osprey build``. Its PRESENCE is the tree reader's first question, because a
#: read-only bind whose host source does not exist is created empty by the
#: container runtime — after which an unprovisioned tree is indistinguishable
#: from a tree in which nobody has switched anything on, and a provisioning
#: fault reads as an operator's answer on exactly the deployment that never
#: provisioned it.
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


class NoOwner:
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
    would cost the one line that says the plan ran for nobody — and printing
    is all it is for. Callers ask ``owner is NO_OWNER``, never truthiness (a real
    name can be falsey only by being empty, which this module never produces)
    and never equality with the text.
    """

    __slots__ = ()

    def __str__(self) -> str:
        return "<no owner>"

    def __repr__(self) -> str:
        return "<no owner>"


#: The answer every owner-less path gives. See :class:`NoOwner`.
NO_OWNER: Final = NoOwner()

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
_owner_var: contextvars.ContextVar[str | NoOwner] = contextvars.ContextVar(
    "osprey_control_owner", default=NO_OWNER
)


def current_owner(owner: str | NoOwner | None = None) -> str | NoOwner:
    """The owner whose switch decides a write, by a five-rung ladder.

    Most specific first:

    1. *owner* when the caller named one, stripped — the same normalisation the
       two rungs below apply, so a padded name and a bare one address one
       person's directory rather than two. Passing :data:`NO_OWNER` explicitly
       is naming one — it says "this work belongs to nobody", which a verdict
       refuses — which is why the "unnamed" spelling is ``None`` and not the
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
    4. :data:`NO_OWNER` for a dispatch run (:data:`DISPATCH_RUN_ENV_VAR` is
       ``"1"``) that none of the rungs above named an owner for. The account a
       scheduled job runs as is not a person, so it never stands in for one.
    5. Otherwise: :data:`NO_OWNER` when a tree is bound (:func:`_tree_is_bound`,
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

    if _is_dispatch_run():
        return NO_OWNER

    if _tree_is_bound():
        return NO_OWNER

    return acting_identity()


def _is_dispatch_run() -> bool:
    """Whether this process is a dispatch job: :data:`DISPATCH_RUN_ENV_VAR` is ``"1"``."""
    return os.environ.get(DISPATCH_RUN_ENV_VAR) == "1"


def bound_owner() -> str | NoOwner:
    """The owner :func:`bind_owner` bound for the running plan, and nothing else.

    Rung 2 of :func:`current_owner` alone: no caller argument, no environment
    stamp, no process account. This is the question an attribution asks — whose
    queued plan is this — and the ladder's lower rungs answer a different one:
    whose switch governs a write here. A record attributed through them would
    name the account the worker runs as rather than the person who queued the
    run, so outside a :func:`bind_owner` block this answers :data:`NO_OWNER`.

    Never raises.
    """
    return _owner_var.get()


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
    binds :data:`NO_OWNER`: the plan reached the queue without a name, and no
    login's switch can cover its writes.

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
    owner: str | NoOwner = NO_OWNER
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
    creates, which is the silently-ignored-switch failure rule 1 exists to prevent.
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
    no record, which is a different thing from having one with every switch off.
    """
    stamped = stamped_agent_data_root()
    if stamped is not None:
        return stamped
    try:
        return resolve_shared_data_root()
    except Exception:  # an unresolvable root is "no record", not a crash
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
    a switch that applies rather than one that silently does not.

    A ``None`` here is what a route reports as ``store_unavailable``: there is
    nowhere to record a switch that any reader would find.
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
    its own), while :func:`write_verdict` picks the reader that governs the
    write. A variable that made one of them say "tree" and the other say "host"
    would send a container with no host record to the host reader, which
    answers for this host's login and not for the owner of the work — the
    split the tree reader exists to close.

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
    :func:`_read_tree_context` is for.

    Normalised by :func:`_absolute_bind`, the same rule the container's own
    state directory goes through, so the two binds cannot disagree about what a
    ``~`` or a relative path means.
    """
    return _absolute_bind(CONTROL_CONTEXT_TREE_ENV_VAR)


# -- parsing ----------------------------------------------------------------


def parse_posture_value(value: Any) -> dict[str, str]:
    """The record's ``posture`` mirror as ``{target: "sandbox"}`` — the switched-off targets.

    The grammar of the mirror field (rule 2), in one place:
    :func:`~osprey_connectors.control_context.parse_posture` reaches this
    filter rather than restating it. No verdict reads the result — the switch
    decides every write — so this is the shape a reader that predates the
    switch sees, and nothing more.

    Never raises. Anything that is not a ``"sandbox"`` entry — a bare
    ``"writes"``, an unknown string, a non-string target key, a hand-edited or
    future-version shape — is dropped, so a field nobody can repair from the
    browser never fails a read.
    """
    if isinstance(value, str):
        if value == POSTURE_SANDBOX:
            return dict.fromkeys(CONTROL_TARGETS, POSTURE_SANDBOX)
        # Bare "writes" (and every unknown string) maps nothing.
        return {}
    if isinstance(value, dict):
        return {
            target: posture
            for target, posture in value.items()
            if isinstance(target, str) and posture == POSTURE_SANDBOX
        }
    return {}


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
    except Exception:  # dropping a cache must not raise into a caller
        logger.debug("Could not drop the control-context cache", exc_info=True)


# -- the read-only tree -----------------------------------------------------


#: Characters that cannot appear in the one path segment an owner names. The
#: same rule :func:`~osprey_connectors.identity.acting_identity` applies to an
#: identity, restated for the name a queue item carries rather than shared with
#: it, because that function ANSWERS a name while this one screens one somebody
#: else supplied.
_TREE_UNSAFE: Final = frozenset({"/", "\\", "\x00"})


class _TreeContext(NamedTuple):
    """One read of one owner's whole record out of the read-only tree.

    Attributes:
        record: The parsed record — its ``switch`` already filtered by
            :mod:`~osprey_connectors.control_context`'s parser to the targets
            at ``"asked"`` or ``"allowed"`` — or ``None`` when the tree holds no record for
            this owner or the record could not be read.
        unavailable: ``None`` when the read reached an answer — a record, or
            a provisioned tree holding none for this owner — and otherwise the
            operator-facing sentence saying what could not be read and what to
            do about it. The caller turns a non-``None`` into its own refusal;
            the remedy is this reader's, because only the read knows which of
            the ways of not knowing happened.
    """

    record: Any
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
    fails to parse or, worse, parses into a different switch than the one on
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


def _read_tree_context(tree: Path, owner: str) -> _TreeContext:
    """*owner*'s whole record out of the read-only *tree*, with the reason when there is none.

    The reader a container uses when it holds the whole tree and no chip of its
    own: a lane's queueserver, the dispatch worker. It fails CLOSED, and that is
    the whole difference from the host reader.
    :func:`~osprey_connectors.control_context.read_record` folds an unreadable
    file into ``None`` — tolerable for a display on the host, and wrong here,
    where an unprovisioned bind, a group the container could not join or a
    record written 0600 must not be answered as the owner's switch. So this
    reader does not go through it.

    Exactly one failure is not a refusal: an ENOENT on the record. A tree that
    is provisioned and readable and holds no record for this owner is an owner
    whose every switch is off. Every other failure on a PRESENT path — EACCES,
    EISDIR, a corrupt payload, a payload from another schema — is a switch that
    may exist and cannot be read, and is reported rather than guessed at.

    **Nothing here follows a symbolic link.** The tree root is group-writable
    (2770) and its group is shared with accounts other than the one reading, so
    any account in it can plant a link under another identity's name; followed,
    that link decides whether a real machine is written to, from a file its
    supposed owner never wrote. The marker and the ``<owner>/`` directory are
    therefore checked with :func:`os.lstat` and the record is opened
    ``O_NOFOLLOW`` — the open is what closes the window between a check and a
    read — and a link found at any of the three answers unavailable NAMING the
    path. Never ENOENT: an absent record means every switch is off, and a
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
            reaches here: work that belongs to nobody is refused before
            any record is read.

    Returns:
        A :class:`_TreeContext`: the parsed record (``None`` when the tree holds
        none for this owner, or when it could not be read) and the sentence when
        it could not. Never raises: this is read on a write path.
    """
    if not tree.is_dir():
        return _TreeContext(None, _tree_unprovisioned(tree, "does not exist"))
    if not os.access(tree, os.R_OK | os.X_OK):
        return _TreeContext(None, _tree_unreadable(tree))

    marker = tree / CONTROL_TREE_MARKER_NAME
    try:
        marker_mode = os.lstat(marker).st_mode
    except FileNotFoundError:
        return _TreeContext(
            None, _tree_unprovisioned(tree, f"carries no {CONTROL_TREE_MARKER_NAME}")
        )
    except OSError:
        return _TreeContext(None, _tree_unreadable(tree))
    if stat.S_ISLNK(marker_mode):
        # A link here would let any account in the tree's group declare an
        # unbuilt tree provisioned, which is the fail-open this marker exists
        # to close.
        return _TreeContext(None, _tree_planted_link(marker, "tree marker"))
    if not stat.S_ISREG(marker_mode):
        return _TreeContext(
            None,
            _tree_unprovisioned(tree, f"carries a {CONTROL_TREE_MARKER_NAME} that is not a file"),
        )

    # An owner that is not one path segment addresses no record in this tree,
    # and must not be answered as an owner whose switches are off: the name
    # arrived on a queue item, and a name nobody can look up is a lookup that
    # did not happen rather than a lookup that found nothing.
    if not isinstance(owner, str) or owner.strip() in {"", ".", ".."} or _TREE_UNSAFE & set(owner):
        return _TreeContext(
            None,
            f"the owner {owner!r} is not a name the control-context tree can hold, "
            "so no narrowing can be read for it — the owner a plan carries must be "
            "one directory name",
        )

    # Imported here, not at module scope: ``control_context`` imports THIS
    # module for the root and the grammar, and the two must not import each
    # other at load time.
    from osprey_connectors import control_context

    owner_dir = tree / owner
    try:
        owner_mode = os.lstat(owner_dir).st_mode
    except FileNotFoundError:
        # The one absence that is an answer: every switch is off for this
        # owner, and a missing ``<owner>/`` directory says exactly that.
        return _TreeContext(None, None)
    except OSError:
        logger.debug("Could not stat the control-context directory %s", owner_dir, exc_info=True)
        return _TreeContext(None, _tree_unreadable(tree))
    if stat.S_ISLNK(owner_mode):
        # ``O_NOFOLLOW`` on the record below guards its last component only, so
        # the directory above it is checked here: a link at this level redirects
        # the read just as effectively.
        return _TreeContext(None, _tree_planted_link(owner_dir, "directory"))

    record_file = owner_dir / control_context.RECORD_FILENAME
    try:
        record = control_context.parse_record(_read_no_follow(record_file))
    except FileNotFoundError:
        # No record for this owner: the same answer as no directory.
        return _TreeContext(None, None)
    except Exception as exc:  # every failure here REFUSES
        # Blanket on purpose, and the opposite of the one this reader avoids:
        # an unexpected failure on a present record leaves a switch that may
        # exist unread, so it answers unavailable rather than guessing.
        logger.debug("Could not read the control-context record at %s", record_file, exc_info=True)
        if getattr(exc, "errno", None) == errno.ELOOP:
            return _TreeContext(None, _tree_planted_link(record_file, "record"))
        if getattr(exc, "errno", None) == errno.EFBIG:
            # Asked by name for the same reason the planted link is: the size
            # bound refuses a file whose mode and group are beyond reproach, so
            # the generic remedy points at two things an operator then finds
            # correct.
            return _TreeContext(None, _tree_record_oversized(record_file))
        detail = getattr(exc, "strerror", None) or type(exc).__name__
        return _TreeContext(
            None, _tree_record_unreadable(record_file, f"could not be read ({detail})")
        )
    if record is None:
        return _TreeContext(
            None,
            _tree_record_unreadable(record_file, "is not a record this version can read"),
        )
    return _TreeContext(record, None)


# -- the launch-time pin ----------------------------------------------------


def launch_posture_stamp(target: str | None, launch_posture: str) -> str:
    """Compose the :data:`LAUNCH_POSTURE_ENV_VAR` value for one launch.

    The one place the wire format is written, so the executor that stamps it
    and :func:`launch_pin_term` which reads it cannot disagree about the
    separator or about how "no target" is spelled.

    Args:
        target: The control target the run is stamped against, or ``None`` when
            the executor could not identify one — spelled
            :data:`LAUNCH_POSTURE_ALL_TARGETS`, which covers every target.
        launch_posture: :data:`POSTURE_SANDBOX` or :data:`POSTURE_WRITES` — the
            recorded answer for that target at the moment of launch.
    """
    return f"{target or LAUNCH_POSTURE_ALL_TARGETS}={launch_posture}"


def launch_pin_term(target: str | None) -> Literal["none", "pinned", "unreadable"]:
    """The launch pin as a term of the write verdict — fail-closed on a bad stamp.

    Reads :data:`LAUNCH_POSTURE_ENV_VAR` on every call and answers, for *target*:

    * ``"none"`` — no stamp (absent or blank), or a ``<t>=writes`` stamp whose
      ``<t>`` is a member of :data:`~osprey_connectors.types.CONTROL_TARGETS` or
      :data:`LAUNCH_POSTURE_ALL_TARGETS`, or a ``<t>=sandbox`` stamp naming a
      different target than *target*;
    * ``"pinned"`` — a ``<t>=sandbox`` stamp with ``<t>`` equal to *target*, or
      ``<t>`` :data:`LAUNCH_POSTURE_ALL_TARGETS`. A caller that names no target
      (``None``) is pinned by any sandbox stamp: the most restrictive answer
      wins;
    * ``"unreadable"`` — anything else, for every target: no separator, an empty
      or unknown target, a posture that is neither :data:`POSTURE_SANDBOX` nor
      :data:`POSTURE_WRITES`.

    ``pinned`` and ``unreadable`` both refuse as ``launch_pinned``; a stamp is
    only ever written by the executor, so one this module cannot read is a run
    whose launch posture is unknown, and that must not widen to writable.
    Surrounding whitespace on the stamp and on each half is ignored.
    """
    text = (os.environ.get(LAUNCH_POSTURE_ENV_VAR) or "").strip()
    if not text:
        return "none"
    stamped, separator, launch_posture = text.partition("=")
    stamped = stamped.strip()
    launch_posture = launch_posture.strip()
    if (
        not separator
        or launch_posture not in VALID_POSTURES
        or (stamped != LAUNCH_POSTURE_ALL_TARGETS and stamped not in CONTROL_TARGETS)
    ):
        return "unreadable"
    if launch_posture != POSTURE_SANDBOX:
        return "none"
    if stamped in (LAUNCH_POSTURE_ALL_TARGETS, target) or not target:
        return "pinned"
    return "none"


# -- the owner rung ---------------------------------------------------------

#: The build-rendered login→tier map: ``{<login>: "never" | "asked" | "allowed"}``,
#: each value the resolved render's ``tier_writes`` for that login. A process
#: that reads other logins' records (a shared container) cannot use its own
#: section's tier for the record's owner, so it reads the owner's entry here.
LOGIN_TIERS_KEY = "control_system.login_tiers"

#: The leaf under the ``control_system`` section that :data:`LOGIN_TIERS_KEY` names.
_LOGIN_TIERS_LEAF: Final = "login_tiers"

#: The three values a map entry may carry — the same three :func:`tier_writes` answers.
_LOGIN_TIERS: Final = TIER_WORDS


def owner_tier_term(
    section: Any, owner: str | NoOwner | None
) -> Literal["skip", "never", "asked", "allowed", "unknown"]:
    """The owner's tier out of :data:`LOGIN_TIERS_KEY` — the verdict's owner rung.

    *section* is the ``control_system`` section; *owner* the login whose record
    governs the write. The rung applies when the section carries the map (any
    value) **or** a control-context tree is bound (:func:`_tree_is_bound`, the
    one marker of a process that reads other logins' records). Answers:

    * ``"skip"`` — the rung does not apply: no map and no tree;
    * ``"never"`` / ``"asked"`` / ``"allowed"`` — the owner's entry;
    * ``"unknown"`` — the rung applies and the owner's tier cannot be read: a
      tree with no map, a map that is not a mapping, a map with any key that is
      not a non-empty string or any value other than the three tier words
      (unreadable as a whole — it is one render's output), an owner that is not
      a non-empty string, or an owner the map does not name. The verdict refuses
      it as ``owner_unknown``; ``never`` refuses as ``tier_never``.

    Logins match exactly: no case folding, no trimming. Never raises.
    """
    try:
        carries = isinstance(section, dict) and _LOGIN_TIERS_LEAF in section
        if not carries:
            return "unknown" if _tree_is_bound() else "skip"
        login_tiers = section[_LOGIN_TIERS_LEAF]
        if not isinstance(login_tiers, dict) or not all(
            isinstance(login, str) and login and isinstance(tier, str) and tier in _LOGIN_TIERS
            for login, tier in login_tiers.items()
        ):
            return "unknown"
        if not isinstance(owner, str) or not owner:
            return "unknown"
        tier = login_tiers.get(owner)
    except Exception:  # a rung that cannot read the map refuses; it never raises
        return "unknown"
    return tier if tier in _LOGIN_TIERS else "unknown"


# -- the write verdict ------------------------------------------------------


@dataclass(frozen=True)
class WriteVerdict:
    """Whether one write may go ahead, and the one reason when it may not.

    Attributes:
        allowed: ``True`` only when every term of :func:`write_verdict` passed.
        reason: ``None`` when allowed; otherwise a member of
            :data:`~osprey_connectors.refusals.REASONS`, the first term that
            refused. :func:`~osprey_connectors.refusals.refusal_phrase` turns it,
            with *kind*, into the words an operator reads.
        target: The control target the caller asked about, verbatim when it was
            a string, else ``None``.
        kind: The kind of machine *target* names here — ``real``, ``rehearsal``
            or ``simulator`` — or ``None`` when the target does not resolve.
        position: The effective switch position when *allowed* — ``asked`` or
            ``allowed``, the stored position capped by the environment's
            ceiling — and ``None`` on every refusal and from
            :func:`backstop_verdict`.
    """

    allowed: bool
    reason: str | None
    target: str | None
    kind: Literal["real", "rehearsal", "simulator"] | None
    position: Literal["asked", "allowed"] | None = None


def write_verdict(
    section: Any,
    target: str | None,
    *,
    owner: str | NoOwner | None = None,
    ignore_launch_pin: bool = False,
) -> WriteVerdict:
    """The verdict on one write to *target*: one total order, first refusal wins.

    The terms, in order — the order is shared with the hook's stdlib twin, so a
    write is refused for the same reason on both sides:

    1. ``readonly_run`` — the run is read-only (``OSPREY_EXECUTION_MODE``).
    2. ``tier_never`` — the section's tier does not allow writes
       (:func:`~osprey_connectors.types.tier_allows`).
    3. ``target_unknown`` — *target* does not resolve in this deployment.
    4. ``owner_unknown`` — the owner ladder (:func:`current_owner`) yields
       :data:`NO_OWNER` or anything that is not a non-empty name. A write with
       no owner is refused; nothing is governed by the ceiling alone here.
       ``owner_unscheduled`` in its place for a dispatch run
       (:data:`DISPATCH_RUN_ENV_VAR`), whether the caller passed
       :data:`NO_OWNER` or the ladder resolved it: a scheduled or triggered
       job that runs for no login has no switch to follow.
       ``owner_foreign`` — with no tree bound, an owner that is not
       :func:`~osprey_connectors.identity.acting_identity`: a host holds only
       its own login's switch and never lends it to another owner.
    5. The owner rung (:func:`owner_tier_term`): a login the map does not name,
       or a map that cannot be read where the rung applies, is
       ``owner_unknown``; a login mapped ``never`` is ``tier_never``.
    6. ``record_unavailable`` — the record governing the write could not be
       read. Under a bound tree that is the owner's record, through the
       no-follow, fail-closed tree reader; a bind that is set but unusable
       refuses rather than falls back to the host. Otherwise it is this
       host's own record, through the status reader, which fails closed too: a
       corrupt file, a foreign schema or an unresolvable root refuses. A record
       that is simply absent is not unavailable — it is every switch off.
    7. ``switch_off`` — the record's stored position for *target*
       (:func:`~osprey_connectors.control_context.switch_position`) is ``off``.
       The record's ``posture`` field is not read: it mirrors the switch, and
       an old ``sandbox`` value left in a file does not refuse.
    8. ``launch_pinned`` — the launch pin (:func:`launch_pin_term`) is
       ``pinned`` or ``unreadable`` for *target*; skipped when
       *ignore_launch_pin*, for a caller that reports the pin on its own.

    An allowed verdict's ``position`` is the lower of the stored position and
    the ceiling: :func:`~osprey_connectors.types.tier_writes` of *section* on a
    host (term 2), the owner's rung under a bound tree (term 5). The ceiling
    adds no refusal.

    Tier and owner rung come before the record, so a demoted login's position
    stays on disk, inert, and arms again if the tier is restored. This function
    reads and never writes. It never answers ``switching`` or ``stopped_since``:
    those belong to the callers that hold the record's lock.

    Args:
        section: The ``control_system`` section of the resolved config.
        target: The control target the write lands on.
        owner: Whose record governs the write; ``None`` asks
            :func:`current_owner`.
        ignore_launch_pin: Skip the last term.

    Returns:
        A :class:`WriteVerdict`. Never raises: an exception anywhere in the
        terms refuses rather than allows.
    """
    named = target if isinstance(target, str) else None
    try:
        # The name decides whether the target resolves at all; the published
        # report decides what it is (C-KIND), so a stand-in the controls server
        # has not vouched for is the real machine in the refusal's words.
        kind = None if machine_kind(section, target) is None else published_machine_kind(target)
    except Exception:  # the kind is a word in a refusal; it must not decide one
        logger.debug("Could not name the machine kind of %r", target, exc_info=True)
        kind = None
    position: Literal["asked", "allowed"] | None = None
    try:
        reason, position = _write_refusal(section, target, owner, ignore_launch_pin)
    except Exception:  # a verdict that cannot be reached refuses; it never raises
        logger.warning("Write verdict for %r could not be reached; refusing", target, exc_info=True)
        reason = "record_unavailable"
    return WriteVerdict(
        allowed=reason is None, reason=reason, target=named, kind=kind, position=position
    )


def backstop_verdict(section: Any) -> WriteVerdict:
    """The verdict for a write no target or owner can be named for.

    The floor under every write path that cannot be traced to one machine: it
    refuses ``readonly_run`` (``OSPREY_EXECUTION_MODE``), then ``tier_never``
    (:func:`~osprey_connectors.types.tier_allows`), in the same order as the
    first two terms of :func:`write_verdict`, and otherwise allows. It is blind
    by design — it resolves no target, asks no owner and opens no record, so a
    switch position never changes its answer and it reports none.

    Args:
        section: The ``control_system`` section of the resolved config.

    Returns:
        A :class:`WriteVerdict` whose ``target``, ``kind`` and ``position`` are
        always ``None``. Never raises: a tier that cannot be read refuses as
        ``tier_never``.
    """
    if is_readonly_run():
        reason: str | None = "readonly_run"
    else:
        try:
            reason = None if tier_allows(section) else "tier_never"
        except Exception:  # tier_allows never raises by contract; refuse if it does
            logger.warning("Backstop tier could not be read; refusing", exc_info=True)
            reason = "tier_never"
    return WriteVerdict(allowed=reason is None, reason=reason, target=None, kind=None)


def _write_refusal(
    section: Any, target: Any, owner: str | NoOwner | None, ignore_launch_pin: bool
) -> tuple[str | None, Literal["asked", "allowed"] | None]:
    """The first term of :func:`write_verdict` that refuses, else the position.

    Answers ``(reason, None)`` or ``(None, position)``. May raise.
    """
    # Imported here, not at module scope: ``control_context`` imports this module.
    from osprey_connectors import control_context

    if is_readonly_run():
        return "readonly_run", None
    if not tier_allows(section):
        return "tier_never", None
    try:
        resolve_target(section, target)
    except Exception:  # ValueError by contract; anything else is no resolution either
        return "target_unknown", None

    resolved = current_owner(owner)
    if resolved is NO_OWNER and _is_dispatch_run():
        return "owner_unscheduled", None
    if resolved is NO_OWNER or not isinstance(resolved, str) or not resolved.strip():
        return "owner_unknown", None
    # A host holds one login's switch: an owner other than that login is refused
    # before its tier or any record is read (a tree bind reads each owner's own).
    if not _tree_is_bound() and resolved.strip() != acting_identity():
        return "owner_foreign", None

    rung = owner_tier_term(section, resolved)
    if rung == "unknown":
        return "owner_unknown", None
    if rung == "never":
        return "tier_never", None

    if _tree_is_bound():
        tree = _control_context_tree()
        if tree is None:
            logger.warning("Write record unreadable for %r: %s", resolved, _tree_bind_unusable())
            return "record_unavailable", None
        read = _read_tree_context(tree, resolved)
        if read.unavailable is not None:
            logger.warning("Write record unreadable for %r: %s", resolved, read.unavailable)
            return "record_unavailable", None
        record = read.record
    else:
        record, status = control_context.read_record_status()
        if status == "unreadable":
            logger.warning("Write record unreadable on this host; refusing")
            return "record_unavailable", None
        if status != "ok":
            record = None

    stored = control_context.switch_position(record, target)
    if stored == "off":
        return "switch_off", None

    if not ignore_launch_pin and launch_pin_term(target) != "none":
        return "launch_pinned", None
    ceiling = rung if _tree_is_bound() else tier_writes(section)
    return None, cast("Literal['asked', 'allowed']", min(stored, ceiling, key=_LOGIN_TIERS.index))
