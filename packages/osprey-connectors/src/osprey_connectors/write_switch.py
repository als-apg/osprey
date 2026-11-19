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
:data:`CONTROL_STATE_DIR_NAME` under the agent-data root, so one directory answers
"control state for whoever this process is acting as". The switch is one
operator's, not the deployment's, and a single shared directory would make one
user's chip decide every other user's writes. The directory resolves by ONE
ladder, used by every writer and every reader:

* :data:`CONTROL_CONTEXT_DIR_ENV_VAR` when it names a non-blank path — the
  container's own state directory, bound in by compose. It is already the
  per-identity directory, so nothing is appended to it: a container sees its
  own dir at a path that has nothing to do with the host's tree, and deriving
  the identity hop inside it would name a directory nobody binds;
* otherwise the ``OSPREY_AGENT_DATA_ROOT`` stamp when it names a non-blank path —
  the stamp a session child carries — else
  the process config's shared agent-data root
  (:attr:`~osprey_connectors.config.LoadedConfig.agent_data_root`), the config
  derivation the state file has always used; and then the two fixed hops,
  :data:`CONTROL_STATE_DIR_NAME` and the acting identity.

There is no third path and no fallback. A root that resolves to nothing makes
:func:`state_dir` ``None``, which a route surfaces as ``store_unavailable``
rather than quietly writing somewhere a reader will never look, and which
:func:`write_verdict` refuses as ``record_unavailable``. The ladder is answered
below this module and the record alike, by the package's private
``_control_state``; :func:`state_dir`, the binds' variables and their kin are
re-exported here, and this is where they are documented.

**2. What the shapes mean.** The record's ``switch`` field is a per-target map
of positions, and :func:`~osprey_connectors.control_context.switch_position` is
its one reader: a machine is armed only at ``"asked"`` or ``"allowed"``, and a
stored ``"on"`` from an older release reads as ``"asked"``. Everything else — no
record, a record that does not name the machine, any other value — is that
machine's writes off. The record's ``posture`` field is a mirror of the switch
kept for readers that predate it, and nothing in this package reads it back.
:func:`~osprey_connectors.control_context._parse_posture_value` is the grammar
of that mirror, the reference the stdlib hook twin's reader of it is held to.

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

Both are blind to a session's hold, because they also choose the gateway a
connector child connects on. The hold is decided where a write executes:
:func:`session_hold_refusal` asks it alone, and :func:`execution_verdict` is
:func:`write_verdict` with the hold placed after its first two terms.

The switch is a property of one OWNER's record, not of a process tree: no
session key indexes it, and a process that belongs to no session — a dispatch
worker, a CLI run — is decided by the same field as one that does.
``OSPREY_POSTURE_SESSION`` remains the audit session id and nothing more.

Inside an executor sandbox one further term is read, and only there:
:data:`LAUNCH_PIN_ENV_VAR`, the switch state the run was LAUNCHED under
(:func:`launch_pin_stamp` writes it, the verdict's pin term reads it). Its
job is asymmetry: a machine switched off mid-run is refused by the record read,
while a machine switched ON never reaches a run that started with it off, which
would otherwise hand a running script write access the operator had not given
it when the run began.

**4. When a change is seen.** Every read re-stats the record and re-parses on
``(st_mtime_ns, st_size, st_ino)`` — the cache lives in
:mod:`~osprey_connectors.control_context`, and
:func:`osprey_connectors.testing.invalidate_cache` is the hook a test drops it
through. A missing record
is every switch off, not an error.

This module reads no config for the switch, holds no cache of its own, and
imports nothing from ``osprey``: it runs inside the connector-host child and
the executor sandbox, where the dependency budget is the lean connector chain.
The import direction is one-way: this module reads the record module, and the
record module never imports this one.

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
import logging
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, ClassVar, Final, Literal, NamedTuple, TypeAlias, cast

from pydantic import Field
from typing_extensions import override

from osprey_connectors import _control_state, control_context, session_hold
from osprey_connectors._control_state import (
    CONTROL_CONTEXT_DIR_ENV_VAR,
    CONTROL_CONTEXT_TREE_ENV_VAR,
    CONTROL_STATE_DIR_NAME,
    CONTROL_TREE_MARKER_NAME,
    DISPATCH_RUN_ENV_VAR,
    LAUNCH_PIN_ENV_VAR,
    LAUNCH_PIN_SANDBOX,
    LAUNCH_PIN_WRITES,
    LaunchPin,
    bound_state_dir,
    launch_pin_stamp,
    stamped_agent_data_root,
    state_dir,
)
from osprey_connectors.config_sections import ControlSystemSection, TierWord, Unreadable
from osprey_connectors.connector_types import ControlTarget, MachineKind, is_control_target
from osprey_connectors.control_context import ArmedPosition, ControlContext
from osprey_connectors.execution_mode import is_readonly_run
from osprey_connectors.identity import acting_identity
from osprey_connectors.model import FrozenModel, FrozenTuple, Instant, evolve
from osprey_connectors.refusals import VerdictReason, VerdictReasonField
from osprey_connectors.targets import machine_kind, resolve_target
from osprey_connectors.write_tier import TIER_WORDS, tier_allows, tier_writes

logger = logging.getLogger("osprey_connectors.write_switch")

__all__ = [
    "Allowed",
    "BudgetSpent",
    "ChannelNotListed",
    "FixedCause",
    "LaunchPin",
    "NoOwner",
    "OverlapHeld",
    "OwnerTierTerm",
    "RefusalCause",
    "Refused",
    "StoppedBy",
    "ValueOutOfRange",
    "WriteCeiling",
    "WrongMachine",
    "CONTROL_CONTEXT_DIR_ENV_VAR",
    "CONTROL_CONTEXT_TREE_ENV_VAR",
    "CONTROL_OWNER_ENV_VAR",
    "CONTROL_STATE_DIR_NAME",
    "CONTROL_TREE_MARKER_NAME",
    "DISPATCH_RUN_ENV_VAR",
    "LAUNCH_PIN_ENV_VAR",
    "LAUNCH_PIN_SANDBOX",
    "LAUNCH_PIN_WRITES",
    "NO_OWNER",
    "RESERVED_OWNER_KWARG",
    "WriteVerdict",
    "backstop_verdict",
    "bind_owner",
    "bound_owner",
    "bound_state_dir",
    "current_owner",
    "execution_verdict",
    "launch_pin_stamp",
    "owner_tier_term",
    "session_hold_refusal",
    "stamped_agent_data_root",
    "state_dir",
    "write_ceiling",
    "write_verdict",
]

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

    @override
    def __str__(self) -> str:
        return "<no owner>"

    @override
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
    5. Otherwise: :data:`NO_OWNER` when a tree is bound (:func:`~osprey_connectors._control_state.tree_is_bound`,
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

    if _control_state.is_dispatch_run():
        return NO_OWNER

    if _control_state.tree_is_bound():
        return NO_OWNER

    return acting_identity()


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
def bind_owner(kwargs: Mapping[str, object]) -> Iterator[dict[str, object]]:
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


# -- the owner rung ---------------------------------------------------------

#: The leaf under the ``control_system`` section holding the build-rendered
#: login→tier map, ``control_system.login_tiers``:
#: ``{<login>: "never" | "asked" | "allowed"}``, each value the resolved
#: render's ``tier_writes`` for that login. A process that reads other logins'
#: records (a shared container) cannot use its own section's tier for the
#: record's owner, so it reads the owner's entry here, as
#: :attr:`~osprey_connectors.config_sections.ControlSystemSection.login_tiers`
#: read it.
_LOGIN_TIERS_LEAF: Final = "login_tiers"

#: The three values a map entry may carry — the same three :func:`tier_writes` answers.
_LOGIN_TIERS: Final = TIER_WORDS

#: What :func:`owner_tier_term` answers: the owner's tier, ``skip`` when the rung
#: does not apply, or ``unknown`` when it applies and cannot be read.
OwnerTierTerm = Literal["skip", TierWord, "unknown"]

#: What :func:`write_ceiling` answers: a tier word, or ``unknown`` for an owner
#: rung that cannot be read.
WriteCeiling = Literal[TierWord, "unknown"]


def owner_tier_term(section: ControlSystemSection, owner: str | NoOwner | None) -> OwnerTierTerm:
    """The owner's tier out of ``control_system.login_tiers`` — the verdict's owner rung.

    *section* is the ``control_system`` section; *owner* the login whose record
    governs the write. The rung applies when the section carries the map (any
    value) **or** a control-context tree is bound (:func:`~osprey_connectors._control_state.tree_is_bound`, the
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
        login_tiers = section.login_tiers
        if login_tiers is None:
            return "unknown" if _control_state.tree_is_bound() else "skip"
        if isinstance(login_tiers, Unreadable):
            return "unknown"
        if not isinstance(owner, str) or not owner:
            return "unknown"
        tier = login_tiers.get(owner)
    except Exception:  # a rung that cannot read the map refuses; it never raises
        return "unknown"
    return tier or "unknown"


def write_ceiling(
    section: ControlSystemSection, owner: str | NoOwner | None, *, tree_bound: bool | None = None
) -> WriteCeiling:
    """The highest position a write by *owner* runs at: the verdict's ceiling.

    Under a bound tree the owner's rung (:func:`owner_tier_term`), otherwise
    :func:`~osprey_connectors.write_tier.tier_writes` of *section*. The ceiling caps
    an allowed verdict's ``position`` and refuses nothing itself; the tier and
    rung terms of :func:`write_verdict` do that. *tree_bound* answers for a
    process other than this one (``None``: :func:`~osprey_connectors._control_state.tree_is_bound`). Never raises.
    """
    if not (_control_state.tree_is_bound() if tree_bound is None else tree_bound):
        return tier_writes(section)
    rung = owner_tier_term(section, owner)
    return "unknown" if rung == "skip" else rung


# -- the write verdict ------------------------------------------------------

#: The environment variable naming the web terminal's session, the hold's key.
_SESSION_KEY_ENV_VAR: Final = "OSPREY_SESSION_ID"


# -- why a refusal refused: the cause beside the reason ------------------------


class ChannelNotListed(FrozenModel):
    """The write names a channel the authorization does not list."""

    kind: Literal["channel_not_listed"] = "channel_not_listed"
    address: str

    def sentence(self) -> str:
        """The words the refusal adds after its phrase."""
        return f"Channel: {self.address}."


class ValueOutOfRange(FrozenModel):
    """A value outside the range the authorization allows its channel.

    ``min`` and ``max`` are the channel's bounds as the authorization holds them.
    """

    kind: Literal["value_out_of_range"] = "value_out_of_range"
    address: str
    min: int | float
    max: int | float

    def sentence(self) -> str:
        """The words the refusal adds after its phrase."""
        return f"Channel: {self.address}; allowed range [{self.min}, {self.max}]."


class WrongMachine(FrozenModel):
    """The authorization covers another machine than the one written to."""

    kind: Literal["wrong_machine"] = "wrong_machine"
    covers: ControlTarget

    def sentence(self) -> str:
        """The words the refusal adds after its phrase."""
        return f"The authorization covers {self.covers}, not this machine."


class StoppedBy(FrozenModel):
    """A stop marked the authorization revoked: whose gesture, and when."""

    kind: Literal["stopped_by"] = "stopped_by"
    by: str
    revoked_at: Instant

    def sentence(self) -> str:
        """The words the refusal adds after its phrase."""
        return f"Stopped by {self.by}."


class BudgetSpent(FrozenModel):
    """Every write the authorization allows has been spent."""

    kind: Literal["budget_spent"] = "budget_spent"
    allowed: int

    def sentence(self) -> str:
        """The words the refusal adds after its phrase."""
        return f"{self.allowed} of {self.allowed} writes used."


class OverlapHeld(FrozenModel):
    """Another program holds a live authorization on a channel this one names."""

    kind: Literal["overlap_held"] = "overlap_held"
    holder: str
    authorization: str
    label: str

    def sentence(self) -> str:
        """The words the refusal adds after its phrase."""
        return f"Held by {self.holder} under {self.authorization} ({self.label})."


class _SessionHeld(FrozenModel):
    """The session's open hold flags, by what raised them, each once in the order opened."""

    kind: Literal["session_held"] = "session_held"
    sources: FrozenTuple[str]

    def sentence(self) -> str:
        """The words the refusal adds after its phrase."""
        return f"Raised by: {', '.join(self.sources)}."


#: The causes that carry nothing but their kind, each with its words.
_FixedCauseKind = Literal[
    "not_an_id",
    "empty_batch",
    "epoch_moved",
    "switch_left_allowed",
    "switch_not_at_allowed",
    "stop_during_issue",
    "no_epoch",
    "no_such_authorization",
    "no_ledger_directory",
    "no_usable_ledger_directory",
    "ledger_unreadable",
    "lease_never_started",
    "lease_already_started",
    "lease_held_elsewhere",
    "renewal_failed",
    "request_expired",
]

_FIXED_SENTENCES: Final[Mapping[_FixedCauseKind, str]] = {
    "not_an_id": "The authorization this process names is not an id.",
    "empty_batch": "The write names no channel.",
    "epoch_moved": "The switch record has moved past its approval.",
    "switch_left_allowed": "The switch is no longer at allowed.",
    "switch_not_at_allowed": "The switch is not at allowed.",
    "stop_during_issue": "A stop came during the issue.",
    "no_epoch": "The switch has no epoch.",
    "no_such_authorization": "No such authorization.",
    "no_ledger_directory": "This process has no write ledger directory.",
    "no_usable_ledger_directory": "This process has no usable write ledger directory.",
    "ledger_unreadable": "The write ledger cannot be read.",
    "lease_never_started": "The holder never started its lease.",
    "lease_already_started": "A lease was already started.",
    "lease_held_elsewhere": "Another process holds the lease.",
    "renewal_failed": "The lease could not be renewed.",
    "request_expired": "The request's expiry has passed.",
}


class FixedCause(FrozenModel):
    """A cause that names only its kind: the words are the kind's own."""

    kind: _FixedCauseKind

    def sentence(self) -> str:
        """The words the refusal adds after its phrase."""
        return _FIXED_SENTENCES[self.kind]


#: Why a refusal refused, beside its reason word: one record per kind, which a
#: program reads by its fields and a person by its :meth:`sentence`. A reason
#: word such as ``auth_revoked`` covers several causes; the cause says which.
RefusalCause: TypeAlias = Annotated[
    ChannelNotListed
    | ValueOutOfRange
    | WrongMachine
    | StoppedBy
    | BudgetSpent
    | OverlapHeld
    | _SessionHeld
    | FixedCause,
    Field(discriminator="kind"),
]


# -- the verdict: allowed, or refused for one reason ---------------------------


class _Verdict(FrozenModel):
    """What every verdict names: the machine asked about, and the authorization."""

    target: str | None
    kind: MachineKind | None
    authorization: str | None = None

    def _decision(self) -> tuple[object, ...]:
        raise NotImplementedError

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self) or not isinstance(other, _Verdict):
            return NotImplemented
        return self._decision() == other._decision()

    @override
    def __hash__(self) -> int:
        return hash(self._decision())


class Allowed(_Verdict):
    """The write may go ahead: every term of the verdict passed.

    An allowed verdict carries no reason, cause or detail; the class has no
    such fields, so one cannot be built with them.

    Attributes:
        target: The control target the caller asked about, verbatim when it was
            a string, else ``None``.
        kind: The kind of machine *target* names here — ``real``, ``rehearsal``
            or ``simulator`` — or ``None`` when the target does not resolve.
        position: The effective switch position — ``asked`` or ``allowed``, the
            stored position capped by the environment's ceiling — and ``None``
            from :func:`backstop_verdict` and for an authorized write, which the
            switch did not decide.
        authorization: The id of the program's write authorization the verdict
            was decided by (:mod:`~osprey_connectors.write_authorization`), or
            ``None`` for a verdict no authorization took part in.
    """

    allowed: Literal[True] = True
    position: ArmedPosition | None
    reason: ClassVar[None] = None
    cause: ClassVar[None] = None
    detail: ClassVar[None] = None

    @override
    def _decision(self) -> tuple[object, ...]:
        """Every field: what two allowed verdicts compare and hash by."""
        return (True, None, self.target, self.kind, self.position, self.authorization)


class Refused(_Verdict):
    """The write may not go ahead, for one reason: the first term that refused.

    Attributes:
        reason: The :class:`~osprey_connectors.refusals.VerdictReason` of the
            term that refused; its word is taken for it. Required: a refusal
            without a reason cannot be built.
            :func:`~osprey_connectors.refusals.refusal_phrase` turns it, with
            *kind*, into the words an operator reads.
        target: As for :class:`Allowed`.
        kind: As for :class:`Allowed`.
        cause: Which of the causes the reason covers refused, as a record a
            program can read (:data:`RefusalCause`), or ``None`` when the
            reason says it all. Not part of the decision, so verdicts compare
            and hash without it.
        authorization: As for :class:`Allowed`.
    """

    allowed: Literal[False] = False
    reason: VerdictReasonField
    cause: RefusalCause | None = None
    position: ClassVar[None] = None

    @property
    def detail(self) -> str | None:
        """The words the refusal adds after its phrase: its cause's sentence, or ``None``."""
        return None if self.cause is None else self.cause.sentence()

    @override
    def _decision(self) -> tuple[object, ...]:
        """Every field but ``cause``: what two refusals compare and hash by."""
        return (False, self.reason, self.target, self.kind, None, self.authorization)


#: Whether one write may go ahead: :class:`Allowed`, or :class:`Refused` with
#: the one reason. Both answer ``allowed``, ``reason``, ``cause``, ``detail``,
#: ``position``, ``target``, ``kind`` and ``authorization``.
WriteVerdict: TypeAlias = Allowed | Refused


def write_verdict(
    section: ControlSystemSection,
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
       (:func:`~osprey_connectors.write_tier.tier_allows`).
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
    8. ``launch_pinned`` — the launch pin (:func:`~osprey_connectors._control_state.launch_pin_term`) is
       ``pinned`` or ``unreadable`` for *target*; skipped when
       *ignore_launch_pin*, for a caller that reports the pin on its own.

    An allowed verdict's ``position`` is the lower of the stored position and
    :func:`write_ceiling`: :func:`~osprey_connectors.write_tier.tier_writes` of
    *section* on a host (term 2), the owner's rung under a bound tree (term 5).
    The ceiling adds no refusal.

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
        An :class:`Allowed` or a :class:`Refused`. Never raises: an exception
        anywhere in the terms refuses rather than allows.
    """
    return _write_verdict(section, target, owner, ignore_launch_pin, None)


class _RecordSeen:
    """The record one verdict read, for a caller that judges more on that same read.

    Filled by :func:`_write_refusal` once the record governing the write has
    been read and found usable: ``read`` is then ``True`` and ``record`` the
    record, or ``None`` for one that is absent (every switch off). A verdict
    refused before the record was reached leaves ``read`` ``False``. The write
    authorization's verdict takes it (:func:`_switch_basis`) so one authorized
    write is judged on one read of the record rather than two that a person's
    gesture could fall between.
    """

    __slots__ = ("read", "record")

    def __init__(self) -> None:
        self.read = False
        self.record: ControlContext | None = None


class _Armed(NamedTuple):
    """The verdict's terms all passed: the effective position the write goes ahead at."""

    position: ArmedPosition


def _write_verdict(
    section: ControlSystemSection,
    target: str | None,
    owner: str | NoOwner | None,
    ignore_launch_pin: bool,
    seen: _RecordSeen | None,
) -> WriteVerdict:
    """:func:`write_verdict`, recording the record it read into *seen* when given."""
    named = target if isinstance(target, str) else None
    kind = _verdict_kind(section, target)
    judged: VerdictReason | _Armed
    try:
        judged = _write_refusal(section, target, owner, ignore_launch_pin, seen)
    except Exception:  # a verdict that cannot be reached refuses; it never raises
        logger.warning("Write verdict for %r could not be reached; refusing", target, exc_info=True)
        judged = VerdictReason.RECORD_UNAVAILABLE
        if seen is not None:
            seen.read = False
    if isinstance(judged, _Armed):
        return Allowed(target=named, kind=kind, position=judged.position)
    return Refused(reason=judged, target=named, kind=kind)


def _verdict_kind(section: ControlSystemSection, target: object) -> MachineKind | None:
    """The machine kind a verdict on *target* names, or ``None``. Never raises.

    The name decides whether the target resolves at all; the published report
    decides what it is (C-KIND), so a stand-in the controls server has not
    vouched for is the real machine in the refusal's words.
    """
    if not is_control_target(target):
        return None
    try:
        if machine_kind(section, target) is None:
            return None
        return control_context.published_machine_kind(target)
    except Exception:  # the kind is a word in a refusal; it must not decide one
        logger.debug("Could not name the machine kind of %r", target, exc_info=True)
        return None


def backstop_verdict(section: ControlSystemSection) -> WriteVerdict:
    """The verdict for a write no target or owner can be named for.

    The floor under every write path that cannot be traced to one machine: it
    refuses ``readonly_run`` (``OSPREY_EXECUTION_MODE``), then ``tier_never``
    (:func:`~osprey_connectors.write_tier.tier_allows`), in the same order as the
    first two terms of :func:`write_verdict`, and otherwise allows. It is blind
    by design — it resolves no target, asks no owner and opens no record, so a
    switch position never changes its answer and it reports none.

    Args:
        section: The ``control_system`` section of the resolved config.

    Returns:
        An :class:`Allowed` or a :class:`Refused` whose ``target``, ``kind`` and
        ``position`` are always ``None``. Never raises: a tier that cannot be
        read refuses as ``tier_never``.
    """
    if is_readonly_run():
        return Refused(reason=VerdictReason.READONLY_RUN, target=None, kind=None)
    try:
        allows = tier_allows(section)
    except Exception:  # tier_allows never raises by contract; refuse if it does
        logger.warning("Backstop tier could not be read; refusing", exc_info=True)
        allows = False
    if not allows:
        return Refused(reason=VerdictReason.TIER_NEVER, target=None, kind=None)
    return Allowed(target=None, kind=None, position=None)


def session_hold_refusal(
    *, root: Path | str | None = None, session_keys: Sequence[str] | None = None
) -> Refused | None:
    """The ``session_held`` refusal for this process's session, or ``None``.

    The session is :data:`_SESSION_KEY_ENV_VAR`; a process without one belongs to
    no session and is never held. A process that serves several sessions at
    once — a panel's data process, which starts before any conversation exists
    and answers a terminal's every session — names them in *session_keys*
    instead, and is held while any one of them is. The store is *root*, else
    the stamped :data:`~osprey_connectors.session_hold.HOLD_DIR_ENV_VAR`;
    without either nothing is held. Under a store, a record that cannot be
    read refuses.

    Args:
        root: The hold root, for a caller that resolved it itself.
        session_keys: The sessions to ask about, in place of the environment's
            one key. An empty sequence belongs to no session; an empty key in it
            names none. ``None`` reads the environment.

    Returns:
        ``None`` when no named session is held; otherwise the first held
        session's refusal, in the order given: ``target`` and ``kind`` are
        ``None`` and ``cause`` names that session's open flags' sources, each
        once in the order opened (its ``detail`` is ``Raised by: <sources>.``).
        A hold that cannot be read, and an unsafe session key, has no flags to
        name and no cause.
        Never raises.
    """
    for key in _session_keys(session_keys):
        refusal = _hold_refusal(key, root)
        if refusal is not None:
            return refusal
    return None


def _session_keys(session_keys: Sequence[str] | None) -> list[str]:
    """The session keys a hold is asked about: *session_keys* as given, else the environment's."""
    if session_keys is None:
        key = os.environ.get(_SESSION_KEY_ENV_VAR, "")
        return [key] if key else []
    return [key for key in session_keys if key]


def _hold_refusal(key: str, root: Path | str | None) -> Refused | None:
    """The ``session_held`` refusal for one session *key*, or ``None`` (see :func:`session_hold_refusal`)."""
    cause = None
    try:
        # One read: the decision and the words it is refused in come from one answer.
        flags = session_hold.open_flags(key, root=root)
        if not flags:
            return None
        cause = _SessionHeld(sources=tuple(dict.fromkeys(flag.source for flag in flags)))
    except ValueError:  # the key is not a safe file name, which holds
        logger.warning("Session key %r is unsafe; held", key)
    except Exception:  # a hold that cannot be read refuses; it never raises
        logger.warning("Session hold for %r could not be read; held", key, exc_info=True)
    return Refused(reason=VerdictReason.SESSION_HELD, target=None, kind=None, cause=cause)


def execution_verdict(
    section: ControlSystemSection,
    target: str | None,
    *,
    owner: str | NoOwner | None = None,
    ignore_launch_pin: bool = False,
    session_keys: Sequence[str] | None = None,
) -> WriteVerdict:
    """The verdict on one write about to execute: :func:`write_verdict` plus the hold.

    The terms are ``readonly_run``, ``tier_never``, ``session_held``
    (:func:`session_hold_refusal`), then the rest of :func:`write_verdict` in its
    order, so a held session with its switch off refuses ``session_held``. Its
    order is shared with the hook's stdlib twin.

    Args:
        section: The ``control_system`` section of the resolved config.
        target: The control target the write lands on.
        owner: As for :func:`write_verdict`.
        ignore_launch_pin: As for :func:`write_verdict`.
        session_keys: As for :func:`session_hold_refusal`: the sessions the
            write belongs to, held while any one is; ``None`` is the
            environment's session.

    Returns:
        An :class:`Allowed` or a :class:`Refused` with the ``target`` and
        ``kind`` of :func:`write_verdict`. Never raises.
    """
    verdict = write_verdict(section, target, owner=owner, ignore_launch_pin=ignore_launch_pin)
    return _with_hold(section, verdict, session_keys)


def _with_hold(
    section: ControlSystemSection, verdict: WriteVerdict, session_keys: Sequence[str] | None
) -> WriteVerdict:
    """*verdict* as :func:`execution_verdict` answers it: the hold after the backstop."""
    if not backstop_verdict(section).allowed:
        return verdict
    held = session_hold_refusal(session_keys=session_keys)
    if held is None:
        return verdict
    return evolve(held, target=verdict.target, kind=verdict.kind)


def _switch_basis(
    section: ControlSystemSection,
    target: str | None,
    owner: str | NoOwner | None,
    session_keys: Sequence[str] | None,
    *,
    hold: bool,
) -> tuple[WriteVerdict, bool, ControlContext | None]:
    """The switch verdict a write authorization's verdict follows, and the record it read.

    :func:`write_verdict` for *target* and *owner*, launch pin honoured, and
    with *hold* the session's hold placed as :func:`execution_verdict` places
    it (*session_keys* as there). Then whether that verdict read the record
    governing the write and found it usable, and the record (``None`` for an
    absent one, every switch off), so the authorization is judged on the same
    read. Never raises.
    """
    seen = _RecordSeen()
    verdict = _write_verdict(section, target, owner, False, seen)
    if hold:
        verdict = _with_hold(section, verdict, session_keys)
    return verdict, seen.read, seen.record


def _write_refusal(
    section: ControlSystemSection,
    target: str | None,
    owner: str | NoOwner | None,
    ignore_launch_pin: bool,
    seen: _RecordSeen | None = None,
) -> VerdictReason | _Armed:
    """The first term of :func:`write_verdict` that refuses, else the position.

    Answers the refusing term's reason, or :class:`_Armed` with the effective
    position once every term passed — the last thing it answers. May raise.
    The usable record it read, if it got that far, is left in *seen*.
    """
    if is_readonly_run():
        return VerdictReason.READONLY_RUN
    if not tier_allows(section):
        return VerdictReason.TIER_NEVER
    if not is_control_target(target):  # resolve_target refuses it too; said here for the type
        return VerdictReason.TARGET_UNKNOWN
    try:
        resolve_target(section, target)
    except Exception:  # ValueError by contract; anything else is no resolution either
        return VerdictReason.TARGET_UNKNOWN

    resolved = current_owner(owner)
    if resolved is NO_OWNER and _control_state.is_dispatch_run():
        return VerdictReason.OWNER_UNSCHEDULED
    if resolved is NO_OWNER or not isinstance(resolved, str) or not resolved.strip():
        return VerdictReason.OWNER_UNKNOWN
    # A host holds one login's switch: an owner other than that login is refused
    # before its tier or any record is read (a tree bind reads each owner's own).
    if not _control_state.tree_is_bound() and resolved.strip() != acting_identity():
        return VerdictReason.OWNER_FOREIGN

    rung = owner_tier_term(section, resolved)
    if rung == "unknown":
        return VerdictReason.OWNER_UNKNOWN
    if rung == "never":
        return VerdictReason.TIER_NEVER

    if _control_state.tree_is_bound():
        tree = _control_state.control_context_tree()
        if tree is None:
            logger.warning(
                "Write record unreadable for %r: %s", resolved, _control_state.tree_bind_unusable()
            )
            return VerdictReason.RECORD_UNAVAILABLE
        read = control_context.read_tree_context(tree, resolved)
        if read.unavailable is not None:
            logger.warning("Write record unreadable for %r: %s", resolved, read.unavailable)
            return VerdictReason.RECORD_UNAVAILABLE
        record = read.record
    else:
        own = control_context.read_record_status()
        record, status = own.value, own.status
        if status == "unreadable":
            logger.warning("Write record unreadable on this host; refusing")
            return VerdictReason.RECORD_UNAVAILABLE
        if status != "ok":
            record = None
    if seen is not None:
        seen.read, seen.record = True, record

    stored = control_context.switch_position(record, target)
    if stored == "off":
        return VerdictReason.SWITCH_OFF

    if not ignore_launch_pin and _control_state.launch_pin_term(target) != "none":
        return VerdictReason.LAUNCH_PINNED
    ceiling = write_ceiling(section, resolved)
    return _Armed(cast("ArmedPosition", min(stored, ceiling, key=_LOGIN_TIERS.index)))
