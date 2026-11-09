"""A program's bounded write authorization: which channels, which values, how many writes.

The write switch is a person's gesture over a whole machine. A program such as
an optimizer needs less than that and must get no more: an **authorization**
(a *grant*) names one machine, the login whose switch it stands in for (the
**owner**), the identity that may use it (the **holder**), a closed range per
channel, a budget of write **steps**, a hard expiry, a **lease** the holder
renews, and who approved it. Every write the holder makes is checked against
it; it ends at an exhausted budget, an expiry, a lapsed lease or a stop, and
nothing renews the budget. The machine's switch may stay off throughout.

This module owns the two files, their readers and their writers, and the
verdict. The write guard of every control-system connector
(:class:`~osprey_connectors.control_system.ControlSystemConnector`) asks
:func:`authorized_execution_verdict` for every write call of a process that
names an authorization (:data:`AUTHORIZATION_ENV_VAR`), spending its step, and
the connector-host pool sets that variable for a child whose key names one; a
gateway choice and a child's report take :func:`authorized_write_verdict`.

**1. Two files, two writers.** The grant is ``<authorizations dir>/<id>.json``,
written once by the approver's process and afterwards only to mark it
revoked (:func:`issue_grant`, :func:`revoke_grants`). The authorizations
directory is the owner's: :data:`AUTHORIZATIONS_DIR_NAME` inside the owner's
directory of the read-only control-context tree when one is bound, else inside
this host's :func:`~osprey_connectors.posture_store.state_dir`. The **ledger**
is ``<ledger dir>/<id>.ledger.json``, written only by the holder: the holder
opens it with its lease (:func:`start_lease`), its heartbeat renews the lease
(:func:`renew_lease`), and the connector child appends one step per write call
(:func:`spend_step`). The ledger directory is :data:`LEDGER_DIR_ENV_VAR`, else,
on a host with no tree bound, the host's authorizations directory; under a
bound tree with no ledger directory there is none, and every write refuses.
The control-context record is not touched: ``control_context.json`` keeps its
schema and its seven fields.

**2. Reads fail closed.** Both files are read the way the posture store reads
the tree (:func:`~osprey_connectors.posture_store._read_tree_context`): no
symbolic link is followed at any directory below the root or at the file (the
file is opened ``O_NOFOLLOW``), the read is bounded, the file must be a regular
file, and the payload must be exactly this schema — every field present, of its
type, no other field, no repeated key, every number finite. A grant that is
absent is no authorization (``switch_off``); every failure on a path that is
there refuses (``auth_unreadable``), and so do a missing or unusable ledger
directory and a ledger that cannot be read. A ledger that is absent has no
lease, and refuses ``auth_lease_lapsed``.

**3. The verdict.** :func:`authorization_verdict` judges one write call — a
whole batch of ``(channel, value)`` operations — in one total order: the launch
pin (:func:`~osprey_connectors.posture_store._launch_pin_term`, read here
because the switch verdict reaches it only with the switch on), the owner, the
record, the grant, its holder and owner, its machine, a stop, its expiry, the
switch position for a self-issued grant, every channel listed and every value a
finite number in its range, the lease, and a step left. It does not repeat the
switch verdict's earlier terms; :func:`authorized_execution_verdict` is the
composition a write path takes: the execution verdict first, and for a process
that names an authorization, whenever that verdict allows or refuses only
``switch_off``, this one in its place. An authorization never lifts a tier, an
unknown owner, an unreadable record, a pin or a hold, and a person turning the
switch on mid-run never lifts the bounds.

**4. One write call is one step.** :func:`spend_step` takes the ledger's lock
(an exclusive ``flock`` on ``<ledger>.lock`` beside it, as
:mod:`~osprey_connectors.session_hold` takes its record's), re-judges the whole
verdict against the ledger it read under that lock, and appends the step
through :func:`~osprey_connectors.safety_file.atomic_replace` before it
answers. The caller writes only after an allowed answer, so a crash between
the two costs the step and never writes unbudgeted. Concurrent appenders in
any number of processes on one kernel are serialised by the lock.

A step is spent before the connector's own checks run: a write the limits
database then refuses has still spent it, and so has one whose put fails.
Both fail closed. A stop that lands after a write call's verdict and before
its put does not reach that one call, which goes out as judged; the next call
is refused. The switch's own verdict has the same window.

**5. The lease.** The holder renews ``lease_until_ns``; a lease that lapses
ends the authorization for good — :func:`renew_lease` refuses once it has
lapsed. Where the reader shares the holder's PID namespace, the holder's PID,
its start time and its state are also asked (:func:`pid_namespace`,
:func:`process_start_time`): a dead, reused or zombie PID lapses the lease
before its time. From another namespace the lease time alone decides.

**6. Self-issue at ``allowed``.** A holder whose owner's switch stands at
``allowed`` for the machine may write its own grant into its ledger directory
(:func:`self_issue`), with approver surface ``write_switch``. Such a grant is
honoured only while the position stays ``allowed``: the first write after it
leaves ``allowed`` is refused and lapses the lease, so the switch coming back
does not revive it. A grant found in the ledger directory that claims any
other surface is not honoured at all, because the holder can write there.

**7. Overlap.** Two programs steering one channel of one machine is never what
an approver meant: :func:`auth_overlap` finds the live grant of another holder
sharing a channel, and :func:`issue_grant` refuses ``auth_overlap`` under the
directory's lock, so two approvals cannot both pass the check. A grant whose
every ledger in sight shows its lease time passed does not count: no write may
use it. The check reads the owner's own directory and the holders'
self-issued grants in the ledger directories this process sees
(:func:`self_issued_directories`); another owner's directory is not read. A
grant file that cannot be read is logged and left out of the check: it
refuses every write under it already, and a file nobody can clear from the
chip must not veto every approval.

**8. Requests, approval and the stop.** A holder asks with a **request**,
``<ledger dir>/requests/<id>.json`` (:func:`write_request`), read the way a
grant is read (:func:`parse_request`, :func:`list_requests`); a web terminal
reads the holders' requests from a read-only bind, :data:`REQUESTS_DIR_ENV_VAR`.
A person acting as the owner approves it (:func:`approve_request`): under the
owner's record lock, then the grants directory's — the one lock order every
writer of both keeps — the grant is written under the request's own id, so the
holder waits for exactly that id (:func:`approved_grant`). The switch's own
writer, :func:`~osprey_connectors.control_context.set_switch`, revokes every
grant on the machines a stop names inside its record lock
(:func:`revoke_grants`), so a stop is one gesture and refuses the next write;
a stop whose marks cannot be written moves the record's epoch instead, which
ends every grant of that owner. Every lock here is created at 0o660 whatever
the umask, because a grants directory is written by several uids through its
group. A listing removes grants that ended more than :data:`PRUNE_AFTER_S` ago
(:func:`prune_grants`).

This module imports nothing from ``osprey`` and sits above the record and the
posture store, which it reads.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import logging
import math
import numbers
import os
import re
import stat
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Literal

from osprey_connectors import control_context, posture_store
from osprey_connectors.identity import acting_identity
from osprey_connectors.posture_store import NO_OWNER, NoOwner, WriteVerdict
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors.safety_file import atomic_replace

logger = logging.getLogger("osprey_connectors.write_authorization")

__all__ = [
    "APPROVAL_SURFACES",
    "AUTHORIZATIONS_DIR_NAME",
    "AUTHORIZATION_ENV_VAR",
    "COMPLETIONS",
    "LEDGER_DIR_ENV_VAR",
    "LEDGER_VIEW_DIR_ENV_VAR",
    "MAX_EXPIRES_S",
    "PRUNE_AFTER_S",
    "REQUESTS_DIR_ENV_VAR",
    "REQUESTS_DIR_NAME",
    "SCHEMA_VERSION",
    "SELF_ISSUE_SURFACE",
    "SURFACES",
    "Approver",
    "AuthorizationRefused",
    "ChannelRange",
    "Grant",
    "GrantStanding",
    "Ledger",
    "LedgerStep",
    "Revocation",
    "WriteRequest",
    "approve_request",
    "approved_grant",
    "auth_overlap",
    "authorization_verdict",
    "authorizations_dir",
    "authorized_execution_verdict",
    "authorized_write_verdict",
    "grant_in_force",
    "grant_standing",
    "issue_grant",
    "ledger_dir",
    "list_grants",
    "list_requests",
    "listed_grants",
    "named_authorization",
    "parse_grant",
    "parse_ledger",
    "parse_request",
    "pid_namespace",
    "process_start_time",
    "prune_grants",
    "read_ledger",
    "read_request",
    "renew_lease",
    "requests_dir",
    "revoke_grants",
    "self_issue",
    "self_issued_directories",
    "self_issued_grants",
    "spend_step",
    "start_lease",
    "withdraw_request",
    "write_request",
]

#: The environment variable a holder's connector child carries: the id of the
#: authorization its writes are judged by.
AUTHORIZATION_ENV_VAR: Final = "OSPREY_WRITE_AUTHORIZATION"

#: The environment variable naming the holder's ledger directory, a path in the
#: holder's own filesystem that it can write.
LEDGER_DIR_ENV_VAR: Final = "OSPREY_WRITE_LEDGER_DIR"

#: The directory under an identity's control-state directory holding its grants.
AUTHORIZATIONS_DIR_NAME: Final = "authorizations"

#: The format of both files. Any other value is unreadable, and refuses.
SCHEMA_VERSION: Final = 1

#: The most write calls one grant or request may allow: far above any run's
#: evaluations, so no plan, typo or forged file asks for an unbounded budget.
MAX_STEPS: Final = 100_000
#: The longest a grant may stand from its issue, or a request from its asking:
#: a day, three times the optimizer's eight-hour default.
MAX_EXPIRES_S: Final = 24 * 3600.0
#: The longest one renewal of a lease may last: ten times the optimizer's
#: minute, so a holder gone unseen (from another PID namespace) is found
#: lapsed within ten minutes, by the verdict and by the overlap check alike.
MAX_LEASE_S: Final = 600.0

#: Who recorded an approval: the web terminal's chip, the CLI, or the switch
#: itself standing at ``allowed`` (a self-issued grant).
SURFACES: Final = ("chip", "cli", "write_switch")

#: The surface of a self-issued grant.
SELF_ISSUE_SURFACE: Final = "write_switch"

_GRANT_SUFFIX: Final = ".json"
_LEDGER_SUFFIX: Final = ".ledger.json"
_LOCK_SUFFIX: Final = ".lock"

#: The lock a grants directory is written under: check-and-issue is one hold.
_DIRECTORY_LOCK_NAME: Final = "authorizations.lock"

#: An authorization id: ``uuid.uuid4().hex``. Anything else names no file.
_ID: Final = re.compile(r"[0-9a-f]{32}")
_GRANT_FILE: Final = re.compile(r"([0-9a-f]{32})\.json")

#: A grant is a few hundred bytes plus one line per channel.
_GRANT_READ_LIMIT: Final = 1024 * 1024
#: A ledger grows by one short entry per step.
_LEDGER_READ_LIMIT: Final = 16 * 1024 * 1024

_LOCK_TIMEOUT_S: Final = 5.0
_LOCK_POLL_S: Final = 0.01
_FILE_MODE: Final = 0o640

_PROC_STAT: Final = "/proc/{pid}/stat"
_PID_NS_LINK: Final = "/proc/self/ns/pid"

#: Characters that cannot appear in the one path segment an owner names.
_UNSAFE_SEGMENT: Final = frozenset({"/", "\\", "\x00"})

_Status = Literal["ok", "absent", "unreadable"]


def _now_ns() -> int:
    """The time every term is judged at: one clock, the record's epoch's."""
    return time.time_ns()


# -- the formats --------------------------------------------------------------


@dataclass(frozen=True)
class ChannelRange:
    """The closed range of values one channel may be written with."""

    min: float
    max: float

    def holds(self, value: object) -> bool:
        """Whether *value* is a finite real number inside the range."""
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return False
        number = float(value)
        return math.isfinite(number) and self.min <= number <= self.max


@dataclass(frozen=True)
class Approver:
    """Who recorded the approval, from which surface, and when."""

    identity: str
    surface: Literal["chip", "cli", "write_switch"]
    at_ns: int


@dataclass(frozen=True)
class Revocation:
    """A stop that ended a grant: when, and whose gesture."""

    at_ns: int
    by: str


@dataclass(frozen=True)
class Grant:
    """One authorization, as its file states it. Treat as immutable.

    Attributes:
        id: ``uuid4().hex``; the file is ``<id>.json``.
        owner: The login whose switch the grant stands in for.
        holder: The acting identity that may write under it.
        target: The control target it covers.
        switch_epoch: The owner's record's ``switch_epoch`` at issue; a record
            whose epoch has moved since (a stop of every machine) ends it.
        channels: ``{address: ChannelRange}``, at least one.
        steps: How many write calls it allows, at least one.
        issued_at_ns: When it was written.
        expires_at_ns: When it ends whatever else happens.
        lease_s: How long one renewal of the holder's lease lasts.
        approver: Who approved it.
        label: The run it was approved for, free text.
        revoked: The stop that ended it, or ``None``.
    """

    id: str
    owner: str
    holder: str
    target: str
    switch_epoch: int
    channels: Mapping[str, ChannelRange]
    steps: int
    issued_at_ns: int
    expires_at_ns: int
    lease_s: float
    approver: Approver
    label: str = ""
    revoked: Revocation | None = None

    def to_payload(self) -> dict[str, Any]:
        """This grant as the JSON object its file holds."""
        return {
            "schema": SCHEMA_VERSION,
            "id": self.id,
            "owner": self.owner,
            "holder": self.holder,
            "target": self.target,
            "switch_epoch": self.switch_epoch,
            "channels": {
                address: {"min": bound.min, "max": bound.max}
                for address, bound in self.channels.items()
            },
            "steps": self.steps,
            "issued_at_ns": self.issued_at_ns,
            "expires_at_ns": self.expires_at_ns,
            "lease_s": self.lease_s,
            "approver": {
                "identity": self.approver.identity,
                "surface": self.approver.surface,
                "at_ns": self.approver.at_ns,
            },
            "label": self.label,
            "revoked": (
                None
                if self.revoked is None
                else {"at_ns": self.revoked.at_ns, "by": self.revoked.by}
            ),
        }

    def live_at(self, now_ns: int) -> bool:
        """Not revoked and not expired at *now_ns*. The lease is the ledger's."""
        return self.revoked is None and now_ns < self.expires_at_ns


@dataclass(frozen=True)
class LedgerStep:
    """One write call spent: its number, when, and the channels it named."""

    n: int
    at_ns: int
    channels: tuple[str, ...]


@dataclass(frozen=True)
class Ledger:
    """The holder's consumption of one authorization. Treat as immutable.

    Attributes:
        authorization: The grant's id.
        holder_pid: The PID of the process that renews the lease, or ``None``
            to rely on the lease time alone.
        holder_start: That process's start time as the kernel states it, so a
            reused PID is not taken for it; ``None`` where nothing states it.
        pid_ns: That process's PID namespace (:func:`pid_namespace`), so a
            reader in another namespace does not look its PID up.
        lease_until_ns: When the lease lapses unless renewed first.
        steps: Every write call spent, in order, numbered from 1.
    """

    authorization: str
    holder_pid: int | None
    holder_start: str | None
    pid_ns: str | None
    lease_until_ns: int
    steps: tuple[LedgerStep, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, Any]:
        """This ledger as the JSON object its file holds."""
        return {
            "schema": SCHEMA_VERSION,
            "authorization": self.authorization,
            "holder_pid": self.holder_pid,
            "holder_start": self.holder_start,
            "pid_ns": self.pid_ns,
            "lease_until_ns": self.lease_until_ns,
            "steps": [
                {"n": step.n, "at_ns": step.at_ns, "channels": list(step.channels)}
                for step in self.steps
            ],
        }


class AuthorizationRefused(Exception):
    """A writer of this module refused, for one of the refusal reasons.

    Attributes:
        reason: A member of :data:`~osprey_connectors.refusals.REASONS`.
        detail: What the refusal adds after its phrase, or ``None``.
    """

    def __init__(self, reason: str, detail: str | None = None) -> None:
        self.reason = reason
        self.detail = detail
        phrase = refusal_phrase(reason)
        super().__init__(f"{phrase} {detail}" if detail else phrase)


# -- parsing ------------------------------------------------------------------


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a finite number")


def _no_repeated_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"the key {key!r} is repeated")
        payload[key] = value
    return payload


def _load_json(raw: str | bytes) -> Any:
    """*raw* as JSON with no ``NaN``/``Infinity`` and no repeated key."""
    return json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_no_repeated_keys)


def _exact_keys(payload: Any, keys: frozenset[str], what: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{what} is not an object")
    if set(payload) != keys:
        raise ValueError(f"{what} has keys {sorted(payload)}, not {sorted(keys)}")
    return payload


def _name(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{what} is not a name")
    return value


def _integer(value: Any, what: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{what} is not an integer of at least {minimum}")
    return int(value)


def _finite(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{what} is not a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} is not finite")
    return number


def _within_ceilings(steps: int, start_ns: int, expires_ns: int, lease_s: float) -> None:
    """Refuse a budget, an expiry or a lease past its ceiling.

    The ceilings are :data:`MAX_STEPS`, :data:`MAX_EXPIRES_S` from *start_ns*,
    and :data:`MAX_LEASE_S`. Every grant and request is parsed through this,
    so an approval and a self-issue meet the same ceilings, and a file written
    past them is unreadable.
    """
    if steps > MAX_STEPS:
        raise ValueError(f"steps {steps} is above the ceiling of {MAX_STEPS}")
    if expires_ns - start_ns > int(MAX_EXPIRES_S * 1e9):
        raise ValueError(f"the expiry is more than {MAX_EXPIRES_S:g} s away")
    if lease_s > MAX_LEASE_S:
        raise ValueError(f"lease_s {lease_s:g} is above the ceiling of {MAX_LEASE_S:g}")


def _authorization_id(value: Any) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError("the id is not a 32-digit lowercase hex string")
    return value


_GRANT_KEYS: Final = frozenset(
    {
        "schema",
        "id",
        "owner",
        "holder",
        "target",
        "switch_epoch",
        "channels",
        "steps",
        "issued_at_ns",
        "expires_at_ns",
        "lease_s",
        "approver",
        "label",
        "revoked",
    }
)
_RANGE_KEYS: Final = frozenset({"min", "max"})
_APPROVER_KEYS: Final = frozenset({"identity", "surface", "at_ns"})
_REVOKED_KEYS: Final = frozenset({"at_ns", "by"})
_LEDGER_KEYS: Final = frozenset(
    {"schema", "authorization", "holder_pid", "holder_start", "pid_ns", "lease_until_ns", "steps"}
)
_STEP_KEYS: Final = frozenset({"n", "at_ns", "channels"})


def _parse_channels(value: Any) -> dict[str, ChannelRange]:
    if not isinstance(value, dict) or not value:
        raise ValueError("channels is not a non-empty object")
    channels: dict[str, ChannelRange] = {}
    for address, bound in value.items():
        if not isinstance(address, str) or not address.strip():
            raise ValueError("a channel address is empty")
        entry = _exact_keys(bound, _RANGE_KEYS, f"the range of {address!r}")
        low = _finite(entry["min"], f"the min of {address!r}")
        high = _finite(entry["max"], f"the max of {address!r}")
        if low > high:
            raise ValueError(f"the range of {address!r} has min above max")
        channels[address] = ChannelRange(min=low, max=high)
    return channels


def parse_grant(raw: Any) -> Grant:
    """A grant out of its JSON text, bytes or decoded object.

    Every field must be present with its type, and nothing else may be: an
    unknown schema, a missing or extra field, a repeated key, a bound that is
    not a finite number, a range with ``min`` above ``max``, a ``steps`` below
    one, an expiry not after the issue, an unknown surface, or a budget, an
    expiry or a lease past its ceiling (:func:`_within_ceilings`) — each is a
    grant this version cannot honour.

    Raises:
        ValueError: The payload is not a grant this version can read.
    """
    payload = _load_json(raw) if isinstance(raw, str | bytes | bytearray) else raw
    payload = _exact_keys(payload, _GRANT_KEYS, "the grant")
    if payload["schema"] is not SCHEMA_VERSION:
        raise ValueError(f"the grant's schema is not {SCHEMA_VERSION}")
    approver = _exact_keys(payload["approver"], _APPROVER_KEYS, "the approver")
    surface = approver["surface"]
    if surface not in SURFACES:
        raise ValueError(f"the approver's surface {surface!r} is not one of {SURFACES}")
    revoked_raw = payload["revoked"]
    revoked = None
    if revoked_raw is not None:
        entry = _exact_keys(revoked_raw, _REVOKED_KEYS, "the revocation")
        revoked = Revocation(
            at_ns=_integer(entry["at_ns"], "revoked.at_ns", 0), by=_name(entry["by"], "revoked.by")
        )
    issued = _integer(payload["issued_at_ns"], "issued_at_ns", 0)
    expires = _integer(payload["expires_at_ns"], "expires_at_ns", 0)
    if expires <= issued:
        raise ValueError("the grant expires before it was issued")
    lease_s = _finite(payload["lease_s"], "lease_s")
    if lease_s <= 0:
        raise ValueError("lease_s is not positive")
    steps = _integer(payload["steps"], "steps", 1)
    _within_ceilings(steps, issued, expires, lease_s)
    label = payload["label"]
    if not isinstance(label, str):
        raise ValueError("label is not a string")
    return Grant(
        id=_authorization_id(payload["id"]),
        owner=_name(payload["owner"], "owner"),
        holder=_name(payload["holder"], "holder"),
        target=_name(payload["target"], "target"),
        switch_epoch=_integer(payload["switch_epoch"], "switch_epoch", 0),
        channels=_parse_channels(payload["channels"]),
        steps=steps,
        issued_at_ns=issued,
        expires_at_ns=expires,
        lease_s=lease_s,
        approver=Approver(
            identity=_name(approver["identity"], "approver.identity"),
            surface=surface,
            at_ns=_integer(approver["at_ns"], "approver.at_ns", 0),
        ),
        label=label,
        revoked=revoked,
    )


def _optional_str(value: Any, what: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{what} is neither a string nor null")
    return value


def parse_ledger(raw: Any) -> Ledger:
    """A ledger out of its JSON text, bytes or decoded object.

    The same strictness as :func:`parse_grant`; the steps must be numbered
    1, 2, … in order, each naming at least one channel.

    Raises:
        ValueError: The payload is not a ledger this version can read.
    """
    payload = _load_json(raw) if isinstance(raw, str | bytes | bytearray) else raw
    payload = _exact_keys(payload, _LEDGER_KEYS, "the ledger")
    if payload["schema"] is not SCHEMA_VERSION:
        raise ValueError(f"the ledger's schema is not {SCHEMA_VERSION}")
    holder_pid = payload["holder_pid"]
    if holder_pid is not None:
        holder_pid = _integer(holder_pid, "holder_pid", 1)
    entries = payload["steps"]
    if not isinstance(entries, list):
        raise ValueError("steps is not a list")
    steps = []
    for index, raw_step in enumerate(entries, start=1):
        entry = _exact_keys(raw_step, _STEP_KEYS, f"step {index}")
        if _integer(entry["n"], f"step {index}'s n", 1) != index:
            raise ValueError(f"step {index} is numbered {entry['n']}")
        channels = entry["channels"]
        if (
            not isinstance(channels, list)
            or not channels
            or not all(isinstance(c, str) and c for c in channels)
        ):
            raise ValueError(f"step {index}'s channels is not a non-empty list of names")
        steps.append(
            LedgerStep(
                n=index,
                at_ns=_integer(entry["at_ns"], f"step {index}'s at_ns", 0),
                channels=tuple(channels),
            )
        )
    return Ledger(
        authorization=_authorization_id(payload["authorization"]),
        holder_pid=holder_pid,
        holder_start=_optional_str(payload["holder_start"], "holder_start"),
        pid_ns=_optional_str(payload["pid_ns"], "pid_ns"),
        lease_until_ns=_integer(payload["lease_until_ns"], "lease_until_ns", 0),
        steps=tuple(steps),
    )


# -- where the files are ------------------------------------------------------


@dataclass(frozen=True)
class _Place:
    """A directory as a trusted *root* and the segments below it.

    The root is a bind or a resolved state directory and is taken as it is;
    every segment below it is checked with :func:`os.lstat` on every read, so
    a link planted at any of them is refused rather than followed.
    """

    root: Path
    segments: tuple[str, ...] = ()

    @property
    def path(self) -> Path:
        return self.root.joinpath(*self.segments)


def _safe_segment(name: Any) -> bool:
    return (
        isinstance(name, str)
        and name.strip() not in {"", ".", ".."}
        and not (_UNSAFE_SEGMENT & set(name))
    )


def _owner_place(owner: str) -> _Place | None:
    """Where *owner*'s grants live for this process, or ``None`` when nowhere usable.

    Under a bound tree, ``<tree>/<owner>/authorizations``; on a host, the
    host's own state directory, which holds only its own login's.
    """
    if posture_store._tree_is_bound():
        tree = posture_store._control_context_tree()
        if tree is None or not _safe_segment(owner):
            return None
        return _Place(tree, (owner, AUTHORIZATIONS_DIR_NAME))
    if owner != acting_identity():
        return None
    root = posture_store.state_dir()
    return None if root is None else _Place(root, (AUTHORIZATIONS_DIR_NAME,))


def _ledger_place() -> _Place | None:
    """The holder's ledger directory, or ``None`` when this process has none."""
    bound = posture_store._absolute_bind(LEDGER_DIR_ENV_VAR)
    if bound is not None:
        return _Place(bound)
    if posture_store._tree_is_bound():
        return None
    root = posture_store.state_dir()
    return None if root is None else _Place(root, (AUTHORIZATIONS_DIR_NAME,))


def authorizations_dir(owner: str | None = None) -> Path | None:
    """The directory *owner*'s grants live in, as this process reaches it.

    *owner* ``None`` is this process's acting identity. ``None`` when this
    process reaches no such directory: a host asked about another login, an
    unusable tree bind or an owner that is not one path segment.
    """
    place = _owner_place(acting_identity() if owner is None else owner)
    return None if place is None else place.path


def ledger_dir() -> Path | None:
    """This process's ledger directory, or ``None`` when it has none (rule 1)."""
    place = _ledger_place()
    return None if place is None else place.path


def named_authorization() -> str | None:
    """The authorization :data:`AUTHORIZATION_ENV_VAR` names, or ``None``.

    Blank is unset. A value that is not a well-formed id is returned as it is,
    so the verdict refuses it rather than treating the process as naming none.
    """
    value = (os.environ.get(AUTHORIZATION_ENV_VAR) or "").strip()
    return value or None


# -- reading ------------------------------------------------------------------


def _dir_status(place: _Place) -> _Status:
    """Whether *place* is a plain directory: ``absent`` at the first missing segment.

    ``unreadable`` for the root not being a directory, a link or a non-directory
    at any segment, and any other failure.
    """
    try:
        if not place.root.is_dir():
            return "absent" if not place.root.exists() else "unreadable"
    except OSError:
        return "unreadable"
    probe = place.root
    for segment in place.segments:
        probe = probe / segment
        try:
            mode = os.lstat(probe).st_mode
        except FileNotFoundError:
            return "absent"
        except OSError:
            return "unreadable"
        if not stat.S_ISDIR(mode):
            return "unreadable"
    return "ok"


def _read_no_follow(path: Path, limit: int) -> bytes:
    """*path*'s bytes, refusing a link, a non-regular file and more than *limit* bytes.

    ``O_NONBLOCK`` so a FIFO planted at the name cannot stall the open; the
    regular-file check after it refuses one.

    Raises:
        OSError: Absent (``FileNotFoundError``), a link (``ELOOP``), not a
            regular file, too long (``EFBIG``), or unreadable.
    """
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, f"{path} is not a regular file")
        chunks = []
        remaining = limit + 1
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(fd)
    data = b"".join(chunks)
    if len(data) > limit:
        raise OSError(errno.EFBIG, f"{path} is larger than {limit} bytes")
    return data


def _read_grant_at(place: _Place, authorization: str) -> tuple[_Status, Grant | None]:
    """The grant ``<id>.json`` under *place*, with how the read ended.

    What went wrong is logged with the path; a refusal never names a file.
    """
    directory = _dir_status(place)
    if directory != "ok":
        if directory == "unreadable":
            logger.warning("The authorizations directory %s is not a plain directory", place.path)
        return directory, None
    path = place.path / f"{authorization}{_GRANT_SUFFIX}"
    try:
        grant = parse_grant(_read_no_follow(path, _GRANT_READ_LIMIT).decode("utf-8"))
    except FileNotFoundError:
        return "absent", None
    except Exception as exc:  # every failure on a present grant refuses
        logger.warning("The write authorization %s cannot be read: %s", path, exc)
        return "unreadable", None
    if grant.id != authorization:
        logger.warning("The write authorization %s names another id, %s", path, grant.id)
        return "unreadable", None
    return "ok", grant


def _read_ledger_at(place: _Place, authorization: str) -> tuple[_Status, Ledger | None]:
    """The ledger ``<id>.ledger.json`` under *place*; a missing directory is unreadable."""
    if _dir_status(place) != "ok":
        logger.warning("The write ledger directory %s is not a plain directory", place.path)
        return "unreadable", None
    path = place.path / f"{authorization}{_LEDGER_SUFFIX}"
    try:
        ledger = parse_ledger(_read_no_follow(path, _LEDGER_READ_LIMIT).decode("utf-8"))
    except FileNotFoundError:
        return "absent", None
    except Exception as exc:  # every failure on a present ledger refuses
        logger.warning("The write ledger %s cannot be read: %s", path, exc)
        return "unreadable", None
    if ledger.authorization != authorization:
        logger.warning("The write ledger %s names another authorization", path)
        return "unreadable", None
    return "ok", ledger


def _find_grant(owner: str, authorization: str) -> tuple[_Status, Grant | None]:
    """The grant for *authorization*: in the owner's directory, else the holder's.

    A grant found only in the holder's ledger directory was written by the
    holder, so it is honoured only as a self-issued grant: any other surface
    there is unreadable.
    """
    owner_place = _owner_place(owner)
    if owner_place is None:
        logger.warning("No authorizations directory is reachable for %r", owner)
        return "unreadable", None
    status, grant = _read_grant_at(owner_place, authorization)
    if status != "absent":
        return status, grant
    holder_place = _ledger_place()
    if holder_place is None or holder_place.path == owner_place.path:
        return "absent", None
    status, grant = _read_grant_at(holder_place, authorization)
    if grant is not None and grant.approver.surface != SELF_ISSUE_SURFACE:
        logger.warning(
            "The grant %s in the holder's own directory claims a %s approval; refused",
            authorization,
            grant.approver.surface,
        )
        return "unreadable", None
    return status, grant


def _governing_record(owner: str) -> tuple[bool, control_context.ControlContext | None]:
    """The owner's record as the switch verdict reads it: ``(readable, record)``.

    Under a bound tree, the owner's record through the tree's fail-closed
    reader; on a host, this host's own record. An absent record is readable
    and ``None``.
    """
    if posture_store._tree_is_bound():
        tree = posture_store._control_context_tree()
        if tree is None:
            return False, None
        read = posture_store._read_tree_context(tree, owner)
        return read.unavailable is None, read.record
    record, status = control_context.read_record_status()
    if status == "unreadable":
        return False, None
    return True, record if status == "ok" else None


# -- the holder's liveness ----------------------------------------------------


def _kernel_stat(pid: int) -> tuple[str, str] | None:
    """*pid*'s ``(state, start time)`` from ``/proc``, or ``None`` where it says nothing."""
    try:
        with open(_PROC_STAT.format(pid=pid), encoding="ascii", errors="replace") as stat_file:
            # The command name can hold anything, spaces and parentheses included.
            fields = stat_file.read().rpartition(")")[2].split()
        return fields[0], fields[19]
    except (OSError, IndexError):
        return None


def process_start_time(pid: int) -> str | None:
    """When *pid* started, in the kernel's own units, or ``None`` where nothing states it.

    Only ever compared with itself, which is what tells a holder from a process
    that reused its PID. ``None`` on a platform with no ``/proc``.
    """
    kernel = _kernel_stat(pid)
    return None if kernel is None else kernel[1]


def pid_namespace() -> str | None:
    """This process's PID namespace as the kernel names it, or ``None`` where nothing does.

    Only ever compared with another process's, which is what says whether a PID
    means the same thing to both. ``None`` on a platform with no ``/proc``.
    """
    try:
        return os.readlink(_PID_NS_LINK)
    except OSError:
        return None


def _holder_live(ledger: Ledger) -> bool:
    """Whether the kernel, asked from the holder's own namespace, still sees the holder.

    From another namespace, or with no PID recorded, a PID means nothing here
    and the answer is ``True``: the lease time decides alone. ``None`` matches
    ``None``, so two processes on a platform without ``/proc`` compare.
    """
    if ledger.holder_pid is None or ledger.pid_ns != pid_namespace():
        return True
    if not control_context.is_process_alive(ledger.holder_pid):
        return False
    kernel = _kernel_stat(ledger.holder_pid)
    if kernel is None:
        return True
    state, start = kernel
    return state != "Z" and (ledger.holder_start is None or start == ledger.holder_start)


def _lease_live(ledger: Ledger, now_ns: int) -> bool:
    return now_ns < ledger.lease_until_ns and _holder_live(ledger)


# -- the verdict --------------------------------------------------------------


#: The operations a gateway choice or a child's report is judged on: whether
#: the authorization would allow a write at all, every channel term skipped.
#: Never a write's own operations: :func:`spend_step` cannot be handed it.
_ANY_WRITE: Final = object()


@dataclass(frozen=True)
class _Basis:
    """What the switch verdict already read, for a judgement that follows it.

    ``seen`` is the record that verdict read (``seen.read`` true), so the grant's
    epoch and the switch are judged on one read of it; ``switch`` is that
    verdict, which stands in for a second read of the switch's position for a
    self-issued grant. Either ``None`` reads afresh.
    """

    seen: posture_store._RecordSeen | None = None
    switch: WriteVerdict | None = None


@dataclass(frozen=True)
class _Judgement:
    """One verdict and what it was reached on, for the writer that appends after it."""

    verdict: WriteVerdict
    ledger: Ledger | None = None
    #: A self-issued grant whose switch has left ``allowed``: the writer that
    #: holds the ledger's lock lapses the lease, so the grant ends for good.
    ends: bool = False


def _resolved_owner(owner: str | NoOwner | None) -> str | None:
    resolved = posture_store.current_owner(owner)
    if resolved is NO_OWNER or not isinstance(resolved, str) or not resolved.strip():
        return None
    return resolved.strip()


def _operations(operations: Any) -> list[tuple[Any, Any]] | None:
    """*operations* as a list of pairs, or ``None`` when it is not a batch of pairs."""
    if isinstance(operations, Mapping):
        return list(operations.items())
    if isinstance(operations, str | bytes) or not isinstance(operations, Iterable):
        return None
    pairs = []
    for entry in operations:
        if isinstance(entry, str | bytes) or not isinstance(entry, Sequence) or len(entry) != 2:
            return None
        pairs.append((entry[0], entry[1]))
    return pairs


def _judge(
    section: Any,
    target: Any,
    operations: Any,
    authorization: Any,
    owner: str | NoOwner | None,
    now_ns: int,
    basis: _Basis | None = None,
) -> _Judgement:
    """The verdict's terms in order (module rule 3). May raise; callers refuse on it.

    *operations* :data:`_ANY_WRITE` skips the channel terms (11): the answer is
    whether some write could go ahead now. *basis* carries what the switch
    verdict already read.
    """
    named = target if isinstance(target, str) else None
    kind = posture_store._verdict_kind(section, target)
    identifier = authorization if isinstance(authorization, str) else None

    def refuse(reason: str, detail: str | None = None) -> _Judgement:
        return _Judgement(
            WriteVerdict(
                allowed=False,
                reason=reason,
                target=named,
                kind=kind,
                detail=detail,
                authorization=identifier,
            )
        )

    if identifier is None or _ID.fullmatch(identifier) is None:
        return refuse("auth_unreadable", "The authorization this process names is not an id.")
    if posture_store._launch_pin_term(named) != "none":
        return refuse("launch_pinned")

    resolved = _resolved_owner(owner)
    if resolved is None:
        if posture_store._is_dispatch_run():
            return refuse("owner_unscheduled")
        return refuse("owner_unknown")
    if not posture_store._tree_is_bound() and resolved != acting_identity():
        return refuse("owner_foreign")

    if basis is not None and basis.seen is not None and basis.seen.read:
        readable, record = True, basis.seen.record
    else:
        readable, record = _governing_record(resolved)
    if not readable:
        return refuse("record_unavailable")

    status, grant = _find_grant(resolved, identifier)
    if status == "absent":
        return refuse("switch_off")
    if status != "ok" or grant is None:
        return refuse("auth_unreadable")

    if grant.owner != resolved or grant.holder != acting_identity():
        return refuse("auth_holder")
    if grant.target != named:
        return refuse("auth_channel", f"The authorization covers {grant.target}, not this machine.")
    if grant.revoked is not None:
        return refuse("auth_revoked", f"Stopped by {grant.revoked.by}.")
    if record is None or record.switch_epoch != grant.switch_epoch:
        return refuse("auth_revoked", "The switch record has moved past its approval.")
    if now_ns >= grant.expires_at_ns:
        return refuse("auth_expired")
    if grant.approver.surface == SELF_ISSUE_SURFACE:
        # The pin was judged first, so the switch verdict that honoured it and
        # one that ignores it agree here.
        switch = (
            basis.switch
            if basis is not None and basis.switch is not None
            else posture_store.write_verdict(section, named, owner=resolved, ignore_launch_pin=True)
        )
        if not switch.allowed or switch.position != "allowed":
            ended = refuse("auth_revoked", "The switch is no longer at allowed.")
            return replace(ended, ends=True)

    pairs = [] if operations is _ANY_WRITE else _operations(operations)
    if pairs is None or (not pairs and operations is not _ANY_WRITE):
        return refuse("auth_channel", "The write names no channel.")
    for address, value in pairs:
        bound = grant.channels.get(address) if isinstance(address, str) else None
        if bound is None:
            return refuse("auth_channel", f"Channel: {address}.")
        if not bound.holds(value):
            return refuse(
                "auth_range", f"Channel: {address}; allowed range [{bound.min}, {bound.max}]."
            )

    ledger_place = _ledger_place()
    if ledger_place is None:
        return refuse("auth_unreadable", "This process has no write ledger directory.")
    ledger_status, ledger = _read_ledger_at(ledger_place, identifier)
    if ledger_status == "absent":
        return refuse("auth_lease_lapsed", "The holder never started its lease.")
    if ledger_status != "ok" or ledger is None:
        return refuse("auth_unreadable", "The write ledger cannot be read.")
    if not _lease_live(ledger, now_ns):
        return refuse("auth_lease_lapsed")
    if len(ledger.steps) >= grant.steps:
        return refuse("auth_exhausted", f"{grant.steps} of {grant.steps} writes used.")

    return _Judgement(
        WriteVerdict(allowed=True, reason=None, target=named, kind=kind, authorization=identifier),
        ledger,
    )


def _unreachable(target: Any, authorization: Any, what: str) -> WriteVerdict:
    logger.warning("The write authorization verdict could not be reached; refusing (%s)", what)
    return WriteVerdict(
        allowed=False,
        reason="auth_unreadable",
        target=target if isinstance(target, str) else None,
        kind=None,
        authorization=authorization if isinstance(authorization, str) else None,
    )


def authorization_verdict(
    section: Any,
    target: str | None,
    operations: Sequence[tuple[str, object]] | Mapping[str, object],
    *,
    authorization: str,
    owner: str | NoOwner | None = None,
) -> WriteVerdict:
    """Whether one write call of *operations* may go ahead under *authorization*.

    The terms, in order; the first that refuses names the reason:

    1. ``auth_unreadable`` — *authorization* is not a well-formed id.
    2. ``launch_pinned`` — the launch pin is ``pinned`` or ``unreadable`` for
       *target*: a sandbox pinned at launch is not opened by a grant.
    3. ``owner_unscheduled`` / ``owner_unknown`` / ``owner_foreign`` — the owner
       ladder names nobody, or (with no tree bound) another login than this
       host's.
    4. ``record_unavailable`` — the owner's record cannot be read.
    5. ``switch_off`` — no grant has this id; ``auth_unreadable`` — one is
       there and cannot be read (module rule 2), or one in the holder's own
       directory that is not self-issued.
    6. ``auth_holder`` — the grant names another owner, or a holder other than
       :func:`~osprey_connectors.identity.acting_identity`.
    7. ``auth_channel`` — the grant covers another machine.
    8. ``auth_revoked`` — the grant is marked revoked, or the record's
       ``switch_epoch`` is no longer the grant's (or there is no record).
    9. ``auth_expired`` — its expiry has passed.
    10. ``auth_revoked`` — a self-issued grant whose owner's switch is not at
        ``allowed`` for *target* now (:func:`spend_step` then ends its lease).
    11. ``auth_channel`` — the batch is empty or names a channel the grant does
        not list; ``auth_range`` — a value that is not a finite number inside
        its channel's range. The whole batch is judged before anything is
        written.
    12. ``auth_unreadable`` — no ledger directory, or a ledger that cannot be
        read; ``auth_lease_lapsed`` — no ledger, a lease past its time, or a
        holder the kernel says is gone.
    13. ``auth_exhausted`` — every step has been spent.

    This reads and never writes, so it is the answer for a report or a
    gateway choice; a write spends its step through :func:`spend_step`. It
    does not repeat the switch verdict's read-only, tier and hold terms:
    :func:`authorized_execution_verdict` is the composition a write path takes.

    Args:
        section: The ``control_system`` section of the resolved config.
        target: The control target the write lands on.
        operations: The write call's ``(channel, value)`` pairs, or a mapping.
        authorization: The id the process names.
        owner: Whose switch the grant stands in for; ``None`` asks
            :func:`~osprey_connectors.posture_store.current_owner`.

    Returns:
        A :class:`~osprey_connectors.posture_store.WriteVerdict` whose
        ``authorization`` names the id and whose ``position`` is ``None``.
        Never raises: a verdict that cannot be reached refuses
        ``auth_unreadable``.
    """
    return _verdict(section, target, operations, authorization, owner, None)


def _verdict(
    section: Any,
    target: Any,
    operations: Any,
    authorization: Any,
    owner: str | NoOwner | None,
    basis: _Basis | None,
) -> WriteVerdict:
    """:func:`authorization_verdict` on *basis*; *operations* may be :data:`_ANY_WRITE`."""
    try:
        return _judge(section, target, operations, authorization, owner, _now_ns(), basis).verdict
    except Exception:  # a verdict that cannot be reached refuses; it never raises
        return _unreachable(target, authorization, "read")


# -- locks and writes ---------------------------------------------------------


def _open_lock(lock: Path) -> int:
    """Open *lock* for :func:`_locked`, creating it at 0o660 whatever the umask.

    A grants directory is group-shared (0o2770) and written under several uids
    — the host's CLI user, a web terminal's uid 1000 — so its lock must be
    openable by every one of them: whichever uid creates it first sets the
    mode, and a lock the others cannot open locks them out of approving,
    listing and revoking, a stop's revocation included. The create is the
    record lock's own (``O_EXCL`` at 0o660, then ``fchmod`` on the new file
    only), and an existing lock is reused as it stands, never followed through
    a link. One left at 0o640 by an earlier version, under another uid, is
    opened read-only: ``flock`` asks for no write access.

    Raises:
        OSError: A link sits at *lock*, or it could not be created or opened.
    """
    try:
        return control_context._open_lock_file(lock)
    except PermissionError:
        return os.open(lock, os.O_RDONLY | os.O_NOFOLLOW)


@contextlib.contextmanager
def _locked(lock: Path) -> Iterator[None]:
    """Hold an exclusive ``flock`` on *lock*, creating it, never following a link.

    Raises:
        OSError: A link sits at *lock*, or it could not be opened.
        RuntimeError: The lock was not acquired in time.
    """
    fd = _open_lock(lock)
    try:
        deadline = time.monotonic() + _LOCK_TIMEOUT_S
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"{lock} stayed locked for {_LOCK_TIMEOUT_S:g} s; nothing was written"
                    ) from None
                time.sleep(_LOCK_POLL_S)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_replace(path, data, mode=_FILE_MODE)


def _ledger_lock_path(place: _Place, authorization: str) -> Path:
    return place.path / f"{authorization}{_LEDGER_SUFFIX}{_LOCK_SUFFIX}"


def _checked_ledger_place(authorization: str, *, create: bool = False) -> _Place:
    """The ledger directory, which must exist as a plain directory.

    *create* is for the holder's own writers on a host, whose ledger directory
    is the host's authorizations directory: it is created there when absent. A
    ledger directory named by :data:`LEDGER_DIR_ENV_VAR` is a bind and is never
    created, and nothing on the write path creates one.

    Raises:
        ValueError: *authorization* is not an id.
        AuthorizationRefused: ``auth_unreadable`` — no usable ledger directory.
    """
    _authorization_id(authorization)
    place = _ledger_place()
    if create and place is not None and place.segments and _dir_status(place) == "absent":
        _prepare_directory(place.path)
    if place is None or _dir_status(place) != "ok":
        raise AuthorizationRefused(
            "auth_unreadable", "This process has no usable write ledger directory."
        )
    return place


def _lapse(place: _Place, authorization: str) -> None:
    """End the lease now, for good. The caller holds the ledger's lock."""
    status, ledger = _read_ledger_at(place, authorization)
    if status == "ok" and ledger is not None and ledger.lease_until_ns > 0:
        lapsed = replace(ledger, lease_until_ns=0)
        _write_json(place.path / f"{authorization}{_LEDGER_SUFFIX}", lapsed.to_payload())


def spend_step(
    section: Any,
    target: str | None,
    operations: Sequence[tuple[str, object]] | Mapping[str, object],
    *,
    authorization: str,
    owner: str | NoOwner | None = None,
) -> WriteVerdict:
    """Judge one write call and, when it is allowed, spend its step first.

    Under the ledger's lock: the whole :func:`authorization_verdict` against
    the ledger read inside the lock, then one step appended (``n`` the next
    number, the time, the batch's channels) and the ledger replaced whole. An
    allowed answer means the step is on disk; the caller writes only then, so
    a crash after this costs the step. Concurrent callers in any process are
    serialised by the lock, so no two of them spend the last step.

    A write the ledger could not record refuses ``auth_unreadable``. A
    self-issued grant refused because its switch has left ``allowed`` has its
    lease lapsed here as well, so the switch coming back does not revive it.

    Returns:
        As :func:`authorization_verdict`. Never raises.
    """
    return _spend(section, target, operations, authorization, owner, None)


def _spend(
    section: Any,
    target: Any,
    operations: Any,
    authorization: Any,
    owner: str | NoOwner | None,
    basis: _Basis | None,
) -> WriteVerdict:
    """:func:`spend_step` on *basis*. A write's own operations only, never :data:`_ANY_WRITE`."""
    if operations is _ANY_WRITE:
        return _unreachable(target, authorization, "spend without a write")
    try:
        place = _checked_ledger_place(authorization)
    except Exception:  # no ledger to spend from: the read-only verdict names why
        verdict = _verdict(section, target, operations, authorization, owner, basis)
        return _unreachable(target, authorization, "ledger") if verdict.allowed else verdict
    try:
        # Taken once: a one-shot iterable is judged and recorded as the same batch.
        batch = _operations(operations)
        with _locked(_ledger_lock_path(place, authorization)):
            now = _now_ns()
            judged = _judge(section, target, batch, authorization, owner, now, basis)
            if judged.ends:
                _lapse(place, authorization)
            if not judged.verdict.allowed or judged.ledger is None:
                return judged.verdict
            ledger = judged.ledger
            channels = tuple(str(address) for address, _value in batch or ())
            step = LedgerStep(n=len(ledger.steps) + 1, at_ns=now, channels=channels)
            _write_json(
                place.path / f"{authorization}{_LEDGER_SUFFIX}",
                replace(ledger, steps=(*ledger.steps, step)).to_payload(),
            )
            return judged.verdict
    except Exception:  # an unrecorded step is never an allowed write
        return _unreachable(target, authorization, "spend")


def authorized_execution_verdict(
    section: Any,
    target: str | None,
    operations: Sequence[tuple[str, object]] | Mapping[str, object],
    *,
    authorization: str | None,
    owner: str | NoOwner | None = None,
    session_keys: Sequence[str] | None = None,
    spend: bool = False,
) -> WriteVerdict:
    """The verdict on one write call about to execute, for a process that may hold a grant.

    :func:`~osprey_connectors.posture_store.execution_verdict` first. A process
    naming no *authorization* gets that verdict unchanged. One that names one
    gets it whenever it refuses for any reason but ``switch_off`` — an
    authorization lifts no tier, owner, record, pin or hold — and otherwise,
    whether the switch allowed or was off, the authorization's verdict in its
    place: a person turning the switch on mid-run does not lift the bounds.

    Args:
        section: The ``control_system`` section of the resolved config.
        target: The control target the write lands on.
        operations: The write call's ``(channel, value)`` pairs.
        authorization: The id the process names, or ``None``.
        owner: As for :func:`authorization_verdict`.
        session_keys: As for
            :func:`~osprey_connectors.posture_store.execution_verdict`.
        spend: Spend the step (:func:`spend_step`) rather than only judge.

    The record is read once: the authorization's terms are judged on the
    record the switch verdict read, and a self-issued grant's position on that
    verdict's own answer.

    Returns:
        A :class:`~osprey_connectors.posture_store.WriteVerdict`. Never raises.
    """
    if authorization is None:
        return posture_store.execution_verdict(
            section, target, owner=owner, session_keys=session_keys
        )
    seen = posture_store._RecordSeen()
    base = posture_store._with_hold(
        section, posture_store._write_verdict(section, target, owner, False, seen), session_keys
    )
    if not base.allowed and base.reason != "switch_off":
        return replace(base, authorization=authorization)
    basis = _Basis(seen=seen, switch=base)
    if spend:
        return _spend(section, target, operations, authorization, owner, basis)
    return _verdict(section, target, operations, authorization, owner, basis)


def authorized_write_verdict(
    section: Any,
    target: str | None,
    *,
    authorization: str | None,
    owner: str | NoOwner | None = None,
) -> WriteVerdict:
    """Whether a process may write to *target* at all: the gateway's and the report's answer.

    :func:`~osprey_connectors.posture_store.write_verdict` for a process naming
    no *authorization*. For one that names one, composed as
    :func:`authorized_execution_verdict` composes the execution verdict, with
    the authorization judged on every term but the channels and values: the
    grant in force, the lease live and a step left. It names no hold, as the
    switch verdict names none, because the gateway a connector selects must not
    follow a session's hold. It reads and never spends.

    Returns:
        A :class:`~osprey_connectors.posture_store.WriteVerdict`. Never raises.
    """
    if authorization is None:
        return posture_store.write_verdict(section, target, owner=owner)
    seen = posture_store._RecordSeen()
    base = posture_store._write_verdict(section, target, owner, False, seen)
    if not base.allowed and base.reason != "switch_off":
        return replace(base, authorization=authorization)
    return _verdict(
        section, target, _ANY_WRITE, authorization, owner, _Basis(seen=seen, switch=base)
    )


# -- the lease ----------------------------------------------------------------


def _live_grant(owner: str | NoOwner | None, authorization: str, now_ns: int) -> Grant:
    """The grant this process holds, still in force (not revoked, epoch, expiry).

    Raises:
        AuthorizationRefused: For the reason it is not.
    """
    resolved = _resolved_owner(owner)
    if resolved is None:
        raise AuthorizationRefused("owner_unknown")
    readable, record = _governing_record(resolved)
    if not readable:
        raise AuthorizationRefused("record_unavailable")
    status, grant = _find_grant(resolved, authorization)
    if status == "absent":
        raise AuthorizationRefused("switch_off", "No such authorization.")
    if status != "ok" or grant is None:
        raise AuthorizationRefused("auth_unreadable")
    if grant.owner != resolved or grant.holder != acting_identity():
        raise AuthorizationRefused("auth_holder")
    if grant.revoked is not None or record is None or record.switch_epoch != grant.switch_epoch:
        raise AuthorizationRefused("auth_revoked")
    if now_ns >= grant.expires_at_ns:
        raise AuthorizationRefused("auth_expired")
    return grant


def start_lease(authorization: str, *, owner: str | NoOwner | None = None) -> Ledger:
    """Open the ledger for *authorization*, with this process as the lease's holder.

    Called once, by the process that will renew the lease. It records this
    process's PID, start time and PID namespace and a lease of the grant's
    ``lease_s`` from now, with no step spent. A ledger is opened once: one that
    exists already — a lease started before, live or lapsed — refuses, so a
    holder that crashed does not come back under the same authorization.

    Raises:
        ValueError: *authorization* is not an id.
        AuthorizationRefused: The grant is absent, unreadable, not this
            holder's, revoked or expired (its reason), there is no usable
            ledger directory (``auth_unreadable``), or the ledger exists
            (``auth_lease_lapsed``).
        RuntimeError: The ledger stayed locked past the bounded wait.
        OSError: The write failed.
    """
    place = _checked_ledger_place(authorization, create=True)
    with _locked(_ledger_lock_path(place, authorization)):
        now = _now_ns()
        grant = _live_grant(owner, authorization, now)
        status, _ledger = _read_ledger_at(place, authorization)
        if status != "absent":
            raise AuthorizationRefused("auth_lease_lapsed", "A lease was already started.")
        pid = os.getpid()
        ledger = Ledger(
            authorization=authorization,
            holder_pid=pid,
            holder_start=process_start_time(pid),
            pid_ns=pid_namespace(),
            lease_until_ns=now + int(grant.lease_s * 1e9),
        )
        _write_json(place.path / f"{authorization}{_LEDGER_SUFFIX}", ledger.to_payload())
    return ledger


def renew_lease(authorization: str, *, owner: str | NoOwner | None = None) -> bool:
    """Move the lease on by the grant's ``lease_s``, if this process holds it and it is live.

    A compare-and-swap under the ledger's lock. ``False`` writes nothing: the
    ledger is absent or unreadable, another process holds the lease, the lease
    has lapsed (a lapsed lease never comes back), or the grant is no longer in
    force. Never raises.
    """
    try:
        place = _checked_ledger_place(authorization)
        with _locked(_ledger_lock_path(place, authorization)):
            now = _now_ns()
            status, ledger = _read_ledger_at(place, authorization)
            if status != "ok" or ledger is None:
                return False
            if ledger.holder_pid != os.getpid() or ledger.pid_ns != pid_namespace():
                return False
            if not _lease_live(ledger, now):
                return False
            grant = _live_grant(owner, authorization, now)
            renewed = replace(ledger, lease_until_ns=now + int(grant.lease_s * 1e9))
            _write_json(place.path / f"{authorization}{_LEDGER_SUFFIX}", renewed.to_payload())
            return True
    except Exception:  # a renewal that cannot be made is a lease left to lapse
        logger.warning("The lease of %r was not renewed", authorization, exc_info=True)
        return False


# -- issuing ------------------------------------------------------------------


def _grant_directory(directory: Path | None, owner: str) -> Path:
    if directory is not None:
        return Path(directory)
    path = authorizations_dir(owner)
    if path is None:
        raise RuntimeError(f"No authorizations directory is reachable for {owner!r}")
    return path


def _prepare_directory(directory: Path) -> None:
    """Create *directory* (0o2770, as the record's) when absent; refuse a link or a file."""
    if not os.path.lexists(directory):
        control_context._ensure_directory(directory)
    mode = os.lstat(directory).st_mode
    if not stat.S_ISDIR(mode):
        raise OSError(errno.ENOTDIR, f"{directory} is not a plain directory; nothing was written")


def list_grants(directory: Path) -> tuple[tuple[Grant, ...], tuple[Path, ...]]:
    """Every grant file in *directory*: the grants read, and the paths that could not be.

    A file is a grant file by its name, ``<id>.json``; ledgers and locks beside
    them are not. A missing directory holds none.

    Raises:
        OSError: The directory is a link or cannot be listed.
    """
    directory = Path(directory)
    try:
        mode = os.lstat(directory).st_mode
    except FileNotFoundError:
        return (), ()
    if not stat.S_ISDIR(mode):
        raise OSError(errno.ENOTDIR, f"{directory} is not a plain directory")
    grants: list[Grant] = []
    unreadable: list[Path] = []
    for name in sorted(os.listdir(directory)):
        match = _GRANT_FILE.fullmatch(name)
        if match is None:
            continue
        status, grant = _read_grant_at(_Place(directory), match.group(1))
        if status == "ok" and grant is not None:
            grants.append(grant)
        elif status == "unreadable":
            unreadable.append(directory / name)
    return tuple(grants), tuple(unreadable)


def _lease_ended(grant: Grant, ledger_dirs: Iterable[Path], now_ns: int) -> bool:
    """Whether every ledger of *grant* in *ledger_dirs* shows its lease time passed.

    The clock alone decides, as :func:`grant_standing` does: the holder's PID,
    judged from the approver's process, may name another host's process. At
    least one ledger must be found, and each must agree, so a stale or planted
    one in a directory searched first cannot free another's live grant. A
    ledger found nowhere is a lease not started yet, or one out of this
    process's sight, and an unreadable one may be live: neither has ended.
    """
    found = list(_ledgers_of(grant.id, ledger_dirs))
    return bool(found) and all(
        ledger is not None and now_ns >= ledger.lease_until_ns for ledger in found
    )


def auth_overlap(
    candidate: Grant,
    existing: Iterable[Grant],
    *,
    now_ns: int | None = None,
    ledger_dirs: Iterable[Path] = (),
) -> Grant | None:
    """The first grant in *existing* that *candidate* must not be issued beside, or ``None``.

    One that is live (not revoked, not expired), held by another holder, on
    the same machine, listing a channel *candidate* lists, and whose lease
    time has not passed in its ledgers in *ledger_dirs* (:func:`_lease_ended`):
    a run that finished, failed or crashed stops renewing, so its grant stops
    blocking once its lease runs out rather than at its expiry. A run that is
    paused still renews, and still blocks. The same holder may hold two: a
    program runs one job at a time and is judged by its own budget. A
    person's own writes are never blocked by a grant.
    """
    now = _now_ns() if now_ns is None else now_ns
    directories = tuple(ledger_dirs)
    for grant in existing:
        if grant.id == candidate.id or not grant.live_at(now):
            continue
        if grant.holder == candidate.holder or grant.target != candidate.target:
            continue
        if set(grant.channels) & set(candidate.channels) and not _lease_ended(
            grant, directories, now
        ):
            return grant
    return None


def _peer_grants(peers: Iterable[Path], own: Path) -> list[Grant]:
    """The self-issued grants in the ledger directories *peers*, *own* left out.

    Only a grant approved by the switch (:func:`self_issued_grants`) is
    counted, because only that one is honoured there. A directory that cannot
    be listed and a grant file that cannot be read are logged and skipped: the
    check is the approver's courtesy, not the boundary, and a file a holder
    can write must not veto every approval. A peer that is absent holds none.
    """
    grants: list[Grant] = []
    seen = {os.path.normpath(own)}
    for peer in peers:
        key = os.path.normpath(peer)
        if key in seen:
            continue
        seen.add(key)
        try:
            found, unreadable = self_issued_grants(Path(peer))
        except OSError as exc:
            logger.warning("The self-issued grants in %s cannot be listed: %s", peer, exc)
            continue
        if unreadable:
            logger.warning(
                "Unreadable grants in %s, left out of the overlap check: %s", peer, unreadable
            )
        grants.extend(found)
    return grants


def issue_grant(grant: Grant, *, directory: Path | None = None, peers: Iterable[Path] = ()) -> Path:
    """Write *grant* as a new file, after checking it overlaps no live grant.

    The approver's writer. *directory* defaults to the grant's owner's
    authorizations directory as this process reaches it. The check and the
    write are one hold of the directory's lock, so two approvals of
    overlapping grants in one directory cannot both pass. A grant file that
    cannot be read in the directory is logged and left out of the check.

    *peers* are the ledger directories holding the holders' self-issued
    grants (:func:`self_issued_directories`), read under the same hold
    (:func:`_peer_grants`); their locks are not taken.

    Returns:
        The path written.

    Raises:
        ValueError: *grant* is not a grant this version can read back.
        FileExistsError: A grant with this id exists.
        AuthorizationRefused: ``auth_overlap``.
        RuntimeError: No directory is reachable, or the lock was not acquired.
        OSError: The directory is not a plain directory, or the write failed.
    """
    payload = grant.to_payload()
    parse_grant(json.dumps(payload))  # what is written must read back
    target_dir = _grant_directory(directory, grant.owner)
    _prepare_directory(target_dir)
    path = target_dir / f"{grant.id}{_GRANT_SUFFIX}"
    with _locked(target_dir / _DIRECTORY_LOCK_NAME):
        if os.path.lexists(path):
            raise FileExistsError(errno.EEXIST, f"{path} exists; nothing was written")
        existing, unreadable = list_grants(target_dir)
        if unreadable:
            logger.warning(
                "Unreadable grants beside %s, left out of the overlap check: %s", path, unreadable
            )
        peers = tuple(peers)
        held = auth_overlap(
            grant,
            [*existing, *_peer_grants(peers, target_dir)],
            ledger_dirs=(target_dir, *peers),
        )
        if held is not None:
            raise AuthorizationRefused(
                "auth_overlap", f"Held by {held.holder} under {held.id} ({held.label})."
            )
        _write_json(path, payload)
    return path


def revoke_grants(
    *,
    by: str,
    targets: Sequence[str] | None = None,
    directory: Path | None = None,
    ids: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Mark every unrevoked grant in *directory* on *targets* revoked; the ids marked.

    The stop's half in the grants directory: *targets* ``None`` is every
    machine. *ids*, when given, narrows the mark to those grants: a person
    revoking one authorization from a list. The mark is ``{"at_ns", "by"}`` and is never removed, so the next
    write under any of these refuses ``auth_revoked``. A grant file that cannot
    be read is left as it is — it already refuses every write. *directory*
    defaults to this process's own authorizations directory.

    Raises:
        RuntimeError: No directory is reachable, or the lock was not acquired.
        OSError: The directory is not a plain directory, or a write failed.
    """
    target_dir = _grant_directory(directory, acting_identity())
    if not os.path.lexists(target_dir):
        return ()
    _prepare_directory(target_dir)
    marked: list[str] = []
    wanted = None if ids is None else frozenset(ids)
    with _locked(target_dir / _DIRECTORY_LOCK_NAME):
        now = _now_ns()
        existing, _unreadable = list_grants(target_dir)
        for grant in existing:
            if grant.revoked is not None:
                continue
            if targets is not None and grant.target not in targets:
                continue
            if wanted is not None and grant.id not in wanted:
                continue
            stopped = replace(grant, revoked=Revocation(at_ns=now, by=by))
            _write_json(target_dir / f"{grant.id}{_GRANT_SUFFIX}", stopped.to_payload())
            marked.append(grant.id)
    return tuple(marked)


def _channel_ranges(channels: Mapping[str, Any]) -> dict[str, ChannelRange]:
    """*channels* as ranges: each value a :class:`ChannelRange` or a ``(min, max)`` pair."""
    ranges: dict[str, ChannelRange] = {}
    for address, bound in channels.items():
        if isinstance(bound, ChannelRange):
            ranges[address] = bound
        else:
            low, high = bound
            ranges[address] = ChannelRange(min=float(low), max=float(high))
    return ranges


def self_issue(
    section: Any,
    target: str,
    channels: Mapping[str, ChannelRange | tuple[float, float]],
    *,
    steps: int,
    expires_at_ns: int,
    lease_s: float,
    label: str = "",
    owner: str | NoOwner | None = None,
) -> Grant:
    """Write this holder's own grant while its owner's switch stands at ``allowed``.

    The switch verdict for *target* must allow at ``allowed`` (launch pin
    included), and the owner's record must carry an epoch. The grant is written
    into this process's ledger directory with approver surface
    ``write_switch``, then the verdict is taken again: if the switch moved or a
    stop came in between, the grant is removed and the issue refused. It is
    honoured afterwards only while the position stays ``allowed``; the first
    write after it leaves ``allowed`` ends it for good (:func:`spend_step`).

    Raises:
        ValueError: The channels, steps, expiry or lease are malformed.
        AuthorizationRefused: The switch verdict's reason; ``auth_revoked``
            when it allows below ``allowed`` or moved during the issue;
            ``record_unavailable`` for a record with no epoch;
            ``auth_unreadable`` for no usable ledger directory.
        RuntimeError: The lock was not acquired in time.
        OSError: The write failed.
    """
    resolved = _resolved_owner(owner)
    if resolved is None:
        raise AuthorizationRefused("owner_unknown")

    def epoch_at_allowed() -> int:
        switch = posture_store.write_verdict(section, target, owner=resolved)
        if not switch.allowed:
            raise AuthorizationRefused(switch.reason or "switch_off")
        if switch.position != "allowed":
            raise AuthorizationRefused("auth_revoked", "The switch is not at allowed.")
        readable, record = _governing_record(resolved)
        if not readable:
            raise AuthorizationRefused("record_unavailable")
        if record is None or record.switch_epoch is None:
            raise AuthorizationRefused("record_unavailable", "The switch has no epoch.")
        return record.switch_epoch

    epoch = epoch_at_allowed()
    now = _now_ns()
    holder = acting_identity()
    grant = Grant(
        id=uuid.uuid4().hex,
        owner=resolved,
        holder=holder,
        target=target,
        switch_epoch=epoch,
        channels=_channel_ranges(channels),
        steps=steps,
        issued_at_ns=now,
        expires_at_ns=expires_at_ns,
        lease_s=lease_s,
        approver=Approver(identity=holder, surface=SELF_ISSUE_SURFACE, at_ns=now),
        label=label,
    )
    place = _checked_ledger_place(grant.id, create=True)
    path = issue_grant(grant, directory=place.path, peers=self_issued_directories())
    try:
        if epoch_at_allowed() != epoch:
            raise AuthorizationRefused("auth_revoked", "A stop came during the issue.")
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink()
        raise
    return grant


# -- where the self-issued grants are: overlap and listing ----------------------

#: The environment variable a web terminal carries: a read-only bind of the
#: holders' ledger directory, where their self-issued grants are, so an
#: approval there checks overlap against them and the chip lists them. Never
#: :data:`LEDGER_DIR_ENV_VAR`: the terminal is not a holder and writes nothing
#: there.
LEDGER_VIEW_DIR_ENV_VAR: Final = "OSPREY_WRITE_LEDGER_VIEW_DIR"


def self_issued_directories() -> tuple[Path, ...]:
    """Where self-issued grants are, as this process sees them.

    This process's own ledger directory (:func:`ledger_dir`), and the holders'
    ledger directory a web terminal carries read-only
    (:data:`LEDGER_VIEW_DIR_ENV_VAR`); either may be absent.
    """
    found: list[Path] = []
    ledger = _ledger_place()
    if ledger is not None:
        found.append(ledger.path)
    view = posture_store._absolute_bind(LEDGER_VIEW_DIR_ENV_VAR)
    if view is not None and all(os.path.normpath(view) != os.path.normpath(p) for p in found):
        found.append(view)
    return tuple(found)


def self_issued_grants(directory: Path) -> tuple[tuple[Grant, ...], tuple[Path, ...]]:
    """The self-issued grants in a ledger *directory*, and the paths that could not be read.

    :func:`list_grants`, keeping only the grants approved by the switch
    (:data:`SELF_ISSUE_SURFACE`): a grant there claiming any other surface is
    never honoured, because the holder can write there, so it is not listed.

    Raises:
        OSError: The directory is a link or cannot be listed.
    """
    grants, unreadable = list_grants(directory)
    return (
        tuple(grant for grant in grants if grant.approver.surface == SELF_ISSUE_SURFACE),
        unreadable,
    )


def grant_in_force(
    grant: Grant, record: control_context.ControlContext | None, *, now_ns: int | None = None
) -> bool:
    """Whether *grant* still stands against its owner's *record*: what a chip shows as live.

    Not revoked, not expired, and issued under the record's epoch. The lease
    and the steps are the holder's and are not asked. A self-issued grant also
    needs the record's switch for its machine at ``allowed``: once it has left,
    the holder's next write ends it for good, so it is not shown standing.
    """
    now = _now_ns() if now_ns is None else now_ns
    if not (
        grant.live_at(now) and record is not None and record.switch_epoch == grant.switch_epoch
    ):
        return False
    if grant.approver.surface == SELF_ISSUE_SURFACE:
        return control_context.switch_position(record, grant.target) == "allowed"
    return True


#: How long a revoked or expired grant is kept before a listing removes it, so
#: a holder whose stop has just landed is still told ``auth_revoked`` (or
#: ``auth_expired``) rather than that no authorization exists.
PRUNE_AFTER_S: Final = 3600.0


def _ended_at(grant: Grant) -> int:
    """When *grant* stopped standing for good: its stop or its expiry, the earlier."""
    if grant.revoked is None:
        return grant.expires_at_ns
    return min(grant.revoked.at_ns, grant.expires_at_ns)


def prune_grants(
    directory: Path, *, now_ns: int | None = None, after_s: float = PRUNE_AFTER_S
) -> tuple[str, ...]:
    """Remove the grants in *directory* that ended more than *after_s* ago; the ids removed.

    A grant has ended once it is revoked or expired; a grant still in force,
    and one that cannot be read, are left alone (an unreadable file refuses
    every write already, and its owner should see it). Under the directory's
    lock, so an issue in flight is not overtaken. A missing directory holds
    none. Never removes a ledger, which is the holder's.

    Raises:
        RuntimeError: The lock was not acquired in time.
        OSError: The directory is not a plain directory, or a removal failed.
    """
    directory = Path(directory)
    if not os.path.lexists(directory):
        return ()
    _prepare_directory(directory)
    now = _now_ns() if now_ns is None else now_ns
    cutoff = now - int(after_s * 1e9)
    removed: list[str] = []
    with _locked(directory / _DIRECTORY_LOCK_NAME):
        existing, _unreadable = list_grants(directory)
        for grant in existing:
            if grant.revoked is None and now < grant.expires_at_ns:
                continue
            if _ended_at(grant) <= cutoff:
                with contextlib.suppress(FileNotFoundError):
                    (directory / f"{grant.id}{_GRANT_SUFFIX}").unlink()
                removed.append(grant.id)
    return tuple(removed)


def _ledgers_of(authorization: str, ledger_dirs: Iterable[Path]) -> Iterator[Ledger | None]:
    """Each ledger of *authorization* in *ledger_dirs*, in order; ``None`` for an unreadable one.

    A directory that is not there, and one without the ledger, are skipped.
    """
    for directory in ledger_dirs:
        if not os.path.isdir(directory):
            continue
        status, ledger = _read_ledger_at(_Place(Path(directory)), authorization)
        if status != "absent":
            yield ledger


def read_ledger(directory: Path, authorization: str) -> Ledger | None:
    """The ledger of *authorization* in *directory*, or ``None`` when absent.

    For a surface that shows a holder's consumption (``osprey status``); the
    same fail-closed reader the verdict uses.

    Raises:
        ValueError: The ledger is there and cannot be read.
    """
    status, ledger = _read_ledger_at(_Place(Path(directory)), _authorization_id(authorization))
    if status == "unreadable":
        raise ValueError(f"The write ledger of {authorization} cannot be read")
    return ledger


@dataclass(frozen=True)
class GrantStanding:
    """One grant as a listing shows it: whether it stands, and what its holder spent.

    Attributes:
        grant: The grant.
        in_force: :func:`grant_in_force`, and the holder's ledger has not
            ended it: a grant whose lease lapsed or whose every step is spent
            refuses the next write, so it is not shown standing.
        used: The steps the ledger counts; ``0`` while there is no ledger yet,
            ``None`` when it cannot be read.
        ledger: The ledger read, or ``None``.
    """

    grant: Grant
    in_force: bool
    used: int | None
    ledger: Ledger | None


def grant_standing(
    grant: Grant,
    record: control_context.ControlContext | None,
    ledger_dirs: Iterable[Path],
    *,
    now_ns: int | None = None,
) -> GrantStanding:
    """Whether *grant* still stands for a listing, its holder's ledger asked.

    The ledger is the first one found in *ledger_dirs* (the holder's ledger
    directory as this process sees it); a directory that is not there is
    skipped. Every surface that lists grants — the chip, its listing, the CLI
    and ``osprey status`` — answers through this, so they agree.
    """
    now = _now_ns() if now_ns is None else now_ns
    in_force = grant_in_force(grant, record, now_ns=now)
    for ledger in _ledgers_of(grant.id, ledger_dirs):
        if ledger is None:
            return GrantStanding(grant, in_force, None, None)
        used = len(ledger.steps)
        ended = now >= ledger.lease_until_ns or used >= grant.steps
        return GrantStanding(grant, in_force and not ended, used, ledger)
    return GrantStanding(grant, in_force, 0, None)


def listed_grants(
    owner_dirs: Iterable[Path], ledger_dirs: Iterable[Path]
) -> tuple[list[tuple[Grant, Path]], list[Path]]:
    """Every grant in *owner_dirs* and every self-issued one in *ledger_dirs*, each once.

    Each with the directory it was found in, and the paths that could not be
    read: a grant file, or a directory that cannot be listed. Nothing is
    written and nothing pruned.
    """
    grants: list[tuple[Grant, Path]] = []
    unreadable: list[Path] = []
    seen: set[str] = set()
    listed: set[str] = set()
    sources = [(Path(d), False) for d in owner_dirs] + [(Path(d), True) for d in ledger_dirs]
    for directory, self_issued in sources:
        if os.path.normpath(directory) in listed:
            continue
        listed.add(os.path.normpath(directory))
        try:
            found, broken = (self_issued_grants if self_issued else list_grants)(directory)
        except OSError:
            logger.warning("Could not list the authorizations in %s", directory, exc_info=True)
            unreadable.append(directory)
            continue
        unreadable.extend(broken)
        for grant in found:
            if grant.id not in seen:
                seen.add(grant.id)
                grants.append((grant, directory))
    return grants, unreadable


def approved_grant(authorization: str, *, owner: str | NoOwner | None = None) -> Grant | None:
    """The grant issued for *authorization*, still in force, or ``None`` while there is none.

    The holder's wait after :func:`write_request`: an approval issues the grant
    under the request's own id, so the holder polls this with the id it asked
    under, then opens its lease (:func:`start_lease`) and names the id
    (:data:`AUTHORIZATION_ENV_VAR`).

    Raises:
        ValueError: *authorization* is not an id.
        AuthorizationRefused: The grant is there and does not stand — unreadable,
            another holder's, revoked or expired — or the owner's record
            cannot be read.
    """
    _authorization_id(authorization)
    try:
        return _live_grant(owner, authorization, _now_ns())
    except AuthorizationRefused as refused:
        if refused.reason == "switch_off":
            return None
        raise


# -- requests -------------------------------------------------------------------

#: The environment variable naming the directory a web terminal reads holders'
#: requests from: a read-only bind of ``<ledger dir>/requests``.
REQUESTS_DIR_ENV_VAR: Final = "OSPREY_WRITE_REQUESTS_DIR"

#: The directory under a holder's ledger directory its requests are written to.
REQUESTS_DIR_NAME: Final = "requests"

#: What a run does once its evaluations are done, each a write the budget pays
#: for: nothing, the best point found, or the values it started from.
COMPLETIONS: Final = ("none", "apply_best", "restore")

#: A request is a few hundred bytes plus one line per channel.
_REQUEST_READ_LIMIT: Final = 1024 * 1024

_REQUEST_KEYS: Final = frozenset(
    {
        "schema",
        "id",
        "owner",
        "holder",
        "target",
        "channels",
        "steps",
        "requested_at_ns",
        "expires_at_ns",
        "lease_s",
        "label",
        "completion",
    }
)


@dataclass(frozen=True)
class WriteRequest:
    """A holder asking for an authorization: what a person approves. Treat as immutable.

    The holder writes it into ``<ledger dir>/requests/<id>.json``; an approval
    issues the grant under the same id, so the holder knows which
    authorization to wait for and name.

    Attributes:
        id: ``uuid4().hex``; the grant approved from it has the same id.
        owner: The login whose switch the run stands in for, who approves.
        holder: The acting identity that will write.
        target: The control target.
        channels: ``{address: ChannelRange}``, each address spelled exactly as
            the holder will write it.
        steps: The write calls the run will make, its completion's included.
        requested_at_ns: When the request was written.
        expires_at_ns: When the authorization is to end whatever happens.
        lease_s: How long one renewal of the holder's lease lasts.
        label: The run, free text.
        completion: Which completion the run will perform (:data:`COMPLETIONS`).
    """

    id: str
    owner: str
    holder: str
    target: str
    channels: Mapping[str, ChannelRange]
    steps: int
    requested_at_ns: int
    expires_at_ns: int
    lease_s: float
    label: str = ""
    completion: Literal["none", "apply_best", "restore"] = "none"

    def to_payload(self) -> dict[str, Any]:
        """This request as the JSON object its file holds."""
        return {
            "schema": SCHEMA_VERSION,
            "id": self.id,
            "owner": self.owner,
            "holder": self.holder,
            "target": self.target,
            "channels": {
                address: {"min": bound.min, "max": bound.max}
                for address, bound in self.channels.items()
            },
            "steps": self.steps,
            "requested_at_ns": self.requested_at_ns,
            "expires_at_ns": self.expires_at_ns,
            "lease_s": self.lease_s,
            "label": self.label,
            "completion": self.completion,
        }

    def grant(self, *, switch_epoch: int, approver: Approver, issued_at_ns: int) -> Grant:
        """The grant approving this request exactly as asked, under its id."""
        return Grant(
            id=self.id,
            owner=self.owner,
            holder=self.holder,
            target=self.target,
            switch_epoch=switch_epoch,
            channels=dict(self.channels),
            steps=self.steps,
            issued_at_ns=issued_at_ns,
            expires_at_ns=self.expires_at_ns,
            lease_s=self.lease_s,
            approver=approver,
            label=self.label,
        )


def parse_request(raw: Any) -> WriteRequest:
    """A request out of its JSON text, bytes or decoded object.

    The grant's strictness (:func:`parse_grant`): every field present with its
    type and nothing else, no repeated key, every bound a finite number with
    ``min <= max``, ``steps`` at least one, an expiry after the request, a
    positive lease, a known completion, and the grant's ceilings.

    Raises:
        ValueError: The payload is not a request this version can read.
    """
    payload = _load_json(raw) if isinstance(raw, str | bytes | bytearray) else raw
    payload = _exact_keys(payload, _REQUEST_KEYS, "the request")
    if payload["schema"] is not SCHEMA_VERSION:
        raise ValueError(f"the request's schema is not {SCHEMA_VERSION}")
    requested = _integer(payload["requested_at_ns"], "requested_at_ns", 0)
    expires = _integer(payload["expires_at_ns"], "expires_at_ns", 0)
    if expires <= requested:
        raise ValueError("the request expires before it was made")
    lease_s = _finite(payload["lease_s"], "lease_s")
    if lease_s <= 0:
        raise ValueError("lease_s is not positive")
    steps = _integer(payload["steps"], "steps", 1)
    _within_ceilings(steps, requested, expires, lease_s)
    label = payload["label"]
    if not isinstance(label, str):
        raise ValueError("label is not a string")
    completion = payload["completion"]
    if completion not in COMPLETIONS:
        raise ValueError(f"completion {completion!r} is not one of {COMPLETIONS}")
    return WriteRequest(
        id=_authorization_id(payload["id"]),
        owner=_name(payload["owner"], "owner"),
        holder=_name(payload["holder"], "holder"),
        target=_name(payload["target"], "target"),
        channels=_parse_channels(payload["channels"]),
        steps=steps,
        requested_at_ns=requested,
        expires_at_ns=expires,
        lease_s=lease_s,
        label=label,
        completion=completion,
    )


def requests_dir() -> Path | None:
    """Where this process reads requests: :data:`REQUESTS_DIR_ENV_VAR`, else its ledger's.

    A web terminal is handed the holders' request directory read-only; a
    holder, or a host with no bind, reads ``<ledger dir>/requests``. ``None``
    when neither is reachable.
    """
    bound = posture_store._absolute_bind(REQUESTS_DIR_ENV_VAR)
    if bound is not None:
        return bound
    ledger = ledger_dir()
    return None if ledger is None else ledger / REQUESTS_DIR_NAME


def read_request(path: Path) -> WriteRequest:
    """The request at *path*, read the fail-closed way.

    The file is opened ``O_NOFOLLOW``, must be a regular file under the read
    limit, must parse, and must be named for its own id (``<id>.json``).

    Raises:
        FileNotFoundError: Nothing is at *path*.
        OSError: A link, not a regular file, too long, or unreadable.
        ValueError: Not a request this version can read, or named for
            another id.
    """
    path = Path(path)
    request = parse_request(_read_no_follow(path, _REQUEST_READ_LIMIT).decode("utf-8"))
    if path.name != f"{request.id}{_GRANT_SUFFIX}":
        raise ValueError(f"{path.name} does not name the request it holds, {request.id}")
    return request


def list_requests(directory: Path) -> tuple[tuple[WriteRequest, ...], tuple[Path, ...]]:
    """Every request file in *directory*: the requests read, and the paths that could not be.

    A missing directory holds none.

    Raises:
        OSError: The directory is a link, not a directory, or cannot be listed.
    """
    directory = Path(directory)
    try:
        mode = os.lstat(directory).st_mode
    except FileNotFoundError:
        return (), ()
    if not stat.S_ISDIR(mode):
        raise OSError(errno.ENOTDIR, f"{directory} is not a plain directory")
    found: list[WriteRequest] = []
    unreadable: list[Path] = []
    for name in sorted(os.listdir(directory)):
        if _GRANT_FILE.fullmatch(name) is None:
            continue
        try:
            found.append(read_request(directory / name))
        except FileNotFoundError:
            continue
        except Exception as exc:  # every failure on a present request is reported
            logger.warning("The write request %s cannot be read: %s", directory / name, exc)
            unreadable.append(directory / name)
    return tuple(found), tuple(unreadable)


def write_request(
    target: str,
    channels: Mapping[str, ChannelRange | tuple[float, float]],
    *,
    steps: int,
    expires_at_ns: int,
    lease_s: float,
    label: str = "",
    completion: Literal["none", "apply_best", "restore"] = "none",
    owner: str | NoOwner | None = None,
) -> WriteRequest:
    """Ask for an authorization: write a request into this holder's ledger directory.

    The holder is this process's acting identity, the owner the login the
    owner ladder names (:func:`~osprey_connectors.posture_store.current_owner`).
    The file is ``<ledger dir>/requests/<id>.json``; the directory is created
    there when absent. Nothing is approved by writing it: a person approves it
    from the chip or the CLI, which issues the grant under the same id
    (:func:`approved_grant` waits for it), or the holder self-issues at
    ``allowed`` (:func:`self_issue`).

    Raises:
        ValueError: The channels, steps, expiry, lease or completion are
            malformed.
        AuthorizationRefused: ``owner_unknown``; ``auth_unreadable`` for no
            usable ledger directory.
        OSError: The write failed.
    """
    resolved = _resolved_owner(owner)
    if resolved is None:
        raise AuthorizationRefused("owner_unknown")
    request = WriteRequest(
        id=uuid.uuid4().hex,
        owner=resolved,
        holder=acting_identity(),
        target=target,
        channels=_channel_ranges(channels),
        steps=steps,
        requested_at_ns=_now_ns(),
        expires_at_ns=expires_at_ns,
        lease_s=lease_s,
        label=label,
        completion=completion,
    )
    payload = request.to_payload()
    parse_request(json.dumps(payload))  # what is written must read back
    place = _checked_ledger_place(request.id, create=True)
    directory = place.path / REQUESTS_DIR_NAME
    _prepare_directory(directory)
    _write_json(directory / f"{request.id}{_GRANT_SUFFIX}", payload)
    return request


def withdraw_request(request_id: str) -> bool:
    """Remove this holder's request *request_id*; whether a file was removed.

    For the holder once its grant is issued, or when its run ends without one.
    The grant, if any, is untouched.

    Raises:
        ValueError: *request_id* is not an id.
        AuthorizationRefused: ``auth_unreadable`` for no usable ledger directory.
    """
    place = _checked_ledger_place(request_id)
    try:
        (place.path / REQUESTS_DIR_NAME / f"{request_id}{_GRANT_SUFFIX}").unlink()
    except FileNotFoundError:
        return False
    return True


# -- approval -------------------------------------------------------------------

#: The surfaces a person approves from.
APPROVAL_SURFACES: Final = ("chip", "cli")


def approve_request(
    request: WriteRequest,
    *,
    surface: Literal["chip", "cli"],
    record_file: Path | None = None,
    peers: Iterable[Path] | None = None,
    approver: str | None = None,
) -> Grant:
    """Issue the grant for *request*, as the person acting as its owner approves it.

    The approver's one writer, for the chip and the CLI. *record_file* is the
    owner's control-context record — on a host, :func:`~osprey_connectors.control_context.record_path`
    when the owner is this process's identity; under a deployment, the
    owner's file in the tree. The grant is written into the ``authorizations``
    directory beside it.

    Inside one hold of the record lock (the lock order every writer of both
    keeps: the record, then the grants directory): the record is read — one
    that cannot be read refuses, one that is not there is created on the
    request's machine with every switch off, and one with no epoch is given
    one — the grant is stamped with that epoch, and :func:`issue_grant` checks
    it against every grant in the directory and the self-issued ones in
    *peers* (default: :func:`self_issued_directories`) before writing. A stop of every machine
    moving the epoch after this ends the grant; a stop of this machine that
    lands after it marks it revoked.

    The caller checks what this module cannot: that the acting person is the
    owner, the owner's tier, and the limits database.

    Args:
        request: What the holder asked for; the grant is exactly that.
        surface: ``chip`` or ``cli``.
        record_file: The owner's record; ``None`` is this host's own, which
            only the owner's own process may use.
        peers: The ledger directories whose self-issued grants are checked
            for overlap.
        approver: Who approves; ``None`` is this process's acting identity.

    Returns:
        The grant written.

    Raises:
        ValueError: *surface* is not an approval surface.
        AuthorizationRefused: ``record_unavailable`` for an unreadable record;
            ``auth_expired`` for a request whose expiry has passed;
            ``auth_overlap`` from the issue; ``owner_foreign``
            for no *record_file* and an owner who is not this process.
        FileExistsError: The request is approved already.
        RuntimeError: A lock was not acquired in time, or no record path resolves.
        OSError: A write failed.
    """
    if surface not in APPROVAL_SURFACES:
        raise ValueError(f"surface {surface!r} is not one of {APPROVAL_SURFACES}")
    identity = approver or acting_identity()
    if record_file is None:
        if request.owner != acting_identity():
            raise AuthorizationRefused("owner_foreign")
        record_file = control_context.record_path()
        if record_file is None:
            raise RuntimeError("No agent-data root: there is no record to approve under")
    record_file = Path(record_file)
    directory = record_file.parent / AUTHORIZATIONS_DIR_NAME
    try:
        with control_context.held_epoch(record_file, request.target) as epoch:
            now = _now_ns()
            if now >= request.expires_at_ns:
                raise AuthorizationRefused("auth_expired", "The request's expiry has passed.")
            grant = request.grant(
                switch_epoch=epoch,
                approver=Approver(identity=identity, surface=surface, at_ns=now),
                issued_at_ns=now,
            )
            issue_grant(
                grant,
                directory=directory,
                peers=self_issued_directories() if peers is None else peers,
            )
    except control_context.RecordUnavailable as exc:
        raise AuthorizationRefused("record_unavailable") from exc
    control_context.invalidate_cache()
    return grant
