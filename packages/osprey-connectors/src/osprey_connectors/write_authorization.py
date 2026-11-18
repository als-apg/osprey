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
:func:`spend_authorized_write` for every write call of a process that names an
authorization (:data:`AUTHORIZATION_ENV_VAR`, which only that guard reads), and
the connector-host pool sets that variable for a child whose key names one; a
gateway choice and a child's report take :func:`authorized_write_verdict`. Each
takes the authorization as an argument: none of them reads the environment.

**1. Two files, two writers.** The grant is ``<authorizations dir>/<id>.json``,
written once by the approver's process and afterwards only to mark it
revoked (:func:`_issue_grant`, :func:`revoke_grants`). The authorizations
directory is the owner's: :data:`AUTHORIZATIONS_DIR_NAME` inside the owner's
directory of the read-only control-context tree when one is bound, else inside
this host's :func:`~osprey_connectors.write_switch.state_dir`. The **ledger**
is ``<ledger dir>/<id>.ledger.json``, written only by the holder: the holder
opens it with its lease (:func:`start_lease`), its heartbeat renews the lease
(:func:`renew_lease`), and the connector child appends one step per write call
(:func:`_spend`). The ledger directory is :data:`LEDGER_DIR_ENV_VAR`, else,
on a host with no tree bound, the host's authorizations directory; under a
bound tree with no ledger directory there is none, and every write refuses.
The control-context record is not touched: ``control_context.json`` keeps its
schema and its seven fields.

**2. Reads fail closed.** Both files are read the way the write switch reads
the tree (:func:`~osprey_connectors.control_context.read_tree_context`): no
symbolic link is followed at any directory below the root or at the file (the
file is opened ``O_NOFOLLOW``), the read is bounded, the file must be a regular
file, and the payload must be exactly this schema — every field present, of its
type, no other field, no repeated key, every number finite. A grant that is
absent is no authorization (``switch_off``); every failure on a path that is
there refuses (``auth_unreadable``), and so do a missing or unusable ledger
directory and a ledger that cannot be read. A ledger that is absent has no
lease, and refuses ``auth_lease_lapsed``.

**3. The verdict.** :func:`_verdict` judges one write call — a
whole batch of ``(channel, value)`` operations — in one total order: the launch
pin (:func:`~osprey_connectors._control_state.launch_pin_term`, read here
because the switch verdict reaches it only with the switch on), the owner, the
record, the grant, its holder and owner, its machine, a stop, its expiry, the
switch position for a self-issued grant, every channel listed and every value a
finite number in its range, the lease, and a step left. It does not repeat the
switch verdict's earlier terms; :func:`authorized_execution_verdict` (and
:func:`spend_authorized_write`, which spends) is the composition a write path
takes: the execution verdict first, and for a process
that names an authorization, whenever that verdict allows or refuses only
``switch_off``, this one in its place. An authorization never lifts a tier, an
unknown owner, an unreadable record, a pin or a hold, and a person turning the
switch on mid-run never lifts the bounds.

**4. One write call is one step.** :func:`_spend` takes the ledger's lock
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

**5. The lease.** The holder renews ``lease_expires_at``; a lease that lapses
ends the authorization for good — :func:`renew_lease` refuses once it has
lapsed. Where the reader shares the holder's PID namespace, the holder's PID,
its start time and its state are also asked (:func:`_pid_namespace`,
:func:`_process_start_time`): a dead, reused or zombie PID lapses the lease
before its time. From another namespace the lease time alone decides.

**6. Self-issue at ``allowed``.** A holder whose owner's switch stands at
``allowed`` for the machine may write its own grant into its ledger directory
(:func:`self_issue`), with approver surface ``write_switch``. Such a grant is
honoured only while the position stays ``allowed``: the first write after it
leaves ``allowed`` is refused and lapses the lease, so the switch coming back
does not revive it. A grant found in the ledger directory that claims any
other surface is not honoured at all, because the holder can write there.

**7. Overlap.** Two programs steering one channel of one machine is never what
an approver meant: :func:`_auth_overlap` finds the live grant of another holder
sharing a channel, and :func:`_issue_grant` refuses ``auth_overlap`` under the
directory's lock, so two approvals cannot both pass the check. A grant whose
every ledger in sight shows its lease time passed, by a grace for clock skew,
does not count: no write may use it. This assumes a holder keeps one ledger
directory per authorization; :func:`start_lease` refuses only a ledger already
in its own directory, so a second lease under the same id elsewhere is not
covered. The check reads the owner's own directory and the holders'
self-issued grants in the ledger directories this process sees
(:func:`self_issued_directories`); another owner's directory is not read. A
grant file that cannot be read is logged and left out of the check: it
refuses every write under it already, and a file nobody can clear from the
chip must not veto every approval.

**8. Requests, approval and the stop.** A holder asks with a **request**,
``<ledger dir>/requests/<id>.json`` (:func:`write_request`), read the way a
grant is read (:func:`_parse_request`, :func:`list_requests`); a web terminal
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
group. A listing removes grants that ended more than :data:`_PRUNE_AFTER_S` ago
(:func:`prune_grants`).

This module imports nothing from ``osprey`` and sits above the record and the
write switch, which it reads.
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
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Final, Literal, NoReturn, Self, TypeVar, get_args

from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    ValidationError,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from osprey_connectors import _control_state, control_context, write_switch
from osprey_connectors.config_sections import ControlSystemSection
from osprey_connectors.connector_types import ControlTarget, is_control_target
from osprey_connectors.errors import (
    AlreadyApprovedError,
    ConnectorsError,
    StateLockedError,
    StateUnavailableError,
    require_aware,
)
from osprey_connectors.identity import acting_identity, is_path_segment
from osprey_connectors.model import (
    FrozenModel,
    Instant,
    Listing,
    Login,
    ReadOnlyMapping,
    ReadStatus,
    evolve,
)
from osprey_connectors.refusals import VerdictReason, refusal_phrase
from osprey_connectors.safety_file import SHARED_FILE_MODE, atomic_replace, ensure_shared_dir
from osprey_connectors.write_switch import (
    NO_OWNER,
    Allowed,
    BudgetSpent,
    ChannelNotListed,
    FixedCause,
    NoOwner,
    OverlapHeld,
    RefusalCause,
    Refused,
    StoppedBy,
    ValueOutOfRange,
    WriteVerdict,
    WrongMachine,
)

logger = logging.getLogger("osprey_connectors.write_authorization")

__all__ = [
    "AUTHORIZATIONS_DIR_NAME",
    "AUTHORIZATION_ENV_VAR",
    "LEDGER_DIR_ENV_VAR",
    "LEDGER_VIEW_DIR_ENV_VAR",
    "MAX_EXPIRES_S",
    "MAX_STEPS",
    "REQUESTS_DIR_ENV_VAR",
    "REQUESTS_DIR_NAME",
    "ApprovalSurface",
    "Approver",
    "ApproverSurface",
    "AuthorizationRefusedError",
    "ChannelRange",
    "Grant",
    "GrantStanding",
    "Ledger",
    "LedgerStep",
    "ListedGrant",
    "Revocation",
    "RunCompletion",
    "WriteRequest",
    "approve_request",
    "approved_grant",
    "authorizations_dir",
    "authorized_execution_verdict",
    "authorized_write_verdict",
    "grant_standing",
    "ledger_dir",
    "list_grants",
    "list_requests",
    "prune_grants",
    "read_request",
    "renew_lease",
    "requests_dir",
    "revoke_grants",
    "self_issue",
    "self_issued_directories",
    "spend_authorized_write",
    "start_lease",
    "withdraw_request",
    "write_request",
]

#: The environment variable a holder's connector child carries: the id of the
#: authorization its writes are judged by. The connector-host pool sets it; the
#: control-system connectors' write guard alone reads it.
AUTHORIZATION_ENV_VAR: Final = "OSPREY_WRITE_AUTHORIZATION"

#: The environment variable naming the holder's ledger directory, a path in the
#: holder's own filesystem that it can write.
LEDGER_DIR_ENV_VAR: Final = "OSPREY_WRITE_LEDGER_DIR"

#: The directory under an identity's control-state directory holding its grants.
AUTHORIZATIONS_DIR_NAME: Final = "authorizations"

#: The format of both files. Any other value is unreadable, and refuses.
_SCHEMA_VERSION: Final = 1

#: The most write calls one grant or request may allow: far above any run's
#: evaluations, so no plan, typo or forged file asks for an unbounded budget.
MAX_STEPS: Final = 100_000
#: The longest a grant may stand from its issue, or a request from its asking:
#: a day, three times the optimizer's eight-hour default.
MAX_EXPIRES_S: Final = 24 * 3600.0
#: The longest one renewal of a lease may last: ten times the optimizer's
#: minute, so a holder gone unseen (from another PID namespace) is found
#: lapsed within ten minutes by the verdict, and by the overlap check a
#: :data:`_LEASE_SKEW_GRACE` later.
_MAX_LEASE_S: Final = 600.0
#: How long past a ledger's lease end the overlap check still counts its grant:
#: the holder writes the lease end from its own clock, and the approver compares
#: it against its own.
_LEASE_SKEW_GRACE: Final = timedelta(seconds=60)

#: The surfaces a person approves from: the web terminal's chip or the CLI.
ApprovalSurface = Literal["chip", "cli"]

#: Every surface a grant's approver can name: a person's, or the switch itself.
ApproverSurface = Literal[ApprovalSurface, "write_switch"]

#: The surface of a self-issued grant: the switch itself standing at
#: ``allowed``.
_SELF_ISSUE_SURFACE: Final = "write_switch"

#: What a run does once its evaluations are done, each a write the budget pays
#: for: nothing, the best point found, or the values it started from.
RunCompletion = Literal["none", "apply_best", "restore"]

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

_PROC_STAT: Final = "/proc/{pid}/stat"
_PID_NS_LINK: Final = "/proc/self/ns/pid"


#: The earliest instant a file may state, and the lease end a lapse writes:
#: a lease that ended at the epoch has ended for good.
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)


def _now() -> datetime:
    """The time every term is judged at: the module's one clock, aware and in UTC."""
    return datetime.now(UTC)


# -- the formats --------------------------------------------------------------
#
# Each file holds one of the records below as a JSON object, with ``schema``
# beside its fields: the record's JSON form, ``model_dump(mode="json")``, is
# that object (:class:`_StoredRecord`), and :func:`_parse_stored` reads it back.
# The reader's rules (module rule 2: names that are names, ids, counts, finite
# bounds, an order between two stamps) are the records' own validators, so a
# record that breaks one is refused when it is built, in this process or from
# a file alike, and nothing that breaks one can be written. Reading a file adds
# only what a file can get wrong and a record built here cannot: the JSON (no
# ``NaN``, no repeated key), the schema, and every field present.

#: The validation context a record is read from a file under: a number field
#: is then held as a float (:func:`_number`). No validity rule depends on it.
_FROM_FILE: Final = object()


def _at_least(value: int, minimum: int) -> int:
    if value < minimum:
        raise ValueError(f"is not an integer of at least {minimum}")
    return value


def _stamp(value: datetime) -> datetime:
    """An instant a record states must not be before the epoch."""
    if value < _EPOCH:
        raise ValueError("is before 1970-01-01T00:00:00+00:00")
    return value


def _number(
    value: object, handler: ValidatorFunctionWrapHandler, info: ValidationInfo[object]
) -> float:
    """A number field's value: a float when read from a file, and as given when built here.

    Only how the number is held depends on where it came from, never whether
    it is valid. Read from a file, an integer is made a float, as the reader
    always made it, so a rewrite of that file writes ``1.0`` for a ``1``.
    Built in this process, an ``int`` or a ``float`` is kept as it was given,
    so a writer handed ``60`` writes ``60``, as before; a ``bool``, and
    anything else JSON would not write as a number (a numpy integer among
    them), is refused.
    """
    if info.context is _FROM_FILE:
        number: float = handler(value)
        return number
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("is not a number")
    return value


def _as_held(value: float) -> int | float:
    """A number field written as the type it holds.

    :func:`_number` keeps an ``int`` a writer was handed, so a file it writes
    says ``60`` for ``60``; JSON mode would otherwise write every ``float``
    field as a float (``60.0``).
    """
    return value


def _positive_finite(value: float) -> float:
    """*value* must be a finite number above zero."""
    if not math.isfinite(value):
        raise ValueError("is not finite")
    if value <= 0:
        raise ValueError("is not positive")
    return value


def _hex_id(value: str) -> str:
    if _ID.fullmatch(value) is None:
        raise ValueError("is not a 32-digit lowercase hex string")
    return value


def _channel_map(value: Mapping[str, ChannelRange]) -> Mapping[str, ChannelRange]:
    """``channels`` must name at least one channel, by a non-blank address."""
    if not value:
        raise ValueError("names no channel")
    if any(not address.strip() for address in value):
        raise ValueError("a channel address is empty")
    return value


class ChannelRange(FrozenModel):
    """The closed range of values one channel may be written with.

    Both bounds are finite and ``min`` is not above ``max``.
    """

    min: float
    max: float

    @classmethod
    def of(cls, lo: float, hi: float) -> Self:
        """The range ``[lo, hi]``, each bound taken as a ``float``.

        Raises:
            ValueError: A bound is not a number, is not finite, or *lo* is
                above *hi*.
        """
        return cls(min=float(lo), max=float(hi))

    def holds(self, value: object) -> bool:
        """Whether *value* is a finite real number inside the range."""
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return False
        number = float(value)
        return math.isfinite(number) and self.min <= number <= self.max

    _held = field_serializer("min", "max")(_as_held)

    @field_validator("min", "max", mode="wrap")
    @classmethod
    def _bound(
        cls, value: object, handler: ValidatorFunctionWrapHandler, info: ValidationInfo[object]
    ) -> float:
        return _number(value, handler, info)

    @model_validator(mode="after")
    def _readable(self) -> Self:
        """Both bounds are finite and ``min`` is not above ``max``."""
        if not (math.isfinite(self.min) and math.isfinite(self.max)):
            raise ValueError("a bound is not finite")
        if self.min > self.max:
            raise ValueError("min is above max")
        return self


#: An instant a record states: aware, in UTC, and not before the epoch.
_Stamp = Annotated[Instant, AfterValidator(_stamp)]


class Approver(FrozenModel):
    """Who recorded the approval, from which surface, and when.

    Attributes:
        by: The login that approved; the file states it as ``identity``.
        surface: Where the approval was given.
        approved_at: When.
    """

    model_config = ConfigDict(serialize_by_alias=True)

    by: Login = Field(alias="identity")
    surface: ApproverSurface
    approved_at: _Stamp


class Revocation(FrozenModel):
    """A stop that ended a grant: when, and whose gesture."""

    revoked_at: _Stamp
    by: Login


class _StoredRecord(FrozenModel):
    """A record one file holds: its JSON form carries ``schema`` beside its fields.

    A field whose name differs from its file key is written under the key
    (``serialize_by_alias``), so every dump of the record is the file's form.
    """

    model_config = ConfigDict(serialize_by_alias=True)

    @model_serializer(mode="wrap")
    def _stored(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        fields: dict[str, object] = handler(self)
        return {"schema": _SCHEMA_VERSION, **fields}


class _AuthorizationTerms(_StoredRecord):
    """The terms a request asks for and its grant carries: the same fields, the same rules.

    A flat base, so each file keeps its keys side by side; :meth:`WriteRequest.grant`
    copies them by name.
    """

    id: str
    owner: Login
    holder: Login
    target: ControlTarget
    channels: ReadOnlyMapping[str, ChannelRange]
    steps: int
    expires_at: _Stamp
    lease_s: float
    label: str = ""

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        return _hex_id(value)

    @field_validator("steps")
    @classmethod
    def _steps(cls, value: int) -> int:
        return _at_least(value, 1)

    @field_validator("channels")
    @classmethod
    def _channels(cls, value: Mapping[str, ChannelRange]) -> Mapping[str, ChannelRange]:
        return _channel_map(value)

    _held = field_serializer("lease_s")(_as_held)

    @field_validator("lease_s", mode="wrap")
    @classmethod
    def _lease(
        cls, value: object, handler: ValidatorFunctionWrapHandler, info: ValidationInfo[object]
    ) -> float:
        return _positive_finite(_number(value, handler, info))


class Grant(_AuthorizationTerms):
    """One authorization, as its file states it.

    Attributes:
        id: ``uuid4().hex``; the file is ``<id>.json``.
        owner: The login whose switch the grant stands in for.
        holder: The acting identity that may write under it.
        target: The control target it covers.
        channels: ``{address: ChannelRange}``, at least one.
        steps: How many write calls it allows, at least one.
        expires_at: When it ends whatever else happens.
        lease_s: How long one renewal of the holder's lease lasts.
        label: The run it was approved for, free text.
        switch_epoch: The owner's record's ``switch_epoch`` at issue, a
            token compared only for equality; a record whose epoch has moved
            since (a stop of every machine) ends it.
        issued_at: When it was written.
        approver: Who approved it.
        revoked: The stop that ended it, or ``None``.
    """

    switch_epoch: int
    issued_at: _Stamp
    approver: Approver
    revoked: Revocation | None = None

    def live_at(self, now: datetime) -> bool:
        """Not revoked and not expired at *now*; at its expiry it is expired. The lease is the ledger's.

        Raises:
            NaiveDatetimeError: *now* names no zone.
        """
        return self.revoked is None and require_aware(now, name="now") < self.expires_at

    @field_validator("switch_epoch")
    @classmethod
    def _epoch(cls, value: int) -> int:
        return _at_least(value, 0)

    @model_validator(mode="after")
    def _expiry(self) -> Self:
        if self.expires_at <= self.issued_at:
            raise ValueError("the grant expires before it was issued")
        return self


class LedgerStep(FrozenModel):
    """One write call spent: its number, when, and the channels it named."""

    n: int
    spent_at: _Stamp
    channels: tuple[str, ...] = Field(strict=False)

    @field_validator("n")
    @classmethod
    def _number(cls, value: int) -> int:
        return _at_least(value, 1)

    @field_validator("channels")
    @classmethod
    def _channels(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or not all(value):
            raise ValueError("is not a non-empty list of names")
        return value


class Ledger(_StoredRecord):
    """The holder's consumption of one authorization.

    Attributes:
        authorization: The grant's id.
        holder_pid: The PID of the process that renews the lease, or ``None``
            to rely on the lease time alone.
        holder_start: That process's start time as the kernel states it, so a
            reused PID is not taken for it; ``None`` where nothing states it.
            A token, not an instant: it is in the kernel's own units and only
            ever compared for equality with itself.
        pid_ns: That process's PID namespace (:func:`_pid_namespace`), so a
            reader in another namespace does not look its PID up. A token
            compared for equality, not a time.
        lease_expires_at: When the lease lapses unless renewed first; the
            epoch once it has lapsed for good.
        spent: Every write call spent, in order, numbered from 1; the file
            states it as ``steps``.
    """

    authorization: str
    holder_pid: int | None
    holder_start: str | None
    pid_ns: str | None
    lease_expires_at: _Stamp
    spent: tuple[LedgerStep, ...] = Field(default=(), strict=False, alias="steps")

    @field_validator("authorization")
    @classmethod
    def _id(cls, value: str) -> str:
        return _hex_id(value)

    @field_validator("holder_pid")
    @classmethod
    def _pid(cls, value: int | None) -> int | None:
        return None if value is None else _at_least(value, 1)

    @model_validator(mode="after")
    def _numbered(self) -> Self:
        for index, step in enumerate(self.spent, start=1):
            if step.n != index:
                raise ValueError(f"step {index} is numbered {step.n}")
        return self


class AuthorizationRefusedError(ConnectorsError):
    """A writer of this module refused, for one of the refusal reasons.

    Attributes:
        reason: The :class:`~osprey_connectors.refusals.VerdictReason` it refused for.
        cause: Which of the causes *reason* covers it was
            (:data:`~osprey_connectors.write_switch.RefusalCause`), or ``None``
            when the reason says it all.
    """

    def __init__(self, reason: VerdictReason, cause: RefusalCause | None = None) -> None:
        self.reason = reason
        self.cause = cause
        phrase = refusal_phrase(reason)
        detail = self.detail
        super().__init__(f"{phrase} {detail}" if detail else phrase)

    @property
    def detail(self) -> str | None:
        """What the refusal adds after its phrase: its cause's sentence, or ``None``."""
        return None if self.cause is None else self.cause.sentence()


# -- parsing ------------------------------------------------------------------


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"{name} is not a finite number")


def _no_repeated_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"the key {key!r} is repeated")
        payload[key] = value
    return payload


def _load_json(raw: str | bytes | bytearray) -> object:
    """*raw* as JSON with no ``NaN``/``Infinity`` and no repeated key."""
    return json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_no_repeated_keys)


def _first_error(exc: ValidationError) -> str:
    """The first thing *exc* found wrong: the field it is at, and why."""
    error = exc.errors(include_url=False)[0]
    where = ".".join(str(part) for part in error["loc"])
    return f"{where}: {error['msg']}" if where else error["msg"]


_Stored = TypeVar("_Stored", bound=FrozenModel)


def _parse_stored(model: type[_Stored], raw: str | bytes | bytearray, what: str) -> _Stored:
    """A file's record out of its JSON text: exactly the record's fields, and the schema.

    The text is JSON with no ``NaN``/``Infinity`` and no repeated key. Its
    object holds ``schema``, exactly :data:`_SCHEMA_VERSION`, and every field
    of *model* under its file key, a field with a default included, and
    nothing else; each field is of its type and meets the record's rules, and
    a number is held as a float (:data:`_FROM_FILE`).

    Raises:
        ValueError: The text is not such a record. The message names *what*,
            the field and the reason.
    """
    payload = _load_json(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{what} is not an object")
    fields = dict(payload)
    if fields.pop("schema", None) is not _SCHEMA_VERSION:
        raise ValueError(f"{what}'s schema is not {_SCHEMA_VERSION}")
    try:
        record = model.model_validate(fields, context=_FROM_FILE)
    except ValidationError as exc:
        raise ValueError(f"{what} is unreadable at {_first_error(exc)}") from None
    unset = sorted(
        info.alias or name
        for name, info in model.model_fields.items()
        if name not in record.model_fields_set
    )
    if unset:
        raise ValueError(f"{what} has no {', '.join(unset)}")
    return record


def _within_ceilings(steps: int, start_at: datetime, expires_at: datetime, lease_s: float) -> None:
    """Refuse a budget, an expiry or a lease past its ceiling.

    The ceilings are :data:`MAX_STEPS`, :data:`MAX_EXPIRES_S` from *start_at*
    (an expiry exactly that far away is within it),
    and :data:`_MAX_LEASE_S`. Every new grant and request is written through
    this (:func:`_issue_grant`, :func:`write_request`), so an approval and a
    self-issue meet the same ceilings. Reading does not apply them: a file
    already on disk past a ceiling is honoured as it was before the ceilings,
    so readers of every version agree on it.
    """
    if steps > MAX_STEPS:
        raise ValueError(f"steps {steps} is above the ceiling of {MAX_STEPS}")
    if expires_at - start_at > timedelta(seconds=MAX_EXPIRES_S):
        raise ValueError(f"the expiry is more than {MAX_EXPIRES_S:g} s away")
    if lease_s > _MAX_LEASE_S:
        raise ValueError(f"lease_s {lease_s:g} is above the ceiling of {_MAX_LEASE_S:g}")


def _authorization_id(value: object) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError("the id is not a 32-digit lowercase hex string")
    return value


def _parse_grant(raw: str | bytes | bytearray) -> Grant:
    """A grant out of its file's JSON text (:func:`_parse_stored`).

    An unknown schema, a missing or extra field, a repeated key, a bound that
    is not a finite number, a range with ``min`` above ``max``, a ``steps``
    below one, an expiry not after the issue, an unknown surface — each is a
    grant this version cannot honour. The ceilings are a writer's check
    (:func:`_within_ceilings`), not a reader's.

    Raises:
        ValueError: The text is not a grant this version can read.
    """
    return _parse_stored(Grant, raw, "the grant")


def _parse_ledger(raw: str | bytes | bytearray) -> Ledger:
    """A ledger out of its file's JSON text (:func:`_parse_stored`).

    The same strictness as :func:`_parse_grant`; the steps must be numbered
    1, 2, … in order, each naming at least one channel.

    Raises:
        ValueError: The text is not a ledger this version can read.
    """
    return _parse_stored(Ledger, raw, "the ledger")


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


def _owner_place(owner: str) -> _Place | None:
    """Where *owner*'s grants live for this process, or ``None`` when nowhere usable.

    Under a bound tree, ``<tree>/<owner>/authorizations``; on a host, the
    host's own state directory, which holds only its own login's.
    """
    if _control_state.tree_is_bound():
        tree = _control_state.control_context_tree()
        if tree is None or not is_path_segment(owner):
            return None
        return _Place(tree, (owner, AUTHORIZATIONS_DIR_NAME))
    if owner != acting_identity():
        return None
    root = _control_state.state_dir()
    return None if root is None else _Place(root, (AUTHORIZATIONS_DIR_NAME,))


def _ledger_place() -> _Place | None:
    """The holder's ledger directory, or ``None`` when this process has none."""
    bound = _control_state.absolute_bind(LEDGER_DIR_ENV_VAR)
    if bound is not None:
        return _Place(bound)
    if _control_state.tree_is_bound():
        return None
    root = _control_state.state_dir()
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


# -- reading ------------------------------------------------------------------


def _dir_status(place: _Place) -> ReadStatus:
    """Whether *place* is a plain directory: ``missing`` at the first missing segment.

    ``unreadable`` for the root not being a directory, a link or a non-directory
    at any segment, and any other failure.
    """
    try:
        if not place.root.is_dir():
            return "missing" if not place.root.exists() else "unreadable"
    except OSError:
        return "unreadable"
    probe = place.root
    for segment in place.segments:
        probe = probe / segment
        try:
            mode = os.lstat(probe).st_mode
        except FileNotFoundError:
            return "missing"
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


def _read_grant_at(place: _Place, authorization: str) -> tuple[ReadStatus, Grant | None]:
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
        grant = _parse_grant(_read_no_follow(path, _GRANT_READ_LIMIT).decode("utf-8"))
    except FileNotFoundError:
        return "missing", None
    except Exception as exc:  # every failure on a present grant refuses
        logger.warning("The write authorization %s cannot be read: %s", path, exc)
        return "unreadable", None
    if grant.id != authorization:
        logger.warning("The write authorization %s names another id, %s", path, grant.id)
        return "unreadable", None
    return "ok", grant


def _read_ledger_at(place: _Place, authorization: str) -> tuple[ReadStatus, Ledger | None]:
    """The ledger ``<id>.ledger.json`` under *place*; a missing directory is unreadable."""
    if _dir_status(place) != "ok":
        logger.warning("The write ledger directory %s is not a plain directory", place.path)
        return "unreadable", None
    path = place.path / f"{authorization}{_LEDGER_SUFFIX}"
    try:
        ledger = _parse_ledger(_read_no_follow(path, _LEDGER_READ_LIMIT).decode("utf-8"))
    except FileNotFoundError:
        return "missing", None
    except Exception as exc:  # every failure on a present ledger refuses
        logger.warning("The write ledger %s cannot be read: %s", path, exc)
        return "unreadable", None
    if ledger.authorization != authorization:
        logger.warning("The write ledger %s names another authorization", path)
        return "unreadable", None
    return "ok", ledger


def _find_grant(owner: str, authorization: str) -> tuple[ReadStatus, Grant | None]:
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
    if status != "missing":
        return status, grant
    holder_place = _ledger_place()
    if holder_place is None or holder_place.path == owner_place.path:
        return "missing", None
    status, grant = _read_grant_at(holder_place, authorization)
    if grant is not None and grant.approver.surface != _SELF_ISSUE_SURFACE:
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
    if _control_state.tree_is_bound():
        tree = _control_state.control_context_tree()
        if tree is None:
            return False, None
        read = control_context.read_tree_context(tree, owner)
        return read.unavailable is None, read.record
    own = control_context.read_record_status()
    if own.status == "unreadable":
        return False, None
    return True, own.value


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


def _process_start_time(pid: int) -> str | None:
    """When *pid* started, in the kernel's own units, or ``None`` where nothing states it.

    Only ever compared with itself, which is what tells a holder from a process
    that reused its PID. ``None`` on a platform with no ``/proc``.
    """
    kernel = _kernel_stat(pid)
    return None if kernel is None else kernel[1]


def _pid_namespace() -> str | None:
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
    if ledger.holder_pid is None or ledger.pid_ns != _pid_namespace():
        return True
    if not control_context.is_process_alive(ledger.holder_pid):
        return False
    kernel = _kernel_stat(ledger.holder_pid)
    if kernel is None:
        return True
    state, start = kernel
    return state != "Z" and (ledger.holder_start is None or start == ledger.holder_start)


def _lease_live(ledger: Ledger, now: datetime) -> bool:
    """Whether the lease holds at *now*: at its end it has lapsed."""
    return now < ledger.lease_expires_at and _holder_live(ledger)


# -- the verdict --------------------------------------------------------------


#: The operations a gateway choice or a child's report is judged on: whether
#: the authorization would allow a write at all, every channel term skipped.
#: Never a write's own operations: :func:`_spend` cannot be handed it.
_ANY_WRITE: Final = object()


@dataclass(frozen=True)
class _Basis:
    """What the switch verdict already read, for a judgement that follows it.

    ``switch`` is that verdict (:func:`~osprey_connectors.write_switch._switch_basis`):
    it names the machine kind every answer here carries, and stands in for a
    second read of the switch's position for a self-issued grant. ``record``
    is the record it read when ``read`` is true, so the grant's epoch and the
    switch are judged on one read of it; a record it did not reach is read
    afresh.
    """

    switch: WriteVerdict
    read: bool
    record: control_context.ControlContext | None


@dataclass(frozen=True)
class _Judgement:
    """One verdict and what it was reached on, for the writer that appends after it."""

    verdict: WriteVerdict
    ledger: Ledger | None = None
    #: A self-issued grant whose switch has left ``allowed``: the writer that
    #: holds the ledger's lock lapses the lease, so the grant ends for good.
    ends: bool = False


def _resolved_owner(owner: str | NoOwner | None) -> str | None:
    resolved = write_switch.current_owner(owner)
    if resolved is NO_OWNER or not isinstance(resolved, str) or not resolved.strip():
        return None
    return resolved.strip()


def _operations(operations: object) -> list[tuple[object, object]] | None:
    """*operations* as a list of pairs, or ``None`` when it is not a batch of pairs."""
    if isinstance(operations, Mapping):
        return list(operations.items())
    if isinstance(operations, str | bytes) or not isinstance(operations, Iterable):
        return None
    pairs: list[tuple[object, object]] = []
    for entry in operations:
        if isinstance(entry, str | bytes) or not isinstance(entry, Sequence) or len(entry) != 2:
            return None
        pairs.append((entry[0], entry[1]))
    return pairs


def _judge(
    section: ControlSystemSection,
    target: object,
    operations: object,
    authorization: object,
    owner: str | NoOwner | None,
    now: datetime,
    basis: _Basis | None = None,
) -> _Judgement:
    """The verdict's terms in order (module rule 3). May raise; callers refuse on it.

    *operations* :data:`_ANY_WRITE` skips the channel terms (11): the answer is
    whether some write could go ahead now. *basis* carries what the switch
    verdict already read, and the machine kind it named; with none, the
    switch is read afresh where a term needs it and the answer names no kind.
    """
    named = target if isinstance(target, str) else None
    kind = None if basis is None else basis.switch.kind
    identifier = authorization if isinstance(authorization, str) else None

    def refuse(reason: VerdictReason, cause: RefusalCause | None = None) -> _Judgement:
        return _Judgement(
            Refused(reason=reason, target=named, kind=kind, cause=cause, authorization=identifier)
        )

    if identifier is None or _ID.fullmatch(identifier) is None:
        return refuse(VerdictReason.AUTH_UNREADABLE, FixedCause(kind="not_an_id"))
    if _control_state.launch_pin_term(named) != "none":
        return refuse(VerdictReason.LAUNCH_PINNED)

    resolved = _resolved_owner(owner)
    if resolved is None:
        if _control_state.is_dispatch_run():
            return refuse(VerdictReason.OWNER_UNSCHEDULED)
        return refuse(VerdictReason.OWNER_UNKNOWN)
    if not _control_state.tree_is_bound() and resolved != acting_identity():
        return refuse(VerdictReason.OWNER_FOREIGN)

    if basis is not None and basis.read:
        readable, record = True, basis.record
    else:
        readable, record = _governing_record(resolved)
    if not readable:
        return refuse(VerdictReason.RECORD_UNAVAILABLE)

    status, grant = _find_grant(resolved, identifier)
    if status == "missing":
        return refuse(VerdictReason.SWITCH_OFF)
    if status != "ok" or grant is None:
        return refuse(VerdictReason.AUTH_UNREADABLE)

    if grant.owner != resolved or grant.holder != acting_identity():
        return refuse(VerdictReason.AUTH_HOLDER)
    if grant.target != named:
        return refuse(VerdictReason.AUTH_CHANNEL, WrongMachine(covers=grant.target))
    if grant.revoked is not None:
        stopped = StoppedBy(by=grant.revoked.by, revoked_at=grant.revoked.revoked_at)
        return refuse(VerdictReason.AUTH_REVOKED, stopped)
    if record is None or record.switch_epoch != grant.switch_epoch:
        return refuse(VerdictReason.AUTH_REVOKED, FixedCause(kind="epoch_moved"))
    if now >= grant.expires_at:
        return refuse(VerdictReason.AUTH_EXPIRED)
    if grant.approver.surface == _SELF_ISSUE_SURFACE:
        # The pin was judged first, so the switch verdict that honoured it and
        # one that ignores it agree here.
        switch = (
            basis.switch
            if basis is not None
            else write_switch.write_verdict(section, named, owner=resolved, ignore_launch_pin=True)
        )
        if not switch.allowed or switch.position != "allowed":
            ended = refuse(VerdictReason.AUTH_REVOKED, FixedCause(kind="switch_left_allowed"))
            return _Judgement(ended.verdict, ends=True)

    pairs = [] if operations is _ANY_WRITE else _operations(operations)
    if pairs is None or (not pairs and operations is not _ANY_WRITE):
        return refuse(VerdictReason.AUTH_CHANNEL, FixedCause(kind="empty_batch"))
    for address, value in pairs:
        bound = grant.channels.get(address) if isinstance(address, str) else None
        if bound is None:
            # Spelt as the sentence always spelt it, whatever the address is.
            return refuse(VerdictReason.AUTH_CHANNEL, ChannelNotListed(address=f"{address}"))
        if not bound.holds(value):
            out_of_range = ValueOutOfRange(address=f"{address}", min=bound.min, max=bound.max)
            return refuse(VerdictReason.AUTH_RANGE, out_of_range)

    ledger_place = _ledger_place()
    if ledger_place is None:
        return refuse(VerdictReason.AUTH_UNREADABLE, FixedCause(kind="no_ledger_directory"))
    ledger_status, ledger = _read_ledger_at(ledger_place, identifier)
    if ledger_status == "missing":
        return refuse(VerdictReason.AUTH_LEASE_LAPSED, FixedCause(kind="lease_never_started"))
    if ledger_status != "ok" or ledger is None:
        return refuse(VerdictReason.AUTH_UNREADABLE, FixedCause(kind="ledger_unreadable"))
    if not _lease_live(ledger, now):
        return refuse(VerdictReason.AUTH_LEASE_LAPSED)
    if len(ledger.spent) >= grant.steps:
        return refuse(VerdictReason.AUTH_EXHAUSTED, BudgetSpent(allowed=grant.steps))

    return _Judgement(
        Allowed(target=named, kind=kind, position=None, authorization=identifier), ledger
    )


def _unreachable(target: object, authorization: object, what: str) -> Refused:
    logger.warning("The write authorization verdict could not be reached; refusing (%s)", what)
    return Refused(
        reason=VerdictReason.AUTH_UNREADABLE,
        target=target if isinstance(target, str) else None,
        kind=None,
        authorization=authorization if isinstance(authorization, str) else None,
    )


def _verdict(
    section: ControlSystemSection,
    target: object,
    operations: object,
    authorization: object,
    owner: str | NoOwner | None,
    basis: _Basis | None,
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
        ``allowed`` for *target* now (:func:`_spend` then ends its lease).
    11. ``auth_channel`` — the batch is empty or names a channel the grant does
        not list; ``auth_range`` — a value that is not a finite number inside
        its channel's range. The whole batch is judged before anything is
        written.
    12. ``auth_unreadable`` — no ledger directory, or a ledger that cannot be
        read; ``auth_lease_lapsed`` — no ledger, a lease past its time, or a
        holder the kernel says is gone.
    13. ``auth_exhausted`` — every step has been spent.

    This reads and never writes, so it is the answer for a report or a
    gateway choice; a write spends its step through :func:`_spend`. It
    does not repeat the switch verdict's read-only, tier and hold terms:
    :func:`authorized_execution_verdict` is the composition a write path takes
    (:func:`spend_authorized_write` when it spends).

    Args:
        section: The ``control_system`` section of the resolved config.
        target: The control target the write lands on.
        operations: The write call's ``(channel, value)`` pairs, or a mapping;
            :data:`_ANY_WRITE` for a verdict on any write.
        authorization: The id the process names.
        owner: Whose switch the grant stands in for; ``None`` asks
            :func:`~osprey_connectors.write_switch.current_owner`.
        basis: The record and switch verdict the caller already read, or
            ``None`` to read the record and the switch here (the answer then
            names no machine kind).

    Returns:
        An :class:`~osprey_connectors.write_switch.Allowed` or a
        :class:`~osprey_connectors.write_switch.Refused` whose
        ``authorization`` names the id; an allowed one has no ``position``.
        Never raises: a verdict that cannot be reached refuses
        ``auth_unreadable``.
    """
    try:
        return _judge(section, target, operations, authorization, owner, _now(), basis).verdict
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
        StateLockedError: The lock was not acquired in time.
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
                    raise StateLockedError(
                        f"{lock} stayed locked for {_LOCK_TIMEOUT_S:g} s; nothing was written",
                        path=lock,
                        waited_s=_LOCK_TIMEOUT_S,
                    ) from None
                time.sleep(_LOCK_POLL_S)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_replace(path, data, mode=SHARED_FILE_MODE)


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
        AuthorizationRefusedError: ``auth_unreadable`` — no usable ledger directory.
    """
    _authorization_id(authorization)
    place = _ledger_place()
    if create and place is not None and place.segments and _dir_status(place) == "missing":
        _prepare_directory(place.path)
    if place is None or _dir_status(place) != "ok":
        raise AuthorizationRefusedError(
            VerdictReason.AUTH_UNREADABLE, FixedCause(kind="no_usable_ledger_directory")
        )
    return place


def _lapse(place: _Place, authorization: str) -> None:
    """End the lease now, for good. The caller holds the ledger's lock."""
    status, ledger = _read_ledger_at(place, authorization)
    if status == "ok" and ledger is not None and ledger.lease_expires_at > _EPOCH:
        lapsed = evolve(ledger, lease_expires_at=_EPOCH)
        _write_json(place.path / f"{authorization}{_LEDGER_SUFFIX}", lapsed.model_dump(mode="json"))


def _spend(
    section: ControlSystemSection,
    target: object,
    operations: object,
    authorization: object,
    owner: str | NoOwner | None,
    basis: _Basis | None,
) -> WriteVerdict:
    """Judge one write call and, when it is allowed, spend its step first.

    Under the ledger's lock: the whole :func:`_verdict` against
    the ledger read inside the lock, then one step appended (``n`` the next
    number, the time, the batch's channels) and the ledger replaced whole. An
    allowed answer means the step is on disk; the caller writes only then, so
    a crash after this costs the step. Concurrent callers in any process are
    serialised by the lock, so no two of them spend the last step.

    A write the ledger could not record refuses ``auth_unreadable``. A
    self-issued grant refused because its switch has left ``allowed`` has its
    lease lapsed here as well, so the switch coming back does not revive it.
    A write's own operations only: :data:`_ANY_WRITE` refuses.

    Returns:
        As :func:`_verdict`. Never raises.
    """
    if operations is _ANY_WRITE:
        return _unreachable(target, authorization, "spend without a write")
    try:
        authorization_id = _authorization_id(authorization)
        place = _checked_ledger_place(authorization_id)
    except Exception:  # no ledger to spend from: the read-only verdict names why
        verdict = _verdict(section, target, operations, authorization, owner, basis)
        return _unreachable(target, authorization, "ledger") if verdict.allowed else verdict
    try:
        # Taken once: a one-shot iterable is judged and recorded as the same batch.
        batch = _operations(operations)
        with _locked(_ledger_lock_path(place, authorization_id)):
            now = _now()
            judged = _judge(section, target, batch, authorization_id, owner, now, basis)
            if judged.ends:
                _lapse(place, authorization_id)
            if not judged.verdict.allowed or judged.ledger is None:
                return judged.verdict
            ledger = judged.ledger
            channels = tuple(str(address) for address, _value in batch or ())
            step = LedgerStep(n=len(ledger.spent) + 1, spent_at=now, channels=channels)
            _write_json(
                place.path / f"{authorization_id}{_LEDGER_SUFFIX}",
                evolve(ledger, spent=(*ledger.spent, step)).model_dump(mode="json"),
            )
            return judged.verdict
    except Exception:  # an unrecorded step is never an allowed write
        return _unreachable(target, authorization, "spend")


def _authorized_basis(
    section: ControlSystemSection,
    target: str | None,
    authorization: str,
    owner: str | NoOwner | None,
    session_keys: Sequence[str] | None,
    *,
    hold: bool,
) -> Refused | _Basis:
    """The switch verdict an authorized write follows: the refusal that stands, or the basis.

    The switch's refusal stands, naming *authorization*, for every reason but
    ``switch_off``: an authorization lifts no tier, owner, record, pin or hold.
    Otherwise — the switch allowed or was off — the authorization decides,
    judged on what the switch verdict read.
    """
    switch, read, record = write_switch._switch_basis(
        section, target, owner, session_keys, hold=hold
    )
    if isinstance(switch, Refused) and switch.reason is not VerdictReason.SWITCH_OFF:
        return evolve(switch, authorization=authorization)
    return _Basis(switch=switch, read=read, record=record)


def authorized_execution_verdict(
    section: ControlSystemSection,
    target: str | None,
    operations: Sequence[tuple[str, object]] | Mapping[str, object],
    *,
    authorization: str,
    owner: str | NoOwner | None = None,
    session_keys: Sequence[str] | None = None,
) -> WriteVerdict:
    """The verdict on one write call about to execute under *authorization*; it spends nothing.

    :func:`~osprey_connectors.write_switch.execution_verdict` first, which
    stands whenever it refuses for any reason but ``switch_off`` — an
    authorization lifts no tier, owner, record, pin or hold — and otherwise,
    whether the switch allowed or was off, the authorization's verdict in its
    place: a person turning the switch on mid-run does not lift the bounds. A
    process that names no authorization asks
    :func:`~osprey_connectors.write_switch.execution_verdict` itself.

    Args:
        section: The ``control_system`` section of the resolved config.
        target: The control target the write lands on.
        operations: The write call's ``(channel, value)`` pairs.
        authorization: The id the process names.
        owner: As for :func:`_verdict`.
        session_keys: As for
            :func:`~osprey_connectors.write_switch.execution_verdict`.

    The record is read once: the authorization's terms are judged on the
    record the switch verdict read, and a self-issued grant's position on that
    verdict's own answer.

    Returns:
        An :class:`~osprey_connectors.write_switch.Allowed` or a
        :class:`~osprey_connectors.write_switch.Refused`. Never raises.
    """
    basis = _authorized_basis(section, target, authorization, owner, session_keys, hold=True)
    if isinstance(basis, Refused):
        return basis
    return _verdict(section, target, operations, authorization, owner, basis)


def spend_authorized_write(
    section: ControlSystemSection,
    target: str | None,
    operations: Sequence[tuple[str, object]] | Mapping[str, object],
    *,
    authorization: str,
    owner: str | NoOwner | None = None,
    session_keys: Sequence[str] | None = None,
) -> WriteVerdict:
    """:func:`authorized_execution_verdict`, and when it allows, the call's step spent first.

    The same composition and the same terms; the authorization's are judged
    again under the ledger's lock and the step appended before the answer
    (:func:`_spend`), so an allowed answer means the step is on disk and the
    caller writes only then. Arguments as for
    :func:`authorized_execution_verdict`.

    Returns:
        An :class:`~osprey_connectors.write_switch.Allowed` or a
        :class:`~osprey_connectors.write_switch.Refused`. Never raises.
    """
    basis = _authorized_basis(section, target, authorization, owner, session_keys, hold=True)
    if isinstance(basis, Refused):
        return basis
    return _spend(section, target, operations, authorization, owner, basis)


def authorized_write_verdict(
    section: ControlSystemSection,
    target: str | None,
    *,
    authorization: str,
    owner: str | NoOwner | None = None,
) -> WriteVerdict:
    """Whether a process may write to *target* at all under *authorization*.

    The gateway's and the report's answer: composed as
    :func:`authorized_execution_verdict` composes the execution verdict, over
    :func:`~osprey_connectors.write_switch.write_verdict`, with the
    authorization judged on every term but the channels and values: the grant
    in force, the lease live and a step left. It names no hold, as the switch
    verdict names none, because the gateway a connector selects must not
    follow a session's hold. It reads and never spends. A process that names
    no authorization asks
    :func:`~osprey_connectors.write_switch.write_verdict` itself.

    Returns:
        An :class:`~osprey_connectors.write_switch.Allowed` or a
        :class:`~osprey_connectors.write_switch.Refused`. Never raises.
    """
    basis = _authorized_basis(section, target, authorization, owner, None, hold=False)
    if isinstance(basis, Refused):
        return basis
    return _verdict(section, target, _ANY_WRITE, authorization, owner, basis)


# -- the lease ----------------------------------------------------------------


def _live_grant(owner: str | NoOwner | None, authorization: str, now: datetime) -> Grant:
    """The grant this process holds, still in force (not revoked, epoch, expiry).

    Raises:
        AuthorizationRefusedError: For the reason it is not.
    """
    resolved = _resolved_owner(owner)
    if resolved is None:
        raise AuthorizationRefusedError(VerdictReason.OWNER_UNKNOWN)
    readable, record = _governing_record(resolved)
    if not readable:
        raise AuthorizationRefusedError(VerdictReason.RECORD_UNAVAILABLE)
    status, grant = _find_grant(resolved, authorization)
    if status == "missing":
        raise AuthorizationRefusedError(
            VerdictReason.SWITCH_OFF, FixedCause(kind="no_such_authorization")
        )
    if status != "ok" or grant is None:
        raise AuthorizationRefusedError(VerdictReason.AUTH_UNREADABLE)
    if grant.owner != resolved or grant.holder != acting_identity():
        raise AuthorizationRefusedError(VerdictReason.AUTH_HOLDER)
    if grant.revoked is not None or record is None or record.switch_epoch != grant.switch_epoch:
        raise AuthorizationRefusedError(VerdictReason.AUTH_REVOKED)
    if now >= grant.expires_at:
        raise AuthorizationRefusedError(VerdictReason.AUTH_EXPIRED)
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
        AuthorizationRefusedError: The grant is absent, unreadable, not this
            holder's, revoked or expired (its reason), there is no usable
            ledger directory (``auth_unreadable``), or the ledger exists
            (``auth_lease_lapsed``).
        StateLockedError: The ledger stayed locked past the bounded wait.
        OSError: The write failed.
    """
    place = _checked_ledger_place(authorization, create=True)
    with _locked(_ledger_lock_path(place, authorization)):
        now = _now()
        grant = _live_grant(owner, authorization, now)
        status, _ledger = _read_ledger_at(place, authorization)
        if status != "missing":
            raise AuthorizationRefusedError(
                VerdictReason.AUTH_LEASE_LAPSED, FixedCause(kind="lease_already_started")
            )
        pid = os.getpid()
        ledger = Ledger(
            authorization=authorization,
            holder_pid=pid,
            holder_start=_process_start_time(pid),
            pid_ns=_pid_namespace(),
            lease_expires_at=now + timedelta(seconds=grant.lease_s),
        )
        _write_json(place.path / f"{authorization}{_LEDGER_SUFFIX}", ledger.model_dump(mode="json"))
    return ledger


def renew_lease(authorization: str, *, owner: str | NoOwner | None = None) -> Ledger:
    """Move the lease on by the grant's ``lease_s``, if this process holds it and it is live.

    A compare-and-swap under the ledger's lock; the renewed ledger is returned.
    A renewal that is refused writes nothing and raises, so a holder never
    writes on after a lease it failed to renew. A lapsed lease never comes
    back.

    Raises:
        AuthorizationRefusedError: The reason the grant no longer stands —
            unreadable, another holder's, revoked or expired, or an unreadable
            record (:func:`approved_grant`'s reasons) — and otherwise
            ``auth_lease_lapsed``, with its cause: no such authorization, no
            usable ledger directory, no ledger (``lease_never_started``), a
            ledger that cannot be read, another process holding the lease
            (``lease_held_elsewhere``), any other failure (``renewal_failed``),
            or none for a lease past its time.
    """
    try:
        place = _checked_ledger_place(authorization)
        with _locked(_ledger_lock_path(place, authorization)):
            now = _now()
            grant = _live_grant(owner, authorization, now)
            status, ledger = _read_ledger_at(place, authorization)
            if status == "missing":
                raise _lapsed(FixedCause(kind="lease_never_started"))
            if status != "ok" or ledger is None:
                raise _lapsed(FixedCause(kind="ledger_unreadable"))
            if ledger.holder_pid != os.getpid() or ledger.pid_ns != _pid_namespace():
                raise _lapsed(FixedCause(kind="lease_held_elsewhere"))
            if not _lease_live(ledger, now):
                raise _lapsed()
            renewed = evolve(ledger, lease_expires_at=now + timedelta(seconds=grant.lease_s))
            _write_json(
                place.path / f"{authorization}{_LEDGER_SUFFIX}", renewed.model_dump(mode="json")
            )
            return renewed
    except AuthorizationRefusedError as refused:
        logger.warning("The lease of %r was not renewed: %s", authorization, refused)
        if refused.reason in (VerdictReason.SWITCH_OFF, VerdictReason.AUTH_UNREADABLE) and (
            refused.cause is not None
        ):
            # No such grant, or no ledger directory: nothing stands to renew.
            raise _lapsed(refused.cause) from refused
        raise
    except Exception as exc:  # a renewal that cannot be made is a lease left to lapse
        logger.warning("The lease of %r was not renewed", authorization, exc_info=True)
        raise _lapsed(FixedCause(kind="renewal_failed")) from exc


def _lapsed(cause: RefusalCause | None = None) -> AuthorizationRefusedError:
    """The refusal of a lease that cannot be renewed, for *cause*."""
    return AuthorizationRefusedError(VerdictReason.AUTH_LEASE_LAPSED, cause)


# -- issuing ------------------------------------------------------------------


def _grant_directory(directory: Path | None, owner: str) -> Path:
    if directory is not None:
        return Path(directory)
    path = authorizations_dir(owner)
    if path is None:
        raise StateUnavailableError(f"No authorizations directory is reachable for {owner!r}")
    return path


def _prepare_directory(directory: Path) -> None:
    """Create *directory* (0o2770, as the record's) when absent; refuse a link or a file."""
    if not os.path.lexists(directory):
        ensure_shared_dir(directory)
    mode = os.lstat(directory).st_mode
    if not stat.S_ISDIR(mode):
        raise OSError(errno.ENOTDIR, f"{directory} is not a plain directory; nothing was written")


def _grants_in(directory: Path) -> tuple[tuple[Grant, ...], tuple[Path, ...]]:
    """Every grant file in *directory*: the grants read, and the paths that could not be.

    A file is a grant file by its name, ``<id>.json``; ledgers and locks beside
    them are not. A missing directory holds none. The writers' listing: a
    directory that cannot be listed raises, so no check is made without it.

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


def _lease_ended(grant: Grant, ledger_dirs: Iterable[Path], now: datetime) -> bool:
    """Whether every ledger of *grant* in *ledger_dirs* shows its lease time passed.

    The clock alone decides, as :func:`grant_standing` does: the holder's PID,
    judged from the approver's process, may name another host's process. A
    lease counts as passed only once more than :data:`_LEASE_SKEW_GRACE` has
    gone since its end (exactly that much still blocks), so an approver's
    clock ahead of the holder's does not free a live grant. At
    least one ledger must be found, and each must agree, so a stale or planted
    one in a directory searched first cannot free another's live grant. A
    ledger found nowhere is a lease not started yet, or one out of this
    process's sight, and an unreadable one may be live: neither has ended.

    This assumes a holder keeps one ledger directory per authorization:
    :func:`start_lease` refuses only a ledger already in its own directory, so
    a holder that starts a second lease under the same id in another directory
    out of this process's sight is not covered.
    """
    found = list(_ledgers_of(grant.id, ledger_dirs))
    return bool(found) and all(
        ledger is not None and now > ledger.lease_expires_at + _LEASE_SKEW_GRACE for ledger in found
    )


def _auth_overlap(
    candidate: Grant,
    existing: Iterable[Grant],
    *,
    now: datetime | None = None,
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
    now = _now() if now is None else require_aware(now, name="now")
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

    Only a grant approved by the switch (:func:`_self_issued_grants`) is
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
            found, unreadable = _self_issued_grants(Path(peer))
        except OSError as exc:
            logger.warning("The self-issued grants in %s cannot be listed: %s", peer, exc)
            continue
        if unreadable:
            logger.warning(
                "Unreadable grants in %s, left out of the overlap check: %s", peer, unreadable
            )
        grants.extend(found)
    return grants


def _issue_grant(
    grant: Grant, *, directory: Path | None = None, peers: Iterable[Path] = ()
) -> Path:
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
        ValueError: Its budget, expiry or lease is past a ceiling
            (:func:`_within_ceilings`); nothing is written.
        AlreadyApprovedError: A grant with this id exists.
        AuthorizationRefusedError: ``auth_overlap``.
        StateUnavailableError: No directory is reachable.
        StateLockedError: The lock was not acquired.
        OSError: The directory is not a plain directory, or the write failed.
    """
    _within_ceilings(grant.steps, grant.issued_at, grant.expires_at, grant.lease_s)
    target_dir = _grant_directory(directory, grant.owner)
    _prepare_directory(target_dir)
    path = target_dir / f"{grant.id}{_GRANT_SUFFIX}"
    with _locked(target_dir / _DIRECTORY_LOCK_NAME):
        if os.path.lexists(path):
            raise AlreadyApprovedError(path, authorization=grant.id)
        existing, unreadable = _grants_in(target_dir)
        if unreadable:
            logger.warning(
                "Unreadable grants beside %s, left out of the overlap check: %s", path, unreadable
            )
        peers = tuple(peers)
        held = _auth_overlap(
            grant,
            [*existing, *_peer_grants(peers, target_dir)],
            ledger_dirs=(target_dir, *peers),
        )
        if held is not None:
            raise AuthorizationRefusedError(
                VerdictReason.AUTH_OVERLAP,
                OverlapHeld(holder=held.holder, authorization=held.id, label=held.label),
            )
        _write_json(path, grant.model_dump(mode="json"))
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
    revoking one authorization from a list. The mark is ``{"revoked_at", "by"}`` and is never removed, so the next
    write under any of these refuses ``auth_revoked``. A grant file that cannot
    be read is left as it is — it already refuses every write. *directory*
    defaults to this process's own authorizations directory.

    Raises:
        ValueError: *by* is not a login (:data:`Login`) and a grant was to be
            marked; nothing is marked.
        StateUnavailableError: No directory is reachable.
        StateLockedError: The lock was not acquired.
        OSError: The directory is not a plain directory, or a write failed.
    """
    target_dir = _grant_directory(directory, acting_identity())
    if not os.path.lexists(target_dir):
        return ()
    _prepare_directory(target_dir)
    marked: list[str] = []
    wanted = None if ids is None else frozenset(ids)
    with _locked(target_dir / _DIRECTORY_LOCK_NAME):
        now = _now()
        existing, _unreadable = _grants_in(target_dir)
        for grant in existing:
            if grant.revoked is not None:
                continue
            if targets is not None and grant.target not in targets:
                continue
            if wanted is not None and grant.id not in wanted:
                continue
            stopped = evolve(grant, revoked=Revocation(revoked_at=now, by=by))
            _write_json(target_dir / f"{grant.id}{_GRANT_SUFFIX}", stopped.model_dump(mode="json"))
            marked.append(grant.id)
    return tuple(marked)


def self_issue(
    section: ControlSystemSection,
    target: str,
    channels: Mapping[str, ChannelRange],
    *,
    steps: int,
    expires_at: datetime,
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
    write after it leaves ``allowed`` ends it for good (:func:`_spend`).

    *channels* maps each address to its :class:`ChannelRange`
    (:meth:`ChannelRange.of` builds one from two bounds). *expires_at* is an
    aware instant (any zone; the grant stores it in UTC), at most
    :data:`MAX_EXPIRES_S` after now.

    Raises:
        ValueError: The channels, steps, expiry or lease are malformed, or
            past a ceiling (:func:`_within_ceilings`); a
            :class:`~osprey_connectors.errors.NaiveDatetimeError` for a naive
            *expires_at*.
        AuthorizationRefusedError: The switch verdict's reason; ``auth_revoked``
            when it allows below ``allowed`` or moved during the issue;
            ``record_unavailable`` for a record with no epoch;
            ``auth_unreadable`` for no usable ledger directory.
        StateLockedError: The lock was not acquired in time.
        OSError: The write failed.
    """
    require_aware(expires_at, name="expires_at")
    resolved = _resolved_owner(owner)
    if resolved is None:
        raise AuthorizationRefusedError(VerdictReason.OWNER_UNKNOWN)

    def epoch_at_allowed() -> int:
        switch = write_switch.write_verdict(section, target, owner=resolved)
        if isinstance(switch, Refused):
            raise AuthorizationRefusedError(switch.reason)
        if switch.position != "allowed":
            raise AuthorizationRefusedError(
                VerdictReason.AUTH_REVOKED, FixedCause(kind="switch_not_at_allowed")
            )
        readable, record = _governing_record(resolved)
        if not readable:
            raise AuthorizationRefusedError(VerdictReason.RECORD_UNAVAILABLE)
        if record is None or record.switch_epoch is None:
            raise AuthorizationRefusedError(
                VerdictReason.RECORD_UNAVAILABLE, FixedCause(kind="no_epoch")
            )
        return record.switch_epoch

    epoch = epoch_at_allowed()
    if not is_control_target(target):  # the verdict refused it already; said for the type
        raise AuthorizationRefusedError(VerdictReason.TARGET_UNKNOWN)
    now = _now()
    holder = acting_identity()
    authorization = uuid.uuid4().hex
    # The ledger directory before the grant and its approver are built: a
    # grant its fields refuse is refused after a missing ledger directory is.
    place = _checked_ledger_place(authorization, create=True)
    grant = Grant(
        id=authorization,
        owner=resolved,
        holder=holder,
        target=target,
        switch_epoch=epoch,
        channels=dict(channels),
        steps=steps,
        issued_at=now,
        expires_at=expires_at,
        lease_s=lease_s,
        approver=Approver(by=holder, surface=_SELF_ISSUE_SURFACE, approved_at=now),
        label=label,
    )
    path = _issue_grant(grant, directory=place.path, peers=self_issued_directories())
    try:
        if epoch_at_allowed() != epoch:
            raise AuthorizationRefusedError(
                VerdictReason.AUTH_REVOKED, FixedCause(kind="stop_during_issue")
            )
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
    view = _control_state.absolute_bind(LEDGER_VIEW_DIR_ENV_VAR)
    if view is not None and all(os.path.normpath(view) != os.path.normpath(p) for p in found):
        found.append(view)
    return tuple(found)


def _self_issued_grants(directory: Path) -> tuple[tuple[Grant, ...], tuple[Path, ...]]:
    """The self-issued grants in a ledger *directory*, and the paths that could not be read.

    :func:`_grants_in`, keeping only the grants approved by the switch
    (:data:`_SELF_ISSUE_SURFACE`): a grant there claiming any other surface is
    never honoured, because the holder can write there, so it is not listed.

    Raises:
        OSError: The directory is a link or cannot be listed.
    """
    grants, unreadable = _grants_in(directory)
    return (
        tuple(grant for grant in grants if grant.approver.surface == _SELF_ISSUE_SURFACE),
        unreadable,
    )


def _grant_in_force(
    grant: Grant, record: control_context.ControlContext | None, *, now: datetime | None = None
) -> bool:
    """Whether *grant* still stands against its owner's *record*: what a chip shows as live.

    Not revoked, not expired, and issued under the record's epoch. The lease
    and the steps are the holder's and are not asked. A self-issued grant also
    needs the record's switch for its machine at ``allowed``: once it has left,
    the holder's next write ends it for good, so it is not shown standing.
    """
    now = _now() if now is None else now
    if not (
        grant.live_at(now) and record is not None and record.switch_epoch == grant.switch_epoch
    ):
        return False
    if grant.approver.surface == _SELF_ISSUE_SURFACE:
        return control_context.switch_position(record, grant.target) == "allowed"
    return True


#: How long a revoked or expired grant is kept before a listing removes it, so
#: a holder whose stop has just landed is still told ``auth_revoked`` (or
#: ``auth_expired``) rather than that no authorization exists.
_PRUNE_AFTER_S: Final = 3600.0


def _ended_at(grant: Grant) -> datetime:
    """When *grant* stopped standing for good: its stop or its expiry, the earlier."""
    if grant.revoked is None:
        return grant.expires_at
    return min(grant.revoked.revoked_at, grant.expires_at)


def prune_grants(
    directory: Path, *, now: datetime | None = None, after_s: float = _PRUNE_AFTER_S
) -> tuple[str, ...]:
    """Remove the grants in *directory* that ended *after_s* or more ago; the ids removed.

    *now* is the aware instant to judge at, the module's clock when omitted.

    A grant has ended once it is revoked or expired; a grant still in force,
    and one that cannot be read, are left alone (an unreadable file refuses
    every write already, and its owner should see it). Under the directory's
    lock, so an issue in flight is not overtaken. A missing directory holds
    none. Never removes a ledger, which is the holder's.

    Raises:
        NaiveDatetimeError: *now* names no zone.
        StateLockedError: The lock was not acquired in time.
        OSError: The directory is not a plain directory, or a removal failed.
    """
    now = _now() if now is None else require_aware(now, name="now")
    directory = Path(directory)
    if not os.path.lexists(directory):
        return ()
    _prepare_directory(directory)
    cutoff = now - timedelta(seconds=after_s)
    removed: list[str] = []
    with _locked(directory / _DIRECTORY_LOCK_NAME):
        existing, _unreadable = _grants_in(directory)
        for grant in existing:
            if grant.revoked is None and now < grant.expires_at:
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
        if status != "missing":
            yield ledger


class GrantStanding(FrozenModel):
    """One grant as a listing shows it: whether it stands, and what its holder spent.

    Attributes:
        grant: The grant.
        in_force: :func:`_grant_in_force`, and the holder's ledger has not
            ended it: a grant whose lease lapsed or whose every step is spent
            refuses the next write, so it is not shown standing.
        steps_spent: The steps the ledger counts; ``0`` while there is no
            ledger yet, ``None`` when it cannot be read.
        ledger: The ledger read, or ``None``.
    """

    grant: Grant
    in_force: bool
    steps_spent: int | None
    ledger: Ledger | None


def grant_standing(
    grant: Grant,
    record: control_context.ControlContext | None,
    ledger_dirs: Iterable[Path],
    *,
    now: datetime | None = None,
) -> GrantStanding:
    """Whether *grant* still stands for a listing, its holder's ledger asked.

    The ledger is the first one found in *ledger_dirs* (the holder's ledger
    directory as this process sees it); a directory that is not there is
    skipped. Every surface that lists grants — the chip, its listing, the CLI
    and ``osprey status`` — answers through this, so they agree. *now* is the
    aware instant to judge at, the module's clock when omitted; a lease is
    ended at its end.

    Raises:
        NaiveDatetimeError: *now* names no zone.
    """
    now = _now() if now is None else require_aware(now, name="now")
    in_force = _grant_in_force(grant, record, now=now)
    for ledger in _ledgers_of(grant.id, ledger_dirs):
        if ledger is None:
            return GrantStanding(grant=grant, in_force=in_force, steps_spent=None, ledger=None)
        spent = len(ledger.spent)
        ended = now >= ledger.lease_expires_at or spent >= grant.steps
        return GrantStanding(
            grant=grant, in_force=in_force and not ended, steps_spent=spent, ledger=ledger
        )
    return GrantStanding(grant=grant, in_force=in_force, steps_spent=0, ledger=None)


class ListedGrant(FrozenModel):
    """One grant a listing found, and the directory it was found in."""

    grant: Grant
    directory: Path


def list_grants(*directories: Path, self_issued: Iterable[Path] = ()) -> Listing[ListedGrant]:
    """Every grant in *directories* and every self-issued one in *self_issued*, each once.

    *directories* are grants directories, every grant in them listed;
    *self_issued* are ledger directories, where only a grant approved by the
    switch (:data:`_SELF_ISSUE_SURFACE`) is listed, because only that one is
    honoured there. A file is a grant file by its name, ``<id>.json``; ledgers
    and locks beside them are not. A directory named twice is listed once, a
    grant found twice is listed where it was found first, and a missing
    directory holds none.

    Returns:
        Each grant with the directory it was found in, and the paths that
        could not be read: a grant file, or a directory that is a link or
        cannot be listed. Nothing is written and nothing pruned; never
        raises :class:`OSError`.
    """
    grants: list[ListedGrant] = []
    unreadable: list[Path] = []
    seen: set[str] = set()
    listed: set[str] = set()
    sources = [(Path(d), False) for d in directories] + [(Path(d), True) for d in self_issued]
    for directory, only_self_issued in sources:
        if os.path.normpath(directory) in listed:
            continue
        listed.add(os.path.normpath(directory))
        try:
            found, broken = (_self_issued_grants if only_self_issued else _grants_in)(directory)
        except OSError:
            logger.warning("Could not list the authorizations in %s", directory, exc_info=True)
            unreadable.append(directory)
            continue
        unreadable.extend(broken)
        for grant in found:
            if grant.id not in seen:
                seen.add(grant.id)
                grants.append(ListedGrant(grant=grant, directory=directory))
    return Listing[ListedGrant](items=tuple(grants), unreadable=tuple(unreadable))


def approved_grant(authorization: str, *, owner: str | NoOwner | None = None) -> Grant | None:
    """The grant issued for *authorization*, still in force, or ``None`` while there is none.

    The holder's wait after :func:`write_request`: an approval issues the grant
    under the request's own id, so the holder polls this with the id it asked
    under, then opens its lease (:func:`start_lease`) and names the id
    (:data:`AUTHORIZATION_ENV_VAR`).

    Raises:
        ValueError: *authorization* is not an id.
        AuthorizationRefusedError: The grant is there and does not stand — unreadable,
            another holder's, revoked or expired — or the owner's record
            cannot be read.
    """
    _authorization_id(authorization)
    try:
        return _live_grant(owner, authorization, _now())
    except AuthorizationRefusedError as refused:
        if refused.reason == "switch_off":
            return None
        raise


# -- requests -------------------------------------------------------------------

#: The environment variable naming the directory a web terminal reads holders'
#: requests from: a read-only bind of ``<ledger dir>/requests``.
REQUESTS_DIR_ENV_VAR: Final = "OSPREY_WRITE_REQUESTS_DIR"

#: The directory under a holder's ledger directory its requests are written to.
REQUESTS_DIR_NAME: Final = "requests"

#: A request is a few hundred bytes plus one line per channel.
_REQUEST_READ_LIMIT: Final = 1024 * 1024


class WriteRequest(_AuthorizationTerms):
    """A holder asking for an authorization: what a person approves.

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
        expires_at: When the authorization is to end whatever happens.
        lease_s: How long one renewal of the holder's lease lasts.
        label: The run, free text.
        requested_at: When the request was written.
        completion: What the run does once its evaluations are done, each a
            write the budget pays for: nothing, the best point found, or the
            values it started from.
    """

    requested_at: _Stamp
    completion: RunCompletion = "none"

    def grant(self, *, switch_epoch: int, approver: Approver, issued_at: datetime) -> Grant:
        """The grant approving this request exactly as asked, under its id."""
        terms = {name: getattr(self, name) for name in _AuthorizationTerms.model_fields}
        return Grant.model_validate(
            {
                **terms,
                "switch_epoch": switch_epoch,
                "issued_at": issued_at,
                "approver": approver,
            }
        )

    @model_validator(mode="after")
    def _expiry(self) -> Self:
        if self.expires_at <= self.requested_at:
            raise ValueError("the request expires before it was made")
        return self


def _parse_request(raw: str | bytes | bytearray) -> WriteRequest:
    """A request out of its file's JSON text (:func:`_parse_stored`).

    The grant's strictness (:func:`_parse_grant`): every field present with its
    type and nothing else, no repeated key, every bound a finite number with
    ``min <= max``, ``steps`` at least one, an expiry after the request, a
    positive lease and a known completion. As for a grant, the ceilings are
    checked when a request is written (:func:`write_request`), not read.

    Raises:
        ValueError: The text is not a request this version can read.
    """
    return _parse_stored(WriteRequest, raw, "the request")


def requests_dir() -> Path | None:
    """Where this process reads requests: :data:`REQUESTS_DIR_ENV_VAR`, else its ledger's.

    A web terminal is handed the holders' request directory read-only; a
    holder, or a host with no bind, reads ``<ledger dir>/requests``. ``None``
    when neither is reachable.
    """
    bound = _control_state.absolute_bind(REQUESTS_DIR_ENV_VAR)
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
    request = _parse_request(_read_no_follow(path, _REQUEST_READ_LIMIT).decode("utf-8"))
    if path.name != f"{request.id}{_GRANT_SUFFIX}":
        raise ValueError(f"{path.name} does not name the request it holds, {request.id}")
    return request


def list_requests(directory: Path) -> Listing[WriteRequest]:
    """Every request file in *directory*: the requests read, and the paths that could not be.

    A missing directory holds none.

    Raises:
        OSError: The directory is a link, not a directory, or cannot be listed.
    """
    directory = Path(directory)
    try:
        mode = os.lstat(directory).st_mode
    except FileNotFoundError:
        return Listing[WriteRequest](items=(), unreadable=())
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
    return Listing[WriteRequest](items=tuple(found), unreadable=tuple(unreadable))


def write_request(
    target: str,
    channels: Mapping[str, ChannelRange],
    *,
    steps: int,
    expires_at: datetime,
    lease_s: float,
    label: str = "",
    completion: RunCompletion = "none",
    owner: str | NoOwner | None = None,
) -> WriteRequest:
    """Ask for an authorization: write a request into this holder's ledger directory.

    The holder is this process's acting identity, the owner the login the
    owner ladder names (:func:`~osprey_connectors.write_switch.current_owner`).
    The file is ``<ledger dir>/requests/<id>.json``; the directory is created
    there when absent. Nothing is approved by writing it: a person approves it
    from the chip or the CLI, which issues the grant under the same id
    (:func:`approved_grant` waits for it), or the holder self-issues at
    ``allowed`` (:func:`self_issue`). *channels* maps each address to its
    :class:`ChannelRange` (:meth:`ChannelRange.of` builds one from two
    bounds). *expires_at* is an aware instant (any zone; the request stores it
    in UTC), at most :data:`MAX_EXPIRES_S` after now; the request is stamped
    with the module's clock.

    Raises:
        ValueError: The target, channels, steps, expiry, lease or completion are
            malformed, or the steps, expiry or lease are past a ceiling
            (:func:`_within_ceilings`); nothing is written. A
            :class:`~osprey_connectors.errors.NaiveDatetimeError` for a naive
            *expires_at*.
        AuthorizationRefusedError: ``owner_unknown``; ``auth_unreadable`` for no
            usable ledger directory.
        OSError: The write failed.
    """
    require_aware(expires_at, name="expires_at")
    resolved = _resolved_owner(owner)
    if resolved is None:
        raise AuthorizationRefusedError(VerdictReason.OWNER_UNKNOWN)
    if not is_control_target(target):
        raise ValueError(f"target {target!r} is not a control target")
    request = WriteRequest(
        id=uuid.uuid4().hex,
        owner=resolved,
        holder=acting_identity(),
        target=target,
        channels=dict(channels),
        steps=steps,
        requested_at=_now(),
        expires_at=expires_at,
        lease_s=lease_s,
        label=label,
        completion=completion,
    )
    _within_ceilings(request.steps, request.requested_at, request.expires_at, request.lease_s)
    place = _checked_ledger_place(request.id, create=True)
    directory = place.path / REQUESTS_DIR_NAME
    _prepare_directory(directory)
    _write_json(directory / f"{request.id}{_GRANT_SUFFIX}", request.model_dump(mode="json"))
    return request


def withdraw_request(request_id: str) -> bool:
    """Remove this holder's request *request_id*; whether a file was removed.

    For the holder once its grant is issued, or when its run ends without one.
    The grant, if any, is untouched.

    Raises:
        ValueError: *request_id* is not an id.
        AuthorizationRefusedError: ``auth_unreadable`` for no usable ledger directory.
    """
    place = _checked_ledger_place(request_id)
    try:
        (place.path / REQUESTS_DIR_NAME / f"{request_id}{_GRANT_SUFFIX}").unlink()
    except FileNotFoundError:
        return False
    return True


# -- approval -------------------------------------------------------------------

_APPROVAL_SURFACES: tuple[ApprovalSurface, ...] = get_args(ApprovalSurface)


def approve_request(
    request: WriteRequest,
    *,
    surface: ApprovalSurface,
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
    one — the grant is stamped with that epoch, and :func:`_issue_grant` checks
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
        ValueError: *surface* is not an approval surface, *approver* is not a
            login (:data:`Login`), or the grant is past a ceiling
            (:func:`_within_ceilings`); nothing is written.
        AuthorizationRefusedError: ``record_unavailable`` for an unreadable record;
            ``auth_expired`` for a request whose expiry has passed;
            ``auth_overlap`` from the issue; ``owner_foreign``
            for no *record_file* and an owner who is not this process.
        AlreadyApprovedError: The request is approved already.
        StateLockedError: A lock was not acquired in time.
        StateUnavailableError: No record path resolves.
        OSError: A write failed.
    """
    if surface not in _APPROVAL_SURFACES:
        raise ValueError(f"surface {surface!r} is not one of {_APPROVAL_SURFACES}")
    identity = approver or acting_identity()
    if record_file is None:
        if request.owner != acting_identity():
            raise AuthorizationRefusedError(VerdictReason.OWNER_FOREIGN)
        record_file = control_context.record_path()
        if record_file is None:
            raise StateUnavailableError("No agent-data root: there is no record to approve under")
    record_file = Path(record_file)
    directory = record_file.parent / AUTHORIZATIONS_DIR_NAME
    try:
        with control_context.held_epoch(record_file, request.target) as epoch:
            now = _now()
            if now >= request.expires_at:
                raise AuthorizationRefusedError(
                    VerdictReason.AUTH_EXPIRED, FixedCause(kind="request_expired")
                )
            grant = request.grant(
                switch_epoch=epoch,
                approver=Approver(by=identity, surface=surface, approved_at=now),
                issued_at=now,
            )
            _issue_grant(
                grant,
                directory=directory,
                peers=self_issued_directories() if peers is None else peers,
            )
    except control_context.RecordUnavailableError as exc:
        raise AuthorizationRefusedError(VerdictReason.RECORD_UNAVAILABLE) from exc
    return grant
