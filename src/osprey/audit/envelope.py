"""The unified audit envelope — one record shape for every safety decision.

Osprey's P1/P2 work left two separate refusal ledgers, each with its own ad-hoc
record shape: ``readonly-refusals.jsonl`` (the python executor) and
``protected-writes.jsonl`` (the framework writers). Neither could answer the
question an operator actually asks — *who did what, under which posture, as
which role* — because neither carried the actor, the posture, or the session
that governed the decision.

This module defines the single envelope that replaces both, and that MCP tool
calls, HTTP mutations, hook decisions and logins all emit:

``{ts, surface, actor, posture, posture_source, session, subject, decision,
reason, detail?, role?, source?}``

**What may go in an envelope.** Every field carries an *identifier* or a
*config key* — a surface name, a username, a tool name, a dotted config key, a
short machine-ish reason. No field ever carries a config **value**, a prompt,
an agent message, a credential, or any other payload the record is merely
*about*. The one deliberate exception is :attr:`AuditEnvelope.source` on the
executor surface, where the refused code *is* the artifact under audit and a
record without it would be an alert rather than an audit trail; that field is
therefore refused on every other surface (:data:`SURFACE_EXECUTOR`).

**Provenance is stated, never inferred.** :attr:`~AuditEnvelope.posture_source`
is a closed set (:data:`POSTURE_SOURCES`) that says *how* the posture in the
record was established — a spawn-time env marker, a live toggle, an app-side
constant, or bare process context. It is passed explicitly by the caller and
validated here; deriving it from the posture *value* would turn an honest
"we do not know why" into a confident guess.

**Bounded by construction.** Every string field is bounded (see the ``MAX_*``
constants) and the bounding happens in :meth:`AuditEnvelope.__post_init__`, so
there is no way to build an unbounded envelope and no bound the writer has to
re-apply. ``detail`` is truncated silently; ``source`` is truncated *and*
flagged via :attr:`~AuditEnvelope.source_truncated`, because a truncated script
that did not say so would read like the whole script.

This module is **schema only** — it writes nothing. The writer that turns an
envelope into a line of ``var/audit/<identity>/<surface>.jsonl`` lives beside
it in ``osprey.audit.writer``. Like ``osprey.utils.sensitive_env``, it stays a
stdlib-only leaf: the writer, the MCP middleware, the HTTP layer and the hooks
all import it, and none of them may inherit an import cycle by doing so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar

# --------------------------------------------------------------------------
# Closed sets
# --------------------------------------------------------------------------

#: The posture in a record was established at spawn time and cannot have
#: changed since — the child was launched with an explicit posture marker.
POSTURE_SOURCE_SPAWN = "spawn"

#: The posture was read live from the posture store at decision time, so it
#: reflects a runtime toggle rather than the value the session started with.
POSTURE_SOURCE_LIVE = "live"

#: The record was emitted by something that is not a session child at all —
#: the HTTP layer, the auth sidecar — which stamps its own app-side posture.
POSTURE_SOURCE_APP = "app"

#: No posture marker was present and the emitter fell back to bare process
#: context (a dispatch worker, a CLI run, a container-level execution mode).
#: The honest "we were not told" value; never a stand-in for the others.
POSTURE_SOURCE_PROCESS = "process"

#: The closed set of legal :attr:`AuditEnvelope.posture_source` values. An
#: envelope carrying anything else is a programming error and is refused at
#: construction — a record whose provenance is unrecognised is worse than no
#: record, because it reads as authoritative.
POSTURE_SOURCES: tuple[str, ...] = (
    POSTURE_SOURCE_SPAWN,
    POSTURE_SOURCE_LIVE,
    POSTURE_SOURCE_APP,
    POSTURE_SOURCE_PROCESS,
)

#: The one surface allowed to carry :attr:`AuditEnvelope.source`: the python
#: executor, where the refused code is the artifact being audited. Every other
#: surface logs identifiers and config keys only, so ``source`` there would be
#: a payload leak rather than evidence.
SURFACE_EXECUTOR = "executor"

#: Canonical spellings for :attr:`AuditEnvelope.decision`. Unlike
#: :data:`POSTURE_SOURCES` this set is *not* enforced — a future surface may
#: need a word these two do not cover — but an emitter with nothing better to
#: say should use one of these rather than invent a synonym.
DECISION_ALLOWED = "allowed"
DECISION_REFUSED = "refused"
DECISIONS: tuple[str, ...] = (DECISION_ALLOWED, DECISION_REFUSED)

# --------------------------------------------------------------------------
# Bounds
# --------------------------------------------------------------------------

#: Characters kept per identifier-shaped field (surface, actor, posture,
#: session, subject, decision, reason, role). Generous for every real
#: identifier and dotted config key, bounded so a caller-supplied name cannot
#: inflate the ledger. Over-long values are truncated silently: an identifier
#: long enough to hit this is already malformed, and flagging it would add a
#: field to every record for a case that should not occur.
MAX_FIELD_CHARS = 256

#: Characters kept in the free-form :attr:`AuditEnvelope.detail` field.
#: Truncated silently and deliberately un-flagged — ``detail`` is
#: supplementary context, so a cut one is still a usable record and the flag
#: would cost a key on records that carry no evidentiary payload.
MAX_DETAIL_CHARS = 1024

#: Characters of offending source kept on the executor surface. Matches the
#: bound the P1 readonly ledger used, for the same reason: enough to hold a
#: whole ordinary script, bounded so a pathological submission cannot inflate
#: the log without limit. A cut record says so via
#: :attr:`AuditEnvelope.source_truncated`.
MAX_SOURCE_CHARS = 8000


def utc_timestamp() -> str:
    """Return the current UTC time in Osprey's audit timestamp format.

    Second resolution with a literal ``Z``, matching the two P1/P2 ledgers this
    envelope subsumes, so records from before and after the migration sort and
    parse the same way. Deliberately not :meth:`~datetime.datetime.isoformat`,
    whose microseconds and ``+00:00`` offset would break that continuity.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _truncate(value: str, limit: int) -> tuple[str, bool]:
    """Return *value* bounded to *limit* characters, and whether it was cut."""
    if len(value) <= limit:
        return value, False
    return value[:limit], True


@dataclass(frozen=True, kw_only=True)
class AuditEnvelope:
    """One safety decision, in the shape every Osprey audit surface emits.

    Frozen, so a record cannot be edited between the decision that produced it
    and the write that stores it, and keyword-only, so a field can never be
    supplied positionally into the wrong slot — ``actor`` and ``subject`` are
    both bare strings and a transposed pair would be silently plausible.

    Validation and bounding happen in :meth:`__post_init__`, which is what
    makes this type the schema: every constructed envelope is already legal and
    already bounded, and the writer never has to second-guess one.

    :param surface: Which surface decided — ``executor``, ``setup_patch``,
        ``http_config``, ``scaffold_restore``, an MCP server name, and so on.
        Kept verbatim even when the writer routes the record elsewhere (the
        maintenance-phase marker), so the record still says who decided.
    :param actor: Whose action it was, from ``identity.acting_identity()``.
    :param posture: The session's true posture at decision time — ``sandbox``
        or ``writes`` as the rest of Osprey spells them.
    :param posture_source: How *posture* was established; one of
        :data:`POSTURE_SOURCES`. Explicit, never derived from *posture*.
    :param session: The posture-store key that governed this record — the chat
        id for chat children, the spawn-time key for PTY sessions. ``None``
        only where no posture-store key exists; *posture_source* already says
        why. A top-level field, never smuggled into *detail*, so toggle events
        and tool records join on one key.
    :param subject: What was acted on — an MCP tool name, a dotted config key,
        a project-relative path, a login subject. An identifier, never a value.
    :param decision: What happened; see :data:`DECISIONS`.
    :param reason: Short machine-ish reason (``protected_key``,
        ``posture``, ``role_mismatch``).
    :param detail: Optional supplementary context, bounded to
        :data:`MAX_DETAIL_CHARS`. Identifiers and config keys only — never a
        config value, a prompt, or an agent message.
    :param role: The role the actor held, where the decision was identity-bound.
    :param source: The offending code, on :data:`SURFACE_EXECUTOR` only.
        Bounded to :data:`MAX_SOURCE_CHARS`; a :class:`ValueError` on any other
        surface, which is the schema's guard against payload leaking into the
        ledger.
    :param ts: Record timestamp; defaults to :func:`utc_timestamp`.
    :raises ValueError: on an unknown *posture_source*, an empty required
        field, or *source* outside the executor surface. Emitters call this
        from inside the writer's own never-raises boundary, so a construction
        bug degrades the audit trail rather than failing the operation it
        describes.
    """

    surface: str
    actor: str
    posture: str
    posture_source: str
    session: str | None
    subject: str
    decision: str
    reason: str
    detail: str | None = None
    role: str | None = None
    source: str | None = None
    ts: str = field(default_factory=utc_timestamp)

    #: Set by :meth:`__post_init__` when *source* was cut to
    #: :data:`MAX_SOURCE_CHARS`. Derived, never caller-supplied: a flag the
    #: emitter could set independently of the text it describes would be able
    #: to lie about it.
    source_truncated: bool = field(init=False, default=False)

    #: Required fields, in envelope order after ``ts``. Named once so
    #: :meth:`__post_init__` and :meth:`to_dict` cannot drift apart, and so a
    #: test can assert the shape off the module's own declaration.
    REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "surface",
        "actor",
        "posture",
        "posture_source",
        "session",
        "subject",
        "decision",
        "reason",
    )

    def __post_init__(self) -> None:
        """Validate the closed sets and bound every field.

        Uses ``object.__setattr__`` because the dataclass is frozen: the
        normalized values are what the envelope was always meant to hold, and
        writing them here is what guarantees no other code path has to.
        """
        if self.posture_source not in POSTURE_SOURCES:
            raise ValueError(
                f"posture_source must be one of {POSTURE_SOURCES}, got {self.posture_source!r}"
            )

        for name in ("surface", "actor", "posture", "subject", "decision", "reason"):
            value = getattr(self, name)
            if not value:
                raise ValueError(f"audit envelope field {name!r} must not be empty")

        if self.source is not None and self.surface != SURFACE_EXECUTOR:
            raise ValueError(
                f"source is recorded on the {SURFACE_EXECUTOR!r} surface only, "
                f"not on {self.surface!r}"
            )

        for name in ("surface", "actor", "posture", "subject", "decision", "reason", "role"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _truncate(value, MAX_FIELD_CHARS)[0])

        if self.session is not None:
            object.__setattr__(self, "session", _truncate(self.session, MAX_FIELD_CHARS)[0])

        if self.detail is not None:
            object.__setattr__(self, "detail", _truncate(self.detail, MAX_DETAIL_CHARS)[0])

        if self.source is not None:
            body, truncated = _truncate(self.source, MAX_SOURCE_CHARS)
            object.__setattr__(self, "source", body)
            object.__setattr__(self, "source_truncated", truncated)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON object this envelope serializes to.

        Required fields are always present, ``session`` included — it is
        emitted as ``null`` rather than dropped, so every line of a surface's
        JSONL carries the same columns and a missing key can only mean an old
        record. Optional fields are omitted when unset, and
        ``source_truncated`` appears only when it is true, matching the P1
        ledger it replaces.
        """
        record: dict[str, Any] = {"ts": self.ts}
        for name in self.REQUIRED_FIELDS:
            record[name] = getattr(self, name)
        if self.detail is not None:
            record["detail"] = self.detail
        if self.role is not None:
            record["role"] = self.role
        if self.source is not None:
            record["source"] = self.source
        if self.source_truncated:
            record["source_truncated"] = True
        return record
