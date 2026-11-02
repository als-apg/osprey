"""What a control session's write answers, and the audit detail it files.

:class:`~osprey.runtime.session.ControlSession` answers every write with a
:class:`SessionWriteResult` built from the names here: an outcome, the audit
reason behind it and the sentence the client shows.

A refusal names its reason on the vocabulary every other write path uses, so
a panel's chip, the agent's hook and a refused write say the same thing:

* a refusal the execution verdict decided names that verdict's reason, a
  member of :data:`osprey_connectors.refusals.REASONS`, and its message is
  :func:`refusal_message` — the verdict's phrase, naming the kind of machine,
  with the detail the verdict adds (which hold was raised);
* a refusal the connector decided names its
  :class:`~osprey_connectors.errors.RefusalReason` and carries the connector's
  own words;
* the session's own states keep their own short texts: no record to write
  under, no connector to write through, a write to the channel still in
  flight, an audit ledger that is down. They are states of this process, not
  reasons a login may not write.

:func:`detail_tokens` is the ``detail`` of the write's audit record.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osprey_connectors.posture_store import WriteVerdict

#: A write's outcome, as the panel's write contract names it.
OUTCOME_REFUSED = "refused"
OUTCOME_TARGET_CHANGED = "target_changed"
OUTCOME_SWITCH_IN_PROGRESS = "switch_in_progress"
OUTCOME_CONFIRMED = "confirmed"
OUTCOME_MISMATCH = "mismatch"
OUTCOME_UNCONFIRMED = "unconfirmed"
OUTCOME_UNKNOWN = "unknown"

#: The verdict reasons the session answers for its own binding, spelled as
#: :data:`osprey_connectors.refusals.REASONS` spells them: the binding moved
#: under the write (the record, the switch position, or a switch in flight),
#: and the switch state could not be checked.
REASON_SWITCHING = "switching"
REASON_RECORD_UNAVAILABLE = "record_unavailable"

#: The connector's limits refusal, as :class:`~osprey_connectors.errors.RefusalReason`
#: spells it: the net's own violation and the monitor's are one reason.
REASON_LIMITS = "LIMITS"

#: The session's own states, and the audit reasons of a write that reached the connector.
REASON_CONTEXT_UNAVAILABLE = "control_context_unavailable"
REASON_AUDIT_UNAVAILABLE = "audit_unavailable"
REASON_WRITE_IN_PROGRESS = "write_in_progress"
REASON_WRITE_ATTEMPT = "write_attempt"
REASON_WRITE_LANDED = "write_landed"
REASON_WRITE_UNCONFIRMED = "write_unconfirmed"

#: The text each of the session's own states answers with, which is the
#: envelope ``detail`` the client shows verbatim.
MESSAGE_NO_RECORD = "no control target record; nothing written"
MESSAGE_UNSERVABLE = "this control target cannot be served here; nothing written"
MESSAGE_AUDIT_UNAVAILABLE = "the audit ledger is unavailable; nothing written"
MESSAGE_WRITE_IN_PROGRESS = "a write to this channel is still in progress; nothing written"
MESSAGE_WRITTEN = "written"
MESSAGE_UNCONFIRMED = "sent, not verified"
MESSAGE_UNKNOWN = "outcome unknown — check the readback"


#: A token the request did not name.
ABSENT = "-"


@dataclass(frozen=True)
class SessionWriteResult:
    """How a write ended. Never an exception for an outcome.

    Attributes:
        outcome: The outcome's name.
        reason: The audit reason behind it.
        message: The sentence for whoever asked.
        observed_value: What the channel was seen to hold afterwards, when a
            confirming read was made.
    """

    outcome: str
    reason: str
    message: str
    observed_value: Any = None


#: The audit ``decision`` of a write record.
DECISION_ALLOWED = "allowed"
DECISION_REFUSED = "refused"

#: Outcomes whose record is a refusal; every other outcome is ``allowed``.
REFUSAL_OUTCOMES = frozenset({OUTCOME_REFUSED, OUTCOME_TARGET_CHANGED, OUTCOME_SWITCH_IN_PROGRESS})


def refusal_message(verdict: WriteVerdict) -> str:
    """The sentence a refused execution verdict answers with.

    The verdict's phrase for its reason and kind of machine
    (:func:`~osprey_connectors.refusals.refusal_phrase`), then the detail the
    verdict adds, such as which hold was raised. Raises ``ValueError`` for a
    verdict that refuses nothing.
    """
    from osprey_connectors.refusals import refusal_phrase

    if verdict.allowed or verdict.reason is None:
        raise ValueError("not a refusal")
    phrase = refusal_phrase(verdict.reason, verdict.kind)
    return f"{phrase} {verdict.detail}" if verdict.detail else phrase


def failed_result(exc: Any) -> SessionWriteResult:
    """The outcome for a raised ``ChannelWriteFailedError``."""
    outcome = getattr(exc, "outcome", None)
    observed = getattr(exc, "observed_value", None)
    if outcome == "mismatch":
        return SessionWriteResult(
            outcome=OUTCOME_MISMATCH,
            reason=REASON_WRITE_UNCONFIRMED,
            message=f"written; the channel holds {observed}",
            observed_value=observed,
        )
    if outcome in ("unconfirmed", "unrequested"):
        return SessionWriteResult(
            outcome=OUTCOME_UNCONFIRMED,
            reason=REASON_WRITE_UNCONFIRMED,
            message=MESSAGE_UNCONFIRMED,
            observed_value=observed,
        )
    # ``failed``, or no outcome at all: both read as unknown.
    return SessionWriteResult(
        outcome=OUTCOME_UNKNOWN, reason=REASON_WRITE_UNCONFIRMED, message=MESSAGE_UNKNOWN
    )


def origin_token(origin: Any, name: str) -> str:
    """One origin field as a ``detail`` token value; ``-`` when the request named none."""
    if origin is None:
        return ABSENT
    value = origin.get(name) if isinstance(origin, Mapping) else getattr(origin, name, None)
    if value is None:
        return ABSENT
    return identifier_token(value)


#: Characters kept of each client-named identifier token (channel, screen,
#: rev), so the whole ``detail`` stays inside the envelope's bound and the
#: value tokens after them are never cut away by a long screen name.
IDENTIFIER_TOKEN_CHARS = 128


#: The channel's own bound: four times any real EPICS name (60 characters), so
#: a real channel is never cut, and still inside the envelope's bound with the
#: other tokens.
CHANNEL_TOKEN_CHARS = 256


def identifier_token(value: Any, limit: int = IDENTIFIER_TOKEN_CHARS) -> str:
    """*value* as one token: whitespace runs joined by ``_``, bounded; ``-`` when empty."""
    text = "_".join(str(value).split())[:limit]
    return text or ABSENT


def detail_tokens(address: str, value: Any, origin: Any) -> str:
    """The audit record's ``detail``: ``channel= screen= rev= ca_user= ca_host= value_len= value=``.

    ``value=`` comes last and is cut at
    :data:`~osprey.audit.envelope.SESSION_VALUE_CHARS`, so a long value never
    hides ``screen=`` and ``rev=`` (the envelope's documented exception for
    the session-write value tokens).
    """
    from osprey.audit.call import write_stamps
    from osprey.audit.envelope import SESSION_VALUE_CHARS

    stamps = write_stamps()
    text = str(value)
    channel = identifier_token(address, CHANNEL_TOKEN_CHARS)
    return " ".join(
        (
            f"channel={channel}",
            f"screen={origin_token(origin, 'screen')}",
            f"rev={origin_token(origin, 'rev')}",
            f"ca_user={stamps.get('ca_user') or ABSENT}",
            f"ca_host={stamps.get('ca_host') or ABSENT}",
            f"value_len={len(text)}",
            f"value={text[:SESSION_VALUE_CHARS]}",
        )
    )
