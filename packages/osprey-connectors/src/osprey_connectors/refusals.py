"""Why a write is refused, and who can lift the refusal, as one importable contract.

Two closed sets live here. :class:`VerdictReason` is why the write verdict
refused: a member's value is the reason word, and its :attr:`~VerdictReason.phrase`
says what stopped the write and ends with who can lift it. :func:`refusal_text`
is the phrase plus a trailer line ``[osprey-refusal: <reason>]``; every refusal a
write path returns to the agent ends with it, so a reader recovers the reason
without parsing the words. The package's verdict, the connector monitor and the
stdlib-only hook twin all speak these words; the hook twin restates the table
and a parity test pins it to this module. :class:`RefusalReason` is the coarser
set a write result and a blocked write's error carry, with who refused and what
to do; a ``WRITES_DISABLED`` refusal narrows to the verdict's reason.

No phrase names a config key, a file, an environment variable or a build command:
the reader of a refusal is an operator or an agent, not the person who edits the
deployment. :data:`STORE_PATH_MARKERS` holds the path spellings of the posture store
(the directory part and the record's file name) so that a text meant for an agent can
be scanned for them.

Reasons are added by the phase that first consumes them; none is speculative.
This module imports nothing from the package, so every other module may read it.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final

from pydantic import BeforeValidator

__all__ = [
    "KIND_WORDS",
    "REASONS",
    "STORE_PATH_MARKERS",
    "RefusalReason",
    "VerdictReason",
    "VerdictReasonField",
    "refusal_phrase",
    "refusal_text",
]


class VerdictReason(StrEnum):
    """Why the write verdict refused a write, with the words an operator reads.

    Each member's value is the reason word a verdict, a write result and the
    hook twin carry; :attr:`phrase` is what it says. The members are declared
    in the verdict's own order of terms, which the hook twin restates and a
    parity test pins (:data:`REASONS`).
    """

    phrase: str

    def __new__(cls, value: str, phrase: str = "") -> VerdictReason:
        member = str.__new__(cls, value)
        member._value_ = value
        member.phrase = phrase
        return member

    READONLY_RUN = (
        "readonly_run",
        (
            "Writes are refused because the deployment is running read-only; "
            "the chip cannot lift this — ask the deployment admin."
        ),
    )
    TIER_NEVER = (
        "tier_never",
        (
            "Writes are refused because this login's write tier does not allow them; "
            "ask the deployment admin."
        ),
    )
    SESSION_HELD = (
        "session_held",
        "Writes are paused: this session has an unanswered hold the operator must answer.",
    )
    TARGET_UNKNOWN = (
        "target_unknown",
        (
            "Writes are refused because the machine this connector writes to is not known; "
            "retry, then ask the deployment admin."
        ),
    )
    OWNER_UNKNOWN = (
        "owner_unknown",
        "Writes are refused because this login's write tier is not known; ask the deployment admin.",
    )
    OWNER_UNSCHEDULED = (
        "owner_unscheduled",
        (
            "Writes are refused because this scheduled or triggered job runs for no login, "
            "so no switch covers it; ask the deployment admin."
        ),
    )
    OWNER_FOREIGN = (
        "owner_foreign",
        (
            "Writes are refused because this machine's writes belong to another login; "
            "only that login can write here."
        ),
    )
    RECORD_UNAVAILABLE = (
        "record_unavailable",
        (
            "Writes are refused because the write switch cannot be read; "
            "retry, then ask the deployment admin."
        ),
    )
    SWITCH_OFF = (
        "switch_off",
        "Writes are off; turn it on from the chip in the web terminal.",
    )
    LAUNCH_PINNED = (
        "launch_pinned",
        (
            "Writes are refused because this run was started with writes held off; "
            "run it again without the pin."
        ),
    )
    SWITCHING = (
        "switching",
        (
            "Writes are refused because the write gateway has not yet followed the switch; "
            "retry, then ask the deployment admin."
        ),
    )
    STOPPED_SINCE = (
        "stopped_since",
        (
            "Writes are off because a stop came after the turn-on began; "
            "turn it on again from the chip."
        ),
    )
    AUTH_UNREADABLE = (
        "auth_unreadable",
        (
            "Writes are refused because this program's write authorization cannot be read; "
            "retry, then ask the deployment admin."
        ),
    )
    AUTH_HOLDER = (
        "auth_holder",
        (
            "Writes are refused because this write authorization was approved for another "
            "program or login; ask the operator to approve one for this program."
        ),
    )
    AUTH_CHANNEL = (
        "auth_channel",
        (
            "Writes are refused because this channel is not in the program's write "
            "authorization; ask the operator to approve one that lists it."
        ),
    )
    AUTH_RANGE = (
        "auth_range",
        (
            "Writes are refused because this value is outside the range the program's write "
            "authorization allows; ask the operator to approve a wider range."
        ),
    )
    AUTH_EXHAUSTED = (
        "auth_exhausted",
        (
            "Writes are refused because the program has used every write its authorization "
            "allows; ask the operator to approve a new authorization."
        ),
    )
    AUTH_LEASE_LAPSED = (
        "auth_lease_lapsed",
        (
            "Writes are refused because the program holding this write authorization stopped "
            "renewing it; ask the operator to approve a new authorization."
        ),
    )
    AUTH_EXPIRED = (
        "auth_expired",
        (
            "Writes are refused because the program's write authorization has expired; "
            "ask the operator to approve a new authorization."
        ),
    )
    AUTH_REVOKED = (
        "auth_revoked",
        (
            "Writes are refused because the program's write authorization has ended: a stop "
            "came, or the switch is no longer at allowed; ask the operator to approve a new "
            "authorization."
        ),
    )
    AUTH_OVERLAP = (
        "auth_overlap",
        (
            "The authorization is refused because another program holds one on a channel of "
            "this machine it names; wait for that run to end, or stop it from the chip."
        ),
    )


#: The closed set of refusal reasons, in their declared order.
REASONS: Final[tuple[VerdictReason, ...]] = tuple(VerdictReason)


def _member(value: object) -> object:
    """*value* as a :class:`VerdictReason` member when it is the word of one.

    A strict model takes a member only, and a word arrives from a file, the
    connector-host IPC or a parent process's text; anything else is left for
    the field's own check to refuse.
    """
    if isinstance(value, str) and not isinstance(value, VerdictReason):
        try:
            return VerdictReason(value)
        except ValueError:
            raise ValueError(f"unknown refusal reason: {value!r}") from None
    return value


#: A :class:`VerdictReason` field that also takes the reason's word.
VerdictReasonField = Annotated[VerdictReason, BeforeValidator(_member)]

#: The words a machine kind adds to a phrase.
KIND_WORDS: Final[dict[str, str]] = {
    "real": "the live machine",
    "rehearsal": "the rehearsal machine",
    "simulator": "the simulator",
}

#: Path spellings of the posture store: the state directory part and the record file.
STORE_PATH_MARKERS: Final[tuple[str, ...]] = ("control_target", "control_context.json")


def _reason(reason: VerdictReason) -> VerdictReason:
    """*reason* as a member; a caller's word for one is taken for it.

    Raises:
        ValueError: *reason* is no reason's word.
    """
    member = _member(reason)
    if not isinstance(member, VerdictReason):
        raise ValueError(f"unknown refusal reason: {reason!r}")
    return member


def refusal_phrase(reason: VerdictReason, kind: str | None = None) -> str:
    """The phrase for *reason*; a known *kind* names the machine it concerns.

    Raises ``ValueError`` for a word that is no reason's. An unknown or absent
    *kind* leaves the phrase as it is: the kind is a word read from a report,
    and a phrase is never refused for it.
    """
    phrase = _reason(reason).phrase
    words = KIND_WORDS.get(kind) if kind is not None else None
    return f"{phrase} (Concerns {words}.)" if words else phrase


def refusal_text(
    reason: VerdictReason, kind: str | None = None, *, detail: str | None = None
) -> str:
    """The refusal a write path returns: the phrase, then ``[osprey-refusal: <reason>]``.

    *detail* — what the refusing surface adds, such as which tool was refused —
    follows the phrase on its line. The trailer is its own last line, so it ends
    the text whatever the caller adds. Raises ``ValueError`` for a word that is no
    reason's, as :func:`refusal_phrase` does.
    """
    member = _reason(reason)
    phrase = refusal_phrase(member, kind)
    if detail:
        phrase = f"{phrase} {detail}"
    return f"{phrase}\n[osprey-refusal: {member.value}]"


_REFERENCE_MONITOR = "the reference monitor"
_CONTROL_SYSTEM = "the control system"
_NEVER_SENT = (
    "This write was refused on policy grounds; it was never sent to the control system.",
)


class RefusalReason(StrEnum):
    """Why a write was refused, with who refused it and what to do about it.

    Each member's value is the reason word a result and an exception carry.
    :attr:`refuser` names who refused a write for the reason, and is ``None``
    for ``CHANNEL_NOT_FOUND``: nothing was denied, the address is not a
    channel. :attr:`remedy` holds the reason's own suggestion sentences, in
    the order a message lists them; it is empty for ``CHANNEL_NOT_FOUND``,
    whose answer is rendered as "not a channel".

    ``RAW_CLIENT_WRITE`` means user code called a control-system client
    library's put directly instead of going through
    ``osprey.runtime.write_channel``; nothing was sent.
    ``CONTROL_SYSTEM_REFUSED`` is the one reason under which the write was
    sent: the control system was asked and denied it.
    ``BATCH_REFUSED`` is a write of a batch that passed every check of its own
    and was not sent because another write of the batch was refused.
    """

    refuser: str | None
    remedy: tuple[str, ...]

    def __new__(
        cls, value: str, refuser: str | None = None, remedy: tuple[str, ...] = ()
    ) -> RefusalReason:
        member = str.__new__(cls, value)
        member._value_ = value
        member.refuser = refuser
        member.remedy = remedy
        return member

    WRITES_DISABLED = ("WRITES_DISABLED", _REFERENCE_MONITOR, _NEVER_SENT)
    LIMITS = ("LIMITS", _REFERENCE_MONITOR, _NEVER_SENT)
    VALIDATION_ERROR = ("VALIDATION_ERROR", _REFERENCE_MONITOR, _NEVER_SENT)
    CONTROL_SYSTEM_REFUSED = (
        "CONTROL_SYSTEM_REFUSED",
        _CONTROL_SYSTEM,
        (
            "The control system itself denied this write (access security); no value was written.",
            "Report it to the operator: write access for this channel is granted "
            "by the control system, not by OSPREY.",
        ),
    )
    RAW_CLIENT_WRITE = ("RAW_CLIENT_WRITE", _REFERENCE_MONITOR, _NEVER_SENT)
    BATCH_REFUSED = (
        "BATCH_REFUSED",
        _REFERENCE_MONITOR,
        (
            "No write of this batch was sent: another write in it was refused.",
            "Resolve that write's refusal and submit the batch again.",
        ),
    )
    CHANNEL_NOT_FOUND = ("CHANNEL_NOT_FOUND", None, ())
