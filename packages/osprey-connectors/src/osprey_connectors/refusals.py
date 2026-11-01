"""Why a write is refused, and who can lift the refusal, as one importable contract.

A refusal carries a *reason* (a member of :data:`REASONS`) and a *phrase* that says
what stopped the write and ends with who can lift it. :func:`refusal_text` is the
phrase plus a trailer line ``[osprey-refusal: <reason>]``; every refusal a write
path returns to the agent ends with it, so a reader recovers the reason without
parsing the words. The package's verdict, the connector
monitor and the stdlib-only hook twin all speak these words; the hook twin restates
the tables and a parity test pins them to this module.

No phrase names a config key, a file, an environment variable or a build command:
the reader of a refusal is an operator or an agent, not the person who edits the
deployment. :data:`STORE_PATH_MARKERS` holds the path spellings of the posture store
(the directory part and the record's file name) so that a text meant for an agent can
be scanned for them.

Reasons are added by the phase that first consumes them; none is speculative.
"""

from __future__ import annotations

__all__ = ["KIND_WORDS", "REASONS", "STORE_PATH_MARKERS", "refusal_phrase", "refusal_text"]

#: The closed set of refusal reasons.
REASONS: tuple[str, ...] = (
    "readonly_run",
    "tier_never",
    "target_unknown",
    "owner_unknown",
    "owner_unscheduled",
    "owner_foreign",
    "record_unavailable",
    "switch_off",
    "launch_pinned",
    "switching",
    "stopped_since",
)

_PHRASES: dict[str, str] = {
    "readonly_run": (
        "Writes are refused because the deployment is running read-only; "
        "the chip cannot lift this — ask the deployment admin."
    ),
    "tier_never": (
        "Writes are refused because this login's write tier does not allow them; "
        "ask the deployment admin."
    ),
    "target_unknown": (
        "Writes are refused because the machine this connector writes to is not known; "
        "retry, then ask the deployment admin."
    ),
    "owner_unknown": (
        "Writes are refused because this login's write tier is not known; ask the deployment admin."
    ),
    "owner_unscheduled": (
        "Writes are refused because this scheduled or triggered job runs for no login, "
        "so no switch covers it; ask the deployment admin."
    ),
    "owner_foreign": (
        "Writes are refused because this machine's writes belong to another login; "
        "only that login can write here."
    ),
    "record_unavailable": (
        "Writes are refused because the write switch cannot be read; "
        "retry, then ask the deployment admin."
    ),
    "switch_off": ("Writes are off; turn it on from the chip in the web terminal."),
    "launch_pinned": (
        "Writes are refused because this run was started with writes held off; "
        "run it again without the pin."
    ),
    "switching": (
        "Writes are refused because the write gateway has not yet followed the switch; "
        "retry, then ask the deployment admin."
    ),
    "stopped_since": (
        "Writes are off because a stop came after the turn-on began; "
        "turn it on again from the chip."
    ),
}

#: The words a machine kind adds to a phrase.
KIND_WORDS: dict[str, str] = {
    "real": "the live machine",
    "rehearsal": "the rehearsal machine",
    "simulator": "the simulator",
}

#: Path spellings of the posture store: the state directory part and the record file.
STORE_PATH_MARKERS: tuple[str, ...] = ("control_target", "control_context.json")


def refusal_phrase(reason: str, kind: str | None = None) -> str:
    """The phrase for *reason*; a known *kind* names the machine it concerns.

    Raises ``ValueError`` for a reason outside :data:`REASONS`. An unknown or absent
    *kind* leaves the phrase as it is.
    """
    try:
        phrase = _PHRASES[reason]
    except KeyError:
        raise ValueError(f"unknown refusal reason: {reason!r}") from None
    words = KIND_WORDS.get(kind) if kind is not None else None
    return f"{phrase} (Concerns {words}.)" if words else phrase


def refusal_text(reason: str, kind: str | None = None, *, detail: str | None = None) -> str:
    """The refusal a write path returns: the phrase, then ``[osprey-refusal: <reason>]``.

    *detail* — what the refusing surface adds, such as which tool was refused —
    follows the phrase on its line. The trailer is its own last line, so it ends
    the text whatever the caller adds. Raises ``ValueError`` for a reason outside
    :data:`REASONS`, as :func:`refusal_phrase` does.
    """
    phrase = refusal_phrase(reason, kind)
    if detail:
        phrase = f"{phrase} {detail}"
    return f"{phrase}\n[osprey-refusal: {reason}]"
