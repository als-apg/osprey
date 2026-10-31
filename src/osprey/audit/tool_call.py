"""The ``tool_call`` audit surface: what one osprey tool call ran against.

The default records (:mod:`osprey.audit.envelope`) say what was decided. This
surface says what a call ran against and what let it through: the control
target and its generation at entry, the approval answer, the facts the tool
noted while it ran (a write's limits verdict and old values, its attribution
stamps), and how long it took. It is joined to the default record and to the
agent harness's own transcript by ``tool_use_id``.

It carries no arguments and no result: the agent's transcript already holds
both, once. Nor does it repeat the decision and reason, which the default
record holds. The MCP audit middleware writes one entry per call that has any
of these facts (:func:`has_facts`), always, through
:func:`osprey.audit.writer.append_record`.
"""

from __future__ import annotations

from typing import Any

from osprey.audit.envelope import terminal_session

__all__ = [
    "SURFACE_TOOL_CALL",
    "build_record",
    "has_facts",
]

#: The surface this record is written on.
SURFACE_TOOL_CALL = "tool_call"


def has_facts(
    *,
    approval: dict[str, Any] | None,
    target: str | None,
    generation: int | None,
    facts: dict[str, Any],
) -> bool:
    """Whether a call has anything this surface records beyond its duration."""
    return approval is not None or target is not None or generation is not None or bool(facts)


def build_record(
    *,
    ts: str,
    actor: str,
    posture: str,
    posture_source: str,
    session: str | None,
    session_id: str | None,
    tool_use_id: str | None,
    server: str,
    subject: str,
    approval: dict[str, Any] | None,
    target: str | None,
    generation: int | None,
    facts: dict[str, Any],
    duration_ms: float,
) -> dict[str, Any]:
    """One record, in its documented key order.

    Every identifier comes first, then ``duration_ms``, then the facts. A
    *session* of ``None`` becomes :func:`~osprey.audit.envelope.terminal_session`.
    """
    return {
        "ts": ts,
        "surface": SURFACE_TOOL_CALL,
        "actor": actor,
        "posture": posture,
        "posture_source": posture_source,
        "session": session if session is not None else terminal_session(),
        "session_id": session_id,
        "tool_use_id": tool_use_id,
        "duration_ms": duration_ms,
        "server": server,
        "subject": subject,
        "approval": approval,
        "target": target,
        "generation": generation,
        "facts": facts,
    }
