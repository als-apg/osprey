"""Shared wording for holders that stay pinned to the deployment baseline.

Most of the system follows the session's control-system target. A few holders
cannot, because they are bound to something the switch does not move:

* the **Phoebus bridge** talks to one running Phoebus product, whose PV context
  was established when that product started — a session-level target switch
  does not re-address it;
* the **health runtime** reports on the deployment as configured, not on
  whatever a session has selected for itself.

A holder in that position must never be silent about it. Two agent-facing
strings come out of this module, and both are rendered from the same computed
facts so a refusal and a label can never disagree:

* :func:`baseline_pinned_line` — the informational line a *read* tool prepends
  to its normal output while the session is switched away from the baseline;
* :func:`baseline_refusal` — the message + suggestions an *action* tool refuses
  with, so a write is never quietly applied to the target the session left.

Both render nothing (``None``) while the session is on the baseline, which is
what keeps unswitched output byte-identical to what it was before this module
existed.

Why this module lives here
--------------------------
:mod:`osprey.mcp_server.control_system.target_state` owns the state-file
contract, and this module is the one place that turns that record plus the
deployment config into the sentence a user reads. Keeping the two together
means there is exactly one in-venv answer to "which target is this session on,
and which one is this deployment's baseline". The phoebus MCP server and the
health runtime are each a *different process* from the controls server that
writes the state file; both import this module rather than restating the rule.
Claude Code hooks, which run outside the venv and cannot import any of this,
restate it stdlib-only — see the ``target_state`` docstring for that contract.

Reuse
-----
The API is deliberately holder-agnostic so the HealthRuntime row can be built
from it without a second implementation:

* :func:`resolve_target_situation` returns the three facts
  (``session_target`` / ``baseline_target`` / ``switched``) and never raises;
* every renderer takes a *subject* (``"Phoebus"``, ``"HealthRuntime"``, …) and
  an optional pre-computed :class:`TargetSituation`, so a caller that needs
  several strings resolves the state once and renders many.

A holder that wants different phrasing than :func:`baseline_pinned_line` should
still take its facts from :func:`resolve_target_situation` — re-deriving the
session target from config is how two holders start telling a user two
different things.

Failure posture
---------------
Every failure mode collapses to "on the baseline": absent, unreadable, or
corrupt state; a state file owned by another session; two ambiguous candidates;
an unreadable config. That means a broken read produces no refusal and no
label rather than a wrong one — the same fail-closed direction the state file's
own readers take.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.target_state import TARGET_LIVE, TARGET_VA
from osprey_connectors import types as connector_types
from osprey_connectors.types import resolve_control_system_type
from osprey_connectors.workspace import load_osprey_config

logger = logging.getLogger("osprey.mcp_server.control_system.target_banner")

#: Subject string for the phoebus holder. Spelled once so the refusal and the
#: read-tool label cannot drift apart across two modules.
PHOEBUS_SUBJECT = "Phoebus"

#: Error type carried by a baseline-pinned refusal envelope. Machine-readable
#: category shared by every holder that refuses for this reason, so a caller can
#: recognise "you are switched away" without matching on prose.
BASELINE_REFUSAL_ERROR_TYPE = "target_switched"

__all__ = [
    "BASELINE_REFUSAL_ERROR_TYPE",
    "PHOEBUS_SUBJECT",
    "TargetSituation",
    "baseline_pinned_line",
    "baseline_refusal",
    "prepend_line",
    "resolve_baseline_target",
    "resolve_session_target",
    "resolve_target_situation",
]


@dataclass(frozen=True)
class TargetSituation:
    """The two targets a baseline-pinned holder has to talk about.

    Attributes:
        session_target: The target this session has selected (``live`` / ``va``).
        baseline_target: The target the deployment config declares.
    """

    session_target: str
    baseline_target: str

    @property
    def switched(self) -> bool:
        """Whether the session has moved off the deployment baseline."""
        return self.session_target != self.baseline_target


# -- resolution ------------------------------------------------------------


def resolve_baseline_target() -> str:
    """The deployment baseline: ``va`` for a virtual accelerator, else ``live``.

    The control-system type comes from
    :func:`osprey_connectors.types.resolve_control_system_type` — the same
    resolver the connector factory uses. Re-implementing that mapping here
    would be a second opinion about what the deployment is, which is exactly
    the bug this module exists to prevent.
    """
    config = load_osprey_config()
    section = config.get("control_system") if isinstance(config, dict) else None
    cs_type = resolve_control_system_type(section)
    return TARGET_VA if cs_type == connector_types.VIRTUAL_ACCELERATOR else TARGET_LIVE


def _int_or_none(value: object) -> int | None:
    """Coerce a record field to ``int``, or ``None`` when it is not a number."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _live_records() -> list[dict]:
    """Every state record whose owning server process is still running."""
    try:
        entries = sorted(target_state.state_dir().glob(target_state.STATE_FILE_GLOB))
    except OSError:
        return []

    records: list[dict] = []
    for entry in entries:
        record = target_state.read_file(entry)
        if record is None:
            continue
        server_pid = _int_or_none(record.get("server_pid"))
        if server_pid is None or not target_state.is_process_alive(server_pid):
            continue
        records.append(record)
    return records


def resolve_session_target(baseline_target: str) -> str:
    """The target this session is on, or *baseline_target* when it is unknowable.

    A checkout can host several sessions at once, each with its own controls
    server and so its own state file. This process is not that server — the
    phoebus server and the health runtime are separate MCP processes — but it
    *is* a child of the same Claude Code process, so the record to trust is the
    one whose ``owner_ppid`` is this process's parent.

    Zero matches (no session has switched, or the switch happened under a
    different parent) and more than one match (ambiguous ownership) both mean
    the same thing: no answer. Both fall back to the baseline, so an unknown
    state produces no refusal and no label rather than a guess.
    """
    own_parent = os.getppid()
    matches = [r for r in _live_records() if _int_or_none(r.get("owner_ppid")) == own_parent]
    if len(matches) != 1:
        if matches:
            logger.debug("Ambiguous target state: %d records own ppid %s", len(matches), own_parent)
        return baseline_target
    target = matches[0].get("target")
    if target not in target_state.TARGET_NAMES:
        logger.debug("Target state names an unknown target %r; using baseline", target)
        return baseline_target
    return str(target)


def resolve_target_situation() -> TargetSituation:
    """Resolve both targets. Never raises; every failure reads as "on baseline"."""
    try:
        baseline = resolve_baseline_target()
    except Exception:  # pragma: no cover - config layer is defensive already
        logger.debug("Could not resolve the deployment baseline target", exc_info=True)
        return TargetSituation(session_target=TARGET_LIVE, baseline_target=TARGET_LIVE)

    try:
        session = resolve_session_target(baseline)
    except Exception:
        logger.debug("Could not resolve the session target; assuming baseline", exc_info=True)
        session = baseline

    return TargetSituation(session_target=session, baseline_target=baseline)


# -- rendering -------------------------------------------------------------


def baseline_pinned_line(subject: str, situation: TargetSituation | None = None) -> str | None:
    """The informational line a baseline-pinned read tool prepends, or ``None``.

    ``None`` — not an empty string — while the session is on the baseline, so a
    caller cannot accidentally prepend a blank line to unswitched output.

    Args:
        subject: The holder speaking, e.g. ``"Phoebus"``.
        situation: Pre-resolved facts; resolved here when omitted.
    """
    situation = resolve_target_situation() if situation is None else situation
    if not situation.switched:
        return None
    return (
        f"{subject} is pinned to the deployment baseline "
        f"({situation.baseline_target}); the session target is {situation.session_target}"
    )


def baseline_refusal(
    subject: str,
    action: str,
    situation: TargetSituation | None = None,
) -> tuple[str, list[str]] | None:
    """Refusal message + suggestions for an action tool, or ``None`` on baseline.

    The message opens with the same sentence :func:`baseline_pinned_line`
    renders, so a user who has already seen the label on a read tool recognises
    the refusal as the same fact rather than a new one.

    Args:
        subject: The holder speaking, e.g. ``"Phoebus"``.
        action: What was refused, as a capitalised noun phrase — e.g.
            ``"Driving a Phoebus widget"``.
        situation: Pre-resolved facts; resolved here when omitted.

    Returns:
        ``(message, suggestions)``, or ``None`` when nothing is refused.
    """
    situation = resolve_target_situation() if situation is None else situation
    line = baseline_pinned_line(subject, situation)
    if line is None:
        return None
    message = (
        f"{line}. {action} would act on the '{situation.baseline_target}' target, "
        f"not the '{situation.session_target}' target this session is on."
    )
    suggestions = [
        f"Switch the session back to the deployment baseline: "
        f"control_target_set(target='{situation.baseline_target}').",
        f"Or act on the '{situation.session_target}' target through the control-system "
        f"tools, which follow the session target.",
    ]
    return message, suggestions


def prepend_line(line: str | None, payload: str) -> str:
    """Put *line* above *payload*, or return *payload* untouched when there is none.

    The untouched branch is the contract that keeps an unswitched tool's output
    byte-identical to what it produced before the holder was labelled.
    """
    return payload if not line else f"{line}\n{payload}"
