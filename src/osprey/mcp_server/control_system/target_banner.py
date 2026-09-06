"""Shared wording for holders that stay pinned to the deployment baseline.

Most of the system follows the deployment's control-system target. A few holders
cannot, because they are bound to something the switch does not move:

* the **Phoebus bridge** talks to one running Phoebus product, whose PV context
  was established when that product started — a target switch does not
  re-address it;
* the **health runtime** reports on the deployment as configured, not on
  whatever target has since been selected.

A holder in that position must never be silent about it. Two agent-facing
strings come out of this module, and both are rendered from the same computed
facts so a refusal and a label can never disagree:

* :func:`baseline_pinned_line` — the informational line a *read* tool prepends
  to its normal output while the deployment is switched away from the baseline;
* :func:`baseline_refusal` — the message + suggestions an *action* tool refuses
  with, so a write is never quietly applied to the target the deployment left.

Both render nothing (``None``) while the deployment is on the baseline, which is
what keeps unswitched output byte-identical to what it was before this module
existed.

Why this module lives here
--------------------------
:mod:`osprey_connectors.control_context` owns the record contract, and this
module is the one place that turns that record plus the deployment config into
the sentence a user reads. Keeping the two together means there is exactly one
in-venv answer to "which target is this deployment on, and which one is its
baseline". The phoebus MCP server and the health runtime are each a *different
process* from the controls server, and both import this module rather than
restating the rule. Claude Code hooks, which run outside the venv and cannot
import any of this, restate it stdlib-only — see the ``control_context``
docstring for that contract.

Reuse
-----
The API is deliberately holder-agnostic so the HealthRuntime row can be built
from it without a second implementation:

* :func:`resolve_target_situation` returns the three facts
  (``control_target`` / ``baseline_target`` / ``switched``) and never raises;
* every renderer takes a *subject* (``"Phoebus"``, ``"HealthRuntime"``, …) and
  an optional pre-computed :class:`TargetSituation`, so a caller that needs
  several strings resolves the state once and renders many.

A holder that wants different phrasing than :func:`baseline_pinned_line` should
still take its facts from :func:`resolve_target_situation` — re-deriving the
deployment's target from config is how two holders start telling a user two
different things.

A surface that needs several published facts at once — the target, its label,
the switch outcome, the reachability sweep — reads
:func:`osprey_connectors.control_context.read_record` directly rather than
asking here fact by fact. This module renders wording; the record is the source.

Failure posture
---------------
Every failure mode collapses to "on the baseline": no record, an unreadable or
corrupt one, a payload from another schema, an unreadable config. That means a
broken read produces no refusal and no label rather than a wrong one — the same
fail-closed direction the record's own reader takes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from osprey.mcp_server.control_system.target_state import TARGET_LIVE
from osprey_connectors import control_context
from osprey_connectors import types as connector_types
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
    "resolve_control_target",
    "resolve_target_situation",
]


@dataclass(frozen=True)
class TargetSituation:
    """The two targets a baseline-pinned holder has to talk about.

    Attributes:
        control_target: The target the deployment is on (``live`` / ``va``).
        baseline_target: The target the deployment config declares.
    """

    control_target: str
    baseline_target: str

    @property
    def switched(self) -> bool:
        """Whether the deployment has moved off its baseline."""
        return self.control_target != self.baseline_target


# -- resolution ------------------------------------------------------------


def resolve_baseline_target() -> str:
    """The deployment baseline: ``va`` for a virtual accelerator, else ``live``.

    The mapping comes from :func:`osprey_connectors.types.baseline_target` —
    the same predicate the connector-host supervisor and the switch-capability
    check read. Re-implementing it here would be a second opinion about what the
    deployment is, which is exactly the bug this module exists to prevent.
    """
    config = load_osprey_config()
    section = config.get("control_system") if isinstance(config, dict) else None
    return connector_types.baseline_target(section)


def resolve_control_target(baseline_target: str) -> str:
    """The deployment's current target, or *baseline_target* when unknowable.

    Read from :func:`osprey_connectors.control_context.read_record` — the one
    record every surface reads, so a refusal, a label and a chip cannot name
    three different machines. No record, an unreadable or corrupt one, and a
    payload from another schema all answer the same way: the baseline, so an
    unknown state produces no refusal and no label rather than a guess.
    """
    record = control_context.read_record()
    return baseline_target if record is None else record.target


def resolve_target_situation() -> TargetSituation:
    """Resolve both targets. Never raises; every failure reads as "on baseline"."""
    try:
        baseline = resolve_baseline_target()
    except Exception:  # pragma: no cover - config layer is defensive already
        logger.debug("Could not resolve the deployment baseline target", exc_info=True)
        return TargetSituation(control_target=TARGET_LIVE, baseline_target=TARGET_LIVE)

    try:
        control_target = resolve_control_target(baseline)
    except Exception:
        logger.debug("Could not resolve the deployment's target; assuming baseline", exc_info=True)
        control_target = baseline

    return TargetSituation(control_target=control_target, baseline_target=baseline)


# -- rendering -------------------------------------------------------------


def baseline_pinned_line(subject: str, situation: TargetSituation | None = None) -> str | None:
    """The informational line a baseline-pinned read tool prepends, or ``None``.

    ``None`` — not an empty string — while the deployment is on the baseline, so a
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
        f"({situation.baseline_target}); the deployment is on the "
        f"{situation.control_target} target"
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
        f"not the '{situation.control_target}' one."
    )
    suggestions = [
        f"Switch the deployment back to its baseline target: "
        f"control_target_set(target='{situation.baseline_target}').",
        f"Or act on the '{situation.control_target}' target through the control-system "
        f"tools, which follow the deployment's target.",
    ]
    return message, suggestions


def prepend_line(line: str | None, payload: str) -> str:
    """Put *line* above *payload*, or return *payload* untouched when there is none.

    The untouched branch is the contract that keeps an unswitched tool's output
    byte-identical to what it produced before the holder was labelled.
    """
    return payload if not line else f"{line}\n{payload}"
