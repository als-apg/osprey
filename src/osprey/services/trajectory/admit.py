"""The one gate every trajectory action passes before it may act.

:meth:`AdmitGate.admit` is synchronous and called twice over: by the action
worker before every action, and by the control path at admission. It either
returns an :class:`Admission` — the facts the action's audit record carries —
or raises :class:`AdmitRefused` with the code and HTTP status the door answers
with. The checks run in this order and the first that fails decides:

1. **The owner's name.** A missing owner, or a floor name — ``unknown``, or this
   process's own ``OSPREY_CONTROL_IDENTITY`` / ``OSPREY_AUDIT_IDENTITY`` —
   names nobody a write could belong to: 403 ``no_operator_identity``.
2. **The deployment ceiling.** :func:`~osprey_connectors.types.target_writes_enabled`
   for the service target, the ceiling
   :func:`~osprey_connectors.posture_store.effective_writes` reads for a target:
   off is 403 ``writes_disabled``.
3. **The owner's record**, out of the read-only control-context tree in one
   :func:`~osprey_connectors.posture_store.owner_context` read:

   * an unbound or unusable tree, or an unreadable record, is 403
     ``control_context_unavailable``;
   * a record naming another target is 409 ``control_target_mismatch`` — no
     record follows the deployment's baseline, so an owner who never switched
     is on the baseline;
   * a record narrowing the service target is 403 ``narrowing``.

Nothing is cached between calls: every action reads the ceiling and the record
as they are when it runs.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from osprey.runtime.guarded_run import ENV_CONTROL_IDENTITY
from osprey.services.trajectory.refusal import Refused

__all__ = [
    "CONTROL_CONTEXT_UNAVAILABLE",
    "CONTROL_TARGET_MISMATCH",
    "NARROWING",
    "NO_OPERATOR_IDENTITY",
    "WRITES_DISABLED",
    "AdmitGate",
    "AdmitRefused",
    "Admission",
]

NO_OPERATOR_IDENTITY = "no_operator_identity"
WRITES_DISABLED = "writes_disabled"
CONTROL_CONTEXT_UNAVAILABLE = "control_context_unavailable"
CONTROL_TARGET_MISMATCH = "control_target_mismatch"
NARROWING = "narrowing"

#: The HTTP status each refusal answers with: 403 posture, 409 state.
_STATUS = {
    NO_OPERATOR_IDENTITY: 403,
    WRITES_DISABLED: 403,
    CONTROL_CONTEXT_UNAVAILABLE: 403,
    CONTROL_TARGET_MISMATCH: 409,
    NARROWING: 403,
}


class AdmitRefused(Refused):
    """An action the gate refuses; its code is one of this module's constants."""

    statuses = _STATUS


@dataclass(frozen=True)
class Admission:
    """What an admitted action's audit record names.

    Attributes:
        owner: Whose action it is, as the door minted it.
        kind: Which door admitted it.
        target: The control target it acts on.
        generation: The generation of the owner's record; 0 when the owner has
            no record and follows the baseline.
        posture: The owner's per-target narrowings, a copy.
    """

    owner: str
    kind: str
    target: str
    generation: int
    posture: dict[str, str] = field(default_factory=dict)


def _target_writes_enabled(section: Mapping[str, Any], target: str) -> bool:
    from osprey_connectors.types import target_writes_enabled

    return target_writes_enabled(section, target)


def _owner_context(owner: str, target: str, *, baseline: str) -> Any:
    from osprey_connectors.posture_store import owner_context

    return owner_context(owner, target, baseline=baseline)


def _floor_names() -> frozenset[str]:
    """The names no write may belong to: ``unknown`` and this process's own identities."""
    from osprey_connectors.identity import AUDIT_IDENTITY_ENV, UNKNOWN_IDENTITY

    names = {UNKNOWN_IDENTITY}
    for env in (ENV_CONTROL_IDENTITY, AUDIT_IDENTITY_ENV):
        value = os.environ.get(env, "").strip()
        if value:
            names.add(value)
    return frozenset(names)


@dataclass(frozen=True)
class AdmitGate:
    """The admission gate for one service target.

    Attributes:
        target: The control target the service drives.
        section: The ``control_system:`` config section the ceiling and the
            baseline are read from.
    """

    target: str
    section: Mapping[str, Any]

    @classmethod
    def from_config(cls, config: Mapping[str, Any], target: str) -> AdmitGate:
        """The gate for *target*, reading the loaded project config's ``control_system:``."""
        section = config.get("control_system")
        return cls(target=target, section=section if isinstance(section, Mapping) else {})

    @property
    def baseline(self) -> str:
        """The deployment's baseline target, which an owner with no record follows."""
        from osprey_connectors.types import baseline_target

        return baseline_target(self.section)

    def admit(self, owner: Any, kind: str) -> Admission:
        """Admit *owner*'s action through door *kind*, or refuse it.

        Args:
            owner: Whose action it is, as the door minted it; ``None`` when the
                door minted nobody.
            kind: Which door admitted the request.

        Returns:
            The facts the action's audit record carries.

        Raises:
            AdmitRefused: A check in the module's list fails.
        """
        if not isinstance(owner, str) or not owner.strip():
            raise AdmitRefused(
                NO_OPERATOR_IDENTITY, "the request names no operator, and an action needs one"
            )
        if owner in _floor_names():
            raise AdmitRefused(
                NO_OPERATOR_IDENTITY,
                f"{owner!r} is not an operator; an action belongs to the person who asked for it",
            )

        if not _target_writes_enabled(self.section, self.target):
            from osprey_connectors.types import target_writes_enabled_key

            key = target_writes_enabled_key(self.section, self.target)
            raise AdmitRefused(
                WRITES_DISABLED,
                f"the deployment does not arm writes for target {self.target!r} ({key})",
            )

        from osprey_connectors.posture_store import StoreVerdict

        baseline = self.baseline
        context = _owner_context(owner, self.target, baseline=baseline)
        verdict = context.detail.verdict
        if verdict == StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE:
            raise AdmitRefused(
                CONTROL_CONTEXT_UNAVAILABLE,
                context.detail.reason or f"the control context of {owner!r} cannot be read",
            )
        record = context.record
        owner_target = record.target if record is not None else baseline
        if owner_target != self.target:
            raise AdmitRefused(
                CONTROL_TARGET_MISMATCH,
                f"{owner!r} is on target {owner_target!r}; this service drives {self.target!r}",
            )
        if verdict == StoreVerdict.NARROWING:
            raise AdmitRefused(
                NARROWING,
                f"{owner!r} has narrowed target {self.target!r}; widen it to act here",
            )
        return Admission(
            owner=owner,
            kind=kind,
            target=self.target,
            generation=record.generation if record is not None else 0,
            posture=dict(record.posture) if record is not None else {},
        )
