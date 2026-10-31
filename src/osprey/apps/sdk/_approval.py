"""The approved-call check an application tool server runs before it acts.

:func:`check_approved_call` is the one place an application reads an approval
stamp. It runs the framework's write chain for one call and adds only the
live-target check, in this order:

1. **Offered** — the render must carry the tool's approval entry
   (:func:`~osprey.mcp_server.approval_record.app_approval_entry`).
2. **Write verdict** — :func:`~osprey_connectors.posture_store.write_verdict`
   for the session's recorded control target, the owner left to its ladder.
3. **Approval record** —
   :func:`~osprey.mcp_server.approval_record.require_approval_record` over the
   call's ``approved_*`` fields, every input digest recomputed over the
   arguments. Every input digest is required: one the arguments cannot produce
   refuses. A tool that asks nobody returns an unbound :class:`BoundState`.
4. **Live target** — the stamp's ``approved_target`` must name the target the
   session is on now: the hook replaces every ``approved_*`` the agent sent, so
   a stamp-versus-call compare cannot see a target switch.

Every refusal is an :class:`ApprovalRefused`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from osprey.mcp_server import approval_record
from osprey.mcp_server.approval_record import APPROVAL_CONFIG_INVALID, APPROVAL_MISSING, Refusal
from osprey_connectors.posture_store import WriteVerdict, write_verdict
from osprey_connectors.refusals import refusal_text
from osprey_connectors.workspace import load_osprey_config

logger = logging.getLogger(__name__)

__all__ = [
    "APP_NAME_ENV_VAR",
    "NOT_OFFERED",
    "ApprovalRefused",
    "BoundState",
    "canonical_json",
    "check_approved_call",
    "input_digest",
]

#: The environment variable naming the application a projected tool server
#: serves; a tool's full name is ``mcp__<app>__<tool>``.
APP_NAME_ENV_VAR = "OSPREY_APP_NAME"
#: :attr:`ApprovalRefused.reason` for a tool the render does not offer.
NOT_OFFERED = "not_offered"

_EMPTY: Mapping[str, str] = MappingProxyType({})


class ApprovalRefused(Exception):
    """An application tool call refused before the application acted.

    Attributes:
        reason: An ``approval_record`` error type, a
            :data:`~osprey_connectors.refusals.REASONS` member, or
            :data:`NOT_OFFERED`.
        message: The sentence to show, guidance included.
        details: Structured facts about the refusal.
    """

    def __init__(self, reason: str, message: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.details: Mapping[str, Any] = dict(details or {})

    def __str__(self) -> str:
        return self.message

    @classmethod
    def from_refusal(cls, refusal: Refusal, guidance: str | None = None) -> ApprovalRefused:
        """The approval record's *refusal*, followed by the application's *guidance*."""
        return cls(refusal.error_type, _with_guidance(refusal.message, guidance))

    @classmethod
    def from_verdict(cls, verdict: WriteVerdict, tool_name: str) -> ApprovalRefused:
        """A refused write *verdict* for the call of *tool_name*."""
        reason = verdict.reason or "record_unavailable"
        message = refusal_text(
            reason,
            verdict.kind,
            detail=f"{tool_name} was refused before reaching the application.",
        )
        return cls(reason, message, {"reason": reason, "target": verdict.target})


@dataclass(frozen=True)
class BoundState:
    """What the approval bound an approved call to.

    Attributes:
        bound: ``False`` when the tool's policy asks nobody; every other
            attribute is then empty.
        approved_target: The control target the call was approved for.
        state_fields: The stamp's value of every bound ``bind.state_fields``
            field, by field name. The application compares them with the state
            it is about to act on.
        input_digests: Every bound ``bind.input_digests`` field, by field name.
        approver: Who approved: ``human`` or ``dispatch_policy``.
    """

    bound: bool
    approved_target: str | None = None
    state_fields: Mapping[str, str] = field(default_factory=lambda: _EMPTY)
    input_digests: Mapping[str, str] = field(default_factory=lambda: _EMPTY)
    approver: str | None = None


def canonical_json(value: object) -> str:
    """Serialize *value* the one way both digest sides agree on.

    Sorted keys, no whitespace, ASCII-only, NaN and infinity refused. The
    approval hook's ``osprey_prompt_render.canonical_json`` is the same
    recipe; ``tests/apps/test_digest_recipe_pin.py`` pins the two equal.
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def input_digest(tool_input: Mapping[str, object], over: Iterable[str]) -> str:
    """Return ``sha256(canonical_json({k: tool_input[k] for k in over}))`` as hex.

    A key in *over* that *tool_input* lacks raises :class:`KeyError`.
    """
    selected = {key: tool_input[key] for key in over}
    return hashlib.sha256(canonical_json(selected).encode("utf-8")).hexdigest()


def _with_guidance(message: str, guidance: str | None) -> str:
    return f"{message}\n{guidance}" if guidance else message


def _guidance(entry: Mapping[str, Any]) -> str | None:
    errors = entry.get("errors")
    refused = errors.get("refused") if isinstance(errors, Mapping) else None
    return refused if isinstance(refused, str) and refused else None


def _bind_entries(bind: Any, kind: str, source: str) -> list[tuple[str, Any]] | None:
    """``(field, source)`` of every *kind* entry of *bind*, ``None`` when malformed."""
    entries = bind.get(kind, []) if isinstance(bind, Mapping) else None
    if not isinstance(entries, list):
        return None
    parsed = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            return None
        name, value = entry.get("field"), entry.get(source)
        if not (isinstance(name, str) and name.startswith("approved_")):
            return None
        parsed.append((name, value))
    return parsed


def _bindings(
    entry: Mapping[str, Any],
) -> tuple[list[tuple[str, Any]], list[tuple[str, list[str]]]] | None:
    """The entry's state fields and input digests, ``None`` when ``bind`` is malformed."""
    bind = entry.get("bind", {})
    if bind is None:
        bind = {}
    if not isinstance(bind, Mapping):
        return None
    states = _bind_entries(bind, "state_fields", "key")
    digests = _bind_entries(bind, "input_digests", "over")
    if states is None or digests is None:
        return None
    if not all(isinstance(key, str) and key for _, key in states):
        return None
    if not all(
        isinstance(over, list) and all(isinstance(k, str) for k in over) for _, over in digests
    ):
        return None
    return states, digests  # type: ignore[return-value]


def _recomputed(arguments: Mapping[str, Any], name: str, over: list[str]) -> str | Refusal:
    """The digest *name* over *arguments*, or the :class:`Refusal` when none can be computed.

    Every bound digest is required: a missing ``over`` key refuses as
    ``approval_missing``, an argument canonical JSON cannot carry as
    ``approval_config_invalid``.
    """
    missing = [key for key in over if key not in arguments]
    if missing:
        keys = ", ".join(f"`{key}`" for key in missing)
        return Refusal(
            APPROVAL_MISSING,
            f"not run: `{name}` is bound over {keys}, which the call does not carry "
            f"({APPROVAL_MISSING}). Call the tool again.",
        )
    try:
        return input_digest(arguments, over)
    except (TypeError, ValueError) as exc:
        return Refusal(
            APPROVAL_CONFIG_INVALID,
            f"not run: `{name}` cannot be computed over the call's arguments: {exc} "
            f"({APPROVAL_CONFIG_INVALID}).",
        )


def _guarded_run_name(target: str | None) -> str | None:
    """The guarded-run name *target* resolves to, ``None`` when it names none."""
    if target is None:
        return None
    from osprey.runtime.guarded_run import guarded_run_target

    try:
        return guarded_run_target(target)
    except Exception:
        logger.debug("Target %r names no guarded-run directory", target, exc_info=True)
        return None


def _verdict(config: Mapping[str, Any]) -> WriteVerdict:
    from osprey.audit.posture import recorded_control_target

    target = recorded_control_target()
    section = config.get("control_system")
    try:
        return write_verdict(section or {}, target)
    except Exception:  # unknown posture must not write
        logger.warning("Write verdict for an application tool could not be reached", exc_info=True)
        return WriteVerdict(allowed=False, reason="record_unavailable", target=target, kind=None)


def check_approved_call(tool_name: str, arguments: Mapping[str, Any]) -> BoundState:
    """Run the approval chain for one call of the application tool *tool_name*.

    Args:
        tool_name: The tool's short name; the full name is
            ``mcp__<OSPREY_APP_NAME>__<tool_name>``.
        arguments: The arguments the call arrived with, ``approved_*`` included.

    Returns:
        The :class:`BoundState` the approval bound the call to; unbound when the
        tool's policy asks nobody.

    Raises:
        ApprovalRefused: The render does not offer the tool, the write verdict
            refuses, the approval record is missing or differs, or the call was
            approved for another target than the session is on.
    """
    app = (os.environ.get(APP_NAME_ENV_VAR) or "").strip()
    full_name = f"mcp__{app}__{tool_name}"
    if not app:
        raise ApprovalRefused(
            NOT_OFFERED,
            f"not run: {tool_name} is served without {APP_NAME_ENV_VAR}, so no render "
            f"offers it ({NOT_OFFERED}).",
        )

    config = load_osprey_config()
    entry = approval_record.app_approval_entry(config, full_name)
    if entry is None:
        raise ApprovalRefused(
            NOT_OFFERED,
            f"not run: this deployment's render does not offer {full_name} ({NOT_OFFERED}).",
        )
    guidance = _guidance(entry)

    verdict = _verdict(config)
    if not verdict.allowed:
        raise ApprovalRefused.from_verdict(verdict, tool_name)

    bindings = _bindings(entry)
    if bindings is None:
        raise ApprovalRefused.from_refusal(
            Refusal(
                APPROVAL_CONFIG_INVALID,
                f"not run: apps.{app}.approval.{tool_name}.bind is malformed "
                f"({APPROVAL_CONFIG_INVALID}).",
            ),
            guidance,
        )
    states, digests = bindings
    from osprey.cli.build_posture_check import ask_capable

    asks = ask_capable(config, full_name)
    approved: dict[str, str | None] = {
        name: value for name, value in arguments.items() if str(name).startswith("approved_")
    }
    for name, over in digests if asks else ():
        digest = _recomputed(arguments, name, over)
        if isinstance(digest, Refusal):
            raise ApprovalRefused.from_refusal(digest, guidance)
        approved[name] = digest

    refusal = approval_record.require_approval_record(
        full_name, config, approved=approved, guarded_run=True
    )
    if refusal is not None:
        raise ApprovalRefused.from_refusal(refusal, guidance)
    if not asks:
        return BoundState(bound=False)

    from osprey.audit.call import current_tool_use_id

    stamp = approval_record.approval_ask_stamp(current_tool_use_id()) or {}
    approved_target = stamp.get("approved_target")
    approved_name = _guarded_run_name(approved_target)
    if approved_name is None or approved_name != _guarded_run_name(verdict.target):
        raise ApprovalRefused.from_refusal(
            Refusal(
                APPROVAL_MISSING,
                f"not run: this call was approved for target `{approved_target}`; the run is "
                f"on `{verdict.target}` ({APPROVAL_MISSING}). Call the tool again.",
            ),
            guidance,
        )
    return BoundState(
        bound=True,
        approved_target=approved_target,
        state_fields=MappingProxyType(
            {name: stamp[name] for name, _ in states if stamp.get(name) is not None}
        ),
        input_digests=MappingProxyType(
            {name: value for name, _ in digests if (value := approved.get(name)) is not None}
        ),
        approver=stamp.get("approver"),
    )
