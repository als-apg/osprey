"""Stage 1 of approval: a call whose policy asks runs only on the prompt's record.

The approval hook puts a call to an approver and, when it asks, files an ask
stamp named for the call's tool-use id beside the control context. A server
that runs a guarded call reads that stamp back before it acts:
:func:`require_approval_record` refuses a call whose tool's policy asks
(:func:`~osprey.cli.build_posture_check.ask_capable`) when the stamp for the
call in scope is missing, unreadable, names another tool, or recorded other
values for the approved fields the call carries. A config whose approval
switch is not true or false refuses every call, asking or not.

A governed tool (the write switch decides it) is ask-capable whatever
``approval.enabled`` or its policy says, so its call needs the record in every
position: the hook files the same stamp at ``asked``, where a person answered,
and at ``allowed``, where the switch did. No server reads the position.

The answer is a :class:`Refusal` or ``None``; each caller turns a refusal into
its own result shape. Who approved is not read: a stamp the dispatch policy
filed (``approver: dispatch_policy``) or the write switch's position filed
(``approver: write_switch``) is a record like a human's. The stamp's
spellings are restated below (:data:`STAMP_APPROVERS`,
:data:`STAMP_POSITIONS`, :data:`ANSWERED_AT_KEY`) for the readers that do look,
such as the executor's posture-store check, which admits only
:data:`PROMPTED_APPROVERS`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

logger = logging.getLogger(__name__)

__all__ = [
    "APPROVAL_ASK_PREFIX",
    "APPROVAL_ASK_SUFFIX",
    "APPROVAL_CONFIG_INVALID",
    "APPROVAL_MISSING",
    "ANSWERED_AT_KEY",
    "APPROVED_FIELDS",
    "APPROVER_DISPATCH_POLICY",
    "APPROVER_HUMAN",
    "APPROVER_WRITE_SWITCH",
    "POSITION_ALLOWED",
    "POSITION_ASKED",
    "PROMPTED_APPROVERS",
    "STAMP_APPROVERS",
    "STAMP_POSITIONS",
    "TOOL_REQUIRED_FIELDS",
    "Refusal",
    "approval_ask_stamp",
    "require_approval_record",
    "required_fields",
]

#: :attr:`Refusal.error_type` for a call whose prompt left no matching record.
APPROVAL_MISSING: Final = "approval_missing"
#: :attr:`Refusal.error_type` for a call under an approval switch that is not a bool.
APPROVAL_CONFIG_INVALID: Final = "approval_config_invalid"

#: The call fields the approval prompt binds a call to. Every one the call
#: carries must equal the value the approval hook recorded for the call.
APPROVED_FIELDS: tuple[str, ...] = ("approved_target", "approved_view_sha256")

#: The approved fields an asking call of one tool must carry beyond what every
#: guarded run carries, keyed by the tool's short name as ``approval.tools``
#: keys it: a measurement is bound to the pyAML view its prompt showed.
TOOL_REQUIRED_FIELDS: Mapping[str, tuple[str, ...]] = {
    "pyaml_measure": ("approved_view_sha256",),
}

#: The ask stamp the approval hook leaves for a call it put to an approver,
#: restated from ``osprey_approval.py`` (a hook script this package cannot
#: import); ``tests/mcp_server/test_approval_record.py`` pins the spellings equal.
APPROVAL_ASK_PREFIX = "approval_ask_"
APPROVAL_ASK_SUFFIX = ".json"

#: The ``approver`` of a stamp a person answered at the approval prompt.
APPROVER_HUMAN: Final = "human"
#: The ``approver`` of a stamp the dispatch policy answered in a dispatch run.
APPROVER_DISPATCH_POLICY: Final = "dispatch_policy"
#: The ``approver`` of a stamp the write switch's ``allowed`` position answered.
APPROVER_WRITE_SWITCH: Final = "write_switch"
#: Every ``approver`` word the hook writes; each is a record for this module.
STAMP_APPROVERS: tuple[str, ...] = (
    APPROVER_HUMAN,
    APPROVER_DISPATCH_POLICY,
    APPROVER_WRITE_SWITCH,
)
#: The approvers that answered a prompt rather than a position: the only ones
#: that let code naming the posture store run.
PROMPTED_APPROVERS: frozenset[str] = frozenset({APPROVER_HUMAN, APPROVER_DISPATCH_POLICY})

#: The ``position`` a governed call's stamp records when a person was asked.
POSITION_ASKED: Final = "asked"
#: The ``position`` a governed call's stamp records when nobody was asked.
POSITION_ALLOWED: Final = "allowed"
#: Every ``position`` word a stamp carries; a stamp off the switch has none.
STAMP_POSITIONS: tuple[str, ...] = (POSITION_ASKED, POSITION_ALLOWED)
#: The stamp key holding when the call was answered, filed at once at ``allowed``.
ANSWERED_AT_KEY: Final = "answered_at"

_NO_RECORD = (
    "not run: this call's approval prompt left no record (approval_missing). "
    "Call the tool again so the approval prompt can ask."
)


@dataclass(frozen=True)
class Refusal:
    """Why a call was not run: its :attr:`error_type` and the sentence to show."""

    error_type: Literal["approval_missing", "approval_config_invalid"]
    message: str

    def __str__(self) -> str:
        return self.message


def approval_ask_stamp(tool_use_id: str | None) -> dict | None:
    """The approval hook's ask stamp for the call *tool_use_id*, or ``None``.

    ``None`` without an id, without a stamp, for one that is not a mapping, and
    when the state directory cannot be resolved: each is a call nobody can
    show was asked.
    """
    if not tool_use_id:
        return None
    try:
        from osprey.mcp_server.control_system import target_state

        stamp = target_state.read_file(
            target_state.state_dir() / f"{APPROVAL_ASK_PREFIX}{tool_use_id}{APPROVAL_ASK_SUFFIX}"
        )
    except Exception:
        logger.debug("Could not read the ask stamp for %s", tool_use_id, exc_info=True)
        return None
    return stamp if isinstance(stamp, dict) else None


def required_fields(tool: str, *, guarded_run: bool = False) -> tuple[str, ...]:
    """The approved fields an asking call of *tool* must carry.

    The one rule, as the approval hook binds a call: a call that runs under
    the control target's run lock (*guarded_run*: a readwrite ``execute`` or
    ``execute_file``, every ``pyaml_measure``) carries its target, and a tool
    :data:`TOOL_REQUIRED_FIELDS` names carries its further fields too — a
    measurement its view digest. A tool under no lock that the table does not
    name must carry nothing; whatever it does carry is still compared.
    """
    fields: tuple[str, ...] = ("approved_target",) if guarded_run else ()
    return (*fields, *TOOL_REQUIRED_FIELDS.get(tool, ()))


def require_approval_record(
    tool: str,
    config: Mapping[str, Any],
    *,
    approved: Mapping[str, str | None] | None = None,
    guarded_run: bool = False,
) -> Refusal | None:
    """Refuse a call of *tool* that its approval prompt left no record of.

    Checked in this order: an ``approval_enabled_fault`` in *config* refuses
    as ``approval_config_invalid``; a tool that is not ``ask_capable`` passes;
    otherwise the ask stamp for the call in scope
    (:func:`~osprey.audit.call.current_tool_use_id`) must exist and name
    *tool*, and every approved field the call carries, and every one
    :func:`required_fields` demands of it, must equal what the stamp recorded.

    A governed tool is ask-capable under ``approval.enabled: false`` and under
    a ``skip`` policy alike. Neither the stamp's ``approver`` nor its
    ``position`` is read: a ``write_switch`` stamp filed at ``allowed`` is a
    record like a human's at ``asked``.

    Args:
        tool: The tool's short name, as ``approval.tools`` keys it.
        config: The rendered ``config.yml``.
        approved: The call's approved fields, ``None`` for an absent one.
            ``None`` when the call carries none.
        guarded_run: Whether the call runs under the control target's run
            lock, and so must carry ``approved_target``.

    Returns:
        ``None`` when the call may run, else the :class:`Refusal`.
    """
    from osprey.cli.build_posture_check import approval_enabled_fault, ask_capable

    fault = approval_enabled_fault(config)
    if fault is not None:
        return Refusal(APPROVAL_CONFIG_INVALID, f"not run: {fault} (approval_config_invalid).")
    if not ask_capable(config, tool):
        return None
    from osprey.audit.call import current_tool_use_id

    stamp = approval_ask_stamp(current_tool_use_id())
    if stamp is None or not str(stamp.get("tool") or "").endswith(tool):
        return Refusal(APPROVAL_MISSING, _NO_RECORD)
    required = required_fields(tool, guarded_run=guarded_run)
    fields = approved or {}
    for name in dict.fromkeys((*APPROVED_FIELDS, *fields, *required)):
        value = fields.get(name)
        if value is None and name not in required:
            continue
        if value is None or stamp.get(name) != value:
            return Refusal(
                APPROVAL_MISSING,
                f"not run: `{name}` differs from what the approval prompt recorded "
                "(approval_missing). Call the tool again.",
            )
    return None
