"""Stage 1 of approval: a call whose policy asks runs only on the prompt's record.

The approval hook puts a call to an approver and, when it asks, files an ask
stamp named for the call's tool-use id beside the control context. A server
that runs a guarded call reads that stamp back before it acts:
:func:`require_approval_record` refuses a call whose tool's policy asks
(:func:`~osprey.cli.build_posture_check.ask_capable`) when the stamp for the
call in scope is missing, unreadable, names another tool, or recorded other
values for the approved fields the call carries. A config whose approval
switch is not true or false refuses every call, asking or not.

The answer is a :class:`Refusal` or ``None``; each caller turns a refusal into
its own result shape. Who approved is not read: a stamp the dispatch policy
filed (``approver: dispatch_policy``) is a record like a human's.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

logger = logging.getLogger(__name__)

__all__ = [
    "APPROVAL_ASK_PREFIX",
    "APPROVAL_ASK_SUFFIX",
    "APPROVAL_CONFIG_INVALID",
    "APPROVAL_MISSING",
    "APPROVED_FIELDS",
    "Refusal",
    "approval_ask_stamp",
    "require_approval_record",
]

#: :attr:`Refusal.error_type` for a call whose prompt left no matching record.
APPROVAL_MISSING: Final = "approval_missing"
#: :attr:`Refusal.error_type` for a call under an approval switch that is not a bool.
APPROVAL_CONFIG_INVALID: Final = "approval_config_invalid"

#: The call fields the approval prompt binds a call to. Every one the call
#: carries must equal the value the approval hook recorded for the call.
APPROVED_FIELDS: tuple[str, ...] = ("approved_target", "approved_view_sha256")

#: The ask stamp the approval hook leaves for a call it put to an approver,
#: restated from ``osprey_approval.py`` (a hook script this package cannot
#: import); ``tests/mcp_server/test_approval_record.py`` pins the spellings equal.
APPROVAL_ASK_PREFIX = "approval_ask_"
APPROVAL_ASK_SUFFIX = ".json"

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


def require_approval_record(
    tool: str,
    config: Mapping[str, Any],
    *,
    approved: Mapping[str, str | None] | None = None,
    required: Sequence[str] = (),
) -> Refusal | None:
    """Refuse a call of *tool* that its approval prompt left no record of.

    Checked in this order: an ``approval_enabled_fault`` in *config* refuses
    as ``approval_config_invalid``; a tool that is not ``ask_capable`` passes;
    otherwise the ask stamp for the call in scope
    (:func:`~osprey.audit.call.current_tool_use_id`) must exist and name
    *tool*, and every approved field the call carries, and every one in
    *required*, must equal what the stamp recorded.

    Args:
        tool: The tool's short name, as ``approval.tools`` keys it.
        config: The rendered ``config.yml``.
        approved: The call's approved fields, ``None`` for an absent one.
            ``None`` when the call carries none.
        required: The approved fields the call must carry.

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
