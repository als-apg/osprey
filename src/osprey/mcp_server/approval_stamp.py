"""The approval hook's ask stamp, as the MCP servers that check it read it.

When the approval hook puts a tool call to an approver it leaves an *ask
stamp* in the shared state directory: one JSON file per tool call, named for
the call's ``tool_use_id``, recording the tool and every ``approved_*`` field
the prompt bound the call to. A server that runs a guarded call reads that
stamp back before it acts, and runs the call only when the stamp exists, names
the tool, and agrees with every approved field the call carries.

This module is the one place those spellings, the fields, the per-tool rule of
which fields a call must carry, and the trajectory values digest are defined
on the package side. The hook script (``osprey_approval.py``) cannot import
``osprey``, so it restates them; ``tests/mcp_server/test_ask_stamp.py`` pins
the two equal.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "APPROVAL_ASK_PREFIX",
    "APPROVAL_ASK_SUFFIX",
    "APPROVAL_CONFIG_INVALID",
    "APPROVAL_MISSING",
    "APPROVED_FIELDS",
    "DEFAULT_REQUIRED_FIELDS",
    "TOOL_REQUIRED_FIELDS",
    "ApprovalConfigInvalid",
    "ApprovalMissing",
    "ApprovalRefused",
    "RequiredFields",
    "approved_values_sha256",
    "ask_stamp_filename",
    "check_asked_call",
    "read_ask_stamp",
    "required_fields",
    "stamp_refusal",
]

#: The ask stamp file of one tool call is
#: ``<APPROVAL_ASK_PREFIX><tool_use_id><APPROVAL_ASK_SUFFIX>`` in the state directory.
APPROVAL_ASK_PREFIX = "approval_ask_"
APPROVAL_ASK_SUFFIX = ".json"

#: The refusal code of a call that should have been asked and has no matching stamp.
APPROVAL_MISSING = "approval_missing"
#: The refusal code of a call under an approval switch that is not true or false.
APPROVAL_CONFIG_INVALID = "approval_config_invalid"

#: The call fields the approval prompt binds a guarded call to. Every one the
#: call carries must equal the value the approval hook recorded for the call.
#:
#: - ``approved_target``: the control target the prompt showed.
#: - ``approved_view_sha256``: the pyAML view digest a measurement prompt showed.
#: - ``approved_step_digest``: the trajectory step an apply prompt showed.
#: - ``approved_values_sha256``: the put-back an apply prompt showed, as
#:   :func:`approved_values_sha256` digests it.
APPROVED_FIELDS: tuple[str, ...] = (
    "approved_target",
    "approved_view_sha256",
    "approved_step_digest",
    "approved_values_sha256",
)


@dataclass(frozen=True)
class RequiredFields:
    """Which approved fields an asking call of one tool must carry.

    Attributes:
        all_of: Fields the call must carry, every one of them.
        exactly_one_of: Groups of fields; of each group the call carries
            exactly one.
    """

    all_of: tuple[str, ...] = ("approved_target",)
    exactly_one_of: tuple[tuple[str, ...], ...] = ()


#: The rule for a tool :data:`TOOL_REQUIRED_FIELDS` does not name.
DEFAULT_REQUIRED_FIELDS = RequiredFields()

#: The per-tool rules, keyed by the tool's short name as ``approval.tools``
#: keys it. A measurement is bound to the pyAML view its prompt showed; a
#: trajectory apply to either the step or the put-back its prompt showed.
TOOL_REQUIRED_FIELDS: Mapping[str, RequiredFields] = {
    "pyaml_measure": RequiredFields(all_of=("approved_target", "approved_view_sha256")),
    "trajectory_shot": RequiredFields(all_of=("approved_target",)),
    "trajectory_apply": RequiredFields(
        all_of=("approved_target",),
        exactly_one_of=(("approved_step_digest", "approved_values_sha256"),),
    ),
}


def required_fields(tool: str) -> RequiredFields:
    """The :class:`RequiredFields` an asking call of *tool* must satisfy."""
    return TOOL_REQUIRED_FIELDS.get(tool, DEFAULT_REQUIRED_FIELDS)


def ask_stamp_filename(tool_use_id: str) -> str:
    """The state-directory file name of the ask stamp for *tool_use_id*."""
    return f"{APPROVAL_ASK_PREFIX}{tool_use_id}{APPROVAL_ASK_SUFFIX}"


def read_ask_stamp(tool_use_id: str | None) -> dict | None:
    """The approval hook's ask stamp for the call *tool_use_id*, or ``None``.

    ``None`` without an id, without a stamp, and when the state directory
    cannot be resolved: each is a call nobody can show was asked.
    """
    if not tool_use_id:
        return None
    try:
        from osprey.mcp_server.control_system import target_state

        stamp = target_state.read_file(target_state.state_dir() / ask_stamp_filename(tool_use_id))
    except Exception:
        logger.debug("Could not read the ask stamp for %s", tool_use_id, exc_info=True)
        return None
    return stamp if isinstance(stamp, dict) else None


def stamp_refusal(
    stamp: Mapping[str, Any] | None,
    tool: str,
    approved: Mapping[str, str | None],
) -> str | None:
    """Why the call may not run on *stamp*, or ``None`` when the stamp admits it.

    The stamp admits the call when it exists, names *tool*, the call carries
    the fields :func:`required_fields` demands of *tool*, and every approved
    field the call carries equals the stamp's value for it.

    Args:
        stamp: The call's ask stamp, from :func:`read_ask_stamp`.
        tool: The tool's short name, as ``approval.tools`` keys it.
        approved: The call's :data:`APPROVED_FIELDS`, ``None`` for an absent one.

    Returns:
        The refusal message, ending in ``(approval_missing)`` and an instruction
        to call again; ``None`` when the call is admitted.
    """
    if stamp is None or not str(stamp.get("tool") or "").endswith(tool):
        return (
            "not run: this call's approval prompt left no record (approval_missing). "
            "Call the tool again so the approval prompt can ask."
        )
    rule = required_fields(tool)
    for group in rule.exactly_one_of:
        carried = [name for name in group if approved.get(name) is not None]
        if len(carried) != 1:
            names = ", ".join(f"`{name}`" for name in group)
            return (
                f"not run: this call must carry exactly one of {names} "
                "(approval_missing). Call the tool again."
            )
    for name in APPROVED_FIELDS:
        value = approved.get(name)
        if value is None and name not in rule.all_of:
            continue
        if value is None or stamp.get(name) != value:
            return (
                f"not run: `{name}` differs from what the approval prompt recorded "
                "(approval_missing). Call the tool again."
            )
    return None


class ApprovalRefused(Exception):
    """A guarded call refused before it ran; :attr:`code` names the refusal."""

    code: str


class ApprovalMissing(ApprovalRefused):
    """The call's policy asks and the approval hook left no matching record."""

    code = APPROVAL_MISSING


class ApprovalConfigInvalid(ApprovalRefused):
    """The config states an ``approval.enabled`` that is not true or false, or
    an ``approval`` that is not a mapping; calling again cannot help."""

    code = APPROVAL_CONFIG_INVALID


def check_asked_call(
    config: Mapping[str, Any], tool: str, approved: Mapping[str, str | None]
) -> bool:
    """Refuse a guarded call unless the approval hook's stamp admits it.

    An invalid approval switch refuses every call, as the hook does. A call
    whose policy puts it to nobody (approval off, or the tool's policy
    ``skip``) was asked of nobody and is not checked. Otherwise the stamp of
    the current tool call must admit it (:func:`stamp_refusal`).

    Args:
        config: The rendered ``config.yml``.
        tool: The tool's short name, as ``approval.tools`` keys it.
        approved: The call's :data:`APPROVED_FIELDS`, ``None`` for an absent one.

    Returns:
        Whether the call was put to a human, so the caller checks its target
        and digests against what it is about to act on.

    Raises:
        ApprovalConfigInvalid: The approval switch is not true or false.
        ApprovalMissing: The policy asks and the stamp does not admit the call.
    """
    from osprey.cli.build_posture_check import approval_enabled_fault, ask_capable

    fault = approval_enabled_fault(config)
    if fault is not None:
        raise ApprovalConfigInvalid(f"not run: {fault} ({APPROVAL_CONFIG_INVALID}).")
    if not ask_capable(config, tool):
        return False
    from osprey.audit.call import current_tool_use_id

    refusal = stamp_refusal(read_ask_stamp(current_tool_use_id()), tool, approved)
    if refusal is not None:
        raise ApprovalMissing(refusal)
    return True


def approved_values_sha256(target: str, model: str, values: Iterable[Mapping[str, Any]]) -> str:
    """The digest an apply prompt binds a put-back to.

    The sha256 hex digest of the canonical JSON object
    ``{"model": model, "pairs": [[name, before], ...], "target": target}``,
    with the pairs sorted by name, keys sorted, no whitespace, ASCII only.
    *values* is the put-back body's ``values`` list of ``{"name", "before"}``
    entries, and *target* is the trajectory service's own target.

    Raises:
        KeyError: A value entry lacks ``name`` or ``before``.
        ValueError: A ``before`` is not finite (NaN or infinity has no JSON form).
    """
    pairs = sorted(([str(entry["name"]), entry["before"]] for entry in values), key=lambda p: p[0])
    canonical = json.dumps(
        {"model": model, "pairs": pairs, "target": target},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()
