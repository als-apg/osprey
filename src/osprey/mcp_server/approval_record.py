"""Stage 1 of approval: a call whose policy asks runs only on the prompt's record.

The approval hook puts a call to an approver and, when it asks, files an ask
stamp named for the call's tool-use id beside the control context. A server
that runs a guarded call reads that stamp back before it acts:
:func:`require_approval_record` refuses a call whose tool's policy asks
(:func:`~osprey.cli.build_posture_check.ask_capable`) when the stamp for the
call in scope is missing, unreadable, names another tool, or recorded other
values for the approved fields the call carries. A config whose approval
switch is not true or false refuses every call, asking or not.

What a call must carry is its tool's :class:`Requirements`. A built-in reads
them from :data:`TOOL_REQUIRED_FIELDS`; an application tool, named by its full
``mcp__<app>__<tool>`` name, from rendered config
``apps.<app>.approval.<tool>.require`` (:func:`app_approval_entry`): an
``all_of`` list and ``one_of`` groups, each satisfied by exactly one field.

The answer is a :class:`Refusal` or ``None``; each caller turns a refusal into
its own result shape. Who approved is not read: a stamp the dispatch policy
filed (``approver: dispatch_policy``) is a record like a human's.
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
    "APPROVED_FIELDS",
    "TOOL_REQUIRED_FIELDS",
    "Refusal",
    "Requirements",
    "app_approval_entry",
    "approval_ask_stamp",
    "require_approval_record",
    "required_fields",
    "requirements",
]

#: :attr:`Refusal.error_type` for a call whose prompt left no matching record.
APPROVAL_MISSING: Final = "approval_missing"
#: :attr:`Refusal.error_type` for a call under an approval switch that is not a bool.
APPROVAL_CONFIG_INVALID: Final = "approval_config_invalid"

#: The call fields the approval prompt binds a call to. Every one the call
#: carries must equal the value the approval hook recorded for the call.
APPROVED_FIELDS: tuple[str, ...] = ("approved_target", "approved_view_sha256")

#: The approved fields an asking call of one built-in tool must carry beyond
#: what every guarded run carries, keyed by the tool's short name as
#: ``approval.tools`` keys it: a measurement is bound to the pyAML view its
#: prompt showed. An application tool is never named here: its requirements
#: are read from rendered config (:func:`app_approval_entry`).
TOOL_REQUIRED_FIELDS: Mapping[str, tuple[str, ...]] = {
    "pyaml_measure": ("approved_view_sha256",),
}

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


@dataclass(frozen=True)
class Requirements:
    """The approved fields an asking call must carry.

    Every :attr:`all_of` name, and of each :attr:`one_of` group exactly one
    name, as the approval prompt recorded it.
    """

    all_of: tuple[str, ...]
    one_of: tuple[tuple[str, ...], ...]


def _app_tool(tool: str) -> tuple[str, str] | None:
    """``(app, short)`` for a tool spelled ``mcp__<app>__<short>``, else ``None``."""
    if not tool.startswith("mcp__"):
        return None
    app, sep, short = tool[len("mcp__") :].partition("__")
    if not (app and sep and short):
        return None
    return app, short


def app_approval_entry(config: Mapping[str, Any], tool: str) -> Mapping[str, Any] | None:
    """The rendered approval entry of the application tool *tool*, or ``None``.

    The one reader of ``apps.<app>.approval.<short>``: *tool* is the full
    ``mcp__<app>__<short>`` name (the projected server is named for its
    application). Any other spelling, an absent leaf, or a leaf that is not a
    mapping answers ``None``.
    """
    split = _app_tool(tool)
    if split is None:
        return None
    app, short = split
    node: Any = config
    for key in ("apps", app, "approval", short):
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node if isinstance(node, Mapping) else None


def _names(value: Any) -> tuple[str, ...] | None:
    """*value* as a tuple of non-empty strings, or ``None`` when it is not one."""
    if not isinstance(value, list) or not all(isinstance(n, str) and n for n in value):
        return None
    return tuple(value)


def _parse_require(require: Any) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]] | None:
    """An entry's ``require`` as ``(all_of, one_of)``, or ``None`` when malformed."""
    if require is None:
        return (), ()
    if not isinstance(require, Mapping):
        return None
    all_of = _names(require.get("all_of", []))
    groups = require.get("one_of", [])
    if all_of is None or not isinstance(groups, list):
        return None
    one_of = tuple(_names(group) for group in groups)
    if any(not group for group in one_of):
        return None
    return all_of, one_of  # type: ignore[return-value]


def _require_fault(config: Mapping[str, Any], tool: str) -> str | None:
    """The sentence naming *tool*'s malformed ``require``, else ``None``."""
    entry = app_approval_entry(config, tool)
    if entry is None or _parse_require(entry.get("require")) is not None:
        return None
    app, short = _app_tool(tool) or ("", "")
    return f"apps.{app}.approval.{short}.require is {entry.get('require')!r}"


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


def requirements(
    tool: str, config: Mapping[str, Any], *, guarded_run: bool = False
) -> Requirements:
    """What an asking call of *tool* must carry under *config*.

    An application tool with an entry (:func:`app_approval_entry`) requires
    ``approved_target`` under *guarded_run* and its ``require.all_of``, and its
    ``require.one_of`` groups; every other tool :func:`required_fields`.

    Raises:
        ValueError: The entry's ``require`` is malformed.
    """
    entry = app_approval_entry(config, tool)
    if entry is None:
        return Requirements(required_fields(tool, guarded_run=guarded_run), ())
    parsed = _parse_require(entry.get("require"))
    if parsed is None:
        raise ValueError(_require_fault(config, tool))
    all_of, one_of = parsed
    guard: tuple[str, ...] = ("approved_target",) if guarded_run else ()
    return Requirements(tuple(dict.fromkeys((*guard, *all_of))), one_of)


def require_approval_record(
    tool: str,
    config: Mapping[str, Any],
    *,
    approved: Mapping[str, str | None] | None = None,
    guarded_run: bool = False,
) -> Refusal | None:
    """Refuse a call of *tool* that its approval prompt left no record of.

    Checked in this order: an ``approval_enabled_fault`` in *config*, or a
    malformed ``require`` in *tool*'s application entry, refuses as
    ``approval_config_invalid``; a tool that is not ``ask_capable`` passes;
    otherwise the ask stamp for the call in scope
    (:func:`~osprey.audit.call.current_tool_use_id`) must exist and name
    *tool*, and every approved field the call carries, and every ``all_of``
    name of its :func:`requirements`, must equal what the stamp recorded; of
    each ``one_of`` group the call carries exactly one name, the stamp records
    it with that value and records none of the group's others.

    Args:
        tool: A built-in's short name, as ``approval.tools`` keys it, or an
            application tool's full ``mcp__<app>__<tool>`` name.
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
    fault = _require_fault(config, tool)
    if fault is not None:
        return Refusal(APPROVAL_CONFIG_INVALID, f"not run: {fault} (approval_config_invalid).")
    if not ask_capable(config, tool):
        return None
    from osprey.audit.call import current_tool_use_id

    stamp = approval_ask_stamp(current_tool_use_id())
    if stamp is None or not str(stamp.get("tool") or "").endswith(tool):
        return Refusal(APPROVAL_MISSING, _NO_RECORD)
    needs = requirements(tool, config, guarded_run=guarded_run)
    required = needs.all_of
    grouped = {name for group in needs.one_of for name in group}
    fields = approved or {}
    for name in dict.fromkeys((*APPROVED_FIELDS, *fields, *required)):
        if name in grouped:
            continue
        value = fields.get(name)
        if value is None and name not in required:
            continue
        if value is None or stamp.get(name) != value:
            return _differs(name)
    for group in needs.one_of:
        carried = [name for name in group if fields.get(name) is not None]
        recorded = [name for name in group if stamp.get(name) is not None]
        if len(carried) != 1 or len(recorded) > 1:
            names = ", ".join(f"`{name}`" for name in group)
            return Refusal(
                APPROVAL_MISSING,
                f"not run: exactly one of {names} must be carried, as the approval prompt "
                "recorded it (approval_missing). Call the tool again.",
            )
        if stamp.get(carried[0]) != fields[carried[0]]:
            return _differs(carried[0])
    return None


def _differs(name: str) -> Refusal:
    return Refusal(
        APPROVAL_MISSING,
        f"not run: `{name}` differs from what the approval prompt recorded "
        "(approval_missing). Call the tool again.",
    )
