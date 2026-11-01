"""Which machine a write tool call writes to, as the hooks place it.

The hooks place every hooked call before they judge it: the stdlib twin's
``hook_write_target`` (``osprey_target_state.py``) answers ``None`` for a call
that writes nothing, the session's target for an application tool, the real
machine for a facility write tool, the lane's target for a lane-addressed tool,
and the session's target for every other framework write tool. :func:`write_target_for_tool` is the package restatement
of that placement, for a surface that has to name the machine before the hook
has run — the Simple chat's approval card (C-FRAME). A parity test
(``tests/hooks/test_write_target_parity.py``) pins the two against each other
for every write tool.

The placement is not the gateway role: which gateway a connector writes
through is the connector's own choice and says nothing about which machine's
switch covers the call.

:data:`KIND_BANNERS` holds the owner-facing banner for each machine kind
(C-WORDS), the words a card or chip shows beside the machine. They are not
:data:`~osprey_connectors.refusals.KIND_WORDS`, which a refusal phrase uses.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from osprey_connectors import control_context, posture_store
from osprey_connectors.types import (
    CONTROL_TARGETS,
    TARGET_LIVE,
    baseline_target,
    published_machine_kind,
)

__all__ = [
    "FALLBACK_WRITE_TOOLS",
    "KIND_BANNERS",
    "UNRESOLVED",
    "Unresolved",
    "frame_target",
    "is_write_call",
    "is_write_tool",
    "short_tool_name",
    "write_target_for_tool",
]

logger = logging.getLogger(__name__)

#: The banner an owner reads beside each kind of machine (C-WORDS).
KIND_BANNERS: dict[str, str] = {
    "real": "Live — writes move hardware",
    "rehearsal": "Rehearsal — nothing moves",
    "simulator": "Simulator — nothing moves",
}

#: The write list a deployment with no rendered ``write_tools`` falls back to.
#: Restated from the hooks' ``FALLBACK_WRITE_TOOLS`` (``osprey_hook_log.py``):
#: this package cannot import the framework, and the hooks cannot import this
#: package. The parity test pins the two.
FALLBACK_WRITE_TOOLS: tuple[str, ...] = (
    "mcp__bluesky__queue_add",
    "mcp__bluesky__queue_start",
    "mcp__controls__channel_write",
    "mcp__phoebus__phoebus_drive",
    "mcp__python__execute",
    "mcp__python__execute_file",
    "mcp__python__pyaml_measure",
)

#: The suffix of a whole-server matcher, ``mcp__<server>__.*``.
_MATCHER_WILDCARD = ".*"

#: The one tool whose call may write nothing: a readonly execution.
_EXECUTE_TOOL = "execute"

#: The hook_config key naming the tools a plan lane addresses, as short names.
_LANE_ADDRESSED_KEY = "lane_addressed_tools"

#: The hook_config key naming the ``control_system.write_tools`` entries the
#: render appended to ``write_tools``: the deployment's own write tools.
_FACILITY_EXTRAS_KEY = "control_system_write_tools"

#: The hook_config key naming the facility (custom-server) write set: exact
#: names for a declaring server, ``mcp__<name>__.*`` for one with no list.
_FACILITY_WRITES_KEY = "facility_write_tools"

#: The hook_config key holding every application tool's approval declaration,
#: keyed by full tool name: an application writes to the session's target.
_APP_APPROVALS_KEY = "app_approvals"

#: The hook_config keys naming server prefixes, longest match stripped.
_PREFIX_KEYS = ("server_prefixes", "approval_prefixes")

#: The service key of the default lane, and every service key that can name a
#: lane, in render order. Restated from ``osprey.bluesky_bridge_connection``.
_LANE_ONE = "bluesky"
_LANE_KEYS = (_LANE_ONE, "bluesky_va", "bluesky_live", "bluesky_standin")


class Unresolved:
    """A write whose machine could not be placed: one instance, compared by identity.

    Distinct from ``None``, which :func:`write_target_for_tool` keeps for a call
    that writes nothing. A write nobody can place is refused by the hooks,
    never waved through as a read.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<unresolved>"

    __str__ = __repr__


#: The answer for a write whose machine cannot be placed.
UNRESOLVED = Unresolved()


def is_write_tool(tool_name: str, write_tools: Any) -> bool:
    """Whether *tool_name* is covered by *write_tools*: exact names and ``.*`` matchers."""
    for entry in write_tools or ():
        if not isinstance(entry, str):
            continue
        if entry == tool_name:
            return True
        if entry.endswith(_MATCHER_WILDCARD) and tool_name.startswith(
            entry[: -len(_MATCHER_WILDCARD)]
        ):
            return True
    return False


def is_write_call(tool_input: Any, short_name: str) -> bool:
    """Whether this call writes: only a readonly (or modeless) ``execute`` does not."""
    if short_name != _EXECUTE_TOOL:
        return True
    if not isinstance(tool_input, Mapping):
        tool_input = {}
    return bool(tool_input.get("execution_mode", "readonly") != "readonly")


def short_tool_name(tool_name: str, prefixes: Any) -> str:
    """*tool_name* without its server prefix; the longest matching prefix wins.

    With no prefix matching, the ``mcp__<server>__<tool>`` shape decides, and a
    name in no MCP shape is its own short name.
    """
    matched = [p for p in prefixes or () if isinstance(p, str) and tool_name.startswith(p)]
    if matched:
        return tool_name[len(max(matched, key=len)) :]
    parts = tool_name.split("__")
    if tool_name.startswith("mcp__") and len(parts) > 2:
        return "__".join(parts[2:])
    return tool_name


def _hook_prefixes(hook_config: Mapping[str, Any]) -> list[str]:
    """Every server prefix *hook_config* names."""
    prefixes: list[str] = []
    for key in _PREFIX_KEYS:
        value = hook_config.get(key)
        if isinstance(value, list):
            prefixes.extend(p for p in value if isinstance(p, str))
    return prefixes


def _is_facility_tool(tool_name: str, hook_config: Mapping[str, Any]) -> bool:
    """Whether *tool_name* is a write tool the deployment declared, not the framework.

    The rendered ``facility_write_tools`` set, plus the ``control_system.write_tools``
    entries the render appended. A render from before that key carried no exact
    names, so there the whole-server matchers in ``write_tools`` are the set.
    """
    declared = hook_config.get(_FACILITY_WRITES_KEY)
    if isinstance(declared, list):
        if is_write_tool(tool_name, declared):
            return True
    else:
        for entry in hook_config.get("write_tools") or ():
            if (
                isinstance(entry, str)
                and entry.endswith(_MATCHER_WILDCARD)
                and is_write_tool(tool_name, [entry])
            ):
                return True
    extras = hook_config.get(_FACILITY_EXTRAS_KEY) or ()
    return tool_name in [extra for extra in extras if isinstance(extra, str)]


def _section(config: Any) -> Any:
    return config.get("control_system") if isinstance(config, Mapping) else None


def _lane_block(config: Any, lane_key: str) -> Mapping[str, Any]:
    services = config.get("services") if isinstance(config, Mapping) else None
    block = services.get(lane_key) if isinstance(services, Mapping) else None
    return block if isinstance(block, Mapping) else {}


def _rendered_lanes(config: Any) -> list[tuple[str, str]]:
    """``[(lane key, the target it serves)]`` in render order; lane one always present."""
    baseline = baseline_target(_section(config))
    lanes: list[tuple[str, str]] = []
    for key in _LANE_KEYS:
        block = _lane_block(config, key)
        if not block:
            continue
        declared = block.get("target")
        lanes.append((key, declared if isinstance(declared, str) and declared else baseline))
    if not any(key == _LANE_ONE for key, _ in lanes):
        lanes.insert(0, (_LANE_ONE, baseline))
    return lanes


def _lane_machine(config: Any, tool_input: Any) -> str | Unresolved:
    """The target the call's lane serves, or :data:`UNRESOLVED`."""
    lanes = _rendered_lanes(config)
    lane = tool_input.get("lane") if isinstance(tool_input, Mapping) else None
    if not lane:
        target = lanes[0][1] if len(lanes) == 1 else None
    else:
        target = next((served for key, served in lanes if key == lane), None)
    return target if target in CONTROL_TARGETS else UNRESOLVED


def _session_machine(
    config: Any, record_path: Path | None, stamped: bool | None
) -> str | Unresolved:
    """The target the session is pointed at, or :data:`UNRESOLVED`.

    The record's target when it is readable. With no record at all, the
    baseline — but only when the agent-data root was stamped: an unstamped
    process that finds no record cannot tell "nobody switched" from "looked in
    the wrong place", and the hook refuses it as ``posture_unknown``.
    """
    if record_path is None:
        record, status = control_context.read_record_status()
    else:
        record, status = control_context.read_record_status(path=record_path)
    if record is not None:
        return record.target
    if stamped is None:
        stamped = posture_store.stamped_agent_data_root() is not None
    if status != "missing" or not stamped:
        return UNRESOLVED
    return baseline_target(_section(config))


def write_target_for_tool(
    config: Any,
    tool_name: str,
    tool_input: Any,
    *,
    hook_config: Mapping[str, Any] | None,
    short_name: str | None = None,
    record_path: Path | None = None,
    stamped: bool | None = None,
) -> str | None | Unresolved:
    """The machine a call writes to, placed as the hooks place it. Never raises.

    * ``None`` for a call that writes nothing: a tool outside the render's
      ``write_tools`` (the fallback list when the render names none), or a
      readonly execution.
    * The session's target for an application tool (``app_approvals``).
    * ``live`` for a facility write tool the deployment declared.
    * The lane's target for a lane-addressed tool.
    * The session's target for every other framework write tool.

    :data:`UNRESOLVED` wherever a write cannot be placed, including any failure
    while answering.

    Args:
        config: The loaded ``config.yml`` mapping, for the lane map and baseline.
        tool_name: The full tool name.
        tool_input: The call's arguments.
        hook_config: The render's ``.claude/hooks/hook_config.json`` mapping;
            ``None`` or ``{}`` reads as an empty one.
        short_name: *tool_name* without its server prefix; ``None`` resolves it
            from *hook_config*'s prefix lists.
        record_path: The control-context record to read; ``None`` reads this
            process's own (:func:`osprey_connectors.control_context.record_path`).
        stamped: Whether the agent-data root of the process that will run the
            call is stamped; ``None`` asks this process's environment.
    """
    try:
        hooks: Mapping[str, Any] = hook_config if isinstance(hook_config, Mapping) else {}
        write_tools = hooks.get("write_tools", FALLBACK_WRITE_TOOLS)
        if not is_write_tool(tool_name, write_tools):
            return None
        if short_name is None:
            short_name = short_tool_name(tool_name, _hook_prefixes(hooks))
        if not is_write_call(tool_input, short_name):
            return None
        apps = hooks.get(_APP_APPROVALS_KEY)
        if isinstance(apps, Mapping) and tool_name in apps:
            return _session_machine(config, record_path, stamped)
        if _is_facility_tool(tool_name, hooks):
            return TARGET_LIVE
        if short_name in tuple(hooks.get(_LANE_ADDRESSED_KEY) or ()):
            return _lane_machine(config, tool_input)
        return _session_machine(config, record_path, stamped)
    except Exception:
        logger.debug("Could not place the write of %s", tool_name, exc_info=True)
        return UNRESOLVED


def frame_target(
    target: str | None | Unresolved,
    *,
    state_dir: Any = None,
) -> dict[str, str] | None:
    """C-FRAME's ``target``: ``{key, machine_kind, kind_words}``, or ``None``.

    ``None`` for a call that writes nothing and for one that cannot be placed.
    The kind is the published one (C-KIND), so a ``standin`` the controls
    server has not vouched for reads as the real machine.

    Args:
        target: An answer of :func:`write_target_for_tool`.
        state_dir: The servers' report directory; ``None`` reads this process's.
    """
    if not isinstance(target, str):
        return None
    kind: Literal["real", "rehearsal", "simulator"] = published_machine_kind(
        target, state_dir=state_dir
    )
    return {"key": target, "machine_kind": kind, "kind_words": KIND_BANNERS[kind]}
