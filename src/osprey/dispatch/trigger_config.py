"""Trigger configuration dataclasses and YAML loader for the event dispatcher.

The pool limits are re-exported from :mod:`osprey.dispatch_pool_defaults`, the
stdlib-only leaf the build profile reads them from as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

from osprey.dispatch.clock_schedule import ClockSchedule, parse_clock_schedule
from osprey.dispatch_pool_defaults import DEFAULT_MAX_CONCURRENT_RUNS, DEFAULT_MAX_QUEUE_DEPTH

__all__ = [
    "DEFAULT_MAX_CONCURRENT_RUNS",
    "DEFAULT_MAX_QUEUE_DEPTH",
    "DispatcherConfig",
    "TriggerConfig",
    "load_triggers",
]

# Tool-name prefix of the event dispatcher's own MCP server. A dispatch job may
# not fire dispatch jobs, so no trigger hands one a tool from this server.
_DISPATCHER_TOOL_PREFIX = "mcp__event_dispatcher__"

_DEFAULT_ON_ERROR: dict[str, Any] = {
    "action": "drop",
    "max_retries": 0,
    "backoff_sec": 0.0,
}


@dataclass
class TriggerConfig:
    """Parsed configuration for a single dispatch trigger.

    Attributes:
        name: Unique trigger name.
        source: Event source type (e.g. ``webhook``, ``cron``).
        action: Free-form action mapping. Only ``action.prompt`` is required;
            unread keys pass through untouched for forward compatibility.
        on_error: Error-handling policy (action/max_retries/backoff_sec).
        source_config: Source-specific settings, always a mapping (a blank or
            absent ``source_config`` is empty).
        allowed_tools: Tool names the dispatched run may use, always a list (an
            absent or blank ``action.allowed_tools`` is empty). ``surface_tools``
            can only narrow it.
        surface: Optional label naming the UI/output surface the triggered
            agent run is associated with (e.g. a dashboard or channel name).
            ``None`` when ``action.surface`` is absent.
        surface_prompt: Optional free-text fragment appended to the agent's
            system prompt at run time. ``None`` when ``action.surface_prompt``
            is absent.
        surface_tools: Optional keep-list of tool names narrowing
            ``action.allowed_tools`` at run time. ``None`` when
            ``action.surface_tools`` is absent or blank; an empty list narrows
            nothing.
        max_turns: Optional per-trigger ceiling on agentic turns. ``None`` when
            ``action.max_turns`` is absent, in which case the worker applies
            the deployment's own ``dispatch.max_turns``.
        schedule: The parsed ``at``/``days`` of a clock-time cron trigger;
            ``None`` for every other trigger.
    """

    name: str
    source: str
    action: dict[str, Any]
    on_error: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_ON_ERROR))
    source_config: dict[str, Any] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)
    surface: str | None = None
    surface_prompt: str | None = None
    surface_tools: list[str] | None = None
    max_turns: int | None = None
    schedule: ClockSchedule | None = None


@dataclass
class DispatcherConfig:
    dispatch_target: str
    max_concurrent_runs: int = DEFAULT_MAX_CONCURRENT_RUNS
    max_queue_depth: int = DEFAULT_MAX_QUEUE_DEPTH


def _parse_trigger(raw: Any, index: int) -> TriggerConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"Trigger at index {index} must be a mapping (got {raw!r})")

    name = raw.get("name")
    if not name:
        raise ValueError(f"Trigger at index {index} is missing required field 'name'")

    source = raw.get("source", "")
    if not source:
        raise ValueError(f"Trigger '{name}' is missing required field 'source'")

    action = raw.get("action")
    if action is not None and not isinstance(action, dict):
        raise ValueError(f"Trigger '{name}' field 'action' must be a mapping")
    if not action or not action.get("prompt"):
        raise ValueError(f"Trigger '{name}' is missing required field 'action.prompt'")

    surface = action.get("surface")
    if surface is not None and not isinstance(surface, str):
        raise ValueError(f"Trigger '{name}' field 'action.surface' must be a string")

    surface_prompt = action.get("surface_prompt")
    if surface_prompt is not None and not isinstance(surface_prompt, str):
        raise ValueError(f"Trigger '{name}' field 'action.surface_prompt' must be a string")

    surface_tools_raw = action.get("surface_tools")
    surface_tools: list[str] | None = None
    if surface_tools_raw is not None:
        if not isinstance(surface_tools_raw, list) or not all(
            isinstance(tool, str) for tool in surface_tools_raw
        ):
            raise ValueError(
                f"Trigger '{name}' field 'action.surface_tools' must be a list of strings"
            )
        surface_tools = list(surface_tools_raw)

    # The worker refuses an unusable ceiling with a 422 at dispatch time, which
    # is the moment an event fires — long after this file was authored — so the
    # value is typed here, where the author is still looking at it. ``bool`` is
    # an ``int`` subclass, so ``max_turns: true`` would otherwise mean one turn.
    max_turns = action.get("max_turns")
    if max_turns is not None and (
        isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1
    ):
        raise ValueError(
            f"Trigger '{name}' field 'action.max_turns' must be an integer >= 1 (got {max_turns!r})"
        )

    # The worker refuses an ``allowed_tools`` that is not a list of tool names
    # when an event fires, so its shape is checked here, where the author wrote
    # it. A dispatch job may not fire dispatch jobs, so naming a dispatcher tool
    # is refused here too, and the message names both the trigger and the tool.
    allowed_tools_raw = action.get("allowed_tools")
    allowed_tools: list[str] = []
    if allowed_tools_raw is not None:
        if not isinstance(allowed_tools_raw, list) or not all(
            isinstance(tool, str) for tool in allowed_tools_raw
        ):
            raise ValueError(
                f"Trigger '{name}' field 'action.allowed_tools' must be a list of strings"
            )
        allowed_tools = list(allowed_tools_raw)
    for tool in allowed_tools:
        if tool.startswith(_DISPATCHER_TOOL_PREFIX):
            raise ValueError(
                f"Trigger '{name}' field 'action.allowed_tools' names the event "
                f"dispatcher's own tool '{tool}'; a dispatch job may not fire "
                f"dispatch jobs, so no '{_DISPATCHER_TOOL_PREFIX}' tool is allowed"
            )

    on_error_raw = raw.get("on_error")
    if on_error_raw is None:
        on_error = dict(_DEFAULT_ON_ERROR)
    elif not isinstance(on_error_raw, dict):
        raise ValueError(f"Trigger '{name}' field 'on_error' must be a mapping")
    else:
        on_error = {
            "action": on_error_raw.get("action", _DEFAULT_ON_ERROR["action"]),
            "max_retries": on_error_raw.get("max_retries", _DEFAULT_ON_ERROR["max_retries"]),
            "backoff_sec": on_error_raw.get("backoff_sec", _DEFAULT_ON_ERROR["backoff_sec"]),
        }

    # Every source reads its settings with ``.get``, so a ``source_config``
    # that is not a mapping is refused here, and no source's ``start`` sees one.
    source_config = raw.get("source_config")
    if source_config is None:
        source_config = {}
    elif not isinstance(source_config, dict):
        raise ValueError(f"Trigger '{name}' field 'source_config' must be a mapping")

    # A schedule that cannot be read would otherwise surface only when the
    # dispatcher starts, or never for a mistyped key, so it is refused here,
    # where the author wrote it.
    schedule = None
    if source == "cron":
        schedule = parse_clock_schedule(name, source_config)

    return TriggerConfig(
        name=name,
        source=source,
        action=action,
        on_error=on_error,
        source_config=source_config,
        allowed_tools=allowed_tools,
        surface=surface,
        surface_prompt=surface_prompt,
        surface_tools=surface_tools,
        max_turns=max_turns,
        schedule=schedule,
    )


def load_triggers(path: str) -> tuple[DispatcherConfig, list[TriggerConfig]]:
    """Parse a triggers YAML file and return (DispatcherConfig, list[TriggerConfig])."""
    with open(path) as f:
        doc = yaml.safe_load(f)

    # Fail loud on an empty or non-mapping document rather than the cryptic
    # AttributeError ``'NoneType' object has no attribute 'get'`` a bare
    # ``doc.get(...)`` would raise on an empty file.
    if doc is None:
        raise ValueError(f"triggers file {path!r} is empty (no YAML document)")
    if not isinstance(doc, dict):
        raise ValueError(f"triggers file {path!r} must be a YAML mapping at the top level")

    dispatcher_raw = doc.get("dispatcher")
    if dispatcher_raw is None:
        dispatcher_raw = {}
    elif not isinstance(dispatcher_raw, dict):
        raise ValueError(f"triggers file {path!r} field 'dispatcher' must be a mapping")
    dispatcher_cfg = DispatcherConfig(
        dispatch_target=dispatcher_raw.get("dispatch_target", ""),
        max_concurrent_runs=dispatcher_raw.get("max_concurrent_runs", DEFAULT_MAX_CONCURRENT_RUNS),
        max_queue_depth=dispatcher_raw.get("max_queue_depth", DEFAULT_MAX_QUEUE_DEPTH),
    )

    raw_triggers = doc.get("triggers")
    if raw_triggers is None:
        raw_triggers = []
    elif not isinstance(raw_triggers, list):
        raise ValueError(f"triggers file {path!r} field 'triggers' must be a list of triggers")
    triggers = [_parse_trigger(t, i) for i, t in enumerate(raw_triggers)]

    # Detect duplicate trigger names at load time. The registry registers
    # triggers by name, so a duplicate would otherwise SILENTLY overwrite the
    # earlier one at registration — fail loud here instead.
    seen: set[str] = set()
    for t in triggers:
        if t.name in seen:
            raise ValueError(
                f"Duplicate trigger name {t.name!r} in {path!r}; trigger names must be unique"
            )
        seen.add(t.name)

    return dispatcher_cfg, triggers
