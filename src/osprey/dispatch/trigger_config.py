"""Trigger configuration dataclasses and YAML loader for the event dispatcher."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

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
        source_config: Free-form source-specific configuration.
        surface: Optional label naming the UI/output surface the triggered
            agent run is associated with (e.g. a dashboard or channel name).
            ``None`` when ``action.surface`` is absent.
        surface_prompt: Optional free-text fragment appended to the agent's
            system prompt at run time. ``None`` when ``action.surface_prompt``
            is absent.
    """

    name: str
    source: str
    action: dict[str, Any]
    on_error: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_ON_ERROR))
    source_config: dict[str, Any] = field(default_factory=dict)
    surface: str | None = None
    surface_prompt: str | None = None


@dataclass
class DispatcherConfig:
    dispatch_target: str
    max_concurrent_runs: int = 5
    max_queue_depth: int = 100


def _parse_trigger(raw: dict[str, Any], index: int) -> TriggerConfig:
    name = raw.get("name")
    if not name:
        raise ValueError(f"Trigger at index {index} is missing required field 'name'")

    source = raw.get("source", "")
    if not source:
        raise ValueError(f"Trigger '{name}' is missing required field 'source'")

    action = raw.get("action")
    if not action or not action.get("prompt"):
        raise ValueError(f"Trigger '{name}' is missing required field 'action.prompt'")

    surface = action.get("surface")
    if surface is not None and not isinstance(surface, str):
        raise ValueError(f"Trigger '{name}' field 'action.surface' must be a string")

    surface_prompt = action.get("surface_prompt")
    if surface_prompt is not None and not isinstance(surface_prompt, str):
        raise ValueError(f"Trigger '{name}' field 'action.surface_prompt' must be a string")

    on_error_raw = raw.get("on_error")
    if on_error_raw is None:
        on_error = dict(_DEFAULT_ON_ERROR)
    else:
        on_error = {
            "action": on_error_raw.get("action", _DEFAULT_ON_ERROR["action"]),
            "max_retries": on_error_raw.get("max_retries", _DEFAULT_ON_ERROR["max_retries"]),
            "backoff_sec": on_error_raw.get("backoff_sec", _DEFAULT_ON_ERROR["backoff_sec"]),
        }

    source_config = raw.get("source_config", {})

    return TriggerConfig(
        name=name,
        source=source,
        action=action,
        on_error=on_error,
        source_config=source_config,
        surface=surface,
        surface_prompt=surface_prompt,
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

    dispatcher_raw = doc.get("dispatcher", {})
    dispatcher_cfg = DispatcherConfig(
        dispatch_target=dispatcher_raw.get("dispatch_target", ""),
        max_concurrent_runs=dispatcher_raw.get("max_concurrent_runs", 5),
        max_queue_depth=dispatcher_raw.get("max_queue_depth", 100),
    )

    raw_triggers = doc.get("triggers") or []
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
