"""Machine-description model and parser for the simulation engine.

Defines the typed model a ``machine.json`` parses into — :class:`SimChannel`
(baseline value or derived expression, with optional physical bounds) and
:class:`Scenario` (override set plus archiver event scripts) — and the
validation that turns the decoded JSON into that model. All schema errors
(bad types, unknown references, reference cycles, malformed events) are raised
here at load time, so the runtime :class:`~osprey.simulation.engine.SimulationEngine`
can assume a well-formed model.

:func:`parse_machine` is the single entry point; the engine calls it once at
construction and consumes the returned :class:`ParsedMachine`.
"""

import ast
from dataclasses import dataclass
from datetime import time as dtime
from pathlib import Path
from typing import Any

from osprey.simulation.expressions import (
    ExpressionError,
    compile_expression,
    extract_channel_refs,
)

DEFAULT_SCENARIO = "nominal"

# Non-position keys required per event shape. Position is validated
# separately: exactly one of 'at' (window fraction), 'at_offset' (seconds
# relative to scenario-activation time), or 'at_time' (daily 'HH:MM:SS'
# wall-clock recurrence; step/spike only); ramps need the matching
# 'until'/'until_offset' flavor.
_EVENT_VALUE_KEYS = {
    "step": ("to",),
    "ramp": ("to",),
    "spike": ("amplitude", "width"),
}


@dataclass(frozen=True)
class SimChannel:
    """Parsed channel definition from the machine file."""

    name: str
    value: float | str | None
    expr: ast.expr | None
    refs: tuple[str, ...]
    units: str
    noise: float
    description: str
    expr_source: str = ""
    min_value: float | None = None
    max_value: float | None = None


@dataclass(frozen=True)
class Scenario:
    """Parsed scenario definition: overrides plus archiver event scripts."""

    name: str
    description: str
    overrides: dict[str, float | str]
    archiver: dict[str, list[dict[str, Any]]]


@dataclass(frozen=True)
class ParsedMachine:
    """Validated machine model: metadata, channels, and scenarios."""

    name: str
    description: str
    channels: dict[str, SimChannel]
    scenarios: dict[str, Scenario]


def parse_machine(machine: Any, machine_path: Path) -> ParsedMachine:
    """Parse and validate a decoded machine description into a typed model.

    Args:
        machine: Decoded machine-file JSON (expected to be a mapping with a
            ``channels`` mapping; ``name``/``description``/``scenarios`` optional).
        machine_path: Path the machine was loaded from (for error messages).

    Returns:
        The validated :class:`ParsedMachine` (a default ``nominal`` scenario is
        injected if the file does not define one).

    Raises:
        ValueError: If the machine description is invalid (bad schema, invalid
            expression, unknown reference, or reference cycle).
    """
    if not isinstance(machine, dict) or not isinstance(machine.get("channels"), dict):
        raise ValueError(f"Machine file {machine_path} must define a 'channels' mapping")

    channels = {pv: _parse_channel(pv, spec) for pv, spec in machine["channels"].items()}
    _check_references(channels)
    scenarios = _parse_scenarios(machine.get("scenarios", {}), channels)
    return ParsedMachine(
        name=str(machine.get("name", "")),
        description=str(machine.get("description", "")),
        channels=channels,
        scenarios=scenarios,
    )


def _parse_channel(pv: str, spec: Any) -> SimChannel:
    """Parse and validate a single channel entry."""
    if not isinstance(spec, dict):
        raise ValueError(f"Channel {pv!r}: entry must be a mapping, got {type(spec).__name__}")
    has_value = "value" in spec
    has_expr = "expr" in spec
    if has_value == has_expr:
        raise ValueError(f"Channel {pv!r}: exactly one of 'value' or 'expr' is required")

    value: float | str | None = None
    expr: ast.expr | None = None
    refs: tuple[str, ...] = ()
    if has_value:
        raw = spec["value"]
        if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            raise ValueError(f"Channel {pv!r}: 'value' must be a number or string")
        value = raw if isinstance(raw, str) else float(raw)
    else:
        source = spec["expr"]
        if not isinstance(source, str):
            raise ValueError(f"Channel {pv!r}: 'expr' must be a string")
        try:
            expr = compile_expression(source)
        except ExpressionError as exc:
            raise ValueError(f"Channel {pv!r}: {exc}") from exc
        refs = tuple(sorted(extract_channel_refs(expr)))

    noise = spec.get("noise", 0.0)
    if isinstance(noise, bool) or not isinstance(noise, (int, float)) or noise < 0:
        raise ValueError(f"Channel {pv!r}: 'noise' must be a non-negative number")

    is_string_channel = has_value and isinstance(value, str)
    min_value = _parse_bound(pv, spec, "min", is_string_channel)
    max_value = _parse_bound(pv, spec, "max", is_string_channel)
    if min_value is not None and max_value is not None and min_value >= max_value:
        raise ValueError(
            f"Channel {pv!r}: 'min' ({min_value:g}) must be less than 'max' ({max_value:g})"
        )

    return SimChannel(
        name=pv,
        value=value,
        expr=expr,
        refs=refs,
        units=str(spec.get("units", "")),
        noise=float(noise),
        description=str(spec.get("description", "")),
        expr_source=spec["expr"] if has_expr else "",
        min_value=min_value,
        max_value=max_value,
    )


def _parse_bound(pv: str, spec: dict[str, Any], key: str, is_string_channel: bool) -> float | None:
    """Parse an optional ``min``/``max`` physical bound from a channel spec."""
    if key not in spec:
        return None
    if is_string_channel:
        raise ValueError(f"Channel {pv!r}: {key!r} is not supported on string-valued channels")
    raw = spec[key]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"Channel {pv!r}: {key!r} must be a number, got {raw!r}")
    return float(raw)


def _check_references(channels: dict[str, SimChannel]) -> None:
    """Validate expression references: all known, no cycles (DFS)."""
    for channel in channels.values():
        for ref in channel.refs:
            if ref not in channels:
                raise ValueError(
                    f"Channel {channel.name!r}: expression references unknown channel {ref!r}"
                )

    white, gray, black = 0, 1, 2
    color = dict.fromkeys(channels, white)

    def visit(node: str, path: list[str]) -> None:
        color[node] = gray
        path.append(node)
        for ref in channels[node].refs:
            if color[ref] == gray:
                cycle = " -> ".join([*path[path.index(ref) :], ref])
                raise ValueError(f"Expression reference cycle detected: {cycle}")
            if color[ref] == white:
                visit(ref, path)
        path.pop()
        color[node] = black

    for pv in channels:
        if color[pv] == white:
            visit(pv, [])


def _parse_scenarios(raw: Any, channels: dict[str, SimChannel]) -> dict[str, Scenario]:
    """Parse and validate the scenarios section; injects a default 'nominal'."""
    if not isinstance(raw, dict):
        raise ValueError("'scenarios' must be a mapping of scenario name to definition")
    scenarios: dict[str, Scenario] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Scenario {name!r}: definition must be a mapping")

        overrides: dict[str, float | str] = {}
        for pv, value in spec.get("overrides", {}).items():
            if pv not in channels:
                raise ValueError(f"Scenario {name!r}: override for unknown channel {pv!r}")
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ValueError(
                    f"Scenario {name!r}: override for {pv!r} must be a number or string"
                )
            overrides[pv] = value if isinstance(value, str) else float(value)

        archiver: dict[str, list[dict[str, Any]]] = {}
        for entry in spec.get("archiver", []):
            if not isinstance(entry, dict):
                raise ValueError(f"Scenario {name!r}: archiver entries must be mappings")
            pv = entry.get("channel")
            if pv not in channels:
                raise ValueError(f"Scenario {name!r}: archiver events for unknown channel {pv!r}")
            events = entry.get("events", [])
            for event in events:
                _validate_event(name, pv, event, channels[pv])
            archiver[pv] = list(events)

        scenarios[name] = Scenario(
            name=name,
            description=str(spec.get("description", "")),
            overrides=overrides,
            archiver=archiver,
        )

    if DEFAULT_SCENARIO not in scenarios:
        scenarios[DEFAULT_SCENARIO] = Scenario(DEFAULT_SCENARIO, "All systems nominal.", {}, {})
    return scenarios


def _require_event_number(
    prefix: str,
    event: dict[str, Any],
    key: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    """Validate that an event key holds a number within optional bounds.

    With both ``minimum`` and ``maximum`` the value must lie inside the closed
    interval (used for window fractions); with only ``minimum`` it must be
    strictly greater than it (used for positive widths).
    """
    value = event[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{prefix}: event key {key!r} must be a number, got {value!r}")
    if minimum is not None and maximum is not None:
        if not (minimum <= value <= maximum):
            raise ValueError(
                f"{prefix}: event key {key!r} must be between {minimum:g} and "
                f"{maximum:g} (window fraction), got {value!r}"
            )
    elif minimum is not None and value <= minimum:
        raise ValueError(
            f"{prefix}: event key {key!r} must be a number > {minimum:g}, got {value!r}"
        )


def _validate_position_keys(prefix: str, event: dict[str, Any], shape: str) -> None:
    """Validate event position-key *presence*: exactly one of ``at`` / ``at_offset``
    / ``at_time``, plus the ramp until-key pairing (no ``at_time``, no mixing of
    fraction and offset flavors, matching until key present)."""
    has_at = "at" in event
    has_offset = "at_offset" in event
    has_time = "at_time" in event
    if has_at + has_offset + has_time != 1:
        raise ValueError(
            f"{prefix}: event requires exactly one of 'at' (window fraction), "
            f"'at_offset' (seconds relative to scenario activation), or "
            f"'at_time' (daily 'HH:MM:SS' wall-clock time)"
        )
    if shape == "ramp":
        if has_time:
            raise ValueError(
                f"{prefix}: 'ramp' events do not support 'at_time' (use 'step' or 'spike')"
            )
        if (has_at and "until_offset" in event) or (has_offset and "until" in event):
            raise ValueError(
                f"{prefix}: 'ramp' event must not mix fraction and offset position keys"
            )
        until_key = "until" if has_at else "until_offset"
        if until_key not in event:
            raise ValueError(f"{prefix}: 'ramp' event missing keys ['{until_key}']")


def _validate_at_time(prefix: str, raw_time: Any) -> None:
    """Validate an ``at_time`` value: a tz-naive ``'HH:MM:SS'`` local time-of-day."""
    if not isinstance(raw_time, str):
        raise ValueError(
            f"{prefix}: event key 'at_time' must be an 'HH:MM:SS' time string, got {raw_time!r}"
        )
    try:
        parsed_time = dtime.fromisoformat(raw_time)
    except ValueError:
        raise ValueError(
            f"{prefix}: event key 'at_time' must be a valid 'HH:MM:SS' time of day, "
            f"got {raw_time!r}"
        ) from None
    if parsed_time.tzinfo is not None:
        raise ValueError(
            f"{prefix}: event key 'at_time' is local time and must not carry a "
            f"timezone offset, got {raw_time!r}"
        )


def _validate_event(scenario: str, pv: str, event: Any, channel: SimChannel) -> None:
    """Validate a single archiver event object (types and ranges at load time)."""
    prefix = f"Scenario {scenario!r}, channel {pv!r}"
    if not isinstance(event, dict):
        raise ValueError(f"{prefix}: event must be a mapping")
    shape = event.get("shape")
    if shape not in _EVENT_VALUE_KEYS:
        raise ValueError(
            f"{prefix}: event shape must be one of {sorted(_EVENT_VALUE_KEYS)}, got {shape!r}"
        )
    missing = [key for key in _EVENT_VALUE_KEYS[shape] if key not in event]
    if missing:
        raise ValueError(f"{prefix}: {shape!r} event missing keys {missing}")

    _validate_position_keys(prefix, event, shape)

    is_string_channel = channel.expr is None and isinstance(channel.value, str)
    if is_string_channel and shape != "step":
        raise ValueError(
            f"{prefix}: {shape!r} events are not supported on string-valued channels (only 'step')"
        )

    if "at" in event:
        _require_event_number(prefix, event, "at", 0.0, 1.0)
    elif "at_offset" in event:
        _require_event_number(prefix, event, "at_offset")
    else:
        _validate_at_time(prefix, event["at_time"])
    if shape == "ramp":
        if "at" in event:
            _require_event_number(prefix, event, "until", 0.0, 1.0)
        else:
            _require_event_number(prefix, event, "until_offset")
    if shape in ("step", "ramp") and not is_string_channel:
        _require_event_number(prefix, event, "to")
    if shape == "spike":
        _require_event_number(prefix, event, "amplitude")
        _require_event_number(prefix, event, "width", minimum=0.0)
