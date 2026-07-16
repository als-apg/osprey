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
import json
from dataclasses import dataclass, field
from datetime import time as dtime
from pathlib import Path
from typing import Any

from osprey.simulation.expressions import (
    ExpressionError,
    compile_expression,
    extract_channel_refs,
)
from osprey.utils.relative_time import RelativeTimestamp

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
class ScenarioLogEntry:
    """A logbook entry owned by a scenario bundle.

    Single source of truth for both the ARIEL DB seed (via ``apply``) and the
    fast per-scenario unit tests, so the telemetry overlay and its narrative
    ship together in one bundle.
    """

    entry_id: str
    when: RelativeTimestamp
    author: str
    title: str
    text: str
    tags: tuple[str, ...]
    categories: tuple[str, ...]
    loto_tag: str | None
    extra: dict[str, Any]


@dataclass(frozen=True)
class BpmErrorSpec:
    """One BPM's seeded error model: offset/gain/polarity/roll/noise.

    Defaults are the identity error (no distortion), mirroring the
    :class:`~osprey.services.virtual_accelerator.ioc.physics_bridge.PhysicsBridge`
    per-BPM error state. ``offset``/``roll``/``noise`` are in the bridge's
    native units (m / rad / m stdev); ``polarity`` is a sign flip, ``gain`` a
    readback scale factor.
    """

    offset: float = 0.0
    gain: float = 1.0
    polarity: int = 1
    roll: float = 0.0
    noise: float = 0.0


@dataclass(frozen=True)
class PhysicsFault:
    """Deploy-time VA physics-fault values a scenario seeds at container boot.

    Keyed by lattice device id (family + zero-padded device number, e.g.
    ``"QF07"``, ``"BPM12"``, ``"HCM01"`` — the same ids
    ``lattice/inventory.py::pyat_coupled_device_ids`` produces), NOT by EPICS
    channel name, and NOT a mock-channel override: it never touches the
    ``overrides``/``channels`` validation path. A render step resolves this
    into VA env (``VA_BPM_ERRORS``/``VA_CORR_GAIN``)
    before ``deploy up``; the VA applies it once at boot (hot-swapping a
    physics fault requires a restart, unlike ``overrides``/``archiver``).
    """

    bpm_errors: dict[str, BpmErrorSpec] = field(default_factory=dict)
    corrector_gain: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    """Parsed scenario definition: overrides, archiver event scripts, logbook.

    ``logbook`` is empty for inline (machine-dict) scenarios; only filesystem
    scenario bundles carry logbook entries (see :func:`load_scenario_bundles`).
    ``physics`` is ``None`` unless the bundle defines a ``physics`` block
    (see :class:`PhysicsFault`); absent block means no physics fault.
    """

    name: str
    description: str
    overrides: dict[str, float | str]
    archiver: dict[str, list[dict[str, Any]]]
    logbook: tuple[ScenarioLogEntry, ...] = field(default_factory=tuple)
    physics: PhysicsFault | None = None


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
    scenarios_dir = machine_path.parent / "scenarios"
    if scenarios_dir.is_dir():
        scenarios = load_scenario_bundles(scenarios_dir, channels)
    else:
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
    """Parse and validate an inline scenarios section; injects a default 'nominal'.

    Used for machine-dict (filesystem-free) inputs. Filesystem scenario bundles
    take the :func:`load_scenario_bundles` path instead.
    """
    if not isinstance(raw, dict):
        raise ValueError("'scenarios' must be a mapping of scenario name to definition")
    scenarios: dict[str, Scenario] = {}
    for name, spec in raw.items():
        scenarios[name] = _parse_scenario_spec(name, spec, channels)

    if DEFAULT_SCENARIO not in scenarios:
        scenarios[DEFAULT_SCENARIO] = _default_nominal()
    return scenarios


def _parse_scenario_spec(
    name: str,
    spec: Any,
    channels: dict[str, SimChannel],
    logbook: tuple[ScenarioLogEntry, ...] = (),
) -> Scenario:
    """Validate one scenario's overrides + archiver events into a :class:`Scenario`.

    Shared by the inline (:func:`_parse_scenarios`) and bundle
    (:func:`load_scenario_bundles`) paths so override/event validation lives in
    exactly one place. ``logbook`` is supplied only by the bundle path.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"Scenario {name!r}: definition must be a mapping")

    overrides: dict[str, float | str] = {}
    for pv, value in spec.get("overrides", {}).items():
        if pv not in channels:
            raise ValueError(f"Scenario {name!r}: override for unknown channel {pv!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(f"Scenario {name!r}: override for {pv!r} must be a number or string")
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

    physics = _parse_physics_fault(name, spec.get("physics"))

    return Scenario(
        name=name,
        description=str(spec.get("description", "")),
        overrides=overrides,
        archiver=archiver,
        logbook=logbook,
        physics=physics,
    )


def _default_nominal() -> Scenario:
    """The auto-injected baseline scenario used when none is defined."""
    return Scenario(DEFAULT_SCENARIO, "All systems nominal.", {}, {}, ())


def _parse_physics_fault(scenario_name: str, raw: Any) -> PhysicsFault | None:
    """Parse and validate an optional 'physics' fault block.

    ``physics`` is a sibling of ``overrides``/``archiver``, not a variant of
    either — its device ids are lattice element ids, never checked against
    ``channels``, so it cannot trip the mock-channel-override validation.
    Absent block (``raw is None``) means no physics fault.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario {scenario_name!r}: 'physics' must be a mapping")

    prefix = f"Scenario {scenario_name!r} physics"
    return PhysicsFault(
        bpm_errors=_parse_bpm_errors(prefix, raw.get("bpm_errors", {})),
        corrector_gain=_parse_corrector_gain(prefix, raw.get("corrector_gain", {})),
    )


def _parse_device_number(prefix: str, block: str, device: Any, value: Any) -> tuple[str, float]:
    """Validate one ``device id -> number`` entry shared by corrector_gain."""
    if not isinstance(device, str) or not device:
        raise ValueError(f"{prefix}: {block!r} keys must be non-empty device id strings")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{prefix}: {block}[{device!r}] must be a number, got {value!r}")
    return device, float(value)


def _parse_corrector_gain(prefix: str, raw: Any) -> dict[str, float]:
    """Parse 'corrector_gain': device id -> calibration factor (1.0 = nominal)."""
    if not isinstance(raw, dict):
        raise ValueError(f"{prefix}: 'corrector_gain' must be a mapping of device id to factor")
    return dict(_parse_device_number(prefix, "corrector_gain", d, v) for d, v in raw.items())


def _parse_bpm_errors(prefix: str, raw: Any) -> dict[str, BpmErrorSpec]:
    """Parse 'bpm_errors': device id -> :class:`BpmErrorSpec` (all keys optional)."""
    if not isinstance(raw, dict):
        raise ValueError(f"{prefix}: 'bpm_errors' must be a mapping of device id to an error spec")
    result: dict[str, BpmErrorSpec] = {}
    for device, spec in raw.items():
        if not isinstance(device, str) or not device:
            raise ValueError(f"{prefix}: 'bpm_errors' keys must be non-empty device id strings")
        if not isinstance(spec, dict):
            raise ValueError(f"{prefix}: bpm_errors[{device!r}] must be a mapping")
        entry_prefix = f"{prefix} bpm_errors[{device!r}]"
        polarity = spec.get("polarity", 1)
        if isinstance(polarity, bool) or polarity not in (1, -1):
            raise ValueError(f"{entry_prefix}: 'polarity' must be 1 or -1, got {polarity!r}")
        result[device] = BpmErrorSpec(
            offset=_bpm_error_number(entry_prefix, spec, "offset", 0.0),
            gain=_bpm_error_number(entry_prefix, spec, "gain", 1.0),
            polarity=int(polarity),
            roll=_bpm_error_number(entry_prefix, spec, "roll", 0.0),
            noise=_bpm_error_number(entry_prefix, spec, "noise", 0.0, minimum=0.0),
        )
    return result


def _bpm_error_number(
    prefix: str, spec: dict[str, Any], key: str, default: float, minimum: float | None = None
) -> float:
    """Validate an optional numeric ``bpm_errors`` field, falling back to ``default``."""
    if key not in spec:
        return default
    value = spec[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{prefix}: {key!r} must be a number, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{prefix}: {key!r} must be >= {minimum:g}, got {value!r}")
    return float(value)


def load_scenario_bundles(
    scenarios_dir: Path, channels: dict[str, SimChannel]
) -> dict[str, Scenario]:
    """Load self-contained scenario bundles from a ``scenarios/`` directory.

    Each immediate subdirectory is a bundle named after the directory: a
    required ``scenario.json`` (``description`` plus optional ``overrides`` /
    ``archiver``, same schema as an inline scenario) and an optional
    ``logbook.json`` (a JSON array of entries with relative timestamps). A
    default ``nominal`` is injected if no ``nominal/`` bundle exists.

    Args:
        scenarios_dir: The ``scenarios/`` directory (sibling of the machine file).
        channels: Parsed channels, for override/event reference validation.

    Returns:
        Scenario name -> :class:`Scenario`, each carrying its logbook entries.

    Raises:
        ValueError: If a bundle is malformed (missing ``scenario.json``, invalid
            scenario/logbook schema, or references an unknown channel).
    """
    scenarios: dict[str, Scenario] = {}
    for bundle in sorted(p for p in scenarios_dir.iterdir() if p.is_dir()):
        name = bundle.name
        scenario_file = bundle / "scenario.json"
        if not scenario_file.is_file():
            raise ValueError(f"Scenario bundle {name!r} is missing scenario.json ({scenario_file})")
        try:
            spec = json.loads(scenario_file.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Scenario bundle {name!r}: invalid scenario.json: {exc}") from exc

        logbook: tuple[ScenarioLogEntry, ...] = ()
        logbook_file = bundle / "logbook.json"
        if logbook_file.is_file():
            try:
                raw_logbook = json.loads(logbook_file.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(f"Scenario bundle {name!r}: invalid logbook.json: {exc}") from exc
            if not isinstance(raw_logbook, list):
                raise ValueError(f"Scenario bundle {name!r}: logbook.json must be a JSON array")
            logbook = tuple(_parse_log_entry(name, entry) for entry in raw_logbook)

        scenarios[name] = _parse_scenario_spec(name, spec, channels, logbook=logbook)

    if DEFAULT_SCENARIO not in scenarios:
        scenarios[DEFAULT_SCENARIO] = _default_nominal()
    return scenarios


def _parse_relative_timestamp(prefix: str, raw: Any) -> RelativeTimestamp:
    """Parse a ``{days_ago, time}`` relative timestamp (reuses at_time rules)."""
    if not isinstance(raw, dict):
        raise ValueError(f"{prefix}: 'when' must be a mapping with 'days_ago' and 'time'")
    days_ago = raw.get("days_ago")
    if isinstance(days_ago, bool) or not isinstance(days_ago, int) or days_ago < 0:
        raise ValueError(f"{prefix}: 'days_ago' must be a non-negative integer, got {days_ago!r}")
    raw_time = raw.get("time")
    _validate_at_time(prefix, raw_time)
    return RelativeTimestamp(days_ago=days_ago, time=dtime.fromisoformat(raw_time))


def _parse_log_entry(scenario_name: str, raw: Any) -> ScenarioLogEntry:
    """Parse and validate one logbook entry from a bundle's logbook.json."""
    prefix = f"Scenario {scenario_name!r} logbook"
    if not isinstance(raw, dict):
        raise ValueError(f"{prefix}: each entry must be a mapping")
    entry_id = raw.get("entry_id")
    if not isinstance(entry_id, str) or not entry_id:
        raise ValueError(f"{prefix}: 'entry_id' must be a non-empty string, got {entry_id!r}")
    entry_prefix = f"{prefix} entry {entry_id!r}"
    when = _parse_relative_timestamp(entry_prefix, raw.get("when"))

    def _require_str(key: str) -> str:
        value = raw.get(key)
        if not isinstance(value, str):
            raise ValueError(f"{entry_prefix}: {key!r} must be a string, got {value!r}")
        return value

    def _str_tuple(key: str) -> tuple[str, ...]:
        value = raw.get(key, [])
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError(f"{entry_prefix}: {key!r} must be a list of strings, got {value!r}")
        return tuple(value)

    loto_tag = raw.get("loto_tag")
    if loto_tag is not None and not isinstance(loto_tag, str):
        raise ValueError(f"{entry_prefix}: 'loto_tag' must be a string or null, got {loto_tag!r}")
    extra = raw.get("extra", {})
    if not isinstance(extra, dict):
        raise ValueError(f"{entry_prefix}: 'extra' must be a mapping, got {extra!r}")

    return ScenarioLogEntry(
        entry_id=entry_id,
        when=when,
        author=_require_str("author"),
        title=_require_str("title"),
        text=_require_str("text"),
        tags=_str_tuple("tags"),
        categories=_str_tuple("categories"),
        loto_tag=loto_tag,
        extra=dict(extra),
    )


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
