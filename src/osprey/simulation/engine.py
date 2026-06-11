"""Data-driven simulation engine for the mock connectors.

Loads a machine description (``machine.json``) defining channels (baseline
values or derived expressions), scenarios (override sets plus archiver event
scripts), and serves reads, writes, and synthesized time-series for the mock
control-system and archiver connectors.

Value precedence per channel: session write > active-scenario override >
baseline (``value`` or ``expr``). Derived channels recompute on every read
from the *effective* values of referenced channels, so overrides and writes
propagate through the physics couplings automatically.

The active scenario lives in a plain-text ``active_scenario`` file next to
the machine file; it is re-read whenever its mtime changes, and switching
(or re-asserting) a scenario clears all session-written state (fresh machine).
"""

import ast
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from osprey.simulation.expressions import (
    ExpressionError,
    compile_expression,
    evaluate,
    extract_channel_refs,
)
from osprey.utils.logger import get_logger

logger = get_logger("simulation_engine")

DEFAULT_SCENARIO = "nominal"
ACTIVE_SCENARIO_FILENAME = "active_scenario"

_EVENT_REQUIRED_KEYS = {
    "step": ("at", "to"),
    "ramp": ("at", "until", "to"),
    "spike": ("at", "amplitude", "width"),
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


@dataclass(frozen=True)
class Scenario:
    """Parsed scenario definition: overrides plus archiver event scripts."""

    name: str
    description: str
    overrides: dict[str, float | str]
    archiver: dict[str, list[dict[str, Any]]]


@dataclass(frozen=True)
class SimReading:
    """Result of reading a simulated channel (noise already applied)."""

    value: float | str
    units: str
    description: str


class SimulationEngine:
    """Scenario-driven machine simulation backing the mock connectors."""

    # Cache engines per machine-file path; invalidated when the file's mtime changes.
    _cache: ClassVar[dict[str, tuple[int, "SimulationEngine"]]] = {}

    def __init__(self, machine: dict[str, Any], machine_path: Path):
        """Parse and validate a machine description.

        Args:
            machine: Decoded machine-file JSON.
            machine_path: Path the machine was loaded from (the active-scenario
                state file lives in the same directory).

        Raises:
            ValueError: If the machine description is invalid (bad schema,
                invalid expression, unknown reference, or reference cycle).
        """
        if not isinstance(machine, dict) or not isinstance(machine.get("channels"), dict):
            raise ValueError(f"Machine file {machine_path} must define a 'channels' mapping")

        self.name: str = str(machine.get("name", ""))
        self.description: str = str(machine.get("description", ""))
        self._machine_path = machine_path
        self._state_path = machine_path.parent / ACTIVE_SCENARIO_FILENAME

        self._channels: dict[str, SimChannel] = {
            pv: _parse_channel(pv, spec) for pv, spec in machine["channels"].items()
        }
        _check_references(self._channels)
        self._scenarios: dict[str, Scenario] = _parse_scenarios(
            machine.get("scenarios", {}), self._channels
        )

        self._rng = np.random.default_rng()
        self._written: dict[str, float | str] = {}
        self._active = DEFAULT_SCENARIO
        # Sentinel that never matches a real mtime so the first refresh always runs.
        self._state_mtime_ns: int | None = -1
        self._refresh_scenario()

    @classmethod
    def from_file(cls, path: Path | str) -> "SimulationEngine":
        """Load an engine from a machine file, cached by (path, mtime).

        Args:
            path: Path to the machine JSON file.

        Returns:
            A (possibly cached) engine instance.

        Raises:
            FileNotFoundError: If the machine file does not exist.
            ValueError: If the machine description is invalid.
        """
        resolved = Path(path).expanduser().resolve()
        mtime_ns = resolved.stat().st_mtime_ns
        cached = cls._cache.get(str(resolved))
        if cached is not None and cached[0] == mtime_ns:
            return cached[1]
        with open(resolved) as f:
            machine = json.load(f)
        engine = cls(machine, resolved)
        cls._cache[str(resolved)] = (mtime_ns, engine)
        logger.debug(
            f"Simulation engine loaded: {engine.name!r} ({len(engine._channels)} channels)"
        )
        return engine

    def list_scenarios(self) -> dict[str, str]:
        """Return scenario name -> description for all defined scenarios."""
        return {name: scenario.description for name, scenario in self._scenarios.items()}

    def active_scenario(self) -> str:
        """Return the currently active scenario name (state file re-read if changed)."""
        self._refresh_scenario()
        return self._active

    def set_active_scenario(self, name: str) -> None:
        """Activate a scenario by writing the state file; clears session writes.

        Re-asserting the already-active scenario also clears session writes
        (fresh machine), matching the state-file path.

        Args:
            name: Scenario name (must exist in the machine file).

        Raises:
            ValueError: If the scenario name is unknown.
        """
        if name not in self._scenarios:
            raise ValueError(f"Unknown scenario {name!r}. Available: {sorted(self._scenarios)}")
        self._state_path.write_text(f"{name}\n")
        # Force a re-read even if filesystem mtime granularity hides the write.
        self._state_mtime_ns = -1
        self._refresh_scenario()

    def has_channel(self, pv: str) -> bool:
        """Return True if the machine file defines this channel."""
        return pv in self._channels

    def read(self, pv: str) -> SimReading:
        """Read a channel's effective value with noise applied.

        Args:
            pv: Channel name.

        Returns:
            SimReading with value, units, and description.

        Raises:
            KeyError: If the channel is not defined in the machine file.
        """
        self._refresh_scenario()
        channel = self._require_channel(pv)
        value = self._effective(pv)
        if not isinstance(value, str) and channel.noise > 0.0:
            value = float(value) * (1.0 + float(self._rng.normal(0.0, channel.noise)))
        return SimReading(value=value, units=channel.units, description=channel.description)

    def write(self, pv: str, value: Any) -> None:
        """Record a session write (takes precedence over overrides and baseline).

        Numeric strings are coerced to float: the MCP/CLI write paths deliver
        all values as strings, and storing one verbatim would poison every
        derived channel that references it. Only values that genuinely fail
        ``float()`` are kept as strings (enum-like string channels).

        Args:
            pv: Channel name.
            value: Value to write (numbers are stored as float).

        Raises:
            KeyError: If the channel is not defined in the machine file.
        """
        self._refresh_scenario()
        self._require_channel(pv)
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        self._written[pv] = value if isinstance(value, str) else float(value)

    def synthesize_series(self, pv: str, timestamps: Sequence[Any]) -> list[Any]:
        """Synthesize an archiver time-series for a channel.

        Baseline-value channels yield a constant baseline plus per-channel
        noise, with the active scenario's archiver events (step/ramp/spike)
        applied at window-relative positions. Expression channels are
        evaluated pointwise over the synthesized series of their referenced
        channels, so derived channels show correlated history.

        Args:
            pv: Channel name.
            timestamps: Timestamps of the requested window (only the count
                matters; events land at fixed fractions of any window).

        Returns:
            List of values, one per timestamp.

        Raises:
            KeyError: If the channel is not defined in the machine file.
        """
        self._refresh_scenario()
        self._require_channel(pv)
        n = len(timestamps)
        if n == 0:
            return []
        cache: dict[str, np.ndarray | list[str]] = {}
        series = self._synthesize(pv, n, cache)
        if isinstance(series, np.ndarray):
            return [float(v) for v in series]
        return list(series)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_channel(self, pv: str) -> SimChannel:
        channel = self._channels.get(pv)
        if channel is None:
            raise KeyError(f"Unknown simulation channel {pv!r}")
        return channel

    def _refresh_scenario(self) -> None:
        """Re-read the active-scenario state file when its mtime changes."""
        try:
            mtime_ns: int | None = self._state_path.stat().st_mtime_ns
        except FileNotFoundError:
            mtime_ns = None
        if mtime_ns == self._state_mtime_ns:
            return
        self._state_mtime_ns = mtime_ns

        name = DEFAULT_SCENARIO
        if mtime_ns is not None:
            try:
                raw = self._state_path.read_text().strip()
            except FileNotFoundError:
                raw = ""
            if raw in self._scenarios:
                name = raw
            elif raw:
                logger.warning(
                    f"Unknown scenario {raw!r} in {self._state_path}; "
                    f"falling back to '{DEFAULT_SCENARIO}'"
                )
        if name != self._active:
            self._written.clear()
            logger.info(f"Simulation scenario switched to {name!r} (session writes cleared)")
            self._active = name
        elif self._written:
            # State file touched with the same scenario name: treat as an
            # explicit re-assert and hand back a fresh machine.
            self._written.clear()
            logger.info(f"Simulation scenario {name!r} re-asserted (session writes cleared)")

    def _effective(self, pv: str) -> float | str:
        """Effective value: session write > scenario override > baseline."""
        if pv in self._written:
            return self._written[pv]
        scenario = self._scenarios[self._active]
        if pv in scenario.overrides:
            return scenario.overrides[pv]
        channel = self._channels[pv]
        if channel.expr is not None:
            try:
                return evaluate(channel.expr, self._numeric_effective)
            except ExpressionError as exc:
                raise ExpressionError(f"Channel {pv!r}: {exc}") from exc
            except (ZeroDivisionError, ValueError, OverflowError) as exc:
                raise ExpressionError(
                    f"Channel {pv!r}: expression {channel.expr_source!r} failed to evaluate: {exc}"
                ) from exc
        assert channel.value is not None  # guaranteed by _parse_channel
        return channel.value

    def _numeric_effective(self, pv: str) -> float:
        value = self._effective(pv)
        if isinstance(value, str):
            raise ExpressionError(
                f"Channel {pv!r} holds a string value and cannot be used in an expression"
            )
        return float(value)

    def _synthesize(
        self, pv: str, n: int, cache: dict[str, "np.ndarray | list[str]"]
    ) -> "np.ndarray | list[str]":
        """Build one channel's series, memoized per synthesis pass."""
        cached = cache.get(pv)
        if cached is not None:
            return cached
        channel = self._channels[pv]
        events = self._scenarios[self._active].archiver.get(pv, [])

        if channel.expr is None and isinstance(channel.value, str):
            string_series = _string_series(channel.value, events, n)
            cache[pv] = string_series
            return string_series

        if channel.expr is not None:
            ref_series = {ref: self._synthesize(ref, n, cache) for ref in channel.refs}
            values: list[float] = []
            try:
                for i in range(n):

                    def resolver(name: str, _index: int = i) -> float:
                        return _ref_value(ref_series, name, _index)

                    values.append(evaluate(channel.expr, resolver))
            except ExpressionError as exc:
                raise ExpressionError(f"Channel {pv!r}: {exc}") from exc
            except (ZeroDivisionError, ValueError, OverflowError) as exc:
                raise ExpressionError(
                    f"Channel {pv!r}: expression {channel.expr_source!r} failed to evaluate: {exc}"
                ) from exc
            series = np.asarray(values, dtype=np.float64)
        else:
            assert channel.value is not None  # guaranteed by _parse_channel
            series = np.full(n, float(channel.value))

        series = _apply_events(series, events, n)
        if channel.noise > 0.0:
            series = series * (1.0 + self._rng.normal(0.0, channel.noise, n))
        cache[pv] = series
        return series


def engine_from_connector_config(config: dict[str, Any]) -> SimulationEngine | None:
    """Load a SimulationEngine from a connector config dict, if configured.

    Mirrors ``LimitsValidator.from_config()`` path resolution: a relative
    ``simulation_file`` path is anchored at the configured project root.

    Args:
        config: Connector-scoped config dict (the connector receives the
            already-scoped sub-dict, so the key is just ``simulation_file``).

    Returns:
        The engine, or None when no ``simulation_file`` is configured.
    """
    sim_file = config.get("simulation_file")
    if not sim_file:
        return None
    path = Path(sim_file).expanduser()
    if not path.is_absolute():
        try:
            from osprey.utils.config import get_config_value

            project_root = get_config_value("project_root", None)
        except (FileNotFoundError, KeyError, RuntimeError):
            project_root = None
        if project_root:
            path = Path(project_root) / path
            logger.debug(f"Resolved simulation file path: {path}")
    engine = SimulationEngine.from_file(path)
    logger.info(f"Simulation engine {engine.name!r} active (machine file: {path})")
    return engine


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

    return SimChannel(
        name=pv,
        value=value,
        expr=expr,
        refs=refs,
        units=str(spec.get("units", "")),
        noise=float(noise),
        description=str(spec.get("description", "")),
        expr_source=spec["expr"] if has_expr else "",
    )


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


def _validate_event(scenario: str, pv: str, event: Any, channel: SimChannel) -> None:
    """Validate a single archiver event object (types and ranges at load time)."""
    prefix = f"Scenario {scenario!r}, channel {pv!r}"
    if not isinstance(event, dict):
        raise ValueError(f"{prefix}: event must be a mapping")
    shape = event.get("shape")
    if shape not in _EVENT_REQUIRED_KEYS:
        raise ValueError(
            f"{prefix}: event shape must be one of {sorted(_EVENT_REQUIRED_KEYS)}, got {shape!r}"
        )
    missing = [key for key in _EVENT_REQUIRED_KEYS[shape] if key not in event]
    if missing:
        raise ValueError(f"{prefix}: {shape!r} event missing keys {missing}")

    is_string_channel = channel.expr is None and isinstance(channel.value, str)
    if is_string_channel and shape != "step":
        raise ValueError(
            f"{prefix}: {shape!r} events are not supported on string-valued channels (only 'step')"
        )

    def _require_number(
        key: str, minimum: float | None = None, maximum: float | None = None
    ) -> None:
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

    _require_number("at", 0.0, 1.0)
    if shape == "ramp":
        _require_number("until", 0.0, 1.0)
    if shape in ("step", "ramp") and not is_string_channel:
        _require_number("to")
    if shape == "spike":
        _require_number("amplitude")
        _require_number("width", minimum=0.0)


def _ref_value(ref_series: dict[str, "np.ndarray | list[str]"], name: str, index: int) -> float:
    """Resolver for pointwise expression evaluation over referenced series."""
    value = ref_series[name][index]
    if isinstance(value, str):
        raise ExpressionError(
            f"Channel {name!r} holds string values and cannot be used in an expression"
        )
    return float(value)


def _string_series(baseline: str, events: list[dict[str, Any]], n: int) -> list[str]:
    """Constant string series; only 'step' events are meaningful for strings."""
    t = np.linspace(0.0, 1.0, n)
    series = [baseline] * n
    for event in events:
        if event["shape"] != "step":
            continue
        at = float(event["at"])
        for i in range(n):
            if t[i] >= at:
                series[i] = str(event["to"])
    return series


def _apply_events(series: "np.ndarray", events: list[dict[str, Any]], n: int) -> "np.ndarray":
    """Apply step/ramp/spike events in order at window-relative positions."""
    if not events:
        return series
    t = np.linspace(0.0, 1.0, n)
    series = series.copy()
    for event in events:
        shape = event["shape"]
        at = float(event["at"])
        if shape == "step":
            series[t >= at] = float(event["to"])
        elif shape == "ramp":
            until = float(event["until"])
            to = float(event["to"])
            if until <= at:
                series[t >= at] = to
                continue
            idx = int(np.searchsorted(t, at))
            start = float(series[min(idx, n - 1)])
            mask = (t >= at) & (t <= until)
            series[mask] = start + (to - start) * (t[mask] - at) / (until - at)
            series[t > until] = to
        else:  # spike (gaussian bump, width as fraction of window)
            amplitude = float(event["amplitude"])
            width = float(event["width"])
            series = series + amplitude * np.exp(-((t - at) ** 2) / (2.0 * width**2))
    return series
