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

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from osprey.simulation.expressions import ExpressionError, evaluate_channel
from osprey.simulation.machine import (
    DEFAULT_SCENARIO,
    Scenario,
    SimChannel,
    parse_machine,
)
from osprey.simulation.series import (
    apply_events,
    clamp,
    epoch_seconds_array,
    ref_value,
    string_series,
)
from osprey.utils.logger import get_logger

logger = get_logger("simulation_engine")

ACTIVE_SCENARIO_FILENAME = "active_scenario"


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
        model = parse_machine(machine, machine_path)

        self.name: str = model.name
        self.description: str = model.description
        self._machine_path = machine_path
        self._state_path = machine_path.parent / ACTIVE_SCENARIO_FILENAME
        self._channels: dict[str, SimChannel] = model.channels
        self._scenarios: dict[str, Scenario] = model.scenarios

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
        if not isinstance(value, str):
            if channel.noise > 0.0:
                value = float(value) * (1.0 + float(self._rng.normal(0.0, channel.noise)))
            value = clamp(float(value), channel.min_value, channel.max_value)
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
        applied. Expression channels are evaluated pointwise over the
        synthesized series of their referenced channels, so derived channels
        show correlated history.

        Event positioning has three flavors: ``at`` places an event at a fixed
        fraction of whatever window is requested, while ``at_offset`` (with
        ``until_offset`` for ramps) anchors it in wall-clock time, in seconds
        relative to the scenario-activation time (the ``active_scenario``
        state-file mtime; negative = past). ``at_time`` (``"HH:MM:SS"``, local
        time; step/spike only) recurs daily: the event fires at that time of
        day on every calendar date inside the requested window. Anchored and
        time-of-day events honor the actual timestamp values, so an event
        outside the requested window does not appear in it. For anchored and
        time-of-day spikes, ``width`` is in seconds.

        Args:
            pv: Channel name.
            timestamps: Timestamps of the requested window (datetime objects
                or epoch seconds). For fraction-positioned events only the
                count matters; anchored events use the actual values.

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
        t_abs = epoch_seconds_array(timestamps)
        anchor = self._scenario_anchor()
        cache: dict[str, np.ndarray | list[str]] = {}
        series = self._synthesize(pv, n, cache, t_abs, anchor)
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
            return evaluate_channel(channel.expr, channel.expr_source, pv, self._numeric_effective)
        assert channel.value is not None  # guaranteed by parse_machine
        return channel.value

    def _numeric_effective(self, pv: str) -> float:
        value = self._effective(pv)
        if isinstance(value, str):
            raise ExpressionError(
                f"Channel {pv!r} holds a string value and cannot be used in an expression"
            )
        channel = self._channels[pv]
        return clamp(float(value), channel.min_value, channel.max_value)

    def _scenario_anchor(self) -> float:
        """Scenario-activation time in epoch seconds (state-file mtime).

        Falls back to the current time when no state file exists (default
        scenario was never explicitly activated).
        """
        if self._state_mtime_ns is not None and self._state_mtime_ns > 0:
            return self._state_mtime_ns / 1e9
        return time.time()

    def _synthesize(
        self,
        pv: str,
        n: int,
        cache: dict[str, "np.ndarray | list[str]"],
        t_abs: "np.ndarray | None",
        anchor: float,
    ) -> "np.ndarray | list[str]":
        """Build one channel's series, memoized per synthesis pass."""
        cached = cache.get(pv)
        if cached is not None:
            return cached
        channel = self._channels[pv]
        events = self._scenarios[self._active].archiver.get(pv, [])

        if channel.expr is None and isinstance(channel.value, str):
            series_str = string_series(channel.value, events, n, t_abs, anchor)
            cache[pv] = series_str
            return series_str

        if channel.expr is not None:
            ref_series = {
                ref: self._synthesize(ref, n, cache, t_abs, anchor) for ref in channel.refs
            }
            values: list[float] = []
            for i in range(n):

                def resolver(name: str, _index: int = i) -> float:
                    return ref_value(ref_series, name, _index)

                values.append(evaluate_channel(channel.expr, channel.expr_source, pv, resolver))
            series = np.asarray(values, dtype=np.float64)
        else:
            assert channel.value is not None  # guaranteed by parse_machine
            series = np.full(n, float(channel.value))

        series = apply_events(series, events, n, t_abs, anchor)
        if channel.noise > 0.0:
            series = series * (1.0 + self._rng.normal(0.0, channel.noise, n))
        if channel.min_value is not None or channel.max_value is not None:
            series = np.clip(series, channel.min_value, channel.max_value)
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


def engine_serves(engine: SimulationEngine | None, channel: str) -> bool:
    """Return True if an engine is present and serves this channel.

    Centralises the optional-engine guard used by the mock connectors: the
    engine is None when no ``simulation_file`` is configured, in which case the
    connector falls back to its generic procedural synthesis.
    """
    return engine is not None and engine.has_channel(channel)
