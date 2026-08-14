"""Data-driven simulation engine for the mock connectors.

Loads a machine description (``machine.json``) defining channels (baseline
values or derived expressions), scenarios (override sets plus archiver event
scripts), and serves reads, writes, and synthesized time-series for the mock
control-system and archiver connectors.

Value precedence per channel: session write > active-scenario override >
baseline (``value`` or ``expr``). Derived channels recompute on every read
from the *effective* values of referenced channels, so overrides and writes
propagate through the physics couplings automatically.

The active scenarios live in a plain-text ``active_scenarios`` file under the
agent-data root (``agent_data.base_dir``, in its ``simulation/`` subdirectory
— the machine file's own directory is build-owned and checksummed, so mutable
state stays out of it): an optional ``anchor=<ISO8601>`` metadata line followed by
one scenario name per line (``nominal`` is always implicitly first). It is
re-read whenever its mtime changes, and switching (or re-asserting) the set
clears all session-written state (fresh machine). Simultaneously active
scenarios must touch disjoint channel sets (see :meth:`validate_composition`);
their overrides and archiver scripts are merged into one composed view. The
legacy single-name ``active_scenario`` file is still read (one-element list)
for backward compatibility, but writes always target ``active_scenarios``.
"""

import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

import numpy as np

from osprey_connectors.config import get_facility_timezone
from osprey_connectors.logger import get_logger
from osprey_connectors.simulation.expressions import ExpressionError, evaluate_channel
from osprey_connectors.simulation.machine import (
    DEFAULT_SCENARIO,
    Scenario,
    ScenarioLogEntry,
    SimChannel,
    TextureSpec,
    parse_machine,
)
from osprey_connectors.simulation.series import (
    apply_events,
    clamp,
    epoch_seconds_array,
    keyed_normals,
    pv_key_bytes,
    ref_value,
    string_series,
    wander,
)
from osprey_connectors.workspace import (
    SIMULATION_STATE_DIR_CONFIG_KEY,
    SIMULATION_STATE_DIR_NAME,
    resolve_simulation_state_dir,
)

logger = get_logger("simulation_engine")

ACTIVE_SCENARIOS_FILENAME = "active_scenarios"
# Legacy single-name state file; read for back-compat, never written.
ACTIVE_SCENARIO_FILENAME = "active_scenario"

#: Config key naming the state directory explicitly (relative paths resolve
#: against the project root). Unset — the normal case — puts it under the
#: agent-data root. Defined in :mod:`osprey_connectors.workspace` and re-exported
#: here so this module and :data:`osprey_connectors.config.RUNTIME_WRITE_PATH_KEYS`
#: cannot drift to different spellings.
STATE_DIR_CONFIG_KEY = SIMULATION_STATE_DIR_CONFIG_KEY

#: Subdirectory of the agent-data root holding the scenario state file.
STATE_DIR_NAME = SIMULATION_STATE_DIR_NAME


#: Resolve the directory holding the ``active_scenarios`` state file. Defined in
#: :mod:`osprey_connectors.workspace` — the engine, the compose generator that renders
#: the container's bind-mount source, and the build injector that pre-creates the
#: directory must agree on it, and the latter two must not import numpy through
#: this module to ask.
resolve_state_dir = resolve_simulation_state_dir


def resolve_active_scenarios(names: Sequence[str]) -> list[str]:
    """The active scenario set a request for ``names`` really means.

    ``nominal`` is the machine's baseline, not a fault: it is always active, so
    it is prepended whether or not the caller named it. The rest keep the
    caller's order and are deduplicated, because activating a scenario twice
    would compose its physics twice.

    One function for a rule two callers depend on — activation writes the state
    file from it, and physics-fault rendering derives ``VA_*`` variables from
    it — so the environment a project is built with and the scenarios its
    engine runs cannot describe different machines.
    """
    resolved: list[str] = [DEFAULT_SCENARIO]
    for name in names:
        if name != DEFAULT_SCENARIO and name not in resolved:
            resolved.append(name)
    return resolved


def default_state_dir() -> Path:
    """Resolve the state directory from the ambient config.

    For callers holding neither a project root nor a loaded config — the mock
    connectors, which are constructed from an already-scoped config sub-dict.
    Callers that do have them (the ``sim`` CLI,
    :func:`osprey.simulation.apply.apply_scenarios`, the VA entrypoint) pass
    ``state_dir`` explicitly instead of relying on this.
    """
    try:
        from osprey_connectors.config import get_config_builder

        builder = get_config_builder()
        config = builder.raw_config
        project_root = Path(config.get("project_root") or builder.config_path.parent)
    except (FileNotFoundError, IsADirectoryError, KeyError, RuntimeError, ValueError):
        config, project_root = {}, Path.cwd()
    return resolve_state_dir(config, project_root)


@dataclass(frozen=True)
class SimReading:
    """Result of reading a simulated channel (noise already applied)."""

    value: float | str
    units: str
    description: str


def _texture_offset(
    pv_key: bytes, texture: TextureSpec, t_abs_s: "np.ndarray | float"
) -> "np.ndarray":
    """Texture contribution at absolute epoch time(s) — shared by both engine paths.

    Synthesis passes the window's epoch-seconds array; live reads pass a scalar
    wall-clock now (``wander`` is scalar/array bit-exact, so the two paths
    agree exactly at a shared timestamp). That agreement holds only because
    both callers route through this one implementation with UNMODIFIED
    absolute epoch seconds — do not normalize, quantize, or re-derive the time
    input on either side. ``kind`` needs no dispatch: ``parse_machine``
    validates the vocabulary and ``"wander"`` is the only kind defined today.

    Args:
        pv_key: Channel key from :func:`osprey_connectors.simulation.series.pv_key_bytes`.
        texture: The channel's declared texture parameters.
        t_abs_s: Absolute epoch seconds — array (synthesis) or scalar (live).

    Returns:
        Texture offsets with the shape of ``t_abs_s`` (0-d for a scalar).
    """
    # asarray is the same first step wander performs — a bit-exact dtype wrap
    # (never a rounding or rebasing), here only to satisfy the array annotation.
    times = np.asarray(t_abs_s, dtype=np.float64)
    return wander(pv_key, times, texture.amplitude, texture.period_s)


def _apply_signal_model(
    pv: str, channel: SimChannel, series: "np.ndarray", t_abs: "np.ndarray | None"
) -> "np.ndarray":
    """Add a channel's declared signal model to its post-event baseline series.

    Texture rides *post-event* levels: step/ramp events hard-overwrite the
    series, so folding it in earlier would leave every stepped plateau dead
    flat. It needs absolute time (that is what makes overlapping windows agree),
    so non-epoch timestamps skip it — same precedent as anchored events in
    :func:`osprey_connectors.simulation.series.apply_events`.

    Both noise terms are keyed draws addressed by (channel, epoch millisecond),
    never by an RNG stream position, so any two windows agree pointwise on
    shared timestamps. The sample-index fallback keeps non-epoch windows
    deterministic (still per-channel via the key).

    Args:
        pv: Channel name; keys the draws and names the channel in the
            skipped-texture debug log.
        channel: Parsed channel supplying ``texture``/``noise``/``noise_abs``.
        series: Post-event baseline series, one entry per sample. Not mutated.
        t_abs: Absolute epoch seconds per sample, or ``None`` when the requested
            timestamps were not convertible.

    Returns:
        The series with texture and both noise terms applied.
    """
    pv_key = pv_key_bytes(pv)
    if channel.texture is not None:
        if t_abs is not None:
            series = series + _texture_offset(pv_key, channel.texture, t_abs)
        else:
            logger.debug(
                f"Skipping texture for {pv!r}: timestamps not convertible to epoch seconds"
            )
    if channel.noise > 0.0 or channel.noise_abs > 0.0:
        counters = t_abs * 1000.0 if t_abs is not None else np.arange(len(series), dtype=np.int64)
        if channel.noise > 0.0:
            relative = keyed_normals(pv_key + b":noise", counters)
            series = series * (1.0 + channel.noise * relative)
        if channel.noise_abs > 0.0:
            absolute = keyed_normals(pv_key + b":noise_abs", counters)
            series = series + channel.noise_abs * absolute
    return series


class SimulationEngine:
    """Scenario-driven machine simulation backing the mock connectors."""

    # Cache engines per (machine-file path, state dir); invalidated when the
    # machine file's mtime changes.
    _cache: ClassVar[dict[tuple[str, str], tuple[int, "SimulationEngine"]]] = {}

    def __init__(
        self,
        machine: dict[str, Any],
        machine_path: Path,
        state_dir: Path | str | None = None,
    ):
        """Parse and validate a machine description.

        Args:
            machine: Decoded machine-file JSON.
            machine_path: Path the machine was loaded from.
            state_dir: Directory holding the ``active_scenarios`` state file.
                Defaults to :func:`default_state_dir` (under the agent-data
                root) — deliberately *not* the machine file's own directory,
                which is build-owned and checksummed.

        Raises:
            ValueError: If the machine description is invalid (bad schema,
                invalid expression, unknown reference, or reference cycle).
        """
        model = parse_machine(machine, machine_path)

        self.name: str = model.name
        self.description: str = model.description
        self._machine_path = machine_path
        self._state_dir = (
            Path(state_dir).expanduser() if state_dir is not None else default_state_dir()
        )
        # Canonical (write) state file plus the legacy single-name file (read-only).
        self._state_path = self._state_dir / ACTIVE_SCENARIOS_FILENAME
        self._legacy_state_path = self._state_dir / ACTIVE_SCENARIO_FILENAME
        self._channels: dict[str, SimChannel] = model.channels
        self._scenarios: dict[str, Scenario] = model.scenarios

        self._rng = np.random.default_rng()
        self._written: dict[str, float | str] = {}
        self._active: tuple[str, ...] = (DEFAULT_SCENARIO,)
        self._composed_overrides: dict[str, float | str] = {}
        self._composed_archiver: dict[str, list[dict[str, Any]]] = {}
        self._anchor_epoch: float | None = None
        self._state_mtime_ns: int | None = None
        # Sentinel that never matches a real (path, mtime) so first refresh runs.
        self._state_signature: tuple[str | None, int | None] = ("", -1)
        self._recompose()
        self._refresh_scenario()

    @classmethod
    def from_file(cls, path: Path | str, state_dir: Path | str | None = None) -> "SimulationEngine":
        """Load an engine from a machine file, cached by (path, state dir, mtime).

        Args:
            path: Path to the machine JSON file.
            state_dir: Directory holding the ``active_scenarios`` state file;
                see :meth:`__init__`. Part of the cache key, so two engines on
                one machine file but different state dirs never alias.

        Returns:
            A (possibly cached) engine instance.

        Raises:
            FileNotFoundError: If the machine file does not exist.
            ValueError: If the machine description is invalid.
        """
        resolved = Path(path).expanduser().resolve()
        # Resolved like the machine path above: two spellings of one directory
        # must not key two engines onto the same state file.
        resolved_state_dir = (
            Path(state_dir) if state_dir is not None else default_state_dir()
        ).expanduser()
        resolved_state_dir = resolved_state_dir.resolve()
        mtime_ns = cls._machine_mtime_ns(resolved)
        cache_key = (str(resolved), str(resolved_state_dir))
        cached = cls._cache.get(cache_key)
        if cached is not None and cached[0] == mtime_ns:
            return cached[1]
        with open(resolved) as f:
            machine = json.load(f)
        engine = cls(machine, resolved, state_dir=resolved_state_dir)
        cls._cache[cache_key] = (mtime_ns, engine)
        logger.debug(
            f"Simulation engine loaded: {engine.name!r} ({len(engine._channels)} channels)"
        )
        return engine

    @staticmethod
    def _machine_mtime_ns(machine_path: Path) -> int:
        """Cache-key mtime: newest of the machine file and every bundle JSON.

        Keying on the machine file alone would serve a stale cached engine after
        an edit to a ``scenarios/<name>/scenario.json`` or ``logbook.json``; the
        ``active_scenarios`` state file is deliberately excluded (it has its own
        mtime-based refresh in :meth:`_refresh_scenario`).
        """
        latest = machine_path.stat().st_mtime_ns
        scenarios_dir = machine_path.parent / "scenarios"
        if scenarios_dir.is_dir():
            for bundle_file in scenarios_dir.rglob("*.json"):
                latest = max(latest, bundle_file.stat().st_mtime_ns)
        return latest

    def list_scenarios(self) -> dict[str, str]:
        """Return scenario name -> description for all defined scenarios."""
        return {name: scenario.description for name, scenario in self._scenarios.items()}

    def scenario_logbook(self, name: str) -> tuple[ScenarioLogEntry, ...]:
        """Return the logbook entries a scenario bundle owns (empty if none)."""
        return self._scenarios[name].logbook

    def active_logbook(self) -> list[ScenarioLogEntry]:
        """Return all logbook entries owned by the active scenarios.

        Concatenated in activation order (nominal-first), so the composed set
        carries every active scenario's narrative. Telemetry-only scenarios
        (no ``logbook.json``) contribute nothing.
        """
        self._refresh_scenario()
        entries: list[ScenarioLogEntry] = []
        for name in self._active:
            entries.extend(self._scenarios[name].logbook)
        return entries

    def active_scenarios(self) -> tuple[str, ...]:
        """Return the active scenario set (nominal-first; state file re-read if changed)."""
        self._refresh_scenario()
        return self._active

    def active_scenario(self) -> str:
        """Return the last-activated scenario name (single-element back-compat view).

        For a nominal-only machine this is ``'nominal'``; for one active fault it
        is that fault. Prefer :meth:`active_scenarios` for the full set.
        """
        self._refresh_scenario()
        return self._active[-1]

    def set_active_scenarios(
        self, names: Sequence[str], anchor: datetime | None = None
    ) -> tuple[str, ...]:
        """Activate a composed set of scenarios by writing the state file.

        ``nominal`` is always implicitly active and prepended. The requested set
        must touch disjoint channel sets (see :meth:`validate_composition`).
        Writing the state file clears session writes (fresh machine). Always
        writes the canonical ``active_scenarios`` file.

        Args:
            names: Scenario names to activate (order preserved, deduped).
            anchor: Optional apply-time anchor T0 written as an ``anchor=`` line;
                shared by telemetry and logbook so both resolve against one clock.
                When omitted, the engine falls back to the state-file mtime.

        Returns:
            The resolved active set (nominal-first).

        Raises:
            ValueError: If a name is unknown or the set does not compose.
        """
        resolved = resolve_active_scenarios(names)
        problems = self.validate_composition(resolved)
        if problems:
            raise ValueError("Cannot activate scenarios: " + "; ".join(problems))

        body: list[str] = []
        if anchor is not None:
            body.append(f"anchor={anchor.isoformat()}")
        faults = [n for n in resolved if n != DEFAULT_SCENARIO]
        body.extend(faults if faults else [DEFAULT_SCENARIO])
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic rename, never a truncate-in-place: the VA IOC polls this file
        # from a bind mount once a second, and a torn read would hand it a
        # half-written scenario set. The temp file is a sibling so the rename
        # stays within one filesystem.
        tmp_path = self._state_path.with_name(f"{self._state_path.name}.tmp")
        tmp_path.write_text("\n".join(body) + "\n")
        os.replace(tmp_path, self._state_path)
        # Force a re-read even if filesystem mtime granularity hides the write.
        self._state_signature = ("", -1)
        self._refresh_scenario()
        return self._active

    def set_active_scenario(self, name: str) -> None:
        """Activate a single scenario (back-compat wrapper over :meth:`set_active_scenarios`).

        Args:
            name: Scenario name (must exist in the machine file).

        Raises:
            ValueError: If the scenario name is unknown.
        """
        self.set_active_scenarios([name])

    def validate_composition(self, names: Sequence[str]) -> list[str]:
        """Return composition problems for a set of scenarios; empty list = OK.

        Active scenarios must touch *disjoint* channel sets — a channel is
        "touched" if a scenario declares an override or an archiver block for it.
        Archiver step/ramp events overwrite the synthesized series (and overrides
        collide on point reads), so two scenarios touching one channel compose
        order-dependently and silently wrong. Returns a message per unknown name
        and per channel collision.

        Args:
            names: Scenario names to check (typically the resolved active set).

        Returns:
            Human-readable problem strings; empty when the set composes cleanly.
        """
        problems: list[str] = []
        owner: dict[str, str] = {}
        for name in names:
            if name not in self._scenarios:
                problems.append(f"Unknown scenario {name!r}. Available: {sorted(self._scenarios)}")
                continue
            scenario = self._scenarios[name]
            touched = set(scenario.overrides) | set(scenario.archiver)
            for pv in sorted(touched):
                if pv in owner and owner[pv] != name:
                    problems.append(
                        f"Channel {pv!r} is touched by both {owner[pv]!r} and {name!r}; "
                        f"active scenarios must touch disjoint channel sets"
                    )
                else:
                    owner[pv] = name
        return problems

    def has_channel(self, pv: str) -> bool:
        """Return True if the machine file defines this channel."""
        return pv in self._channels

    def read(self, pv: str) -> SimReading:
        """Read a channel's effective value with the live signal model applied.

        Numeric pipeline: effective value -> + texture at wall-clock now ->
        relative ``noise`` (multiplicative) -> + ``noise_abs`` (additive) ->
        clamp. Texture is the shared deterministic term (:func:`_texture_offset`,
        the same implementation synthesis uses), so for a NON-OVERRIDDEN,
        noise-free channel a live read equals the synthesized sample at the
        same timestamp exactly. That agreement is scoped: the effective value
        resolves write > override > baseline, while synthesized history builds
        on ``channel.value`` plus archiver events — an override shifts the live
        read but not the history (scenarios declare matching archiver events to
        keep the two consistent). Both noise draws stay stochastic on the
        engine's RNG, deliberately: only synthesis draws are counter-based.

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
            value = float(value)
            if channel.texture is not None:
                value += float(_texture_offset(pv_key_bytes(pv), channel.texture, time.time()))
            if channel.noise > 0.0:
                value *= 1.0 + float(self._rng.normal(0.0, channel.noise))
            if channel.noise_abs > 0.0:
                value += float(self._rng.normal(0.0, channel.noise_abs))
            value = clamp(value, channel.min_value, channel.max_value)
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

        Baseline-value channels yield their baseline with the active scenario's
        archiver events (step/ramp/spike) applied, then the channel's declared
        signal model: ``texture`` (slow baseline wander, riding post-event
        levels), relative ``noise`` (multiplicative sigma) and ``noise_abs``
        (additive sigma). All stochastic terms are deterministic keyed draws
        addressed by (channel, absolute timestamp), so repeated or overlapping
        queries agree pointwise on shared timestamps — with two caveats: noise
        counters are quantized to the nearest millisecond, so sub-millisecond
        timestamps share draws; and when timestamps are not convertible to
        epoch seconds, texture is skipped and noise falls back to sample-index
        counters (deterministic for a given window shape, but without
        cross-window agreement). Expression channels are evaluated pointwise
        over the synthesized series of their referenced channels, so derived
        channels show correlated history.

        Event positioning has three flavors: ``at`` places an event at a fixed
        fraction of whatever window is requested, while ``at_offset`` (with
        ``until_offset`` for ramps) anchors it in wall-clock time, in seconds
        relative to the apply-time anchor T0 (the ``anchor=`` line in the state
        file, falling back to its mtime; negative = past). ``at_time``
        (``"HH:MM:SS"`` in the facility timezone; step/spike only) recurs daily:
        the event fires at that time of
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
        # Resolve the facility timezone once per synthesis pass (cheap: config and
        # ZoneInfo are both cached singletons) and thread it down, so daily
        # ``at_time`` events are placed in facility-local time regardless of the
        # deploy host's ``$TZ``. Resolved here, not at engine construction, so it
        # reflects config loaded after connector init.
        tz = get_facility_timezone()
        cache: dict[str, np.ndarray | list[str]] = {}
        series = self._synthesize(pv, n, cache, t_abs, anchor, tz)
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

    def _active_state_file(self) -> Path | None:
        """The state file to read: canonical ``active_scenarios``, else legacy."""
        if self._state_path.exists():
            return self._state_path
        if self._legacy_state_path.exists():
            return self._legacy_state_path
        return None

    def _refresh_scenario(self) -> None:
        """Re-read and recompose the active-scenario set when the state file changes."""
        state_file = self._active_state_file()
        if state_file is None:
            signature: tuple[str | None, int | None] = (None, None)
        else:
            try:
                signature = (str(state_file), state_file.stat().st_mtime_ns)
            except FileNotFoundError:
                signature, state_file = (None, None), None
        if signature == self._state_signature:
            return
        self._state_signature = signature
        self._state_mtime_ns = signature[1]

        names: list[str] = []
        anchor_epoch: float | None = None
        if state_file is not None:
            try:
                text = state_file.read_text()
            except FileNotFoundError:
                text = ""
            raw_names, anchor_epoch = self._parse_state(text)
            for raw in raw_names:
                if raw in self._scenarios:
                    if raw not in names:
                        names.append(raw)
                else:
                    logger.warning(f"Unknown scenario {raw!r} in {state_file}; ignoring")

        resolved = resolve_active_scenarios(names)
        problems = self.validate_composition(resolved)
        if problems:
            logger.error(
                f"Active scenarios {resolved!r} do not compose ({'; '.join(problems)}); "
                f"falling back to ['{DEFAULT_SCENARIO}']"
            )
            resolved = [DEFAULT_SCENARIO]

        self._anchor_epoch = anchor_epoch
        new_active = tuple(resolved)
        if new_active != self._active:
            self._written.clear()
            logger.info(
                f"Simulation scenarios switched to {list(new_active)!r} (session writes cleared)"
            )
            self._active = new_active
        elif self._written:
            # State file touched with the same set: treat as an explicit
            # re-assert and hand back a fresh machine.
            self._written.clear()
            logger.info(
                f"Simulation scenarios {list(new_active)!r} re-asserted (session writes cleared)"
            )
        self._recompose()

    @staticmethod
    def _parse_state(text: str) -> tuple[list[str], float | None]:
        """Parse state-file text into (scenario names, anchor epoch seconds or None).

        Skips blank lines and ``#`` comments; any ``key=value`` line is metadata
        (only ``anchor=<ISO8601>`` is recognised), everything else is a name.
        """
        names: list[str] = []
        anchor_epoch: float | None = None
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" in stripped:
                key, _, value = stripped.partition("=")
                if key.strip() == "anchor":
                    try:
                        parsed = datetime.fromisoformat(value.strip())
                        # Anchors written by apply.py are UTC-aware; attach the
                        # facility zone to a naive (hand-edited) anchor so
                        # ``.timestamp()`` does not silently fall back to box-local.
                        if parsed.tzinfo is None:
                            parsed = parsed.replace(tzinfo=get_facility_timezone())
                        anchor_epoch = parsed.timestamp()
                    except ValueError:
                        logger.warning(f"Ignoring malformed anchor in state file: {stripped!r}")
                continue
            names.append(stripped)
        return names, anchor_epoch

    def _recompose(self) -> None:
        """Merge the active set's overrides and archiver scripts into composed views.

        Safe to merge by plain update because :meth:`validate_composition`
        guarantees the active scenarios touch disjoint channel sets.
        """
        overrides: dict[str, float | str] = {}
        archiver: dict[str, list[dict[str, Any]]] = {}
        for name in self._active:
            scenario = self._scenarios[name]
            overrides.update(scenario.overrides)
            archiver.update(scenario.archiver)
        self._composed_overrides = overrides
        self._composed_archiver = archiver

    def _effective(self, pv: str) -> float | str:
        """Effective value: session write > composed scenario override > baseline."""
        if pv in self._written:
            return self._written[pv]
        if pv in self._composed_overrides:
            return self._composed_overrides[pv]
        channel = self._channels[pv]
        if channel.expr is not None:
            return evaluate_channel(channel.expr, channel.expr_source, pv, self._numeric_effective)
        assert channel.value is not None  # guaranteed by parse_machine
        return channel.value

    def _numeric_effective(self, pv: str) -> float:
        """Numeric effective value of a referenced channel — texture-free.

        Live expression evaluation resolves references through the effective
        value only: texture, like noise, is a top-level measurement effect,
        applied solely to the channel actually being read. Synthesized derived
        series differ — they inherit referenced channels' texture and noise
        through the cached per-window series (:meth:`_synthesize`). This
        existing live/archive asymmetry for derived channels is documented and
        pinned (``TestExprRefTextureSemantics``), not silently widened.
        """
        value = self._effective(pv)
        if isinstance(value, str):
            raise ExpressionError(
                f"Channel {pv!r} holds a string value and cannot be used in an expression"
            )
        channel = self._channels[pv]
        return clamp(float(value), channel.min_value, channel.max_value)

    def _scenario_anchor(self) -> float:
        """Apply-time anchor T0 in epoch seconds.

        Prefers the explicit ``anchor=`` line from the state file (so telemetry
        and the seeded logbook share one clock), falling back to the state-file
        mtime, then to the current time when neither is available.
        """
        if self._anchor_epoch is not None:
            return self._anchor_epoch
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
        tz: ZoneInfo,
    ) -> "np.ndarray | list[str]":
        """Build one channel's series, memoized per synthesis pass.

        Expression channels evaluate pointwise over their references'
        *synthesized* series, so a derived series inherits referenced
        channels' texture and noise — unlike live derived reads, which
        resolve references texture-free via :meth:`_numeric_effective`.
        """
        cached = cache.get(pv)
        if cached is not None:
            return cached
        channel = self._channels[pv]
        events = self._composed_archiver.get(pv, [])

        if channel.expr is None and isinstance(channel.value, str):
            series_str = string_series(channel.value, events, n, t_abs, anchor, tz)
            cache[pv] = series_str
            return series_str

        if channel.expr is not None:
            ref_series = {
                ref: self._synthesize(ref, n, cache, t_abs, anchor, tz) for ref in channel.refs
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

        series = apply_events(series, events, n, t_abs, anchor, tz)
        series = _apply_signal_model(pv, channel, series, t_abs)
        if channel.min_value is not None or channel.max_value is not None:
            series = np.clip(series, channel.min_value, channel.max_value)
        cache[pv] = series
        return series


def engine_from_connector_config(config: dict[str, Any]) -> SimulationEngine | None:
    """Load a SimulationEngine from a connector config dict, if configured.

    Mirrors ``LimitsValidator.from_config()`` path resolution: a relative
    ``simulation_file`` path is anchored at the configured project root. The
    scenario state file is resolved from the same ambient config (see
    :func:`default_state_dir`), so a connector reads the state the ``sim`` CLI
    writes.

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
            from osprey_connectors.config import get_config_value

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
