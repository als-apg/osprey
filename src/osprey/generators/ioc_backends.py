"""IOC Backend implementations and runtime loader.

This module provides simulation backends for soft IOCs that can be loaded
at runtime from config.yml. This allows changing backends without regenerating
the IOC code.

Backends implement the SimulationBackend protocol:
- initialize(pv_definitions) -> initial values
- on_write(pv_name, value) -> cascading updates or None to delegate
- step(dt) -> time-evolved updates

Usage in generated IOC:
    from osprey.generators.ioc_backends import load_backends_from_config

    config = yaml.safe_load(Path("config.yml").read_text())
    backend = load_backends_from_config(config.get("simulation", {}), PAIRINGS)
"""

from __future__ import annotations

import importlib
import importlib.util
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


class PassthroughBackend:
    """No-op backend - PVs retain written values.

    This backend does not simulate any physics. PVs keep their written values
    and no time evolution occurs.

    Returns empty dict (not None) to indicate it handles all writes.
    """

    def initialize(self, pv_definitions: list[dict]) -> dict[str, Any]:
        """Initialize - no special setup needed."""
        return {}

    def on_write(self, pv_name: str, value: Any) -> dict[str, Any] | None:
        """Handle write - no updates to other PVs."""
        return {}

    def step(self, dt: float) -> dict[str, Any]:
        """Step - no time evolution."""
        return {}


class MockStyleBackend:
    """Archiver-style simulation with keyword-based behaviors.

    This backend replicates the mock archiver connector's value generation logic.
    PV behaviors are determined by keyword matching on PV names.

    Supported behaviors:
    - BPM/position: Random equilibrium offset + slow drift + noise
    - Beam current: 500 mA base with decay
    - Current: 150 mA base with linear trend
    - Voltage: 5000 V base, stable with small oscillation
    - Pressure: 1e-9 Torr base with gradual increase
    - Temperature: 25 C base with gradual increase
    - Lifetime: 10 hours base, decreasing trend
    - Default: 100.0 base with linear trend

    Args:
        noise_level: Noise level for SP->RB tracking (0.01 = 1%)
        pairings: Dict mapping setpoint PV names to readback PV names
    """

    def __init__(
        self,
        noise_level: float = 0.01,
        pairings: dict[str, str] | None = None,
    ):
        """Initialize backend.

        Args:
            noise_level: Default noise level for SP->RB tracking (0.01 = 1%)
            pairings: Dict mapping setpoint PV names to readback PV names
        """
        self.noise_level = noise_level
        self.pairings = pairings or {}
        self._state: dict[str, float] = {}
        self._rng_per_pv: dict[str, np.random.Generator] = {}

    def initialize(self, pv_definitions: list[dict]) -> dict[str, float]:
        """Initialize all PVs with type-appropriate values."""
        initial_values = {}

        for pv in pv_definitions:
            name = pv["name"]

            # Create reproducible RNG for this PV
            seed = hash(name) % (2**32)
            self._rng_per_pv[name] = np.random.default_rng(seed=seed)

            # Generate initial value based on PV name keywords
            value = self._get_initial_value(name)
            self._state[name] = value
            initial_values[name] = value

        return initial_values

    def _get_initial_value(self, pv_name: str) -> float:
        """Get initial value based on PV name keywords (hardcoded heuristics)."""
        name = pv_name.lower()
        rng = self._rng_per_pv.get(pv_name)

        # BPM positions: random offset around zero
        if "bpm" in name or "position" in name:
            return float(rng.uniform(-0.1, 0.1)) if rng else 0.0

        # Beam current (DCCT): 500 mA
        if "beam" in name and "current" in name:
            return 500.0
        if "dcct" in name:
            return 500.0

        # Other currents: 150 mA
        if "current" in name:
            return 150.0

        # Voltages: 5000 V
        if "voltage" in name:
            return 5000.0

        # Pressures: 1e-9 Torr (vacuum)
        if "pressure" in name:
            return 1e-9

        # Temperatures: 25 C
        if "temp" in name:
            return 25.0

        # Lifetime: 10 hours
        if "lifetime" in name:
            return 10.0

        # Default
        return 100.0

    def on_write(self, pv_name: str, value: float) -> dict[str, float] | None:
        """Handle setpoint write, update paired readback.

        Returns dict (handles write) not None (would delegate).
        """
        updates = {}
        self._state[pv_name] = value

        # Check explicit pairings
        if pv_name in self.pairings:
            rb_name = self.pairings[pv_name]

            # Apply noise
            if self.noise_level > 0 and value != 0:
                noisy = value + random.gauss(0, abs(value) * self.noise_level)
            else:
                noisy = value

            self._state[rb_name] = noisy
            updates[rb_name] = noisy

        return updates

    def step(self, dt: float) -> dict[str, float]:
        """Advance simulation by dt seconds (hardcoded behaviors)."""
        updates = {}

        for pv_name, current in list(self._state.items()):
            name = pv_name.lower()
            rng = self._rng_per_pv.get(pv_name)

            # BPM positions: slow drift + noise
            if "bpm" in name or "position" in name:
                if rng:
                    drift = float(rng.normal(0, 0.001 * dt))
                    noise = float(rng.normal(0, 0.001))
                    new_value = current + drift + noise
                    # Mean reversion toward zero
                    new_value -= 0.01 * current * dt
                    self._state[pv_name] = new_value
                    updates[pv_name] = new_value

            # Beam current: slow decay
            elif ("beam" in name and "current" in name) or "dcct" in name:
                # ~5% decay per hour
                decay = current * 0.00001 * dt
                new_value = current - decay
                if rng:
                    new_value += float(rng.normal(0, 0.5))
                self._state[pv_name] = new_value
                updates[pv_name] = new_value

            # Pressures: slight increase over time
            elif "pressure" in name:
                increase = current * 0.0001 * dt
                new_value = current + increase
                if rng:
                    new_value *= 1 + float(rng.normal(0, 0.01))
                self._state[pv_name] = new_value
                updates[pv_name] = new_value

            # Temperatures: gradual increase
            elif "temp" in name:
                increase = 0.001 * dt
                new_value = current + increase
                if rng:
                    new_value += float(rng.normal(0, 0.1))
                self._state[pv_name] = new_value
                updates[pv_name] = new_value

        return updates


class ChainedBackend:
    """Composes multiple backends in a chain.

    For on_write: iterate from last to first, first non-None result wins.
    For step: all backends run in order, merge results (last wins conflicts).
    For initialize: all backends run in order, merge results (last wins).
    """

    def __init__(self, backends: list):
        """Initialize chained backend.

        Args:
            backends: List of backend instances, in order [base, override1, override2, ...]
        """
        self.backends = backends

    def initialize(self, pv_definitions: list[dict]) -> dict[str, Any]:
        """Initialize all backends and merge results (last wins)."""
        values: dict[str, Any] = {}
        for backend in self.backends:
            backend_values = backend.initialize(pv_definitions)
            values.update(backend_values)
        return values

    def on_write(self, pv_name: str, value: Any) -> dict[str, Any] | None:
        """Handle write - iterate from last to first, first non-None wins."""
        # Later backends get first chance to handle
        for backend in reversed(self.backends):
            result = backend.on_write(pv_name, value)
            if result is not None:
                return result
        return {}

    def step(self, dt: float) -> dict[str, Any]:
        """Step all backends and merge results (last wins)."""
        updates: dict[str, Any] = {}
        for backend in self.backends:
            backend_updates = backend.step(dt)
            updates.update(backend_updates)
        return updates


def load_backends_from_config(
    sim_config: dict,
    pairings: dict[str, str] | None = None,
    config_dir: Path | str | None = None,
) -> PassthroughBackend | MockStyleBackend | ChainedBackend:
    """Load and chain backends from simulation config section.

    Args:
        sim_config: The 'simulation' section from config.yml, containing:
            - base: Base backend config dict
            - overlays: List of overlay backend config dicts
        pairings: SP->RB pairings dict (passed to MockStyleBackend)
        config_dir: Directory containing config.yml (for resolving file_path)

    Returns:
        Backend instance (possibly ChainedBackend if multiple backends)

    Example config.yml simulation section:
        simulation:
          base:
            type: mock_style
            noise_level: 0.01
          overlays:
            - file_path: my_backends/physics.py
              class_name: PhysicsBackend
              params:
                tau: 2.0
    """
    base = sim_config.get("base", {"type": "mock_style"})
    overlays = sim_config.get("overlays", [])

    if config_dir is not None:
        config_dir = Path(config_dir)

    backends = []
    for config in [base] + overlays:
        backend = _instantiate_backend(config, pairings, config_dir)
        backends.append(backend)

    if len(backends) == 1:
        return backends[0]
    return ChainedBackend(backends)


def _instantiate_backend(
    config: dict,
    pairings: dict[str, str] | None,
    config_dir: Path | None = None,
) -> PassthroughBackend | MockStyleBackend | Any:
    """Instantiate a single backend from config.

    Args:
        config: Backend configuration dict with:
            - type: "mock_style", "passthrough", or "custom"
            - For custom via file: file_path, class_name, params
            - For custom via module: module_path, class_name, params
            - For mock_style: noise_level
        pairings: SP->RB pairings (used by MockStyleBackend)
        config_dir: Directory containing config.yml (for resolving file_path)

    Returns:
        Backend instance

    Raises:
        ValueError: If backend config is invalid
        ImportError: If custom backend module cannot be imported
        AttributeError: If custom backend class not found in module
    """
    backend_type = config.get("type")

    if backend_type == "mock_style":
        return MockStyleBackend(
            noise_level=config.get("noise_level", 0.01),
            pairings=pairings,
        )
    elif backend_type == "passthrough":
        return PassthroughBackend()
    elif config.get("file_path") and config.get("class_name"):
        # Load from file path (relative to config.yml)
        file_path = Path(config["file_path"])
        if config_dir and not file_path.is_absolute():
            file_path = config_dir / file_path
        file_path = file_path.resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"Backend file not found: {file_path}")

        # Load module from file
        module_name = file_path.stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec from: {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        cls = getattr(module, config["class_name"])
        params = config.get("params", {})
        return cls(**params)
    elif config.get("module_path") and config.get("class_name"):
        # Dynamic import for custom backends (legacy module_path style)
        module = importlib.import_module(config["module_path"])
        cls = getattr(module, config["class_name"])
        params = config.get("params", {})
        return cls(**params)
    else:
        raise ValueError(f"Unknown backend config: {config}")
