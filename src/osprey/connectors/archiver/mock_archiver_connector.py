"""
Mock archiver connector for development and testing.

Generates synthetic time-series data for any PV names.
Ideal for R&D and development without archiver access.

"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from osprey.connectors.archiver.base import ArchiverConnector, ArchiverMetadata
from osprey.connectors.pv_taxonomy import classify_pv
from osprey.simulation import engine_serves
from osprey.utils.config import get_facility_timezone
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.simulation import SimulationEngine

logger = get_logger("mock_archiver_connector")

ARCHIVER_KEY = "archiver.mock_archiver.simulation_file"


def _anchor(path: Path, project_root: Path | None) -> Path:
    """Anchor a relative path at the project root, mirroring the engine loader."""
    if path.is_absolute() or project_root is None:
        return path
    return project_root / path


def _control_system_simulation_file() -> tuple[Path | None, Path | None]:
    """Resolve the control-system-side simulation file for the active connector.

    Returns ``(machine_path, project_root)``; both are None when no project
    config is reachable (a bare connector constructed in tests, for instance).
    """
    from osprey.simulation.apply import resolve_simulation_file
    from osprey.utils.config import get_config_value, load_config

    try:
        config = load_config()
        project_root = get_config_value("project_root", None)
    except (FileNotFoundError, IsADirectoryError, KeyError, RuntimeError, ValueError) as e:
        logger.debug(f"No project config available to derive a simulation file: {e}")
        return None, None

    root = Path(project_root) if project_root else None
    machine_path, _active_type, _type_key, _mock_key = resolve_simulation_file(
        config, root or Path()
    )
    return machine_path, root


def _with_derived_simulation_file(config: dict[str, Any]) -> dict[str, Any]:
    """Fill in ``simulation_file`` from the control-system side when unset.

    The mock archiver has no machine model of its own — it synthesizes history
    from the same file the live connector serves, so a project that declares the
    path under both sections is one edit away from archived history that
    contradicts live reads. When the archiver block omits ``simulation_file`` it
    is derived from ``control_system.connector.<type>.simulation_file`` through
    the resolver the ``sim`` CLI already shares. An explicit archiver-side value
    still wins; when the two disagree, the divergence is reported.

    Returns the connector config, with ``simulation_file`` filled in when it was
    derived. The input dict is never mutated.
    """
    own = config.get("simulation_file")
    derived, project_root = _control_system_simulation_file()

    if derived is None:
        return config

    if not own:
        logger.debug(f"Derived {ARCHIVER_KEY} from the control-system config: {derived}")
        return {**config, "simulation_file": str(derived)}

    own_path = _anchor(Path(str(own)).expanduser(), project_root)
    if own_path != derived:
        logger.warning(
            f"{ARCHIVER_KEY} ({own_path}) differs from the control-system simulation "
            f"file ({derived}); the archiver value wins, so archived history will not "
            f"match live reads. Remove {ARCHIVER_KEY} to derive it."
        )
    return config


class MockArchiverConnector(ArchiverConnector):
    """
    Mock archiver for development - generates synthetic time-series data.

    This connector simulates an archiver system without requiring real
    archiver access. It generates realistic time-series data for any PV name.

    Features:
    - Accepts any PV names
    - Generates realistic time series with trends and noise
    - Configurable sampling rate and noise level
    - Returns pandas DataFrames matching real archiver format

    Example:
        >>> config = {
        >>>     'sample_rate_hz': 1.0,
        >>>     'noise_level': 0.01
        >>> }
        >>> connector = MockArchiverConnector()
        >>> await connector.connect(config)
        >>> df = await connector.get_data(
        >>>     pv_list=['BEAM:CURRENT'],
        >>>     start_date=datetime(2024, 1, 1),
        >>>     end_date=datetime(2024, 1, 2)
        >>> )
    """

    def __init__(self):
        self._connected = False
        self._sim_engine: SimulationEngine | None = None

    async def connect(self, config: dict[str, Any]) -> None:
        """
        Initialize mock archiver.

        Args:
            config: Configuration with keys:
                - sample_rate_hz: Sampling rate (default: 1.0)
                - noise_level: Relative noise level (default: 0.1)
                - simulation_file: Optional path to a machine.json driving the
                  data-driven simulation engine (relative paths resolve against
                  the project root). Engine-known channels are synthesized from
                  the machine model; without it, every channel uses the generic
                  PV-type procedural synthesizer. Unset, it is derived from the
                  control system's own ``simulation_file`` so live reads and
                  archived history come from one machine model.
        """
        self._sample_rate_hz = config.get("sample_rate_hz", 1.0)
        self._noise_level = config.get("noise_level", 0.1)

        # Optional data-driven simulation engine (machine file), derived from
        # the control-system config when this section does not name one.
        from osprey.simulation import engine_from_connector_config

        self._sim_engine = engine_from_connector_config(_with_derived_simulation_file(config))

        self._connected = True
        logger.debug("Mock archiver connector initialized")

    async def disconnect(self) -> None:
        """Cleanup mock archiver."""
        self._sim_engine = None
        self._connected = False
        logger.debug("Mock archiver connector disconnected")

    async def get_data(
        self,
        pv_list: list[str],
        start_date: datetime,
        end_date: datetime,
        precision_ms: int = 1000,
        timeout: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic historical data.

        Args:
            pv_list: List of PV names (all accepted)
            start_date: Start of time range
            end_date: End of time range
            precision_ms: Time precision (affects downsampling)
            timeout: Ignored for mock archiver

        Returns:
            DataFrame with datetime index and columns for each PV
        """
        duration = (end_date - start_date).total_seconds()

        # Limit number of points for performance
        # Use precision_ms to determine sampling
        num_points = min(int(duration / (precision_ms / 1000.0)), 10000)
        num_points = max(num_points, 10)  # At least 10 points

        # Generate timestamps
        time_step = duration / (num_points - 1) if num_points > 1 else 0
        timestamps = [start_date + timedelta(seconds=i * time_step) for i in range(num_points)]

        # Generate data for each PV. Channels known to the simulation engine
        # are synthesized from the machine model; everything else uses the
        # generic procedural generation.
        data = {}
        for pv in pv_list:
            if engine_serves(self._sim_engine, pv):
                data[pv] = self._sim_engine.synthesize_series(pv, timestamps)
            else:
                data[pv] = self._generate_time_series(pv, num_points)

        df = pd.DataFrame(data, index=pd.to_datetime(timestamps))

        logger.debug(
            f"Mock archiver generated {len(df)} points for "
            f"{len(pv_list)} PVs from {start_date} to {end_date}"
        )

        return df

    async def get_metadata(self, pv_name: str) -> ArchiverMetadata:
        """Get mock archiver metadata."""
        # Mock returns fake metadata indicating "infinite" retention
        return ArchiverMetadata(
            pv_name=pv_name,
            is_archived=True,
            # Both bounds tz-aware (facility zone) so a consumer can subtract or
            # compare them without a naive/aware TypeError.
            archival_start=datetime(2000, 1, 1, tzinfo=get_facility_timezone()),
            archival_end=datetime.now(get_facility_timezone()),
            sampling_period=1.0 / self._sample_rate_hz,
            description=f"Mock archived PV: {pv_name}",
        )

    async def check_availability(self, pv_names: list[str]) -> dict[str, bool]:
        """All PVs are available in mock archiver."""
        return dict.fromkeys(pv_names, True)

    def _generate_time_series(
        self,
        pv_name: str,
        num_points: int,
    ) -> np.ndarray:
        """
        Generate generic synthetic time series with trends and noise.

        Used for PVs that are not known to the data-driven simulation engine
        (e.g. projects without a machine file). Creates realistic-looking data
        with:
        - Sinusoidal variations
        - Linear trends
        - Random noise
        - PV-type-specific characteristics (current/voltage/power/pressure/
          temperature/lifetime/default)
        - BPMs use random offsets with slow oscillations

        Scenario-specific physics (vacuum bursts, RF thermal excursions, etc.)
        is supplied by the simulation engine, not this generic generator.
        """
        t = np.linspace(0, 1, num_points)
        pv_lower = pv_name.lower()
        rng = np.random.default_rng(seed=hash(pv_name) % (2**32))

        # BPM channels — reproducible random offsets with slow oscillations
        if "position" in pv_lower or "pos" in pv_lower or "bpm" in pv_lower:
            base = 0.0
            offset_range = 0.1  # ±100 µm equilibrium position
            perturbation_amp = 0.01  # ±10 µm oscillation
            trend = np.ones(num_points) * base

            offset = rng.uniform(-offset_range, offset_range)
            phase = rng.uniform(0, 2 * np.pi)
            frequency = rng.uniform(0.01, 0.5)

            wave = perturbation_amp * np.sin(2 * np.pi * t * frequency + phase)

            noise_amplitude = perturbation_amp * self._noise_level
            noise = rng.normal(0, noise_amplitude, num_points)

            return trend + offset + wave + noise

        # Shape a trend + wave per PV kind. Classification (name -> kind/base)
        # is shared with the control-system mock via classify_pv; the synthesis
        # shapes below are archiver-specific.
        kind = classify_pv(pv_name)
        base = kind.base_value
        if kind.name == "beam_current":
            trend = np.ones(num_points) * base
            for i in range(num_points):
                decay_phase = i % (num_points // 10)
                trend[i] = base * (1 - 0.05 * (decay_phase / (num_points // 10)))
            wave = 5 * np.sin(2 * np.pi * t * 5)
        elif kind.name == "current":
            trend = base + 10 * t
            wave = 10 * np.sin(2 * np.pi * t * 3)
        elif kind.name == "voltage":
            trend = np.ones(num_points) * base
            wave = 50 * np.sin(2 * np.pi * t * 2)
        elif kind.name == "power":
            trend = base + 5 * t
            wave = 5 * np.sin(2 * np.pi * t * 4)
        elif kind.name == "pressure":
            trend = base * (1 + 0.1 * t)
            wave = base * 0.05 * np.sin(2 * np.pi * t * 10)
        elif kind.name == "temperature":
            trend = base + 2 * t
            wave = 0.5 * np.sin(2 * np.pi * t * 8)
        elif kind.name == "lifetime":
            trend = base - 2 * t
            wave = 1 * np.sin(2 * np.pi * t * 3)
        else:
            trend = base + 20 * t
            wave = 10 * np.sin(2 * np.pi * t * 2)

        noise_amplitude = abs(base) * self._noise_level
        noise = np.random.normal(0, noise_amplitude, num_points)

        return trend + wave + noise
