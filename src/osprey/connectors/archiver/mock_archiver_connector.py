"""
Mock archiver connector for development and testing.

Generates synthetic time-series data for any PV names.
Ideal for R&D and development without archiver access.

"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from osprey.connectors.archiver._timerange import (
    aggregate_long_frame,
    resolve_processing,
    utc_window,
)
from osprey.connectors.archiver.base import ArchiverConnector, ArchiverMetadata
from osprey.simulation import engine_serves
from osprey.simulation.procedural import generate_series
from osprey.simulation.series import epoch_seconds_array
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
    - Generates realistic time series with texture and noise
    - Values are a pure function of (channel, absolute timestamp), so two
      overlapping windows agree on every timestamp they share
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
                  the machine model; without it, every channel goes through the
                  shared procedural generator
                  (:mod:`osprey.simulation.procedural`), whose baselines are the
                  ones the Virtual Accelerator serves. Unset, it is derived from
                  the control system's own ``simulation_file`` so live reads and
                  archived history come from one machine model.
        """
        # A zero or negative rate would divide by zero later; reject it at
        # configuration time.
        sample_rate_hz = config.get("sample_rate_hz", 1.0)
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be > 0 (got {sample_rate_hz})")
        self._sample_rate_hz = sample_rate_hz
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
        processing: str = "raw",
    ) -> pd.DataFrame:
        """
        Generate synthetic historical data.

        Args:
            pv_list: List of PV names (all accepted)
            start_date: Start of time range
            end_date: End of time range
            precision_ms: Time precision (affects downsampling). ``<= 0`` means
                full resolution: samples are generated at the connector's own
                configured ``sample_rate_hz``. Either way the generator is
                capped at 10,000 points.
            timeout: Ignored for mock archiver
            processing: Aggregation applied within each precision_ms bin. One of
                "raw", "mean", "min", "max", "median", "std", "count". Applied
                client-side via pandas resampling. Anything else raises ValueError.

        Returns:
            The canonical long frame — see :meth:`ArchiverConnector.get_data`.

        Raises:
            ValueError: If ``processing`` other than ``"raw"`` is requested for
                a channel that synthesizes non-numeric values.
        """
        # long_frame requires a UTC-aware index; a naive start/end means
        # facility wall-clock, as in every other archiver connector.
        start_date, end_date = utc_window(start_date, end_date)
        duration = (end_date - start_date).total_seconds()

        # Limit number of points for performance. precision_ms <= 0 means full
        # resolution, which for the mock (no backing store) means generating at
        # the configured native rate; this also avoids dividing by zero.
        effective_precision_ms = precision_ms if precision_ms > 0 else 1000.0 / self._sample_rate_hz
        num_points = min(int(duration / (effective_precision_ms / 1000.0)), 10000)
        num_points = max(num_points, 10)  # At least 10 points

        # Generate timestamps
        index = pd.date_range(start=start_date, end=end_date, periods=num_points)

        # Generate data for each PV. Channels known to the simulation engine
        # are synthesized from the machine model; everything else goes through
        # the shared procedural generator, evaluated at this grid's absolute
        # timestamps — the same values a store seeded from it holds.
        t_abs = epoch_seconds_array(index)
        if t_abs is None:  # pragma: no cover - the index above is always datetimes
            # Refusing beats the alternative the engine can afford: it falls
            # back to sample-index counters, which is deterministic per window
            # but not per timestamp — and history that disagrees with a store
            # seeded from the same generator is the failure this path replaced.
            raise ValueError(f"Cannot derive epoch seconds for the {start_date} to {end_date} grid")

        resolved = resolve_processing(processing, precision_ms)
        series = {}
        for pv in pv_list:
            if engine_serves(self._sim_engine, pv):
                values = self._sim_engine.synthesize_series(pv, index)
            else:
                values = generate_series(pv, t_abs, noise_level=self._noise_level)
            series[pv] = pd.Series(values, index=index, name=pv)

        data = aggregate_long_frame(series, resolved)

        logger.debug(
            f"Mock archiver generated {len(data)} rows across "
            f"{len(pv_list)} PVs from {start_date} to {end_date}"
        )

        return data

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
