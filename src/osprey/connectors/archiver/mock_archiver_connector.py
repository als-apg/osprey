"""
Mock archiver connector for development and testing.

Generates synthetic time-series data for any PV names.
Ideal for R&D and development without archiver access.

"""

import zlib
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from osprey.connectors.archiver._timerange import (
    aggregate_series,
    long_frame,
    resolve_processing,
    to_utc,
)
from osprey.connectors.archiver.base import ArchiverConnector, ArchiverMetadata
from osprey.connectors.pv_taxonomy import classify_pv
from osprey.simulation import engine_serves
from osprey.utils.config import get_facility_timezone
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.simulation import SimulationEngine

logger = get_logger("mock_archiver_connector")


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
                  PV-type procedural synthesizer.
        """
        # A zero or negative rate has no meaning and would divide by zero in
        # get_data's full-resolution path and in get_metadata's sampling_period,
        # so reject it at configuration time rather than at the first query.
        sample_rate_hz = config.get("sample_rate_hz", 1.0)
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be > 0 (got {sample_rate_hz})")
        self._sample_rate_hz = sample_rate_hz
        self._noise_level = config.get("noise_level", 0.1)

        # Optional data-driven simulation engine (machine file)
        from osprey.simulation import engine_from_connector_config

        self._sim_engine = engine_from_connector_config(config)

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
                configured ``sample_rate_hz`` instead of a density derived from
                ``precision_ms``, since the mock has no backing store of real
                samples to return in full. Either way the generator is capped at
                10,000 points, so a window longer than
                ``10000 / sample_rate_hz`` seconds comes back sparser than the
                configured rate.
            timeout: Ignored for mock archiver
            processing: Aggregation applied within each precision_ms bin. One of
                "raw", "mean", "min", "max", "median", "std", "count". Backends
                that aggregate server-side (EPICS) push it down; the rest apply
                it client-side. Anything else raises ValueError.

        Returns:
            The canonical long frame — see :meth:`ArchiverConnector.get_data`
            for the full contract (columns, dtypes, ordering, and the rule that
            nothing is ever manufactured).

        Raises:
            ValueError: If ``processing`` other than ``"raw"`` is requested for
                a channel that synthesizes non-numeric values — aggregation
                over non-numeric samples is undefined (see
                :func:`~osprey.connectors.archiver._timerange.aggregate_series`).
        """
        # long_frame requires a UTC-aware index on every per-channel series; a
        # naive start/end (facility wall-clock, per the framework's convention)
        # is normalized the same way every other archiver connector normalizes
        # its query window, before any timestamps are generated from it.
        start_date = to_utc(start_date)
        end_date = to_utc(end_date)
        duration = (end_date - start_date).total_seconds()

        # Limit number of points for performance.
        # Use precision_ms to determine sampling density. precision_ms <= 0
        # means full resolution (the same convention resolve_processing and
        # decimate_raw use), but unlike a real archiver the mock has no
        # backing store of "real" samples to return in full — it synthesizes
        # them. "Full resolution" here is therefore defined as generating at
        # the connector's own configured native rate (sample_rate_hz) rather
        # than deriving a density from a bin width that doesn't exist; this
        # also keeps a precision_ms=0 request from dividing by zero.
        effective_precision_ms = precision_ms if precision_ms > 0 else 1000.0 / self._sample_rate_hz
        num_points = min(int(duration / (effective_precision_ms / 1000.0)), 10000)
        num_points = max(num_points, 10)  # At least 10 points

        # Generate timestamps
        time_step = duration / (num_points - 1) if num_points > 1 else 0
        timestamps = [start_date + timedelta(seconds=i * time_step) for i in range(num_points)]
        index = pd.DatetimeIndex(timestamps)

        # Generate data for each PV. Channels known to the simulation engine
        # are synthesized from the machine model; everything else uses the
        # generic procedural generation. Each channel is then aggregated on
        # its own — aggregate_series drops any bin with no samples rather
        # than emitting one, so no upsampling floor is needed here regardless
        # of how much wider the real sample spacing is than precision_ms.
        resolved = resolve_processing(processing, precision_ms)
        series = {}
        for pv in pv_list:
            if engine_serves(self._sim_engine, pv):
                values = self._sim_engine.synthesize_series(pv, timestamps)
            else:
                values = self._generate_time_series(pv, num_points)
            # No dtype is forced: a numeric series (list[float] or a float
            # ndarray) infers float64 as always; an enum/status channel's
            # list[str] is passed through untouched, per the long_frame
            # contract that leaves `value` dtype-unconstrained.
            raw_series = pd.Series(values, index=index, name=pv)
            series[pv] = aggregate_series(raw_series, precision_ms, resolved)

        data = long_frame(series)

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
        # crc32, not hash(): Python salts str hashing per process (PYTHONHASHSEED),
        # so a hash()-derived seed made this generator reproducible only *within*
        # one run. Two processes asked for the same PV over the same window got
        # different data, which defeats the point of seeding at all.
        rng = np.random.default_rng(seed=zlib.crc32(pv_name.encode()))

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
            # Ten sawtooth refill cycles across the window: current decays 5%
            # over each cycle, then jumps back to base.
            period = num_points // 10
            trend = base * (1 - 0.05 * (np.arange(num_points) % period) / period)
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
        # rng, not the global np.random: the BPM branch above already uses the
        # per-PV generator, and drawing this noise from the unseeded global made
        # every non-BPM channel non-reproducible even inside a single process.
        noise = rng.normal(0, noise_amplitude, num_points)

        return trend + wave + noise
