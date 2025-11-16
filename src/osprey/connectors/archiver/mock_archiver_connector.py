"""
Mock archiver connector for development and testing.

Generates synthetic time-series data for any PV names.
Ideal for R&D and development without archiver access.

Related to Issue #18 - Control System Abstraction (Layer 2 - Mock Implementation)
"""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from osprey.connectors.archiver.base import ArchiverConnector, ArchiverMetadata
from osprey.utils.logger import get_logger

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

    async def connect(self, config: dict[str, Any]) -> None:
        """
        Initialize mock archiver.

        Args:
            config: Configuration with keys:
                - sample_rate_hz: Sampling rate (default: 1.0)
                - noise_level: Relative noise level (default: 0.01)
        """
        self._sample_rate_hz = config.get('sample_rate_hz', 1.0)
        self._noise_level = config.get('noise_level', 0.01)
        self._connected = True
        logger.debug("Mock archiver connector initialized")

    async def disconnect(self) -> None:
        """Cleanup mock archiver."""
        self._connected = False
        logger.debug("Mock archiver connector disconnected")

    async def get_data(
        self,
        pv_list: list[str],
        start_date: datetime,
        end_date: datetime,
        precision_ms: int = 1000,
        timeout: int | None = None
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
        timestamps = [
            start_date + timedelta(seconds=i * time_step)
            for i in range(num_points)
        ]

        # Generate data for each PV
        data = {}
        for pv in pv_list:
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
            archival_start=datetime(2000, 1, 1),  # Arbitrary old date
            archival_end=datetime.now(),
            sampling_period=1.0 / self._sample_rate_hz,
            description=f"Mock archived PV: {pv_name}"
        )

    async def check_availability(self, pv_names: list[str]) -> dict[str, bool]:
        """All PVs are available in mock archiver."""
        return dict.fromkeys(pv_names, True)

    def _generate_time_series(self, pv_name: str, num_points: int) -> np.ndarray:
        """
        Generate synthetic time series with trends and noise.

        Creates realistic-looking data with:
        - Sinusoidal variations
        - Linear trends
        - Random noise
        - PV-type-specific characteristics
        """
        t = np.linspace(0, 1, num_points)
        pv_lower = pv_name.lower()

        # Base value depends on PV type
        if ('beam' in pv_lower and 'current' in pv_lower) or 'dcct' in pv_lower:
            # Beam current: slow decay with refills
            base = 500.0
            # Simulate refills 10 times over the time range
            trend = np.ones(num_points) * base
            for i in range(num_points):
                # Decay between refills (5% loss per cycle)
                decay_phase = i % (num_points // 10)
                trend[i] = base * (1 - 0.05 * (decay_phase / (num_points // 10)))
            wave = 5 * np.sin(2 * np.pi * t * 5)
        elif 'current' in pv_lower:
            base = 150.0
            trend = base + 10 * t
            wave = 10 * np.sin(2 * np.pi * t * 3)
        elif 'voltage' in pv_lower:
            base = 5000.0
            trend = np.ones(num_points) * base
            wave = 50 * np.sin(2 * np.pi * t * 2)
        elif 'power' in pv_lower:
            base = 50.0
            trend = base + 5 * t
            wave = 5 * np.sin(2 * np.pi * t * 4)
        elif 'pressure' in pv_lower:
            base = 1e-9
            trend = base * (1 + 0.1 * t)
            wave = base * 0.05 * np.sin(2 * np.pi * t * 10)
        elif 'temp' in pv_lower:
            base = 25.0
            trend = base + 2 * t
            wave = 0.5 * np.sin(2 * np.pi * t * 8)
        elif 'lifetime' in pv_lower:
            base = 10.0
            trend = base - 2 * t  # Lifetime decreases with current
            wave = 1 * np.sin(2 * np.pi * t * 3)
        elif 'position' in pv_lower or 'pos' in pv_lower:
            base = 0.0
            trend = 0.1 * np.sin(2 * np.pi * t * 0.5)  # Slow drift
            wave = 0.01 * np.sin(2 * np.pi * t * 20)  # Fast oscillations
        else:
            base = 100.0
            trend = base + 20 * t
            wave = 10 * np.sin(2 * np.pi * t * 2)

        # Add random noise
        noise_amplitude = abs(base) * self._noise_level
        noise = np.random.normal(0, noise_amplitude, num_points)

        return trend + wave + noise

