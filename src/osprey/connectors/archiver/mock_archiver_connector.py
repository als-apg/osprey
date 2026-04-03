"""
Mock archiver connector for development and testing.

Generates synthetic time-series data for any PV names.
Ideal for R&D and development without archiver access.

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
                - noise_level: Relative noise level (default: 0.1)
        """
        self._sample_rate_hz = config.get("sample_rate_hz", 1.0)
        self._noise_level = config.get("noise_level", 0.1)
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
            description=f"Mock archived PV: {pv_name}",
        )

    async def check_availability(self, pv_names: list[str]) -> dict[str, bool]:
        """All PVs are available in mock archiver."""
        return dict.fromkeys(pv_names, True)

    def _is_rf_channel(self, pv_lower: str) -> bool:
        """Check if PV belongs to the RF system (cavity or klystron)."""
        return "rf" in pv_lower and ("cavity" in pv_lower or "klystron" in pv_lower)

    def _rf_event_envelope(
        self, t: np.ndarray, events: list[tuple[float, float]], width: float = 0.024
    ) -> np.ndarray:
        """
        Generate a Gaussian event envelope from a list of (center, intensity) pairs.

        Width of 0.024 ≈ 4 hours in a 7-day window.
        """
        envelope = np.zeros_like(t)
        for center, intensity in events:
            envelope += intensity * np.exp(-((t - center) ** 2) / (2 * width**2))
        return envelope

    def _generate_rf_time_series(
        self, pv_lower: str, t: np.ndarray, num_points: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Generate RF system data with correlated thermal excursion / trip patterns.

        C1/K1 is the "problematic" cavity — three thermal excursion events where
        body temperature rises, reflected power spikes, and forward power trips.
        C2/K2 is the "stable" reference — one minor event for contrast.

        Event positions are hardcoded so temperature, power, and voltage channels
        all show correlated behavior at the same time points.
        """
        is_primary = "c1" in pv_lower or "k1" in pv_lower

        if is_primary:
            events = [(0.20, 1.0), (0.55, 0.7), (0.85, 1.2)]
        else:
            events = [(0.55, 0.25)]

        envelope = self._rf_event_envelope(t, events)

        if "temperature" in pv_lower or "temp" in pv_lower:
            base = 27.0 if is_primary else 26.5
            daily = 1.0 * np.sin(2 * np.pi * t * 7)
            excursion = envelope * 7.0
            noise = rng.normal(0, 0.2, num_points)
            return base + daily + excursion + noise

        if "power" in pv_lower:
            if "fwd" in pv_lower or "forward" in pv_lower:
                base = 450.0
                dip = -envelope * 440.0
                noise = rng.normal(0, 5, num_points)
                return np.maximum(base + dip + noise, 0)

            if "rev" in pv_lower or "reflect" in pv_lower:
                base = 5.0
                spike = envelope * 80.0
                noise = rng.normal(0, 1, num_points)
                return np.maximum(base + spike + noise, 0)

            if "net" in pv_lower:
                base = 445.0
                dip = -envelope * 440.0
                spike = envelope * 80.0
                noise = rng.normal(0, 5, num_points)
                return np.maximum(base + dip - spike + noise, 0)

            # Klystron output power or generic RF power
            base = 400.0
            dip = -envelope * 390.0
            noise = rng.normal(0, 4, num_points)
            return np.maximum(base + dip + noise, 0)

        if "voltage" in pv_lower:
            if "klystron" in pv_lower:
                base = 80.0  # kV — stable
                noise = rng.normal(0, 0.5, num_points)
                return base + noise
            # Cavity voltage drops during trips
            base = 2.5  # MV
            dip = -envelope * 2.4
            noise = rng.normal(0, 0.02, num_points)
            return np.maximum(base + dip + noise, 0)

        if "frequency" in pv_lower or "freq" in pv_lower:
            base = 499.654  # MHz
            thermal_shift = -envelope * 0.001  # detuning from thermal expansion
            noise = rng.normal(0, 0.0001, num_points)
            return base + thermal_shift + noise

        if "tuner" in pv_lower:
            base = 5.0  # cm
            compensation = envelope * 0.05
            noise = rng.normal(0, 0.01, num_points)
            return base + compensation + noise

        # Fallback for status/interlock/other RF channels
        base = 1.0
        noise = rng.normal(0, 0.01, num_points)
        return base + noise

    def _generate_time_series(self, pv_name: str, num_points: int) -> np.ndarray:
        """
        Generate synthetic time series with trends and noise.

        Creates realistic-looking data with:
        - Sinusoidal variations
        - Linear trends
        - Random noise
        - PV-type-specific characteristics
        - BPMs use random offsets with slow oscillations
        - RF system channels use correlated event patterns
        """
        t = np.linspace(0, 1, num_points)
        pv_lower = pv_name.lower()
        rng = np.random.default_rng(seed=hash(pv_name) % (2**32))

        # RF system channels — correlated thermal excursion / trip patterns
        if self._is_rf_channel(pv_lower):
            return self._generate_rf_time_series(pv_lower, t, num_points, rng)

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

        # Original behavior for all other PV types
        if ("beam" in pv_lower and "current" in pv_lower) or "dcct" in pv_lower:
            base = 500.0
            trend = np.ones(num_points) * base
            for i in range(num_points):
                decay_phase = i % (num_points // 10)
                trend[i] = base * (1 - 0.05 * (decay_phase / (num_points // 10)))
            wave = 5 * np.sin(2 * np.pi * t * 5)
        elif "current" in pv_lower:
            base = 150.0
            trend = base + 10 * t
            wave = 10 * np.sin(2 * np.pi * t * 3)
        elif "voltage" in pv_lower:
            base = 5000.0
            trend = np.ones(num_points) * base
            wave = 50 * np.sin(2 * np.pi * t * 2)
        elif "power" in pv_lower:
            base = 50.0
            trend = base + 5 * t
            wave = 5 * np.sin(2 * np.pi * t * 4)
        elif "pressure" in pv_lower:
            base = 1e-9
            trend = base * (1 + 0.1 * t)
            wave = base * 0.05 * np.sin(2 * np.pi * t * 10)
        elif "temp" in pv_lower:
            base = 25.0
            trend = base + 2 * t
            wave = 0.5 * np.sin(2 * np.pi * t * 8)
        elif "lifetime" in pv_lower:
            base = 10.0
            trend = base - 2 * t
            wave = 1 * np.sin(2 * np.pi * t * 3)
        else:
            base = 100.0
            trend = base + 20 * t
            wave = 10 * np.sin(2 * np.pi * t * 2)

        noise_amplitude = abs(base) * self._noise_level
        noise = np.random.normal(0, noise_amplitude, num_points)

        return trend + wave + noise
