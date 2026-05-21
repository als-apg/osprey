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

# Wall-clock anchor for the "Sector-7 vacuum burst + beam-loss" scenario.
# The mock archiver emits a correlated vacuum spike + DCCT step-down at this
# time of day for every date that falls inside a requested window.
VACUUM_EVENT_HOUR = 14
VACUUM_EVENT_MINUTE = 32
VACUUM_EVENT_SECOND = 8
VACUUM_EVENT_SECTOR = 7  # SR07 is the "problematic" sector


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

        # Vacuum-event positions are window-aware: each 14:32:08 of every
        # date in [start_date, end_date] becomes a normalized t ∈ [0, 1]
        # event position shared by Sector-7 gauge and DCCT channels.
        vacuum_event_positions = self._vacuum_event_positions(start_date, end_date)

        # Generate data for each PV
        data = {}
        for pv in pv_list:
            data[pv] = self._generate_time_series(pv, num_points, vacuum_event_positions)

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

    def _is_cold_cathode_gauge(self, pv_lower: str) -> bool:
        """Check if PV is a cold-cathode vacuum pressure gauge."""
        return "gauge" in pv_lower and "pressure" in pv_lower

    def _is_sector7_gauge(self, pv_lower: str) -> bool:
        """Check if PV is the SR07 (problematic) cold-cathode gauge."""
        sector_tag = f"sr{VACUUM_EVENT_SECTOR:02d}"
        return self._is_cold_cathode_gauge(pv_lower) and sector_tag in pv_lower

    def _is_dcct_channel(self, pv_lower: str) -> bool:
        """Check if PV is a DC current transformer / beam-current monitor."""
        return "dcct" in pv_lower or ("beam" in pv_lower and "current" in pv_lower)

    def _vacuum_event_positions(self, start_date: datetime, end_date: datetime) -> list[float]:
        """
        Find every wall-clock occurrence of 14:32:08 within [start, end] and
        return their fractional positions in [0, 1] across the window.

        Returns an empty list when the requested window does not span any
        scheduled event time — in which case downstream channels fall back to
        their normal baseline behavior.
        """
        duration = (end_date - start_date).total_seconds()
        if duration <= 0:
            return []

        positions: list[float] = []
        cursor = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        # Walk every calendar date that the window touches.
        while cursor <= end_date:
            candidate = cursor.replace(
                hour=VACUUM_EVENT_HOUR,
                minute=VACUUM_EVENT_MINUTE,
                second=VACUUM_EVENT_SECOND,
                microsecond=0,
            )
            if start_date <= candidate <= end_date:
                positions.append((candidate - start_date).total_seconds() / duration)
            cursor += timedelta(days=1)
        return positions

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

        C1/K1 (a.k.a. cavity/klystron device "01") is the "problematic" cavity
        — three thermal excursion events where body temperature rises, reflected
        power spikes, and forward power trips. C2/K2 (device "02") is the
        "stable" reference — one minor event for contrast.

        Operators and logbook entries refer to the primary cavity as "C1"; the
        tier-1 channel database addresses it as device "01" (channel name like
        ``SR:RF:CAVITY:01:...``). Both naming conventions resolve to the same
        underlying instrument, so the matcher accepts either.

        Event positions are hardcoded so temperature, power, and voltage channels
        all show correlated behavior at the same time points.
        """
        is_primary = "c1" in pv_lower or "k1" in pv_lower or ":01:" in pv_lower

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

    def _generate_time_series(
        self,
        pv_name: str,
        num_points: int,
        vacuum_event_positions: list[float] | None = None,
    ) -> np.ndarray:
        """
        Generate synthetic time series with trends and noise.

        Creates realistic-looking data with:
        - Sinusoidal variations
        - Linear trends
        - Random noise
        - PV-type-specific characteristics
        - BPMs use random offsets with slow oscillations
        - RF system channels use correlated event patterns
        - Cold-cathode gauges + DCCT share a Sector-7 vacuum-burst event
          whenever the requested window straddles 14:32:08 of any date
        """
        t = np.linspace(0, 1, num_points)
        pv_lower = pv_name.lower()
        rng = np.random.default_rng(seed=hash(pv_name) % (2**32))
        vacuum_event_positions = vacuum_event_positions or []

        # RF system channels — correlated thermal excursion / trip patterns
        if self._is_rf_channel(pv_lower):
            return self._generate_rf_time_series(pv_lower, t, num_points, rng)

        # Cold-cathode vacuum gauges — flat ~5e-8 Torr baseline.  Sector 7
        # spikes to ~2e-7 Torr at each scheduled vacuum-event time; all other
        # sectors remain flat for visual contrast.
        if self._is_cold_cathode_gauge(pv_lower):
            base = 5e-8
            series = np.full(num_points, base) + rng.normal(0, base * 0.03, num_points)
            if self._is_sector7_gauge(pv_lower) and vacuum_event_positions:
                # Width 0.025 ≈ ~15 s in a 10-min window: sharp, visible spike.
                spike_envelope = self._rf_event_envelope(
                    t,
                    [(pos, 1.0) for pos in vacuum_event_positions],
                    width=0.025,
                )
                series = series + spike_envelope * 1.5e-7
            return np.maximum(series, 0)

        # DCCT / beam-current — when a vacuum event is in the window, override
        # the default sawtooth with a ~5 mA Gaussian dip aligned with the
        # Sector-7 gauge spike.  The dip width (0.05) is intentionally about
        # twice the gauge-spike width (0.025) so Pearson r against SR07 lands
        # around -0.88 over a 10-min window — matching the talk slide.  A
        # small residual offset of 0.3 mA keeps a faint "we lost some beam"
        # signal after top-up refills the ring.
        if self._is_dcct_channel(pv_lower) and vacuum_event_positions:
            base = 500.0
            series = np.full(num_points, base)
            for pos in vacuum_event_positions:
                dip = 5.0 * np.exp(-((t - pos) ** 2) / (2 * 0.05**2))
                residual = 0.3 / (1.0 + np.exp(-200.0 * (t - pos)))
                series = series - dip - residual
            series += rng.normal(0, 0.05, num_points)
            return series

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
