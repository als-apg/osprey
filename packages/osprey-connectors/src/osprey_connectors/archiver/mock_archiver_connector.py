"""
Mock archiver connector for development and testing.

Serves the history of the addresses in the built simulator view, read from the
archive composite (:func:`osprey_connectors.simulation.archive.build`) at the
active scenario set. An address outside the view is refused.

"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from osprey_connectors.archiver._timerange import (
    aggregate_long_frame,
    resolve_processing,
    utc_window,
)
from osprey_connectors.archiver.base import ArchiverConnector, ArchiverMetadata
from osprey_connectors.config import get_facility_timezone
from osprey_connectors.control_system.mock_connector import (
    SIMULATOR_VIEW_SETTING,
    not_in_facility,
    simulation_state_dir,
    simulator_view_dir,
)
from osprey_connectors.logger import get_logger
from osprey_connectors.simulation.series import epoch_seconds_array
from osprey_connectors.simulation.view import SimulatorView, ViewSchemaError

if TYPE_CHECKING:
    from osprey_connectors.simulation.archive import ArchiveComposite

logger = get_logger("mock_archiver_connector")


class MockArchiverConnector(ArchiverConnector):
    """
    Mock archiver for development - the history of the built simulator view.

    Features:
    - Serves the addresses of the built simulator view and refuses any other
    - Values are a pure function of (channel, absolute timestamp), so two
      overlapping windows agree on every timestamp they share
    - Follows the active scenario set, rebuilt when it changes
    - Configurable sampling rate
    - Returns pandas DataFrames matching real archiver format

    Example:
        >>> config = {
        >>>     'sample_rate_hz': 1.0,
        >>> }
        >>> connector = MockArchiverConnector()
        >>> await connector.connect(config)
        >>> df = await connector.get_data(
        >>>     channels=['SR:DIAG:BPM:01:POSITION:X'],
        >>>     start_date=datetime(2024, 1, 1),
        >>>     end_date=datetime(2024, 1, 2)
        >>> )
    """

    def __init__(self):
        self._connected = False
        self._archive: ArchiveComposite | None = None
        self._view: SimulatorView | None = None
        self._state_file: Path | None = None
        self._state_signature: tuple[int, int] | None = None

    async def connect(self, config: dict[str, Any]) -> None:
        """
        Build the archive composite of the simulator view.

        Args:
            config: Configuration with keys:
                - sample_rate_hz: Sampling rate (default: 1.0)
                - simulator_view: The view directory to serve; absent serves
                  ``data/simulator/`` beside the loaded config.

        Raises:
            ValueError: ``sample_rate_hz`` is not greater than zero.
            RuntimeError: There is no built simulator view, it is from an
                older build, or a physics model fails to build at the start
                state.
        """
        # A zero or negative rate would divide by zero later; reject it at
        # configuration time.
        sample_rate_hz = config.get("sample_rate_hz", 1.0)
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be > 0 (got {sample_rate_hz})")
        self._sample_rate_hz = sample_rate_hz

        from osprey_connectors.simulation.state import ACTIVE_SCENARIOS_FILENAME

        self._view = simulator_view_dir(config.get(SIMULATOR_VIEW_SETTING))
        self._state_file = simulation_state_dir(self._view.path) / ACTIVE_SCENARIOS_FILENAME
        try:
            self._rebuild()
        except ViewSchemaError as error:
            raise RuntimeError(str(error)) from None

        self._connected = True
        logger.debug(f"Mock archiver connector serving {self._view.path}")

    def _signature(self) -> tuple[int, int] | None:
        if self._state_file is None:
            return None
        try:
            stat = self._state_file.stat()
        except FileNotFoundError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def _rebuild(self) -> None:
        """Build the archive composite at the active set the state file names."""
        from osprey_connectors.simulation.archive import build
        from osprey_connectors.simulation.state import parse_active_state

        assert self._view is not None
        # The signature is taken before the file is read, so a rewrite between
        # the two shows as a change on the next read.
        signature = self._signature()
        names: list[str] = []
        anchor_s: float | None = None
        if signature is not None and self._state_file is not None:
            try:
                names, anchor_s = parse_active_state(self._state_file.read_text(encoding="utf-8"))
            except FileNotFoundError:
                signature = None
        self._archive = build(self._view, names, anchor_s=anchor_s)
        self._state_signature = signature

    def _current(self) -> "ArchiveComposite":
        """The archive composite, rebuilt when the active set changed since the last read."""
        if self._archive is None:
            raise RuntimeError("mock archiver is not connected")
        if self._signature() != self._state_signature:
            self._rebuild()
        assert self._archive is not None
        return self._archive

    def _require(self, channels: list[str]) -> "ArchiveComposite":
        archive = self._current()
        served = set(archive.addresses)
        for channel in channels:
            if channel not in served:
                raise ValueError(not_in_facility(channel))
        return archive

    async def disconnect(self) -> None:
        """Cleanup mock archiver."""
        self._archive = None
        self._connected = False
        logger.debug("Mock archiver connector disconnected")

    async def get_data(
        self,
        channels: list[str],
        start_date: datetime,
        end_date: datetime,
        precision_ms: int = 1000,
        timeout: int | None = None,  # noqa: ARG002 - ArchiverConnector.get_data signature; generated samples never wait on a transport
        processing: str = "raw",
    ) -> pd.DataFrame:
        """
        The archived history of channels of the simulator view.

        Args:
            channels: Channel addresses of the simulator view
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
            ValueError: A channel is not in the built facility file, or
                ``processing`` other than ``"raw"`` is requested for a channel
                holding non-numeric values.
        """
        archive = self._require(list(channels))

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

        index = pd.date_range(start=start_date, end=end_date, periods=num_points)

        # Every sample is evaluated at the grid's absolute timestamps, so a
        # store seeded from the same archive composite holds the same values.
        t_abs = epoch_seconds_array(index)
        if t_abs is None:  # pragma: no cover - the index above is always datetimes
            raise ValueError(f"Cannot derive epoch seconds for the {start_date} to {end_date} grid")

        resolved = resolve_processing(processing, precision_ms, start_date)
        series = {
            channel: pd.Series(archive.series(channel, t_abs), index=index, name=channel)
            for channel in channels
        }

        data = aggregate_long_frame(series, resolved)

        logger.debug(
            f"Mock archiver generated {len(data)} rows across "
            f"{len(channels)} channels from {start_date} to {end_date}"
        )

        return data

    async def get_metadata(self, channel: str) -> ArchiverMetadata:
        """Mock archiver metadata for a channel of the simulator view.

        Raises:
            ValueError: The channel is not in the built facility file.
        """
        self._require([channel])
        return ArchiverMetadata(
            channel=channel,
            is_archived=True,
            # Both bounds tz-aware (facility zone) so a consumer can subtract or
            # compare them without a naive/aware TypeError.
            archival_start=datetime(2000, 1, 1, tzinfo=get_facility_timezone()),
            archival_end=datetime.now(get_facility_timezone()),
            sampling_period=1.0 / self._sample_rate_hz,
            description=f"Mock archived channel: {channel}",
        )

    async def check_availability(self, channels: list[str]) -> dict[str, bool]:
        """Whether the simulator view holds each channel."""
        served = set(self._current().addresses)
        return {channel: channel in served for channel in channels}
