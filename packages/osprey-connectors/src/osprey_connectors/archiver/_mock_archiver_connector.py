"""
Mock archiver connector for development and testing.

Serves the history of the addresses in the built simulator view, read from the
archive composite (:func:`osprey_connectors.simulation.archive.build`) at the
active scenario set. An address outside the view was never recorded: it has no
rows, and its coverage says it is not archived.

"""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pandas as pd
from pydantic import model_validator
from typing_extensions import override

from osprey_connectors.archiver._base import ArchiveCoverage, ChannelArchive
from osprey_connectors.archiver._timerange import (
    ProcessingMode,
    aggregate_long_frame,
    resolve_processing,
    utc_window,
)
from osprey_connectors.config import facility_timezone
from osprey_connectors.config_sections import ConnectorSettings
from osprey_connectors.control_system._va_in_process_connector import (
    SIMULATOR_VIEW_SETTING,
    simulation_state_dir,
    simulator_view_dir,
)
from osprey_connectors.errors import ConfigurationError
from osprey_connectors.logger import get_logger
from osprey_connectors.service_answers import BackendSpec, Unavailable
from osprey_connectors.simulation.series import _epoch_seconds_array
from osprey_connectors.simulation.view import SimulatorView, ViewSchemaError

if TYPE_CHECKING:
    from osprey_connectors.simulation.archive import ArchiveComposite

logger = get_logger("mock_archiver_connector")


class MockArchiverSettings(ConnectorSettings):
    """The mock archive's block, ``archiver.settings``, read.

    Attributes:
        sample_rate_hz: The sampling rate; a number greater than zero, 1.0
            when unset.
        simulator_view: The view directory to serve; ``None`` serves
            ``data/simulator/`` beside the loaded config. A value that is not a
            path is ``None`` too.
    """

    sample_rate_hz: float = 1.0
    simulator_view: str | Path | None = None

    @model_validator(mode="before")
    @classmethod
    def _read(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        # A zero or negative rate would divide by zero later; reject it at
        # configuration time, along with a rate that is not a number at all.
        sample_rate_hz: object = data.get("sample_rate_hz", 1.0)
        if not isinstance(sample_rate_hz, int | float) or sample_rate_hz <= 0:
            raise ConfigurationError(
                f"sample_rate_hz must be > 0 (got {sample_rate_hz})",
                key="archiver.settings.sample_rate_hz",
            )
        view = data.get(SIMULATOR_VIEW_SETTING)
        return {
            **data,
            "sample_rate_hz": float(sample_rate_hz),
            SIMULATOR_VIEW_SETTING: view if isinstance(view, str | Path) else None,
        }


class MockArchiverConnector(ChannelArchive):
    """
    Mock archiver for development - the history of the built simulator view.

    Features:
    - Serves the addresses of the built simulator view; any other address was
      never recorded
    - Values are a pure function of (channel, absolute timestamp), so two
      overlapping windows agree on every timestamp they share
    - Follows the active scenario set, rebuilt when it changes
    - Configurable sampling rate
    - Returns pandas DataFrames matching real archiver format

    The spec's settings:

    - sample_rate_hz: Sampling rate (default: 1.0)
    - simulator_view: The view directory to serve; absent serves
      ``data/simulator/`` beside the loaded config.

    The view is opened, and its archive composite built, on the first call; a
    view that is missing, from an older build, or whose physics model fails to
    build at the start state answers ``Unavailable``.

    The settings are read into :class:`MockArchiverSettings`.

    Raises:
        ConfigurationError: ``sample_rate_hz`` is not a number greater than zero.

    Example:
        >>> archive = MockArchiverConnector(BackendSpec(
        >>>     kind="archive", type="mock_archiver", target="va", rehearsal=False,
        >>>     settings={'sample_rate_hz': 1.0},
        >>> ))
        >>> frame = await archive.history(
        >>>     ['SR:DIAG:BPM:01:POSITION:X'],
        >>>     start_at=datetime(2024, 1, 1, tzinfo=UTC),
        >>>     end_at=datetime(2024, 1, 2, tzinfo=UTC),
        >>> )
    """

    settings_model: ClassVar[type[ConnectorSettings]] = MockArchiverSettings
    SETTINGS_KEY: ClassVar[str] = "archiver.settings"

    def __init__(
        self, spec: BackendSpec, *, project_dir: Path | None = None, **seams: object
    ) -> None:
        super().__init__(spec, project_dir=project_dir, **seams)
        config = self._settings(MockArchiverSettings)
        self._sample_rate_hz: float = config.sample_rate_hz
        self._view_setting: str | Path | None = config.simulator_view
        self._archive: ArchiveComposite | None = None
        self._view: SimulatorView | None = None
        self._state_file: Path | None = None
        self._state_signature: tuple[int, int] | None = None

    def _signature(self) -> tuple[int, int] | None:
        if self._state_file is None:
            return None
        try:
            stat = self._state_file.stat()
        except FileNotFoundError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def _rebuild(self) -> None:
        """Build the archive composite at the active set the state file names.

        A failed build keeps the previous composite serving.
        """
        from osprey_connectors.simulation.archive import build
        from osprey_connectors.simulation.state import ActiveState, parse_active_state

        assert self._view is not None
        # The signature is taken before the file is read, so a rewrite between
        # the two shows as a change on the next read.
        signature = self._signature()
        active = ActiveState()
        if signature is not None and self._state_file is not None:
            try:
                active = parse_active_state(self._state_file.read_text(encoding="utf-8"))
            except FileNotFoundError:
                signature = None
        self._archive = build(self._view, active, tz=facility_timezone())
        self._state_signature = signature

    def _open(self) -> None:
        """Open the simulator view and build its archive composite, the first time.

        Raises:
            RuntimeError: There is no built simulator view, it is from an older
                build, or a physics model fails to build at the start state.
        """
        from osprey_connectors.simulation.state import ACTIVE_SCENARIOS_FILENAME

        self._view = simulator_view_dir(self._view_setting)
        self._state_file = simulation_state_dir(self._view.path) / ACTIVE_SCENARIOS_FILENAME
        try:
            self._rebuild()
        except ViewSchemaError as error:
            raise RuntimeError(str(error)) from None
        logger.debug(f"Mock archiver connector serving {self._view.path}")

    def _current(self) -> "ArchiveComposite | Unavailable":
        """The archive composite, rebuilt when the active set changed since the last read."""
        if self._archive is None:
            try:
                self._open()
            except RuntimeError as error:
                return Unavailable(cause="misconfigured", reason=str(error))
        elif self._signature() != self._state_signature:
            try:
                self._rebuild()
            except (RuntimeError, ViewSchemaError) as error:
                return Unavailable(cause="failed", reason=str(error))
        assert self._archive is not None
        return self._archive

    @override
    async def aclose(self) -> None:
        """Drop the archive composite; the next call builds it again."""
        self._archive = None
        logger.debug("Mock archiver connector closed")

    @override
    async def history(
        self,
        addresses: Sequence[str],
        *,
        start_at: datetime,
        end_at: datetime,
        precision_ms: int = 1000,
        processing: ProcessingMode = "raw",
        timeout_s: float | None = None,
    ) -> pd.DataFrame | Unavailable:
        """
        The archived history of channels of the simulator view.

        Args:
            addresses: Channel addresses of the simulator view; any other
                address has no rows
            start_at: Start of the time range, timezone-aware.
            end_at: End of the time range, timezone-aware.
            precision_ms: Time precision (affects downsampling). ``<= 0`` means
                full resolution: samples are generated at the connector's own
                configured ``sample_rate_hz``. Either way the generator is
                capped at 10,000 points.
            processing: Aggregation applied within each precision_ms bin. One of
                "raw", "mean", "min", "max", "median", "std", "count". Applied
                client-side via pandas resampling. Anything else raises
                ArchiveQueryError.
            timeout_s: Ignored for mock archiver

        Returns:
            The canonical long frame — see :meth:`ChannelArchive.history` — or
            ``Unavailable`` when the simulator view cannot be served.

        Raises:
            ArchiveQueryError: ``processing`` other than ``"raw"`` is requested
                for a channel holding non-numeric values.
        """
        # long_frame requires a UTC-aware index; a naive bound is refused.
        start_utc, end_utc = utc_window(start_at, end_at)
        resolved = resolve_processing(processing, precision_ms, start_utc)

        archive = self._current()
        if isinstance(archive, Unavailable):
            return archive
        served = set(archive.addresses)
        recorded = [channel for channel in addresses if channel in served]

        duration = (end_utc - start_utc).total_seconds()

        # Limit number of points for performance. precision_ms <= 0 means full
        # resolution, which for the mock (no backing store) means generating at
        # the configured native rate; this also avoids dividing by zero.
        effective_precision_ms = precision_ms if precision_ms > 0 else 1000.0 / self._sample_rate_hz
        num_points = min(int(duration / (effective_precision_ms / 1000.0)), 10000)
        num_points = max(num_points, 10)  # At least 10 points

        index = pd.date_range(start=start_utc, end=end_utc, periods=num_points)

        # Every sample is evaluated at the grid's absolute timestamps, so a
        # store seeded from the same archive composite holds the same values.
        t_abs = _epoch_seconds_array(index)
        if t_abs is None:  # pragma: no cover - the index above is always datetimes
            raise ValueError(f"Cannot derive epoch seconds for the {start_utc} to {end_utc} grid")

        series = {
            channel: pd.Series(archive.series(channel, t_abs), index=index, name=channel)
            for channel in recorded
        }

        data = aggregate_long_frame(series, resolved)

        logger.debug(
            f"Mock archiver generated {len(data)} rows across "
            f"{len(recorded)} channels from {start_utc} to {end_utc}"
        )

        return data

    @override
    async def coverage(
        self, addresses: Sequence[str], *, timeout_s: float | None = None
    ) -> Mapping[str, ArchiveCoverage] | Unavailable:
        """Whether the simulator view holds each channel; a held one spans 2000 to now."""
        archive = self._current()
        if isinstance(archive, Unavailable):
            return archive
        served = set(archive.addresses)
        now = datetime.now(UTC)
        return {
            address: ArchiveCoverage(
                address=address,
                is_archived=True,
                first_sample_at=datetime(2000, 1, 1, tzinfo=UTC),
                last_sample_at=now,
                sampling_period_s=1.0 / self._sample_rate_hz,
            )
            if address in served
            else ArchiveCoverage(address=address, is_archived=False)
            for address in addresses
        }

    @override
    async def probe(self, *, timeout_s: float | None = None) -> str | Unavailable:
        """How many channels the simulator view serves, or why it cannot be served."""
        archive = self._current()
        if isinstance(archive, Unavailable):
            return archive
        return f"the simulator view serves {len(archive.addresses)} archived channels"
