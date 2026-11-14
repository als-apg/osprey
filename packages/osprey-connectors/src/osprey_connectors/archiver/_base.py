"""
Abstract base class for archiver connectors.

Provides protocol-agnostic interfaces for retrieving historical data
from various archiver systems (EPICS Archiver Appliance, Tango HDB++, LabVIEW, etc.).

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class ArchiverMetadata:
    """Archiving metadata for one channel.

    ``channel`` is the same protocol-agnostic channel address the control-system
    connectors take, and the same value that appears in the ``channel`` column of
    :meth:`ArchiverConnector.get_data`.

    ``is_archived`` is ``None`` when this archiver cannot say whether it archives
    the channel; ``description`` then says why.
    """

    channel: str
    is_archived: bool | None
    archival_start: datetime | None = None
    archival_end: datetime | None = None
    sampling_period: float | None = None
    description: str | None = None


@dataclass(frozen=True)
class ArchiveExtent:
    """What an archive store holds, as far as the archiver can say.

    Every field is ``None`` when unknown. ``channels`` is the set of channel
    addresses the store records now; ``oldest`` and ``newest`` are the instants
    of its oldest and newest samples; ``reason`` says why a part is unknown.
    """

    channels: frozenset[str] | None = None
    oldest: datetime | None = None
    newest: datetime | None = None
    reason: str | None = None


#: The reason the default :meth:`ArchiverConnector.store_extent` answers.
NO_EXTENT_REASON = "this archiver does not report what it holds"


class ArchiverConnector(ABC):
    """
    Abstract base class for archiver connectors.

    Implementations provide interfaces to different archiver systems
    using a unified API that returns pandas DataFrames.

    Example:
        >>> connector = await ConnectorFactory.create_archiver_connector()
        >>> try:
        >>>     df = await connector.get_data(
        >>>         channels=['BEAM:CURRENT', 'BEAM:LIFETIME'],
        >>>         start_date=datetime(2024, 1, 1),
        >>>         end_date=datetime(2024, 1, 2)
        >>>     )
        >>>     print(df.head())
        >>> finally:
        >>>     await connector.disconnect()
    """

    @abstractmethod
    async def connect(self, config: dict[str, Any]) -> None:
        """
        Establish connection to archiver.

        Args:
            config: Archiver-specific configuration

        Raises:
            ConnectorUnavailableError: If connection cannot be established
        """

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to archiver and cleanup resources."""

    @abstractmethod
    async def get_data(
        self,
        channels: list[str],
        start_date: datetime,
        end_date: datetime,
        precision_ms: int = 1000,
        timeout: int | None = None,
        processing: str = "raw",
    ) -> pd.DataFrame:
        """
        Retrieve historical data for one or more channels.

        Args:
            channels: List of channel addresses to retrieve. Addresses are
                protocol-agnostic: EPICS PV names, DOOCS property addresses, or
                whatever the configured backend keys its store by.
            start_date: Start of time range
            end_date: End of time range
            precision_ms: Bin width in milliseconds; ``<= 0`` means full
                resolution. A backend that cannot express the requested width
                must raise ``ValueError`` rather than serve a different one.
                Bins are anchored on ``start_date``: the same window and width
                cut the same lattice for every channel in the request and for
                every processing mode. A backend that bins on a grid of its own
                is the documented exception — the EPICS Archiver Appliance's
                operators are epoch-anchored, and the window merely clips them.
            timeout: Optional timeout in seconds
            processing: Aggregation applied within each precision_ms bin. One of
                "raw", "mean", "min", "max", "median", "std", "count". Anything
                else raises ValueError.

        Waveform channels:
            A waveform channel's sample is one whole trace: its ``value`` is a
            ``list[float]`` with one entry per element of the channel's shape.
            Only ``"raw"`` processing is defined for waveforms, and it returns
            every stored trace in the window at its own timestamp, unbinned,
            whatever ``precision_ms`` is — dropping or averaging traces would
            describe pulses nothing ever captured. Any other mode on a waveform
            channel raises ``ValueError`` naming the channel. A backend that
            cannot read waveforms at all says so with its own error rather than
            returning something else in their place.

        Returns:
            A long-format DataFrame with columns ``timestamp``
            (datetime64[ns, UTC]), ``channel`` (str), and ``value`` (``float64``
            unless a channel is non-numeric or a waveform), sorted by channel
            then timestamp. Each channel contributes only its own real samples or per-bin
            aggregates — no forward-fill, no shared grid, no bin for a period
            with no samples. A channel with no data in range contributes no rows; an
            empty result is an empty frame with these columns.

        Raises:
            ConnectorUnavailableError: If archiver cannot be reached
            TimeoutError: If operation times out
            ValueError: If time range, channel addresses, or processing mode are
                invalid, or if a non-"raw" processing mode is requested for a
                channel whose values are non-numeric or that is a waveform channel
        """

    @abstractmethod
    async def get_metadata(self, channel: str) -> ArchiverMetadata:
        """
        Get archiving metadata for a channel.

        Args:
            channel: Channel address

        Returns:
            ArchiverMetadata with archiving information

        Raises:
            ConnectorUnavailableError: If archiver cannot be reached
            ValueError: If the channel address is invalid
        """

    @abstractmethod
    async def check_availability(self, channels: list[str]) -> dict[str, bool | None]:
        """
        Check which channels are archived.

        Args:
            channels: List of channel addresses to check

        Returns:
            Dictionary mapping channel address to availability status; ``None``
            where this archiver cannot say.

        Raises:
            RuntimeError: If the archiver is not connected; a disconnected
                archiver has no store to ask, so it answers nothing.
        """

    async def store_extent(self) -> ArchiveExtent:
        """What the store holds: its channels, oldest and newest sample.

        Optional. The default answers every part unknown, with a reason, so an
        archiver that cannot list its store keeps working unchanged.
        """
        return ArchiveExtent(reason=NO_EXTENT_REASON)
