"""The archive service: what a channel's past was, on one control target.

:class:`ChannelArchive` is the fifth service kind. It is built like every other
backend, ``Backend(spec, *, project_dir=None, **seams)`` through
:meth:`~osprey_connectors.factory.ConnectorFactory.create_service`, from a
``BackendSpec`` of kind ``archive``: the constructor reads the settings and
does no I/O, and there is no connect step, so a backend opens whatever client
it needs on its first call. Each method answers its value or
:class:`~osprey_connectors.service_answers.Unavailable`, and raises only for
the caller's own arguments (:class:`~osprey_connectors.errors.ArchiveQueryError`,
and a naive or non-datetime instant).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Self

import pandas as pd
from pydantic import model_validator

from osprey_connectors.archiver._timerange import ProcessingMode
from osprey_connectors.model import FrozenModel, Instant
from osprey_connectors.service_answers import Unavailable, _Backend


class ArchiveCoverage(FrozenModel):
    """Whether the archive records one channel, and the span of what it holds.

    ``address`` is the same protocol-agnostic channel address the control-system
    connectors take, and the same value that appears in the ``address`` column of
    :meth:`ChannelArchive.history`.

    ``is_archived`` is ``None`` when this archive cannot say whether it records
    the channel; ``unknown_reason`` then says why, and is set only then.

    ``first_sample_at`` and ``last_sample_at`` are the instants of the oldest and
    newest samples the archive holds for the channel, in UTC; ``None`` when it
    cannot say. ``sampling_period_s`` is the seconds between its samples, where
    the archive states one.
    """

    address: str
    is_archived: bool | None
    first_sample_at: Instant | None = None
    last_sample_at: Instant | None = None
    sampling_period_s: float | None = None
    unknown_reason: str | None = None

    @model_validator(mode="after")
    def _reason_only_when_unknown(self) -> Self:
        if self.unknown_reason is not None and self.is_archived is not None:
            raise ValueError("unknown_reason is set only when is_archived is None")
        return self


class ArchiveExtent(FrozenModel):
    """What an archive store holds.

    ``addresses`` is the set of channel addresses the store records now;
    ``oldest_at`` and ``newest_at`` are the instants of its oldest and newest
    samples, in UTC. A field is ``None`` when the store holds nothing to say it
    with; a store that cannot be asked answers ``Unavailable`` instead.
    """

    addresses: frozenset[str] | None = None
    oldest_at: Instant | None = None
    newest_at: Instant | None = None


#: The reason the default :meth:`ChannelArchive.extent` answers.
NO_EXTENT_REASON = "this archiver does not report what it holds"

#: The reason the default :meth:`ChannelArchive.probe` answers.
_NO_PROBE_REASON = "this archiver has no probe of its own; a history read asks it"


class ChannelArchive(_Backend, ABC):
    """The archived history of the channels of one control target.

    Example:
        >>> cfg = current_config()
        >>> table = service_targets(
        >>>     cfg.pv_directory,
        >>>     cfg.archiver,
        >>>     simulator_port=virtual_accelerator_port(cfg.raw_tree()),
        >>>     built_simulator=has_simulator(cfg.raw_tree()),
        >>> )
        >>> archive = await ConnectorFactory.create_service(table.archive["live"])
        >>> try:
        >>>     frame = await archive.history(
        >>>         ['BEAM:CURRENT', 'BEAM:LIFETIME'],
        >>>         start_at=datetime(2024, 1, 1, tzinfo=UTC),
        >>>         end_at=datetime(2024, 1, 2, tzinfo=UTC),
        >>>     )
        >>> finally:
        >>>     await archive.aclose()
    """

    @abstractmethod
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
        The archived history of one or more channels.

        Args:
            addresses: The channel addresses to read. Addresses are
                protocol-agnostic: EPICS PV names, DOOCS property addresses, or
                whatever the configured backend keys its store by.
            start_at: Start of the time range, timezone-aware. A naive
                datetime is refused: it names no instant.
            end_at: End of the time range, timezone-aware.
            precision_ms: Bin width in milliseconds; ``<= 0`` means full
                resolution. A backend that cannot express the requested width
                must raise ``ArchiveQueryError`` rather than serve a different one.
                Bins are anchored on ``start_at``: the same window and width
                cut the same lattice for every channel in the request and for
                every processing mode. A backend that bins on a grid of its own
                is the documented exception — the EPICS Archiver Appliance's
                operators are epoch-anchored, and the window merely clips them.
            processing: Aggregation applied within each precision_ms bin. One of
                "raw", "mean", "min", "max", "median", "std", "count". Anything
                else raises ``ArchiveQueryError``.
            timeout_s: The seconds this call may spend; ``None`` is the
                backend's configured default.

        Waveform channels:
            A waveform channel's sample is one whole trace: its ``value`` is a
            ``list[float]`` with one entry per element of the channel's shape.
            Only ``"raw"`` processing is defined for waveforms, and it returns
            every stored trace in the window at its own timestamp, unbinned,
            whatever ``precision_ms`` is — dropping or averaging traces would
            describe pulses nothing ever captured. Any other mode on a waveform
            channel raises ``ArchiveQueryError`` naming the channel. A backend
            that cannot read waveforms at all answers ``Unavailable`` with cause
            ``unsupported`` rather than returning something else in their place.

        Returns:
            A long-format DataFrame with columns ``timestamp``
            (datetime64[ns, UTC]), ``address`` (str), and ``value`` (``float64``
            unless a channel is non-numeric or a waveform), sorted by address
            then timestamp. Each channel contributes only its own real samples or per-bin
            aggregates — no forward-fill, no shared grid, no bin for a period
            with no samples. A channel with no data in range, or one the archive
            does not record, contributes no rows; an empty result is an empty
            frame with these columns. ``Unavailable`` when the archive gave no
            answer, its ``cause`` saying why.

        Raises:
            TypeError: If ``start_at`` or ``end_at`` is not a ``datetime``
            NaiveDatetimeError: If ``start_at`` or ``end_at`` is naive (a
                ``ValueError``)
            ArchiveQueryError: If the time range, the bin width or the
                processing mode cannot be asked of this archive, or a non-"raw"
                processing mode is requested for a channel whose values are
                non-numeric or that is a waveform channel (a ``ValueError``)
        """

    @abstractmethod
    async def coverage(
        self, addresses: Sequence[str], *, timeout_s: float | None = None
    ) -> Mapping[str, ArchiveCoverage] | Unavailable:
        """Whether the archive records each channel, and the span it holds for it.

        Args:
            addresses: The channel addresses to ask about.
            timeout_s: The seconds this call may spend; ``None`` is the
                backend's configured default.

        Returns:
            One :class:`ArchiveCoverage` per address, keyed by it; a channel
            the archive cannot say anything about has ``is_archived`` ``None``
            and the reason. ``Unavailable`` when the archive gave no answer.

        Raises:
            ArchiveQueryError: If an address cannot be asked of this archive.
        """

    async def extent(self, *, timeout_s: float | None = None) -> ArchiveExtent | Unavailable:  # noqa: ARG002 - the contract's budget, unused by the default
        """What the store holds: its channels, oldest and newest sample, in UTC.

        Optional. The default answers ``Unavailable`` with cause ``unsupported``,
        so an archive that cannot list its store keeps working unchanged.
        """
        return Unavailable(cause="unsupported", reason=NO_EXTENT_REASON)

    async def probe(self, *, timeout_s: float | None = None) -> str | Unavailable:  # noqa: ARG002 - the contract's budget, unused by the default
        """Whether the archive answers: the evidence in plain words, or why not.

        Optional. The default answers ``Unavailable`` with cause ``unsupported``.
        """
        return Unavailable(cause="unsupported", reason=_NO_PROBE_REASON)


class _NoAnswer(Exception):
    """Carries an ``Unavailable`` out of a backend's worker thread to the method that answers it.

    Private to the archives: raised where a failure is classified, caught by the
    method that returns ``answer``; it never leaves a :class:`ChannelArchive`.
    """

    def __init__(self, answer: Unavailable) -> None:
        self.answer = answer
        super().__init__(answer.reason)
