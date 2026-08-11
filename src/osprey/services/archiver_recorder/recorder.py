"""The recording loop: sample the machine on a grid, store what answered.

Three behaviours here are load-bearing and each has a quiet failure mode it is
written to avoid:

* **Sampling lands on an absolute grid.** Timestamps are floored to a multiple
  of the sample cadence since the epoch, not spaced from whenever the process
  started. That is what puts recorded documents on the same instants the seeder
  used, which is in turn what lets the scenario seeder rewrite an event window
  across the seam without leaving two interleaved grids behind.
* **A channel that did not answer contributes nothing.** No last-known value is
  carried forward and no placeholder is written. A gap in the archive is the
  honest record of a channel that was not answering, and it is the only record
  that lets anyone later tell a quiet channel from an absent one.
* **Enablement is polled, and a torn read changes nothing.** Config writes are
  truncate-in-place rather than atomic, so a poll can land mid-write and see a
  half file. Treating that as an answer would stop and restart recording on a
  write that changed nothing at all.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from .config import RecorderConfigError, RecorderSettings, read_control_system_type
from .store import ArchiveWriter

logger = logging.getLogger(__name__)

#: The one control system whose readings belong in this archive. A simulated
#: machine's history is synthesized for the past and recorded for the present,
#: and the two halves are the same kind of thing. A real machine's readings are
#: not: splicing them onto a synthesized past would produce exactly the
#: two-world archive this service exists to prevent.
RECORDING_CONTROL_SYSTEM = "virtual_accelerator"

#: Reads a set of addresses and returns the ones that answered.
ChannelReader = Callable[[Sequence[str]], Awaitable[Mapping[str, float]]]


class Recorder:
    """Samples the control system into the archive for as long as it runs."""

    def __init__(
        self,
        *,
        settings: RecorderSettings,
        addresses: Sequence[str],
        writer: ArchiveWriter,
        config_path: Path,
        reader: ChannelReader | None = None,
        recording: bool = False,
    ) -> None:
        self._settings = settings
        self._addresses = list(addresses)
        self._writer = writer
        self._config_path = Path(config_path)
        self._reader = reader if reader is not None else channel_access_reader(settings)
        self._recording = recording
        self._announced = False
        self._last_slot: int | None = None
        self._last_poll: float | None = None
        self._non_numeric: set[str] = set()
        self._stop = asyncio.Event()

    @property
    def recording(self) -> bool:
        """Whether the last enablement answer said to record."""
        return self._recording

    def refresh_enablement(self, now: datetime) -> bool:
        """Re-read ``control_system.type`` if the poll interval has elapsed.

        Returns the enablement state in force after this call — unchanged when
        the interval has not elapsed yet, and unchanged when the read failed.
        State transitions log one line each; steady state logs nothing, so a
        recorder idling for a week does not fill the logs saying so.
        """
        stamp = now.timestamp()
        if self._last_poll is not None and stamp - self._last_poll < self._settings.poll_sec:
            return self._recording
        self._last_poll = stamp

        try:
            control_system = read_control_system_type(self._config_path)
        except RecorderConfigError as exc:
            # Keep the last known answer. See the module docstring: this is
            # very likely a config write in progress, and the next poll will
            # read the finished file.
            logger.debug("enablement poll could not read the config, keeping last state: %s", exc)
            return self._recording

        should_record = control_system == RECORDING_CONTROL_SYSTEM
        if should_record != self._recording or not self._announced:
            if should_record:
                logger.info(
                    "Recording %d channels every %ds into %s.%s (control_system.type=%s).",
                    len(self._addresses),
                    self._settings.cadence_sec,
                    self._settings.database,
                    self._settings.collection,
                    control_system,
                )
            else:
                logger.info(
                    "Idle, writing nothing: control_system.type=%s, and only %s is recorded. "
                    "Flipping it in %s takes effect within %ds — no restart needed.",
                    control_system or "<unset>",
                    RECORDING_CONTROL_SYSTEM,
                    self._config_path,
                    self._settings.poll_sec,
                )
            self._recording = should_record
            self._announced = True
        return self._recording

    def slot_for(self, now: datetime) -> int:
        """The absolute grid instant, in whole seconds, that ``now`` falls in."""
        cadence = self._settings.cadence_sec
        return int(now.timestamp()) // cadence * cadence

    async def tick(self, now: datetime) -> bool:
        """Run one iteration: poll enablement, and record the current slot.

        Returns True if a document was written. A slot already recorded is
        skipped rather than written twice, so a restart mid-interval cannot
        double-write the instant the previous run just covered.
        """
        if not self.refresh_enablement(now):
            return False

        slot = self.slot_for(now)
        if slot == self._last_slot:
            return False

        values = await self._read_channels()
        if not values:
            logger.warning(
                "No channel answered at %s; writing nothing for this sample. "
                "A gap here is the honest record of an unreachable control system.",
                datetime.fromtimestamp(slot, tz=UTC).isoformat(),
            )
            self._last_slot = slot
            return False

        timestamp = datetime.fromtimestamp(slot, tz=UTC)
        try:
            await asyncio.to_thread(self._writer.write_sample, timestamp, values)
        except Exception as exc:  # noqa: BLE001 — see below
            # Never let one failed write end the loop. The store may be
            # restarting, and the next tick is seconds away; a crash here would
            # instead take the Channel Access channel cache down with it and
            # reconnect every channel from scratch.
            logger.warning("Failed to store the sample at %s: %s", timestamp.isoformat(), exc)
            self._last_slot = slot
            return False

        self._last_slot = slot
        logger.debug("Stored %d channel values at %s", len(values), timestamp.isoformat())
        return True

    async def run_forever(self) -> None:
        """Tick on the sample grid until :meth:`stop` is called.

        Sleeps to the next grid boundary rather than a fixed interval, so
        samples stay on the absolute grid however long a tick took.
        """
        while not self._stop.is_set():
            await self.tick(datetime.now(tz=UTC))
            delay = self._seconds_to_next_slot(datetime.now(tz=UTC))
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            except TimeoutError:
                continue

    def stop(self) -> None:
        """Ask :meth:`run_forever` to return at the next opportunity."""
        self._stop.set()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _seconds_to_next_slot(self, now: datetime) -> float:
        cadence = self._settings.cadence_sec
        elapsed = now.timestamp() % cadence
        return cadence - elapsed

    async def _read_channels(self) -> dict[str, float]:
        raw = await self._reader(self._addresses)
        values: dict[str, float] = {}
        for address, value in raw.items():
            try:
                values[address] = float(value)
            except (TypeError, ValueError):
                # The archive schema is one number per channel per instant.
                # A channel that is not a number (a string status, a waveform)
                # has no place in it — said once per address, because saying it
                # every cadence would bury everything else.
                if address not in self._non_numeric:
                    self._non_numeric.add(address)
                    logger.warning(
                        "Channel %s returned a non-numeric value (%r); it will not be recorded.",
                        address,
                        value,
                    )
        return values


def channel_access_reader(settings: RecorderSettings) -> ChannelReader:
    """The production reader: one long-lived Channel Access client.

    ``aioca`` caches its channels across calls, so the connection cost is paid
    once and every later sample reuses it — and its reconnection is what makes
    this service survive an IOC restart without any logic here: the channels go
    disconnected, those reads fail (contributing nothing), and they resume
    answering once the IOC is back.

    Failed reads come back as falsy ``CANothing`` sentinels rather than
    exceptions, which is why ``throw=False`` is safe: one unreachable channel
    must not cost the whole sample.
    """

    async def read(addresses: Sequence[str]) -> Mapping[str, float]:
        from aioca import caget

        # Bounded by the sample cadence: a read still outstanding when the next
        # sample is due has already missed its slot, and waiting longer would
        # only push every later sample off the grid.
        results = await caget(list(addresses), timeout=settings.cadence_sec, throw=False)
        return {
            address: value
            for address, value in zip(addresses, results, strict=True)
            if getattr(value, "ok", True)
        }

    return read
