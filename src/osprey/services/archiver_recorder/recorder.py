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
  write that changed nothing at all. The poll asks who is on the other end, not
  merely what the config calls them — see :data:`RECORDING_CONTROL_SYSTEM`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from osprey_connectors.standin import DEPLOYED_SERVICES_KEY, LIVE_STANDIN_PORT_KEY

from .config import RecorderConfigError, RecorderSettings, read_recording_facts
from .store import ArchiveWriter

logger = logging.getLogger(__name__)

#: The control-system type that, on its own, makes this deployment's machine one
#: whose past this deployment authored.
#:
#: **The archive belongs to the machine it records, and a model has no past.**
#: This service samples whatever machine the deployment calls its machine, and
#: stores it in the collection that already holds that machine's history. For a
#: virtual accelerator — the type named here — the past was synthesized and the
#: present is recorded, and the two halves are the same kind of thing on one
#: timeline. A sandbox model nobody calls their machine has no archive of its
#: own to be written into. And a real machine's readings are never spliced onto
#: a synthesized past: that two-world archive is what this gate exists to
#: prevent.
#:
#: Which is why this constant is only half the gate. A deployment can stand a
#: *stand-in* up — a second virtual accelerator wired in as its own ``standin``
#: control target, with its own physics state and its own history seeded to
#: match — and a deployment that records its own store beside one records the
#: stand-in: the machine whose present is sampled and the machine whose past was
#: seeded are then the same one. Nothing is relabelled ``live``; ``live`` always
#: means the facility's authored ``epics`` block. But ``osprey set
#: connector=epics`` still rewrites the rendered ``control_system.type`` to
#: something that does not match here, while the machine on the other end of the
#: Channel Access connection has not changed at all. Recording follows the
#: machine, not the spelling, so the second half of the gate asks
#: :func:`~osprey_connectors.standin.archive_belongs_to_standin` — whose past is
#: in this store — through
#: :func:`~osprey.services.archiver_recorder.config.read_recording_facts`, which
#: then derives the ``standin`` target's own endpoint through the same resolver
#: the roster's label uses. It is the one predicate the recorder's compose entry
#: and the deploy-time archive seed bind to as well, so a machine that is still
#: the stand-in cannot be quietly dropped out of its own archive.
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
        """Re-read who is on the other end, if the poll interval has elapsed.

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
            facts = read_recording_facts(self._config_path)
        except RecorderConfigError as exc:
            # Keep the last known answer. See the module docstring: this is
            # very likely a config write in progress, and the next poll will
            # read the finished file.
            logger.debug("enablement poll could not read the config, keeping last state: %s", exc)
            return self._recording

        # Two ways for the machine on the other end to be one whose past this
        # deployment authored, and neither implies the other: a deployment whose
        # own type is the virtual accelerator, and a deployment whose archive is
        # its stand-in's. See RECORDING_CONTROL_SYSTEM.
        should_record = facts.control_system_type == RECORDING_CONTROL_SYSTEM or facts.live_standin
        if should_record != self._recording or not self._announced:
            if should_record:
                logger.info(
                    "Recording %d channels every %ds into %s.%s "
                    "(control_system.type=%s, this archive is the stand-in's: %s).",
                    len(self._addresses),
                    self._settings.cadence_sec,
                    self._settings.database,
                    self._settings.collection,
                    facts.control_system_type or "<unset>",
                    facts.live_standin,
                )
            else:
                logger.info(
                    "Idle, writing nothing: control_system.type=%s is not %s, and this "
                    "deployment's archive is not its stand-in's — either no stand-in was "
                    "stood up, or `%s` does not list this recorder, or the `standin` "
                    "target's own gateways no longer select the stand-in. History is "
                    "recorded for a machine whose past this deployment authored, and "
                    "either of those two makes it one. Setting either in %s takes effect "
                    "within %ds — no restart needed; a stand-in is named by `%s`.",
                    facts.control_system_type or "<unset>",
                    RECORDING_CONTROL_SYSTEM,
                    DEPLOYED_SERVICES_KEY,
                    self._config_path,
                    self._settings.poll_sec,
                    LIVE_STANDIN_PORT_KEY,
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
