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

Waveform channels take their own path. A trace is not one number, so it is read
apart from the scalar channels and written into the waveform collection, one
document per channel per instant, and only on the tail grid: a trace is hundreds
of samples, and the tail cadence is the density the seeder stores them at.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from osprey_connectors.standin import DEPLOYED_SERVICES_KEY, LIVE_STANDIN_PORT_KEY
from osprey_connectors.types import TRANSPORT_IN_PROCESS, VIRTUAL_ACCELERATOR

from .config import ChannelSet, RecorderConfigError, RecorderSettings, read_recording_facts
from .store import ArchiveWriter

if TYPE_CHECKING:
    from osprey_connectors.control_system.base import ControlSystemConnector

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
#:
#: The type is necessary, not sufficient: the simulator served in process has
#: no recorder, because nothing on a network serves its channels to sample, so
#: the type's half of the gate also asks that the recorded section talks to a
#: network.
RECORDING_CONTROL_SYSTEM = VIRTUAL_ACCELERATOR

#: Reads a set of addresses and returns the ones that answered.
#: A waveform channel answers with an array.
ChannelReader = Callable[[Sequence[str]], Awaitable[Mapping[str, Any]]]


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
        # A plain sequence carries no value types, so all of it is scalar.
        if isinstance(addresses, ChannelSet):
            self._addresses = list(addresses.scalars)
            self._waveforms = list(addresses.waveforms)
        else:
            self._addresses = list(addresses)
            self._waveforms = []
        self._writer = writer
        self._config_path = Path(config_path)
        self._reader = reader if reader is not None else channel_access_reader(settings)
        self._recording = recording
        self._announced = False
        self._last_slot: int | None = None
        self._last_waveform_slot: int | None = None
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
        # its stand-in's. The first is recorded only when that type is served
        # over a network: the simulator served in process has nothing to sample.
        # The stand-in is always a soft IOC on the network. See
        # RECORDING_CONTROL_SYSTEM.
        should_record = (
            facts.control_system_type == RECORDING_CONTROL_SYSTEM
            and facts.transport != TRANSPORT_IN_PROCESS
        ) or facts.live_standin
        if should_record != self._recording or not self._announced:
            if should_record:
                logger.info(
                    "Recording %d channels every %ds and %d waveforms every %ds into %s.%s "
                    "(control_system.type=%s, this archive is the stand-in's: %s).",
                    len(self._addresses),
                    self._settings.cadence_sec,
                    len(self._waveforms),
                    self._settings.tail_cadence_sec,
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
        double-write the instant the previous run just covered. A slot on the
        tail grid also records the waveform channels.
        """
        if not self.refresh_enablement(now):
            return False

        slot = self.slot_for(now)
        wrote_waveforms = await self._record_waveforms(slot)
        if slot == self._last_slot:
            return wrote_waveforms
        return await self._record_scalars(slot) or wrote_waveforms

    async def _record_scalars(self, slot: int) -> bool:
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
        except Exception as exc:  # see below
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

    async def _record_waveforms(self, slot: int) -> bool:
        """Record the waveform channels, when ``slot`` is a tail-grid instant.

        Only the tail grid's own instants are recorded — the ones the seeder
        stores traces at — so a slot the loop missed is a gap rather than a
        trace stamped with an instant it was not read at.
        """
        if (
            not self._waveforms
            or slot % self._settings.tail_cadence_sec != 0
            or slot == self._last_waveform_slot
        ):
            return False
        self._last_waveform_slot = slot

        traces = await self._read_waveforms()
        timestamp = datetime.fromtimestamp(slot, tz=UTC)
        if not traces:
            logger.warning(
                "No waveform channel answered at %s; writing no traces for this instant.",
                timestamp.isoformat(),
            )
            return False
        try:
            await asyncio.to_thread(self._writer.write_waveforms, timestamp, traces)
        except Exception as exc:  # the same reasoning as a failed sample write
            logger.warning("Failed to store the traces at %s: %s", timestamp.isoformat(), exc)
            return False
        logger.debug("Stored %d traces at %s", len(traces), timestamp.isoformat())
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

    async def _read_waveforms(self) -> dict[str, list[float]]:
        raw = await self._reader(self._waveforms)
        traces: dict[str, list[float]] = {}
        for address, value in raw.items():
            try:
                trace = np.asarray(value, dtype=np.float64).ravel()
            except (TypeError, ValueError):
                trace = None
            if trace is None or trace.size == 0:
                # A waveform channel that answered with something that is not
                # a trace has no document to go into — said once per address.
                if address not in self._non_numeric:
                    self._non_numeric.add(address)
                    logger.warning(
                        "Waveform channel %s returned no trace (%r); it will not be recorded.",
                        address,
                        value,
                    )
                continue
            traces[address] = trace.tolist()
        return traces


async def _connect() -> ControlSystemConnector:
    """Build and connect the Channel Access connector the recorder samples through.

    Built by the connector factory as type ``epics`` with an empty settings
    block, rather than from the deployment's ``control_system`` section, the
    way the dispatch trigger source builds its own. The recorder's gateway is
    the one its compose entry hands it in ``EPICS_CA_*``: an in-network address
    for a co-deployed virtual accelerator, the operator's for an external IOC.
    A gateway block from the section is written for the agent's side of the
    network and would rewrite that environment with an address this container
    may not reach. With no gateway configured, ``connect()`` leaves the
    environment as it stands.
    """
    from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors

    register_builtin_connectors()  # idempotent; the factory resolves "epics" through it
    return await ConnectorFactory.create_control_system_connector({"type": "epics"})


def channel_access_reader(settings: RecorderSettings) -> ChannelReader:
    """The production reader: one long-lived control-system connector.

    The connector runs every Channel Access call on the process's one CA loop,
    and aioca caches its channels there across calls, so the connection cost is
    paid once and every later sample reuses it. aioca's reconnection is also
    what makes this service survive an IOC restart without any logic here: the
    channels go disconnected, those reads fail (contributing nothing), and they
    resume answering once the IOC is back. Each sample costs one round trip per
    channel; the connector fetches a channel's units and labels on its first
    read only.

    The connector is built on the first sample rather than at startup, so a
    recorder idling on a deployment it does not record never touches Channel
    Access at all.

    Every channel is read on its own and all of them under one deadline, the
    sample cadence, so one unreachable channel cannot cost the whole sample:
    what answered by then is kept, and the rest is dropped.
    """
    cadence = float(settings.cadence_sec)
    connector: ControlSystemConnector | None = None

    async def read(addresses: Sequence[str]) -> Mapping[str, Any]:
        nonlocal connector
        # Bounded by the sample cadence: a read still outstanding when the next
        # sample is due has already missed its slot, and waiting longer would
        # only push every later sample off the grid.
        deadline = asyncio.get_running_loop().time() + cadence
        if connector is None:
            try:
                connector = await asyncio.wait_for(_connect(), cadence)
            except TimeoutError:
                logger.warning(
                    "The Channel Access loop did not connect within %ss; trying again "
                    "at the next sample.",
                    cadence,
                )
                return {}
        reads = {
            address: asyncio.create_task(connector.read_channel(address, timeout=cadence))
            for address in addresses
        }
        if not reads:
            return {}
        try:
            remaining = max(deadline - asyncio.get_running_loop().time(), 0.0)
            await asyncio.wait(reads.values(), timeout=remaining)
        finally:
            # Cancelling a read cancels its call on the CA loop too. The channel
            # stays in aioca's cache, so it is still connecting for next time.
            for pending in reads.values():
                pending.cancel()
        values: dict[str, Any] = {}
        for address, done in reads.items():
            # Never answered, or answered with a failure: either way the
            # channel contributes nothing to this sample.
            if done.done() and not done.cancelled() and done.exception() is None:
                values[address] = done.result().value
        return values

    return read
