"""Manifest -> co-hosted process-variable database, plus the record shims.

The serving layer hands a Channel Access server one *process-variable
database* (``pvdb``): a plain ``{name: spec}`` dict describing every PV's
type, element count, initial value and display limits. This module is the
only place that translates the namespace-union channel manifest
(``manifest.build_manifest()["channels"]``) into that shape. It owns no
physics, opens no socket, and imports neither the CA server library nor the
LUME serving package -- it produces data plus a thin per-channel shim, and
the runner injects both into the server it hosts.

Two consumers already exist and are deliberately not changed by this module:
``ioc.engine_source.EngineSource`` (1 Hz telemetry) and
``ioc.physics_bridge.PhysicsBridge`` (BPM readbacks after a solve). Both
drive their records through exactly two calls -- ``record.set(value)`` and
``record.get()`` -- so :class:`PVRecord` reproduces that surface and they
plug in unchanged.

Three properties of the served database are contracts, not preferences:

* **The channel set is closed.** One PV per manifest channel, no more, no
  fewer: the pinned counts (348 SR magnet ``:SP`` / 348 paired ``:RB`` /
  144 BPM readings / 2,908 channels total) are what clients, the channel
  finder databases and the safety limits file all agree on. This module
  adds no diagnostic or control PV of its own.
* **Drive limits are not alarm limits.** ``drive_limits`` becomes
  ``lolim``/``hilim`` (the CA *control/display* band, what DRVL/DRVH mean
  to a client). The alarm-limit keys (``lolo``/``low``/``high``/``hihi``)
  are never emitted, so a value clamped to the top of its band reads back
  ``NO_ALARM`` rather than a limit violation -- clamping is a normal,
  non-alarming outcome. Leaving those four keys unset is exactly what
  disables the server's alarm evaluation: it only compares against them
  when ``lolo < hihi`` (resp. ``low < high``), which the shared default of
  0.0 never satisfies.
* **No PV uses the server's own ``scan`` field.** A scanning PV is polled
  by the server and is *skipped* by the driver-side monitor post, so a
  served value pushed with :meth:`PVRecord.set` on a scanning PV would
  silently never reach a subscriber. Every value here is driver-posted;
  none is server-scanned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    PARTITION_SP_ECHO,
    PARTITION_STATIC_NOISY,
    RECORD_TYPE_ANALOG,
    RECORD_TYPE_BINARY,
    RECORD_TYPE_LONG_STRING,
    RECORD_TYPE_MBB,
    RECORD_TYPE_STRING,
)

SETPOINT_SUBFIELD = "SP"
READBACK_SUBFIELD = "RB"

# Gateway "long string" channels are 512-byte char waveforms. The width is
# declared explicitly (never derived from the boot value's length) so a wire
# value the real channel carries in full is never truncated by a short seed.
LONG_STRING_LENGTH = 512

# DBR_STRING's payload is 40 bytes, fixed by the protocol -- a `string` PV
# carries one such element. Named here because the record-type table is the
# documentation of record widths; nothing configures it.
STRING_LENGTH = 40

# An EPICS multi-bit-binary record has 16 discrete states. The manifest
# schema carries no state labels, so an mbb channel is served as a *numeric*
# enum: the label of state N is "N". Should the manifest ever grow labels,
# they replace `MBB_ENUM_STATES` and nothing else changes.
MBB_STATE_COUNT = 16
MBB_ENUM_STATES = tuple(str(state) for state in range(MBB_STATE_COUNT))

# The two states of a binary channel. These are flag channels (STATUS:VALID,
# :FAULT, :READY, :ON), whose value is a boolean, not a device power state.
BINARY_ENUM_STATES = ("FALSE", "TRUE")

# Display precision for analog channels. The server defaults this to 0, which
# makes every precision-aware client render an analog channel as an integer --
# unusable for BPM positions in metres (micron-scale) and visibly lossy for
# magnet currents. Six digits covers both. Display metadata only: the value a
# client reads is always the full double.
ANALOG_PRECISION = 6


def _analog_spec() -> dict[str, Any]:
    return {"type": "float", "count": 1, "prec": ANALOG_PRECISION}


def _binary_spec() -> dict[str, Any]:
    return {"type": "enum", "count": 1, "enums": list(BINARY_ENUM_STATES)}


def _mbb_spec() -> dict[str, Any]:
    return {"type": "enum", "count": 1, "enums": list(MBB_ENUM_STATES)}


def _string_spec() -> dict[str, Any]:
    return {"type": "string", "count": 1}


def _long_string_spec() -> dict[str, Any]:
    return {"type": "char", "count": LONG_STRING_LENGTH}


# Manifest record type -> a *fresh* PV spec. Factories rather than literals:
# every channel gets its own dict (and its own `enums` list), so seeding one
# channel's boot value or drive limits can never leak into another's.
_SPEC_FACTORIES = {
    RECORD_TYPE_ANALOG: _analog_spec,
    RECORD_TYPE_BINARY: _binary_spec,
    RECORD_TYPE_MBB: _mbb_spec,
    RECORD_TYPE_STRING: _string_spec,
    RECORD_TYPE_LONG_STRING: _long_string_spec,
}

# Boot value for a channel with no seed of its own, per record type.
_DEFAULT_VALUE: dict[str, Any] = {
    RECORD_TYPE_ANALOG: 0.0,
    RECORD_TYPE_BINARY: 0,
    RECORD_TYPE_MBB: 0,
    RECORD_TYPE_STRING: "",
    RECORD_TYPE_LONG_STRING: "",
}


def _coerce_analog(value: Any) -> float:
    return float(value)


def _coerce_discrete(value: Any) -> int:
    # bool is an int subclass, so an engine-supplied True/False lands as 1/0.
    return int(value)


def _coerce_text(value: Any) -> str:
    return str(value)


# Coercion applied to every value entering a PV -- boot seed and live `.set()`
# alike. The manifest's scenario data stores every channel as a float, even
# the ones served as discrete flags, and the CA server has no record layer to
# convert on its behalf: an unconverted float on an enum PV would be served as
# a garbage state. Coercing here keeps that conversion in one place instead of
# in each of the two value sources.
_COERCE = {
    RECORD_TYPE_ANALOG: _coerce_analog,
    RECORD_TYPE_BINARY: _coerce_discrete,
    RECORD_TYPE_MBB: _coerce_discrete,
    RECORD_TYPE_STRING: _coerce_text,
    RECORD_TYPE_LONG_STRING: _coerce_text,
}

# Record types whose values are discrete or textual: simulated measurement
# noise is meaningless for them, so a manifest flagging one noisy is a
# contract violation the telemetry source would otherwise act on.
_NOISE_REJECTING_TYPES = frozenset({RECORD_TYPE_BINARY, RECORD_TYPE_MBB, RECORD_TYPE_LONG_STRING})

# Alarm-limit keys this module must never emit; see the module docstring.
ALARM_LIMIT_KEYS = ("lolo", "low", "high", "hihi")


class ManifestContractError(ValueError):
    """A manifest entry violates a contract the serving layer relies on."""


class PVRecord:
    """One served channel, in the shape its value sources already speak.

    ``EngineSource`` and ``PhysicsBridge`` drive their channels with
    ``record.set(value)`` and ``record.get()``. This class is that surface
    over a CA driver: ``set`` commits the value into the driver's parameter
    store and posts a monitor event for this address alone; ``get`` reads
    the committed value back *through the driver*, so a setpoint a client
    wrote over CA is visible here too (a local cache would go stale the
    moment the write path bypassed it).

    A record is usable before a driver exists, which matters because the
    database has to be built -- and its initial values fixed -- before the
    server and driver are constructed. Until :meth:`attach` is called,
    ``set`` writes straight into the PV spec's boot value and ``get`` reads
    it, so values pushed during assembly (notably ``PhysicsBridge.bind()``'s
    first BPM push) become the values the server comes up with.
    """

    __slots__ = ("address", "_spec", "_coerce", "_driver")

    def __init__(self, address: str, spec: dict[str, Any], coerce: Any) -> None:
        self.address = address
        self._spec = spec
        self._coerce = coerce
        self._driver: Any = None

    def attach(self, driver: Any) -> None:
        """Bind this record to the live CA driver (see :meth:`ServingRecords.attach_driver`)."""
        self._driver = driver

    def set(self, value: Any) -> None:
        """Commit ``value`` and post a monitor event for this address.

        Posting is per-address (``updatePV``), not database-wide
        (``updatePVs``): the telemetry source sets on the order of a
        thousand records per poll tick, and a database-wide sweep on each of
        them would be quadratic in the channel count for no added
        delivery -- ``updatePV`` posts exactly the address whose value just
        changed. The driver itself suppresses the event when the value is
        unchanged, so an unmoving telemetry channel costs no CA traffic.
        """
        value = self._coerce(value)
        if self._driver is None:
            self._spec["value"] = value
            return
        self._driver.setParam(self.address, value)
        self._driver.updatePV(self.address)

    def get(self) -> Any:
        """Return the currently served value."""
        if self._driver is None:
            return self._spec["value"]
        return self._driver.getParam(self.address)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"<PVRecord {self.address!r} type={self._spec['type']!r}>"


@dataclass
class ServingRecords:
    """The built database plus the shims that drive it.

    ``pvdb`` is what the server is handed. The three record dicts mirror the
    partitions the value sources consume: ``pyat_coupled`` is what
    ``PhysicsBridge.bind()`` takes, ``static_noisy`` is what ``EngineSource``
    drives, and ``all`` is every record by address for whole-namespace
    consumers. ``setpoint_readbacks`` maps each writable ``:SP`` address to
    its paired ``:RB`` address, which is the echo the write path owes a
    client on an accepted write.
    """

    pvdb: dict[str, dict[str, Any]] = field(default_factory=dict)
    all: dict[str, PVRecord] = field(default_factory=dict)
    pyat_coupled: dict[str, PVRecord] = field(default_factory=dict)
    static_noisy: dict[str, PVRecord] = field(default_factory=dict)
    setpoint_readbacks: dict[str, str] = field(default_factory=dict)

    def attach_driver(self, driver: Any) -> None:
        """Point every record at the live driver.

        Called once, after the server has created the PVs and the driver
        exists. Before this, records serve and accept values through the PV
        spec; after it, through the driver.
        """
        for record in self.all.values():
            record.attach(driver)


def _channel_key(channel: dict) -> tuple[str, str, str, str, str]:
    """Identity of a channel's device+field, independent of subfield.

    Two channels sharing this key and differing only in subfield (SP vs RB)
    are the two halves of one setpoint/readback pair.
    """
    return (
        channel["ring"],
        channel["system"],
        channel["family"],
        channel["device"],
        channel["field"],
    )


def _initial_value(channel: dict, boot_values: dict[str, Any] | None) -> Any:
    """Boot value for one channel: the type default, unless this is a
    ``:SP``/``:RB`` channel with its own seed in ``boot_values``.

    Restricting the seed to setpoints and their readbacks is deliberate:
    those are the channels whose boot state is a machine state worth
    reproducing. Telemetry and status channels get their values from the
    engine source on the first poll tick instead.
    """
    record_type = channel["record_type"]
    default = _DEFAULT_VALUE[record_type]
    if boot_values is None or channel["subfield"] not in (SETPOINT_SUBFIELD, READBACK_SUBFIELD):
        return _COERCE[record_type](default)
    return _COERCE[record_type](boot_values.get(channel["address"], default))


def build_serving_pvdb(
    channels: list[dict],
    *,
    drive_limits: dict[str, tuple[float, float]] | None = None,
    boot_values: dict[str, Any] | None = None,
    async_setpoints: bool = False,
) -> ServingRecords:
    """Build the co-hosted PV database and its record shims from the manifest.

    Args:
        channels: the ``channels`` list of the namespace-union manifest;
            each entry needs ``address``, ``ring``, ``system``, ``family``,
            ``device``, ``field``, ``subfield``, ``partition``,
            ``record_type`` and ``noise``.
        drive_limits: optional ``{address: (low, high)}`` map, applied as the
            PV's ``lolim``/``hilim`` control band -- the DRVL/DRVH a client
            reads. Never alarm limits (see the module docstring). This module
            stays file-blind: deriving the map from ``channel_limits.json``
            is the caller's job. The server does not enforce the band on a
            write; the write path clamps to it.
        boot_values: optional ``{address: value}`` map giving boot-time
            values for ``:SP``/``:RB`` channels (e.g. derived from a
            scenario's ``machine.json`` by the caller -- this module never
            loads it). Any other channel, and any ``:SP``/``:RB`` with no
            entry, boots at its type default (0.0 / 0 / "").
        async_setpoints: declare every ``:SP`` PV as an asynchronous write
            (``asyn``). Off by default, because an asynchronous PV leaves the
            client blocked until the driver explicitly completes the write:
            only a write path that does complete it may turn this on.

    Returns:
        A :class:`ServingRecords` carrying the database, the per-partition
        record shims and the setpoint->readback pairing.

    Raises:
        ManifestContractError: a channel's record_type/noise combination, or
            the channel set's setpoint/readback pairing, violates a contract
            the serving layer relies on.
    """
    records = ServingRecords()
    readback_addresses: dict[tuple[str, str, str, str, str], str] = {}
    setpoint_channels: list[dict] = []

    for channel in channels:
        address = channel["address"]
        record_type = channel["record_type"]

        if record_type not in _SPEC_FACTORIES:
            raise ManifestContractError(f"unknown record_type {record_type!r} for {address!r}")
        if record_type in _NOISE_REJECTING_TYPES and channel["noise"]:
            raise ManifestContractError(
                f"channel {address!r} is flagged noisy -- {record_type} records reject noise"
            )
        if address in records.pvdb:
            raise ManifestContractError(f"duplicate channel address {address!r}")

        spec = _SPEC_FACTORIES[record_type]()
        spec["value"] = _initial_value(channel, boot_values)
        if drive_limits is not None and address in drive_limits:
            low, high = drive_limits[address]
            spec["lolim"] = float(low)
            spec["hilim"] = float(high)
        if async_setpoints and channel["subfield"] == SETPOINT_SUBFIELD:
            spec["asyn"] = True

        record = PVRecord(address, spec, _COERCE[record_type])
        records.pvdb[address] = spec
        records.all[address] = record

        partition = channel["partition"]
        if partition == PARTITION_PYAT_COUPLED:
            records.pyat_coupled[address] = record
        elif partition == PARTITION_STATIC_NOISY:
            records.static_noisy[address] = record

        if channel["subfield"] == READBACK_SUBFIELD:
            readback_addresses[_channel_key(channel)] = address
        elif channel["subfield"] == SETPOINT_SUBFIELD:
            setpoint_channels.append(channel)

    for channel in setpoint_channels:
        readback = readback_addresses.get(_channel_key(channel))
        if readback is None:
            if channel["partition"] == PARTITION_SP_ECHO:
                # An echo setpoint with nothing to echo into would accept
                # writes that no client could ever observe.
                raise ManifestContractError(
                    f"sp-echo setpoint {channel['address']!r} has no matching RB readback channel"
                )
            # A pyat-coupled setpoint may legitimately stand alone (a
            # synthetic single-channel set); its physics effect is observed
            # on the BPM readbacks, not on a readback of its own.
            continue
        records.setpoint_readbacks[channel["address"]] = readback

    return records


__all__ = [
    "ALARM_LIMIT_KEYS",
    "ANALOG_PRECISION",
    "BINARY_ENUM_STATES",
    "LONG_STRING_LENGTH",
    "MBB_ENUM_STATES",
    "MBB_STATE_COUNT",
    "READBACK_SUBFIELD",
    "SETPOINT_SUBFIELD",
    "STRING_LENGTH",
    "ManifestContractError",
    "PVRecord",
    "ServingRecords",
    "build_serving_pvdb",
]
