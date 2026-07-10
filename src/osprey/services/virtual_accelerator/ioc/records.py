"""EPICS record factory for the PyAT virtual accelerator IOC.

Pure record-construction layer: consumes the namespace-union channel manifest
(``manifest.build_manifest()["channels"]``) and builds one pythonSoftIOC
record per channel, typed per the manifest's ``record_type``. This module
owns no physics and no simulation state -- it never imports PyAT or
SimulationEngine. Instead it exposes callback-shaped interfaces so the two
downstream tasks can plug in without this module knowing what they do:

  * partition (a) pyat-coupled setpoints call an injected
    ``on_pyat_setpoint`` hook on write, then echo the written value onto
    their own paired ``:RB`` -- a plain value copy, same as partition (b),
    so a magnet's own current readback tracks its own setpoint honestly
    even though nothing else in this module knows what the hook did with
    it. The injected hook (the physics bridge, ioc-physics-bridge) is
    responsible for computing the new orbit and pushing *other* records --
    the BPM POSITION readbacks -- back out via ``.set()`` on the records in
    the returned ``pyat_coupled`` dict; it never needs to touch the writing
    setpoint's own ``:RB``. The echo only happens once the hook returns, so
    a write the hook rejects (``OrbitSolveError``) never gets echoed.
  * partition (b) sp-echo setpoints are wired to their paired readback
    entirely inside this module -- "no physics" means the echo is a plain
    value copy (write SP, RB follows immediately), so there is nothing for
    another task to inject here.
  * partition (c) static-noisy channels are built as plain In-type records
    holding a type-appropriate default value; the returned ``static_noisy``
    dict is the "source hook" the engine-source task (ioc-engine-source)
    drives with its own periodic updater.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from softioc import builder

from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    PARTITION_SP_ECHO,
    PARTITION_STATIC_NOISY,
    RECORD_TYPE_ANALOG,
    RECORD_TYPE_BINARY,
    RECORD_TYPE_STRING,
)

SETPOINT_SUBFIELD = "SP"
READBACK_SUBFIELD = "RB"

# SR corrector current band, Amps -- matches the SR:MAG:{HCM,VCM}:*:CURRENT:SP
# channel_limits.json band. Set as DRVL/DRVH on the built aOut record so
# pythonSoftIOC clamps an out-of-range caput to this band at the record
# itself, before on_update (hence the physics hook and the RB echo) ever
# sees it -- a real second bound below the ORM plan's own pydantic schema,
# enforced against any writer, not only the plan.
CORRECTOR_FAMILIES = frozenset({"HCM", "VCM"})
CORRECTOR_DRIVE_LIMIT_A = 12.0

_IN_BUILDERS = {
    RECORD_TYPE_ANALOG: builder.aIn,
    RECORD_TYPE_BINARY: builder.boolIn,
    RECORD_TYPE_STRING: builder.stringIn,
}
_OUT_BUILDERS = {
    RECORD_TYPE_ANALOG: builder.aOut,
    RECORD_TYPE_BINARY: builder.boolOut,
    RECORD_TYPE_STRING: builder.stringOut,
}
_DEFAULT_VALUE = {
    RECORD_TYPE_ANALOG: 0.0,
    RECORD_TYPE_BINARY: False,
    RECORD_TYPE_STRING: "",
}


class ManifestContractError(ValueError):
    """A manifest entry violates a contract this factory (and its downstream
    consumers, the physics bridge and engine source) rely on."""


@dataclass
class IOCRecords:
    """Handle to a built record set.

    ``all`` is every record keyed by address -- used by whole-namespace
    consumers such as the full-sweep check. ``pyat_coupled`` and
    ``static_noisy`` are the callback-slot dicts the physics bridge and
    engine source plug into; see the module docstring.
    """

    all: dict[str, Any] = field(default_factory=dict)
    pyat_coupled: dict[str, Any] = field(default_factory=dict)
    static_noisy: dict[str, Any] = field(default_factory=dict)


def _channel_key(channel: dict) -> tuple[str, str, str, str, str]:
    """Identity of a channel's device+field, independent of subfield.

    Used to pair a partition (b) setpoint with its readback: two channels
    sharing this key and differing only in subfield (SP vs RB) are the two
    halves of one echo pair.
    """
    return (
        channel["ring"],
        channel["system"],
        channel["family"],
        channel["device"],
        channel["field"],
    )


def build_records(
    channels: list[dict],
    *,
    on_pyat_setpoint: Callable[[str, float], None] | None = None,
    stuck_setpoints: frozenset[str] = frozenset(),
) -> IOCRecords:
    """Build one softioc record per manifest channel.

    Args:
        channels: the ``channels`` list from the namespace-union manifest;
            each entry needs ``address``, ``ring``, ``system``, ``family``,
            ``device``, ``field``, ``subfield``, ``partition``,
            ``record_type``, and ``noise``.
        on_pyat_setpoint: callback invoked as ``on_pyat_setpoint(address,
            value)`` whenever a partition (a) setpoint is written, before
            that setpoint's own ``:RB`` is echoed (see the module
            docstring). The physics bridge supplies this and uses the
            ``pyat_coupled`` dict on the returned :class:`IOCRecords` to
            push recomputed BPM positions back out. If omitted (as in these
            unit tests), setpoint writes are accepted but produce no
            readback movement at all -- there is no hook to call and
            nothing to echo without one.
        stuck_setpoints: build-time, per-channel apply fault. A ``:SP``
            address in this set still latches the caput value onto its own
            ``aOut`` record, but its normal write-time behavior (the sp-echo
            copy into ``:RB``, or the pyat-coupled hook call plus ``:RB``
            echo) is replaced with a no-op -- so that device's readback
            simply never moves.
            This is a substrate-honesty fixture: any Channel Access client
            reading the frozen readback sees the identical stale value, not
            a per-client divergence. Empty by default (no fault).

    Returns:
        An :class:`IOCRecords` with every built record plus the two
        callback-slot dicts.

    Raises:
        ManifestContractError: if a channel's record_type/noise/partition
            combination violates a contract this factory relies on (e.g. a
            binary channel flagged noisy, or an sp-echo setpoint with no
            matching readback channel).
    """
    records = IOCRecords()

    # Pass 1: build every non-setpoint (In-type) record first, so partition
    # (b) setpoints built in pass 2 can look up and echo into their
    # already-constructed readback companion.
    readback_index: dict[tuple[str, str, str, str, str], Any] = {}
    setpoint_channels: list[dict] = []

    for channel in channels:
        record_type = channel["record_type"]
        if record_type not in _IN_BUILDERS:
            raise ManifestContractError(
                f"unknown record_type {record_type!r} for {channel['address']!r}"
            )
        if record_type == RECORD_TYPE_BINARY and channel["noise"]:
            raise ManifestContractError(
                f"binary channel {channel['address']!r} is flagged noisy -- bi records reject noise"
            )

        if channel["subfield"] == SETPOINT_SUBFIELD:
            setpoint_channels.append(channel)
            continue

        rec = _IN_BUILDERS[record_type](
            channel["address"], initial_value=_DEFAULT_VALUE[record_type]
        )
        records.all[channel["address"]] = rec

        if channel["subfield"] == READBACK_SUBFIELD:
            readback_index[_channel_key(channel)] = rec

        if channel["partition"] == PARTITION_PYAT_COUPLED:
            records.pyat_coupled[channel["address"]] = rec
        elif channel["partition"] == PARTITION_STATIC_NOISY:
            records.static_noisy[channel["address"]] = rec

    # Pass 2: build setpoints, wiring each to its partition's write behavior.
    for channel in setpoint_channels:
        record_type = channel["record_type"]
        partition = channel["partition"]
        address = channel["address"]

        if partition == PARTITION_SP_ECHO:
            rb = readback_index.get(_channel_key(channel))
            if rb is None:
                raise ManifestContractError(
                    f"sp-echo setpoint {address!r} has no matching RB readback channel"
                )
            on_update = rb.set
        elif partition == PARTITION_PYAT_COUPLED:
            rb = readback_index.get(_channel_key(channel))
            if on_pyat_setpoint is None:
                on_update = lambda value: None  # noqa: E731
            elif rb is None:
                # No paired RB in this channel set (e.g. a synthetic
                # single-channel test) -- still call the hook, just nothing
                # to echo into.
                on_update = lambda value, addr=address, hook=on_pyat_setpoint: hook(addr, value)  # noqa: E731
            else:

                def on_update(value, addr=address, hook=on_pyat_setpoint, rb=rb):
                    # Hook first: it re-solves the orbit and rolls the
                    # lattice element back on OrbitSolveError, so :RB must
                    # only ever echo an accepted setpoint, never a rejected
                    # transient one. This is the scan-hang fix -- without
                    # this echo a corrector's own CURRENT:RB stays 0.0
                    # forever, since the physics bridge only pushes BPM
                    # POSITION readbacks, never a magnet's own :RB.
                    hook(addr, value)
                    rb.set(value)
        else:
            raise ManifestContractError(
                f"unexpected partition {partition!r} for setpoint channel {address!r}"
            )

        if address in stuck_setpoints:
            # Apply fault: the SP still latches its own written value below,
            # but whatever would normally propagate that write outward (the
            # sp-echo copy, or the pyat-coupled hook call) is dropped -- the
            # readback freezes at its last value, honestly, for every reader.
            on_update = lambda value: None  # noqa: E731

        drive_limit_kwargs: dict[str, float] = {}
        if partition == PARTITION_PYAT_COUPLED and channel["family"] in CORRECTOR_FAMILIES:
            drive_limit_kwargs = {
                "DRVL": -CORRECTOR_DRIVE_LIMIT_A,
                "DRVH": CORRECTOR_DRIVE_LIMIT_A,
            }

        rec = _OUT_BUILDERS[record_type](
            address,
            initial_value=_DEFAULT_VALUE[record_type],
            on_update=on_update,
            **drive_limit_kwargs,
        )
        records.all[address] = rec
        if partition == PARTITION_PYAT_COUPLED:
            records.pyat_coupled[address] = rec

    return records
