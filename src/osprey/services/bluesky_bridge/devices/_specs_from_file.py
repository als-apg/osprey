"""Read the queueserver worker's plan-device set from a YAML/JSON device file.

The worker runs in its own container and venv: it cannot import OSPREY's
virtual-accelerator manifest, the channel-finder database, or anything else
host-side to discover which channels are scan devices. So the device set
arrives as a file, mounted into the container and named by
``BLUESKY_DEVICES_FILE``, and is parsed here into the control-room-neutral
``SettableSpec``/``TriggerableSpec``/``ReadableSpec``/``DetectorSpec`` shapes
that ``devices/connector.py`` builds from.

The document has four device keys, all optional, and an optional ``schema``
line naming the document and its version::

    schema: osprey.facility.bluesky_devices/2
    settables:
      - name: SR:C01:QF:1          # device name; also the event-data column key
        setpoint: SR:C01:QF:1:SP
        readback: SR:C01:QF:1:RB   # optional; omitted => reads the setpoint PV
        settle_tolerance: 0.05     # optional; or {relative: 1.0e-4} of the demand
        value_type: float          # optional; the value type of the value under `name`
        unit: A                    # optional; the unit of the value under `name`
        setpoint_value_type: float # optional; the setpoint's value type, with a readback
        setpoint_unit: A           # optional; the setpoint's unit, with a readback
      - name: SR:C01:RF:MODE:SP
        setpoint: SR:C01:RF:MODE:SP
        value_type: enum
        options: [OFF, ON, AUTO]   # optional; an enum's words, in option-index order
      - name: SR:C01:QD:1
        setpoint: SR:C01:QD:1:SP
        motion_band: 0.011         # optional; the readback's simulated motion
      - name: ES1:STAGE:POS:SP     # a positioner: the move starts on a command
        setpoint: ES1:STAGE:POS:SP
        readback: ES1:STAGE:POS:RB
        started_by:                # optional; the command that starts the move
          command: ES1:STAGE:MOVE  # the channel a move writes 1 to
          status: ES1:STAGE:STATUS # the channel that reports the move
          idle: Done               # what the status reads while no move runs
          timeout_s: 10.0          # the longest one move may take
        stopped_by: ES1:STAGE:STOP # optional, beside started_by; written 1 to stop a move
    triggerables:
      - name: ES1:MCA01:ACQ        # device name; its status is recorded under it
        command: ES1:MCA01:ACQ     # the channel a trigger writes 1 to
        status: ES1:MCA01:STATUS   # the channel that reports the act
        idle: Idle                 # what the status reads while no act runs
        timeout_s: 5.0             # the longest one act may take
        value_type: enum           # optional; the status's value type
        options: [Idle, Busy]      # optional; the status's words
    readables:
      - name: SR:C01:BPM:1:X
        pv: SR:C01:BPM:1:X:RB
        unit: mm                   # optional
      - name: SR:INJ:SEPTUM:WAVEFORM
        pv: SR:INJ:SEPTUM:WAVEFORM
        value_type: waveform       # optional
        shape: [500]               # optional; the length of a 1-D array reading
    triggered_readables:
      - name: ES1-MCA01            # device name; its results are recorded under their pvs
        command: ES1:MCA01:ACQ     # the channel a trigger writes 1 to
        status: ES1:MCA01:STATUS   # the channel that reports the acquisition
        idle: Idle                 # what the status reads while none runs
        timeout_s: 5.0             # the longest one acquisition may take
        results:                   # the channels it records, each as a readable's
          - pv: ES1:MCA01:COUNTS
            unit: counts           # optional
          - pv: ES1:MCA01:SPECTRUM
            value_type: waveform   # optional
            shape: [1024]          # optional

``yaml.safe_load`` parses it, so a ``.json`` file is read by the same code
path. Nothing here splits a value on any character, which is the point of the
format: EPICS PV names may contain commas (16 of the bundled demo tree's BTS
quadrupoles do), and the env-var channel this replaced used commas as its
entry separator.

Parsing is **fail-soft**. A malformed entry is skipped with a warning rather
than raised on — one typo must not cost every other device in a 13k-entry file
its connection — and an unreadable, empty or structurally wrong file yields an
empty device set, which the caller reports as "this worker will expose no
plans". The *build* is where a bad file is refused loudly:
``validate_device_document`` returns the human-readable problem list it uses.

Which channels are scan devices, and which readback pairs with which setpoint,
is a projection of the facility's namespace. That same namespace is described
by the channel-finder database and the knowledge graph; the three are to be
kept in step, and unifying them is a later piece of work.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from osprey_connectors.command import CommandAct

from .specs import DetectorSpec, DeviceSpecs, ReadableSpec, SettableSpec, TriggerableSpec

logger = logging.getLogger("osprey.services.bluesky_bridge.devices._specs_from_file")

SETTABLES_KEY = "settables"
"""Top-level key holding the settable (setpoint/readback) device entries."""

READABLES_KEY = "readables"
"""Top-level key holding the read-only device entries."""

TRIGGERABLES_KEY = "triggerables"
"""Top-level key holding the entries of commands a plan triggers."""

TRIGGERED_READABLES_KEY = "triggered_readables"
"""Top-level key holding the entries of detectors a plan triggers and records."""

RESULTS_KEY = "results"
"""Required detector key: the channels the detector records, a non-empty list of
``{pv, value_type?, unit?, shape?}`` entries, each read as a readable's."""

SCHEMA_KEY = "schema"
"""Top-level key naming the document and its version; the parser does not read it."""

TOP_LEVEL_KEYS = (SETTABLES_KEY, TRIGGERABLES_KEY, READABLES_KEY, TRIGGERED_READABLES_KEY)
"""The device keys the document may carry, in the order names are claimed."""

ACCEPTED_KEYS = (SCHEMA_KEY, *TOP_LEVEL_KEYS)
"""The only keys the document may carry; anything else rejects the document."""

UNIT_KEY = "unit"
"""Optional settable and readable key: the unit of the value a device reports
under its own name, a non-empty string. A device without one declares no unit."""
SHAPE_KEY = "shape"
"""Optional readable key: the length of a 1-D array reading, a list of one int
>= 1. A readable without one reads a scalar."""
SETPOINT_UNIT_KEY = "setpoint_unit"
"""Optional settable key: the unit of the setpoint channel of a device with a
separate readback, a non-empty string."""
STOPPED_BY_KEY = "stopped_by"
"""Optional settable key, only beside ``started_by``: the command that stops the
setpoint's move, a non-empty address. A run that stops the move while it is
unfinished writes it."""
VALUE_TYPE_KEY = "value_type"
"""Optional key of every entry: the facility value type of the value a device
reports under its own name, one of :data:`VALUE_TYPES`. A device without one
reports a float."""
SETPOINT_VALUE_TYPE_KEY = "setpoint_value_type"
"""Optional settable key: the facility value type of the setpoint channel of a
device with a separate readback, one of :data:`VALUE_TYPES`."""
OPTIONS_KEY = "options"
"""Optional key of every entry, beside ``value_type``: the words of an enum
channel, in option-index order, a non-empty list of non-empty strings. A run
records the channel as its word."""
SETPOINT_OPTIONS_KEY = "setpoint_options"
"""Optional settable key, beside ``setpoint_value_type``: the words of an enum
setpoint channel, as ``options``."""
VALUE_TYPES = ("float", "int", "bool", "enum", "string", "waveform")
"""The facility value types a channel carries."""
_SETTABLE_REQUIRED = ("name", "setpoint")
_SETTABLE_OPTIONAL = ("readback", UNIT_KEY, SETPOINT_UNIT_KEY, STOPPED_BY_KEY)
_SETTABLE_VALUE_TYPES = (VALUE_TYPE_KEY, SETPOINT_VALUE_TYPE_KEY)
_SETTABLE_OPTIONS = (OPTIONS_KEY, SETPOINT_OPTIONS_KEY)
SETTLE_TOLERANCE_KEY = "settle_tolerance"
"""Optional settable key: how far the readback may sit from the demand once settled.

A number >= 0 in the channel's unit, or ``{relative: <number >= 0>}``, the bound
as a fraction of the demand. A device with one settles within the larger of it
and the profile's ``bluesky.settle_tolerance`` on every lane."""
MOTION_BAND_KEY = "motion_band"
"""Optional settable key: how far the readback's simulated motion carries it.

A number >= 0 in the channel's unit, written for a device that declares no
tolerance. Only a simulated lane settles within it; any other lane settles
within the profile's floor."""
RELATIVE_KEY = "relative"
"""The one key of a relative ``settle_tolerance``."""
_SETTABLE_NUMERIC_OPTIONAL = (MOTION_BAND_KEY,)
_SETTABLE_TOLERANCE_OPTIONAL = (SETTLE_TOLERANCE_KEY,)
_READABLE_REQUIRED = ("name", "pv")
_READABLE_OPTIONAL = (UNIT_KEY,)
_READABLE_SHAPE_OPTIONAL = (SHAPE_KEY,)
_ONE_VALUE_TYPE = (VALUE_TYPE_KEY,)
_ONE_OPTIONS = (OPTIONS_KEY,)
IDLE_KEY = "idle"
"""Required triggerable key: what the status reads while no act runs, a
non-null scalar (text, a number or a bool)."""
TIMEOUT_KEY = "timeout_s"
"""Required triggerable key: the longest one act may take, a finite number of
seconds > 0."""
_TRIGGERABLE_REQUIRED = ("name", "command", "status")
_TRIGGERABLE_SCALARS = (IDLE_KEY,)
_TRIGGERABLE_POSITIVES = (TIMEOUT_KEY,)
STARTED_BY_KEY = "started_by"
"""Optional settable key: the command that starts the setpoint's move.

A mapping of exactly ``command`` and ``status`` (non-empty addresses), ``idle``
(a non-null scalar) and ``timeout_s`` (a finite number of seconds > 0): the
command a move writes, the status that reports the move, what the status reads
while no move runs and the longest one move may take."""
COMMAND_KEY = "command"
"""A start act's key: the command a move writes."""
STATUS_KEY = "status"
"""A start act's key: the status that reports the move."""
_START_ACT_KEYS = (COMMAND_KEY, STATUS_KEY, IDLE_KEY, TIMEOUT_KEY)
_SETTABLE_ACT_OPTIONAL = (STARTED_BY_KEY,)
_STOP_WITHOUT_START = f"{STOPPED_BY_KEY!r} is present without {STARTED_BY_KEY!r}"
_START_ACT_TEXT = (
    "'command' and 'status' (non-empty strings), 'idle' (a non-null scalar) "
    "and 'timeout_s' (a finite number > 0)"
)


def _clean(value: Any) -> str | None:
    """Return ``value`` as a stripped non-empty string, or ``None`` if it is not
    one (wrong type, empty, or whitespace only).

    Names are stripped because a name with trailing whitespace becomes a
    subtly wrong event-data column key rather than an obvious error.
    """
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _optional_strings(
    entry: dict[str, Any], keys: tuple[str, ...], where: str, name: str
) -> dict[str, str | None] | None:
    """Each optional string key of ``entry``, or ``None`` when one is malformed.

    Absent or null leaves the key ``None``; present but not a non-empty string
    is a malformed entry, warned about by name.
    """
    values: dict[str, str | None] = {}
    for key in keys:
        values[key] = None
        if entry.get(key) is None:
            continue
        values[key] = _clean(entry[key])
        if values[key] is None:
            logger.warning(
                "skipping %s (%r): %r is present but not a non-empty string", where, name, key
            )
            return None
    return values


def _value_types(
    entry: dict[str, Any], keys: tuple[str, ...], where: str, name: str
) -> dict[str, str | None] | None:
    """Each optional value-type key of ``entry``, or ``None`` when one is malformed.

    Absent or null leaves the key ``None``; present but not one of
    :data:`VALUE_TYPES` is a malformed entry, warned about by name.
    """
    values: dict[str, str | None] = {}
    for key in keys:
        values[key] = entry.get(key)
        if values[key] is not None and values[key] not in VALUE_TYPES:
            logger.warning(
                "skipping %s (%r): %r is present but not one of %r",
                where,
                name,
                key,
                list(VALUE_TYPES),
            )
            return None
    return values


def _words(value: Any) -> tuple[str, ...] | None:
    """Return ``options`` as a tuple, or ``None`` if it is not a non-empty list of
    non-empty strings."""
    if not isinstance(value, list) or not value:
        return None
    words = tuple(_clean(word) for word in value)
    return None if None in words else words  # type: ignore[return-value]


def _options(
    entry: dict[str, Any], keys: tuple[str, ...], where: str, name: str
) -> dict[str, tuple[str, ...] | None] | None:
    """Each optional words key of ``entry``, or ``None`` when one is malformed.

    Absent or null leaves the key ``None``; present but not a non-empty list of
    non-empty strings is a malformed entry, warned about by name.
    """
    values: dict[str, tuple[str, ...] | None] = {}
    for key in keys:
        values[key] = None
        if entry.get(key) is None:
            continue
        values[key] = _words(entry[key])
        if values[key] is None:
            logger.warning(
                "skipping %s (%r): %r is present but not a non-empty list of non-empty strings",
                where,
                name,
                key,
            )
            return None
    return values


def _tolerance(value: Any) -> float | None:
    """Return ``value`` as a float >= 0, or ``None`` if it is not one.

    A bool is refused although Python counts it as an int: ``true`` in a
    device file is a typo, never a tolerance of 1. NaN compares false and is
    refused with the negatives.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if number >= 0 else None


def _shape(value: Any) -> tuple[int] | None:
    """Return a ``shape`` as a one-element tuple, or ``None`` if it is not a list
    of one int >= 1.

    A bool is refused as in ``_tolerance``; a shape of more than one dimension
    is no plan device, so it is malformed here.
    """
    if not isinstance(value, list) or len(value) != 1:
        return None
    (length,) = value
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        return None
    return (length,)


def _scalar(value: Any) -> str | int | float | bool | None:
    """Return ``value`` when it is a non-null scalar, else ``None``."""
    return value if isinstance(value, str | int | float | bool) else None


def _positive_seconds(value: Any) -> float | None:
    """Return ``value`` as a finite float > 0, or ``None`` if it is not one.

    A bool is refused as in ``_tolerance``.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def _start_act(value: Any) -> CommandAct | None:
    """Return a ``started_by`` block as a :class:`CommandAct`, or ``None`` if malformed.

    The block is a mapping of exactly the four act keys, its addresses
    non-empty strings, its ``idle`` a non-null scalar and its ``timeout_s`` a
    finite number > 0.
    """
    if not isinstance(value, dict) or set(value) != set(_START_ACT_KEYS):
        return None
    command = _clean(value[COMMAND_KEY])
    status = _clean(value[STATUS_KEY])
    idle = _scalar(value[IDLE_KEY])
    timeout_s = _positive_seconds(value[TIMEOUT_KEY])
    if command is None or status is None or idle is None or timeout_s is None:
        return None
    return CommandAct(command, status, idle, timeout_s)


def _settle(value: Any) -> tuple[float | None, float | None] | None:
    """Return a ``settle_tolerance`` as ``(absolute, relative)``, or ``None`` if malformed.

    A number >= 0 is absolute; a mapping holding exactly ``relative``, a number
    >= 0, is relative. Anything else is malformed.
    """
    if isinstance(value, dict):
        if set(value) != {RELATIVE_KEY}:
            return None
        relative = _tolerance(value[RELATIVE_KEY])
        return None if relative is None else (None, relative)
    absolute = _tolerance(value)
    return None if absolute is None else (absolute, None)


def _load_document(path: Path) -> Any:
    """Parse the file at ``path``, returning ``None`` when it cannot be used.

    ``UnicodeDecodeError`` joins the caught errors because a binary file
    handed to the loader is a bad *file*, not a bug to crash the worker over.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("device file %s could not be read (%s); no devices built", path, exc)
        return None
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        logger.warning("device file %s is not valid YAML/JSON (%s); no devices built", path, exc)
        return None


def _entries(section: Any, key: str) -> list[Any]:
    """Return the entry list under ``key``, warning and yielding none when the
    section is present but is not a list."""
    if section is None:
        return []
    if not isinstance(section, list):
        logger.warning(
            "%r must be a list of device entries, got %s; skipping the section",
            key,
            type(section).__name__,
        )
        return []
    return section


def _parse_settables(section: Any) -> list[SettableSpec]:
    """Parse the ``settables`` section, skipping malformed entries with a warning.

    An entry needs a non-empty ``name`` and ``setpoint``. ``readback`` is
    optional: absent or explicitly null means the device reads its setpoint PV;
    present but not a non-empty string is a malformed entry, because silently
    reading the setpoint would hide a typo'd readback address.
    ``settle_tolerance`` and ``motion_band`` are optional too: absent or null
    leaves the device without that bound; a ``settle_tolerance`` that is not a
    number >= 0 or ``{relative: <number >= 0>}``, or a ``motion_band`` that is
    not a number >= 0, is a malformed entry. ``unit`` and ``setpoint_unit`` are
    optional strings: absent or null declares no unit; present but not a
    non-empty string is a malformed entry. ``value_type`` and
    ``setpoint_value_type`` are optional: absent or null states a float;
    present but not one of :data:`VALUE_TYPES` is a malformed entry.
    ``options`` and ``setpoint_options`` are optional: absent or null lists no
    words; present but not a non-empty list of non-empty strings is a
    malformed entry. ``started_by`` is optional: absent or null leaves the
    setpoint moving on its own write; present but not a well-formed act (see
    ``_start_act``) is a malformed entry. ``stopped_by`` is an optional string, and an entry that
    states it without ``started_by`` is malformed.
    """
    specs: list[SettableSpec] = []
    for index, entry in enumerate(_entries(section, SETTABLES_KEY)):
        where = f"{SETTABLES_KEY}[{index}]"
        if not isinstance(entry, dict):
            logger.warning("skipping %s: entry is not a mapping (%r)", where, entry)
            continue

        name = _clean(entry.get("name"))
        setpoint_pv = _clean(entry.get("setpoint"))
        if name is None or setpoint_pv is None:
            logger.warning(
                "skipping malformed %s (%r): needs a non-empty 'name' and 'setpoint'",
                where,
                entry.get("name"),
            )
            continue

        readback_pv: str | None = None
        if entry.get("readback") is not None:
            readback_pv = _clean(entry["readback"])
            if readback_pv is None:
                logger.warning(
                    "skipping %s (%r): 'readback' is present but not a non-empty string",
                    where,
                    name,
                )
                continue

        settle: tuple[float | None, float | None] | None = (None, None)
        if entry.get(SETTLE_TOLERANCE_KEY) is not None:
            settle = _settle(entry[SETTLE_TOLERANCE_KEY])
        if settle is None:
            logger.warning(
                "skipping %s (%r): %r is present but not a number >= 0 or {relative: <number>}",
                where,
                name,
                SETTLE_TOLERANCE_KEY,
            )
            continue

        band: float | None = None
        if entry.get(MOTION_BAND_KEY) is not None:
            band = _tolerance(entry[MOTION_BAND_KEY])
            if band is None:
                logger.warning(
                    "skipping %s (%r): %r is present but not a number >= 0",
                    where,
                    name,
                    MOTION_BAND_KEY,
                )
                continue

        units = _optional_strings(entry, (UNIT_KEY, SETPOINT_UNIT_KEY, STOPPED_BY_KEY), where, name)
        if units is None:
            continue
        kinds = _value_types(entry, _SETTABLE_VALUE_TYPES, where, name)
        if kinds is None:
            continue
        words = _options(entry, _SETTABLE_OPTIONS, where, name)
        if words is None:
            continue

        started_by: CommandAct | None = None
        if entry.get(STARTED_BY_KEY) is not None:
            started_by = _start_act(entry[STARTED_BY_KEY])
            if started_by is None:
                logger.warning(
                    "skipping %s (%r): %r is present but not a mapping of %s",
                    where,
                    name,
                    STARTED_BY_KEY,
                    _START_ACT_TEXT,
                )
                continue
        if units[STOPPED_BY_KEY] is not None and started_by is None:
            logger.warning("skipping %s (%r): %s", where, name, _STOP_WITHOUT_START)
            continue

        specs.append(
            SettableSpec(
                name=name,
                setpoint_pv=setpoint_pv,
                readback_pv=readback_pv,
                settle_tolerance=settle[0],
                settle_relative=settle[1],
                motion_band=band,
                unit=units[UNIT_KEY],
                setpoint_unit=units[SETPOINT_UNIT_KEY],
                started_by=started_by,
                stopped_by=units[STOPPED_BY_KEY],
                value_type=kinds[VALUE_TYPE_KEY],
                setpoint_value_type=kinds[SETPOINT_VALUE_TYPE_KEY],
                options=words[OPTIONS_KEY],
                setpoint_options=words[SETPOINT_OPTIONS_KEY],
            )
        )
    return specs


def _readable(entry: dict[str, Any], where: str, name: str, read_pv: str) -> ReadableSpec | None:
    """The readable ``entry`` states under ``name``, or ``None`` when its kind is malformed.

    ``unit``, ``value_type`` and ``options`` are optional, as on a settable.
    ``shape`` is optional: absent or null reads a scalar; present but not a
    list of one int >= 1 is a malformed entry, warned about by name.
    """
    units = _optional_strings(entry, (UNIT_KEY,), where, name)
    if units is None:
        return None
    kinds = _value_types(entry, _ONE_VALUE_TYPE, where, name)
    if kinds is None:
        return None
    words = _options(entry, _ONE_OPTIONS, where, name)
    if words is None:
        return None
    shape: tuple[int] | None = None
    if entry.get(SHAPE_KEY) is not None:
        shape = _shape(entry[SHAPE_KEY])
        if shape is None:
            logger.warning(
                "skipping %s (%r): %r is present but not a list of one int >= 1",
                where,
                name,
                SHAPE_KEY,
            )
            return None
    return ReadableSpec(
        name=name,
        read_pv=read_pv,
        unit=units[UNIT_KEY],
        shape=shape,
        value_type=kinds[VALUE_TYPE_KEY],
        options=words[OPTIONS_KEY],
    )


def _parse_readables(section: Any) -> list[ReadableSpec]:
    """Parse the ``readables`` section, skipping malformed entries with a warning.

    An entry needs a non-empty ``name`` and ``pv``; its kind is read as
    ``_readable`` reads it. See ``_parse_settables``.
    """
    specs: list[ReadableSpec] = []
    for index, entry in enumerate(_entries(section, READABLES_KEY)):
        where = f"{READABLES_KEY}[{index}]"
        if not isinstance(entry, dict):
            logger.warning("skipping %s: entry is not a mapping (%r)", where, entry)
            continue

        name = _clean(entry.get("name"))
        read_pv = _clean(entry.get("pv"))
        if name is None or read_pv is None:
            logger.warning(
                "skipping malformed %s (%r): needs a non-empty 'name' and 'pv'",
                where,
                entry.get("name"),
            )
            continue

        spec = _readable(entry, where, name, read_pv)
        if spec is not None:
            specs.append(spec)
    return specs


def _act_entry(entry: dict[str, Any], where: str) -> tuple[str, CommandAct] | None:
    """The name and the act an entry states, or ``None`` when either is malformed.

    The entry needs a non-empty ``name``, ``command`` and ``status``, an ``idle``
    that is a non-null scalar and a ``timeout_s`` that is a finite number > 0;
    a malformed one is warned about by name.
    """
    name = _clean(entry.get("name"))
    command = _clean(entry.get(COMMAND_KEY))
    status = _clean(entry.get(STATUS_KEY))
    if name is None or command is None or status is None:
        logger.warning(
            "skipping malformed %s (%r): needs a non-empty 'name', 'command' and 'status'",
            where,
            entry.get("name"),
        )
        return None

    idle = _scalar(entry.get(IDLE_KEY))
    if idle is None:
        logger.warning("skipping %s (%r): %r must be a non-null scalar", where, name, IDLE_KEY)
        return None

    timeout_s = _positive_seconds(entry.get(TIMEOUT_KEY))
    if timeout_s is None:
        logger.warning("skipping %s (%r): %r must be a finite number > 0", where, name, TIMEOUT_KEY)
        return None
    return name, CommandAct(command, status, idle, timeout_s)


def _parse_triggerables(section: Any) -> list[TriggerableSpec]:
    """Parse the ``triggerables`` section, skipping malformed entries with a warning.

    An entry states its name and act as ``_act_entry`` reads them;
    ``value_type`` and ``options``, the status's, are optional, as on a
    settable. See ``_parse_settables``.
    """
    specs: list[TriggerableSpec] = []
    for index, entry in enumerate(_entries(section, TRIGGERABLES_KEY)):
        where = f"{TRIGGERABLES_KEY}[{index}]"
        if not isinstance(entry, dict):
            logger.warning("skipping %s: entry is not a mapping (%r)", where, entry)
            continue

        stated = _act_entry(entry, where)
        if stated is None:
            continue
        name, act = stated

        kinds = _value_types(entry, _ONE_VALUE_TYPE, where, name)
        if kinds is None:
            continue
        words = _options(entry, _ONE_OPTIONS, where, name)
        if words is None:
            continue

        specs.append(
            TriggerableSpec(
                name=name,
                command=act.command,
                status=act.status,
                idle=act.idle,
                timeout_s=act.timeout_s,
                value_type=kinds[VALUE_TYPE_KEY],
                options=words[OPTIONS_KEY],
            )
        )
    return specs


def _results(value: Any, where: str, name: str) -> tuple[ReadableSpec, ...] | None:
    """A detector's ``results`` as readable specs, or ``None`` when they are malformed.

    ``results`` is a non-empty list of mappings, each with a non-empty ``pv``
    and its kind read as ``_readable`` reads it; the result is named by its
    address. One malformed result makes the whole list malformed.
    """
    if not isinstance(value, list) or not value:
        logger.warning(
            "skipping %s (%r): %r must be a non-empty list of result entries",
            where,
            name,
            RESULTS_KEY,
        )
        return None
    results: list[ReadableSpec] = []
    for index, entry in enumerate(value):
        place = f"{where}.{RESULTS_KEY}[{index}]"
        read_pv = _clean(entry.get("pv")) if isinstance(entry, dict) else None
        if read_pv is None:
            logger.warning("skipping %s (%r): %s needs a non-empty 'pv'", where, name, place)
            return None
        result = _readable(entry, where, name, read_pv)
        if result is None:
            return None
        results.append(replace(result, name=read_pv))
    return tuple(results)


def _parse_detectors(section: Any) -> list[DetectorSpec]:
    """Parse the ``triggered_readables`` section, skipping malformed entries with a warning.

    An entry states its name and act as a triggerable does (``_act_entry``)
    and its ``results`` as ``_results`` reads them; a malformed act or result,
    or an entry with no result, skips the detector. See ``_parse_settables``.
    """
    specs: list[DetectorSpec] = []
    for index, entry in enumerate(_entries(section, TRIGGERED_READABLES_KEY)):
        where = f"{TRIGGERED_READABLES_KEY}[{index}]"
        if not isinstance(entry, dict):
            logger.warning("skipping %s: entry is not a mapping (%r)", where, entry)
            continue

        stated = _act_entry(entry, where)
        if stated is None:
            continue
        name, act = stated

        results = _results(entry.get(RESULTS_KEY), where, name)
        if results is None:
            continue

        specs.append(
            DetectorSpec(
                name=name,
                command=act.command,
                status=act.status,
                idle=act.idle,
                timeout_s=act.timeout_s,
                results=results,
            )
        )
    return specs


def _drop_duplicate_names(
    setpoints: list[SettableSpec],
    triggerables: list[TriggerableSpec],
    readbacks: list[ReadableSpec],
    detectors: list[DetectorSpec],
) -> DeviceSpecs:
    """Drop any spec whose device name was already seen (setpoints first, then
    triggerables, readbacks and detectors), warning on each collision.

    Device names become ophyd-async device names *and* event-data column keys;
    two devices sharing a name would make the scanned column ambiguous (see the
    bridge's device-column lookup), so a later entry that reuses an
    already-claimed name is dropped rather than silently shadowing the first.
    """
    seen: set[str] = set()

    def _keep(specs: list) -> list:
        kept = []
        for spec in specs:
            if spec.name in seen:
                logger.warning(
                    "skipping device %r: name already claimed by an earlier device entry",
                    spec.name,
                )
                continue
            seen.add(spec.name)
            kept.append(spec)
        return kept

    settables = _keep(setpoints)
    kept_triggerables = _keep(triggerables)
    kept_readables = _keep(readbacks)
    return DeviceSpecs(settables, kept_readables, kept_triggerables, _keep(detectors))


def specs_from_file(path: str | Path) -> DeviceSpecs:
    """Read the device file at ``path`` into settable, triggerable, readable and detector specs.

    Returns no devices — never raises — when the file is unreadable, empty,
    unparseable, not a mapping, or carries an unknown top-level key. The last
    two reject the *whole* document with a single warning rather than loading
    the half of it that parsed: a typo'd ``readable:`` key would otherwise ship
    a deployment that looks healthy while quietly exposing no readbacks.
    Individual malformed entries are skipped with their own warning, and any
    device name that collides with an earlier one is dropped
    (``_drop_duplicate_names``, settables resolving first, then triggerables,
    readables and detectors).
    """
    nothing = DeviceSpecs([], [], [], [])
    doc = _load_document(Path(path))
    if doc is None:
        return nothing

    if not isinstance(doc, dict):
        logger.warning(
            "device file %s must hold a mapping with %r keys, got %s; no devices built",
            path,
            list(TOP_LEVEL_KEYS),
            type(doc).__name__,
        )
        return nothing

    unknown = [key for key in doc if key not in ACCEPTED_KEYS]
    if unknown:
        logger.warning(
            "device file %s has unknown top-level key(s) %r (expected only %r); no devices built",
            path,
            unknown,
            list(ACCEPTED_KEYS),
        )
        return nothing

    return _drop_duplicate_names(
        _parse_settables(doc.get(SETTABLES_KEY)),
        _parse_triggerables(doc.get(TRIGGERABLES_KEY)),
        _parse_readables(doc.get(READABLES_KEY)),
        _parse_detectors(doc.get(TRIGGERED_READABLES_KEY)),
    )


def _validate_section(
    section: Any,
    key: str,
    required: tuple[str, ...],
    optional: tuple[str, ...],
    numeric: tuple[str, ...] = (),
    tolerances: tuple[str, ...] = (),
    shapes: tuple[str, ...] = (),
    scalars: tuple[str, ...] = (),
    positives: tuple[str, ...] = (),
    acts: tuple[str, ...] = (),
    value_types: tuple[str, ...] = (),
    options: tuple[str, ...] = (),
    nested: tuple[str, ...] = (),
) -> list[str]:
    """Return the problems in one device section (empty list ⇒ none).

    ``optional`` keys are strings when present; ``numeric`` keys are numbers
    >= 0 when present; ``tolerances`` keys are numbers >= 0 or ``{relative:
    <number >= 0>}`` when present; ``shapes`` keys are lists of one int >= 1
    when present. ``scalars`` keys are required non-null scalars, and
    ``positives`` keys required finite numbers > 0. ``acts`` keys are start
    acts (see ``_start_act``) when present. ``value_types`` keys are one of
    :data:`VALUE_TYPES` when present, and ``options`` keys non-empty lists of
    non-empty strings. ``nested`` keys are known here and checked by the
    caller.
    """
    if section is None:
        return []
    if not isinstance(section, list):
        return [f"{key!r} must be a list of device entries, got {type(section).__name__}"]

    known = (
        set(required)
        | set(optional)
        | set(numeric)
        | set(tolerances)
        | set(shapes)
        | set(scalars)
        | set(positives)
        | set(acts)
        | set(value_types)
        | set(options)
        | set(nested)
    )
    problems: list[str] = []
    for index, entry in enumerate(section):
        where = f"{key}[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{where}: entry must be a mapping, got {type(entry).__name__}")
            continue
        problems.extend(
            f"{where}: {field!r} must be a non-empty string"
            for field in required
            if _clean(entry.get(field)) is None
        )
        problems.extend(
            f"{where}: {field!r} must be a non-empty string when present"
            for field in optional
            if entry.get(field) is not None and _clean(entry[field]) is None
        )
        problems.extend(
            f"{where}: {field!r} must be a number >= 0 when present"
            for field in numeric
            if entry.get(field) is not None and _tolerance(entry[field]) is None
        )
        problems.extend(
            f"{where}: {field!r} must be a number >= 0 or {{relative: <number >= 0>}} when present"
            for field in tolerances
            if entry.get(field) is not None and _settle(entry[field]) is None
        )
        problems.extend(
            f"{where}: {field!r} must be a list of one int >= 1 when present"
            for field in shapes
            if entry.get(field) is not None and _shape(entry[field]) is None
        )
        problems.extend(
            f"{where}: {field!r} must be a non-null scalar"
            for field in scalars
            if _scalar(entry.get(field)) is None
        )
        problems.extend(
            f"{where}: {field!r} must be a finite number > 0"
            for field in positives
            if _positive_seconds(entry.get(field)) is None
        )
        problems.extend(
            f"{where}: {field!r} must be a mapping of {_START_ACT_TEXT} when present"
            for field in acts
            if entry.get(field) is not None and _start_act(entry[field]) is None
        )
        problems.extend(
            f"{where}: {field!r} must be one of {list(VALUE_TYPES)!r} when present"
            for field in value_types
            if entry.get(field) is not None and entry[field] not in VALUE_TYPES
        )
        problems.extend(
            f"{where}: {field!r} must be a non-empty list of non-empty strings when present"
            for field in options
            if entry.get(field) is not None and _words(entry[field]) is None
        )
        unknown = [name for name in entry if name not in known]
        if unknown:
            problems.append(f"{where}: unknown key(s) {unknown!r}; expected only {sorted(known)!r}")
    return problems


def _stop_without_start_problems(section: Any) -> list[str]:
    """Return one problem per settable that states ``stopped_by`` without ``started_by``."""
    if not isinstance(section, list):
        return []
    return [
        f"{SETTABLES_KEY}[{index}]: {_STOP_WITHOUT_START}; only a positioner names a stop"
        for index, entry in enumerate(section)
        if isinstance(entry, dict)
        and entry.get(STOPPED_BY_KEY) is not None
        and entry.get(STARTED_BY_KEY) is None
    ]


def _detector_problems(section: Any) -> list[str]:
    """Return the problems in the ``triggered_readables`` section, its results included.

    An entry is checked as a triggerable is, and its ``results`` is a non-empty
    list of entries each checked as a readable's, with ``pv`` and no ``name``.
    """
    problems = _validate_section(
        section,
        TRIGGERED_READABLES_KEY,
        _TRIGGERABLE_REQUIRED,
        (),
        scalars=_TRIGGERABLE_SCALARS,
        positives=_TRIGGERABLE_POSITIVES,
        nested=(RESULTS_KEY,),
    )
    if not isinstance(section, list):
        return problems
    for index, entry in enumerate(section):
        if not isinstance(entry, dict):
            continue
        where = f"{TRIGGERED_READABLES_KEY}[{index}]"
        results = entry.get(RESULTS_KEY)
        if not isinstance(results, list) or not results:
            problems.append(f"{where}: {RESULTS_KEY!r} must be a non-empty list of result entries")
            continue
        problems += _validate_section(
            results,
            f"{where}.{RESULTS_KEY}",
            ("pv",),
            _READABLE_OPTIONAL,
            shapes=_READABLE_SHAPE_OPTIONAL,
            value_types=_ONE_VALUE_TYPE,
            options=_ONE_OPTIONS,
        )
    return problems


def _duplicate_name_problems(doc: dict) -> list[str]:
    """Return one problem per device name already claimed by an earlier entry.

    The loader drops these (see ``_drop_duplicate_names``), so a file the build
    accepted with duplicates would ship fewer devices than it lists.
    """
    seen: set[str] = set()
    problems: list[str] = []
    for key in TOP_LEVEL_KEYS:
        section = doc.get(key)
        if not isinstance(section, list):
            continue
        for index, entry in enumerate(section):
            if not isinstance(entry, dict):
                continue
            name = _clean(entry.get("name"))
            if name is None:
                continue
            if name in seen:
                problems.append(
                    f"{key}[{index}]: device name {name!r} is already claimed by an earlier "
                    "entry and would be dropped"
                )
            else:
                seen.add(name)
    return problems


def validate_device_document(doc: Any) -> list[str]:
    """Return the problems in a parsed device document; empty list ⇒ valid.

    This is the strict counterpart to ``specs_from_file``: the loader degrades
    quietly so a running worker survives a bad file, while the build calls this
    and refuses to stage a file that reports anything. It reports every problem
    it finds rather than the first, each naming its section and index so a
    13k-entry file can be repaired without bisecting it, and it never raises —
    the caller may hand it any object ``yaml.safe_load`` could return.

    An empty (``None``) or partial document is *valid*: a device file may
    legitimately list only readables, or nothing at all. Unknown keys are not,
    at either level, because they are how a typo presents itself.
    """
    if doc is None:
        return []
    if not isinstance(doc, dict):
        return [
            f"top-level document must be a mapping with {list(TOP_LEVEL_KEYS)!r} keys, "
            f"got {type(doc).__name__}"
        ]

    problems: list[str] = []
    unknown = [key for key in doc if key not in ACCEPTED_KEYS]
    if unknown:
        problems.append(
            f"unknown top-level key(s) {unknown!r}; expected only {list(ACCEPTED_KEYS)!r}"
        )
    problems += _validate_section(
        doc.get(SETTABLES_KEY),
        SETTABLES_KEY,
        _SETTABLE_REQUIRED,
        _SETTABLE_OPTIONAL,
        _SETTABLE_NUMERIC_OPTIONAL,
        _SETTABLE_TOLERANCE_OPTIONAL,
        acts=_SETTABLE_ACT_OPTIONAL,
        value_types=_SETTABLE_VALUE_TYPES,
        options=_SETTABLE_OPTIONS,
    )
    problems += _stop_without_start_problems(doc.get(SETTABLES_KEY))
    problems += _validate_section(
        doc.get(TRIGGERABLES_KEY),
        TRIGGERABLES_KEY,
        _TRIGGERABLE_REQUIRED,
        (),
        scalars=_TRIGGERABLE_SCALARS,
        positives=_TRIGGERABLE_POSITIVES,
        value_types=_ONE_VALUE_TYPE,
        options=_ONE_OPTIONS,
    )
    problems += _validate_section(
        doc.get(READABLES_KEY),
        READABLES_KEY,
        _READABLE_REQUIRED,
        _READABLE_OPTIONAL,
        shapes=_READABLE_SHAPE_OPTIONAL,
        value_types=_ONE_VALUE_TYPE,
        options=_ONE_OPTIONS,
    )
    problems += _detector_problems(doc.get(TRIGGERED_READABLES_KEY))
    problems += _duplicate_name_problems(doc)
    return problems
