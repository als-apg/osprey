"""Virtual Accelerator entrypoint.

Assembles the whole virtual accelerator in one process, in dependency order:

    manifest -> serving database -> physics bridge (partition a)
             -> runner (Channel Access + PVA) -> engine source (partition c)

and then hands the calling thread to the runner, which blocks serving until
the process is signalled.

The order is a contract, not a convenience. The serving database is built
first and every value that must be on the wire at boot is pushed into it
*before* the runner exists, because the Channel Access server copies each
PV's spec when it creates the PV: a value written into a spec afterwards is
never served. That is why the physics bridge is bound here -- its first push
of BPM readings is the boot state -- and why nothing re-seeds the database
from type defaults after that point.

Both transports come from the same runner: the facility's whole channel
namespace is co-hosted on Channel Access, and the physics model's own
variables are served on PVA. Channel Access is the authoritative view of the
machine; see
:mod:`~osprey.services.virtual_accelerator.serving.runner` for why the two
are not synchronised.

Run contract (see docker/virtual-accelerator/README.md for the full version):

    -v <project>/data/simulation:/data/simulation             # the DIRECTORY, never a file
    -v <repo>/var/agent_data/simulation:/state/simulation:ro  # scenario state
    -e VA_CHANNELS_FILE=channel_manifest.json                 # required; see below
    -p 5064:5064/tcp

``VA_DATA_DIR`` overrides the mount point (default ``/data/simulation``) for
local testing without an actual bind mount.

``VA_STATE_DIR`` names the directory holding the ``active_scenarios`` file the
IOC polls for scenario switches. It is a *separate* mount because the host
writes it at run time (``osprey sim apply``) while ``data/`` is build-owned and
checksummed. Unset, it falls back to the data dir — the single-directory layout,
for a hand-run container whose state file sits next to ``machine.json``.

Facility-neutral source configuration:

``VA_CHANNELS_FILE``
    **Required.** Path to a ``{"channels": [...]}`` manifest JSON (see
    ``manifest.loaders.load_manifest_file``). Relative paths resolve against
    the data dir. The IOC's drive limits come from ``<data dir>/``
    ``channel_limits.json`` when present (none otherwise) -- the copy ``osprey
    build`` writes in beside the manifest, so the clamp needs nothing mounted
    but the served directory -- and boot values from the mounted
    ``machine.json``. There is no default: the only namespace this process
    could pick on its own is the framework's bundled demo one, and a
    container quietly serving that under a facility's name is
    indistinguishable, on the wire, from one serving the facility. ``osprey
    build`` writes this variable into a project's ``.env``; a standalone
    demo names the packaged demo manifest
    (``manifest/channel_manifest.json``, resolvable as
    ``manifest.paths.MANIFEST_OUTPUT``) explicitly.
``VA_LATTICE``
    The lattice file to serve, named relative to the data dir, or ``none``.
    Defaults to ``none`` -- a manifest names a facility's channels and says
    nothing about whether the mounted tree holds a model for them. With
    ``none``, PyAT is never imported, the served model is the empty stub in
    ``serving.model_stub``, and pyat-coupled setpoint writes (if any) latch
    without physics. The name keeps its case: the served tree is searched for
    that file verbatim, and a name that is not there refuses the boot.

Serving a lattice asks one more thing of the mount than serving a manifest
does. The model is built from a whole facility data tree, resolved through
:class:`~osprey.services.virtual_accelerator.manifest.paths.ManifestPaths`:
the lattice and the bindings that tie channels to it sit under the tree's
``simulation/``, and the write bands the model builds its variables from sit
at the tree's root. So the directory this process is given is the tree's
``simulation/`` directory, the data root is its parent, and that root --
carrying its ``channel_limits.json`` -- has to be inside the mount as well.
A lattice-backed boot refuses, by name, both a mount that is not laid out
that way and a data root whose files the model needs and cannot read, rather
than modelling a different ring than the one it serves.

Mounting ``simulation/`` alone is a whole mount for the other boot: the
namespace, the boot values and the IOC's own clamp all come from files
inside it, so such a container serves the facility's channels with no
physics behind them.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import threading
from collections.abc import Container
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pydantic

from osprey.services.virtual_accelerator.ioc.engine_source import (
    DEFAULT_NOISE_LEVEL,
    DEFAULT_POLL_INTERVAL_S,
    EngineSource,
)
from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    PARTITION_SP_ECHO,
    READBACK_SUBFIELD,
    setpoint_addresses,
)
from osprey.services.virtual_accelerator.manifest.build import MANIFEST_FILENAME
from osprey.services.virtual_accelerator.manifest.loaders import (
    load_machine_json_channels,
    load_manifest_file,
)
from osprey.services.virtual_accelerator.manifest.paths import (
    MANIFEST_OUTPUT,
    ManifestPaths,
)
from osprey.services.virtual_accelerator.serving.pvdb import build_serving_pvdb
from osprey.simulation.engine import SimulationEngine

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lume.model import LUMEModel

DEFAULT_DATA_DIR = "/data/simulation"

# The line this process prints once it is serving, and the marker everything
# that waits on that boot greps for: the image boot check
# (``scripts/va/build_and_boot_check.sh``), the container e2e fixtures, and
# anyone reading ``docker logs``. Both halves are load-bearing -- the marker
# is matched as a prefix, the channel count is read out of the remainder --
# so the whole line is one contract and is written out in exactly one place.
READY_MARKER = "virtual accelerator IOC serving PVs"


def _ready_line(channel_count: int) -> str:
    """The readiness announcement for a namespace of ``channel_count`` PVs."""
    return f"{READY_MARKER}: {channel_count} channels"


LATTICE_NONE = "none"

# Fault-seed bounds, checked at parse so an impossible instrument is refused
# before construction. Each one states what a monitor or a corrector can BE --
# a gain outside this window or a roll beyond this angle describes no device
# the model could stand in for. A magnitude an operator deliberately asked the
# simulator for is not bounded at all -- see the comment on
# _BPM_ERROR_FIELD_BOUNDS for why a displacement and a noise amplitude carry no
# bound.
MIN_BPM_GAIN = 0.1
MAX_BPM_GAIN = 10.0
MAX_BPM_ROLL_RAD = 0.1
MAX_CORR_GAIN_FACTOR = 5.0  # |factor|=1 is polarity flip; beyond 5x is absurd

# Every field VA_BPM_ERRORS knows, in the order a rendered field list spells
# them. A field named in an entry and missing here is refused as unknown.
_BPM_ERROR_FIELDS = (
    "offset_x",
    "offset_y",
    "gain_x",
    "gain_y",
    "polarity_x",
    "polarity_y",
    "roll",
    "noise_x",
    "noise_y",
)

# VA_BPM_ERRORS field -> (min, max) bound, checked at parse. Only the fields
# describing what a monitor IS appear here; a field the map does not name is
# bounded by nothing.
#
# A displacement and a noise amplitude are among those. Neither is a property
# of any monitor: they are the magnitudes an operator asked the simulator to
# seed, in whatever unit that monitor publishes, and a seeded fault is whatever
# was asked for. What still refuses them is well-formedness and not size -- a
# value that is not a finite number names no magnitude, and a negative noise
# amplitude names no distribution, since it is a standard deviation.
_BPM_ERROR_FIELD_BOUNDS: dict[str, tuple[float, float]] = {
    "gain_x": (MIN_BPM_GAIN, MAX_BPM_GAIN),
    "gain_y": (MIN_BPM_GAIN, MAX_BPM_GAIN),
    "roll": (-MAX_BPM_ROLL_RAD, MAX_BPM_ROLL_RAD),
}
# A polarity is a direction rather than a range: it must land exactly on +1 or
# -1, which is checked in _parse_bpm_errors rather than read off a bound.
_BPM_POLARITY_FIELDS = frozenset({"polarity_x", "polarity_y"})
_BPM_NOISE_FIELDS = frozenset({"noise_x", "noise_y"})


def _parse_device_float_map(env_var: str, *, bound: float) -> dict[str, float]:
    """Parse a `VA_STUCK_SETPOINTS`-shaped `"DEVICE=value,DEVICE=value,..."`
    env var into `{device: value}`, rejecting a magnitude beyond `bound`."""
    result: dict[str, float] = {}
    for entry in os.environ.get(env_var, "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        device, sep, raw_value = entry.partition("=")
        device = device.strip()
        if not sep or not device or not raw_value.strip():
            raise SystemExit(f"FATAL: {env_var} entry {entry!r} is not 'DEVICE=value'")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise SystemExit(f"FATAL: {env_var} entry {entry!r} has a non-numeric value") from exc
        if not (-bound <= value <= bound):
            raise SystemExit(
                f"FATAL: {env_var} entry {entry!r} magnitude {abs(value)} exceeds bound {bound}"
            )
        result[device] = value
    return result


def _parse_bpm_errors(env_var: str = "VA_BPM_ERRORS") -> dict[str, dict[str, float]]:
    """Parse `"BPM01:offset_x=50e-6,gain_y=1.05;BPM07:polarity_x=-1"` into
    `{"BPM01": {"offset_x": 5e-5, "gain_y": 1.05}, "BPM07": {"polarity_x": -1.0}}`,
    refusing a field `_BPM_ERROR_FIELDS` does not name, refusing any value that
    is not a finite number, and holding the fields that describe an instrument
    inside `_BPM_ERROR_FIELD_BOUNDS`. A seeded displacement or noise amplitude
    passes through at whatever size it was written, in the unit the monitor
    publishes.

    The device token is everything before the LAST colon of an entry, because
    a field list carries none and a device is as often known by the address
    its reading is published on -- colon-separated at every level -- as by the
    element it sits at. Both spellings therefore survive parsing as one token,
    and which of the two a facility's people use is theirs to decide; an entry
    with no colon at all names no fields and is refused."""
    result: dict[str, dict[str, float]] = {}
    for entry in os.environ.get(env_var, "").split(";"):
        entry = entry.strip()
        if not entry:
            continue
        device, sep, fields_raw = entry.rpartition(":")
        device = device.strip()
        if not sep or not device or not fields_raw.strip():
            raise SystemExit(
                f"FATAL: {env_var} entry {entry!r} is not 'DEVICE:field=value[,field=value...]'"
            )
        fields: dict[str, float] = {}
        for field_kv in fields_raw.split(","):
            field_kv = field_kv.strip()
            if not field_kv:
                continue
            field, fsep, raw_value = field_kv.partition("=")
            field = field.strip()
            if not fsep or field not in _BPM_ERROR_FIELDS:
                raise SystemExit(f"FATAL: {env_var} entry {entry!r} names unknown field {field!r}")
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise SystemExit(
                    f"FATAL: {env_var} entry {entry!r} field {field!r} is non-numeric"
                ) from exc
            if not math.isfinite(value):
                # `nan` and `inf` survive float() and name no magnitude, so
                # every field refuses them here. A seeded one would reach the
                # error model and be drawn with: a non-finite standard
                # deviation returns nan rather than raising, and the monitor
                # would publish nan on every read while the boot log reports
                # the seed as applied.
                raise SystemExit(
                    f"FATAL: {env_var} entry {entry!r} field {field!r}={raw_value.strip()!r} is "
                    "not a finite number"
                )
            if field in _BPM_POLARITY_FIELDS:
                if value not in (-1.0, 1.0):
                    raise SystemExit(
                        f"FATAL: {env_var} entry {entry!r} field {field!r}={value} must be +1 or -1"
                    )
            elif field in _BPM_NOISE_FIELDS:
                if value < 0.0:
                    raise SystemExit(
                        f"FATAL: {env_var} entry {entry!r} field {field!r}={value} is a standard "
                        "deviation and cannot be negative"
                    )
            elif (bound := _BPM_ERROR_FIELD_BOUNDS.get(field)) is not None:
                lo, hi = bound
                if not (lo <= value <= hi):
                    raise SystemExit(
                        f"FATAL: {env_var} entry {entry!r} field {field!r}={value} outside bound "
                        f"[{lo}, {hi}]"
                    )
            fields[field] = value
        result[device] = fields
    return result


def _positive_float_env(env_var: str, default: float) -> float:
    """Read *env_var* as a float greater than zero, or fall back to *default*.

    Args:
        env_var: Name of the variable to read.
        default: Value used when the variable is unset or empty.

    Returns:
        The configured value, or *default*.

    Raises:
        SystemExit: If the variable is set to something that is not a number,
            or to a value at or below zero — a poll interval of zero is a busy
            loop, and a negative one is not a duration at all. Refused at boot
            rather than clamped, so a typo is visible in ``docker logs``.
    """
    return _float_env(env_var, default, minimum=0.0, inclusive=False)


def _non_negative_float_env(env_var: str, default: float) -> float:
    """Read *env_var* as a float of zero or more, or fall back to *default*.

    Zero is meaningful here — it asks for no noise at all — so it is accepted
    where :func:`_positive_float_env` refuses it.
    """
    return _float_env(env_var, default, minimum=0.0, inclusive=True)


def _float_env(env_var: str, default: float, *, minimum: float, inclusive: bool) -> float:
    """Shared reader behind the two float-env helpers above."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(
            f"FATAL: {env_var}={raw!r} is not a number. Set it to a decimal "
            f"value, or unset it for the default of {default}."
        ) from None
    if value < minimum or (value == minimum and not inclusive):
        bound = f">= {minimum}" if inclusive else f"> {minimum}"
        raise SystemExit(f"FATAL: {env_var}={raw!r} must be {bound}.")
    return value


def _resolve_channels_file(data_dir: Path) -> Path:
    """Resolve ``VA_CHANNELS_FILE`` into the channel source path.

    A relative path resolves against the data dir -- the bind mount is the
    natural home for facility-supplied data files.

    Unset or empty (the compose passthrough sends ``""`` when the host var is
    absent) is refused rather than defaulted, and that refusal is the point:
    the only namespace this process could choose on its own is the
    framework's bundled demo one, and a container serving those addresses
    under a facility's name looks, to every client, exactly like one serving
    the facility. The demo namespace is still available -- it is a committed
    file
    (:data:`~osprey.services.virtual_accelerator.manifest.paths.MANIFEST_OUTPUT`)
    like any other manifest -- but it has to be asked for.
    """
    raw = os.environ.get("VA_CHANNELS_FILE", "").strip()
    if not raw:
        raise SystemExit(
            "FATAL: VA_CHANNELS_FILE names no channel manifest. The virtual "
            "accelerator serves the manifest it is given and never falls back to "
            "the framework's bundled demo namespace.\n"
            "  Project deployment: run `osprey build`, which generates "
            f"{MANIFEST_FILENAME} into the project's data/simulation/ and writes "
            "VA_CHANNELS_FILE into its .env.\n"
            "  Standalone demo: ask for the packaged demo manifest by name --\n"
            f"    -e VA_CHANNELS_FILE={MANIFEST_OUTPUT} "
            f"-e VA_LATTICE={ManifestPaths(data_root=data_dir.parent).lattice_json.name}\n"
            "  (that path is where this installation carries it)."
        )
    path = Path(raw)
    return path if path.is_absolute() else data_dir / path


def _resolve_lattice(data_dir: Path) -> Path | None:
    """Resolve ``VA_LATTICE`` into the served lattice file, or ``None``.

    The value names a file in the served tree, relative to the data dir, the
    way ``VA_CHANNELS_FILE`` names the manifest beside it; :data:`LATTICE_NONE`
    is the one value naming no file at all, and an empty or unset variable
    reads as that. A manifest names a facility's channels and says nothing
    about whether the mounted tree holds a model for them, so a deployment
    that wants one asks for it by name: ``osprey build`` writes the name it
    derived from the project's own tree.

    Case is preserved, because the name is looked up in that tree verbatim and
    a case-folding resolver would find a file on one host and miss it on
    another.

    Returning a path rather than the bare name is what makes the variable a
    *source* like the manifest and not a flag: a name the mount does not carry
    is refused here, against the tree this process was handed, instead of
    surfacing later as a decoding failure from inside the lattice loader.
    """
    raw = os.environ.get("VA_LATTICE", "").strip()
    if not raw or raw == LATTICE_NONE:
        return None
    named = Path(raw)
    path = named if named.is_absolute() else data_dir / named
    if not path.is_file():
        raise SystemExit(
            f"FATAL: VA_LATTICE={raw!r} names no file in the served tree ({path} "
            f"is not there). The name is looked up verbatim, case included. Set "
            f"VA_LATTICE={LATTICE_NONE} to serve the manifest's channels without "
            f"physics, or mount a tree that carries that lattice."
        )
    return path


def _load_drive_limits(path: Path, *, setpoints: Container[str]) -> dict[str, tuple[float, float]]:
    """Derive the ``build_records(drive_limits=...)`` map from
    ``channel_limits.json``: one ``(min_value, max_value)`` entry per
    writable setpoint, used as the IOC's drive band.

    ``path`` is the file to read, and there is no default: the bands belong to
    one particular tree, and a process that serves a facility was handed that
    facility's directory. A caller that wants the bundled tree's bands names
    the bundled file.

    ``setpoints`` is the manifest's own setpoint set -- the channels
    whose ``subfield`` says they are written. The limits file carries an
    entry per address, read-only ones included, so which of them are
    setpoints has to come from the manifest; reading it off the address text
    would hand an empty band map to every facility whose setpoints are not
    spelled ``...:SP``."""
    raw = json.loads(path.read_text())
    defaults = raw.get("defaults", {})
    limits: dict[str, tuple[float, float]] = {}
    for address, entry in raw.items():
        if address.startswith("_") or address == "defaults":
            continue
        if address not in setpoints:
            continue
        merged = {**defaults, **entry}
        if not merged.get("writable", True):
            continue
        min_value = merged.get("min_value")
        max_value = merged.get("max_value")
        if min_value is None or max_value is None:
            continue
        limits[address] = (float(min_value), float(max_value))
    return limits


def _load_boot_values(machine_path: Path | None = None) -> dict[str, float]:
    """Derive the ``build_records(boot_values=...)`` map from
    machine.json's scenario-seed channels (see ``ioc/records.py``'s
    ``build_records`` docstring). A handful of derived channels (e.g. RF
    net power, computed via an ``expr`` rather than a stored ``value``)
    carry no static value and are skipped -- harmless here since none of
    them are ``:SP``/``:RB`` addresses, the only subfields this map is ever
    consulted for. ``machine_path`` selects which machine.json to read;
    ``None`` (the default) reads the bundled template."""
    return {
        address: entry["value"]
        for address, entry in load_machine_json_channels(machine_path).items()
        if "value" in entry
    }


def _served_tree(data_dir: Path, lattice_path: Path) -> ManifestPaths:
    """The facility data tree behind the served directory, for the model.

    The model is built from a whole tree, not from a lattice file: the
    bindings that say which channel drives which element, the ``machine.json``
    nominals it boots holding and the ``channel_limits.json`` bands those
    nominals are weighed against all have to come from the same one, and
    :class:`~osprey.services.virtual_accelerator.manifest.paths.ManifestPaths`
    is where that layout is written down -- once, for build time and serve
    time alike. Its ``simulation/`` directory is what this process is given
    and its root is that directory's parent.

    Refusing a mount that does not match is the point of asking here. The
    alternative is a process that serves one lattice and models another: the
    resolution that builds the model is not the one that answered
    ``VA_LATTICE``, so a disagreement between them is invisible from the wire
    -- every channel is served, every write is accepted, and the physics
    behind them belongs to a different ring.

    Raises:
        SystemExit: the served directory is not the tree's ``simulation/``
            directory, or ``VA_LATTICE`` names a lattice that is not the one
            that tree carries.
    """
    paths = ManifestPaths(data_root=data_dir.parent)
    if paths.lattice_json.parent != lattice_path.parent:
        raise SystemExit(
            f"FATAL: serving a lattice needs the mounted directory to be a facility "
            f"tree's simulation/ directory, because the model is built from the whole "
            f"tree around it (bindings, nominals and write bands included). "
            f"{data_dir} is not: the model would look for its lattice at "
            f"{paths.lattice_json}. Bind-mount <project>/build/data/simulation, or "
            f"point VA_DATA_DIR at it."
        )
    if lattice_path != paths.lattice_json:
        raise SystemExit(
            f"FATAL: VA_LATTICE names {lattice_path.name}, but the tree's own lattice -- "
            f"the file its va_bindings.json was derived against, and the only one the "
            f"model reads -- is {paths.lattice_json.name}. Serving one ring and "
            f"modelling another is indistinguishable, on the wire, from serving the "
            f"one the bindings describe. Rename the file, or set "
            f"VA_LATTICE={paths.lattice_json.name}."
        )
    return paths


def _refuse_unbound_coupled_channels(channels: list[dict], document: Any, source: Path) -> None:
    """Refuse a manifest whose pyat-coupled channels the bindings do not bind.

    The manifest's pyat-coupled partition is derived from the bindings
    document at build time, so the two describe the same addresses whenever
    they came from one build. When they did not, the model is handed a
    coupled channel nothing binds: it reaches the variable catalog as a plain
    declared scalar and the model layer rejects it for its *type*, naming
    neither the address nor the file the binding is missing from. That is a
    diagnosable failure here and an opaque one three layers down, so it is
    answered here.

    Readbacks are excluded, as the catalog excludes them: the serving layer
    mirrors a setpoint onto its readback record, so a readback is not a
    variable and is bound by nothing.

    Raises:
        SystemExit: at least one pyat-coupled channel is unbound; the message
            names them and the bindings file they are missing from.
    """
    bound = {binding.setpoint_address for binding in document.bindings}
    unbound = sorted(
        channel["address"]
        for channel in channels
        if channel["partition"] == PARTITION_PYAT_COUPLED
        and channel["subfield"] != READBACK_SUBFIELD
        and channel["address"] not in bound
    )
    if unbound:
        raise SystemExit(
            f"FATAL: {len(unbound)} channel(s) the manifest calls pyat-coupled are "
            f"bound to nothing by {source}: {unbound[:5]}. The manifest's coupled "
            f"partition is derived from that document, so the two came from "
            f"different builds -- rebuild the deployment so its manifest and its "
            f"bindings describe one accelerator."
        )


def _raise_keyboard_interrupt(signum: int, _frame: Any) -> None:
    """Signal handler: turn a stop signal into the interrupt the runner exits on."""
    raise KeyboardInterrupt(f"signal {signum}")


def _install_shutdown_signals() -> None:
    """Make SIGTERM behave exactly as Ctrl-C already does.

    The runner's ``run()`` blocks on its queue forever and returns on one
    thing only: a ``KeyboardInterrupt`` reaching the thread that called it.
    SIGINT raises one by Python's own default; SIGTERM -- what ``docker
    stop`` sends -- terminates the process outright unless a handler says
    otherwise. Pointing both at the same handler is what makes a container
    stop leave through the runner's documented exit rather than through an
    abrupt kill. SIGINT is installed explicitly too, so the pair is
    symmetric and neither depends on an inherited disposition.

    Installed only once the servers are up: before that the process is still
    assembling, and the default dispositions (die immediately) are the right
    answer to a stop signal arriving mid-assembly.
    """
    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)


def _start_engine_source(engine_source: EngineSource, interval: float) -> threading.Thread:
    """Run the telemetry poll loop on a daemon thread of its own.

    There is no shared dispatcher to schedule it on: the runner owns the
    calling thread (``run()`` blocks on it) and runs its Channel Access server
    on one of its own. So the poll loop gets one too.

    Daemon deliberately. It holds nothing worth draining -- each tick reads
    the scenario files afresh and pushes values it recomputes -- and the
    process must never wait on a loop that has no end.

    ``run_forever`` rather than a hand-rolled loop, so the source's own
    per-record failure isolation (see its docstring: an escaping exception
    would freeze every telemetry channel at its boot value, silently) stays
    on this path.
    """
    thread = threading.Thread(
        target=lambda: asyncio.run(engine_source.run_forever(interval)),
        name="engine-source",
        daemon=True,
    )
    thread.start()
    return thread


def main() -> None:
    from osprey.utils.logger import configure_logging

    # Container entry point: without this the serving/PyAT/framework log records
    # this process drives would have no handler. Records go to stderr.
    configure_logging()

    data_dir = Path(os.environ.get("VA_DATA_DIR", DEFAULT_DATA_DIR))
    # Two facility-network facts, read here rather than fixed in the image: how
    # often the telemetry thread republishes engine values, and how much noise
    # the synthesised channels carry. Both default to the `engine_source`
    # constants, which are the one place either number is written down.
    poll_interval_s = _positive_float_env("VA_POLL_INTERVAL_S", DEFAULT_POLL_INTERVAL_S)
    noise_level = _non_negative_float_env("VA_NOISE_LEVEL", DEFAULT_NOISE_LEVEL)
    state_dir = Path(os.environ.get("VA_STATE_DIR", "").strip() or data_dir)
    machine_path = data_dir / "machine.json"
    if not machine_path.is_file():
        raise SystemExit(
            f"FATAL: no machine.json at {machine_path}. "
            f"Bind-mount a project's data/simulation/ DIRECTORY (never a single "
            f"file) to {DEFAULT_DATA_DIR}, or set VA_DATA_DIR -- see README.md."
        )

    channels_file = _resolve_channels_file(data_dir)
    lattice_path = _resolve_lattice(data_dir)

    # Every data file comes from the manifest that was named and the mount
    # beside it -- never from the bundled tutorial data, whose addresses
    # belong to one particular facility's namespace. The packaged demo
    # manifest reaches this the same way any other does: by being named.
    print(f"Loading channel manifest from {channels_file} ...", flush=True)
    channels = load_manifest_file(channels_file)
    setpoints = setpoint_addresses(channels)
    limits_path = data_dir / "channel_limits.json"
    drive_limits = (
        _load_drive_limits(limits_path, setpoints=setpoints) if limits_path.is_file() else {}
    )
    boot_values = _load_boot_values(machine_path)

    stuck_setpoints = frozenset(
        addr.strip() for addr in os.environ.get("VA_STUCK_SETPOINTS", "").split(",") if addr.strip()
    )
    if stuck_setpoints:
        print(f"VA apply-fault active: {sorted(stuck_setpoints)}", flush=True)

    bpm_errors = _parse_bpm_errors("VA_BPM_ERRORS")
    # VA_CORR_GAIN feeds PhysicsBridge's magnet_cal, which is family-agnostic
    # (any magnet, not just correctors) despite the "CORR" name here.
    corr_gain = _parse_device_float_map("VA_CORR_GAIN", bound=MAX_CORR_GAIN_FACTOR)
    corrector_gains = {device: {"factor": factor} for device, factor in corr_gain.items()}
    if bpm_errors:
        print(f"VA apply-fault active: bpm_errors={bpm_errors}", flush=True)
    if corrector_gains:
        print(f"VA apply-fault active: corrector_gains={corrector_gains}", flush=True)

    # The physics the runner serves: the ring in a lattice-backed boot, the
    # empty stub otherwise. One or the other, never both, and never none --
    # the runner is built around a model.
    model: LUMEModel
    # What the bindings document says each writable address does on readback.
    # Empty with no lattice: there is no document, and every coupled setpoint
    # (if the manifest names any) latches.
    bound: dict[str, Any] = {}
    if lattice_path is not None:
        # Deferred import: all of these reach PyAT or lume at module level,
        # and the whole point of VA_LATTICE=none is booting without PyAT
        # installed or importable.
        from osprey.services.virtual_accelerator.bindings import BindingsError, load_bindings
        from osprey.services.virtual_accelerator.ioc.physics_bridge import (
            OrbitSolveError,
            PhysicsBridge,
        )
        from osprey.services.virtual_accelerator.lattice.errors import (
            BpmErrorSeedError,
            resolve_bpm_errors,
        )
        from osprey.services.virtual_accelerator.model.pyat import PyATRingModel
        from osprey.services.virtual_accelerator.serving.write_path import bound_setpoints

        paths = _served_tree(data_dir, lattice_path)
        # Read here as well as inside the model: the served path needs the
        # readback rule each binding declares, and the model needs the
        # bindings themselves. One file, read twice, rather than a document
        # threaded through a constructor that has no parameter for it.
        try:
            document = load_bindings(paths.va_bindings)
        except FileNotFoundError as exc:
            raise SystemExit(
                f"FATAL: {lattice_path} is served but the tree carries no bindings at "
                f"{paths.va_bindings}. A lattice on its own models nothing any channel "
                f"can reach, which is why the bindings file is what makes a tree a "
                f"virtual-accelerator tree."
            ) from exc
        except BindingsError as exc:
            raise SystemExit(
                f"FATAL: {paths.va_bindings} is not a bindings document: {exc}"
            ) from exc
        _refuse_unbound_coupled_channels(channels, document, paths.va_bindings)

        # The model is constructed here rather than left to the bridge to
        # build, because two things now need the same one: the bridge serves
        # writes through it, and the runner serves its variables on PVA and
        # owns the thread every access to it happens on. One instance, one
        # lattice; a second model would be a second lattice, silently
        # diverging from the one whose orbit the BPM readings come from.
        #
        # Every refusal the model raises is turned into a SystemExit naming
        # the served tree. Ending the process is the serving layer's
        # decision, which is why the model itself never does it -- and an
        # unhandled traceback out of a container's PID 1 is the one shape of
        # boot failure nobody can read.
        try:
            model = PyATRingModel(paths.data_root, channels)
        except OrbitSolveError as exc:
            raise SystemExit(
                f"FATAL: the served lattice has no stable closed orbit at boot ({exc})"
            ) from exc
        except BindingsError as exc:
            raise SystemExit(
                f"FATAL: the tree at {paths.data_root} does not describe one accelerator: {exc}"
            ) from exc
        except FileNotFoundError as exc:
            raise SystemExit(
                f"FATAL: the tree at {paths.data_root} is missing a file its model needs: {exc}"
            ) from exc
        except pydantic.ValidationError as exc:
            # The band refusal: a machine.json nominal outside its
            # channel_limits.json band, on the served files. It is what makes
            # the model's declared state the facility's own, so it reads as a
            # statement about this tree rather than as a validation dump. It
            # is the variable's own model that refuses it, which is why this
            # clause is narrower than the one below and comes before it.
            raise SystemExit(
                f"FATAL: the tree at {paths.data_root} contradicts itself -- a "
                f"machine.json nominal falls outside the band "
                f"{paths.channel_limits} states for it ({exc})"
            ) from exc
        except ValueError as exc:
            # Everything else the tree can be wrong about: a slice weight, an
            # energy table, a file that is not JSON, a binding the model
            # cannot make a variable of. Each of those names its own file and
            # key in the text it carries, so the headline stays neutral --
            # naming a cause here that is not the cause sends an operator to
            # the wrong file.
            raise SystemExit(
                f"FATAL: the model refused the tree at {paths.data_root}: {exc}"
            ) from exc

        # What each coupled setpoint serves on readback, from the document
        # rather than from the manifest's setpoint/readback pairing: only the
        # document knows that a readback is the setpoint's value mapped back
        # along the facility's own reverse curve. Asked for here, where the
        # tree is resolved, because the runner resolves no tree of its own.
        try:
            bound = dict(bound_setpoints(document, model.supported_variables))
        except ValueError as exc:
            raise SystemExit(
                f"FATAL: {paths.va_bindings} declares a readback the served namespace "
                f"cannot produce: {exc}"
            ) from exc

        # A seeded readout error names its device the way whoever seeded it
        # knows the device: by the address the reading is published on, or by
        # the element the deck carries it at. The bridge holds its state by
        # element, because a reading is a pair of planes at one monitor, so
        # the document -- the one thing that carries both spellings -- is what
        # turns the seeds into that key. A spelling it knows neither way ends
        # the boot here: a machine serving unperturbed readings while looking
        # configured is the failure a fault seed exists to make visible.
        monitors = {
            binding.setpoint_address: binding.element
            for binding in document.bindings
            if binding.kind == "monitor" and binding.element is not None
        }
        try:
            bpm_errors = resolve_bpm_errors(bpm_errors, monitors)
        except BpmErrorSeedError as exc:
            raise SystemExit(f"FATAL: VA_BPM_ERRORS: {exc}") from exc
        if bpm_errors:
            # The second half of the pair printed above: what was asked for,
            # and the monitors it resolved to. An operator reading the log of
            # a container that serves addresses needs both to see that the
            # device they meant is the device that was perturbed.
            print(f"VA apply-fault active: bpm_errors at elements {bpm_errors}", flush=True)

        bridge = PhysicsBridge(
            model=model,
            bpm_errors=bpm_errors or None,
            corrector_gains=corrector_gains or None,
        )
        on_pyat_setpoint = bridge.on_setpoint
    else:
        if bpm_errors or corrector_gains:
            raise SystemExit(
                "FATAL: VA_BPM_ERRORS/VA_CORR_GAIN are lattice-physics faults "
                "and require VA_LATTICE to name a lattice file in the data dir"
            )
        print("No lattice configured (VA_LATTICE=none): PhysicsBridge skipped", flush=True)
        # The served namespace is the manifest's, whole, either way -- the
        # model only ever describes the physics behind part of it. With none,
        # it describes nothing and the co-hosted namespace is all there is.
        from osprey.services.virtual_accelerator.serving.model_stub import NullModel

        model = NullModel()
        bridge = None
        on_pyat_setpoint = None

    # async_setpoints is not optional here: every setpoint that routes through
    # the model is completed only once the solve behind it has finished, and a
    # synchronous PV would tell the client its write had landed before the
    # solve had even started. The write path refuses to be built without it.
    records = build_serving_pvdb(
        channels,
        drive_limits=drive_limits,
        boot_values=boot_values,
        async_setpoints=True,
    )
    print(
        f"Built serving database: {len(records.all)} channels "
        f"({len(records.pyat_coupled)} pyat-coupled, {len(records.static_noisy)} static-noisy)",
        flush=True,
    )
    if bridge is not None:
        # Pushes the boot BPM readings into the database's specs. Before the
        # runner exists, and it has to be: these are the values the Channel
        # Access server comes up serving.
        bridge.bind(records.pyat_coupled)

    print(f"Loading simulation engine from {machine_path} ...", flush=True)
    engine = SimulationEngine.from_file(machine_path, state_dir=state_dir)

    # With no lattice, the engine is the only physics in the process: sync
    # each sp-echo readback into it every tick so machine-file expression
    # channels can respond to accepted setpoints (see EngineSource's
    # setpoint_echo_records docstring). With a lattice, physics coupling
    # flows through PhysicsBridge and the engine stays a pure scenario source.
    setpoint_echoes: dict[str, Any] | None = None
    if lattice_path is None:
        setpoint_echoes = {
            ch["address"]: records.all[ch["address"]]
            for ch in channels
            if ch["partition"] == PARTITION_SP_ECHO
            and ch["subfield"] == READBACK_SUBFIELD
            and ch["address"] in records.all
        }

    engine_source = EngineSource(
        engine,
        channels,
        records.static_noisy,
        data_dir,
        state_dir=state_dir,
        noise_level=noise_level,
        setpoint_echo_records=setpoint_echoes,
    )

    # Import the runner only now: it is the one module here that reaches the
    # Channel Access server extension, and a lattice-free boot on a host
    # without it must still get this far.
    from osprey.services.virtual_accelerator.serving.runner import CohostRunner

    # Constructing the runner creates the servers and starts serving. Nothing
    # after this may write into a PV spec -- the specs have been copied into
    # live PVs -- which is why the runner points every record at the driver
    # itself as its last act.
    runner = CohostRunner(
        model,
        records,
        on_setpoint=on_pyat_setpoint,
        drive_limits=drive_limits,
        bound_setpoints=bound,
        stuck_setpoints=stuck_setpoints,
    )

    # Telemetry starts only once the driver is attached, so its first tick
    # posts monitor events to the server rather than editing boot specs
    # behind it.
    _start_engine_source(engine_source, poll_interval_s)

    _install_shutdown_signals()
    print(_ready_line(len(records.all)), flush=True)

    # `run()` is the run loop: it blocks on the queue and returns only on a
    # KeyboardInterrupt, which the handlers installed above raise for SIGINT
    # and SIGTERM alike. It catches that itself; catching it again here
    # covers the window in which a signal arrives between the loop's own
    # try and this call.
    #
    # Shutdown is process exit, not a server teardown: there is no stop API
    # to call, and nothing in this process holds state that outlives it. The
    # Channel Access server thread is a daemon, the telemetry thread is a
    # daemon, the PVA server's threads are the server library's own, and
    # every one of them is released when the process image is. A write
    # already queued when the signal arrives is never applied and its
    # put-completion never fires -- the client's put times out rather than
    # being told a value landed that did not.
    try:
        runner.run()
    except KeyboardInterrupt:
        pass
    print("virtual accelerator IOC stopped", flush=True)


if __name__ == "__main__":
    main()
