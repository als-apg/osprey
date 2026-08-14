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
    -p 5064:5064/tcp

``VA_DATA_DIR`` overrides the mount point (default ``/data/simulation``) for
local testing without an actual bind mount.

``VA_STATE_DIR`` names the directory holding the ``active_scenarios`` file the
IOC polls for scenario switches. It is a *separate* mount because the host
writes it at run time (``osprey sim apply``) while ``data/`` is build-owned and
checksummed. Unset, it falls back to the data dir — the single-directory layout,
for a hand-run container whose state file sits next to ``machine.json``.

Facility-neutral source configuration. Every variable below is optional, and
unset they serve the built-in tutorial machine byte-for-byte — a deployment
that sets none of them is unaffected by all of them:

``VA_CHANNELS_FILE``
    Path to a ``{"channels": [...]}`` manifest JSON (see
    ``manifest.loaders.load_manifest_file``). Relative paths resolve against
    the data dir. Unset/empty -> the built-in generated manifest
    (``build_manifest()``). With a file source, drive limits come
    from ``<data dir>/channel_limits.json`` when present (none otherwise)
    and boot values from the mounted ``machine.json`` -- never from the
    bundled tutorial data.
``VA_LATTICE``
    ``builtin`` or ``none``: whether to construct the PyAT-backed
    ``PhysicsBridge``. Defaults to ``builtin`` for the built-in manifest and
    ``none`` for a file-backed one. With ``none``, PyAT is never imported,
    the served model is the empty stub in
    ``serving.model_stub``, and pyat-coupled setpoint writes (if any) latch
    without physics.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.services.virtual_accelerator.ioc.engine_source import EngineSource
from osprey.services.virtual_accelerator.manifest import PARTITION_SP_ECHO, build_manifest
from osprey.services.virtual_accelerator.manifest.loaders import (
    load_machine_json_channels,
    load_manifest_file,
)
from osprey.services.virtual_accelerator.serving.pvdb import (
    READBACK_SUBFIELD,
    build_serving_pvdb,
)
from osprey.simulation.engine import SimulationEngine

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lume.model import LUMEModel

DEFAULT_DATA_DIR = "/data/simulation"
ENGINE_POLL_INTERVAL_S = 1.0

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


LATTICE_BUILTIN = "builtin"
LATTICE_NONE = "none"

# FR4 fault-seed bounds -- "bound each magnitude at parse (reject absurd
# values before construction)". Generous vs. plausible commissioning-error
# magnitudes, so real faults always parse, but tight enough to reject a
# fat-fingered/unit-confused entry (e.g. millimeters typed where meters were
# meant) before it ever reaches PhysicsBridge.
MAX_BPM_OFFSET_M = 1e-2
MIN_BPM_GAIN = 0.1
MAX_BPM_GAIN = 10.0
MAX_BPM_ROLL_RAD = 0.1
MAX_BPM_NOISE_M = 1e-2
MAX_CORR_GAIN_FACTOR = 5.0  # |factor|=1 is polarity flip; beyond 5x is absurd

# VA_BPM_ERRORS field -> (min, max) bound, checked at parse. polarity_x/y are
# additionally required to land exactly on a bound (see _parse_bpm_errors).
_BPM_ERROR_FIELD_BOUNDS: dict[str, tuple[float, float]] = {
    "offset_x": (-MAX_BPM_OFFSET_M, MAX_BPM_OFFSET_M),
    "offset_y": (-MAX_BPM_OFFSET_M, MAX_BPM_OFFSET_M),
    "gain_x": (MIN_BPM_GAIN, MAX_BPM_GAIN),
    "gain_y": (MIN_BPM_GAIN, MAX_BPM_GAIN),
    "polarity_x": (-1.0, 1.0),
    "polarity_y": (-1.0, 1.0),
    "roll": (-MAX_BPM_ROLL_RAD, MAX_BPM_ROLL_RAD),
    "noise_x": (0.0, MAX_BPM_NOISE_M),
    "noise_y": (0.0, MAX_BPM_NOISE_M),
}
_BPM_POLARITY_FIELDS = frozenset({"polarity_x", "polarity_y"})


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
    bounding every field per `_BPM_ERROR_FIELD_BOUNDS`."""
    result: dict[str, dict[str, float]] = {}
    for entry in os.environ.get(env_var, "").split(";"):
        entry = entry.strip()
        if not entry:
            continue
        device, sep, fields_raw = entry.partition(":")
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
            bound = _BPM_ERROR_FIELD_BOUNDS.get(field)
            if not fsep or bound is None:
                raise SystemExit(f"FATAL: {env_var} entry {entry!r} names unknown field {field!r}")
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise SystemExit(
                    f"FATAL: {env_var} entry {entry!r} field {field!r} is non-numeric"
                ) from exc
            if field in _BPM_POLARITY_FIELDS:
                if value not in (-1.0, 1.0):
                    raise SystemExit(
                        f"FATAL: {env_var} entry {entry!r} field {field!r}={value} must be +1 or -1"
                    )
            else:
                lo, hi = bound
                if not (lo <= value <= hi):
                    raise SystemExit(
                        f"FATAL: {env_var} entry {entry!r} field {field!r}={value} outside bound "
                        f"[{lo}, {hi}]"
                    )
            fields[field] = value
        result[device] = fields
    return result


def _resolve_channels_file(data_dir: Path) -> Path | None:
    """Resolve ``VA_CHANNELS_FILE`` into the file-backed channel source path.

    Unset or empty (the compose passthrough sends ``""`` when the host var
    is absent) means the built-in generated manifest. A relative path
    resolves against the data dir -- the bind mount is the natural home for
    facility-supplied data files.
    """
    raw = os.environ.get("VA_CHANNELS_FILE", "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else data_dir / path


def _resolve_lattice_mode(channels_file: Path | None) -> str:
    """Resolve ``VA_LATTICE`` into ``builtin`` or ``none``.

    The default follows the channel source: the built-in manifest describes
    the lattice-backed tutorial machine (so ``builtin``), while a
    file-backed manifest comes from a facility with no PyAT model (so
    ``none``). An explicit value overrides either default -- a facility MAY
    pair a file manifest with the built-in lattice if its addresses map.
    """
    raw = os.environ.get("VA_LATTICE", "").strip().lower()
    if not raw:
        return LATTICE_NONE if channels_file is not None else LATTICE_BUILTIN
    if raw not in (LATTICE_BUILTIN, LATTICE_NONE):
        raise SystemExit(
            f"FATAL: VA_LATTICE must be {LATTICE_BUILTIN!r} or {LATTICE_NONE!r}, got {raw!r}"
        )
    return raw


def _channel_limits_path() -> Path:
    """Locate ``channel_limits.json`` from the installed ``osprey.templates``
    package -- the same convention ``manifest/paths.py`` uses for the other
    control-assistant data files -- so this works identically from an
    editable checkout, a built wheel, or the wheel-drop context an image
    build stages."""
    import osprey.templates

    return (
        Path(osprey.templates.__file__).parent
        / "apps"
        / "control_assistant"
        / "data"
        / "channel_limits.json"
    )


def _load_drive_limits(path: Path | None = None) -> dict[str, tuple[float, float]]:
    """Derive the ``build_records(drive_limits=...)`` map from
    ``channel_limits.json``: one ``(min_value, max_value)`` entry per
    writable ``:SP`` address with numeric bounds. ``ioc/records.py`` stays
    file-blind (see its ``build_records`` docstring) -- this is the file
    read its ``drive_limits`` argument replaces. ``path`` selects which
    limits file to parse; ``None`` (the default) reads the bundled
    template."""
    raw = json.loads((path or _channel_limits_path()).read_text())
    defaults = raw.get("defaults", {})
    limits: dict[str, tuple[float, float]] = {}
    for address, entry in raw.items():
        if address.startswith("_") or address == "defaults" or not address.endswith(":SP"):
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
    state_dir = Path(os.environ.get("VA_STATE_DIR", "").strip() or data_dir)
    machine_path = data_dir / "machine.json"
    if not machine_path.is_file():
        raise SystemExit(
            f"FATAL: no machine.json at {machine_path}. "
            f"Bind-mount a project's data/simulation/ DIRECTORY (never a single "
            f"file) to {DEFAULT_DATA_DIR}, or set VA_DATA_DIR -- see README.md."
        )

    channels_file = _resolve_channels_file(data_dir)
    lattice_mode = _resolve_lattice_mode(channels_file)

    if channels_file is not None:
        # File-backed facility: every data file comes from the mount, never
        # from the bundled tutorial data (whose addresses belong to another
        # facility's namespace).
        print(f"Loading channel manifest from {channels_file} ...", flush=True)
        channels = load_manifest_file(channels_file)
        limits_path = data_dir / "channel_limits.json"
        drive_limits = _load_drive_limits(limits_path) if limits_path.is_file() else {}
        boot_values = _load_boot_values(machine_path)
    else:
        print(f"Building channel manifest (data dir: {data_dir}) ...", flush=True)
        channels = build_manifest()["channels"]
        drive_limits = _load_drive_limits()
        boot_values = _load_boot_values()

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
    if lattice_mode == LATTICE_BUILTIN:
        # Deferred import: both of these reach PyAT at module level, and the
        # whole point of VA_LATTICE=none is booting without PyAT installed
        # or importable.
        from osprey.services.virtual_accelerator.ioc.physics_bridge import (
            OrbitSolveError,
            PhysicsBridge,
        )
        from osprey.services.virtual_accelerator.model.pyat import PyATRingModel

        # The model is constructed here rather than left to the bridge to
        # build, because two things now need the same one: the bridge serves
        # writes through it, and the runner serves its variables on PVA and
        # owns the thread every access to it happens on. One instance, one
        # lattice; a second model would be a second lattice, silently
        # diverging from the one whose orbit the BPM readings come from.
        try:
            model = PyATRingModel()
        except OrbitSolveError as exc:
            # Ending the process is the serving layer's decision, which is
            # why the model itself never does it: this turns an opaque boot
            # crash into a diagnosable one.
            raise SystemExit(
                f"FATAL: the SR lattice has no stable closed orbit at boot ({exc})"
            ) from exc

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
                f"and require VA_LATTICE={LATTICE_BUILTIN!r}"
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
    if lattice_mode == LATTICE_NONE:
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
        stuck_setpoints=stuck_setpoints,
    )

    # Telemetry starts only once the driver is attached, so its first tick
    # posts monitor events to the server rather than editing boot specs
    # behind it.
    _start_engine_source(engine_source, ENGINE_POLL_INTERVAL_S)

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
