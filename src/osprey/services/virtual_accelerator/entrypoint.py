"""Virtual Accelerator IOC entrypoint.

Assembles the full VA soft-IOC in one process, in dependency order:

    manifest -> records -> physics bridge (partition a) -> engine source (partition c)

then serves Channel Access via the probe-proven configuration (TCP
name-server; see src/osprey/services/virtual_accelerator/probe/README.md and
src/osprey/templates/data/facility_gateways.py's "Local Simulation" preset,
which points at exactly this container's published port).

Run contract (see docker/virtual-accelerator/README.md for the full version):

    -v <project>/data/simulation:/data/simulation   # the DIRECTORY, never a file
    -p 5064:5064/tcp

``VA_DATA_DIR`` overrides the mount point (default ``/data/simulation``) for
local testing without an actual bind mount.

Facility-neutral source configuration (all optional; defaults reproduce the
historical behaviour exactly):

``VA_CHANNELS_FILE``
    Path to a ``{"channels": [...]}`` manifest JSON (see
    ``manifest.loaders.load_manifest_file``). Relative paths resolve against
    the data dir. Unset/empty -> the built-in generated manifest
    (``build_manifest()``), as before. With a file source, drive limits come
    from ``<data dir>/channel_limits.json`` when present (none otherwise)
    and boot values from the mounted ``machine.json`` -- never from the
    bundled tutorial data.
``VA_LATTICE``
    ``builtin`` or ``none``: whether to construct the PyAT-backed
    ``PhysicsBridge``. Defaults to ``builtin`` for the built-in manifest and
    ``none`` for a file-backed one. With ``none``, PyAT is never imported
    and pyat-coupled setpoint writes (if any) latch without physics.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from osprey.services.virtual_accelerator.ioc.engine_source import EngineSource
from osprey.services.virtual_accelerator.ioc.records import (
    READBACK_SUBFIELD,
    build_records,
)
from osprey.services.virtual_accelerator.manifest import PARTITION_SP_ECHO, build_manifest
from osprey.services.virtual_accelerator.manifest.loaders import (
    load_machine_json_channels,
    load_manifest_file,
)
from osprey.simulation.engine import SimulationEngine

DEFAULT_DATA_DIR = "/data/simulation"
ENGINE_POLL_INTERVAL_S = 1.0

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
    limits file to parse; ``None`` (the default) keeps the historical
    bundled-template read."""
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
    ``None`` (the default) keeps the historical bundled-template read."""
    return {
        address: entry["value"]
        for address, entry in load_machine_json_channels(machine_path).items()
        if "value" in entry
    }


def main() -> None:
    data_dir = Path(os.environ.get("VA_DATA_DIR", DEFAULT_DATA_DIR))
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
        print(f"Building channel manifest and IOC records (data dir: {data_dir}) ...", flush=True)
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

    if lattice_mode == LATTICE_BUILTIN:
        # Deferred import: physics_bridge imports PyAT at module level, and
        # the whole point of VA_LATTICE=none is booting without PyAT
        # installed or importable.
        from osprey.services.virtual_accelerator.ioc.physics_bridge import PhysicsBridge

        bridge = PhysicsBridge(
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
        bridge = None
        on_pyat_setpoint = None

    records = build_records(
        channels,
        on_pyat_setpoint=on_pyat_setpoint,
        stuck_setpoints=stuck_setpoints,
        drive_limits=drive_limits,
        boot_values=boot_values,
    )
    if bridge is not None:
        bridge.bind(records.pyat_coupled)

    print(f"Loading simulation engine from {machine_path} ...", flush=True)
    engine = SimulationEngine.from_file(machine_path)

    # With no lattice, the engine is the only physics in the process: sync
    # each sp-echo readback into it every tick so machine-file expression
    # channels can respond to accepted setpoints (see EngineSource's
    # setpoint_echo_records docstring). With a lattice, physics coupling
    # flows through PhysicsBridge and the engine stays a pure scenario
    # source -- exactly the historical behaviour.
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
        setpoint_echo_records=setpoint_echoes,
    )

    # Import softioc only now: constructing softioc records (build_records/
    # PhysicsBridge above, both already done) must happen before iocInit, and
    # this module's own CA client-poisoning caveat (see
    # tests/va/test_record_factory.py's docstring) is irrelevant here -- this
    # process is the IOC server only, never also a CA client.
    from softioc import asyncio_dispatcher, builder, softioc

    dispatcher = asyncio_dispatcher.AsyncioDispatcher()
    builder.LoadDatabase()
    softioc.iocInit(dispatcher)

    asyncio.run_coroutine_threadsafe(
        engine_source.run_forever(ENGINE_POLL_INTERVAL_S), dispatcher.loop
    )

    print(
        f"virtual accelerator IOC serving PVs: {len(records.all)} channels "
        f"({len(records.pyat_coupled)} pyat-coupled, {len(records.static_noisy)} static-noisy)",
        flush=True,
    )

    # wait_for_quit() installs SIGINT/SIGTERM handlers and blocks until either
    # fires, so `docker stop`/Ctrl-C shut the container down cleanly.
    try:
        dispatcher.wait_for_quit()
    except AttributeError:  # pragma: no cover -- defensive: older softioc without wait_for_quit
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
