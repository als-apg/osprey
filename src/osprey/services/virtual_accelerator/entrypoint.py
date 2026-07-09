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
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from osprey.services.virtual_accelerator.ioc.engine_source import EngineSource
from osprey.services.virtual_accelerator.ioc.physics_bridge import PhysicsBridge
from osprey.services.virtual_accelerator.ioc.records import build_records
from osprey.services.virtual_accelerator.manifest import build_manifest
from osprey.simulation.engine import SimulationEngine

DEFAULT_DATA_DIR = "/data/simulation"
ENGINE_POLL_INTERVAL_S = 1.0


def main() -> None:
    data_dir = Path(os.environ.get("VA_DATA_DIR", DEFAULT_DATA_DIR))
    machine_path = data_dir / "machine.json"
    if not machine_path.is_file():
        raise SystemExit(
            f"FATAL: no machine.json at {machine_path}. "
            f"Bind-mount a project's data/simulation/ DIRECTORY (never a single "
            f"file) to {DEFAULT_DATA_DIR}, or set VA_DATA_DIR -- see README.md."
        )

    print(f"Building channel manifest and IOC records (data dir: {data_dir}) ...", flush=True)
    channels = build_manifest()["channels"]

    stuck_setpoints = frozenset(
        addr.strip() for addr in os.environ.get("VA_STUCK_SETPOINTS", "").split(",") if addr.strip()
    )
    if stuck_setpoints:
        print(f"VA apply-fault active: {sorted(stuck_setpoints)}", flush=True)

    bridge = PhysicsBridge()
    records = build_records(
        channels, on_pyat_setpoint=bridge.on_setpoint, stuck_setpoints=stuck_setpoints
    )
    bridge.bind(records.pyat_coupled)

    print(f"Loading simulation engine from {machine_path} ...", flush=True)
    engine = SimulationEngine.from_file(machine_path)
    engine_source = EngineSource(engine, channels, records.static_noisy, data_dir)

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
