#!/usr/bin/env python3
"""Phase-1 hard gate: SC1/SC2 -- drive a real read_channel + verified write_channel
round trip against the toy probe IOC
(src/osprey/services/virtual_accelerator/probe) through the UNMODIFIED,
production `EPICSConnector`, via `ConnectorFactory`.

This script makes NO changes to src/osprey/. It only:
  1. Starts a container from the probe image built by task 1.1 (building it first
     if the image isn't present yet -- reads the existing, unmodified Containerfile).
  2. Writes a throwaway scratch config (a tempfile, never under the repo) that sets
     control_system.writes_enabled = true, so EPICSConnector routes through its
     write_access gateway path.
  3. Uses ConnectorFactory.create_control_system_connector() to build a real
     EPICSConnector pointed at the container via CA name-server mode (the one
     configuration task 1.1 found to be portable across container runtimes).
  4. Calls connector.read_channel() and connector.write_channel(..., verification_level
     ="readback") -- both real CA operations, no mocking.

Exit code 0 = round trip verified. Any failure exits non-zero and this is a hard
gate: per the task, a failure here is a finding that re-scopes downstream work,
not something to silently patch around.

KNOWN FINDING -- libca teardown assertion in pure name-server (TCP-only) mode:
when the probe container is torn down right after EPICSConnector.disconnect(),
pyepics's underlying libca occasionally hits `assert(this->pudpiiu)` in
cac.cpp (thread 'CAC-TCP-send') and suspends that thread instead of exiting.
This looks like a real libca quirk specific to EPICS_CA_NAME_SERVERS-only
operation (no UDP search/beacon IIU exists in that mode, and this teardown
path assumes one does) rather than anything wrong with EPICSConnector or this
script's use of it -- the round trip itself (read/write/verify) completes and
passes before the crash, and it only shows up during teardown of the CA
circuit, not during normal operation. Because it leaves a non-daemon-like
suspended C thread behind, the interpreter's normal shutdown sequence hangs
forever waiting to join it. This script routes around that by calling
`os._exit()` right after computing the pass/fail result (all our own cleanup
-- container removal, scratch config deletion -- has already run by then), so
the gate still exits promptly and correctly instead of hanging. This is
reported upstream as a finding, not patched around in src/osprey/.
"""

import asyncio
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_DIR = REPO_ROOT / "src" / "osprey" / "services" / "virtual_accelerator" / "probe"
IMAGE = "osprey-va-probe:latest"
CONTAINER = "osprey-va-probe-roundtrip"
LABEL = "io.osprey=va-probe"
CA_PORT = 5064
BOOT_TIMEOUT_SECS = 20


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _runtime_available(runtime: str) -> bool:
    """Health-check: can this runtime actually create a new container?

    Podman on this host has previously shown a broken storage layer that lets
    `info` succeed but fails every new container's overlay mount. A real,
    uniquely-named, immediately-removed container run is the only reliable
    check. Never a filter-based sweep -- exact name only, per containment
    rules: operate only on resources this script itself creates.
    """
    if _run(["which", runtime]).returncode != 0:
        return False
    probe_name = f"osprey-va-probe-healthcheck-{os.getpid()}"
    result = _run(
        [runtime, "run", "--rm", "--name", probe_name, "--label", LABEL, "alpine:latest", "true"]
    )
    return result.returncode == 0


def _select_runtime() -> str:
    override = os.environ.get("OSPREY_VA_RUNTIME")
    if override:
        return override
    if _runtime_available("podman"):
        return "podman"
    if _runtime_available("docker"):
        return "docker"
    print("FATAL: neither a working podman nor docker found on PATH", file=sys.stderr)
    sys.exit(1)


def _ensure_image(runtime: str) -> None:
    check = _run([runtime, "image", "inspect", IMAGE])
    if check.returncode == 0:
        return
    print(f"--- Image {IMAGE} not found, building from {PROBE_DIR} ---")
    build = subprocess.run(
        [
            runtime,
            "build",
            "-t",
            IMAGE,
            "-f",
            str(PROBE_DIR / "Containerfile"),
            str(PROBE_DIR),
        ]
    )
    if build.returncode != 0:
        print("FATAL: failed to build probe image", file=sys.stderr)
        sys.exit(1)


def _start_container(runtime: str) -> None:
    # Idempotency / containment: remove only this script's own exact-named
    # container, never a broader sweep.
    _run([runtime, "rm", "-f", CONTAINER])
    result = subprocess.run(
        [
            runtime,
            "run",
            "-d",
            "--name",
            CONTAINER,
            "--label",
            LABEL,
            "-p",
            f"127.0.0.1:{CA_PORT}:{CA_PORT}/tcp",
            "-p",
            f"127.0.0.1:{CA_PORT}:{CA_PORT}/udp",
            IMAGE,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"FATAL: failed to start probe container: {result.stderr}", file=sys.stderr)
        sys.exit(1)


def _wait_for_boot(runtime: str) -> None:
    deadline = time.monotonic() + BOOT_TIMEOUT_SECS
    while time.monotonic() < deadline:
        logs = _run([runtime, "logs", CONTAINER])
        if "probe IOC serving PVs" in (logs.stdout + logs.stderr):
            return
        time.sleep(0.5)
    logs = _run([runtime, "logs", CONTAINER])
    print(f"FATAL: IOC did not report ready within {BOOT_TIMEOUT_SECS}s", file=sys.stderr)
    print(logs.stdout, logs.stderr, file=sys.stderr)
    sys.exit(1)


def _write_scratch_config() -> str:
    """Write a throwaway config.yml to a tempfile (never under the repo) that
    sets control_system.writes_enabled = true, so EPICSConnector.connect()
    routes through the write_access gateway rather than assuming read-only."""
    fd, path = tempfile.mkstemp(prefix="osprey-va-probe-roundtrip-", suffix=".yml")
    with os.fdopen(fd, "w") as f:
        f.write("control_system:\n  writes_enabled: true\n")
    return path


async def _run_roundtrip() -> bool:
    from osprey.connectors.factory import ConnectorFactory, register_builtin_connectors

    register_builtin_connectors()

    connector_config = {
        "type": "epics",
        "connector": {
            "epics": {
                "timeout": 10.0,
                "gateways": {
                    "read_only": {
                        "address": "localhost",
                        "port": CA_PORT,
                        "use_name_server": True,
                    },
                    "write_access": {
                        "address": "localhost",
                        "port": CA_PORT,
                        "use_name_server": True,
                    },
                },
            }
        },
    }

    connector = await ConnectorFactory.create_control_system_connector(connector_config)
    resolved_class = f"{type(connector).__module__}.{type(connector).__name__}"
    print(f"ConnectorFactory resolved connector class: {resolved_class}")
    expected_class = "osprey.connectors.control_system.epics_connector.EPICSConnector"
    if resolved_class != expected_class:
        print(
            f"FATAL: expected {expected_class}, ConnectorFactory resolved {resolved_class} "
            "instead -- the round trip would not be exercising the real EPICS stack",
            file=sys.stderr,
        )
        return False
    try:
        # SC1: read_channel against the real probe IOC.
        baseline = await connector.read_channel("PROBE:BPM:POSITION:X")
        print(f"read_channel PROBE:BPM:POSITION:X -> {baseline.value}")

        # SC2: verified write_channel -- write SP, EPICSConnector reads it back
        # internally and confirms within tolerance.
        write_value = 75.0
        result = await connector.write_channel(
            "PROBE:HCM:CURRENT:SP",
            write_value,
            verification_level="readback",
            tolerance=0.5,
        )
        print(
            f"write_channel PROBE:HCM:CURRENT:SP = {write_value} -> "
            f"success={result.success}, verified={result.verification.verified}, "
            f"readback={result.verification.readback_value}"
        )

        if not result.success or not result.verification or not result.verification.verified:
            print(f"FATAL: write_channel did not verify: {result}", file=sys.stderr)
            return False

        # Bonus independent confirmation (beyond the minimum SC1/SC2 gate):
        # read the RB and BPM PVs the SP write should have driven, proving the
        # write actually reached the IOC and its physics side effects, not
        # just that write_channel's own readback-of-SP round-tripped.
        rb = await connector.read_channel("PROBE:HCM:CURRENT:RB")
        bpm = await connector.read_channel("PROBE:BPM:POSITION:X")
        print(f"read_channel PROBE:HCM:CURRENT:RB -> {rb.value}")
        print(f"read_channel PROBE:BPM:POSITION:X (post-write) -> {bpm.value}")

        expected_bpm = write_value * 1e-4 * 2.0 * 1000.0  # kick_rad * drift_m * mm/m
        if abs(rb.value - write_value) > 0.5:
            print(
                f"FATAL: RB did not reflect the write: {rb.value} != {write_value}", file=sys.stderr
            )
            return False
        if abs(bpm.value - expected_bpm) > 0.5:
            print(
                f"FATAL: BPM did not reflect the physics side effect: "
                f"{bpm.value} != {expected_bpm} (expected)",
                file=sys.stderr,
            )
            return False

        return True
    finally:
        await connector.disconnect()


def main() -> int:
    runtime = _select_runtime()
    print(f"Using container runtime: {runtime}")

    scratch_config_path = _write_scratch_config()
    print(f"Scratch config (not in repo): {scratch_config_path}")
    os.environ["CONFIG_FILE"] = scratch_config_path

    try:
        _ensure_image(runtime)
        _start_container(runtime)

        print(f"--- Waiting up to {BOOT_TIMEOUT_SECS}s for PVs to serve ---")
        _wait_for_boot(runtime)

        ok = asyncio.run(_run_roundtrip())
        if ok:
            print("--- Gate PASSED: read_channel + verified write_channel round trip confirmed ---")
            return 0
        else:
            print("--- Gate FAILED: round trip did not verify ---", file=sys.stderr)
            return 1
    finally:
        _run([runtime, "rm", "-f", CONTAINER])
        try:
            os.remove(scratch_config_path)
        except OSError:
            pass


if __name__ == "__main__":
    _code = main()
    # See the module docstring's "KNOWN FINDING": a libca teardown assertion
    # can leave a suspended background thread that hangs normal interpreter
    # shutdown. All of this script's own cleanup has already run by this
    # point, so force-exit rather than risk hanging the gate indefinitely.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
