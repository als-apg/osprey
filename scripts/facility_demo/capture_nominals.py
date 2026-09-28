#!/usr/bin/env python3
"""Freeze the nominal value of every demo address, once per serving substrate.

Two substrates serve the control-assistant demo addresses, and each gets its own
golden file:

``mock``
    The in-process mock connector. For an address the simulation engine serves,
    the nominal is the engine's effective value (write > scenario override >
    baseline, expressions evaluated), which excludes the texture and both noise
    draws a live read adds; sigma is the machine file's relative ``noise`` scaled
    by the nominal and its absolute ``noise_abs``, combined in quadrature. For
    every other address the mock falls back to the procedural taxonomy: the
    nominal is the kind's base value and sigma is the kind's ``noise_sigma`` at
    the mock's noise level. Written as ``{address: {nominal, sigma}}``.

``va``
    A virtual-accelerator container, booted on a staged copy of the demo data
    tree whose machine file declares no ``texture``, ``noise`` or ``noise_abs``
    on any channel, with ``VA_NOISE_LEVEL=0`` for the procedural addresses and
    no ``VA_BPM_ERRORS``. Every address is read over Channel Access until two
    snapshots taken a settle interval apart agree exactly. Written as
    ``{address: {nominal}}``.

Each file carries a ``header`` naming the command that reproduces it and the
noise settings it was captured under, and a top-level ``_reproduce`` line with
the same command. ``--check PATH`` recomputes the document
under the same settings and byte-compares it with ``PATH``.

The engine is built on a fresh, empty scenario state directory, so the active
scenario set of whatever host runs this cannot shift a nominal.

Usage::

    uv run python scripts/facility_demo/capture_nominals.py --substrate mock \\
        --out tests/facility/golden/nominal_mock.json
    uv run python scripts/facility_demo/capture_nominals.py --substrate mock \\
        --check tests/facility/golden/nominal_mock.json
    uv run python scripts/facility_demo/capture_nominals.py --substrate va \\
        --out tests/facility/golden/nominal_va.json

The ``va`` substrate needs a container runtime and the virtual-accelerator image
(``--image``, built by ``scripts/va/build_and_boot_check.sh``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESET_DATA = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data"
PRESET_SIM_DIR = PRESET_DATA / "simulation"
MACHINE_JSON = PRESET_SIM_DIR / "machine.json"
LIMITS_JSON = PRESET_DATA / "channel_limits.json"
MANIFEST_JSON = REPO_ROOT / "src/osprey/services/virtual_accelerator/manifest/channel_manifest.json"

#: The mock connector's noise level when the preset sets none, which the
#: control-assistant preset does not.
MOCK_NOISE_LEVEL = 0.01

#: Machine-file keys that move a reading on its own: slow drift, relative
#: noise, absolute noise.
MOTION_KEYS = ("texture", "noise", "noise_abs")

DEFAULT_IMAGE = "osprey-va-full:latest"
CONTAINER_CA_PORT = 5064
READY_LOG_MARKER = "virtual accelerator IOC serving PVs"
BOOT_TIMEOUT_S = 240.0
CONNECT_TIMEOUT_S = 120.0
#: Longer than the telemetry thread's republish interval, so two equal
#: snapshots straddle at least one republish.
SETTLE_S = 3.0
MAX_SNAPSHOTS = 20
#: Reads of one address before a missing value fails the capture.
READ_ATTEMPTS = 3


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _command(substrate: str, image: str) -> str:
    parts = [
        "uv run python scripts/facility_demo/capture_nominals.py",
        f"--substrate {substrate}",
    ]
    if substrate == "va" and image != DEFAULT_IMAGE:
        parts.append(f"--image {image}")
    parts.append(f"--out tests/facility/golden/nominal_{substrate}.json")
    return " ".join(parts)


def demo_addresses() -> list[str]:
    """Every address the demo serves, sorted."""
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    return sorted({entry["address"] for entry in manifest["channels"]})


def capture_mock(addresses: list[str]) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Nominal and sigma per address from today's mock engine and taxonomy."""
    from osprey_connectors.channel_taxonomy import classify_channel
    from osprey_connectors.simulation.engine import SimulationEngine, engine_serves

    channels: dict[str, dict[str, float]] = {}
    served = 0
    with tempfile.TemporaryDirectory(prefix="capture-nominals-state-") as state_dir:
        engine = SimulationEngine.from_file(MACHINE_JSON, state_dir=state_dir)
        unknown = sorted(set(engine._channels) - set(addresses))
        if unknown:
            raise SystemExit(f"machine file channels outside the address set: {unknown[:5]}")
        for address in addresses:
            if engine_serves(engine, address):
                served += 1
                effective = engine._effective(address)
                if isinstance(effective, str):
                    raise SystemExit(f"{address}: string channel has no numeric nominal")
                nominal = float(effective)
                channel = engine._channels[address]
                sigma = math.sqrt((abs(nominal) * channel.noise) ** 2 + channel.noise_abs**2)
            else:
                kind = classify_channel(address)
                nominal = float(kind.base_value)
                sigma = float(kind.noise_sigma(nominal, MOCK_NOISE_LEVEL))
            channels[address] = {"nominal": nominal, "sigma": sigma}
    header = {
        "command": _command("mock", DEFAULT_IMAGE),
        "substrate": "mock",
        "addresses": {"source": _rel(MANIFEST_JSON), "count": len(addresses)},
        "machine_file": _rel(MACHINE_JSON),
        "active_scenarios": [],
        "engine_served": served,
        "noise_level": MOCK_NOISE_LEVEL,
        "nominal": (
            "engine-served: the engine's effective value, without texture or noise; "
            "others: classify_channel(address).base_value"
        ),
        "sigma": (
            "engine-served: sqrt((abs(nominal) * noise)**2 + noise_abs**2); "
            "others: classify_channel(address).noise_sigma(nominal, noise_level)"
        ),
    }
    return header, channels


def _stage_still_tree(root: Path) -> list[str]:
    """Stage the demo data tree under ``root`` with every channel's motion removed."""
    served = root / "simulation"
    shutil.copytree(PRESET_SIM_DIR, served)
    shutil.copy2(MANIFEST_JSON, served / "channel_manifest.json")
    shutil.copy2(LIMITS_JSON, served / "channel_limits.json")
    shutil.copy2(LIMITS_JSON, root / "channel_limits.json")
    machine_path = served / "machine.json"
    machine = json.loads(machine_path.read_text(encoding="utf-8"))
    stripped: set[str] = set()
    for entry in machine["channels"].values():
        for key in MOTION_KEYS:
            if key in entry:
                del entry[key]
                stripped.add(key)
    machine_path.write_text(json.dumps(machine, indent=2) + "\n", encoding="utf-8")
    return sorted(stripped)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_ready(runtime: str, name: str) -> None:
    deadline = time.monotonic() + BOOT_TIMEOUT_S
    while time.monotonic() < deadline:
        logs = subprocess.run(
            [runtime, "logs", name], capture_output=True, text=True, timeout=30, check=False
        )
        if READY_LOG_MARKER in logs.stdout + logs.stderr:
            return
        state = subprocess.run(
            [runtime, "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if state.stdout.strip() != "true":
            raise SystemExit(f"container exited before serving:\n{logs.stdout}\n{logs.stderr}")
        time.sleep(1.0)
    raise SystemExit(f"container not serving after {BOOT_TIMEOUT_S:.0f} s")


def _read_stable(addresses: list[str]) -> dict[str, float]:
    """Read every address until two snapshots a settle interval apart agree."""
    import epics

    pvs = {address: epics.PV(address, auto_monitor=False) for address in addresses}
    try:
        deadline = time.monotonic() + CONNECT_TIMEOUT_S
        pending = set(addresses)
        while pending and time.monotonic() < deadline:
            pending = {address for address in pending if not pvs[address].connected}
            if pending:
                time.sleep(0.1)
        if pending:
            raise SystemExit(f"{len(pending)} addresses never connected: {sorted(pending)[:5]}")

        def snapshot() -> dict[str, float]:
            values: dict[str, float] = {}
            for address in addresses:
                value = None
                for _ in range(READ_ATTEMPTS):
                    value = pvs[address].get(timeout=10.0, use_monitor=False)
                    if value is not None:
                        break
                if value is None:
                    raise SystemExit(f"{address}: no value after {READ_ATTEMPTS} reads")
                values[address] = float(value)
            return values

        before: dict[str, float] = {}
        previous = snapshot()
        for _ in range(MAX_SNAPSHOTS):
            time.sleep(SETTLE_S)
            current = snapshot()
            if current == previous:
                return current
            before, previous = previous, current
        moving = sorted(a for a in addresses if before[a] != previous[a])
        raise SystemExit(f"{len(moving)} addresses never settled: {moving[:5]}")
    finally:
        for pv in pvs.values():
            try:
                pv.disconnect()
            except Exception:
                # Teardown must not mask the capture's outcome.
                pass


def capture_va(
    addresses: list[str], image: str, runtime: str
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Nominal per address from a virtual-accelerator container serving a still tree."""
    port = _free_port()
    name = f"capture-nominals-{os.getpid()}"
    env_settings = {
        "VA_NOISE_LEVEL": "0",
        "VA_BPM_ERRORS": "",
        "VA_CHANNELS_FILE": "channel_manifest.json",
        "VA_LATTICE": "lattice.json",
    }
    with tempfile.TemporaryDirectory(prefix="capture-nominals-va-") as tmp:
        root = Path(tmp) / "data"
        root.mkdir()
        state = Path(tmp) / "state"
        state.mkdir()
        stripped = _stage_still_tree(root)
        run = [
            runtime,
            "run",
            "-d",
            "--name",
            name,
            "-p",
            f"127.0.0.1:{port}:{CONTAINER_CA_PORT}/tcp",
            "-v",
            f"{root}:/data:ro",
            "-e",
            "VA_DATA_DIR=/data/simulation",
            "-v",
            f"{state}:/state/simulation:ro",
            "-e",
            "VA_STATE_DIR=/state/simulation",
        ]
        for key, value in env_settings.items():
            run += ["-e", f"{key}={value}"]
        run.append(image)
        subprocess.run([runtime, "rm", "-f", name], capture_output=True, timeout=30, check=False)
        started = subprocess.run(run, capture_output=True, text=True, timeout=60, check=False)
        if started.returncode != 0:
            raise SystemExit(f"{runtime} run failed:\n{started.stdout}\n{started.stderr}")
        try:
            _wait_for_ready(runtime, name)
            os.environ["EPICS_CA_NAME_SERVERS"] = f"127.0.0.1:{port}"
            os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
            os.environ.pop("EPICS_CA_ADDR_LIST", None)
            os.environ.pop("EPICS_CA_SERVER_PORT", None)
            values = _read_stable(addresses)
        finally:
            subprocess.run(
                [runtime, "rm", "-f", name], capture_output=True, timeout=60, check=False
            )
    header = {
        "command": _command("va", image),
        "substrate": "va",
        "addresses": {"source": _rel(MANIFEST_JSON), "count": len(addresses)},
        "machine_file": _rel(MACHINE_JSON),
        "active_scenarios": [],
        "stripped_keys": stripped,
        "environment": env_settings,
        "nominal": (
            f"Channel Access read, repeated every {SETTLE_S:g} s until two snapshots agree exactly"
        ),
    }
    return header, {address: {"nominal": values[address]} for address in addresses}


def render(header: dict[str, Any], channels: dict[str, dict[str, float]]) -> str:
    """The golden document, byte-stable for equal inputs."""
    document = {
        "_reproduce": header["command"],
        "header": header,
        "channels": {a: channels[a] for a in sorted(channels)},
    }
    return json.dumps(document, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--substrate", choices=("mock", "va"), required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--out", type=Path, help="write the golden here")
    target.add_argument("--check", type=Path, help="byte-compare a fresh capture with this file")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="virtual-accelerator image")
    parser.add_argument("--runtime", default="docker", help="container runtime")
    args = parser.parse_args(argv)

    addresses = demo_addresses()
    if args.substrate == "mock":
        header, channels = capture_mock(addresses)
    else:
        header, channels = capture_va(addresses, args.image, args.runtime)
    text = render(header, channels)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {len(channels)} addresses to {args.out}")
        return 0

    frozen = args.check.read_text(encoding="utf-8")
    if frozen == text:
        print(f"{args.check}: {len(channels)} addresses match")
        return 0
    try:
        frozen_document = json.loads(frozen)
    except json.JSONDecodeError:
        frozen_document = {}
    frozen_header = frozen_document.get("header", {})
    frozen_channels = frozen_document.get("channels", {})
    header_keys = sorted(
        k for k in set(header) | set(frozen_header) if header.get(k) != frozen_header.get(k)
    )
    differing = sorted(
        a for a in set(channels) | set(frozen_channels) if channels.get(a) != frozen_channels.get(a)
    )
    print(
        f"{args.check}: differs from a fresh capture; "
        f"{len(header_keys)} header keys and {len(differing)} addresses differ"
    )
    for key in header_keys:
        print(f"  header.{key}: golden {frozen_header.get(key)!r} now {header.get(key)!r}")
    for address in differing[:20]:
        print(f"  {address}: golden {frozen_channels.get(address)} now {channels.get(address)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
