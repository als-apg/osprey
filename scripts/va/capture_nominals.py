#!/usr/bin/env python3
"""Capture the nominal value of every address the virtual accelerator serves for the demo.

The capture is ``tests/facility/golden/nominal_va.json``, the golden
``tests/va/e2e/test_substrate_parity.py`` holds the live container's served
nominals to. It is read off the wire, independently of the in-process
composite, so a regression in the composite moves the served values and not
the golden.

The project is staged as the live-container suite stages it
(``stage_va_project`` in ``tests/va/e2e/conftest.py``): the example facility's
simulator view, rendered with the suite's synthetic additions. Its state
directory starts empty but for ``nominal``, and ``osprey sim apply still``
then makes the active set ``nominal`` and ``still``, under which no captured
address moves; the capture asserts every one has a motion envelope of zero
before it reads. The container is booted as the suite boots it
(``boot_va_container``), from the image the suite runs:
``osprey-va-full:latest``, built by ``scripts/va/build_and_boot_check.sh``, or
the one ``OSPREY_VA_E2E_IMAGE`` names.

Every captured address is read over Channel Access until two snapshots taken a
settle interval apart agree exactly. Each value is written in the composite's
stored representation: a bool or enum as its label, a string as its text, a
number as read. Two kinds of served address are not captured: the suite's
synthetic string channel, which the demo does not carry, and every waveform,
whose trace is computed when it is read.

The document is ``{_reproduce, header, channels}``: ``_reproduce`` and
``header.command`` carry the command that writes it, ``header`` the settings it
was captured under, and ``channels`` maps each address, sorted, to
``{"nominal": value}``. ``--check PATH`` recaptures and byte-compares with
``PATH``, printing every header key and address that differs.

Usage::

    uv run python scripts/va/capture_nominals.py --out tests/facility/golden/nominal_va.json
    uv run python scripts/va/capture_nominals.py --check tests/facility/golden/nominal_va.json

Needs a container runtime (``docker``) and the image.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "scripts/va/capture_nominals.py"

#: The scenario the capture applies on top of ``nominal``.
STILL = "still"
#: Longer than the runner's tick, so two equal snapshots straddle at least one pass.
SETTLE_S = 3.0
MAX_SNAPSHOTS = 20
#: How long one read of every address may take, connection included.
READ_TIMEOUT_S = 120.0


def _command(path: Path) -> str:
    """The command that writes the capture to ``path``."""
    resolved = path.resolve()
    shown = resolved.relative_to(REPO_ROOT) if resolved.is_relative_to(REPO_ROOT) else resolved
    return f"uv run python {SCRIPT} --out {shown.as_posix()}"


async def _read_wire(addresses: list[str], timeout: float) -> dict[str, Any]:
    """Every address's value off the wire, as a JSON scalar or list."""
    import aioca

    results = await aioca.caget(addresses, timeout=timeout, throw=False)
    values: dict[str, Any] = {}
    missing: list[str] = []
    for address, value in zip(addresses, results, strict=True):
        if not value.ok:
            missing.append(address)
        elif hasattr(value, "tolist"):
            values[address] = value.tolist()
        elif isinstance(value, float):
            values[address] = float(value)
        elif isinstance(value, int):
            values[address] = int(value)
        else:
            values[address] = str(value)
    if missing:
        raise RuntimeError(f"no value for {missing[:10]} ({len(missing)} in all)")
    return values


def _snapshot(addresses: list[str]) -> dict[str, Any]:
    """One read of every address, taken in a child process.

    Channel Access runs out of process: libca's teardown blocks for as long as
    a name-server circuit retries a port nothing serves, which is this
    process's state once the container is gone. The child leaves through
    ``os._exit`` once its answer is written.
    """
    result = subprocess.run(
        [sys.executable, __file__, "--read"],
        input=json.dumps(addresses),
        capture_output=True,
        text=True,
        timeout=READ_TIMEOUT_S + 60.0,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise SystemExit(f"Channel Access read failed:\n{result.stdout}\n{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _read_stable(addresses: list[str]) -> dict[str, Any]:
    """Read every address until two snapshots a settle interval apart agree exactly."""
    before: dict[str, Any] = {}
    previous = _snapshot(addresses)
    for _ in range(MAX_SNAPSHOTS):
        time.sleep(SETTLE_S)
        current = _snapshot(addresses)
        if current == previous:
            return current
        before, previous = previous, current
    moving = sorted(address for address in addresses if before[address] != previous[address])
    raise SystemExit(f"{len(moving)} addresses never settled: {moving[:5]}")


def _stored(view: Any, address: str, raw: Any) -> Any:
    """A wire value in the composite's stored representation."""
    from osprey_connectors.simulation import decode_char_waveform
    from osprey_connectors.simulation.values import DEFAULT_BOOL_OPTIONS

    if address in set(view.status_addresses().values()):
        return decode_char_waveform(raw)
    channel = view.channel(address)
    value_type = str(channel.value_type or "float")
    if value_type in ("bool", "enum"):
        return list(channel.options or DEFAULT_BOOL_OPTIONS)[int(raw)]
    if value_type == "string":
        return decode_char_waveform(raw)
    if value_type == "int":
        return int(raw)
    return float(raw)


def capture() -> tuple[dict[str, Any], dict[str, Any]]:
    """The header and the captured nominal of every captured address."""
    sys.path.insert(0, str(REPO_ROOT))
    from tests.va.e2e import conftest as e2e

    from osprey_connectors.simulation.state import read_active_state, resolve_active_scenarios
    from osprey_connectors.simulation.values import is_waveform
    from osprey_connectors.simulation.view import VIEW_RELPATH, SimulatorView

    with tempfile.TemporaryDirectory(prefix="capture-nominals-") as scratch:
        project = e2e.stage_va_project(Path(scratch))
        applied = project.sim_apply(STILL)
        if applied.returncode != 0:
            raise SystemExit(
                f"osprey sim apply {STILL} failed:\n{applied.stdout}\n{applied.stderr}"
            )
        active = resolve_active_scenarios(read_active_state(project.state_dir)[0])
        view = SimulatorView.open(project.data_dir / VIEW_RELPATH.name)
        status = set(view.status_addresses().values())
        addresses = sorted(
            address
            for address in (*view.channels(), *status)
            if address != e2e.STRING_CHANNEL
            and (address in status or not is_waveform(view.channel(address).value_type))
        )
        moving = [address for address in addresses if view.motion_envelope(address, active)]
        if moving:
            raise SystemExit(f"{len(moving)} addresses move under {active}: {moving[:5]}")
        with e2e.boot_va_container(project):
            raw = _read_stable(addresses)
        channels = {
            address: {"nominal": _stored(view, address, raw[address])} for address in addresses
        }
    header = {
        "substrate": "va",
        "image": (
            "osprey-va-full:latest, built by scripts/va/build_and_boot_check.sh, "
            "or the one OSPREY_VA_E2E_IMAGE names"
        ),
        "render": (
            "the example facility's simulator view, staged as tests/va/e2e/conftest.py "
            "stage_va_project stages it"
        ),
        "active_scenarios": active,
        "addresses": {
            "count": len(addresses),
            "excluded": "the suite's synthetic string channel and every waveform",
        },
        "nominal": (
            f"Channel Access read, repeated every {SETTLE_S:g} s until two snapshots agree "
            "exactly; a bool or enum as its label, a string as its text"
        ),
    }
    return header, channels


def render(command: str, header: dict[str, Any], channels: dict[str, Any]) -> str:
    """The golden document, byte-stable for equal inputs."""
    document = {
        "_reproduce": command,
        "header": {"command": command, **header},
        "channels": {address: channels[address] for address in sorted(channels)},
    }
    return json.dumps(document, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--out", type=Path, help="write the capture here")
    target.add_argument("--check", type=Path, help="byte-compare a fresh capture with this file")
    args = parser.parse_args(argv)

    path = args.out if args.out is not None else args.check
    header, channels = capture()
    text = render(_command(path), header, channels)

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
    fresh = json.loads(text)
    frozen_header = frozen_document.get("header", {})
    frozen_channels = frozen_document.get("channels", {})
    header_keys = sorted(
        key
        for key in set(fresh["header"]) | set(frozen_header)
        if fresh["header"].get(key) != frozen_header.get(key)
    )
    differing = sorted(
        address
        for address in set(channels) | set(frozen_channels)
        if fresh["channels"].get(address) != frozen_channels.get(address)
    )
    print(
        f"{args.check}: differs from a fresh capture; "
        f"{len(header_keys)} header keys and {len(differing)} addresses differ"
    )
    for key in header_keys:
        print(f"  header.{key}: golden {frozen_header.get(key)!r} now {fresh['header'].get(key)!r}")
    for address in differing:
        print(
            f"  {address}: golden {frozen_channels.get(address)} "
            f"now {fresh['channels'].get(address)}"
        )
    return 1


if __name__ == "__main__" and sys.argv[1:] == ["--read"]:
    import asyncio

    try:
        values = asyncio.run(_read_wire(json.loads(sys.stdin.read()), READ_TIMEOUT_S))
        print(json.dumps(values), flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    os._exit(0)

if __name__ == "__main__":
    sys.exit(main())
