#!/usr/bin/env python3
"""Full-namespace Channel Access reachability sweep against a running VA container.

Batched, single-shared-timeout bulk read of every address in the namespace-union
manifest (``docker/virtual-accelerator/manifest``) -- the same manifest baked
into the ``osprey-va-full`` image. Never reads channels one at a time: every
address gets its own ``epics.PV`` (auto-monitoring) up front so connections
happen concurrently, then one shared deadline is used to wait for the whole
set to connect before reading values back.

Usable two ways:

* As a script, against a container already published on the host (see
  ``scripts/va/run_va.sh``)::

      export EPICS_CA_NAME_SERVERS=localhost:5064
      export EPICS_CA_AUTO_ADDR_LIST=NO
      python scripts/va/sweep_check.py

* Imported, so ``tests/va/e2e/test_full_sweep.py`` can drive the exact same
  sweep function against its own container fixture without duplicating the
  bulk-read logic.

Never imports ``softioc``/``ioc.records`` -- this process (like any of this
suite's CA clients) must stay eligible to act as a pyepics Channel Access
client (see the poisoning caveat documented in ``tests/va/test_record_factory.py``).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VA_PARENT = _REPO_ROOT / "docker" / "virtual-accelerator"
if str(_VA_PARENT) not in sys.path:
    sys.path.insert(0, str(_VA_PARENT))


def all_manifest_addresses() -> list[str]:
    """Return every address in the namespace-union manifest (softioc-free import)."""
    from manifest import build_manifest

    return [c["address"] for c in build_manifest()["channels"]]


@dataclass
class SweepResult:
    total: int
    connected: int
    missing_connect: list[str] = field(default_factory=list)
    missing_value: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.missing_connect and not self.missing_value


def sweep(addresses: list[str], *, timeout: float = 45.0, value_timeout: float = 5.0) -> SweepResult:
    """Bulk-read every address in ``addresses`` under one shared connect deadline.

    Args:
        addresses: PV addresses to read.
        timeout: Shared wall-clock budget (seconds) for every PV to connect.
            Not a per-channel timeout -- one deadline for the whole batch.
        value_timeout: Per-PV ``get()`` timeout once connected (fast; the PV
            is already monitoring, so this only guards against a stuck get).
    """
    import epics

    start = time.monotonic()
    pvs = {addr: epics.PV(addr, auto_monitor=True) for addr in addresses}

    deadline = start + timeout
    pending = set(pvs)
    while pending and time.monotonic() < deadline:
        pending = {addr for addr in pending if not pvs[addr].connected}
        if pending:
            time.sleep(0.05)

    missing_connect = sorted(addr for addr in pvs if not pvs[addr].connected)

    missing_value = []
    for addr, pv in pvs.items():
        if addr in missing_connect:
            continue
        value = pv.get(timeout=value_timeout)
        if value is None:
            missing_value.append(addr)

    elapsed = time.monotonic() - start
    return SweepResult(
        total=len(addresses),
        connected=len(pvs) - len(missing_connect),
        missing_connect=missing_connect,
        missing_value=sorted(missing_value),
        elapsed_s=elapsed,
    )


def main() -> int:
    import os

    os.environ.setdefault("EPICS_CA_NAME_SERVERS", "localhost:5064")
    os.environ.setdefault("EPICS_CA_AUTO_ADDR_LIST", "NO")

    addresses = all_manifest_addresses()
    print(f"Sweeping {len(addresses)} manifest addresses ...")
    result = sweep(addresses)

    print(f"Connected: {result.connected}/{result.total} in {result.elapsed_s:.1f}s")
    if result.missing_connect:
        print(f"MISSING (never connected): {len(result.missing_connect)}")
        for addr in result.missing_connect:
            print(f"  - {addr}")
    if result.missing_value:
        print(f"MISSING (connected, no value): {len(result.missing_value)}")
        for addr in result.missing_value:
            print(f"  - {addr}")

    if result.ok:
        print("OK: full namespace reachable.")
        return 0
    print("FAIL: see missing addresses above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
