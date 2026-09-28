#!/usr/bin/env python3
"""Full-namespace Channel Access reachability sweep against a running VA container.

Batched, single-shared-timeout bulk read of every address in the namespace-union
manifest (``osprey.services.virtual_accelerator.manifest``) -- the same
manifest baked into the ``osprey-va-full`` image. Never reads channels one at
a time: every address gets its own pvapy ``pvaccess.Channel`` and an
asynchronous get issued up front, so connections and reads happen
concurrently, under one shared deadline for the whole set.

Call :func:`sweep` from a main thread (or a daemon thread that never exits):
on macOS a joined thread that touched pvapy hangs at interpreter exit.

Usable two ways:

* As a script, against a container already published on the host (see
  ``scripts/va/run_va.sh``)::

      export EPICS_CA_NAME_SERVERS=localhost:5064
      export EPICS_CA_AUTO_ADDR_LIST=NO
      python scripts/va/sweep_check.py

* Imported, so ``tests/va/e2e/test_full_sweep.py`` can drive the exact same
  sweep function against its own container fixture without duplicating the
  bulk-read logic.

Never imports the CA *server* stack -- this process (like any of this
suite's CA clients) must stay eligible to act as a Channel Access client (see
the poisoning caveat documented in
``scripts/va/probe_pcaspy/coexistence_check.py``).
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field


def all_manifest_addresses() -> list[str]:
    """Return every address in the namespace-union manifest (server-free import)."""
    from osprey.services.virtual_accelerator.manifest import build_manifest

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


def sweep(
    addresses: list[str], *, timeout: float = 45.0, value_timeout: float = 5.0
) -> SweepResult:
    """Bulk-read every address in ``addresses`` under one shared connect deadline.

    Args:
        addresses: PV addresses to read.
        timeout: Shared wall-clock budget (seconds) for every PV to connect and
            answer. Not a per-channel budget in practice: every get is issued
            up front, so they all run against the one deadline together.
        value_timeout: Extra grace (seconds) past ``timeout`` for a get whose
            channel connected but whose value has not arrived yet.
    """
    import pvaccess

    start = time.monotonic()
    answered: dict[str, bool] = {}
    lock = threading.Lock()

    def settle(address: str, ok: bool):
        def callback(*_: object) -> None:
            with lock:
                answered[address] = ok

        return callback

    channels = {}
    for address in addresses:
        channel = pvaccess.Channel(address, pvaccess.CA)
        # Bounds both the connect and the get: an address that never connects
        # errors out of its asyncGet once this elapses.
        channel.setTimeout(timeout)
        channel.asyncGet(settle(address, True), settle(address, False), "field(value)")
        channels[address] = channel

    deadline = start + timeout + value_timeout
    while time.monotonic() < deadline:
        with lock:
            if len(answered) == len(channels):
                break
        time.sleep(0.05)

    with lock:
        got = {address for address, ok in answered.items() if ok}
    missing_connect = sorted(
        address for address, channel in channels.items() if not channel.isConnected()
    )
    missing_value = [
        address for address in channels if address not in got and address not in missing_connect
    ]

    elapsed = time.monotonic() - start
    return SweepResult(
        total=len(addresses),
        connected=len(channels) - len(missing_connect),
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
