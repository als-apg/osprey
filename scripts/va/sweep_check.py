#!/usr/bin/env python3
"""Full-namespace Channel Access reachability sweep against a running VA container.

Batched, single-shared-timeout bulk read of every channel the simulator view
lists -- the namespace the container serves. Never reads channels one at a
time: every address is connected concurrently through aioca under one shared
deadline, and every connected address is then read back concurrently too. Both
batches run on the process's one Channel Access loop (``get_ca_loop()``, the
loop the connector uses), so an in-process caller shares its channels and its
libca context rather than starting a second of either.

Usable two ways:

* As a script, from a built project's root against its running container::

      export EPICS_CA_NAME_SERVERS=localhost:5064
      export EPICS_CA_AUTO_ADDR_LIST=NO
      python scripts/va/sweep_check.py [build/data/simulator]

* Imported, so ``tests/va/e2e/test_full_sweep.py`` can drive the exact same
  sweep function against its own container fixture without duplicating the
  bulk-read logic.

Never imports the CA *server* stack: pcaspy's extension carries a second copy
of the ``ca_*`` client symbols, and this process (like any of this suite's CA
clients) must keep its calls bound to the libca aioca loads.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from osprey_connectors.simulation.view import VIEW_RELPATH, SimulatorView

#: The simulator view of a built project, relative to its root.
DEFAULT_VIEW_DIR = Path("build") / VIEW_RELPATH


def served_addresses(view_dir: Path = DEFAULT_VIEW_DIR) -> list[str]:
    """Every channel address the simulator view in ``view_dir`` lists, sorted."""
    return list(SimulatorView.open(view_dir).channels())


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
        timeout: Shared wall-clock budget (seconds) for every PV to connect.
            Not a per-channel timeout -- one deadline for the whole batch.
        value_timeout: Shared timeout for the read-back of every connected
            address (fast; each channel is already connected, so this only
            guards against a stuck get).
    """
    import aioca

    from osprey_connectors.control_system.ca_loop import get_ca_loop

    async def connect_then_read() -> tuple[list[str], list[str]]:
        # One connect per address, all in flight at once against one absolute
        # deadline (aioca's ``(deadline,)`` form): a budget for the whole
        # batch, not one per channel.
        deadline = (time.time() + timeout,)
        connected = await aioca.connect(addresses, timeout=deadline, throw=False)
        missing_connect = [
            addr for addr, done in zip(addresses, connected, strict=True) if not done.ok
        ]
        reachable = [addr for addr, done in zip(addresses, connected, strict=True) if done.ok]
        values = await aioca.caget(reachable, timeout=value_timeout, throw=False)
        missing_value = [
            addr
            for addr, value in zip(reachable, values, strict=True)
            if isinstance(value, aioca.CANothing)
        ]
        return missing_connect, missing_value

    start = time.monotonic()
    # No subscriptions are opened, so nothing is left for teardown to close:
    # a sweep leaves only cached channels behind, which aioca purges at exit.
    missing_connect, missing_value = get_ca_loop().run_sync(
        connect_then_read(), timeout=timeout + value_timeout + 10.0
    )
    elapsed = time.monotonic() - start
    return SweepResult(
        total=len(addresses),
        connected=len(addresses) - len(missing_connect),
        missing_connect=sorted(missing_connect),
        missing_value=sorted(missing_value),
        elapsed_s=elapsed,
    )


def main(argv: list[str]) -> int:
    os.environ.setdefault("EPICS_CA_NAME_SERVERS", "localhost:5064")
    os.environ.setdefault("EPICS_CA_AUTO_ADDR_LIST", "NO")

    source = Path(argv[0]) if argv else DEFAULT_VIEW_DIR
    addresses = served_addresses(source)
    print(f"Sweeping {len(addresses)} addresses from {source} ...")
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
    code = main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    # os._exit rather than sys.exit: with the name server unreachable, aioca's
    # atexit teardown holds the interpreter for ~30 s after the verdict.
    os._exit(code)
