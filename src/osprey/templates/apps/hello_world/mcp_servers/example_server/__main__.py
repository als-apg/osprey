"""Stdio entry point: ``python -m example_server``."""

from __future__ import annotations

import sys
import time

_started = time.perf_counter()

from .server import mcp  # noqa: E402 — the import is the cost being measured

if __name__ == "__main__":
    # Say on stderr that the frames can start, the way every OSPREY-framework
    # server does. Until this line a launcher cannot tell a server still
    # importing from one that is stuck: both are silence on stdout. Stdlib
    # only, so this package keeps depending on nothing but ``fastmcp``.
    print(
        f"[STARTUP-TIMING] example_server | total_startup: "
        f"{(time.perf_counter() - _started) * 1000:.0f}ms",
        file=sys.stderr,
        flush=True,
    )
    mcp.run()
