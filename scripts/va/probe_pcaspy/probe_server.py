"""Minimal pcaspy portable-CAS probe server.

Serves five PVs that exercise exactly the three Channel Access server
behaviours the LUME serving layer depends on:

- ``SYNC:SP`` / ``SYNC:RB``    -- plain synchronous write + echo (transport).
- ``ASYNC:SP`` / ``ASYNC:RB``  -- an ``asyn: True`` PV whose value is committed
  only after a deliberate delay, then completed with ``callbackPV``. A client
  using put-completion must stay blocked for that whole delay, and must observe
  the committed value the instant it unblocks.
- ``TELEM:COUNTER``            -- bumped by a server-owned thread on its own
  schedule, with no client write involved, so a host ``camonitor``
  subscription can be checked for server-initiated monitor events.

No physics here: this is a transport/behaviour probe, not a model. See
README.md for what is being proven and why.
"""

from __future__ import annotations

import os
import threading
import time

from pcaspy import Driver, SimpleServer

PREFIX = os.environ.get("PROBE_PV_PREFIX", "PCASPY:")
# How long the asynchronous write withholds completion. Must be long enough
# that a client blocking for it is unambiguously distinguishable from ordinary
# round-trip latency.
ASYNC_DELAY_S = float(os.environ.get("PROBE_ASYNC_DELAY_S", "2.0"))
TELEM_PERIOD_S = float(os.environ.get("PROBE_TELEM_PERIOD_S", "0.5"))

pvdb = {
    "SYNC:SP": {"type": "float", "prec": 4},
    "SYNC:RB": {"type": "float", "prec": 4},
    "ASYNC:SP": {"type": "float", "prec": 4, "asyn": True},
    "ASYNC:RB": {"type": "float", "prec": 4},
    "TELEM:COUNTER": {"type": "float", "prec": 3},
}


class ProbeDriver(Driver):
    """Driver with one synchronous PV, one asynchronous PV, and a telemetry loop."""

    def __init__(self) -> None:
        super().__init__()
        self._telem = 0.0
        threading.Thread(target=self._telemetry_loop, daemon=True).start()

    def write(self, reason, value):  # noqa: D102 - pcaspy Driver contract
        if reason == "ASYNC:SP":
            # Deliberately do NOT commit here. The value lands, and completion
            # is signalled, only from the delayed worker below -- the same
            # shape as "a setpoint write must not complete until the physics
            # solve has committed".
            threading.Thread(target=self._delayed_commit, args=(reason, value), daemon=True).start()
            return True
        if reason == "SYNC:SP":
            self.setParam(reason, value)
            self.setParam("SYNC:RB", value)
            self.updatePVs()
            return True
        # Readback / telemetry PVs are not writable.
        return False

    def _delayed_commit(self, reason: str, value: float) -> None:
        time.sleep(ASYNC_DELAY_S)
        # Commit *before* signalling completion, so a client that unblocks on
        # put-completion is guaranteed to read the committed value.
        self.setParam(reason, value)
        self.setParam("ASYNC:RB", value)
        self.updatePVs()
        self.callbackPV(reason)

    def _telemetry_loop(self) -> None:
        while True:
            time.sleep(TELEM_PERIOD_S)
            self._telem += 1.0
            self.setParam("TELEM:COUNTER", self._telem)
            self.updatePVs()


def main() -> None:
    server = SimpleServer()
    server.createPV(PREFIX, pvdb)
    ProbeDriver()
    names = ", ".join(PREFIX + reason for reason in pvdb)
    print(
        f"pcaspy probe server serving PVs: {names} "
        f"(async delay {ASYNC_DELAY_S}s, telemetry period {TELEM_PERIOD_S}s)",
        flush=True,
    )
    while True:
        server.process(0.1)


if __name__ == "__main__":
    main()
