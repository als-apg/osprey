#!/usr/bin/env python3
"""Out-of-process host-side Channel Access op for the substrate-equivalence e2e.

Runs ONE ``connect -> (optional verified write) -> read`` sequence against the
co-deployed Virtual Accelerator through the real, unmodified
``VirtualAcceleratorConnector`` (built by the production ``ConnectorFactory``),
prints the result as one marker-prefixed JSON line, then force-exits.

Why a subprocess per proof (mirrors P2's ``sweep_check`` subprocess): repeated
CA-context create/destroy cycles in a single process eventually trip a libca
teardown assertion (``assert(this->pudpiiu)``, ``cac.cpp``, thread
``CAC-TCP-send``) in ``EPICS_CA_NAME_SERVERS``-only mode -- a known
pyepics/libca quirk on the CA-circuit teardown path, unrelated to the connector
or the substrate: the read/write round trip completes and is correct *before*
it fires (see ``scripts/va/probe_roundtrip.py``'s "KNOWN FINDING"). Isolating
each proof's host CA work in its own process -- which emits its result and
``os._exit()``s *before* any connector disconnect / CA teardown runs -- means
the fault can never recur in the pytest process.

Deliberately never calls ``connector.disconnect()`` and never lets the
interpreter shut down normally: the whole point is to skip the CA-circuit
teardown that carries the assertion. All CA work is finished once the values
are in hand; ``os._exit(0)`` hands the socket back to the OS untorn.

Protocol -- ``argv[1]`` is a JSON spec::

    {"connector_config": {...},          # the ConnectorFactory connector config
     "config_overrides": {...},          # osprey.utils.config.get_config_value overrides
     "read": "<address>",                # the readback address to return
     "write": {"address": "<addr>", "value": <float>} | null,
     "settle_read": <bool>}              # poll the read until it == write.value (sp-echo)

On success, emits exactly one stdout line::

    __HOST_CA_RESULT__{"read_value": <float>, "read_settled": <bool>,
                       "write_success": <bool|null>, "write_verified": <bool|null>}

``read_settled`` is ``true`` when ``settle_read`` was not requested, or when the
readback reached the written value within the settle deadline; ``false`` means
the asynchronous echo never propagated in time (the caller should fail loudly).

Exit 0 on success; a non-zero exit (or a native SIGBUS) means the host CA op
failed and the caller must surface it -- never silently pass.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any
from unittest.mock import patch

RESULT_MARKER = "__HOST_CA_RESULT__"

# sp-echo SP->RB propagation is asynchronous in the IOC (the setpoint record's
# on_update fires on the IOC's own loop AFTER the caput that write_channel
# verifies against the SP), so an immediate readback read can beat the echo.
# When a caller sets settle_read, poll the readback until it reflects the
# written value rather than guess a fixed delay (condition-based waiting).
_SETTLE_TOL = 1e-6
_SETTLE_TIMEOUT_SEC = 10.0
_SETTLE_POLL_SEC = 0.05


async def _do(spec: dict[str, Any]) -> dict[str, Any]:
    """Connect, optionally verify-write, read, and emit -- then force-exit.

    Uses the SAME config the in-process host connector used (same
    ``get_config_value`` overrides, same connector config), so this
    out-of-process read/write drives a REAL production connector -- isolated in
    its own process only for CA-teardown safety, never a mocked or bypassed path.
    """
    from osprey.connectors.factory import ConnectorFactory, register_builtin_connectors

    register_builtin_connectors()

    overrides = spec["config_overrides"]

    def _get_config_value(key: str, default: Any = None) -> Any:
        return overrides.get(key, default)

    with patch("osprey.utils.config.get_config_value", side_effect=_get_config_value):
        connector = await ConnectorFactory.create_control_system_connector(spec["connector_config"])
        result: dict[str, Any] = {"write_success": None, "write_verified": None}

        write = spec.get("write")
        if write is not None:
            # SP always latches its own readback (verification_level="readback"
            # re-reads the SAME channel it wrote) -- exactly what P3/P5 assert.
            wr = await connector.write_channel(
                write["address"], write["value"], verification_level="readback"
            )
            result["write_success"] = bool(wr.success)
            result["write_verified"] = bool(
                wr.verification is not None and wr.verification.verified
            )

        # Read the target address. For an sp-echo pair, wait (bounded) for the
        # asynchronous SP->RB echo to reflect the written value when the caller
        # asks (settle_read) -- see the module constants above. Without a write,
        # or without settle_read, this is a single immediate read.
        settle_to = write["value"] if (write is not None and spec.get("settle_read")) else None
        read_value = float((await connector.read_channel(spec["read"])).value)
        settled = settle_to is None or abs(read_value - float(settle_to)) <= _SETTLE_TOL
        if settle_to is not None and not settled:
            deadline = time.monotonic() + _SETTLE_TIMEOUT_SEC
            while not settled and time.monotonic() < deadline:
                await asyncio.sleep(_SETTLE_POLL_SEC)
                read_value = float((await connector.read_channel(spec["read"])).value)
                settled = abs(read_value - float(settle_to)) <= _SETTLE_TOL
        result["read_value"] = read_value
        result["read_settled"] = settled

    # Emit the result and force-exit from inside the running loop, BEFORE the
    # connector is disconnected or the event loop is closed -- the libca
    # CA-teardown assertion must never get a chance to run. See module docstring.
    sys.stdout.write(RESULT_MARKER + json.dumps(result) + "\n")
    sys.stdout.flush()
    os._exit(0)


def main() -> int:
    spec = json.loads(sys.argv[1])
    asyncio.run(_do(spec))
    return 0  # unreachable: _do() os._exit()s on success


if __name__ == "__main__":
    sys.exit(main())
