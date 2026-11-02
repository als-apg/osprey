#!/usr/bin/env python3
"""Out-of-process host-side Channel Access op for the substrate-equivalence e2e.

Runs ONE ``connect -> (optional confirmed write) -> read`` sequence against the
co-deployed Virtual Accelerator through the real, unmodified
``VirtualAcceleratorConnector`` (built by the production ``ConnectorFactory``),
prints the result as one marker-prefixed JSON line, then force-exits.

Why a subprocess per proof (mirrors P2's ``sweep_check`` subprocess): repeated
CA-context create/destroy cycles in a single process eventually trip a libca
teardown assertion (``assert(this->pudpiiu)``, ``cac.cpp``, thread
``CAC-TCP-send``) in ``EPICS_CA_NAME_SERVERS``-only mode -- a known libca
quirk on the CA-circuit teardown path, unrelated to the connector or the
substrate: the read/write round trip completes and is correct *before* it fires
(see ``scripts/va/probe_roundtrip.py``'s "KNOWN FINDING"). It was found under
pyepics, but it lives in libca itself, which aioca loads from the same
``epicscorelibs`` build, so the guard stays. Isolating
each proof's host CA work in its own process -- which emits its result and
``os._exit()``s *before* any connector disconnect / CA teardown runs -- means
the fault can never recur in the pytest process.

Deliberately never calls ``connector.disconnect()`` and never lets the
interpreter shut down normally: the whole point is to skip the CA-circuit
teardown that carries the assertion. All CA work is finished once the values
are in hand; ``os._exit(0)`` hands the socket back to the OS untorn.

Protocol -- ``argv[1]`` is a JSON spec::

    {"connector_config": {...},          # the ConnectorFactory connector config
     "config_overrides": {...},          # osprey_connectors.config.get_config_value overrides
     "read": "<address>",                # the readback address to return
     "write": {"address": "<addr>", "value": <float>} | null,
     "settle_read": <bool>,              # poll the read until it is within
                                         # settle_tolerance of write.value
     "settle_tolerance": <float>,        # required when settle_read is true
     "arm_target": "<target>" | null}    # optional: the control target this
                                         # connector writes to, armed (see
                                         # _arm_switch) before it is built

On success, emits exactly one stdout line::

    __HOST_CA_RESULT__{"read_value": <float>, "read_settled": <bool>,
                       "write_outcome": <str|null>,
                       "write_error_message": <str|null>}

``write_outcome`` is the ``WriteOutcome`` word the connector reported, or
``null`` when no write was asked for. The word rather than a bool: on a
container flake the caller has to be able to tell a refusal from a mismatch
from a re-read that failed, and ``write_error_message`` carries the reason the
outcome came with (``null`` for the outcomes that carry none).

``read_settled`` is ``true`` when ``settle_read`` was not requested, or when the
readback came within ``settle_tolerance`` of the written value within the settle
deadline; ``false`` means the asynchronous echo never propagated in time (the
caller should fail loudly). ``settle_tolerance`` is the band of the motion the
readback's seed declares, which a served readback carries on top of the value
it holds.

Exit 0 on success; a non-zero exit (or a native SIGBUS) means the host CA op
failed and the caller must surface it -- never silently pass.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from typing import Any
from unittest.mock import patch

RESULT_MARKER = "__HOST_CA_RESULT__"

# sp-echo SP->RB propagation is asynchronous in the IOC (the setpoint record's
# on_update fires on the IOC's own loop AFTER the caput that write_channel
# confirms against the SP), so an immediate readback read can beat the echo.
# When a caller sets settle_read, poll the readback until it reflects the
# written value, within the spec's settle_tolerance, rather than guess a fixed
# delay (condition-based waiting).
_SETTLE_TIMEOUT_SEC = 10.0
_SETTLE_POLL_SEC = 0.05


def _settle_tolerance(spec: dict[str, Any]) -> float:
    """The spec's ``settle_tolerance``, which a settling read cannot go without."""
    if spec.get("settle_tolerance") is None:
        raise ValueError("a spec with settle_read must carry settle_tolerance")
    return float(spec["settle_tolerance"])


#: The environment that would point the write verdict at a record, an owner or a
#: launch pin other than this process's own (posture_store's verdict variables).
#: An armed op is the operator's own put from the host, so none of them applies.
_FOREIGN_VERDICT_ENV = (
    "OSPREY_CONTROL_OWNER",
    "OSPREY_DISPATCH_RUN",
    "OSPREY_CONTROL_CONTEXT_TREE",
    "OSPREY_CONTROL_CONTEXT_DIR",
    "OSPREY_LAUNCH_POSTURE",
    "OSPREY_EXECUTION_MODE",
)


def _arm_switch(target: str) -> str:
    """Turn this login's switch on for *target* in a record of this process's own.

    A connector write is allowed only when the login's tier allows writes (the
    caller's ``control_system.writes`` override), the connector is stamped with
    the machine it writes to (passed to the factory as ``control_target``) and
    the login's control-context record has that machine switched on. The record
    lives under a fresh agent-data root that only this process reads, so arming
    one op never reaches a deployment's record or another op's. Written through
    the production writers, the way the chip does: the record first, then the
    person's gesture against the epoch it was stamped with.

    Returns the agent-data root, which the caller removes before it
    force-exits: ``os._exit`` skips atexit handlers, so the one registered
    here only covers a run that fails and exits normally.
    """
    import atexit
    import tempfile

    from osprey_connectors import control_context, posture_store

    for name in _FOREIGN_VERDICT_ENV:
        os.environ.pop(name, None)
    root = tempfile.mkdtemp(prefix="osprey-ca-op-")
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    os.environ[posture_store.AGENT_DATA_ROOT_ENV_VAR] = root
    posture_store.invalidate_cache()
    control_context.invalidate_cache()
    path = control_context.record_path()
    if path is None:
        raise RuntimeError("the control-context record path did not resolve")
    control_context.write_record(control_context.ControlContext(target=target, generation=1))
    control_context.invalidate_cache()
    current = control_context.read_record(path=path)
    epoch = (current.switch_epoch if current is not None else None) or 0
    try:
        control_context.set_switch([target], "asked", epoch=epoch, path=path)
    except control_context.StoppedSince as refused:
        if not refused.stamped or refused.epoch is None:
            raise
        control_context.set_switch([target], "asked", epoch=refused.epoch, path=path)
    control_context.invalidate_cache()
    return root


async def _do(spec: dict[str, Any]) -> dict[str, Any]:
    """Connect, optionally write (confirmed), read, and emit -- then force-exit.

    Uses the SAME config the in-process host connector used (same
    ``get_config_value`` overrides, same connector config), so this
    out-of-process read/write drives a REAL production connector -- isolated in
    its own process only for CA-teardown safety, never a mocked or bypassed path.
    """
    from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors

    register_builtin_connectors()

    overrides = spec["config_overrides"]

    def _get_config_value(key: str, default: Any = None) -> Any:
        # Answer a SECTION read the way the real loader does: a nested mapping
        # assembled from every override under that prefix. The per-type write
        # posture reads ``control_system`` whole and indexes the connector
        # block by name (a custom type is a dotted module path, so the leaf
        # cannot be looked up by dotted path), and a flat map that only
        # answered exact keys would leave that read empty and writes unarmed.
        if key in overrides:
            return overrides[key]
        prefix = key + "."
        section: dict[str, Any] = {}
        for dotted, value in overrides.items():
            if not dotted.startswith(prefix):
                continue
            node = section
            *parents, leaf = dotted[len(prefix) :].split(".")
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = value
        return section or default

    arm_target = spec.get("arm_target")
    armed_root = _arm_switch(arm_target) if arm_target is not None else None

    with patch("osprey_connectors.config.get_config_value", side_effect=_get_config_value):
        connector = await ConnectorFactory.create_control_system_connector(
            spec["connector_config"], control_target=arm_target
        )
        result: dict[str, Any] = {"write_outcome": None, "write_error_message": None}

        write = spec.get("write")
        if write is not None:
            # SP always latches its own readback (confirmation re-reads the SAME
            # channel it wrote) -- exactly what P3/P5 assert.
            wr = await connector.write_channel(write["address"], write["value"], confirm=True)
            # WriteOutcome is a StrEnum, so str() is the plain word and the
            # emitted JSON stays a string the caller can compare directly.
            result["write_outcome"] = str(wr.outcome)
            result["write_error_message"] = wr.error_message

        # Read the target address. For an sp-echo pair, wait (bounded) for the
        # asynchronous SP->RB echo to reflect the written value when the caller
        # asks (settle_read) -- see the module constants above. Without a write,
        # or without settle_read, this is a single immediate read.
        settle_to = write["value"] if (write is not None and spec.get("settle_read")) else None
        tolerance = _settle_tolerance(spec) if settle_to is not None else 0.0
        read_value = float((await connector.read_channel(spec["read"])).value)
        settled = settle_to is None or abs(read_value - float(settle_to)) <= tolerance
        if settle_to is not None and not settled:
            deadline = time.monotonic() + _SETTLE_TIMEOUT_SEC
            while not settled and time.monotonic() < deadline:
                await asyncio.sleep(_SETTLE_POLL_SEC)
                read_value = float((await connector.read_channel(spec["read"])).value)
                settled = abs(read_value - float(settle_to)) <= tolerance
        result["read_value"] = read_value
        result["read_settled"] = settled

    # Emit the result and force-exit from inside the running loop, BEFORE the
    # connector is disconnected or the event loop is closed -- the libca
    # CA-teardown assertion must never get a chance to run. See module docstring.
    sys.stdout.write(RESULT_MARKER + json.dumps(result) + "\n")
    sys.stdout.flush()
    if armed_root is not None:
        shutil.rmtree(armed_root, ignore_errors=True)
    os._exit(0)


def main() -> int:
    spec = json.loads(sys.argv[1])
    asyncio.run(_do(spec))
    return 0  # unreachable: _do() os._exit()s on success


if __name__ == "__main__":
    sys.exit(main())
