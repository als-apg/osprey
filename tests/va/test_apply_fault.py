"""Tests for the build-time stuck-setpoint apply fault (``stuck_setpoints``).

Same process-boundary gotchas as ``test_record_factory.py`` apply here (see
that module's docstring): softioc's ``builder``/``softioc`` are process-global
and constructing a record poisons the process as a future CA client. The
live-round-trip check therefore reuses the same three-process split -- IOC
subprocess, CA-client subprocess, this pytest process -- with its own
dedicated address set so it never collides with ``test_record_factory.py``'s.
"""

from __future__ import annotations

import json
import os
import socket
import sys


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# See test_record_factory.py for why these must be set before any epics/
# softioc import, and why setdefault (not overwrite) is correct when both
# test modules run in the same pytest session.
os.environ.setdefault("EPICS_CA_ADDR_LIST", "127.0.0.1")
os.environ.setdefault("EPICS_CA_AUTO_ADDR_LIST", "NO")
if "EPICS_CA_SERVER_PORT" not in os.environ:
    os.environ["EPICS_CA_SERVER_PORT"] = str(_free_port())
if "EPICS_CA_REPEATER_PORT" not in os.environ:
    os.environ["EPICS_CA_REPEATER_PORT"] = str(_free_port())


def _run_ca_client_subprocess(argv: list[str]) -> None:
    """See test_record_factory.py's identical helper -- must import only
    ``epics``, dispatched before this module ever imports ``ioc.records``."""
    import epics

    op = argv[0]
    if op == "caget":
        (address,) = argv[1:]
        value = epics.caget(address, timeout=5, connection_timeout=5)
        print(json.dumps({"value": value}), flush=True)
    elif op == "caput":
        address, value = argv[1], float(argv[2])
        ok = epics.caput(address, value, wait=True, timeout=5)
        print(json.dumps({"ok": bool(ok)}), flush=True)
    else:
        raise ValueError(f"unknown CA client op {op!r}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--ca-client":
    _run_ca_client_subprocess(sys.argv[2:])
    sys.exit(0)

# Everything below this point is only reached by pytest's own import of this
# module, or by the --run-live-ioc-subprocess entry point further down --
# never by the --ca-client path above.

import subprocess  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402

from osprey.services.virtual_accelerator.ioc.records import build_records  # noqa: E402
from osprey.services.virtual_accelerator.manifest import (  # noqa: E402
    PARTITION_SP_ECHO,
    RECORD_TYPE_ANALOG,
)

# Two sp-echo pairs, distinct devices so _channel_key() can't collide them:
# one is faulted via stuck_setpoints, the other is a plain control to prove
# the fault is truly per-channel, not a partition-wide switch.
_STUCK_SP = "ZZTEST:MAG:STUCK:01:CURRENT:SP"
_STUCK_RB = "ZZTEST:MAG:STUCK:01:CURRENT:RB"
_LIVE_SP = "ZZTEST:MAG:LIVE:01:CURRENT:SP"
_LIVE_RB = "ZZTEST:MAG:LIVE:01:CURRENT:RB"


def _channel(
    address: str,
    *,
    partition: str,
    subfield: str,
    record_type: str = RECORD_TYPE_ANALOG,
    noise: bool = False,
    ring: str = "ZZTEST",
    system: str = "MAG",
    family: str = "HCM",
    device: str = "99",
    field: str = "CURRENT",
) -> dict:
    """Same synthetic manifest-entry builder as test_record_factory.py."""
    return {
        "address": address,
        "ring": ring,
        "system": system,
        "family": family,
        "device": device,
        "field": field,
        "subfield": subfield,
        "partition": partition,
        "record_type": record_type,
        "noise": noise,
    }


class TestStuckSetpointNonLive:
    """build_records() itself, without a live IOC -- default behavior and
    the fault's default-off contract."""

    def test_default_is_no_fault(self):
        sp = _channel(
            "ZZTEST:MAG:DEFAULTOFF:01:CURRENT:SP", partition=PARTITION_SP_ECHO, subfield="SP"
        )
        rb = _channel(
            "ZZTEST:MAG:DEFAULTOFF:01:CURRENT:RB", partition=PARTITION_SP_ECHO, subfield="RB"
        )
        # Must not raise, and stuck_setpoints defaults to frozenset() -- no
        # channel is faulted unless explicitly named.
        records = build_records([rb, sp])
        assert set(records.all) == {
            "ZZTEST:MAG:DEFAULTOFF:01:CURRENT:SP",
            "ZZTEST:MAG:DEFAULTOFF:01:CURRENT:RB",
        }

    def test_stuck_setpoints_accepts_arbitrary_addresses_without_matching_channels(self):
        # An address in stuck_setpoints that doesn't correspond to any built
        # setpoint channel is simply inert -- this is a per-channel fixture,
        # not a channel selector.
        sp = _channel(
            "ZZTEST:MAG:UNRELATED:01:CURRENT:SP", partition=PARTITION_SP_ECHO, subfield="SP"
        )
        rb = _channel(
            "ZZTEST:MAG:UNRELATED:01:CURRENT:RB", partition=PARTITION_SP_ECHO, subfield="RB"
        )
        records = build_records([rb, sp], stuck_setpoints=frozenset({"NOT:A:REAL:CHANNEL:SP"}))
        assert "ZZTEST:MAG:UNRELATED:01:CURRENT:SP" in records.all


def _run_live_ioc_subprocess() -> None:
    """Subprocess entry point: builds one stuck and one live sp-echo pair,
    serves them until killed."""
    from softioc import asyncio_dispatcher, builder, softioc

    channels = [
        _channel(_STUCK_RB, partition=PARTITION_SP_ECHO, subfield="RB", device="01"),
        _channel(_STUCK_SP, partition=PARTITION_SP_ECHO, subfield="SP", device="01"),
        _channel(_LIVE_RB, partition=PARTITION_SP_ECHO, subfield="RB", device="02"),
        _channel(_LIVE_SP, partition=PARTITION_SP_ECHO, subfield="SP", device="02"),
    ]
    build_records(channels, stuck_setpoints=frozenset({_STUCK_SP}))

    dispatcher = asyncio_dispatcher.AsyncioDispatcher()
    builder.LoadDatabase()
    softioc.iocInit(dispatcher)

    print("apply_fault live_ioc subprocess ready", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--run-live-ioc-subprocess":
    _run_live_ioc_subprocess()


def _ca_client(*args: str, timeout: float = 10) -> dict:
    result = subprocess.run(
        [sys.executable, __file__, "--ca-client", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"CA client subprocess failed: {result.stdout}\n{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _caget(address: str) -> float | None:
    return _ca_client("caget", address)["value"]


def _caput(address: str, value: float) -> bool:
    return _ca_client("caput", address, str(value))["ok"]


@pytest.fixture(scope="module")
def live_ioc():
    """Launch the IOC subprocess and wait for it to start serving PVs."""
    proc = subprocess.Popen(
        [sys.executable, __file__, "--run-live-ioc-subprocess"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 20
    connected = False
    last_error = None
    while time.monotonic() < deadline:
        try:
            value = _caget(_LIVE_RB)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            value = None
        if value is not None:
            connected = True
            break
        time.sleep(0.5)

    if not connected:
        proc.terminate()
        output = proc.stdout.read() if proc.stdout else ""
        proc.wait(timeout=5)
        raise RuntimeError(
            f"apply_fault live_ioc subprocess never came up "
            f"(last_error={last_error!r}); output:\n{output}"
        )

    yield

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


class TestLiveStuckSetpointRoundTrip:
    """A real Channel Access client, in its own process, must observe the
    apply fault honestly: the SP latches the written value, but the paired
    RB never moves -- and only for the channel named in stuck_setpoints."""

    def test_stuck_sp_write_leaves_rb_at_prior_value(self, live_ioc):
        before = _caget(_STUCK_RB)
        assert _caput(_STUCK_SP, 7.25)
        time.sleep(0.3)
        # The SP record itself still latches the caput -- only the outward
        # propagation into RB is suppressed.
        assert _caget(_STUCK_SP) == pytest.approx(7.25)
        assert _caget(_STUCK_RB) == pytest.approx(before)

    def test_non_stuck_sibling_sp_still_echoes_to_rb(self, live_ioc):
        assert _caput(_LIVE_SP, 4.5)
        time.sleep(0.3)
        assert _caget(_LIVE_RB) == pytest.approx(4.5)
