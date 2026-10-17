"""Host-side Channel Access client for the pcaspy transport probe.

Run under the worktree venv (`.venv/bin/python`) with the CA name-server
transport already exported in the environment. Prints one
``BEHAVIOR <name> PASS|FAIL -- <detail>`` line per behaviour and exits 0 only
if every behaviour passed.

The client is aioca, which is asyncio-only: every behaviour is a coroutine, and
the whole run is one ``asyncio.run``.

Modes:

- default: run the three behaviours against a reachable probe server.
- ``--expect-unreachable``: negative control. Exits 0 only if the PVs are
  *not* reachable, proving the positive run really depended on the configured
  name server rather than on some other discovery path.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

PREFIX = os.environ.get("PROBE_PV_PREFIX", "PCASPY:")
ASYNC_DELAY_S = float(os.environ.get("PROBE_ASYNC_DELAY_S", "2.0"))
TELEM_PERIOD_S = float(os.environ.get("PROBE_TELEM_PERIOD_S", "0.5"))
CONNECT_TIMEOUT_S = float(os.environ.get("PROBE_CONNECT_TIMEOUT_S", "15"))

# Observation window for server-initiated monitor events, after the initial
# subscription value has been drained.
MONITOR_WINDOW_S = max(4.0, TELEM_PERIOD_S * 6)


def report_env() -> list[str]:
    """Print the CA transport environment and return any misconfigurations.

    A probe that passes with the transport misconfigured proves nothing, so the
    positive run refuses to score itself unless name-server mode is genuinely
    the only discovery path available to the client.
    """
    name_servers = os.environ.get("EPICS_CA_NAME_SERVERS", "")
    auto_addr = os.environ.get("EPICS_CA_AUTO_ADDR_LIST", "")
    addr_list = os.environ.get("EPICS_CA_ADDR_LIST", "")
    print(
        "client CA env: "
        f"EPICS_CA_NAME_SERVERS={name_servers!r} "
        f"EPICS_CA_AUTO_ADDR_LIST={auto_addr!r} "
        f"EPICS_CA_ADDR_LIST={addr_list!r}",
        flush=True,
    )
    problems = []
    if auto_addr.strip().upper() != "NO":
        problems.append("EPICS_CA_AUTO_ADDR_LIST is not NO (UDP broadcast still enabled)")
    if not name_servers.strip():
        problems.append("EPICS_CA_NAME_SERVERS is empty (no TCP name server configured)")
    if addr_list.strip():
        problems.append("EPICS_CA_ADDR_LIST is non-empty (a broadcast path remains)")
    return problems


async def caget(aioca, name: str, timeout: float) -> object | None:
    """One read over the wire, or ``None`` if nothing answered in time."""
    value = await aioca.caget(name, timeout=timeout, throw=False)
    return None if isinstance(value, aioca.CANothing) else value


async def behavior_transport(aioca) -> tuple[bool, str, float]:
    """1. Basic transport: host caget and caput reach the served PVs."""
    sp = PREFIX + "SYNC:SP"
    rb = PREFIX + "SYNC:RB"
    initial = await caget(aioca, sp, CONNECT_TIMEOUT_S)
    if initial is None:
        return False, f"caget {sp} found nothing (PV unreachable)", 0.0

    target = 12.5
    started = time.monotonic()
    put_result = await aioca.caput(sp, target, wait=True, timeout=10, throw=False)
    elapsed = time.monotonic() - started
    if not put_result.ok:
        return False, f"caput {sp} did not complete: {put_result!r}", elapsed

    echoed = await caget(aioca, rb, CONNECT_TIMEOUT_S)
    if echoed is None or abs(echoed - target) > 1e-9:
        return False, f"caput {sp}={target} but {rb} reads {echoed!r}", elapsed
    return (
        True,
        f"caget {sp}={initial!r}; caput {sp}={target} -> {rb}={echoed!r}; "
        f"synchronous put-completion returned in {elapsed:.3f}s",
        elapsed,
    )


async def behavior_async_completion(aioca, sync_elapsed: float) -> tuple[bool, str]:
    """2. Asynchronous write completion observed with put-completion."""
    sp = PREFIX + "ASYNC:SP"
    rb = PREFIX + "ASYNC:RB"
    target = 42.0
    started = time.monotonic()
    put_result = await aioca.caput(sp, target, wait=True, timeout=ASYNC_DELAY_S + 20, throw=False)
    elapsed = time.monotonic() - started
    committed = await caget(aioca, rb, CONNECT_TIMEOUT_S)

    detail = (
        f"caput {sp}={target} wait=True completed={put_result.ok} after "
        f"{elapsed:.3f}s (server delay {ASYNC_DELAY_S}s); {rb}={committed!r} "
        f"on unblock; synchronous control put took {sync_elapsed:.3f}s"
    )
    if not put_result.ok:
        return False, "put-completion did not report success: " + detail
    if elapsed < ASYNC_DELAY_S * 0.8:
        return False, "client did not block for the server delay: " + detail
    if sync_elapsed >= ASYNC_DELAY_S * 0.5:
        return False, "blocking is not attributable to asyn (control put was slow too): " + detail
    if committed is None or abs(committed - target) > 1e-9:
        return False, "value was not committed before completion was signalled: " + detail
    return True, detail


async def behavior_monitor_events(aioca) -> tuple[bool, str]:
    """3. Server-initiated monitor events reach a host camonitor subscription."""
    name = PREFIX + "TELEM:COUNTER"
    connected = await aioca.connect(name, timeout=CONNECT_TIMEOUT_S, throw=False)
    if not connected.ok:
        return False, f"{name} never connected for monitoring"

    events: list[tuple[float, object]] = []
    subscription = aioca.camonitor(name, lambda value: events.append((time.monotonic(), value)))
    try:
        # Drain the initial subscription value, which every CA subscription
        # delivers on connect and which is therefore not server-initiated.
        await asyncio.sleep(1.0)
        events.clear()

        # No client write happens in this window: anything that arrives was
        # posted by the server on its own schedule.
        await asyncio.sleep(MONITOR_WINDOW_S)
        observed = list(events)

        values = [value for _, value in observed]
        if len(observed) < 3:
            return False, (
                f"only {len(observed)} monitor event(s) on {name} in "
                f"{MONITOR_WINDOW_S:.1f}s with no client write; values={values!r}"
            )
        increasing = all(
            b is not None and a is not None and b > a
            for a, b in zip(values, values[1:], strict=False)
        )
        if not increasing:
            return False, f"monitor values on {name} did not advance: {values!r}"
        return True, (
            f"{len(observed)} server-initiated monitor event(s) on {name} in "
            f"{MONITOR_WINDOW_S:.1f}s with no client write; values={values!r}"
        )
    finally:
        subscription.close()


async def run_negative_control(aioca) -> int:
    """Confirm the PVs are unreachable when the name server address is wrong."""
    sp = PREFIX + "SYNC:SP"
    value = await caget(aioca, sp, 5)
    if value is None:
        print(
            f"NEGATIVE-CONTROL PASS -- caget {sp} found nothing with a wrong "
            "name-server address, so the positive run really used the "
            "configured name server",
            flush=True,
        )
        return 0
    print(
        f"NEGATIVE-CONTROL FAIL -- caget {sp} returned {value!r} despite a wrong "
        "name-server address; the probe is not testing the transport it claims",
        flush=True,
    )
    return 1


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-unreachable",
        action="store_true",
        help="negative control: succeed only if the PVs cannot be reached",
    )
    args = parser.parse_args()

    problems = report_env()
    import aioca  # imported after the env is reported and checked

    if args.expect_unreachable:
        return await run_negative_control(aioca)

    if problems:
        for problem in problems:
            print(f"FATAL: {problem}", flush=True)
        print(
            "BEHAVIOR transport FAIL -- CA transport env is not name-server-only; "
            "a pass here would prove nothing",
            flush=True,
        )
        return 1

    transport_ok, transport_detail, sync_elapsed = await behavior_transport(aioca)
    print(
        f"BEHAVIOR transport {'PASS' if transport_ok else 'FAIL'} -- {transport_detail}", flush=True
    )

    # Monitors are scored before the asynchronous write on purpose: a server
    # that cannot serve an asyn write may die trying, and a dead server would
    # otherwise be misreported as "monitors do not work".
    if transport_ok:
        monitor_ok, monitor_detail = await behavior_monitor_events(aioca)
    else:
        monitor_ok, monitor_detail = False, "skipped: transport did not come up"
    print(
        f"BEHAVIOR monitor-events {'PASS' if monitor_ok else 'FAIL'} -- {monitor_detail}",
        flush=True,
    )

    if transport_ok:
        async_ok, async_detail = await behavior_async_completion(aioca, sync_elapsed)
    else:
        async_ok, async_detail = False, "skipped: transport did not come up"
    print(
        f"BEHAVIOR async-completion {'PASS' if async_ok else 'FAIL'} -- {async_detail}", flush=True
    )

    return 0 if (transport_ok and async_ok and monitor_ok) else 1


if __name__ == "__main__":
    code = asyncio.run(main())
    sys.stdout.flush()
    sys.stderr.flush()
    # Exit without unwinding: the verdict is already decided and printed, and
    # nothing in libca's teardown is allowed a chance to change the exit code.
    os._exit(code)
