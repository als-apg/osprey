"""Child-process driver: the EPICS connector against a live soft IOC.

Run by ``test_epics_ca_softioc.py`` as ``python _epics_ca_driver.py <scenario>
<prefix>``, with ``EPICS_CA_*`` pointing at the test's IOC. Prints one line,
``__RESULT__<json>``, and exits; the exit code is part of what is tested,
because a crash in Channel Access teardown only shows there. Underscore-
prefixed, so pytest never collects it.

Each scenario runs in a process of its own: a process has one Channel Access
context, which every connect repoints, so no two IOCs, and no test of what
happens at exit, can share one.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import threading
from typing import Any
from unittest.mock import patch

from osprey_connectors.control_system.ca_loop import get_ca_loop
from osprey_connectors.control_system.epics_connector import EPICSConnector

RESULT_MARKER = "__RESULT__"
ATEXIT_MARKER = "__ATEXIT__"


def _metadata(value: Any) -> dict[str, Any]:
    meta = value.metadata
    plain = value.value.tolist() if hasattr(value.value, "tolist") else value.value
    return {
        "value": plain,
        "type": type(value.value).__name__,
        "units": meta.units,
        "precision": meta.precision,
        "alarm_status": meta.alarm_status,
        "alarm_severity": meta.alarm_severity,
        "enum_label": meta.enum_label,
        "enum_labels": meta.enum_labels,
    }


def _write(result: Any) -> dict[str, Any]:
    return {
        "outcome": result.outcome.name,
        "observed": result.observed_value,
        "alarm_status": result.alarm_status,
        "refusal_reason": result.refusal_reason,
        "error": result.error_message,
    }


async def _connect() -> EPICSConnector:
    connector = EPICSConnector()
    await connector.connect({"timeout_s": 2})
    return connector


async def contract(prefix: str) -> dict[str, Any]:
    """Every read shape and every write outcome, over the wire."""
    out: dict[str, Any] = {"reads": {}, "writes": {}}
    connector = await _connect()
    for record in ("AO", "LONG", "MBBO", "STR"):
        out["reads"][record] = _metadata(await connector.read_channel(f"{prefix}:{record}"))
    try:
        await connector.read_channel(f"{prefix}:NOPE", timeout=0.5)
    except Exception as exc:  # the exception class is the assertion
        out["reads"]["NOPE"] = type(exc).__name__

    cases = {
        "confirmed": (f"{prefix}:AO", 2.25, {}),
        "into_alarm": (f"{prefix}:AO", 9.0, {}),
        "enum_label": (f"{prefix}:MBBO", "Off", {}),
        "unrequested": (f"{prefix}:AO", 3.0, {"confirm": False}),
        "not_sent": (f"{prefix}:NOPE", 1.0, {"timeout": 0.5}),
        "unacknowledged": (f"{prefix}:SLOW", 1.0, {"timeout": 1.0}),
        "access_denied": (f"{prefix}:RO", 5.0, {}),
        "text_hex_to_long": (f"{prefix}:LONG", "0x10", {}),
        "text_exponent_to_long": (f"{prefix}:LONG", "1e3", {}),
        "fraction_to_long": (f"{prefix}:LONG", 1.5, {}),
        "overflow_to_long": (f"{prefix}:LONG", 2**31, {"confirm": False}),
        "text_exponent_to_ao": (f"{prefix}:AO", "2.5e0", {}),
        "text_to_char_waveform": (f"{prefix}:WF", "/data/run42", {}),
        "disabled_record": (f"{prefix}:DISABLED", 1.0, {}),
    }
    for name, (address, value, kwargs) in cases.items():
        out["writes"][name] = _write(await connector.write_channel(address, value, **kwargs))
    out["long_after_refusal"] = (await connector.read_channel(f"{prefix}:LONG")).value

    updates: list[float] = []
    sub_id = await connector.subscribe(f"{prefix}:AO", lambda v: updates.append(v.value))
    await asyncio.sleep(0.5)
    await connector.write_channel(f"{prefix}:AO", 4.5)
    await asyncio.sleep(0.5)
    await connector.unsubscribe(sub_id)
    await connector.write_channel(f"{prefix}:AO", 5.5)
    await asyncio.sleep(0.5)
    out["monitor"] = updates

    read_current = connector._current_value_reader()
    out["max_step"] = [
        await asyncio.to_thread(read_current, f"{prefix}:AO"),
        await asyncio.to_thread(read_current, f"{prefix}:NOPE"),
    ]
    return out


def runtime_style(prefix: str) -> dict[str, Any]:
    """Reads the way ``osprey.runtime`` makes them: a fresh loop (and thread) per call."""
    from aioca._catools import _Context

    connector = asyncio.run(_connect())
    values = [asyncio.run(connector.read_channel(f"{prefix}:AO")).value for _ in range(10)]

    def in_a_thread() -> None:
        values.append(asyncio.run(connector.read_channel(f"{prefix}:AO")).value)

    for _ in range(10):
        worker = threading.Thread(target=in_a_thread)
        worker.start()
        worker.join()
    return {"reads": len(values), "channel_caches": len(_Context._channel_caches)}


async def aioca_first(prefix: str) -> dict[str, Any]:
    """aioca used directly on this thread before the connector: one context, clean exit."""
    import aioca

    direct = float(await aioca.caget(f"{prefix}:AO", timeout=2))
    connector = await _connect()
    read = await connector.read_channel(f"{prefix}:AO")
    write = await connector.write_channel(f"{prefix}:AO", 2.0)
    await connector.subscribe(f"{prefix}:AO", lambda _value: None)
    return {"direct": direct, "read": read.value, "write": write.outcome.name}


async def connector_first(prefix: str) -> dict[str, Any]:
    import aioca

    connector = await _connect()
    write = await connector.write_channel(f"{prefix}:AO", 3.0)
    await connector.subscribe(f"{prefix}:AO", lambda _value: None)
    direct = float(await aioca.caget(f"{prefix}:AO", timeout=2))
    return {"direct": direct, "write": write.outcome.name}


async def dispatch(prefix: str) -> dict[str, Any]:
    """The dispatch trigger source and the connector, on one CA loop."""
    import types

    from osprey.dispatch.sources.epics_ca import EpicsCaSource

    fired: list[list[float]] = []

    async def fire(_trigger: Any, payload: dict[str, Any]) -> None:
        fired.append([payload["previous_value"], payload["value"]])

    trigger = types.SimpleNamespace(
        name="crossing",
        source_config={"pv": f"{prefix}:AO", "threshold": 5.0, "cool_down_sec": 0},
    )
    source = EpicsCaSource()
    await source.start([trigger], fire)
    connector = await _connect()
    await asyncio.sleep(0.5)
    for value in (1.0, 6.0, 2.0, 7.5):
        await connector.write_channel(f"{prefix}:AO", value)
        await asyncio.sleep(0.3)
    await source.stop()
    return {"fired": fired}


async def stopped_before_teardown(prefix: str) -> dict[str, Any]:
    """Report, from an ``atexit`` hook, whether the CA loop still runs then.

    The hook is registered after the connector's first CA call, so after
    aioca's own teardown hook, and ``atexit`` runs it first: it sees what
    aioca's teardown is about to see.
    """
    import atexit

    connector = await _connect()
    await connector.read_channel(f"{prefix}:AO")
    await connector.subscribe(f"{prefix}:AO", lambda _value: None)

    def report() -> None:
        alive = any(t.name == "osprey-ca-loop" and t.is_alive() for t in threading.enumerate())
        print(ATEXIT_MARKER + json.dumps({"ca_loop_alive": alive}), flush=True)

    atexit.register(report)
    return {}


async def unreachable_name_server(prefix: str) -> dict[str, Any]:
    """A read through a name-server gateway nobody answers, then exit."""
    connector = EPICSConnector()
    await connector.connect(
        {
            "timeout_s": 1,
            "gateways": {"read_only": {"address": "127.0.0.1", "port": 1, "use_name_server": True}},
        }
    )
    try:
        await connector.read_channel(f"{prefix}:AO", timeout=1)
    except Exception as exc:  # the exception class is the assertion
        return {"read": type(exc).__name__}
    return {"read": "answered"}


async def repoint(prefix: str) -> dict[str, Any]:
    """Connect at the IOC, then at a port nothing serves, then at the IOC again.

    Each connect repoints the process's one CA context. The read in the middle
    goes unanswered only if the context really moved. The monitor opened on the
    first context is closed with it, re-opened on each new one, and delivers
    again once CA is back at the IOC.
    """
    import os

    def at(port: int | str) -> dict[str, Any]:
        return {
            "timeout_s": 1,
            "gateways": {"read_only": {"address": "127.0.0.1", "port": port}},
        }

    ioc_port = os.environ["EPICS_CA_SERVER_PORT"]
    first = EPICSConnector()
    await first.connect(at(ioc_port))
    before = (await first.read_channel(f"{prefix}:AO")).value
    updates: list[float] = []
    await first.subscribe(f"{prefix}:AO", lambda value: updates.append(value.value))
    await asyncio.sleep(0.3)
    seen_before_repoint = len(updates)

    nowhere = EPICSConnector()
    await nowhere.connect(at(1))
    try:
        await nowhere.read_channel(f"{prefix}:AO", timeout=1)
        moved = "answered"
    except Exception as exc:  # the exception class is the assertion
        moved = type(exc).__name__

    back = EPICSConnector()
    await back.connect(at(ioc_port))
    await asyncio.sleep(0.3)
    write = await back.write_channel(f"{prefix}:AO", 4.0)
    after = (await back.read_channel(f"{prefix}:AO")).value
    await asyncio.sleep(0.3)
    return {
        "before": before,
        "moved": moved,
        "write": write.outcome.name,
        "after": after,
        "monitor_after_repoint": updates[seen_before_repoint:],
    }


async def dispatch_repoint(prefix: str) -> dict[str, Any]:
    """A dispatch trigger, armed before CA is pointed away and back, still fires."""
    import os
    import types

    from osprey.dispatch.sources.epics_ca import EpicsCaSource

    fired: list[list[float]] = []

    async def fire(_trigger: Any, payload: dict[str, Any]) -> None:
        fired.append([payload["previous_value"], payload["value"]])

    trigger = types.SimpleNamespace(
        name="crossing",
        source_config={"pv": f"{prefix}:AO", "threshold": 5.0, "cool_down_sec": 0},
    )
    source = EpicsCaSource()
    await source.start([trigger], fire)
    await asyncio.sleep(0.3)

    ioc_port = os.environ["EPICS_CA_SERVER_PORT"]
    for port in (1, ioc_port):
        await EPICSConnector().connect(
            {"timeout_s": 1, "gateways": {"read_only": {"address": "127.0.0.1", "port": port}}}
        )
    await asyncio.sleep(0.5)
    writer = await _connect()
    for value in (1.0, 6.0):
        await writer.write_channel(f"{prefix}:AO", value)
        await asyncio.sleep(0.3)
    await source.stop()
    return {"fired": fired}


def recorder(prefix: str) -> dict[str, Any]:
    """The archiver recorder's sampling, each sample on a loop of its own.

    Sampled from fresh loops on purpose: the recorder's loop is never the CA
    loop, and its channels must still be cached on the CA loop alone.
    """
    import time

    from aioca._catools import _Context

    from osprey.services.archiver_recorder.config import RecorderSettings
    from osprey.services.archiver_recorder.recorder import channel_access_reader

    settings = RecorderSettings(
        host="archiver-mongodb",
        port=27017,
        database="osprey_archiver",
        collection="pv_history",
        auth_source="admin",
        username="osprey",
        password_env="MONGO_ROOT_PASSWORD",
        timeout_s=5,
        cadence_sec=1,
        tail_cadence_sec=60,
        poll_sec=30,
        hot_span_hours=48,
        retention_days=30,
    )
    read = channel_access_reader(settings)
    addresses = [f"{prefix}:{record}" for record in ("AO", "LONG", "MBBO", "NOPE")]
    samples, elapsed = [], []
    for _ in range(2):
        started = time.monotonic()
        sample = asyncio.run(read(addresses))
        elapsed.append(time.monotonic() - started)
        samples.append({name.split(":", 1)[1]: value for name, value in sample.items()})
    ca_loop_caches = [loop is get_ca_loop().loop for loop in _Context._channel_caches]
    return {"samples": samples, "elapsed": elapsed, "ca_loop_caches": ca_loop_caches}


SCENARIOS = {
    "contract": contract,
    "runtime_style": runtime_style,
    "aioca_first": aioca_first,
    "connector_first": connector_first,
    "dispatch": dispatch,
    "stopped_before_teardown": stopped_before_teardown,
    "unreachable_name_server": unreachable_name_server,
    "repoint": repoint,
    "dispatch_repoint": dispatch_repoint,
    "recorder": recorder,
}


def main() -> None:
    scenario, prefix = sys.argv[1], sys.argv[2]
    run = SCENARIOS[scenario]
    with patch.object(EPICSConnector, "_writes_enabled", property(lambda _self: True)):
        result = asyncio.run(run(prefix)) if inspect.iscoroutinefunction(run) else run(prefix)
    print(RESULT_MARKER + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
