"""The exact bytes every frame the connector host and its parent exchange is written as.

A pin, captured from the codec as it stood when ``PROTOCOL_VERSION`` 3 was
defined: one request for each method, the ``init`` payload with and without
its optional keys, one result for each record a frame carries and the
post-connect report, one error frame for each class the exception registry
carries, an unregistered class and a subclass of a registered one, and a batch
read whose failure crosses as an error node. A refactor of the codec, the
exception registry or the child's error path that changes any byte fails here,
and a version that never shipped is only kept by changing no byte.

Each frame is produced the way the process that sends it produces it: a
request by the proxy's own call (the bytes it writes to the pipe), a result by
the codec, and an error by the child's error path, so the pin holds what
crosses the pipe, not what one helper happens to build.
"""

import asyncio
from datetime import UTC, datetime

import numpy as np
import pytest

from osprey_connectors.control_system import (
    Alarm,
    ChannelMetadata,
    ChannelReadBatch,
    ChannelValue,
    ChannelWriteResult,
)
from osprey_connectors.errors import (
    ChannelError,
    ChannelLimitsViolationError,
    ChannelNotFoundError,
    ChannelUnreachableError,
    ChannelWriteBlockedError,
    ChannelWriteFailedError,
    ConnectorInternalError,
    ConnectorUnavailableError,
    ConnectRefusedError,
    LimitsViolation,
)
from osprey_connectors.ipc import _exceptions, _host
from osprey_connectors.ipc import _frames as frames
from osprey_connectors.ipc._proxy import ConnectorHostProxy
from osprey_connectors.refusals import RefusalReason

_RID = "req-0001"
_AT = datetime(2026, 1, 2, 3, 4, 5, 250000, tzinfo=UTC)

_SECTION = {
    "type": "epics",
    "connector": {
        "epics": {
            "gateways": {
                "read_only": {"address": "gw.example", "port": 5064},
                "write_access": {"address": "gw.example", "port": 5084},
            }
        }
    },
}


# -- requests: the bytes the proxy writes --------------------------------------


class _Silent:
    """A pipe read half that never answers."""

    def __init__(self) -> None:
        self._never = asyncio.Event()

    async def read(self, _n: int = -1) -> bytes:
        await self._never.wait()
        return b""


class _Capture:
    """A pipe write half that keeps every frame written to it."""

    def __init__(self) -> None:
        self.written: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.written.append(bytes(data))

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None


async def _sent(call) -> bytes:
    """The one frame *call* (proxy -> awaitable) writes, its reply never awaited."""
    writer = _Capture()
    proxy = ConnectorHostProxy(_Silent(), writer)
    task = asyncio.ensure_future(call(proxy))
    while not writer.written:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await proxy.disconnect(ack_timeout_s=0)
    return writer.written[0]


def _init(**optional: object) -> dict[str, object]:
    """The init payload as a supervisor builds it: the section and target, then the optional keys."""
    return {"control_system": _SECTION, "target": "live", **optional}


_REQUESTS = {
    "read_channel": lambda proxy: proxy.read_channel("SR:DCCT", timeout_s=1.5),
    "read_channel/no_timeout": lambda proxy: proxy.read_channel("SR:DCCT"),
    "read_multiple_channels": lambda proxy: proxy.read_multiple_channels(
        ["SR:A", "SR:B"], timeout_s=2.0
    ),
    "write_channel": lambda proxy: proxy.write_channel("SR:Q", 1.25, timeout_s=3.0),
    "write_channel/confirm_false": lambda proxy: proxy.write_channel(
        "SR:Q", 1.25, timeout_s=3.0, confirm=False
    ),
    "write_channel/array": lambda proxy: proxy.write_channel(
        "SR:WAVE", np.arange(3, dtype=np.float64)
    ),
    "write_multiple_channels": lambda proxy: proxy.write_multiple_channels(
        [("SR:A", 1), ("SR:B", "on")], timeout_s=4.0, confirm=True
    ),
    "validate_channel": lambda proxy: proxy.validate_channel("SR:Q"),
    "init": lambda proxy: proxy.supervisor_request("init", _init(), timeout_s=30.0),
    "init/config_file": lambda proxy: proxy.supervisor_request(
        "init", _init(config_file="/srv/project/config.yml"), timeout_s=30.0
    ),
    "init/execution_mode": lambda proxy: proxy.supervisor_request(
        "init", _init(execution_mode="readonly"), timeout_s=30.0
    ),
    "init/config_file+execution_mode": lambda proxy: proxy.supervisor_request(
        "init",
        _init(config_file="/srv/project/config.yml", execution_mode="readonly"),
        timeout_s=30.0,
    ),
    "ping": lambda proxy: proxy.supervisor_request("ping", {}, timeout_s=2.0),
    "spawn_probe": lambda proxy: proxy.supervisor_request(
        "spawn_probe", {"address": "SR:DCCT", "timeout_s": 5.0}, timeout_s=5.0
    ),
}


async def _disconnect_request() -> bytes:
    writer = _Capture()
    await ConnectorHostProxy(_Silent(), writer).disconnect(ack_timeout_s=0)
    return writer.written[0]


# -- results: the bytes the codec writes for each value a reply carries ----------


def _reading(value: object = 4.5) -> ChannelValue:
    return ChannelValue(
        address="SR:Q",
        value=value,
        sampled_at=_AT,
        alarm=Alarm(status="HIGH", severity=1),
        enum_label=None,
        metadata=ChannelMetadata(units="A", precision=3, display_low=0.0, display_high=10.0),
    )


_LIMITS = LimitsViolation(
    kind="MAX_EXCEEDED",
    reason="Value 999.0 above maximum 500.0",
    min_value=0.0,
    max_value=500.0,
    max_step=None,
    current_value=120.0,
)

#: The post-connect report, its keys in the order the child writes them.
_REPORT = {
    "selected_role": "read_only",
    "mode": "gateway",
    "host": "gw.example",
    "port": 5064,
    "_epics_configured": True,
    "connector_type": "epics",
    "transport": "ca",
    "target": "live",
    "writes_enabled": False,
    "readonly_run": True,
    "authorization": None,
    "epics_env": {"EPICS_CA_ADDR_LIST": "gw.example", "EPICS_CA_AUTO_ADDR_LIST": "NO"},
    "pid": 4242,
}

_RESULTS = {
    "Alarm": Alarm(status="LOLO", severity=2),
    "ChannelValue": _reading(),
    "ChannelValue/enum": ChannelValue(
        address="SR:MODE",
        value=1,
        sampled_at=_AT,
        enum_label="On",
        metadata=ChannelMetadata(enum_labels=("Off", "On"), raw_metadata={"DESC": "mode"}),
    ),
    "ChannelValue/array": _reading(np.array([1, 2, 3], dtype=np.int32)),
    "ChannelMetadata": ChannelMetadata(
        units="mm",
        precision=2,
        description="horizontal position",
        display_low=-5.0,
        display_high=5.0,
        raw_metadata={"EGU": "mm", "PREC": 2},
        control_low=-4.0,
        control_high=4.0,
        warning_low=-3.0,
        warning_high=3.0,
        alarm_low=-4.5,
        alarm_high=4.5,
        writable=True,
    ),
    "ChannelReadBatch": ChannelReadBatch(
        values={"SR:Q": _reading()},
        failures={
            "SR:GONE": ChannelUnreachableError("SR:GONE", "SR:GONE did not answer in 2.0s"),
            "SR:NOPE": ChannelNotFoundError("SR:NOPE"),
        },
    ),
    "ChannelWriteResult/confirmed": ChannelWriteResult.confirmed("SR:Q", 4.5, _reading()),
    "ChannelWriteResult/refused": ChannelWriteResult.refused(
        "SR:Q",
        999.0,
        RefusalReason.LIMITS,
        "Value 999.0 above maximum 500.0",
        limits=_LIMITS,
    ),
    "ChannelWriteResult/list": [
        ChannelWriteResult.unrequested("SR:A", 1),
        ChannelWriteResult.failed("SR:B", "on", "the IOC refused the put"),
    ],
    "LimitsViolation": _LIMITS,
    "post_connect_report": _REPORT,
    "ping": 4242,
    "validate_channel": True,
    "disconnect": None,
    "numpy_scalar": np.float32(2.5),
}


# -- errors: the bytes the child writes for each failure it reports -------------


class _OwnUnreachable(ChannelUnreachableError):
    """A connector's own subclass of a registered class whose text its base re-renders alike."""


#: One instance per class the registry carries, by tag.
_ERRORS: dict[str, BaseException] = {
    "ConnectionError": ConnectionError("gateway unreachable"),
    "TimeoutError": TimeoutError("read timed out after 2.0s"),
    "ConnectRefusedError": ConnectRefusedError("no read-only gateway to connect through"),
    "ChannelError": ChannelError("SR:Q", "SR:Q failed its own way"),
    "ChannelLimitsViolationError": ChannelLimitsViolationError(
        address="SR:Q",
        value=999.0,
        violation_type="MAX_EXCEEDED",
        violation_reason="Value 999.0 above maximum 500.0",
        min_value=0.0,
        max_value=500.0,
        max_step=10.0,
        current_value=120.0,
    ),
    "ChannelNotFoundError": ChannelNotFoundError("SR:Q"),
    "ChannelUnreachableError": ChannelUnreachableError("SR:Q", "SR:Q timed out"),
    "ChannelWriteBlockedError": ChannelWriteBlockedError(
        "SR:Q",
        "WRITES_DISABLED",
        message="Writes are disabled for this deployment",
        verdict_reason="switch_off",
    ),
    "ChannelWriteBlockedError/limits": ChannelWriteBlockedError(
        "SR:Q", "LIMITS", message="Value 999.0 above maximum 500.0", violation=_LIMITS
    ),
    "ChannelWriteFailedError": ChannelWriteFailedError(
        "SR:Q", "mismatch", value_written=1.7, observed_value=1.5
    ),
    "ConnectorUnavailableError": ConnectorUnavailableError("the transport is gone"),
    "ConnectorInternalError": ConnectorInternalError(
        "connector raised KeyError('x')", class_tag="KeyError"
    ),
    "unregistered": KeyError("x"),
    "subclass": _OwnUnreachable("SR:Q", "SR:Q timed out in the connector's own words"),
}


def test_every_registered_class_has_a_pinned_error_frame():
    tags = {name.partition("/")[0] for name in _ERRORS}
    assert set(_exceptions.EXCEPTION_SPECS) <= tags


def test_every_record_a_frame_carries_has_a_pinned_result():
    tags = {name.partition("/")[0] for name in _RESULTS}
    assert set(frames._RECORDS) <= tags


def test_every_method_the_child_serves_has_a_pinned_request():
    methods = {name.partition("/")[0] for name in _REQUESTS} | {"disconnect"}
    assert methods == {*_host.PROXY_METHODS, _host.INIT_METHOD, "ping", "spawn_probe"}


@pytest.mark.parametrize("name", list(_REQUESTS))
async def test_a_request_is_written_as_pinned(name, monkeypatch):
    monkeypatch.setattr(frames, "new_request_id", lambda: _RID)
    assert await _sent(_REQUESTS[name]) == _PINNED[f"request:{name}"]


async def test_a_disconnect_request_is_written_as_pinned(monkeypatch):
    monkeypatch.setattr(frames, "new_request_id", lambda: _RID)
    assert await _disconnect_request() == _PINNED["request:disconnect"]


@pytest.mark.parametrize("name", list(_RESULTS))
def test_a_result_is_written_as_pinned(name):
    assert frames.encode_result(_RID, _RESULTS[name]) == _PINNED[f"result:{name}"]


@pytest.mark.parametrize("name", list(_ERRORS))
def test_an_error_frame_is_written_as_pinned(name):
    assert _host._error_frame(_RID, _ERRORS[name]) == _PINNED[f"error:{name}"]


def test_a_batch_failure_crosses_as_the_pinned_error_node():
    batch = ChannelReadBatch(
        failures={"SR:GONE": ChannelUnreachableError("SR:GONE", "SR:GONE did not answer")}
    )
    assert frames.encode_result(_RID, batch) == _PINNED["result:batch_failure_node"]


def test_the_protocol_version_is_the_pinned_one():
    assert frames.PROTOCOL_VERSION == 3


#: The pinned bytes, captured from the codec as it stood (see the module docstring).
_PINNED: dict[str, bytes] = {
    "request:read_channel": (
        b'OSPF\x00\x00\x00\xbd\x00\x00\x00\xb9{"kind": "request", "request_id": "req-0001", "m'
        b'ethod": "read_channel", "kwargs": {"__osprey_ipc__": "dict",'
        b' "items": [["address", "SR:DCCT"], ["timeout_s", 1.5]]}, "v"'
        b': 3, "blobs": []}'
    ),
    "request:read_channel/no_timeout": (
        b'OSPF\x00\x00\x00\xbe\x00\x00\x00\xba{"kind": "request", "request_id": "req-0001", "m'
        b'ethod": "read_channel", "kwargs": {"__osprey_ipc__": "dict",'
        b' "items": [["address", "SR:DCCT"], ["timeout_s", null]]}, "v'
        b'": 3, "blobs": []}'
    ),
    "request:read_multiple_channels": (
        b'OSPF\x00\x00\x00\xd0\x00\x00\x00\xcc{"kind": "request", "request_id": "req-0001", "m'
        b'ethod": "read_multiple_channels", "kwargs": {"__osprey_ipc__'
        b'": "dict", "items": [["addresses", ["SR:A", "SR:B"]], ["time'
        b'out_s", 2.0]]}, "v": 3, "blobs": []}'
    ),
    "request:write_channel": (
        b'OSPF\x00\x00\x00\xcc\x00\x00\x00\xc8{"kind": "request", "request_id": "req-0001", "m'
        b'ethod": "write_channel", "kwargs": {"__osprey_ipc__": "dict"'
        b', "items": [["address", "SR:Q"], ["value", 1.25], ["timeout_'
        b's", 3.0]]}, "v": 3, "blobs": []}'
    ),
    "request:write_channel/confirm_false": (
        b'OSPF\x00\x00\x00\xe0\x00\x00\x00\xdc{"kind": "request", "request_id": "req-0001", "m'
        b'ethod": "write_channel", "kwargs": {"__osprey_ipc__": "dict"'
        b', "items": [["address", "SR:Q"], ["value", 1.25], ["timeout_'
        b's", 3.0], ["confirm", false]]}, "v": 3, "blobs": []}'
    ),
    "request:write_channel/array": (
        b'OSPF\x00\x00\x01\x8f\x00\x00\x00\xf3{"kind": "request", "request_id": "req-0001", "m'
        b'ethod": "write_channel", "kwargs": {"__osprey_ipc__": "dict"'
        b', "items": [["address", "SR:WAVE"], ["value", {"__osprey_ipc'
        b'__": "ndarray", "blob": 0}], ["timeout_s", null]]}, "v": 3, '
        b"\"blobs\": [152]}\x93NUMPY\x01\x00v\x00{'descr': '<f8', 'fortran_order': F"
        b"alse, 'shape': (3,), }                                      "
        b"                      \n\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@"
    ),
    "request:write_multiple_channels": (
        b'OSPF\x00\x00\x00\xf2\x00\x00\x00\xee{"kind": "request", "request_id": "req-0001", "m'
        b'ethod": "write_multiple_channels", "kwargs": {"__osprey_ipc_'
        b'_": "dict", "items": [["operations", [["SR:A", 1], ["SR:B", '
        b'"on"]]], ["timeout_s", 4.0], ["confirm", true]]}, "v": 3, "b'
        b'lobs": []}'
    ),
    "request:validate_channel": (
        b'OSPF\x00\x00\x00\xaa\x00\x00\x00\xa6{"kind": "request", "request_id": "req-0001", "m'
        b'ethod": "validate_channel", "kwargs": {"__osprey_ipc__": "di'
        b'ct", "items": [["address", "SR:Q"]]}, "v": 3, "blobs": []}'
    ),
    "request:init": (
        b'OSPF\x00\x00\x02M\x00\x00\x02I{"kind": "request", "request_id": "req-0001", "m'
        b'ethod": "init", "kwargs": {"__osprey_ipc__": "dict", "items"'
        b': [["control_system", {"__osprey_ipc__": "dict", "items": [['
        b'"type", "epics"], ["connector", {"__osprey_ipc__": "dict", "'
        b'items": [["epics", {"__osprey_ipc__": "dict", "items": [["ga'
        b'teways", {"__osprey_ipc__": "dict", "items": [["read_only", '
        b'{"__osprey_ipc__": "dict", "items": [["address", "gw.example'
        b'"], ["port", 5064]]}], ["write_access", {"__osprey_ipc__": "'
        b'dict", "items": [["address", "gw.example"], ["port", 5084]]}'
        b']]}]]}]]}]]}], ["target", "live"]]}, "v": 3, "blobs": []}'
    ),
    "request:init/config_file": (
        b'OSPF\x00\x00\x02y\x00\x00\x02u{"kind": "request", "request_id": "req-0001", "m'
        b'ethod": "init", "kwargs": {"__osprey_ipc__": "dict", "items"'
        b': [["control_system", {"__osprey_ipc__": "dict", "items": [['
        b'"type", "epics"], ["connector", {"__osprey_ipc__": "dict", "'
        b'items": [["epics", {"__osprey_ipc__": "dict", "items": [["ga'
        b'teways", {"__osprey_ipc__": "dict", "items": [["read_only", '
        b'{"__osprey_ipc__": "dict", "items": [["address", "gw.example'
        b'"], ["port", 5064]]}], ["write_access", {"__osprey_ipc__": "'
        b'dict", "items": [["address", "gw.example"], ["port", 5084]]}'
        b']]}]]}]]}]]}], ["target", "live"], ["config_file", "/srv/pro'
        b'ject/config.yml"]]}, "v": 3, "blobs": []}'
    ),
    "request:init/execution_mode": (
        b'OSPF\x00\x00\x02m\x00\x00\x02i{"kind": "request", "request_id": "req-0001", "m'
        b'ethod": "init", "kwargs": {"__osprey_ipc__": "dict", "items"'
        b': [["control_system", {"__osprey_ipc__": "dict", "items": [['
        b'"type", "epics"], ["connector", {"__osprey_ipc__": "dict", "'
        b'items": [["epics", {"__osprey_ipc__": "dict", "items": [["ga'
        b'teways", {"__osprey_ipc__": "dict", "items": [["read_only", '
        b'{"__osprey_ipc__": "dict", "items": [["address", "gw.example'
        b'"], ["port", 5064]]}], ["write_access", {"__osprey_ipc__": "'
        b'dict", "items": [["address", "gw.example"], ["port", 5084]]}'
        b']]}]]}]]}]]}], ["target", "live"], ["execution_mode", "reado'
        b'nly"]]}, "v": 3, "blobs": []}'
    ),
    "request:init/config_file+execution_mode": (
        b'OSPF\x00\x00\x02\x99\x00\x00\x02\x95{"kind": "request", "request_id": "req-0001", "m'
        b'ethod": "init", "kwargs": {"__osprey_ipc__": "dict", "items"'
        b': [["control_system", {"__osprey_ipc__": "dict", "items": [['
        b'"type", "epics"], ["connector", {"__osprey_ipc__": "dict", "'
        b'items": [["epics", {"__osprey_ipc__": "dict", "items": [["ga'
        b'teways", {"__osprey_ipc__": "dict", "items": [["read_only", '
        b'{"__osprey_ipc__": "dict", "items": [["address", "gw.example'
        b'"], ["port", 5064]]}], ["write_access", {"__osprey_ipc__": "'
        b'dict", "items": [["address", "gw.example"], ["port", 5084]]}'
        b']]}]]}]]}]]}], ["target", "live"], ["config_file", "/srv/pro'
        b'ject/config.yml"], ["execution_mode", "readonly"]]}, "v": 3,'
        b' "blobs": []}'
    ),
    "request:ping": (
        b'OSPF\x00\x00\x00\x8b\x00\x00\x00\x87{"kind": "request", "request_id": "req-0001", "m'
        b'ethod": "ping", "kwargs": {"__osprey_ipc__": "dict", "items"'
        b': []}, "v": 3, "blobs": []}'
    ),
    "request:spawn_probe": (
        b'OSPF\x00\x00\x00\xbc\x00\x00\x00\xb8{"kind": "request", "request_id": "req-0001", "m'
        b'ethod": "spawn_probe", "kwargs": {"__osprey_ipc__": "dict", '
        b'"items": [["address", "SR:DCCT"], ["timeout_s", 5.0]]}, "v":'
        b' 3, "blobs": []}'
    ),
    "request:disconnect": (
        b'OSPF\x00\x00\x00\x91\x00\x00\x00\x8d{"kind": "request", "request_id": "req-0001", "m'
        b'ethod": "disconnect", "kwargs": {"__osprey_ipc__": "dict", "'
        b'items": []}, "v": 3, "blobs": []}'
    ),
    "result:Alarm": (
        b'OSPF\x00\x00\x00\x98\x00\x00\x00\x94{"kind": "result", "request_id": "req-0001", "va'
        b'lue": {"__osprey_ipc__": "Alarm", "fields": {"status": "LOLO'
        b'", "severity": 2}}, "v": 3, "blobs": []}'
    ),
    "result:ChannelValue": (
        b'OSPF\x00\x00\x02\xd8\x00\x00\x02\xd4{"kind": "result", "request_id": "req-0001", "va'
        b'lue": {"__osprey_ipc__": "ChannelValue", "fields": {"address'
        b'": "SR:Q", "value": 4.5, "sampled_at": {"__osprey_ipc__": "d'
        b'atetime", "iso": "2026-01-02T03:04:05.250000+00:00"}, "alarm'
        b'": {"__osprey_ipc__": "Alarm", "fields": {"status": "HIGH", '
        b'"severity": 1}}, "enum_label": null, "metadata": {"__osprey_'
        b'ipc__": "ChannelMetadata", "fields": {"units": "A", "precisi'
        b'on": 3, "description": null, "display_low": 0.0, "display_hi'
        b'gh": 10.0, "enum_labels": null, "raw_metadata": {"__osprey_i'
        b'pc__": "dict", "items": []}, "control_low": null, "control_h'
        b'igh": null, "warning_low": null, "warning_high": null, "alar'
        b'm_low": null, "alarm_high": null, "writable": null}}}}, "v":'
        b' 3, "blobs": []}'
    ),
    "result:ChannelValue/enum": (
        b'OSPF\x00\x00\x02\xb1\x00\x00\x02\xad{"kind": "result", "request_id": "req-0001", "va'
        b'lue": {"__osprey_ipc__": "ChannelValue", "fields": {"address'
        b'": "SR:MODE", "value": 1, "sampled_at": {"__osprey_ipc__": "'
        b'datetime", "iso": "2026-01-02T03:04:05.250000+00:00"}, "alar'
        b'm": null, "enum_label": "On", "metadata": {"__osprey_ipc__":'
        b' "ChannelMetadata", "fields": {"units": "", "precision": nul'
        b'l, "description": null, "display_low": null, "display_high":'
        b' null, "enum_labels": ["Off", "On"], "raw_metadata": {"__osp'
        b'rey_ipc__": "dict", "items": [["DESC", "mode"]]}, "control_l'
        b'ow": null, "control_high": null, "warning_low": null, "warni'
        b'ng_high": null, "alarm_low": null, "alarm_high": null, "writ'
        b'able": null}}}}, "v": 3, "blobs": []}'
    ),
    "result:ChannelValue/array": (
        b'OSPF\x00\x00\x03\x8c\x00\x00\x02\xfc{"kind": "result", "request_id": "req-0001", "va'
        b'lue": {"__osprey_ipc__": "ChannelValue", "fields": {"address'
        b'": "SR:Q", "value": {"__osprey_ipc__": "ndarray", "blob": 0}'
        b', "sampled_at": {"__osprey_ipc__": "datetime", "iso": "2026-'
        b'01-02T03:04:05.250000+00:00"}, "alarm": {"__osprey_ipc__": "'
        b'Alarm", "fields": {"status": "HIGH", "severity": 1}}, "enum_'
        b'label": null, "metadata": {"__osprey_ipc__": "ChannelMetadat'
        b'a", "fields": {"units": "A", "precision": 3, "description": '
        b'null, "display_low": 0.0, "display_high": 10.0, "enum_labels'
        b'": null, "raw_metadata": {"__osprey_ipc__": "dict", "items":'
        b' []}, "control_low": null, "control_high": null, "warning_lo'
        b'w": null, "warning_high": null, "alarm_low": null, "alarm_hi'
        b'gh": null, "writable": null}}}}, "v": 3, "blobs": [140]}\x93NUM'
        b"PY\x01\x00v\x00{'descr': '<i4', 'fortran_order': False, 'shape': (3,)"
        b", }                                                         "
        b"   \n\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00"
    ),
    "result:ChannelMetadata": (
        b'OSPF\x00\x00\x01\xe4\x00\x00\x01\xe0{"kind": "result", "request_id": "req-0001", "va'
        b'lue": {"__osprey_ipc__": "ChannelMetadata", "fields": {"unit'
        b's": "mm", "precision": 2, "description": "horizontal positio'
        b'n", "display_low": -5.0, "display_high": 5.0, "enum_labels":'
        b' null, "raw_metadata": {"__osprey_ipc__": "dict", "items": ['
        b'["EGU", "mm"], ["PREC", 2]]}, "control_low": -4.0, "control_'
        b'high": 4.0, "warning_low": -3.0, "warning_high": 3.0, "alarm'
        b'_low": -4.5, "alarm_high": 4.5, "writable": true}}, "v": 3, '
        b'"blobs": []}'
    ),
    "result:ChannelReadBatch": (
        b'OSPF\x00\x00\x05\x12\x00\x00\x05\x0e{"kind": "result", "request_id": "req-0001", "va'
        b'lue": {"__osprey_ipc__": "ChannelReadBatch", "fields": {"val'
        b'ues": {"__osprey_ipc__": "dict", "items": [["SR:Q", {"__ospr'
        b'ey_ipc__": "ChannelValue", "fields": {"address": "SR:Q", "va'
        b'lue": 4.5, "sampled_at": {"__osprey_ipc__": "datetime", "iso'
        b'": "2026-01-02T03:04:05.250000+00:00"}, "alarm": {"__osprey_'
        b'ipc__": "Alarm", "fields": {"status": "HIGH", "severity": 1}'
        b'}, "enum_label": null, "metadata": {"__osprey_ipc__": "Chann'
        b'elMetadata", "fields": {"units": "A", "precision": 3, "descr'
        b'iption": null, "display_low": 0.0, "display_high": 10.0, "en'
        b'um_labels": null, "raw_metadata": {"__osprey_ipc__": "dict",'
        b' "items": []}, "control_low": null, "control_high": null, "w'
        b'arning_low": null, "warning_high": null, "alarm_low": null, '
        b'"alarm_high": null, "writable": null}}}}]]}, "failures": {"_'
        b'_osprey_ipc__": "dict", "items": [["SR:GONE", {"__osprey_ipc'
        b'__": "error", "class_tag": "ChannelUnreachableError", "messa'
        b'ge": "SR:GONE did not answer in 2.0s", "fields": {"__osprey_'
        b'ipc__": "dict", "items": [["address", "SR:GONE"]]}}], ["SR:N'
        b'OPE", {"__osprey_ipc__": "error", "class_tag": "ChannelNotFo'
        b'undError", "message": "SR:NOPE is not a channel of this faci'
        b'lity", "fields": {"__osprey_ipc__": "dict", "items": [["addr'
        b'ess", "SR:NOPE"]]}}]]}}}, "v": 3, "blobs": []}'
    ),
    "result:ChannelWriteResult/confirmed": (
        b'OSPF\x00\x00\x03\xcb\x00\x00\x03\xc7{"kind": "result", "request_id": "req-0001", "va'
        b'lue": {"__osprey_ipc__": "ChannelWriteResult", "fields": {"a'
        b'ddress": "SR:Q", "value_written": 4.5, "outcome": "confirmed'
        b'", "refusal_reason": null, "error_message": null, "observed"'
        b': {"__osprey_ipc__": "ChannelValue", "fields": {"address": "'
        b'SR:Q", "value": 4.5, "sampled_at": {"__osprey_ipc__": "datet'
        b'ime", "iso": "2026-01-02T03:04:05.250000+00:00"}, "alarm": {'
        b'"__osprey_ipc__": "Alarm", "fields": {"status": "HIGH", "sev'
        b'erity": 1}}, "enum_label": null, "metadata": {"__osprey_ipc_'
        b'_": "ChannelMetadata", "fields": {"units": "A", "precision":'
        b' 3, "description": null, "display_low": 0.0, "display_high":'
        b' 10.0, "enum_labels": null, "raw_metadata": {"__osprey_ipc__'
        b'": "dict", "items": []}, "control_low": null, "control_high"'
        b': null, "warning_low": null, "warning_high": null, "alarm_lo'
        b'w": null, "alarm_high": null, "writable": null}}}}, "notes":'
        b' null, "verdict_reason": null, "limits_violation": null}}, "'
        b'v": 3, "blobs": []}'
    ),
    "result:ChannelWriteResult/refused": (
        b'OSPF\x00\x00\x02*\x00\x00\x02&{"kind": "result", "request_id": "req-0001", "va'
        b'lue": {"__osprey_ipc__": "ChannelWriteResult", "fields": {"a'
        b'ddress": "SR:Q", "value_written": 999.0, "outcome": "refused'
        b'", "refusal_reason": "LIMITS", "error_message": "Value 999.0'
        b' above maximum 500.0", "observed": null, "notes": null, "ver'
        b'dict_reason": null, "limits_violation": {"__osprey_ipc__": "'
        b'LimitsViolation", "fields": {"kind": "MAX_EXCEEDED", "reason'
        b'": "Value 999.0 above maximum 500.0", "min_value": 0.0, "max'
        b'_value": 500.0, "max_step": null, "current_value": 120.0}}}}'
        b', "v": 3, "blobs": []}'
    ),
    "result:ChannelWriteResult/list": (
        b'OSPF\x00\x00\x02U\x00\x00\x02Q{"kind": "result", "request_id": "req-0001", "va'
        b'lue": [{"__osprey_ipc__": "ChannelWriteResult", "fields": {"'
        b'address": "SR:A", "value_written": 1, "outcome": "unrequeste'
        b'd", "refusal_reason": null, "error_message": null, "observed'
        b'": null, "notes": null, "verdict_reason": null, "limits_viol'
        b'ation": null}}, {"__osprey_ipc__": "ChannelWriteResult", "fi'
        b'elds": {"address": "SR:B", "value_written": "on", "outcome":'
        b' "failed", "refusal_reason": null, "error_message": "the IOC'
        b' refused the put", "observed": null, "notes": null, "verdict'
        b'_reason": null, "limits_violation": null}}], "v": 3, "blobs"'
        b": []}"
    ),
    "result:LimitsViolation": (
        b'OSPF\x00\x00\x01\x16\x00\x00\x01\x12{"kind": "result", "request_id": "req-0001", "va'
        b'lue": {"__osprey_ipc__": "LimitsViolation", "fields": {"kind'
        b'": "MAX_EXCEEDED", "reason": "Value 999.0 above maximum 500.'
        b'0", "min_value": 0.0, "max_value": 500.0, "max_step": null, '
        b'"current_value": 120.0}}, "v": 3, "blobs": []}'
    ),
    "result:post_connect_report": (
        b'OSPF\x00\x00\x02\x0f\x00\x00\x02\x0b{"kind": "result", "request_id": "req-0001", "va'
        b'lue": {"__osprey_ipc__": "dict", "items": [["selected_role",'
        b' "read_only"], ["mode", "gateway"], ["host", "gw.example"], '
        b'["port", 5064], ["_epics_configured", true], ["connector_typ'
        b'e", "epics"], ["transport", "ca"], ["target", "live"], ["wri'
        b'tes_enabled", false], ["readonly_run", true], ["authorizatio'
        b'n", null], ["epics_env", {"__osprey_ipc__": "dict", "items":'
        b' [["EPICS_CA_ADDR_LIST", "gw.example"], ["EPICS_CA_AUTO_ADDR'
        b'_LIST", "NO"]]}], ["pid", 4242]]}, "v": 3, "blobs": []}'
    ),
    "result:ping": (
        b'OSPF\x00\x00\x00T\x00\x00\x00P{"kind": "result", "request_id": "req-0001", "va'
        b'lue": 4242, "v": 3, "blobs": []}'
    ),
    "result:validate_channel": (
        b'OSPF\x00\x00\x00T\x00\x00\x00P{"kind": "result", "request_id": "req-0001", "va'
        b'lue": true, "v": 3, "blobs": []}'
    ),
    "result:disconnect": (
        b'OSPF\x00\x00\x00T\x00\x00\x00P{"kind": "result", "request_id": "req-0001", "va'
        b'lue": null, "v": 3, "blobs": []}'
    ),
    "result:numpy_scalar": (
        b'OSPF\x00\x00\x01\x00\x00\x00\x00x{"kind": "result", "request_id": "req-0001", "va'
        b'lue": {"__osprey_ipc__": "npscalar", "blob": 0}, "v": 3, "bl'
        b"obs\": [132]}\x93NUMPY\x01\x00v\x00{'descr': '<f4', 'fortran_order': Fals"
        b"e, 'shape': (), }                                           "
        b"                   \n\x00\x00 @"
    ),
    "result:batch_failure_node": (
        b'OSPF\x00\x00\x01\xa8\x00\x00\x01\xa4{"kind": "result", "request_id": "req-0001", "va'
        b'lue": {"__osprey_ipc__": "ChannelReadBatch", "fields": {"val'
        b'ues": {"__osprey_ipc__": "dict", "items": []}, "failures": {'
        b'"__osprey_ipc__": "dict", "items": [["SR:GONE", {"__osprey_i'
        b'pc__": "error", "class_tag": "ChannelUnreachableError", "mes'
        b'sage": "SR:GONE did not answer", "fields": {"__osprey_ipc__"'
        b': "dict", "items": [["address", "SR:GONE"]]}}]]}}}, "v": 3, '
        b'"blobs": []}'
    ),
    "error:ConnectionError": (
        b'OSPF\x00\x00\x00\xb9\x00\x00\x00\xb5{"kind": "error", "request_id": "req-0001", "cla'
        b'ss_tag": "ConnectionError", "message": "gateway unreachable"'
        b', "fields": {"__osprey_ipc__": "dict", "items": []}, "v": 3,'
        b' "blobs": []}'
    ),
    "error:TimeoutError": (
        b'OSPF\x00\x00\x00\xbc\x00\x00\x00\xb8{"kind": "error", "request_id": "req-0001", "cla'
        b'ss_tag": "TimeoutError", "message": "read timed out after 2.'
        b'0s", "fields": {"__osprey_ipc__": "dict", "items": []}, "v":'
        b' 3, "blobs": []}'
    ),
    "error:ConnectRefusedError": (
        b'OSPF\x00\x00\x00\xd1\x00\x00\x00\xcd{"kind": "error", "request_id": "req-0001", "cla'
        b'ss_tag": "ConnectRefusedError", "message": "no read-only gat'
        b'eway to connect through", "fields": {"__osprey_ipc__": "dict'
        b'", "items": []}, "v": 3, "blobs": []}'
    ),
    "error:ChannelError": (
        b'OSPF\x00\x00\x00\xcd\x00\x00\x00\xc9{"kind": "error", "request_id": "req-0001", "cla'
        b'ss_tag": "ChannelError", "message": "SR:Q failed its own way'
        b'", "fields": {"__osprey_ipc__": "dict", "items": [["address"'
        b', "SR:Q"]]}, "v": 3, "blobs": []}'
    ),
    "error:ChannelLimitsViolationError": (
        b'OSPF\x00\x00\x03\xb2\x00\x00\x03\xae{"kind": "error", "request_id": "req-0001", "cla'
        b'ss_tag": "ChannelLimitsViolationError", "message": "\\n======'
        b"============================================================"
        b"====\\nCHANNEL LIMITS VIOLATION DETECTED\\n==================="
        b"===================================================\\nChannel"
        b" Address: SR:Q\\nAttempted Value: 999.0\\nCurrent Value: 120.0"
        b"\\nViolation: Value 999.0 above maximum 500.0\\nAllowed Range:"
        b" [0.0, 500.0]\\nMaximum Step Size: 10.0\\n===================="
        b"==================================================\\n\\u26a0\\u"
        b"fe0f  Write operation BLOCKED for safety\\n=================="
        b'====================================================", "fiel'
        b'ds": {"__osprey_ipc__": "dict", "items": [["address", "SR:Q"'
        b'], ["attempted_value", 999.0], ["violation_type", "MAX_EXCEE'
        b'DED"], ["violation_reason", "Value 999.0 above maximum 500.0'
        b'"], ["min_value", 0.0], ["max_value", 500.0], ["max_step", 1'
        b'0.0], ["current_value", 120.0]]}, "v": 3, "blobs": []}'
    ),
    "error:ChannelNotFoundError": (
        b'OSPF\x00\x00\x00\xe4\x00\x00\x00\xe0{"kind": "error", "request_id": "req-0001", "cla'
        b'ss_tag": "ChannelNotFoundError", "message": "SR:Q is not a c'
        b'hannel of this facility", "fields": {"__osprey_ipc__": "dict'
        b'", "items": [["address", "SR:Q"]]}, "v": 3, "blobs": []}'
    ),
    "error:ChannelUnreachableError": (
        b'OSPF\x00\x00\x00\xcf\x00\x00\x00\xcb{"kind": "error", "request_id": "req-0001", "cla'
        b'ss_tag": "ChannelUnreachableError", "message": "SR:Q timed o'
        b'ut", "fields": {"__osprey_ipc__": "dict", "items": [["addres'
        b's", "SR:Q"]]}, "v": 3, "blobs": []}'
    ),
    "error:ChannelWriteBlockedError": (
        b'OSPF\x00\x00\x01?\x00\x00\x01;{"kind": "error", "request_id": "req-0001", "cla'
        b'ss_tag": "ChannelWriteBlockedError", "message": "Writes are '
        b'disabled for this deployment", "fields": {"__osprey_ipc__": '
        b'"dict", "items": [["address", "SR:Q"], ["reason", "WRITES_DI'
        b'SABLED"], ["verdict_reason", "switch_off"], ["violation", nu'
        b'll]]}, "v": 3, "blobs": []}'
    ),
    "error:ChannelWriteBlockedError/limits": (
        b'OSPF\x00\x00\x01\xe8\x00\x00\x01\xe4{"kind": "error", "request_id": "req-0001", "cla'
        b'ss_tag": "ChannelWriteBlockedError", "message": "Value 999.0'
        b' above maximum 500.0", "fields": {"__osprey_ipc__": "dict", '
        b'"items": [["address", "SR:Q"], ["reason", "LIMITS"], ["verdi'
        b'ct_reason", null], ["violation", {"__osprey_ipc__": "LimitsV'
        b'iolation", "fields": {"kind": "MAX_EXCEEDED", "reason": "Val'
        b'ue 999.0 above maximum 500.0", "min_value": 0.0, "max_value"'
        b': 500.0, "max_step": null, "current_value": 120.0}}]]}, "v":'
        b' 3, "blobs": []}'
    ),
    "error:ChannelWriteFailedError": (
        b'OSPF\x00\x00\x01P\x00\x00\x01L{"kind": "error", "request_id": "req-0001", "cla'
        b'ss_tag": "ChannelWriteFailedError", "message": "Write to \'SR'
        b":Q' not confirmed (MISMATCH): sent 1.7, channel holds 1.5\", "
        b'"fields": {"__osprey_ipc__": "dict", "items": [["address", "'
        b'SR:Q"], ["outcome", "mismatch"], ["value_written", 1.7], ["o'
        b'bserved_value", 1.5]]}, "v": 3, "blobs": []}'
    ),
    "error:ConnectorUnavailableError": (
        b'OSPF\x00\x00\x00\xc5\x00\x00\x00\xc1{"kind": "error", "request_id": "req-0001", "cla'
        b'ss_tag": "ConnectorUnavailableError", "message": "the transp'
        b'ort is gone", "fields": {"__osprey_ipc__": "dict", "items": '
        b'[]}, "v": 3, "blobs": []}'
    ),
    "error:ConnectorInternalError": (
        b'OSPF\x00\x00\x00\xe4\x00\x00\x00\xe0{"kind": "error", "request_id": "req-0001", "cla'
        b'ss_tag": "ConnectorInternalError", "message": "connector rai'
        b'sed KeyError(\'x\')", "fields": {"__osprey_ipc__": "dict", "it'
        b'ems": [["class_tag", "KeyError"]]}, "v": 3, "blobs": []}'
    ),
    "error:unregistered": (
        b'OSPF\x00\x00\x01\x02\x00\x00\x00\xfe{"kind": "error", "request_id": "req-0001", "cla'
        b'ss_tag": "ConnectorInternalError", "message": "connector hos'
        b't reported an unrecognised error: KeyError(\'x\')", "fields": '
        b'{"__osprey_ipc__": "dict", "items": [["class_tag", "KeyError'
        b'"]]}, "v": 3, "blobs": []}'
    ),
    "error:subclass": (
        b'OSPF\x00\x00\x00\xec\x00\x00\x00\xe8{"kind": "error", "request_id": "req-0001", "cla'
        b'ss_tag": "ChannelUnreachableError", "message": "SR:Q timed o'
        b'ut in the connector\'s own words", "fields": {"__osprey_ipc__'
        b'": "dict", "items": [["address", "SR:Q"]]}, "v": 3, "blobs":'
        b" []}"
    ),
}
