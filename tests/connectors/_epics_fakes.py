"""Shared fakes and fixtures for the EPICS / PVA / VA connector tests.

One stand-in for the one client library the EPICS connector uses: a fake
``pvaccess`` (pvapy) module, serving both transports the way pvapy does — a
``Channel(address, provider)`` whose provider is ``CA`` or ``PVA`` — plus
connectors wired without going through ``connect()``.

The fake is installed with ``monkeypatch.setitem(sys.modules, "pvaccess", ...)``
(the :func:`fake_pvaccess` fixture), so ``connect()``'s lazy ``import pvaccess``
picks it up; the builders below inject it directly instead. It records every
call it receives in :attr:`FakePvaccess.log`, so a test asserts on the request
string, the provider and the timeout a call actually carried — never merely
that a call "didn't raise".

What the fake models, because the connector depends on it:

* **Channels** — ``get(request)``, ``put(value[, request])``, ``setTimeout``,
  ``isConnected``, and the monitor quartet ``subscribe`` / ``startMonitor`` /
  ``stopMonitor`` / ``unsubscribe``. A channel nobody serves raises the exact
  text pvapy raises for it (``"Channel X timed out."``).
* **pvRequests** — ``field(a,b)`` returns only the top-level fields named, and
  ``field()`` returns them all, so the Channel Access split read (value, alarm
  and timestamp in one get, ``display`` in another) and PVA's field-limited
  metadata get behave as they do against a server.
* **Results** — a :class:`FakePvObject` exposing what the connector calls on a
  pvapy ``PvObject``: ``keys()``, ``obj[name]``, ``getStructureDict()`` and
  ``toDict()``. A union is spelled as pvapy spells it, a ``(value_dict,
  type_dict)`` tuple.

Fixtures defined here reach a test module only when that module imports them,
for example ``from tests.connectors._epics_fakes import fake_pvaccess  # noqa: F401``.
"""

from __future__ import annotations

import re
import sys
import threading
import types
from collections.abc import Callable
from typing import Any

import pytest

from osprey.connectors.control_system.epics_connector import EPICSConnector

EPICS_CA_VARS = (
    "EPICS_CA_ADDR_LIST",
    "EPICS_CA_SERVER_PORT",
    "EPICS_CA_NAME_SERVERS",
    "EPICS_CA_AUTO_ADDR_LIST",
)
EPICS_PVA_VARS = (
    "EPICS_PVA_ADDR_LIST",
    "EPICS_PVA_NAME_SERVERS",
    "EPICS_PVA_AUTO_ADDR_LIST",
)

PVA_GLOB = "SR:CAM*:IMAGE"

#: The pvRequests the connector sends, spelled once for the assertions.
CA_READ_REQUEST = "field(value,alarm,timeStamp)"
CA_DISPLAY_REQUEST = "field(display)"
PVA_READ_REQUEST = "field()"
PVA_METADATA_REQUEST = "field(alarm,timeStamp,display)"
VALUE_REQUEST = "field(value)"
CONFIRMING_PUT_REQUEST = "record[block=true]field(value)"

#: A record's processing time, as a POSIX timestamp.
TIMESTAMP = 1_750_000_000


# ---------------------------------------------------------------------------
# Environment and posture
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_epics_env(monkeypatch):
    """Start each test with no EPICS_CA_* or EPICS_PVA_* variable set.

    This only removes variables the host had set. ``monkeypatch.delenv`` records
    nothing for a variable that is absent, so it cannot undo a value that
    ``connect()`` later writes into ``os.environ``. The suite-wide autouse
    ``restore_environ`` fixture (``tests/conftest.py``) restores those writes.
    """
    for var in EPICS_CA_VARS + EPICS_PVA_VARS:
        monkeypatch.delenv(var, raising=False)


def patch_writes_enabled(monkeypatch, enabled: bool) -> None:
    """Answer the deployment-wide ``control_system.writes_enabled`` key with ``enabled``."""

    def fake_get_config_value(key, default=None):
        if key == "control_system.writes_enabled":
            return enabled
        return default

    monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)


@pytest.fixture
def writes_enabled(monkeypatch):
    """Open the base-class writes gate so ``write_channel`` reaches its real body.

    ``ControlSystemConnector.__init_subclass__`` wraps ``write_channel`` with a
    ``_writes_enabled`` pre-check that is False in a config-less test
    environment; the tests using this are about what happens after that gate.
    """
    monkeypatch.setattr(EPICSConnector, "_writes_enabled", property(lambda self: True))


# ---------------------------------------------------------------------------
# pvapy's exception and its two classified texts
# ---------------------------------------------------------------------------


class FakePvaException(Exception):
    """Stands in for ``pvaccess.PvaException``, pvapy's one exception type."""


def timed_out(address: str) -> FakePvaException:
    """The failure pvapy raises for a channel that did not connect or answer."""
    return FakePvaException(f"Channel {address} timed out.")


def access_denied(address: str) -> FakePvaException:
    """The failure pvapy raises for a put the IOC's access security refused."""
    return FakePvaException(f"channel {address} PvaClientPut::put Write access denied")


# ---------------------------------------------------------------------------
# PvObject
# ---------------------------------------------------------------------------


class Unconvertible:
    """A field value pvapy cannot hand over: reading it with ``obj[name]`` raises."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error or TypeError("No to_python (by-value) converter found")


def _structure_of(value: Any) -> Any:
    """pvapy's introspection spelling of a value: types only, never data.

    A sub-structure is a dict, a union a tuple, a structure array a
    one-element list; anything else is named by its Python type.
    """
    if isinstance(value, dict):
        return {key: _structure_of(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return value[1:] if len(value) > 1 else ({},)
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return [_structure_of(value[0])]
    if isinstance(value, Unconvertible):
        return "unconvertible"
    return type(value).__name__


class FakePvObject:
    """A pvapy ``PvObject`` stand-in over plain top-level fields.

    ``structure`` overrides the introspection dict (``None`` derives it from
    the fields); ``structure_error`` makes ``getStructureDict`` raise, which the
    connector must survive.
    """

    def __init__(
        self,
        fields: dict[str, Any],
        *,
        structure: dict[str, Any] | None = None,
        structure_error: BaseException | None = None,
    ) -> None:
        self._fields = dict(fields)
        self._structure = structure
        self._structure_error = structure_error

    def keys(self) -> list[str]:
        return list(self._fields)

    def __getitem__(self, name: str) -> Any:
        value = self._fields[name]
        if isinstance(value, Unconvertible):
            raise value.error
        return value

    def getStructureDict(self) -> dict[str, Any]:  # pvapy's spelling
        if self._structure_error is not None:
            raise self._structure_error
        if self._structure is not None:
            return self._structure
        return {key: _structure_of(value) for key, value in self._fields.items()}

    def toDict(self) -> dict[str, Any]:  # pvapy's spelling
        return {k: v for k, v in self._fields.items() if not isinstance(v, Unconvertible)}


_FIELD_LIST = re.compile(r"field\(([^)]*)\)")


def _select(fields: dict[str, Any], request: str | None) -> dict[str, Any]:
    """The top-level fields a pvRequest asks for: ``field()`` is all of them."""
    match = _FIELD_LIST.search(request or "")
    if match is None or not match.group(1).strip():
        return dict(fields)
    wanted = [name.strip() for name in match.group(1).split(",") if name.strip()]
    return {name: fields[name] for name in wanted if name in fields}


# ---------------------------------------------------------------------------
# Normative-type field builders
# ---------------------------------------------------------------------------


def alarm(severity: int = 0, status: int = 0, message: str = "") -> dict[str, Any]:
    return {"severity": severity, "status": status, "message": message}


def time_stamp(seconds: float = TIMESTAMP, nanoseconds: int = 0) -> dict[str, Any]:
    return {"secondsPastEpoch": seconds, "nanoseconds": nanoseconds, "userTag": 0}


def record(
    value: Any = 1.0,
    *,
    units: str = "mA",
    fmt: str = "F9.3",
    description: str = "",
    severity: int = 0,
    status: int = 0,
    message: str = "",
    seconds: float = TIMESTAMP,
    nanoseconds: int = 0,
    limit_low: float = 0.0,
    limit_high: float = 10.0,
) -> dict[str, Any]:
    """The fields pvapy serves for an analog record (or a PVA NTScalar).

    ``display`` carries its precision inside ``format``, the way pvapy's
    Channel Access provider spells it (``"F9.3"`` is 3 decimals).
    """
    return {
        "value": value,
        "alarm": alarm(severity, status, message),
        "timeStamp": time_stamp(seconds, nanoseconds),
        "display": {
            "limitLow": limit_low,
            "limitHigh": limit_high,
            "description": description,
            "format": fmt,
            "units": units,
        },
    }


def enum_record(
    index: int = 0,
    choices: list[str] | tuple[str, ...] = ("OFF", "ON"),
    *,
    severity: int = 0,
    message: str = "",
) -> dict[str, Any]:
    """An enum (a CA mbbo/bo, or an NTEnum): ``value`` is ``{index, choices}``.

    No ``display``: pvapy's CA provider serves none for an enum record.
    """
    return {
        "value": {"index": index, "choices": list(choices)},
        "alarm": alarm(severity, 0, message),
        "timeStamp": time_stamp(),
    }


def ndarray_record(
    array: Any,
    dimensions: list[int],
    *,
    member: str = "ushortValue",
    codec: str = "",
    color_mode: int = 0,
    units: str = "",
) -> dict[str, Any]:
    """An NTNDArray: the payload in a union, its dimensions innermost first."""
    return {
        "value": ({member: array}, {member: ["USHORT"]}),
        "codec": {"name": codec, "parameters": ({}, {})},
        "compressedSize": 0,
        "uncompressedSize": 0,
        "dimension": [
            {"size": size, "offset": 0, "fullSize": size, "binning": 1, "reverse": False}
            for size in dimensions
        ],
        "uniqueId": 7,
        "attribute": [{"name": "ColorMode", "value": ({"value": color_mode}, {"value": "INT"})}],
        "alarm": alarm(),
        "timeStamp": time_stamp(),
        "display": {
            "limitLow": 0.0,
            "limitHigh": 0.0,
            "description": "",
            "format": "",
            "units": units,
        },
    }


# ---------------------------------------------------------------------------
# The fake pvaccess module
# ---------------------------------------------------------------------------


class FakeChannel:
    """A pvapy ``Channel`` over the fake module's served records."""

    def __init__(self, module: FakePvaccess, address: str, provider: str) -> None:
        self.module = module
        self.address = address
        self.provider = provider
        self.timeout: float | None = None
        self.connected = False
        self.subscribers: dict[str, Callable[[Any], None]] = {}
        self.monitor_request: str | None = None
        self.monitoring = False

    # -- connection -------------------------------------------------------

    def setTimeout(self, timeout: float) -> None:  # pvapy's spelling
        self.timeout = timeout

    def isConnected(self) -> bool:  # pvapy's spelling
        return self.connected

    def _record(self) -> dict[str, Any]:
        fields = self.module.served.get(self.address)
        if fields is None:
            raise timed_out(self.address)
        self.connected = True
        return fields

    # -- get / put --------------------------------------------------------

    def get(self, request: str = "field(value)") -> FakePvObject:
        module = self.module
        module._note("get", self, request)
        hook = module.get_hooks.get(self.address)
        if hook is not None:
            answer = hook(request)
            if answer is not None:
                self.connected = True
                return answer if isinstance(answer, FakePvObject) else FakePvObject(answer)
        return FakePvObject(_select(self._record(), request))

    def put(self, value: Any, request: str | None = None) -> None:
        module = self.module
        module._note("put", self, request, value=value)
        hook = module.put_hooks.get(self.address)
        if hook is not None:
            hook(value, request)
            return
        fields = self._record()
        current = fields.get("value")
        if isinstance(current, dict) and "index" in current:
            choices = current.get("choices") or []
            index = choices.index(value) if isinstance(value, str) else int(value)
            fields["value"] = {**current, "index": index}
        else:
            fields["value"] = value

    # -- monitor ----------------------------------------------------------

    def subscribe(self, name: str, callback: Callable[[Any], None]) -> None:
        self.subscribers[name] = callback

    def unsubscribe(self, name: str) -> None:
        self.module._note("unsubscribe", self, name)
        self.subscribers.pop(name, None)

    def startMonitor(self, request: str = "") -> None:  # pvapy's spelling
        self.module._note("startMonitor", self, request)
        error = self.module.monitor_errors.get(self.address)
        if error is not None:
            raise error
        self.monitor_request = request
        self.monitoring = True
        fields = self.module.served.get(self.address)
        if fields is not None:  # the first update is the current value
            self.fire(fields)

    def stopMonitor(self) -> None:  # pvapy's spelling
        self.module._note("stopMonitor", self, None)
        self.monitoring = False

    def fire(self, update: dict[str, Any] | FakePvObject) -> None:
        """Deliver one update to every subscriber, as pvapy's monitor thread would."""
        if not self.monitoring:
            return
        obj = (
            update
            if isinstance(update, FakePvObject)
            else FakePvObject(_select(update, self.monitor_request))
        )
        for callback in list(self.subscribers.values()):
            callback(obj)


class FakePvaccess(types.ModuleType):
    """A ``pvaccess`` module: ``Channel``, ``CA``, ``PVA`` and ``PvaException``.

    ``served`` maps an address to its record's top-level fields (mutated by a
    put); an address it does not name is unreachable. ``get_hooks`` /
    ``put_hooks`` replace the default behaviour for one address — a get hook
    returning ``None`` falls through to the served record, and either may
    raise or sleep. ``monitor_errors`` makes ``startMonitor`` raise.
    """

    CA = "CA"
    PVA = "PVA"
    PvaException = FakePvaException

    def __init__(self) -> None:
        super().__init__("pvaccess")
        self.served: dict[str, dict[str, Any]] = {}
        self.get_hooks: dict[str, Callable[[str], Any]] = {}
        self.put_hooks: dict[str, Callable[[Any, str | None], None]] = {}
        self.monitor_errors: dict[str, BaseException] = {}
        self.channels: list[FakeChannel] = []
        self.log: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def Channel(self, address: str, provider: str = "PVA") -> FakeChannel:  # pvapy's spelling
        channel = FakeChannel(self, address, provider)
        with self._lock:
            self.channels.append(channel)
        return channel

    def serve(self, address: str, fields: dict[str, Any]) -> dict[str, Any]:
        """Serve ``fields`` at ``address``; returns the live dict a put mutates."""
        self.served[address] = fields
        return fields

    def _note(self, op: str, channel: FakeChannel, request: str | None, **extra: Any) -> None:
        entry = {
            "op": op,
            "address": channel.address,
            "provider": channel.provider,
            "request": request,
            "timeout": channel.timeout,
            **extra,
        }
        with self._lock:
            self.log.append(entry)

    # -- assertion helpers --------------------------------------------------

    def calls(self, op: str | None = None, **match: Any) -> list[dict[str, Any]]:
        """The logged calls of kind ``op`` whose fields equal every ``match`` item."""
        return [
            entry
            for entry in self.log
            if (op is None or entry["op"] == op)
            and all(entry.get(key) == value for key, value in match.items())
        ]

    def channels_for(self, address: str) -> list[FakeChannel]:
        return [channel for channel in self.channels if channel.address == address]


def install_fake_pvaccess(monkeypatch) -> FakePvaccess:
    """Put a :class:`FakePvaccess` in ``sys.modules`` for ``connect()`` to import."""
    module = FakePvaccess()
    monkeypatch.setitem(sys.modules, "pvaccess", module)
    return module


@pytest.fixture
def fake_pvaccess(monkeypatch) -> FakePvaccess:
    """:func:`install_fake_pvaccess` as a fixture; yields the stand-in module."""
    return install_fake_pvaccess(monkeypatch)


# ---------------------------------------------------------------------------
# Connectors wired without connect()
# ---------------------------------------------------------------------------


def ca_connector(
    pvaccess: FakePvaccess | None = None,
    *,
    limits_validator: Any = None,
    timeout: float = 5.0,
    globs: tuple[str, ...] | list[str] = (),
) -> EPICSConnector:
    """A connector that skips ``connect()`` by injecting its runtime state.

    ``pvaccess`` is the client module it holds (a fresh :class:`FakePvaccess`
    when not given; reach it again as ``connector._pvaccess``).
    """
    connector = EPICSConnector()
    connector._pvaccess = pvaccess if pvaccess is not None else FakePvaccess()
    connector._limits_validator = limits_validator
    connector._timeout = timeout
    connector._pva_channel_globs = list(globs)
    connector._connected = True
    connector._epics_configured = True
    return connector


def pva_connector(
    pvaccess: FakePvaccess | None = None,
    *,
    globs: tuple[str, ...] | list[str] = (PVA_GLOB,),
    timeout: float = 3.0,
    limits_validator: Any = None,
) -> EPICSConnector:
    """A connector routing ``globs`` over PVAccess, without ``connect()``."""
    return ca_connector(pvaccess, limits_validator=limits_validator, timeout=timeout, globs=globs)
