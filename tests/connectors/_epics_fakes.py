"""Shared fakes and fixtures for the EPICS / PVA / VA connector tests.

One copy of the stand-ins that the EPICS connector test files had each written
for themselves: a fake ``aioca`` and an inline CA loop for the connector's
Channel Access path, a fake ``p4p`` package, a p4p ``Value`` stand-in, and
connectors wired without going through ``connect()``.

Fixtures defined here reach a test module only when that module imports them,
for example ``from tests.connectors._epics_fakes import fake_aioca  # noqa: F401``.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import aioca
import numpy
import pytest

from osprey_connectors.control_system.epics_connector import EPICSConnector

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

    monkeypatch.setattr("osprey_connectors.config.get_config_value", fake_get_config_value)


@pytest.fixture
def writes_enabled(monkeypatch):
    """Open the base-class writes gate so ``write_channel`` reaches its real body.

    ``ControlSystemConnector.__init_subclass__`` wraps ``write_channel`` with a
    ``_writes_enabled`` pre-check that is False in a config-less test
    environment; the tests using this are about what happens after that gate.
    """
    monkeypatch.setattr(EPICSConnector, "_writes_enabled", property(lambda self: True))


# ---------------------------------------------------------------------------
# A stand-in p4p
# ---------------------------------------------------------------------------


class FakeDisconnected(RuntimeError):
    """Stands in for p4p's Disconnected, a RuntimeError and not a ConnectionError."""


class FakeRemoteError(RuntimeError):
    """Stands in for p4p's RemoteError."""


class FakeCancelled(RuntimeError):
    """Stands in for p4p's Cancelled."""


def fake_p4p_module(*, errors: bool = True, cancelled: bool = False) -> types.ModuleType:
    """A ``p4p`` package exposing what the connector looks up on it.

    ``p4p.client.thread.Context`` is a ``MagicMock``. With ``errors`` the thread
    module also carries ``Disconnected``, ``RemoteError`` and ``TimeoutError``;
    ``cancelled`` adds ``Cancelled``. Without ``errors`` it carries only
    ``Context``, so any other use of the module is visible.
    """
    thread_mod = types.ModuleType("p4p.client.thread")
    thread_mod.Context = MagicMock(name="Context")
    if errors:
        thread_mod.Disconnected = FakeDisconnected
        thread_mod.RemoteError = FakeRemoteError
        thread_mod.TimeoutError = TimeoutError
        if cancelled:
            thread_mod.Cancelled = FakeCancelled
    client_mod = types.ModuleType("p4p.client")
    client_mod.thread = thread_mod
    p4p_mod = types.ModuleType("p4p")
    p4p_mod.client = client_mod
    return p4p_mod


def install_fake_p4p(monkeypatch) -> tuple[types.ModuleType, MagicMock]:
    """Put a fake ``p4p`` package in ``sys.modules``; return (module, Context class)."""
    p4p_mod = fake_p4p_module(errors=False)
    monkeypatch.setitem(sys.modules, "p4p", p4p_mod)
    monkeypatch.setitem(sys.modules, "p4p.client", p4p_mod.client)
    monkeypatch.setitem(sys.modules, "p4p.client.thread", p4p_mod.client.thread)
    return p4p_mod, p4p_mod.client.thread.Context


class FakeValue:
    """Minimal p4p ``Value`` stand-in: a struct id plus (possibly nested) fields."""

    def __init__(self, type_id: str, fields: dict):
        self._type_id = type_id
        self._fields = {
            key: FakeValue("", value) if isinstance(value, dict) else value
            for key, value in fields.items()
        }

    def getID(self) -> str:  # p4p's spelling
        return self._type_id

    def get(self, name, default=None):
        return self._fields.get(name, default)

    def __contains__(self, name) -> bool:
        return name in self._fields


# ---------------------------------------------------------------------------
# Channel Access PVs and connectors wired without connect()
# ---------------------------------------------------------------------------


# Channel Access status codes the fake answers with (EPICS base caerr.h).
ECA_TIMEOUT = 80
ECA_DISCONN = 192
ECA_NOWTACCESS = 376

DBR_STRING, DBR_SHORT, DBR_ENUM, DBR_CHAR, DBR_LONG, DBR_DOUBLE = 0, 1, 3, 4, 5, 6


def ca_reading(
    value: Any = 1.0,
    *,
    status: int | None = 0,
    severity: int | None = 0,
    datatype: int | None = None,
    timestamp: float | None = 1_750_000_000.0,
    count: int | None = None,
    units: str | None = None,
    precision: int | None = None,
    enums: list[str] | tuple[str, ...] | None = None,
) -> Any:
    """An aioca-style reading: ``value`` as a subclass carrying CA attributes.

    aioca hands back ``str``/``int``/``float``/``ndarray`` subclasses with the
    reading's metadata as attributes; this builds the same shape. ``datatype``
    defaults from the value's type, and an attribute left ``None`` is not set,
    as aioca sets only the fields the requested format carries.
    """
    if isinstance(value, (list, tuple, numpy.ndarray)):
        reading: Any = numpy.asarray(value).view(type("ca_array", (numpy.ndarray,), {}))
        default_type = DBR_DOUBLE
        default_count = len(reading)
    else:
        if isinstance(value, str):
            base, default_type = str, DBR_STRING
        elif isinstance(value, int) and not isinstance(value, bool):
            base, default_type = int, DBR_LONG
        else:
            base, default_type = float, DBR_DOUBLE
        reading = type(f"ca_{base.__name__}", (base,), {})(value)
        default_count = 1
    attrs = {
        "status": status,
        "severity": severity,
        "datatype": default_type if datatype is None else datatype,
        "timestamp": timestamp,
        "element_count": default_count if count is None else count,
        "units": units,
        "precision": precision,
        "enums": list(enums) if enums is not None else None,
        "ok": True,
    }
    for name, attr in attrs.items():
        if attr is not None:
            setattr(reading, name, attr)
    return reading


class FakeChannel:
    """One channel the fake aioca answers for.

    ``value`` is what a TIME-form ``caget`` returns (a :func:`ca_reading`, or
    an ``aioca.CANothing`` for a failed get); ``ctrl`` is what a CTRL-form
    ``caget`` returns. ``put`` is the code a ``caput`` answers with: ``None``
    for success, else an ECA code. ``connected=False`` makes every call fail to
    connect.
    """

    def __init__(self, value=None, *, ctrl=None, put=None, connected=True):
        self.value = value if value is not None else ca_reading(1.0)
        self.ctrl = ctrl if ctrl is not None else ca_reading(0.0, units="mA", precision=3)
        self.put = put
        self.connected = connected


class FakeSubscription:
    """What ``camonitor`` returns: the callback to fire, and whether it was closed."""

    def __init__(self, name, callback):
        self.name = name
        self.callback = callback
        self.closed = False

    def close(self):
        self.closed = True


class FakeAioca(types.ModuleType):
    """A stand-in ``aioca`` module over a dict of :class:`FakeChannel`.

    A name with no channel never connects, unless ``default`` answers for it.
    Every call is recorded in
    ``calls`` as ``(function, name, kwargs)`` so a test can say what was asked
    of the control system, and ``puts`` lists ``(name, value, wait)``.
    """

    CANothing = aioca.CANothing
    FORMAT_RAW = aioca.FORMAT_RAW
    FORMAT_TIME = aioca.FORMAT_TIME
    FORMAT_CTRL = aioca.FORMAT_CTRL
    DBR_CHAR_STR = aioca.DBR_CHAR_STR

    def __init__(
        self, channels: dict[str, FakeChannel] | None = None, *, default: FakeChannel | None = None
    ):
        super().__init__("aioca")
        self.channels: dict[str, FakeChannel] = dict(channels or {})
        # Answers for every name not in ``channels``; None leaves them unconnected.
        self.default = default
        self.calls: list[tuple[str, str, dict]] = []
        self.puts: list[tuple[str, Any, bool]] = []
        self.monitors: list[FakeSubscription] = []

    def _channel(self, name):
        channel = self.channels.get(name, self.default)
        return channel if channel is not None and channel.connected else None

    # Only the keywords the connector passes are named; the rest of aioca's
    # signature is accepted and ignored.

    async def connect(self, name, *, timeout=None, **_aioca_kwargs):
        self.calls.append(("connect", name, {"timeout": timeout}))
        if self._channel(name) is None:
            return aioca.CANothing(name, ECA_TIMEOUT)
        return aioca.CANothing(name)

    async def caget(
        self, name, *, datatype=None, format=aioca.FORMAT_RAW, timeout=None, **_aioca_kwargs
    ):
        self.calls.append(
            ("caget", name, {"format": format, "timeout": timeout, "datatype": datatype})
        )
        channel = self._channel(name)
        if channel is None:
            return aioca.CANothing(name, ECA_TIMEOUT)
        answer = channel.ctrl if format == aioca.FORMAT_CTRL else channel.value
        # A CANothing is returned, as aioca does under throw=False; any other
        # exception stands for one the client raised regardless.
        if isinstance(answer, BaseException) and not isinstance(answer, aioca.CANothing):
            raise answer
        return answer

    async def caput(self, name, value, *, datatype=None, wait=False, timeout=None, **_aioca_kwargs):
        self.calls.append(("caput", name, {"wait": wait, "timeout": timeout, "datatype": datatype}))
        channel = self._channel(name)
        if channel is None:
            return aioca.CANothing(name, ECA_TIMEOUT)
        self.puts.append((name, value, wait))
        if channel.put is None:
            return aioca.CANothing(name)
        return aioca.CANothing(name, channel.put)

    async def cainfo(self, name, *, timeout=None, **_aioca_kwargs):
        self.calls.append(("cainfo", name, {"timeout": timeout}))
        channel = self._channel(name)
        if channel is None:
            return aioca.CANothing(name, ECA_TIMEOUT)
        return types.SimpleNamespace(
            name=name,
            datatype=getattr(channel.value, "datatype", DBR_DOUBLE),
            count=getattr(channel.value, "element_count", 1),
        )

    def camonitor(
        self, name, callback, *, format=aioca.FORMAT_RAW, notify_disconnect=False, **_aioca_kwargs
    ):
        self.calls.append(
            ("camonitor", name, {"format": format, "notify_disconnect": notify_disconnect})
        )
        subscription = FakeSubscription(name, callback)
        self.monitors.append(subscription)
        return subscription


class InlineCaLoop:
    """A CA loop that runs every coroutine on the caller's own loop.

    The production loop is a thread of its own; the fake aioca does no I/O, so
    there is nothing to marshal and the tests stay single-threaded. ``run_sync``
    is called from a worker thread (the limits validator's), where a fresh
    ``asyncio.run`` is safe.
    """

    def __init__(self):
        self.pointed_at: list[dict] = []
        self.reopeners: list = []

    async def run(self, coro):
        return await coro

    def run_sync(self, coro, timeout=None):  # noqa: ARG002 - the real loop's signature
        return asyncio.run(coro)

    def call_soon(self, callback, *args):
        callback(*args)

    async def point_at(self, env):
        self.pointed_at.append(dict(env))
        for key, setting in env.items():
            if setting is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = setting

    def add_reopener(self, reopen):
        self.reopeners.append(reopen)

    def discard_reopener(self, reopen):
        if reopen in self.reopeners:
            self.reopeners.remove(reopen)

    def repoint(self):
        """What a repoint does to monitors: every one is re-opened on the new context."""
        for reopen in list(self.reopeners):
            reopen()


def install_fake_aioca(monkeypatch, channels: dict[str, FakeChannel] | None = None) -> FakeAioca:
    """Put a :class:`FakeAioca` in ``sys.modules`` and an inline CA loop behind ``connect()``.

    ``connect()`` imports aioca and binds the process's CA loop; both are
    replaced, so a test that connects never starts the real loop thread or
    repoints the process's Channel Access.
    """
    fake = FakeAioca(channels)
    monkeypatch.setitem(sys.modules, "aioca", fake)
    monkeypatch.setattr(
        "osprey_connectors.control_system.epics_connector.get_ca_loop", InlineCaLoop
    )
    return fake


@pytest.fixture
def fake_aioca(monkeypatch):
    """:func:`install_fake_aioca` as a fixture; yields the stand-in ``aioca``."""
    return install_fake_aioca(monkeypatch)


def ca_connector(
    *, aioca=None, channels=None, limits_validator=None, timeout=5.0
) -> EPICSConnector:
    """A Channel Access connector that skips ``connect()`` by injecting its runtime state.

    Pass a :class:`FakeAioca` as ``aioca``, or ``channels`` to build one.
    """
    connector = EPICSConnector()
    connector._aioca = aioca if aioca is not None else FakeAioca(channels)
    connector._ca = InlineCaLoop()
    connector._limits_validator = limits_validator
    connector._timeout = timeout
    connector._connected = True
    connector._epics_configured = True
    return connector


def pva_connector(
    context=None, globs=(PVA_GLOB,), aioca=None, p4p=None, timeout=3.0
) -> EPICSConnector:
    """A connector wired for PVA (and a fake CA client) without ``connect()``."""
    connector = EPICSConnector()
    connector._pva_channel_globs = list(globs)
    connector._p4p = p4p if p4p is not None else fake_p4p_module()
    connector._pva_context = context
    connector._timeout = timeout
    connector._aioca = aioca if aioca is not None else FakeAioca()
    connector._ca = InlineCaLoop()
    connector._connected = True
    return connector
