"""Shared fakes and fixtures for the EPICS / PVA / VA connector tests.

One copy of the stand-ins that the EPICS connector test files had each written
for themselves: a fake ``epics`` (pyepics) package, a fake ``p4p`` package, a
p4p ``Value`` stand-in, a connected Channel Access PV, and connectors wired
without going through ``connect()``.

Fixtures defined here reach a test module only when that module imports them,
for example ``from tests.connectors._epics_fakes import fake_pyepics  # noqa: F401``.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

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
# A stand-in pyepics
# ---------------------------------------------------------------------------


class UnreachablePV:
    """A pyepics ``PV`` that never connects: every Channel Access search goes unanswered."""

    connected = False

    def __init__(self, pvname, *args, **kwargs):  # noqa: ARG002 - pyepics' signature
        self.pvname = pvname

    def wait_for_connection(self, timeout=None):  # noqa: ARG002 - pyepics' signature
        return False

    def disconnect(self):
        pass


def install_fake_pyepics(monkeypatch) -> types.ModuleType:
    """Put a stand-in ``epics`` package in ``sys.modules`` and return its ``epics.ca``.

    ``EPICSConnector.connect()`` sets ``epics.ca.AUTO_CLEANUP``, unregisters
    ``epics.ca.finalize_libca`` from ``atexit`` and, with a gateway configured,
    calls ``epics.ca.clear_cache()``, which loads libca into the process. Against
    the real pyepics none of that is undone, so every later test in the worker
    would inherit a live CA context and a pyepics with its shutdown hook
    removed. The stand-in takes all of it instead; ``monkeypatch`` puts the real
    modules back.

    ``ca.seen_at_first_libca_use`` records ``AUTO_CLEANUP`` each time
    ``clear_cache`` runs, which is where pyepics would first load libca.
    ``epics.PV`` builds an :class:`UnreachablePV`.
    """
    ca = types.ModuleType("epics.ca")
    ca.AUTO_CLEANUP = True
    ca.finalize_libca = lambda: None
    ca.seen_at_first_libca_use = []

    def clear_cache():
        ca.seen_at_first_libca_use.append(ca.AUTO_CLEANUP)

    ca.clear_cache = clear_cache
    package = types.ModuleType("epics")
    package.ca = ca
    package.PV = UnreachablePV
    monkeypatch.setitem(sys.modules, "epics", package)
    monkeypatch.setitem(sys.modules, "epics.ca", ca)
    return ca


@pytest.fixture
def fake_pyepics(monkeypatch):
    """:func:`install_fake_pyepics` as a fixture; yields the stand-in ``epics.ca``."""
    return install_fake_pyepics(monkeypatch)


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

    def getID(self) -> str:  # noqa: N802 - p4p's spelling
        return self._type_id

    def get(self, name, default=None):
        return self._fields.get(name, default)

    def __contains__(self, name) -> bool:
        return name in self._fields


# ---------------------------------------------------------------------------
# Channel Access PVs and connectors wired without connect()
# ---------------------------------------------------------------------------


def connected_pv(
    value=1.0,
    *,
    status=0,
    severity=0,
    pv_type=None,
    labels=None,
    timestamp=1_750_000_000.0,
):
    """A fake pyepics PV that is connected and reports a value and an alarm state.

    ``pv_type`` sets ``pv.type`` and ``labels`` sets ``pv.enum_strs``; each is
    left as a ``MagicMock`` attribute when not given.
    """
    pv = MagicMock()
    pv.wait_for_connection.return_value = True
    pv.connected = True
    pv.get.return_value = value
    pv.timestamp = timestamp
    pv.units = "mA"
    pv.precision = 3
    pv.status = status
    pv.severity = severity
    if pv_type is not None:
        pv.type = pv_type
    if labels is not None:
        pv.enum_strs = labels
    return pv


def ca_connector(*, epics=None, limits_validator=None, timeout=5.0) -> EPICSConnector:
    """A Channel Access connector that skips ``connect()`` by injecting its runtime state."""
    connector = EPICSConnector()
    connector._epics = epics if epics is not None else MagicMock()
    connector._limits_validator = limits_validator
    connector._timeout = timeout
    connector._connected = True
    connector._epics_configured = True
    return connector


def pva_connector(
    context=None, globs=(PVA_GLOB,), epics=None, p4p=None, timeout=3.0
) -> EPICSConnector:
    """A connector wired for PVA (and a ``MagicMock`` CA client) without ``connect()``."""
    connector = EPICSConnector()
    connector._pva_channel_globs = list(globs)
    connector._p4p = p4p if p4p is not None else fake_p4p_module()
    connector._pva_context = context
    connector._timeout = timeout
    connector._epics = epics if epics is not None else MagicMock()
    connector._connected = True
    return connector
