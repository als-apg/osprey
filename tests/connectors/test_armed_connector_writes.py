"""Every connector's own write passes the armed raw-put block; a raw put does not.

With the armed block installed on the client puts the product's own table
names, each connector's ``write_channel`` still reaches its client — the base
connector opens the write door around it and the put hops into
``asyncio.to_thread`` carrying the door — and the client sees the door open.
The same client put made directly from test code, outside any connector, is
refused with ``ChannelWriteBlockedError(RAW_CLIENT_WRITE)``: the negative
control that proves the block is armed on exactly the put the connector used.
The mock connector has no client library; for it the contract is only that
installing the block leaves its writes working.

Each client is a fake module built from source and registered in
``sys.modules`` under the name the connector imports (``epics``, ``tango``,
``doocs4py``), so the block's rows resolve to the fake and the connector talks
to it. Only the rows whose owner is one of those fakes are installed, so a
real client library already imported by another test is never patched; the
fixture from :mod:`tests.runtime._patch_restore` restores the fakes and drops
the block's import-hook finder.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
from types import ModuleType

import pytest

from osprey.runtime import raw_put_block
from osprey.services.python_executor.write_surface import _ARMED_BLOCKED
from osprey_connectors.control_system import doocs_connector, tango_connector
from osprey_connectors.control_system.base import WriteOutcome
from osprey_connectors.control_system.doocs_connector import DOOCSConnector
from osprey_connectors.control_system.epics_connector import EPICSConnector
from osprey_connectors.control_system.mock_connector import MockConnector
from osprey_connectors.control_system.tango_connector import TangoConnector
from osprey_connectors.control_system.va_connector import VirtualAcceleratorConnector
from osprey_connectors.control_system.write_door import door_is_open
from osprey_connectors.errors import RAW_CLIENT_WRITE_MARKER, ChannelWriteBlockedError
from tests.runtime._patch_restore import restore_patches  # noqa: F401

#: Row owners that resolve to a fake below and to nothing a real library owns.
_FAKE_OWNERS = frozenset({"epics", "epics.PV", "tango.DeviceProxy", "doocs4py"})

_EPICS_SOURCE = """
    from osprey_connectors.control_system.write_door import door_is_open

    calls = []

    def caput(pvname, value, wait=False, timeout=None):
        calls.append((pvname, value, door_is_open()))
        return True

    class PV:
        def __init__(self, pvname, *args, **kwargs):
            self.pvname = pvname

        def put(self, value, **kwargs):
            calls.append((self.pvname, value, door_is_open()))
            return True
"""

_TANGO_SOURCE = """
    from osprey_connectors.control_system.write_door import door_is_open

    __version__ = "10.0.0"
    calls = []

    class DeviceProxy:
        def __init__(self, name):
            self.name = name

        def dev_name(self):
            return self.name

        def set_timeout_millis(self, millis):
            pass

        def write_attribute(self, attribute, value):
            calls.append((f"{self.name}/{attribute}", value, door_is_open()))

    class Database:
        def __init__(self, *args):
            pass

        def get_info(self):
            return "TANGO Database sys/database/2"
"""

_DOOCS_SOURCE = """
    from osprey_connectors.control_system.write_door import door_is_open

    __version__ = "2.0.0"
    calls = []

    def names(pattern):
        return [("FACILITY", "XFEL")]

    def set(address, value):
        calls.append((address, value, door_is_open()))
"""


def _armed_rows() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The product's armed blocked rows, restricted to the fake owners."""
    rows: dict[str, list[str]] = {}
    for dotted, attr in _ARMED_BLOCKED:
        if dotted in _FAKE_OWNERS:
            rows.setdefault(dotted, []).append(attr)
    return tuple((dotted, tuple(attrs)) for dotted, attrs in rows.items())


@pytest.fixture
def fakes(monkeypatch, restore_patches):  # noqa: F811
    """Register the fake clients, then install the armed block over them."""
    modules: dict[str, ModuleType] = {}
    for name, source in (
        ("epics", _EPICS_SOURCE),
        ("tango", _TANGO_SOURCE),
        ("doocs4py", _DOOCS_SOURCE),
    ):
        module = ModuleType(name)
        module.__file__ = f"<fake {name}>"
        exec(compile(textwrap.dedent(source), module.__file__, "exec"), vars(module))
        monkeypatch.setitem(sys.modules, name, module)
        restore_patches(module)
        for value in list(vars(module).values()):
            if isinstance(value, type):
                restore_patches(value)
        modules[name] = module

    raw_put_block.install(
        "armed",
        blocked_targets=_armed_rows(),
        rpc_targets=(),
        refuse_rpc=False,
        marker=RAW_CLIENT_WRITE_MARKER,
        rpc_refusals={},
    )
    return modules


def _writes_on(monkeypatch, cls) -> None:
    monkeypatch.setattr(cls, "_writes_enabled", property(lambda self: True))


def _assert_refused(call, address: str) -> None:
    with pytest.raises(ChannelWriteBlockedError) as caught:
        call()
    assert caught.value.reason == "RAW_CLIENT_WRITE"
    assert caught.value.channel_address == address
    assert RAW_CLIENT_WRITE_MARKER in str(caught.value)


def test_the_product_table_names_every_connector_put():
    rows = dict(_armed_rows())
    assert "caput" in rows["epics"]
    assert "put" in rows["epics.PV"]
    assert "write_attribute" in rows["tango.DeviceProxy"]
    assert "set" in rows["doocs4py"]


def test_the_block_is_armed_on_the_fakes(fakes):
    assert getattr(fakes["epics"].caput, "_osprey_armed_block", False)
    assert getattr(fakes["epics"].PV.put, "_osprey_armed_block", False)
    assert getattr(fakes["tango"].DeviceProxy.write_attribute, "_osprey_armed_block", False)
    assert getattr(fakes["doocs4py"].set, "_osprey_armed_block", False)


@pytest.mark.parametrize("cls", [EPICSConnector, VirtualAcceleratorConnector])
def test_epics_family_connector_write_passes_and_raw_caput_is_refused(fakes, monkeypatch, cls):
    epics = fakes["epics"]
    _writes_on(monkeypatch, cls)
    connector = cls()
    connector._epics = epics
    connector._limits_validator = None
    connector._timeout = 5.0
    connector._connected = True
    connector._epics_configured = True

    result = asyncio.run(connector.write_channel("SR:CH", 1.5, confirm=False))

    assert result.outcome is WriteOutcome.UNREQUESTED, result.error_message
    assert epics.calls == [("SR:CH", 1.5, True)]
    assert door_is_open() is False
    _assert_refused(lambda: epics.caput("SR:CH", 2.0), "SR:CH")
    _assert_refused(lambda: epics.PV("SR:CH").put(2.0), "SR:CH")
    assert len(epics.calls) == 1


def test_tango_connector_write_passes_and_raw_write_attribute_is_refused(fakes, monkeypatch):
    tango = fakes["tango"]
    _writes_on(monkeypatch, TangoConnector)
    monkeypatch.setattr(
        tango_connector.LimitsValidator, "from_config", staticmethod(lambda **_kw: None)
    )

    async def main():
        connector = TangoConnector()
        await connector.connect({})
        result = await connector.write_channel("sr/ps/01/Current", 3.0, confirm=False)
        await connector.disconnect()
        return result

    result = asyncio.run(main())

    assert result.outcome is WriteOutcome.UNREQUESTED, result.error_message
    assert tango.calls == [("sr/ps/01/Current", 3.0, True)]
    assert door_is_open() is False
    _assert_refused(
        lambda: tango.DeviceProxy("sr/ps/01").write_attribute("Current", 4.0),
        "sr/ps/01/Current",
    )
    assert len(tango.calls) == 1


def test_doocs_connector_write_passes_and_raw_set_is_refused(fakes, monkeypatch):
    doocs4py = fakes["doocs4py"]
    _writes_on(monkeypatch, DOOCSConnector)
    monkeypatch.setattr(
        doocs_connector.LimitsValidator, "from_config", staticmethod(lambda **_kw: None)
    )

    async def main():
        connector = DOOCSConnector()
        await connector.connect({})
        result = await connector.write_channel("FAC/DEV/LOC/PROP", 7, confirm=False)
        await connector.disconnect()
        return result

    result = asyncio.run(main())

    assert result.outcome is WriteOutcome.UNREQUESTED, result.error_message
    assert doocs4py.calls == [("FAC/DEV/LOC/PROP", 7, True)]
    assert door_is_open() is False
    _assert_refused(lambda: doocs4py.set("FAC/DEV/LOC/PROP", 8), "FAC/DEV/LOC/PROP")
    assert len(doocs4py.calls) == 1


def test_mock_connector_writes_still_work_with_the_block_armed(fakes, monkeypatch):
    _writes_on(monkeypatch, MockConnector)

    async def main():
        connector = MockConnector()
        await connector.connect({"response_delay_ms": 0, "noise_level": 0.0})
        result = await connector.write_channel("TEST:CHANNEL:SP", 4.25, confirm=False)
        read = await connector.read_channel("TEST:CHANNEL:SP")
        await connector.disconnect()
        return result, read

    result, read = asyncio.run(main())

    assert result.outcome is WriteOutcome.UNREQUESTED, result.error_message
    assert read.value == pytest.approx(4.25)
    assert door_is_open() is False
    assert all(not calls for calls in (m.calls for m in fakes.values()))
