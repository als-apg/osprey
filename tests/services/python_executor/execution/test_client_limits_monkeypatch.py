"""Limits parity for the non-EPICS clients in the generated wrapper monkeypatch.

A readwrite run is limits-checked, but only for the clients the wrapper knows
how to intercept. Tango, DOOCS and caproto were in the *readonly* write surface
— so a readonly run refused them — while a readwrite run let them past the
limits database that ``epics.caput`` and every p4p flavour are checked against.
That asymmetry is what these tests pin shut.

Like the p4p monkeypatch tests, they execute the generated *source text*
against fake client modules injected into ``sys.modules``: the block is emitted
to run inside the executor subprocess, so there is no object to patch here.
"""

import contextlib
import io
import sys
from types import ModuleType

import pytest

from osprey.connectors.control_system.limits_validator import (
    ChannelLimitsConfig,
    LimitsValidator,
)
from osprey.errors import ChannelLimitsViolationError
from osprey.services.python_executor.execution.wrapper import ExecutionWrapper

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _make_fake_epics():
    """Keep the block's own epics branch off any real pyepics install."""
    mod = ModuleType("epics")

    def caput(pvname, value, wait=False, timeout=60, **kwargs):
        return 1

    class PV:
        def __init__(self, pvname):
            self.pvname = pvname

        def put(self, value, wait=False, timeout=60, **kwargs):
            return 1

    mod.caput = caput
    mod.PV = PV
    return mod


def _install_fake_tango(monkeypatch, writes):
    mod = ModuleType("tango")

    class DeviceProxy:
        def __init__(self, name):
            self._name = name

        def dev_name(self):
            return self._name

        def write_attribute(self, attr, value):
            writes.append((attr, value))
            return "written"

        def write_attributes(self, name_val):
            writes.extend(list(name_val))
            return "written"

        def read_attribute(self, attr):
            return 1.0

    mod.DeviceProxy = DeviceProxy
    monkeypatch.setitem(sys.modules, "tango", mod)
    return mod


def _install_fake_doocs(monkeypatch, writes):
    mod = ModuleType("doocs4py")

    def _set(address, value):
        writes.append((address, value))
        return "written"

    mod.set = _set
    monkeypatch.setitem(sys.modules, "doocs4py", mod)
    return mod


def _install_fake_caproto(monkeypatch, writes):
    """Inject a fake ``caproto`` with both write entry points."""
    caproto_mod = ModuleType("caproto")
    sync_mod = ModuleType("caproto.sync")
    sync_client = ModuleType("caproto.sync.client")
    threading_mod = ModuleType("caproto.threading")
    threading_client = ModuleType("caproto.threading.client")

    def write(pv_name, data, **kwargs):
        writes.append((pv_name, data))
        return "written"

    class PV:
        def __init__(self, name):
            self.name = name

        def write(self, data, **kwargs):
            writes.append((self.name, data))
            return "written"

        def read(self):
            return 1.0

    sync_client.write = write
    threading_client.PV = PV
    sync_mod.client = sync_client
    threading_mod.client = threading_client
    caproto_mod.sync = sync_mod
    caproto_mod.threading = threading_mod

    for name, mod in (
        ("caproto", caproto_mod),
        ("caproto.sync", sync_mod),
        ("caproto.sync.client", sync_client),
        ("caproto.threading", threading_mod),
        ("caproto.threading.client", threading_client),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return sync_client, threading_client


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _make_validator():
    limits = {
        "sys/tg_test/1/current": ChannelLimitsConfig(
            channel_address="sys/tg_test/1/current", min_value=0.0, max_value=10.0
        ),
        "sys/tg_test/1/voltage": ChannelLimitsConfig(
            channel_address="sys/tg_test/1/voltage", min_value=0.0, max_value=10.0
        ),
        "FACILITY/MAGNET/H1/CURRENT.SP": ChannelLimitsConfig(
            channel_address="FACILITY/MAGNET/H1/CURRENT.SP",
            min_value=0.0,
            max_value=10.0,
        ),
        "TEST:MAG:SP": ChannelLimitsConfig(
            channel_address="TEST:MAG:SP", min_value=0.0, max_value=10.0
        ),
    }
    return LimitsValidator(limits, {"allow_unlisted_channels": False})


def _run_monkeypatch(monkeypatch):
    """Execute the generated monkeypatch block; return its stdout."""
    monkeypatch.setitem(sys.modules, "epics", _make_fake_epics())

    import osprey.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_limits_validator", None, raising=False)

    source = ExecutionWrapper(limits_validator=_make_validator())._get_limits_checking_monkeypatch()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(source, "<generated-wrapper>", "exec"), {})
    out = buf.getvalue()
    assert "Limits checking setup failed" not in out, out
    return out


# ---------------------------------------------------------------------------
# Tango
# ---------------------------------------------------------------------------


def test_tango_write_attribute_within_limits_reaches_the_device(monkeypatch):
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    assert proxy.write_attribute("current", 5.0) == "written"
    assert writes == [("current", 5.0)]


def test_tango_write_attribute_out_of_bounds_raises_before_the_device(monkeypatch):
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    with pytest.raises(ChannelLimitsViolationError):
        proxy.write_attribute("current", 99.0)
    assert writes == []


def test_tango_channel_is_the_full_device_attribute_address(monkeypatch):
    """A limits database is keyed by ``device/attribute``, not by the attribute.

    Validating the bare attribute name would look up a channel the database
    has never heard of — which, on a deployment that allows unlisted channels,
    is a write that passes without being checked at all.
    """
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    # Another device with the same attribute name is not in the database.
    other = mod.DeviceProxy("sys/tg_test/2")
    with pytest.raises(ChannelLimitsViolationError):
        other.write_attribute("current", 5.0)
    assert writes == []


def test_tango_write_attributes_checks_every_pair(monkeypatch):
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    assert proxy.write_attributes([("current", 5.0), ("voltage", 4.0)]) == "written"
    assert writes == [("current", 5.0), ("voltage", 4.0)]

    writes.clear()
    with pytest.raises(ChannelLimitsViolationError):
        proxy.write_attributes([("current", 5.0), ("voltage", 99.0)])
    assert writes == []


def test_tango_write_attributes_forwards_a_consumed_iterator(monkeypatch):
    """A generator of pairs must still reach the device.

    The guard has to materialise the argument to check every pair, which
    exhausts an iterator. Forwarding the original then writes nothing at all —
    silently, with the caller told the write succeeded.
    """
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    pairs = iter([("current", 5.0), ("voltage", 4.0)])
    assert proxy.write_attributes(pairs) == "written"
    assert writes == [("current", 5.0), ("voltage", 4.0)]


def test_tango_write_attributes_fails_closed_on_an_unpairable_shape(monkeypatch):
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    with pytest.raises(ValueError, match="pairs"):
        proxy.write_attributes([object()])
    assert writes == []


# ---------------------------------------------------------------------------
# DOOCS
# ---------------------------------------------------------------------------


def test_doocs_set_within_limits_reaches_the_machine(monkeypatch):
    writes: list = []
    mod = _install_fake_doocs(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    address = "FACILITY/MAGNET/H1/CURRENT.SP"
    assert mod.set(address, 5.0) == "written"
    assert writes == [(address, 5.0)]


def test_doocs_set_out_of_bounds_raises_before_the_machine(monkeypatch):
    writes: list = []
    mod = _install_fake_doocs(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError):
        mod.set("FACILITY/MAGNET/H1/CURRENT.SP", 99.0)
    assert writes == []


# ---------------------------------------------------------------------------
# caproto
# ---------------------------------------------------------------------------


def test_caproto_sync_write_is_limits_checked(monkeypatch):
    writes: list = []
    sync_client, _ = _install_fake_caproto(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    assert sync_client.write("TEST:MAG:SP", 5.0) == "written"
    assert writes == [("TEST:MAG:SP", 5.0)]

    writes.clear()
    with pytest.raises(ChannelLimitsViolationError):
        sync_client.write("TEST:MAG:SP", 99.0)
    assert writes == []


def test_caproto_threading_pv_write_is_limits_checked(monkeypatch):
    writes: list = []
    _, threading_client = _install_fake_caproto(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    pv = threading_client.PV("TEST:MAG:SP")
    assert pv.write(5.0) == "written"
    assert writes == [("TEST:MAG:SP", 5.0)]

    writes.clear()
    with pytest.raises(ChannelLimitsViolationError):
        pv.write(99.0)
    assert writes == []
    assert pv.read() == 1.0, "reads must survive the guard untouched"


# ---------------------------------------------------------------------------
# Absence
# ---------------------------------------------------------------------------


def test_absent_clients_do_not_stop_the_block(monkeypatch):
    """One client missing must not skip the guards after it.

    Each client gets its own try/except for the same reason the p4p flavours
    do: the outer swallow-all handler would drop every guard that had not been
    installed yet.
    """
    for name in (
        "tango",
        "doocs4py",
        "caproto",
        "caproto.sync",
        "caproto.sync.client",
        "caproto.threading",
        "caproto.threading.client",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    writes: list = []
    _install_fake_doocs(monkeypatch, writes)

    out = _run_monkeypatch(monkeypatch)
    assert "tango not available" in out
    assert "✅ Monkeypatched doocs4py.set()" in out
