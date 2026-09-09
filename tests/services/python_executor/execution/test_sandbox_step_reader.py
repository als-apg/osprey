"""The sandbox's write guards read the current value with the script's own client.

``max_step`` is the one limit that needs a fresh read, and the shared validator
owns no control-system client. In the generated wrapper each guard supplies
one: the Channel Access guard reads back through ``epics.ca.get`` on the
channel id being written, the p4p guard the very Context the put is going
through, the Tango guard the DeviceProxy doing the write, the DOOCS guard
``doocs4py.get``, and the caproto guards that client's own read — so the step
is measured over the client and the addressing the write itself uses.

One client answers no reader: p4p's asyncio flavour has only a coroutine
``get``, and the validator is synchronous. There the step cannot be measured
and a ``max_step`` channel fails closed, which is what a guard that cannot
measure a step owes the write.

Like the other monkeypatch tests, this executes the generated *source text*
against fake client modules injected into ``sys.modules``.
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

CHANNEL = "TEST:MAG:SP"
CURRENT = 5.0


def _step_validator(channel=CHANNEL):
    limits = {
        channel: ChannelLimitsConfig(
            channel_address=channel, min_value=0.0, max_value=100.0, max_step=2.0
        )
    }
    return LimitsValidator(limits, {"allow_unlisted_channels": False})


def _raising_read(_channel):
    """A Channel Access read that fails, as a disconnected channel's does."""
    raise RuntimeError("channel is not connected")


def _absent_read(_channel):
    """A Channel Access read that answers nothing, as a timed-out one does."""
    return None


def _install_fake_epics(monkeypatch, reads, writes, read=None):
    """A pyepics whose three write spellings all funnel through ``ca.put``.

    Real pyepics reaches the network from ``caput`` through ``PV.put`` and
    from ``PV.put`` through ``ca.put``, which is why the guard sits on the
    last of the three. Keeping that chain in the fake is what makes a test
    written against ``caput`` an honest exercise of the wrapper.

    ``read`` replaces what ``ca.get`` answers, and may raise: a client that
    cannot read is the case a step check has to fail closed on.
    """
    mod = ModuleType("epics")
    ca = ModuleType("epics.ca")

    class _Chid:
        """What ``epics.ca`` addresses a channel by: an opaque id, not a name."""

        def __init__(self, pvname):
            self.pvname = pvname

    def name(chid):
        return chid.pvname

    def create_channel(pvname, **kwargs):
        return _Chid(pvname)

    def get(chid, timeout=None, **kwargs):
        reads.append(name(chid))
        if read is not None:
            return read(name(chid))
        return CURRENT

    def put(chid, value, wait=False, timeout=60, **kwargs):
        writes.append((name(chid), value))
        return 1

    ca.name = name
    ca.create_channel = create_channel
    ca.get = get
    ca.put = put

    class PV:
        def __init__(self, pvname):
            self.pvname = pvname
            self.chid = ca.create_channel(pvname)

        def put(self, value, wait=False, timeout=60, **kwargs):
            return ca.put(self.chid, value, wait=wait, timeout=timeout, **kwargs)

    def caget(pvname, timeout=None, **kwargs):
        return ca.get(ca.create_channel(pvname), timeout=timeout)

    def caput(pvname, value, wait=False, timeout=60, **kwargs):
        return PV(pvname).put(value, wait=wait, timeout=timeout, **kwargs)

    mod.ca = ca
    mod.caget = caget
    mod.caput = caput
    mod.PV = PV
    monkeypatch.setitem(sys.modules, "epics", mod)
    monkeypatch.setitem(sys.modules, "epics.ca", ca)
    return mod


class _RecordingContext:
    """A p4p Context that records the reads and the puts it was asked for."""

    def __init__(self):
        self.reads: list = []
        self.puts: list = []

    def get(self, name, request=None, timeout=5.0):
        self.reads.append(name)
        return CURRENT

    def put(self, name, values, request=None, timeout=5.0, **kwargs):
        self.puts.append((name, values))
        return "put-done"


class _AsyncioContext:
    """A p4p asyncio Context: an ``async def get``, like the real one."""

    def __init__(self):
        self.reads: list = []
        self.puts: list = []

    async def get(self, name, request=None, timeout=5.0):
        self.reads.append(name)
        return CURRENT

    def put(self, name, values, request=None, timeout=5.0, **kwargs):
        self.puts.append((name, values))
        return "put-done"


def _install_fake_p4p(monkeypatch, asyncio_flavor=False):
    p4p_mod = ModuleType("p4p")
    client_mod = ModuleType("p4p.client")
    p4p_mod.client = client_mod
    monkeypatch.setitem(sys.modules, "p4p", p4p_mod)
    monkeypatch.setitem(sys.modules, "p4p.client", client_mod)

    thread_mod = ModuleType("p4p.client.thread")
    ctx_cls = type("ThreadContext", (_RecordingContext,), {})
    thread_mod.Context = ctx_cls
    client_mod.thread = thread_mod
    monkeypatch.setitem(sys.modules, "p4p.client.thread", thread_mod)
    monkeypatch.setitem(sys.modules, "p4p.client.cothread", None)

    if not asyncio_flavor:
        monkeypatch.setitem(sys.modules, "p4p.client.asyncio", None)
        return ctx_cls

    asyncio_mod = ModuleType("p4p.client.asyncio")
    asyncio_cls = type("AsyncioContext", (_AsyncioContext,), {})
    asyncio_mod.Context = asyncio_cls
    client_mod.asyncio = asyncio_mod
    monkeypatch.setitem(sys.modules, "p4p.client.asyncio", asyncio_mod)
    return asyncio_cls


def _run_monkeypatch(monkeypatch, channel=CHANNEL):
    import osprey.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_limits_validator", None, raising=False)

    wrapper = ExecutionWrapper(limits_validator=_step_validator(channel))
    source = wrapper._get_limits_checking_monkeypatch()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(source, "<generated-wrapper>", "exec"), {})
    out = buf.getvalue()
    assert "Limits checking setup failed" not in out, out
    return out


# ---------------------------------------------------------------------------
# Channel Access
# ---------------------------------------------------------------------------


def test_caput_measures_the_step_with_the_scripts_own_ca_get(monkeypatch):
    reads: list = []
    writes: list = []
    epics = _install_fake_epics(monkeypatch, reads, writes)
    _run_monkeypatch(monkeypatch)

    assert epics.caput(CHANNEL, CURRENT + 1.0) == 1
    assert reads == [CHANNEL]
    assert writes == [(CHANNEL, CURRENT + 1.0)]


def test_caput_beyond_max_step_never_reaches_the_control_system(monkeypatch):
    reads: list = []
    writes: list = []
    epics = _install_fake_epics(monkeypatch, reads, writes)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError) as exc:
        epics.caput(CHANNEL, CURRENT + 50.0)

    assert exc.value.violation_type == "MAX_STEP_EXCEEDED"
    assert writes == []


def test_pv_put_measures_the_step_with_the_scripts_own_ca_get(monkeypatch):
    reads: list = []
    writes: list = []
    epics = _install_fake_epics(monkeypatch, reads, writes)
    _run_monkeypatch(monkeypatch)

    epics.PV(CHANNEL).put(CURRENT + 1.0)

    assert reads == [CHANNEL]
    assert writes == [(CHANNEL, CURRENT + 1.0)]


def test_ca_get_that_raises_fails_the_step_check_closed(monkeypatch):
    """A read that blows up is not a step of zero — it is no measurement at all.

    The guard swallows the error so the refusal names the missing measurement
    rather than the client's own exception, but what it must never do is let
    the write through: an unmeasured step is an unapproved one.
    """
    reads: list = []
    writes: list = []
    epics = _install_fake_epics(monkeypatch, reads, writes, read=_raising_read)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError) as exc:
        epics.caput(CHANNEL, CURRENT + 0.5)

    assert exc.value.violation_type == "STEP_CHECK_FAILED"
    assert reads == [CHANNEL]
    assert writes == []


def test_ca_get_that_answers_none_fails_the_step_check_closed(monkeypatch):
    """pyepics answers a timed-out read with ``None``, not an exception.

    The step is as unmeasured as it is when the read raises, and the write is
    inside ``max_step`` only if the current value is assumed — which is the
    assumption the step check exists to refuse.
    """
    reads: list = []
    writes: list = []
    epics = _install_fake_epics(monkeypatch, reads, writes, read=_absent_read)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError) as exc:
        epics.caput(CHANNEL, CURRENT + 0.5)

    assert exc.value.violation_type == "STEP_CHECK_FAILED"
    assert reads == [CHANNEL]
    assert writes == []


# ---------------------------------------------------------------------------
# PVAccess
# ---------------------------------------------------------------------------


def test_p4p_put_measures_the_step_with_the_context_doing_the_put(monkeypatch):
    reads: list = []
    writes: list = []
    _install_fake_epics(monkeypatch, reads, writes)
    ctx_cls = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    context = ctx_cls()
    assert context.put(CHANNEL, CURRENT + 1.0) == "put-done"

    assert context.reads == [CHANNEL]
    assert context.puts == [(CHANNEL, CURRENT + 1.0)]
    # The step was NOT measured over Channel Access.
    assert reads == []


def test_p4p_put_beyond_max_step_never_reaches_the_control_system(monkeypatch):
    reads: list = []
    writes: list = []
    _install_fake_epics(monkeypatch, reads, writes)
    ctx_cls = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    context = ctx_cls()
    with pytest.raises(ChannelLimitsViolationError) as exc:
        context.put(CHANNEL, CURRENT + 50.0)

    assert exc.value.violation_type == "MAX_STEP_EXCEEDED"
    assert context.puts == []


def test_p4p_asyncio_put_fails_closed_with_no_synchronous_read(monkeypatch):
    """The asyncio flavour has no synchronous read, so a max_step channel is refused.

    Calling its coroutine ``get`` would hand the validator an un-awaited
    coroutine and refuse the write with a TypeError about it; the guard answers
    "no reader" instead, and the refusal names the missing measurement.
    """
    reads: list = []
    writes: list = []
    _install_fake_epics(monkeypatch, reads, writes)
    ctx_cls = _install_fake_p4p(monkeypatch, asyncio_flavor=True)
    _run_monkeypatch(monkeypatch)

    context = ctx_cls()
    with pytest.raises(ChannelLimitsViolationError) as exc:
        context.put(CHANNEL, CURRENT + 1.0)

    assert exc.value.violation_type == "STEP_CHECK_FAILED"
    assert context.puts == []
    # No coroutine was created, so none was left un-awaited.
    assert context.reads == []


# ---------------------------------------------------------------------------
# Tango
# ---------------------------------------------------------------------------

TANGO_DEVICE = "sys/tg_test/1"
TANGO_ATTR = "current"
TANGO_CHANNEL = f"{TANGO_DEVICE}/{TANGO_ATTR}"


class _TangoAttribute:
    def __init__(self, value):
        self.value = value


def _install_fake_tango(monkeypatch, reads, writes):
    mod = ModuleType("tango")

    class DeviceProxy:
        def __init__(self, name):
            self._name = name

        def dev_name(self):
            return self._name

        def read_attribute(self, attr):
            reads.append(attr)
            return _TangoAttribute(CURRENT)

        def write_attribute(self, attr, value):
            writes.append((attr, value))
            return "written"

        def write_attributes(self, name_val):
            writes.extend(list(name_val))
            return "written"

    mod.DeviceProxy = DeviceProxy
    monkeypatch.setitem(sys.modules, "tango", mod)
    return mod


def test_tango_write_measures_the_step_with_the_writing_proxy(monkeypatch):
    reads: list = []
    writes: list = []
    mod = _install_fake_tango(monkeypatch, reads, writes)
    _run_monkeypatch(monkeypatch, channel=TANGO_CHANNEL)

    proxy = mod.DeviceProxy(TANGO_DEVICE)
    assert proxy.write_attribute(TANGO_ATTR, CURRENT + 1.0) == "written"

    assert reads == [TANGO_ATTR]
    assert writes == [(TANGO_ATTR, CURRENT + 1.0)]


def test_tango_write_beyond_max_step_never_reaches_the_device(monkeypatch):
    reads: list = []
    writes: list = []
    mod = _install_fake_tango(monkeypatch, reads, writes)
    _run_monkeypatch(monkeypatch, channel=TANGO_CHANNEL)

    proxy = mod.DeviceProxy(TANGO_DEVICE)
    with pytest.raises(ChannelLimitsViolationError) as exc:
        proxy.write_attribute(TANGO_ATTR, CURRENT + 50.0)

    assert exc.value.violation_type == "MAX_STEP_EXCEEDED"
    assert writes == []


def test_tango_batch_write_measures_the_step_too(monkeypatch):
    reads: list = []
    writes: list = []
    mod = _install_fake_tango(monkeypatch, reads, writes)
    _run_monkeypatch(monkeypatch, channel=TANGO_CHANNEL)

    proxy = mod.DeviceProxy(TANGO_DEVICE)
    with pytest.raises(ChannelLimitsViolationError) as exc:
        proxy.write_attributes([(TANGO_ATTR, CURRENT + 50.0)])

    assert exc.value.violation_type == "MAX_STEP_EXCEEDED"
    assert writes == []


# ---------------------------------------------------------------------------
# DOOCS
# ---------------------------------------------------------------------------

DOOCS_CHANNEL = "FACILITY/MAGNET/H1/CURRENT.SP"


class _EqData:
    def __init__(self, value):
        self._value = value

    def get_data(self):
        return self._value


def _install_fake_doocs(monkeypatch, reads, writes):
    mod = ModuleType("doocs4py")

    def _get(address):
        reads.append(address)
        return _EqData(CURRENT)

    def _set(address, value):
        writes.append((address, value))
        return "written"

    mod.get = _get
    mod.set = _set
    monkeypatch.setitem(sys.modules, "doocs4py", mod)
    return mod


def test_doocs_set_measures_the_step_with_doocs4py(monkeypatch):
    reads: list = []
    writes: list = []
    mod = _install_fake_doocs(monkeypatch, reads, writes)
    _run_monkeypatch(monkeypatch, channel=DOOCS_CHANNEL)

    assert mod.set(DOOCS_CHANNEL, CURRENT + 1.0) == "written"

    assert reads == [DOOCS_CHANNEL]
    assert writes == [(DOOCS_CHANNEL, CURRENT + 1.0)]


def test_doocs_set_beyond_max_step_never_reaches_the_property(monkeypatch):
    reads: list = []
    writes: list = []
    mod = _install_fake_doocs(monkeypatch, reads, writes)
    _run_monkeypatch(monkeypatch, channel=DOOCS_CHANNEL)

    with pytest.raises(ChannelLimitsViolationError) as exc:
        mod.set(DOOCS_CHANNEL, CURRENT + 50.0)

    assert exc.value.violation_type == "MAX_STEP_EXCEEDED"
    assert writes == []


# ---------------------------------------------------------------------------
# caproto
# ---------------------------------------------------------------------------


class _CaprotoResponse:
    """caproto answers a read with an array, even for a scalar channel."""

    def __init__(self, value):
        self.data = [value]


def _install_fake_caproto(monkeypatch, reads, writes):
    caproto_mod = ModuleType("caproto")
    sync_mod = ModuleType("caproto.sync")
    sync_client = ModuleType("caproto.sync.client")
    threading_mod = ModuleType("caproto.threading")
    threading_client = ModuleType("caproto.threading.client")

    def read(pv_name, **kwargs):
        reads.append(pv_name)
        return _CaprotoResponse(CURRENT)

    def write(pv_name, data, **kwargs):
        writes.append((pv_name, data))
        return "written"

    class PV:
        def __init__(self, name):
            self.name = name

        def read(self, **kwargs):
            reads.append(self.name)
            return _CaprotoResponse(CURRENT)

        def write(self, data, **kwargs):
            writes.append((self.name, data))
            return "written"

    sync_client.read = read
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


def test_caproto_sync_write_measures_the_step_with_caprotos_own_read(monkeypatch):
    reads: list = []
    writes: list = []
    _install_fake_epics(monkeypatch, [], [])
    sync_client, _ = _install_fake_caproto(monkeypatch, reads, writes)
    _run_monkeypatch(monkeypatch)

    assert sync_client.write(CHANNEL, CURRENT + 1.0) == "written"

    assert reads == [CHANNEL]
    assert writes == [(CHANNEL, CURRENT + 1.0)]


def test_caproto_sync_write_beyond_max_step_never_reaches_the_ioc(monkeypatch):
    reads: list = []
    writes: list = []
    _install_fake_epics(monkeypatch, [], [])
    sync_client, _ = _install_fake_caproto(monkeypatch, reads, writes)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError) as exc:
        sync_client.write(CHANNEL, CURRENT + 50.0)

    assert exc.value.violation_type == "MAX_STEP_EXCEEDED"
    assert writes == []


def test_caproto_pv_write_measures_the_step_with_the_writing_pv(monkeypatch):
    reads: list = []
    writes: list = []
    _install_fake_epics(monkeypatch, [], [])
    _, threading_client = _install_fake_caproto(monkeypatch, reads, writes)
    _run_monkeypatch(monkeypatch)

    assert threading_client.PV(CHANNEL).write(CURRENT + 1.0) == "written"

    assert reads == [CHANNEL]
    assert writes == [(CHANNEL, CURRENT + 1.0)]
