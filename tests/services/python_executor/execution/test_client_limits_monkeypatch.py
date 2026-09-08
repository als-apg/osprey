"""Limits parity for the non-EPICS clients in the generated wrapper monkeypatch.

A readwrite run is limits-checked, but only for the clients the wrapper knows
how to intercept. Tango, DOOCS and caproto were in the *readonly* write surface
— so a readonly run refused them — while a readwrite run let them past the
limits database that ``epics.caput`` and every p4p flavour are checked against.
That asymmetry is what these tests pin shut.

Like the p4p monkeypatch tests, they execute the generated *source text*
against fake client modules injected into ``sys.modules``: the block is emitted
to run inside the executor subprocess, so there is no object to patch here.

Because that source runs in the *test* process, every client it imports is
faked before it is exec'd — aioca and p4p are really installed in this
environment, and exec'ing the block against them would rebind the installed
libraries for the rest of the session. The autouse fixture below is the second
net under that: whatever a test does manage to patch is put back afterwards.
"""

import asyncio
import contextlib
import importlib
import io
import sys
from types import ModuleType

import pytest

from osprey.connectors.control_system.limits_validator import (
    DEFAULT_STEP_READ_TIMEOUT_SECONDS,
    ChannelLimitsConfig,
    LimitsValidator,
)
from osprey.errors import ChannelLimitsViolationError
from osprey.services.python_executor.execution.wrapper import ExecutionWrapper
from osprey.services.python_executor.write_surface import (
    _CLIENT_WRITE_TARGETS,
    _LIMITS_REFUSED,
    _LIMITS_WRAPPED,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Restoration
# ---------------------------------------------------------------------------


def _resolve_target(dotted):
    """Resolve a write-surface row to the object that carries its attributes.

    A copy of the resolver the emitted guard uses, kept for the same reason
    ``test_readonly_guard.py`` keeps one: the guard has to be self-contained —
    it runs before user code in a subprocess that may not be able to import
    OSPREY at all — so there is no function to share.
    """
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for attr in parts[cut:]:
            try:
                obj = getattr(obj, attr)
            except AttributeError:
                return None
        return obj
    return None


@pytest.fixture(autouse=True)
def _restore_patched_targets():
    """Undo any client patch that escaped the fakes, after every test.

    The block these tests exec is written to patch installed control-system
    clients, and it is exec'd in the test process. The fakes below are what
    normally absorbs that, but a client the fakes miss — a row added to the
    write surface, a spelling the harness does not yet stand in for — would
    otherwise leave the installed library patched for the remainder of the
    session, and unrelated tests would fail on a limits refusal far from here.

    Snapshotting happens before a test injects its fakes, so it captures the
    real objects; restoring writes back to those same objects and is therefore
    unaffected by whatever ``sys.modules`` held in between.
    """
    saved = []
    for dotted, attrs in _CLIENT_WRITE_TARGETS:
        target = _resolve_target(dotted)
        if target is None:
            continue
        for attr in attrs:
            if hasattr(target, attr):
                saved.append((target, attr, getattr(target, attr)))
    yield
    for target, attr, value in saved:
        setattr(target, attr, value)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


#: Marks a module as one of this suite's stand-ins, so ``_run_monkeypatch``
#: can tell a fake a test already installed from a real installed client.
_FAKE_CLIENT_MARKER = "_osprey_fake_client"


def _fake_module(name):
    """A module object marked as one of this suite's client stand-ins."""
    mod = ModuleType(name)
    setattr(mod, _FAKE_CLIENT_MARKER, True)
    return mod


class _FakeChid:
    """What ``epics.ca`` addresses a channel by: an opaque id, not a name."""

    def __init__(self, pvname):
        self.pvname = pvname


def _install_fake_epics(monkeypatch, writes=None):
    """Inject an ``epics`` whose every write funnels through ``epics.ca.put``.

    Real pyepics spells a write three ways — ``caput``, ``PV.put`` and
    ``ca.put`` — and the first two reach the network through the third. The
    fake keeps that chain, so a wrapper installed on ``ca.put`` alone is
    provably the choke point for all three, and it looks ``ca.put`` up on the
    module at call time so a ``PV.put`` reaches the wrapper the block installs
    rather than the function captured when the class was defined.
    """
    writes = [] if writes is None else writes
    mod = _fake_module("epics")
    ca = _fake_module("epics.ca")

    def _ca_name(chid):
        return chid.pvname

    def _ca_create_channel(pvname, **kwargs):
        return _FakeChid(pvname)

    def _ca_put(chid, value, wait=False, timeout=60, **kwargs):
        writes.append((ca.name(chid), value))
        return 1

    def _ca_get(chid, timeout=60, **kwargs):
        return 1.0

    ca.name = _ca_name
    ca.create_channel = _ca_create_channel
    ca.put = _ca_put
    ca.get = _ca_get

    class PV:
        def __init__(self, pvname):
            self.pvname = pvname
            self.chid = ca.create_channel(pvname)

        def put(self, value, wait=False, timeout=60, **kwargs):
            return ca.put(self.chid, value, wait=wait, timeout=timeout, **kwargs)

        def get(self, **kwargs):
            return ca.get(self.chid)

    def caput(pvname, value, wait=False, timeout=60, **kwargs):
        return PV(pvname).put(value, wait=wait, timeout=timeout, **kwargs)

    def caget(pvname, timeout=60, **kwargs):
        return ca.get(ca.create_channel(pvname), timeout=timeout)

    mod.ca = ca
    mod.PV = PV
    mod.caput = caput
    mod.caget = caget
    monkeypatch.setitem(sys.modules, "epics", mod)
    monkeypatch.setitem(sys.modules, "epics.ca", ca)
    return mod


def _install_fake_aioca(monkeypatch, writes=None, reads=None, read=None):
    """Inject an ``aioca`` that shares one ``caput``/``caget`` with ``_catools``.

    On a real install ``aioca.caput is aioca._catools.caput`` — the package
    namespace re-exports what the submodule defines. A fake giving each module
    its own function would let a wrapper installed on one spelling look
    effective while the other still wrote.

    The array form keeps aioca's own re-entry: real ``caput`` dispatches a
    list of channels to ``caput_array``, which writes each pair by calling the
    module-global ``caput``. That is why the guard needs no array wrapper of
    its own — and a fake that wrote the pairs directly would hide the fact.

    ``read`` replaces what ``caget`` answers, and may raise: a client that
    cannot read is the case a step check has to fail closed on.
    """
    writes = [] if writes is None else writes
    reads = [] if reads is None else reads
    mod = _fake_module("aioca")
    catools = _fake_module("aioca._catools")

    async def caput(pv, value, *args, **kwargs):
        if isinstance(pv, (list, tuple)):
            return [
                await catools.caput(one_pv, one_value, *args, **kwargs)
                for one_pv, one_value in zip(pv, value, strict=True)
            ]
        writes.append((pv, value))
        return "put-done"

    async def caget(pv, *args, **kwargs):
        reads.append(pv)
        if read is not None:
            return read(pv)
        return 1.0

    for target in (mod, catools):
        target.caput = caput
        target.caget = caget
    mod._catools = catools
    monkeypatch.setitem(sys.modules, "aioca", mod)
    monkeypatch.setitem(sys.modules, "aioca._catools", catools)
    return mod


class _RecordingRawContext:
    """Stand-in for ``p4p.client.raw.Context``, the base every flavour puts through.

    Its ``put`` takes the raw signature — a handler and a ``builder`` — which
    is what distinguishes a direct raw put from a flavour's own put.
    """

    def __init__(self, provider="pva", **kwargs):
        self.puts = []
        self.rpcs = []

    def put(self, name, handler, builder=None, request=None, **kwargs):
        self.puts.append((name, builder))
        return "put-done"

    def rpc(self, name, handler, value=None, **kwargs):
        self.rpcs.append((name, value))
        return "rpc-done"

    def get(self, name, handler=None, request=None, **kwargs):
        return 1.0


def _install_fake_p4p(monkeypatch):
    """Inject a ``p4p`` package: one Context per flavour plus the raw base.

    p4p is installed in this environment, so without this the block would
    rebind the real ``Context.put`` of every flavour. The p4p *behaviour* is
    tested in ``test_p4p_monkeypatch.py``; here the fake exists only so these
    tests leave the installed library alone.
    """
    p4p_mod = _fake_module("p4p")
    client_mod = _fake_module("p4p.client")
    raw_mod = _fake_module("p4p.client.raw")
    raw_mod.Context = type("RawContext", (_RecordingRawContext,), {})

    class FlavorContext(raw_mod.Context):
        """A flavour Context: subclasses raw as p4p's flavours do, and takes
        the flavour ``put`` signature (a name and values, no handler)."""

        def put(self, name, values, request=None, timeout=5.0, **kwargs):
            self.puts.append((name, values))
            return "put-done"

        def rpc(self, name, value=None, request=None, timeout=5.0):
            self.rpcs.append((name, value))
            return "rpc-done"

        def get(self, name, request=None, timeout=5.0):
            return 1.0

    p4p_mod.client = client_mod
    client_mod.raw = raw_mod
    modules = {"p4p": p4p_mod, "p4p.client": client_mod, "p4p.client.raw": raw_mod}

    for flavor in ("thread", "asyncio", "cothread"):
        flavor_mod = _fake_module(f"p4p.client.{flavor}")
        flavor_mod.Context = type(f"{flavor.capitalize()}Context", (FlavorContext,), {})
        setattr(client_mod, flavor, flavor_mod)
        modules[f"p4p.client.{flavor}"] = flavor_mod

    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    # The C extension is not faked: it holds an immutable type the block never
    # patches, and leaving it unimportable keeps the real one out of reach.
    monkeypatch.setitem(sys.modules, "p4p._p4p", None)
    return p4p_mod


class _FakePvObject:
    """The structure a pvaccess ``Channel`` hands back and takes in.

    Shaped after the real binding, because pvaccess is not installed here and
    this fake is therefore the only evidence the guard binds correctly on it.
    pvaPy's no-arg ``getPyObject()`` answers the VALUE of the structure's
    ``value`` field and raises ``pvaccess.InvalidRequest`` when there is none;
    ``toDict()`` is the spelling that answers the whole structure as a dict.
    (The exception type is not part of the guard's contract — it lets whatever
    is raised propagate, and the write fails closed either way.)
    """

    def __init__(self, data):
        self._data = dict(data)

    def getPyObject(self):  # noqa: N802 - pvaccess spells it this way
        if "value" not in self._data:
            raise RuntimeError("PvObject has no value field")
        return self._data["value"]

    def toDict(self):  # noqa: N802 - pvaccess spells it this way
        return dict(self._data)


def _install_fake_pvaccess(monkeypatch, writes=None, reads=None, read=None):
    """Inject a ``pvaccess`` with the typed put family a Channel really carries.

    pvaPy spells a write as ``put`` plus a typed setter per scalar kind
    (``putDouble``, ``putInt``, …), which is why the block sweeps every
    ``put``-prefixed attribute rather than naming one. ``asyncPut``,
    ``parsePut`` and ``parsePutGet`` are the three writes whose names fall
    outside that prefix, and they are here so the sweep is tested against them.

    ``get`` answers a structure, as pvaPy's does, so a guard that reduces the
    payload but not the read is caught here. ``read`` replaces what it
    answers, and may raise: a channel that cannot be read is the case a step
    check has to fail closed on.
    """
    writes = [] if writes is None else writes
    reads = [] if reads is None else reads
    mod = _fake_module("pvaccess")

    class Channel:
        def __init__(self, name, provider=None):
            self._name = name
            self.current = 1.0

        def getName(self):  # noqa: N802 - pvaccess spells it this way
            return self._name

        def get(self, request=""):
            reads.append(self._name)
            if read is not None:
                return read(self._name)
            return _FakePvObject({"value": self.current})

        def put(self, value, request=""):
            writes.append((self._name, value))
            return "put-done"

        def putDouble(self, value, request=""):  # noqa: N802
            writes.append((self._name, value))
            return "put-done"

        def putGet(self, value, request=""):  # noqa: N802
            writes.append((self._name, value))
            return _FakePvObject({"value": value})

        def asyncPut(self, value, callback=None, request=""):  # noqa: N802
            # pvaPy's asynchronous write: a PvObject first, the completion
            # callback second. A write whose name does not start with "put".
            writes.append((self._name, value))
            return "async-put-done"

        def parsePut(self, args, request=""):  # noqa: N802
            # Takes a LIST OF JSON STRINGS, not a value object.
            writes.append((self._name, args))
            return "parse-put-done"

        def parsePutGet(self, args, request=""):  # noqa: N802
            writes.append((self._name, args))
            return _FakePvObject({"value": args})

    mod.Channel = Channel
    mod.PvObject = _FakePvObject
    monkeypatch.setitem(sys.modules, "pvaccess", mod)
    return mod


def _install_fake_tango(monkeypatch, writes=None, commands=None, group_calls=None, reads=None):
    """Inject a ``tango`` whose proxies and Group record what reaches them.

    PyTango is not installed in this environment, so this module is the only
    surface the tango branch of the block is ever exercised against, which
    makes it the tree's one statement of PyTango's binding layout. Two things
    it copies exactly:

    * ``Connection`` DEFINES both command spellings and ``DeviceProxy``
      merely inherits them (upstream ``tango/connection.py``), so a guard
      that patches only the subclass leaves
      ``tango.Connection.command_inout(proxy, 'On')`` live.
    * the signatures — ``command_inout(name, cmd_param=None, *, green_mode,
      wait, timeout)`` after ``green()`` rewrites it, and
      ``command_inout_asynch(cmd_name, *args)``, which takes ``forget``
      positionally and rejects it as a keyword.

    ``AttributeProxy`` is here for the third: it is bound to one attribute for
    its whole life, so its writes carry a value alone and its channel address
    has to be rebuilt from the device proxy behind it. Its ``read()`` answers a
    ``DeviceAttribute``, so the value the step is measured from sits behind
    ``.value`` rather than being the answer itself, and its three writes forward
    to the DeviceProxy spellings the way upstream binds them --- which is what
    makes an AttributeProxy write pass the limits check twice.

    ``Group`` is here for the same reason and is deliberately NOT a subclass of
    anything: upstream it is a plain Python class (``tango/group.py``) defining
    its own two commands and its own two attribute writes, which is why neither
    the DeviceProxy nor the Connection install reaches it. Upstream,
    ``write_attribute_asynch`` and ``command_inout_asynch`` are not in the
    class body at all: ``group_init()`` binds each as a ``(*args, **kwds)``
    proxy onto the C++ group, so the guard can only find them by attribute
    name, never by signature.

    Both classes are built fresh on every call, so a test that patches them
    leaves nothing behind for the next one.
    """
    writes = [] if writes is None else writes
    commands = [] if commands is None else commands
    group_calls = [] if group_calls is None else group_calls
    reads = [] if reads is None else reads
    mod = _fake_module("tango")

    class DeviceAttribute:
        """What a Tango read answers: the value sits behind ``.value``."""

        def __init__(self, value):
            self.value = value

    class AttributeInfo:
        """An attribute named by an object rather than by a string.

        PyTango accepts one of these wherever a write takes an attribute name,
        which is why the guard reduces the argument to ``.name`` before it
        builds the channel address.
        """

        def __init__(self, name):
            self.name = name

    class Connection:
        """The class PyTango really defines the two command spellings on."""

        def command_inout(self, name, cmd_param=None, *, green_mode=None, wait=None, timeout=None):
            commands.append((name, cmd_param))
            return "commanded"

        def command_inout_asynch(self, cmd_name, *args):
            commands.append((cmd_name, args[0] if args else None))
            return 1

    class DeviceProxy(Connection):
        def __init__(self, name):
            self._name = name

        def dev_name(self):
            return self._name

        def write_attribute(self, attr, value):
            writes.append((attr, value))
            return "written"

        def write_attribute_asynch(self, attr, value):
            writes.append((attr, value))
            return 1

        def write_read_attribute(self, attr, value):
            writes.append((attr, value))
            return DeviceAttribute(value)

        def write_attributes(self, name_val):
            writes.extend(list(name_val))
            return "written"

        def write_attributes_asynch(self, attr_values, cb=None):
            # PyTango names the pairs ``attr_values`` on this spelling alone.
            writes.extend(list(attr_values))
            return 2

        def write_read_attributes(self, name_val, attr_read_names=None):
            writes.extend(list(name_val))
            return "read-back-many"

        def read_attribute(self, attr):
            reads.append(attr)
            return 1.0

    class AttributeProxy:
        """PyTango's attribute proxy: one attribute, writes that carry a value alone."""

        def __init__(self, full_name):
            self._device, self._attr = full_name.rsplit("/", 1)

        def get_device_proxy(self):
            return DeviceProxy(self._device)

        def name(self):
            return self._attr

        def read(self):
            reads.append(self._attr)
            return DeviceAttribute(1.0)

        # Upstream these three are bound onto the class as calls on the device
        # proxy behind the attribute, so each one reaches the machine through
        # a DeviceProxy write spelling rather than on its own.
        def write(self, value):
            return self.get_device_proxy().write_attribute(self._attr, value)

        def write_asynch(self, value):
            return self.get_device_proxy().write_attribute_asynch(self._attr, value)

        def write_read(self, value):
            return self.get_device_proxy().write_read_attribute(self._attr, value)

    class Group:
        """PyTango's group: its own class, with its own four write spellings."""

        def __init__(self, name):
            self._name = name

        def get_name(self):
            return self._name

        def add(self, pattern):
            return None

        def write_attribute(self, attr_name, value, forward=True, multi=False):
            group_calls.append(("write_attribute", attr_name, value))
            return "written"

        def write_attribute_asynch(self, attr_name, value, forward=True, multi=False):
            group_calls.append(("write_attribute_asynch", attr_name, value))
            return 1

        def command_inout(self, cmd_name, param=None, forward=True):
            group_calls.append(("command_inout", cmd_name, param))
            return "commanded"

        def command_inout_asynch(self, cmd_name, param=None, forget=False, forward=True):
            group_calls.append(("command_inout_asynch", cmd_name, param))
            return 1

    mod.Connection = Connection
    mod.DeviceProxy = DeviceProxy
    mod.AttributeInfo = AttributeInfo
    mod.AttributeProxy = AttributeProxy
    mod.Group = Group
    mod.group_calls = group_calls
    monkeypatch.setitem(sys.modules, "tango", mod)
    return mod


def _install_fake_doocs(monkeypatch, writes=None):
    writes = [] if writes is None else writes
    mod = _fake_module("doocs4py")

    def _set(address, value):
        writes.append((address, value))
        return "written"

    mod.set = _set
    monkeypatch.setitem(sys.modules, "doocs4py", mod)
    return mod


class _CaprotoResponse:
    """caproto answers a read with a response whose ``data`` is an array."""

    def __init__(self, value):
        self.data = [value]


def _install_fake_caproto(monkeypatch, writes=None, reads=None, read_kwargs=None):
    """Inject a fake ``caproto`` with every write entry point, in all three flavors."""
    writes = [] if writes is None else writes
    reads = [] if reads is None else reads
    read_kwargs = [] if read_kwargs is None else read_kwargs
    caproto_mod = _fake_module("caproto")
    sync_mod = _fake_module("caproto.sync")
    sync_client = _fake_module("caproto.sync.client")
    threading_mod = _fake_module("caproto.threading")
    threading_client = _fake_module("caproto.threading.client")
    asyncio_mod = _fake_module("caproto.asyncio")
    asyncio_client = _fake_module("caproto.asyncio.client")

    def write(pv_name, data, **kwargs):
        writes.append((pv_name, data))
        return "written"

    def read(pv_name, **kwargs):
        reads.append(pv_name)
        read_kwargs.append(kwargs)
        return _CaprotoResponse(1.0)

    def read_write_read(pv_name, data, **kwargs):
        # Upstream this reads the channel, writes it, and reads it back; the
        # value it drives the channel with is the one the plain write drives.
        writes.append((pv_name, data))
        return "read-write-read"

    class PV:
        def __init__(self, name):
            self.name = name

        def write(self, data, **kwargs):
            writes.append((self.name, data))
            return "written"

        # The bare-value shape ``_caproto_scalar`` reduces alongside a
        # response object, so both of its branches are exercised here.
        def read(self, **kwargs):
            reads.append(self.name)
            read_kwargs.append(kwargs)
            return 1.0

    class Batch:
        """caproto's request batcher: its write carries the PV to drive."""

        def write(self, pv, data, callback=None, **kwargs):
            writes.append((pv.name, data))
            return "batched"

    class AsyncPV:
        """caproto's asyncio PV: read and write are coroutines."""

        def __init__(self, name):
            self.name = name

        async def write(self, data, **kwargs):
            writes.append((self.name, data))
            return "written"

        async def read(self, **kwargs):
            reads.append(self.name)
            read_kwargs.append(kwargs)
            return _CaprotoResponse(1.0)

    sync_client.write = write
    sync_client.read = read
    sync_client.read_write_read = read_write_read
    threading_client.PV = PV
    threading_client.Batch = Batch
    asyncio_client.PV = AsyncPV
    sync_mod.client = sync_client
    threading_mod.client = threading_client
    asyncio_mod.client = asyncio_client
    caproto_mod.sync = sync_mod
    caproto_mod.threading = threading_mod
    caproto_mod.asyncio = asyncio_mod

    for name, mod in (
        ("caproto", caproto_mod),
        ("caproto.sync", sync_mod),
        ("caproto.sync.client", sync_client),
        ("caproto.threading", threading_mod),
        ("caproto.threading.client", threading_client),
        ("caproto.asyncio", asyncio_mod),
        ("caproto.asyncio.client", asyncio_client),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return sync_client, threading_client, asyncio_client


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
        # The Tango channel that configures max_step, so a test can tell a
        # guard that reads the present value from one that never does.
        "sys/tg_test/1/step": ChannelLimitsConfig(
            channel_address="sys/tg_test/1/step",
            min_value=0.0,
            max_value=100.0,
            max_step=2.0,
        ),
        "FACILITY/MAGNET/H1/CURRENT.SP": ChannelLimitsConfig(
            channel_address="FACILITY/MAGNET/H1/CURRENT.SP",
            min_value=0.0,
            max_value=10.0,
        ),
        "TEST:MAG:SP": ChannelLimitsConfig(
            channel_address="TEST:MAG:SP", min_value=0.0, max_value=10.0
        ),
        # The one channel here that configures max_step, so a test can tell a
        # guard that reads the present value from one that never does.
        "TEST:MAG:STEP": ChannelLimitsConfig(
            channel_address="TEST:MAG:STEP",
            min_value=0.0,
            max_value=100.0,
            max_step=2.0,
        ),
    }
    return LimitsValidator(limits, {"allow_unlisted_channels": False})


#: Every client the emitted block imports, with the fake that stands in for it.
#: A client missing from this list is one the block could patch for real.
_CLIENT_FAKES = (
    ("epics", _install_fake_epics),
    ("aioca", _install_fake_aioca),
    ("p4p", _install_fake_p4p),
    ("pvaccess", _install_fake_pvaccess),
    ("tango", _install_fake_tango),
    ("doocs4py", _install_fake_doocs),
    ("caproto", _install_fake_caproto),
)

_UNSET = object()


def _needs_fake(name):
    """Whether ``_run_monkeypatch`` should stand in for this client itself.

    A test that installed its own fake keeps it — recording writes is the
    point of most of them — and so does a test that set the module to ``None``
    to exercise the block's "not available" branch. Anything else, including a
    client really installed in this environment, is replaced.
    """
    entry = sys.modules.get(name, _UNSET)
    if entry is None:
        return False
    return entry is _UNSET or not getattr(entry, _FAKE_CLIENT_MARKER, False)


def _install_client_fakes(monkeypatch):
    """Stand in for every client the block imports, without running it yet.

    ``_run_monkeypatch`` opens with this. A test that has to look at a client
    object on *both* sides of the block calls it first, so the objects it
    snapshots are the ones the block goes on to patch.
    """
    for name, installer in _CLIENT_FAKES:
        if _needs_fake(name):
            installer(monkeypatch)


def _run_monkeypatch(monkeypatch):
    """Execute the generated monkeypatch block against fakes; return its stdout."""
    _install_client_fakes(monkeypatch)

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
# pyepics
# ---------------------------------------------------------------------------


def _count_validations(monkeypatch):
    """Record every ``LimitsValidator.validate`` the emitted block performs.

    Refusing an out-of-range write proves a guard is somewhere on the path;
    it does not prove there is only one. The count is what separates a single
    wrapper at the choke point from wrappers stacked on ``caput``, ``PV.put``
    and ``ca.put`` alike — three validations, and up to three ``max_step``
    reads, for one write.
    """
    calls: list = []
    original = LimitsValidator.validate

    def _recording(self, channel_address, value, **kwargs):
        calls.append((channel_address, value))
        return original(self, channel_address, value, **kwargs)

    monkeypatch.setattr(LimitsValidator, "validate", _recording)
    return calls


def test_ca_put_within_limits_reaches_the_channel(monkeypatch):
    writes: list = []
    mod = _install_fake_epics(monkeypatch, writes)
    calls = _count_validations(monkeypatch)
    _run_monkeypatch(monkeypatch)

    chid = mod.ca.create_channel("TEST:MAG:SP")
    assert mod.ca.put(chid, 5.0) == 1
    assert writes == [("TEST:MAG:SP", 5.0)]
    assert calls == [("TEST:MAG:SP", 5.0)]
    assert mod.ca.get(chid) == 1.0, "reads must survive the guard untouched"


def test_ca_put_out_of_bounds_raises_before_the_channel(monkeypatch):
    """The lowest pyepics spelling is checked, not just the convenience ones.

    ``epics.ca.put`` is the function every other pyepics write ends at, and it
    is also callable directly — a script that creates its own channel and puts
    through it used to reach the network with no limits check at all.
    """
    writes: list = []
    mod = _install_fake_epics(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    chid = mod.ca.create_channel("TEST:MAG:SP")
    with pytest.raises(ChannelLimitsViolationError):
        mod.ca.put(chid, 99.0)
    assert writes == []


def test_ca_put_names_the_channel_from_the_chid(monkeypatch):
    """A chid is opaque, so the guard has to ask ``ca.name`` what it addresses.

    Validating the chid itself looks up a channel the database has never heard
    of, which on a deployment that allows unlisted channels is a write that
    passes unchecked.
    """
    writes: list = []
    mod = _install_fake_epics(monkeypatch, writes)
    calls = _count_validations(monkeypatch)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError):
        mod.ca.put(mod.ca.create_channel("TEST:MAG:SP"), 99.0)
    assert calls == [("TEST:MAG:SP", 99.0)]


def test_pv_put_is_refused_at_the_ca_put_choke_point(monkeypatch):
    writes: list = []
    mod = _install_fake_epics(monkeypatch, writes)
    calls = _count_validations(monkeypatch)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError):
        mod.PV("TEST:MAG:SP").put(99.0)
    assert writes == []
    assert calls == [("TEST:MAG:SP", 99.0)]


def test_caput_validates_once_on_its_way_through_ca_put(monkeypatch):
    """``caput`` reaches the network through ``PV.put`` and then ``ca.put``.

    With a wrapper on each spelling the same write was validated at every
    layer it passed; with one at the bottom it is validated once, and pays for
    one ``max_step`` read instead of three.
    """
    writes: list = []
    mod = _install_fake_epics(monkeypatch, writes)
    calls = _count_validations(monkeypatch)
    _run_monkeypatch(monkeypatch)

    assert mod.caput("TEST:MAG:SP", 5.0) == 1
    assert writes == [("TEST:MAG:SP", 5.0)]
    assert calls == [("TEST:MAG:SP", 5.0)]


def test_caput_out_of_bounds_is_refused_at_the_choke_point(monkeypatch):
    writes: list = []
    mod = _install_fake_epics(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError):
        mod.caput("TEST:MAG:SP", 99.0)
    assert writes == []


def test_the_block_wraps_only_ca_put(monkeypatch):
    """``caput`` and ``PV.put`` are left as pyepics defined them.

    They are covered because they route through ``ca.put``, so wrapping them
    as well buys nothing and costs a duplicate validation per write.
    """
    mod = _install_fake_epics(monkeypatch)
    original_caput = mod.caput
    original_pv_put = mod.PV.put
    original_ca_put = mod.ca.put
    _run_monkeypatch(monkeypatch)

    assert mod.caput is original_caput
    assert mod.PV.put is original_pv_put
    assert mod.ca.put is not original_ca_put


def test_absent_pyepics_does_not_stop_the_block(monkeypatch):
    monkeypatch.setitem(sys.modules, "epics", None)
    monkeypatch.setitem(sys.modules, "epics.ca", None)
    writes: list = []
    _install_fake_doocs(monkeypatch, writes)

    out = _run_monkeypatch(monkeypatch)
    assert "pyepics not available" in out
    assert "✅ Monkeypatched doocs4py.set()" in out


# ---------------------------------------------------------------------------
# aioca
# ---------------------------------------------------------------------------


def test_aioca_caput_within_limits_reaches_the_channel(monkeypatch):
    writes: list = []
    reads: list = []
    mod = _install_fake_aioca(monkeypatch, writes, reads)
    _run_monkeypatch(monkeypatch)

    assert asyncio.run(mod.caput("TEST:MAG:SP", 5.0)) == "put-done"
    assert writes == [("TEST:MAG:SP", 5.0)]
    # A channel with no max_step has no step to measure, so the guard buys no
    # read for it — the round trip is paid for only where it is needed.
    assert reads == []


def test_aioca_caput_out_of_bounds_raises_before_the_channel(monkeypatch):
    """aioca is a second Channel Access client, not a spelling of pyepics.

    Its writes never pass through ``epics.ca.put``, so the pyepics choke point
    leaves an ``await aioca.caput(...)`` unchecked; this is the guard that
    closes it.
    """
    writes: list = []
    mod = _install_fake_aioca(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError):
        asyncio.run(mod.caput("TEST:MAG:SP", 99.0))
    assert writes == []


def test_aioca_array_caput_validates_every_pair(monkeypatch):
    """The list form is checked per pair, through aioca's own ``caput_array``.

    A list of channels is forwarded unchanged, because aioca dispatches it to
    ``caput_array``, which writes each pair by calling the module-global
    ``caput`` — this guard. Validating the list against a single channel's
    limits here would refuse the whole write for the wrong reason and still
    check no pair.
    """
    writes: list = []
    mod = _install_fake_aioca(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError):
        asyncio.run(mod.caput(["TEST:MAG:SP", "TEST:MAG:SP"], [5.0, 99.0]))

    # The in-range pair went through on its own way past the guard; the
    # out-of-range one was refused before it reached the channel.
    assert writes == [("TEST:MAG:SP", 5.0)]


def test_aioca_caput_beyond_max_step_reads_through_aiocas_own_caget(monkeypatch):
    """The step is measured over aioca, awaiting its own ``caget``.

    The validator is synchronous and owns no client; the guard is a coroutine,
    so it can await the read itself and hand the answer over as a plain value.
    """
    writes: list = []
    reads: list = []
    mod = _install_fake_aioca(monkeypatch, writes, reads)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError) as exc:
        asyncio.run(mod.caput("TEST:MAG:STEP", 50.0))

    assert exc.value.violation_type == "MAX_STEP_EXCEEDED"
    assert reads == ["TEST:MAG:STEP"]
    assert writes == []


def test_aioca_caput_within_max_step_reaches_the_channel(monkeypatch):
    writes: list = []
    reads: list = []
    mod = _install_fake_aioca(monkeypatch, writes, reads)
    _run_monkeypatch(monkeypatch)

    # The fake reads back 1.0, so this is a step of 1.0 against a max of 2.0.
    assert asyncio.run(mod.caput("TEST:MAG:STEP", 2.0)) == "put-done"
    assert reads == ["TEST:MAG:STEP"]
    assert writes == [("TEST:MAG:STEP", 2.0)]


def test_aioca_caput_fails_closed_when_its_caget_raises(monkeypatch):
    """A client that cannot read refuses the write; it does not skip the check.

    The guard swallows the read error so the refusal comes from the validator
    with the channel and value attached — but swallowing it must not turn into
    letting the write past, which is what an unmeasured step would be.
    """
    writes: list = []
    reads: list = []

    def _raise(_pv):
        raise OSError("channel unreachable")

    mod = _install_fake_aioca(monkeypatch, writes, reads, read=_raise)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError) as exc:
        asyncio.run(mod.caput("TEST:MAG:STEP", 2.0))

    assert exc.value.violation_type == "STEP_CHECK_FAILED"
    assert reads == ["TEST:MAG:STEP"]
    assert writes == []


def test_aioca_caput_fails_closed_when_its_caget_answers_none(monkeypatch):
    """A read that succeeds with nothing in it is no measurement either."""
    writes: list = []
    reads: list = []
    mod = _install_fake_aioca(monkeypatch, writes, reads, read=lambda _pv: None)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError) as exc:
        asyncio.run(mod.caput("TEST:MAG:STEP", 2.0))

    assert exc.value.violation_type == "STEP_CHECK_FAILED"
    assert reads == ["TEST:MAG:STEP"]
    assert writes == []


def test_the_block_rebinds_both_aioca_caput_spellings(monkeypatch):
    """``aioca.caput`` and ``aioca._catools.caput`` are the same function.

    The package namespace re-exports what the submodule defines, so patching
    only the re-export would leave ``from aioca._catools import caput``
    unchecked — and would also break the array form, whose per-pair re-entry
    goes through the submodule global.
    """
    mod = _install_fake_aioca(monkeypatch)
    original_caput = mod.caput
    _run_monkeypatch(monkeypatch)

    catools = sys.modules["aioca._catools"]
    assert mod.caput is not original_caput
    assert catools.caput is mod.caput


def test_absent_aioca_does_not_stop_the_block(monkeypatch):
    monkeypatch.setitem(sys.modules, "aioca", None)
    monkeypatch.setitem(sys.modules, "aioca._catools", None)
    writes: list = []
    _install_fake_doocs(monkeypatch, writes)

    out = _run_monkeypatch(monkeypatch)
    assert "aioca not available" in out
    assert "✅ Monkeypatched doocs4py.set()" in out


# ---------------------------------------------------------------------------
# pvaPy (pvaccess)
# ---------------------------------------------------------------------------


def test_pvaccess_put_within_limits_reaches_the_channel(monkeypatch):
    writes: list = []
    reads: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes, reads)
    _run_monkeypatch(monkeypatch)

    assert mod.Channel("TEST:MAG:SP").put(5.0) == "put-done"
    assert writes == [("TEST:MAG:SP", 5.0)]
    # No max_step on this channel, so the guard buys no read for it.
    assert reads == []


def test_pvaccess_put_out_of_bounds_raises_before_the_channel(monkeypatch):
    """pvaPy is a PVAccess client of its own, not a p4p spelling.

    A ``pvaccess.Channel`` put never passes through a p4p ``Context``, so the
    p4p guards leave it unchecked; this is the guard that closes it.
    """
    writes: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError):
        mod.Channel("TEST:MAG:SP").put(99.0)
    assert writes == []


def test_pvaccess_typed_put_is_refused_out_of_range(monkeypatch):
    """The typed setters are the same write under another name.

    pvaPy spells one setter per scalar and array kind — ``putDouble``,
    ``putInt``, ``putScalarArray`` — so the guard sweeps every ``put``-prefixed
    attribute rather than naming ``put``. Naming them would go stale against
    the binding, and every name missed would be an unchecked write.
    """
    writes: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError):
        mod.Channel("TEST:MAG:SP").putDouble(99.0)
    assert writes == []


def test_pvaccess_structure_payload_is_reduced_before_validation(monkeypatch):
    """A ``PvObject`` payload is unwrapped to the number it carries.

    Handed the structure itself, the validator finds nothing numeric to check
    and lets the write past — so an out-of-range value dressed as a structure
    would reach the machine.
    """
    writes: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError):
        mod.Channel("TEST:MAG:SP").put(_FakePvObject({"value": 99.0}))
    assert writes == []


def test_pvaccess_dict_payload_is_reduced_before_validation(monkeypatch):
    """The plain dict a structure converts to is unwrapped the same way."""
    writes: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError):
        mod.Channel("TEST:MAG:SP").put({"value": 99.0})
    assert writes == []


def test_pvaccess_dict_without_a_value_field_fails_closed(monkeypatch):
    """A shape with no value under it is refused, not written unchecked."""
    writes: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ValueError):
        mod.Channel("TEST:MAG:SP").put({"unit": "A"})
    assert writes == []


def test_pvaccess_callable_payload_fails_closed(monkeypatch):
    """A callable is not a value a limits database can be asked about."""
    writes: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ValueError):
        mod.Channel("TEST:MAG:SP").put(lambda: 5.0)
    assert writes == []


def test_pvaccess_step_check_reads_through_the_channels_own_get(monkeypatch):
    """The step is measured over the same Channel the put goes through.

    pvaPy answers a get with a structure, so the reader reduces it exactly as
    it reduces a payload. Left unreduced, the structure is non-numeric, the
    validator SKIPS the step check, and a 49-unit step past a max of 2 reaches
    the machine — which is why this asserts the refusal is ``MAX_STEP_EXCEEDED``
    and not merely that something raised.
    """
    writes: list = []
    reads: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes, reads)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError) as exc:
        mod.Channel("TEST:MAG:STEP").put(50.0)

    assert exc.value.violation_type == "MAX_STEP_EXCEEDED"
    assert reads == ["TEST:MAG:STEP"]
    assert writes == []


def test_pvaccess_put_within_max_step_reaches_the_channel(monkeypatch):
    """The reader is a real read, not a blanket refusal on max_step channels."""
    writes: list = []
    reads: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes, reads)
    _run_monkeypatch(monkeypatch)

    # The fake reads back 1.0, so this is a step of 1.0 against a max of 2.0.
    assert mod.Channel("TEST:MAG:STEP").put(2.0) == "put-done"
    assert reads == ["TEST:MAG:STEP"]
    assert writes == [("TEST:MAG:STEP", 2.0)]


def test_pvaccess_step_check_fails_closed_when_the_read_fails(monkeypatch):
    """A Channel that cannot be read refuses the write rather than skipping it."""
    writes: list = []
    reads: list = []

    def _raise(_name):
        raise OSError("channel unreachable")

    mod = _install_fake_pvaccess(monkeypatch, writes, reads, read=_raise)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError) as exc:
        mod.Channel("TEST:MAG:STEP").put(2.0)

    assert exc.value.violation_type == "STEP_CHECK_FAILED"
    assert writes == []


def test_pvaccess_async_put_out_of_range_is_refused(monkeypatch):
    """``asyncPut`` is a write whose name does not start with ``put``.

    pvaPy spells the asynchronous write ``asyncPut(pvObject, callback)`` — the
    same put with a completion callback bolted on, and the same value going to
    the machine. A sweep keyed on the ``put`` prefix alone leaves it out, so an
    approved run could send an out-of-range value with no check at all.
    """
    writes: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError):
        mod.Channel("TEST:MAG:SP").asyncPut(_FakePvObject({"value": 99.0}), lambda _r: None)
    assert writes == []


def test_pvaccess_async_put_within_limits_reaches_the_channel(monkeypatch):
    """The first positional is a PvObject, so the ordinary reduction covers it."""
    writes: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    channel = mod.Channel("TEST:MAG:SP")
    assert channel.asyncPut(_FakePvObject({"value": 5.0}), lambda _r: None) == "async-put-done"
    assert len(writes) == 1
    assert writes[0][0] == "TEST:MAG:SP"


def test_pvaccess_parse_put_cannot_be_limits_checked(monkeypatch):
    """``parsePut``/``parsePutGet`` carry JSON strings, so they are refused.

    Their payload is a list of ``field=value`` strings parsed against the
    channel's introspected structure — there is no value object to reduce, and
    the wrapper cannot know which string names the field the limits are about.
    Guessing would be a check that silently means nothing, so the write is
    refused with the same posture the p4p guard takes for a callable payload:
    an in-range value is refused too, and the caller is pointed at ``put``.
    """
    writes: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ValueError, match="cannot be limits-checked"):
        mod.Channel("TEST:MAG:SP").parsePut(["value=5.0"])
    with pytest.raises(ValueError, match="cannot be limits-checked"):
        mod.Channel("TEST:MAG:SP").parsePutGet(["value=5.0"])
    assert writes == []


def test_pvaccess_structure_payload_over_max_step_is_refused(monkeypatch):
    """A structure payload is stepped against a structure read, both reduced.

    This is the shape a real pvaPy run has on both sides — ``PvObject`` in,
    ``PvObject`` back from ``get`` — and neither is a number until the guard
    reduces it. Unreduced, the validator finds nothing numeric and SKIPS the
    step check, so the assertion is on ``MAX_STEP_EXCEEDED`` specifically.
    """
    writes: list = []
    reads: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes, reads)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError) as exc:
        mod.Channel("TEST:MAG:STEP").put(_FakePvObject({"value": 50.0}))

    assert exc.value.violation_type == "MAX_STEP_EXCEEDED"
    assert reads == ["TEST:MAG:STEP"]
    assert writes == []


def test_pvaccess_valueless_structure_payload_fails_closed(monkeypatch):
    """A structure with no value under it is refused by pvaPy's own accessor.

    ``PvObject.getPyObject()`` raises when the structure has no ``value``
    field, which happens before the guard's own dict branch is reached. The
    exception propagates and the write never leaves — a different exception
    type from the plain-dict refusal, the same fail-closed outcome.
    """
    writes: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(RuntimeError):
        mod.Channel("TEST:MAG:SP").put(_FakePvObject({"unit": "A"}))
    assert writes == []


def test_fake_pv_object_answers_the_way_pvapy_does():
    """The fake is the only surface standing in for binding on the real library.

    pvaPy's no-arg ``PvObject.getPyObject()`` answers the VALUE of the
    ``value`` field and raises ``pvaccess.InvalidRequest`` when there is none;
    ``toDict()`` is the spelling that answers the whole structure as a dict. A
    fake that answered a dict from ``getPyObject()`` would test the guard only
    against a shape a real PvObject never produces. (The exception type is not
    part of the contract — the guard lets whatever is raised propagate.)
    """
    payload = _FakePvObject({"value": 99.0, "unit": "A"})

    assert payload.getPyObject() == 99.0
    assert payload.toDict() == {"value": 99.0, "unit": "A"}
    with pytest.raises(RuntimeError):
        _FakePvObject({"unit": "A"}).getPyObject()


def test_pvaccess_without_a_channel_class_reports_the_gap(monkeypatch):
    """A pvaccess module with no ``Channel`` must not report itself guarded.

    The success line is the operator's only evidence the guard is on. A stub or
    broken install that carries no ``Channel`` wraps nothing, and saying so is
    the difference between a known gap and an unchecked write believed checked.
    """
    monkeypatch.setitem(sys.modules, "pvaccess", _fake_module("pvaccess"))
    writes: list = []
    _install_fake_doocs(monkeypatch, writes)

    out = _run_monkeypatch(monkeypatch)
    assert "no Channel class" in out
    assert "✅ Monkeypatched pvaccess" not in out
    # The rest of the block still runs.
    assert "✅ Monkeypatched doocs4py.set()" in out


def test_pvaccess_channel_without_a_put_reports_the_gap(monkeypatch):
    """A ``Channel`` the sweep found nothing on is the same gap, and says so."""
    mod = _fake_module("pvaccess")

    class Channel:
        def get(self):
            return 1.0

    mod.Channel = Channel
    monkeypatch.setitem(sys.modules, "pvaccess", mod)

    out = _run_monkeypatch(monkeypatch)
    assert "⚠️  pvaccess guard failed: Channel has no put method" in out
    assert "✅ Monkeypatched pvaccess" not in out


def test_absent_pvaccess_does_not_stop_the_block(monkeypatch):
    monkeypatch.setitem(sys.modules, "pvaccess", None)
    writes: list = []
    _install_fake_doocs(monkeypatch, writes)

    out = _run_monkeypatch(monkeypatch)
    assert "pvaccess not available" in out
    assert "✅ Monkeypatched doocs4py.set()" in out


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


def test_tango_write_attribute_asynch_within_limits_reaches_the_device(monkeypatch):
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    assert proxy.write_attribute_asynch("current", 5.0) == 1, (
        "the request id the caller polls on must come back untouched"
    )
    assert writes == [("current", 5.0)]


def test_tango_write_attribute_asynch_out_of_bounds_raises_before_the_device(monkeypatch):
    """The asynchronous spelling drives the same device with the same value."""
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    with pytest.raises(ChannelLimitsViolationError):
        proxy.write_attribute_asynch("current", 99.0)
    assert writes == []


def test_tango_write_read_attribute_within_limits_reaches_the_device(monkeypatch):
    """The read-back the call answers is the caller's result, not the guard's."""
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    assert proxy.write_read_attribute("current", 5.0).value == 5.0
    assert writes == [("current", 5.0)]


def test_tango_write_read_attribute_out_of_bounds_raises_before_the_device(monkeypatch):
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    with pytest.raises(ChannelLimitsViolationError):
        proxy.write_read_attribute("current", 99.0)
    assert writes == []


def test_tango_write_attributes_asynch_checks_every_pair(monkeypatch):
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    assert proxy.write_attributes_asynch([("current", 5.0), ("voltage", 4.0)]) == 2
    assert writes == [("current", 5.0), ("voltage", 4.0)]


def test_tango_write_attributes_asynch_out_of_bounds_raises_before_the_device(monkeypatch):
    """One pair out of range stops the batch: none of it reaches the device."""
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    with pytest.raises(ChannelLimitsViolationError):
        proxy.write_attributes_asynch([("current", 5.0), ("voltage", 99.0)])
    assert writes == []


def test_tango_write_read_attributes_checks_every_pair(monkeypatch):
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    written = proxy.write_read_attributes([("current", 5.0), ("voltage", 4.0)])
    assert written == "read-back-many"
    assert writes == [("current", 5.0), ("voltage", 4.0)]


def test_tango_write_read_attributes_out_of_bounds_raises_before_the_device(monkeypatch):
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    with pytest.raises(ChannelLimitsViolationError):
        proxy.write_read_attributes([("current", 5.0), ("voltage", 99.0)])
    assert writes == []


def test_tango_write_read_attribute_measures_the_step_over_the_writing_proxy(monkeypatch):
    """A max_step channel is read through the proxy the write goes through."""
    writes: list = []
    reads: list = []
    mod = _install_fake_tango(monkeypatch, writes, reads=reads)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    with pytest.raises(ChannelLimitsViolationError) as exc:
        proxy.write_read_attribute("step", 50.0)

    assert exc.value.violation_type == "MAX_STEP_EXCEEDED"
    assert reads == ["step"]
    assert writes == []

    reads.clear()
    # The fake reads back 1.0, so this is a step of 1.0 against a max of 2.0.
    assert proxy.write_read_attribute("step", 2.0).value == 2.0
    assert reads == ["step"]
    assert writes == [("step", 2.0)]


def test_tango_attribute_proxy_write_within_limits_reaches_the_device(monkeypatch):
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    attribute = mod.AttributeProxy("sys/tg_test/1/current")
    assert attribute.write(5.0) == "written"
    assert writes == [("current", 5.0)]


def test_tango_attribute_proxy_write_out_of_bounds_raises_before_the_device(monkeypatch):
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    attribute = mod.AttributeProxy("sys/tg_test/1/current")
    with pytest.raises(ChannelLimitsViolationError):
        attribute.write(99.0)
    assert writes == []


def test_tango_attribute_proxy_write_asynch_within_limits_reaches_the_device(monkeypatch):
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    attribute = mod.AttributeProxy("sys/tg_test/1/current")
    assert attribute.write_asynch(5.0) == 1, (
        "the request id the caller polls on must come back untouched"
    )
    assert writes == [("current", 5.0)]


def test_tango_attribute_proxy_write_asynch_out_of_bounds_raises_before_the_device(monkeypatch):
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    attribute = mod.AttributeProxy("sys/tg_test/1/current")
    with pytest.raises(ChannelLimitsViolationError):
        attribute.write_asynch(99.0)
    assert writes == []


def test_tango_attribute_proxy_write_read_within_limits_reaches_the_device(monkeypatch):
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    attribute = mod.AttributeProxy("sys/tg_test/1/current")
    assert attribute.write_read(5.0).value == 5.0
    assert writes == [("current", 5.0)]


def test_tango_attribute_proxy_write_read_out_of_bounds_raises_before_the_device(monkeypatch):
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    attribute = mod.AttributeProxy("sys/tg_test/1/current")
    with pytest.raises(ChannelLimitsViolationError):
        attribute.write_read(99.0)
    assert writes == []


def test_tango_attribute_proxy_channel_is_the_full_device_attribute_address(monkeypatch):
    """An AttributeProxy write carries a value alone, so it names no channel.

    The address is rebuilt from the device proxy behind the attribute and the
    attribute's own name. Validating anything narrower would look up a channel
    the database has never heard of --- which, on a deployment that allows
    unlisted channels, is a write that passes without being checked at all.
    """
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    # The same attribute name on another device is not in the database.
    attribute = mod.AttributeProxy("sys/tg_test/2/current")
    with pytest.raises(ChannelLimitsViolationError):
        attribute.write(5.0)
    assert writes == []


def test_tango_attribute_proxy_measures_the_step_over_its_own_read(monkeypatch):
    """The step is measured from the value behind the read-back's ``.value``.

    An AttributeProxy read answers a ``DeviceAttribute``, not a number. Left
    unreduced it is non-numeric, the validator SKIPS the step check, and a
    49-unit step past a max of 2 reaches the machine --- which is why this
    asserts the refusal is ``MAX_STEP_EXCEEDED`` and not merely that something
    raised.
    """
    writes: list = []
    reads: list = []
    mod = _install_fake_tango(monkeypatch, writes, reads=reads)
    _run_monkeypatch(monkeypatch)

    attribute = mod.AttributeProxy("sys/tg_test/1/step")
    with pytest.raises(ChannelLimitsViolationError) as exc:
        attribute.write(50.0)

    assert exc.value.violation_type == "MAX_STEP_EXCEEDED"
    assert reads == ["step"]
    assert writes == []

    reads.clear()
    # The fake reads back 1.0, so this is a step of 1.0 against a max of 2.0.
    # The write forwards to the wrapped DeviceProxy spelling, so the step is
    # measured once by each proxy's guard.
    assert attribute.write(2.0) == "written"
    assert reads == ["step", "step"]
    assert writes == [("step", 2.0)]


def test_tango_attribute_proxy_asynch_and_read_back_writes_measure_the_step(monkeypatch):
    """The other two AttributeProxy spellings carry a guard of their own.

    All three forward to a DeviceProxy spelling that is itself wrapped, so a
    refusal alone cannot say which guard produced it: these would still refuse
    with the AttributeProxy wrappers gone. Counting the reads does say it --- a
    step measured twice is one measurement per proxy, which only happens while
    both guards are installed.
    """
    writes: list = []
    reads: list = []
    mod = _install_fake_tango(monkeypatch, writes, reads=reads)
    _run_monkeypatch(monkeypatch)

    attribute = mod.AttributeProxy("sys/tg_test/1/step")

    # The fake reads back 1.0, so each of these is a step of 1.0 against a max
    # of 2.0 --- within limits, and through both guards on the way out.
    assert attribute.write_asynch(2.0) == 1, (
        "the request id the caller polls on must come back untouched"
    )
    assert reads == ["step", "step"]
    assert writes == [("step", 2.0)]

    reads.clear()
    writes.clear()
    assert attribute.write_read(2.0).value == 2.0
    assert reads == ["step", "step"]
    assert writes == [("step", 2.0)]


def test_tango_write_attributes_takes_the_pairs_by_keyword(monkeypatch):
    """PyTango's own keyword form reaches the device through the guard."""
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    assert proxy.write_attributes(name_val=[("current", 5.0), ("voltage", 4.0)]) == "written"
    assert writes == [("current", 5.0), ("voltage", 4.0)]

    writes.clear()
    with pytest.raises(ChannelLimitsViolationError):
        proxy.write_attributes(name_val=[("current", 5.0), ("voltage", 99.0)])
    assert writes == []


def test_tango_write_attributes_asynch_takes_the_pairs_by_its_own_keyword(monkeypatch):
    """This spelling names the pairs ``attr_values``, and stock PyTango accepts it.

    One factory serves all three many-write spellings, so a guard that knew
    only ``name_val`` would break a correct call with a TypeError naming its
    own parameter rather than anything the caller wrote.
    """
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    assert proxy.write_attributes_asynch(attr_values=[("current", 5.0), ("voltage", 4.0)]) == 2
    assert writes == [("current", 5.0), ("voltage", 4.0)]

    writes.clear()
    with pytest.raises(ChannelLimitsViolationError):
        proxy.write_attributes_asynch(attr_values=[("current", 5.0), ("voltage", 99.0)])
    assert writes == []


def test_tango_write_read_attributes_takes_the_pairs_by_keyword(monkeypatch):
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    written = proxy.write_read_attributes(name_val=[("current", 5.0), ("voltage", 4.0)])
    assert written == "read-back-many"
    assert writes == [("current", 5.0), ("voltage", 4.0)]

    writes.clear()
    with pytest.raises(ChannelLimitsViolationError):
        proxy.write_read_attributes(name_val=[("current", 5.0), ("voltage", 99.0)])
    assert writes == []


def test_tango_many_write_without_pairs_fails_closed(monkeypatch):
    """A call carrying no pairs at all cannot be checked, so it does not write."""
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    with pytest.raises(ValueError, match="pairs"):
        proxy.write_attributes()
    assert writes == []


def test_tango_write_attribute_addresses_an_attribute_object_by_its_name(monkeypatch):
    """An attribute named by an object addresses the channel its name does.

    PyTango takes an ``AttributeInfo`` wherever a write takes an attribute
    name. Interpolated whole, it builds an address the limits database has
    never heard of --- which on a deployment that allows unlisted channels is a
    write that passes with no bound applied, so this asserts the refusal is
    ``MAX_EXCEEDED`` and not merely that something raised.
    """
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    attribute = mod.AttributeInfo("current")
    with pytest.raises(ChannelLimitsViolationError) as exc:
        proxy.write_attribute(attribute, 99.0)

    assert exc.value.violation_type == "MAX_EXCEEDED"
    assert writes == []

    # The object itself still reaches the device: it is reduced only to build
    # the address the value is bounded under.
    assert proxy.write_attribute(attribute, 5.0) == "written"
    assert writes == [(attribute, 5.0)]


def test_tango_write_attributes_addresses_an_attribute_object_by_its_name(monkeypatch):
    """The many-write shape reduces each pair's attribute the same way."""
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    attribute = mod.AttributeInfo("current")
    with pytest.raises(ChannelLimitsViolationError) as exc:
        proxy.write_attributes([(attribute, 99.0)])

    assert exc.value.violation_type == "MAX_EXCEEDED"
    assert writes == []

    assert proxy.write_attributes([(attribute, 5.0)]) == "written"
    assert writes == [(attribute, 5.0)]


def test_tango_command_inout_is_refused(monkeypatch):
    """A command is refused in a limits-checked run, argument or not.

    A Tango command names an operation, not a channel: there is no address the
    limits database is keyed by and no number to bound. Letting one through
    because "no value was found to check" would be a limits-checked run
    driving the device unchecked, so the call refuses outright — including
    with an argument that would pass every bound if it were a write.
    """
    commands: list = []
    mod = _install_fake_tango(monkeypatch, commands=commands)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    with pytest.raises(RuntimeError, match="a command carries no value to bound"):
        proxy.command_inout("On")
    with pytest.raises(RuntimeError, match="a command carries no value to bound"):
        proxy.command_inout("SetCurrent", 5.0)
    assert commands == []


def test_tango_command_inout_asynch_is_refused(monkeypatch):
    """The asynchronous spelling reaches the same device and refuses too."""
    commands: list = []
    mod = _install_fake_tango(monkeypatch, commands=commands)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    with pytest.raises(RuntimeError, match="a command carries no value to bound"):
        proxy.command_inout_asynch("On")
    with pytest.raises(RuntimeError, match="a command carries no value to bound"):
        proxy.command_inout_asynch("SetCurrent", 5.0, True)
    assert commands == []


def test_tango_attribute_writes_stay_checked_alongside_the_command_refusal(monkeypatch):
    """Refusing commands must not cost write_attribute its limits check.

    Both installs run against the same DeviceProxy class, so a mistake in
    either — an exception escaping the command loop, a name overwritten — is
    a write that stops being validated. This pins the two together.
    """
    writes: list = []
    commands: list = []
    mod = _install_fake_tango(monkeypatch, writes, commands)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    assert proxy.write_attribute("current", 5.0) == "written"
    with pytest.raises(ChannelLimitsViolationError):
        proxy.write_attribute("current", 99.0)
    with pytest.raises(RuntimeError, match="a command carries no value to bound"):
        proxy.command_inout("On")

    assert writes == [("current", 5.0)]
    assert commands == []


def test_tango_command_refusal_reaches_the_class_that_defines_it(monkeypatch):
    """The unbound call on the definer refuses too, not just the subclass.

    PyTango defines both command spellings on ``Connection``; ``DeviceProxy``
    inherits them. Patching only the subclass installs a shadow and leaves the
    original reachable as ``tango.Connection.command_inout(proxy, 'On')`` —
    an unbound call with the proxy as ``self``, which drives the real device
    with the guard reporting itself installed. Both spellings are pinned on
    the definer here, and the instance spelling with them.
    """
    commands: list = []
    mod = _install_fake_tango(monkeypatch, commands=commands)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    with pytest.raises(RuntimeError, match="a command carries no value to bound"):
        mod.Connection.command_inout(proxy, "On")
    with pytest.raises(RuntimeError, match="a command carries no value to bound"):
        mod.Connection.command_inout_asynch(proxy, "On")
    with pytest.raises(RuntimeError, match="a command carries no value to bound"):
        proxy.command_inout("On")

    assert commands == []


def test_tango_definer_refusal_leaves_write_attribute_checked(monkeypatch):
    """Patching the base class must not cost the subclass its write check."""
    writes: list = []
    commands: list = []
    mod = _install_fake_tango(monkeypatch, writes, commands)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    assert proxy.write_attribute("current", 5.0) == "written"
    with pytest.raises(ChannelLimitsViolationError):
        proxy.write_attribute("current", 99.0)
    with pytest.raises(RuntimeError, match="a command carries no value to bound"):
        mod.Connection.command_inout(proxy, "On")

    assert writes == [("current", 5.0)]
    assert commands == []


def test_tango_group_writes_and_commands_are_all_refused(monkeypatch):
    """A group write is a broadcast, so all four Group spellings refuse.

    ``tango.Group`` is its own class: it inherits neither from ``DeviceProxy``
    nor from ``Connection``, so the two installs above do not reach it, and one
    ``g.write_attribute('current', 5.0)`` fans that value out to every device
    the group matched. There is no single channel address to look limits up
    under and no one device to bound the step against, so the value being in
    range buys nothing — all four refuse, commands with the command message and
    writes with the group-write message.
    """
    mod = _install_fake_tango(monkeypatch)
    _run_monkeypatch(monkeypatch)

    group = mod.Group("all")
    group.add("sys/tg_test/*")
    with pytest.raises(RuntimeError, match="a command carries no value to bound"):
        group.command_inout("On")
    with pytest.raises(RuntimeError, match="a command carries no value to bound"):
        group.command_inout_asynch("On")
    with pytest.raises(RuntimeError, match="fans one value out to many devices"):
        group.write_attribute("current", 5.0)
    with pytest.raises(RuntimeError, match="fans one value out to many devices"):
        group.write_attribute_asynch("current", 5.0)

    assert mod.group_calls == []


def test_tango_group_refusal_leaves_device_writes_checked(monkeypatch):
    """Refusing the group must not cost a single-device write its check."""
    writes: list = []
    mod = _install_fake_tango(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    proxy = mod.DeviceProxy("sys/tg_test/1")
    group = mod.Group("all")
    assert proxy.write_attribute("current", 5.0) == "written"
    with pytest.raises(ChannelLimitsViolationError):
        proxy.write_attribute("current", 99.0)
    with pytest.raises(RuntimeError, match="fans one value out to many devices"):
        group.write_attribute("current", 5.0)

    assert writes == [("current", 5.0)]
    assert mod.group_calls == []


def test_tango_success_line_names_the_commands_it_refused(monkeypatch):
    """The block reports what it actually installed on this binding."""
    _install_fake_tango(monkeypatch)

    out = _run_monkeypatch(monkeypatch)

    assert "✅ Monkeypatched tango: " in out
    assert (
        "wrapped on DeviceProxy: write_attribute(), write_attribute_asynch(), "
        "write_read_attribute(), write_attributes(), write_attributes_asynch(), "
        "write_read_attributes()" in out
    )
    assert "wrapped on AttributeProxy: write(), write_asynch(), write_read()" in out
    assert "refused on DeviceProxy: command_inout(), command_inout_asynch()" in out
    assert "refused on Connection: command_inout(), command_inout_asynch()" in out
    assert (
        "refused on Group: command_inout(), command_inout_asynch(), "
        "write_attribute(), write_attribute_asynch()" in out
    )


def test_tango_without_a_device_proxy_does_not_report_itself_guarded(monkeypatch):
    """A tango carrying no DeviceProxy wrapped nothing and must say so."""
    mod = _fake_module("tango")
    monkeypatch.setitem(sys.modules, "tango", mod)

    out = _run_monkeypatch(monkeypatch)

    assert "⚠️  tango guard failed: no DeviceProxy class" in out
    assert "✅ Monkeypatched tango" not in out


def test_tango_group_is_refused_without_a_device_proxy(monkeypatch):
    """The Group install does not hang off the DeviceProxy branch.

    A tango carrying a ``Group`` but no ``DeviceProxy`` still has four live
    write spellings; installed under the DeviceProxy branch, the guard would
    warn about the missing class and leave every one of them writing. The
    success line names only the class that was actually patched.
    """
    real = _install_fake_tango(monkeypatch)
    mod = _fake_module("tango")
    mod.Group = real.Group
    mod.group_calls = real.group_calls
    monkeypatch.setitem(sys.modules, "tango", mod)

    out = _run_monkeypatch(monkeypatch)

    assert "⚠️  tango guard failed: no DeviceProxy class" in out
    success = [line for line in out.splitlines() if line.startswith("✅ Monkeypatched tango")]
    assert success == [
        "✅ Monkeypatched tango: refused on Group: command_inout(), "
        "command_inout_asynch(), write_attribute(), write_attribute_asynch()"
    ]

    group = mod.Group("all")
    with pytest.raises(RuntimeError, match="fans one value out to many devices"):
        group.write_attribute("current", 5.0)
    with pytest.raises(RuntimeError, match="a command carries no value to bound"):
        group.command_inout("On")
    assert mod.group_calls == []


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
    sync_client, _, _ = _install_fake_caproto(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    assert sync_client.write("TEST:MAG:SP", 5.0) == "written"
    assert writes == [("TEST:MAG:SP", 5.0)]

    writes.clear()
    with pytest.raises(ChannelLimitsViolationError):
        sync_client.write("TEST:MAG:SP", 99.0)
    assert writes == []


def test_caproto_threading_pv_write_is_limits_checked(monkeypatch):
    writes: list = []
    _, threading_client, _ = _install_fake_caproto(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    pv = threading_client.PV("TEST:MAG:SP")
    assert pv.write(5.0) == "written"
    assert writes == [("TEST:MAG:SP", 5.0)]

    writes.clear()
    with pytest.raises(ChannelLimitsViolationError):
        pv.write(99.0)
    assert writes == []
    assert pv.read() == 1.0, "reads must survive the guard untouched"


def test_caproto_sync_read_write_read_refuses_an_out_of_range_value(monkeypatch):
    """The write-and-read-back spelling is bounded like the plain write.

    It drives the channel with the same value, so a value the plain write is
    refused for must not reach the machine through this one either.
    """
    writes: list = []
    sync_client, _, _ = _install_fake_caproto(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    with pytest.raises(ChannelLimitsViolationError) as exc:
        sync_client.read_write_read("TEST:MAG:SP", 99.0)

    assert exc.value.violation_type == "MAX_EXCEEDED"
    assert writes == []


def test_caproto_sync_read_write_read_passes_an_in_range_value_through(monkeypatch):
    writes: list = []
    sync_client, _, _ = _install_fake_caproto(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    assert sync_client.read_write_read("TEST:MAG:SP", 5.0) == "read-write-read"
    assert writes == [("TEST:MAG:SP", 5.0)]

    # caproto names the pair, so the keyword call has to be checked too.
    assert sync_client.read_write_read(pv_name="TEST:MAG:SP", data=6.0) == "read-write-read"
    assert writes == [("TEST:MAG:SP", 5.0), ("TEST:MAG:SP", 6.0)]


def test_caproto_batch_write_refuses_an_out_of_range_value(monkeypatch):
    """A batched write is bounded under the limits of the PV it carries."""
    writes: list = []
    _, threading_client, _ = _install_fake_caproto(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    pv = threading_client.PV("TEST:MAG:SP")
    with pytest.raises(ChannelLimitsViolationError) as exc:
        threading_client.Batch().write(pv, 99.0)

    assert exc.value.violation_type == "MAX_EXCEEDED"
    assert writes == []


def test_caproto_batch_write_passes_an_in_range_value_through(monkeypatch):
    writes: list = []
    _, threading_client, _ = _install_fake_caproto(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    pv = threading_client.PV("TEST:MAG:SP")
    assert threading_client.Batch().write(pv, 5.0) == "batched"
    assert writes == [("TEST:MAG:SP", 5.0)]

    # caproto names the pair, so the keyword call has to be checked too.
    assert threading_client.Batch().write(pv=pv, data=6.0) == "batched"
    assert writes == [("TEST:MAG:SP", 5.0), ("TEST:MAG:SP", 6.0)]


def test_caproto_asyncio_pv_write_refuses_an_out_of_range_value(monkeypatch):
    """The asyncio client is a third module the other two guards never reach."""
    writes: list = []
    _, _, asyncio_client = _install_fake_caproto(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    pv = asyncio_client.PV("TEST:MAG:SP")
    with pytest.raises(ChannelLimitsViolationError) as exc:
        asyncio.run(pv.write(99.0))

    assert exc.value.violation_type == "MAX_EXCEEDED"
    assert writes == []


def test_caproto_asyncio_pv_write_passes_an_in_range_value_through(monkeypatch):
    writes: list = []
    _, _, asyncio_client = _install_fake_caproto(monkeypatch, writes)
    _run_monkeypatch(monkeypatch)

    pv = asyncio_client.PV("TEST:MAG:SP")
    assert asyncio.run(pv.write(5.0)) == "written"
    assert writes == [("TEST:MAG:SP", 5.0)]


def test_caproto_asyncio_pv_write_reads_only_for_a_max_step_channel(monkeypatch):
    """The pre-read is bought by max_step alone, and awaited over the PV itself.

    The guard is a coroutine, so it awaits caproto's own read and hands the
    validator a plain value. A channel without a max_step never pays for that
    round trip.
    """
    writes: list = []
    reads: list = []
    _, _, asyncio_client = _install_fake_caproto(monkeypatch, writes, reads)
    _run_monkeypatch(monkeypatch)

    assert asyncio.run(asyncio_client.PV("TEST:MAG:SP").write(5.0)) == "written"
    assert reads == []

    step_pv = asyncio_client.PV("TEST:MAG:STEP")
    with pytest.raises(ChannelLimitsViolationError) as exc:
        asyncio.run(step_pv.write(50.0))

    assert exc.value.violation_type == "MAX_STEP_EXCEEDED"
    assert reads == ["TEST:MAG:STEP"]
    assert writes == [("TEST:MAG:SP", 5.0)]

    # The fake reads back 1.0, so this is a step of 1.0 against a max of 2.0.
    assert asyncio.run(step_pv.write(2.0)) == "written"
    assert reads == ["TEST:MAG:STEP", "TEST:MAG:STEP"]
    assert writes == [("TEST:MAG:SP", 5.0), ("TEST:MAG:STEP", 2.0)]


def test_caproto_pre_reads_carry_the_step_read_ceiling(monkeypatch):
    """Every max_step pre-read bounds itself, on all four write surfaces.

    caproto lets a client be built with no timeout of its own, and a pre-read
    inheriting that would hold an operator's write open on a channel that
    never answers. The ceiling is the guard's, not the client's.
    """
    writes: list = []
    reads: list = []
    read_kwargs: list = []
    sync_client, threading_client, asyncio_client = _install_fake_caproto(
        monkeypatch, writes, reads, read_kwargs
    )
    _run_monkeypatch(monkeypatch)

    # The fake reads back 1.0, so each of these is a step of 1.0 against a
    # max of 2.0 — in range, so the write goes through and the read is real.
    pv = threading_client.PV("TEST:MAG:STEP")
    sync_client.write("TEST:MAG:STEP", 2.0)
    pv.write(2.0)
    threading_client.Batch().write(pv, 2.0)
    asyncio.run(asyncio_client.PV("TEST:MAG:STEP").write(2.0))

    assert reads == ["TEST:MAG:STEP"] * 4
    assert read_kwargs == [{"timeout": DEFAULT_STEP_READ_TIMEOUT_SECONDS}] * 4
    assert writes == [("TEST:MAG:STEP", 2.0)] * 4


def test_caproto_report_lines_name_every_wrapped_spelling(monkeypatch):
    """The success lines are the operator's only evidence the guard is on."""
    _install_fake_caproto(monkeypatch)
    out = _run_monkeypatch(monkeypatch)

    success = [line for line in out.splitlines() if line.startswith("✅ Monkeypatched caproto")]
    assert success == [
        "✅ Monkeypatched caproto.sync.client: write(), read_write_read()",
        "✅ Monkeypatched caproto.threading.client: PV.write(), Batch.write()",
        "✅ Monkeypatched caproto.asyncio.client PV.write()",
    ]


def test_caproto_asyncio_absence_leaves_the_other_flavors_guarded(monkeypatch):
    """One flavor missing must not take the guards for the other two with it.

    Each flavor is imported and patched in its own try/except; collapsed into
    one, an environment without the asyncio client would lose the sync and
    threading guards and say nothing about it.
    """
    writes: list = []
    sync_client, threading_client, _ = _install_fake_caproto(monkeypatch, writes)
    monkeypatch.setitem(sys.modules, "caproto.asyncio.client", None)

    out = _run_monkeypatch(monkeypatch)

    assert "ℹ️  caproto.asyncio.client not available" in out
    success = [line for line in out.splitlines() if line.startswith("✅ Monkeypatched caproto")]
    assert success == [
        "✅ Monkeypatched caproto.sync.client: write(), read_write_read()",
        "✅ Monkeypatched caproto.threading.client: PV.write(), Batch.write()",
    ]

    # The two remaining flavors are guarded, not merely reported.
    with pytest.raises(ChannelLimitsViolationError):
        sync_client.read_write_read("TEST:MAG:SP", 99.0)
    with pytest.raises(ChannelLimitsViolationError):
        threading_client.Batch().write(threading_client.PV("TEST:MAG:SP"), 99.0)
    assert writes == []


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
        "caproto.asyncio",
        "caproto.asyncio.client",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    writes: list = []
    _install_fake_doocs(monkeypatch, writes)

    out = _run_monkeypatch(monkeypatch)
    assert "tango not available" in out
    assert "✅ Monkeypatched doocs4py.set()" in out


# ---------------------------------------------------------------------------
# The limits buckets, row by row
# ---------------------------------------------------------------------------


_WRAPPED_ROWS = sorted(_LIMITS_WRAPPED)
_REFUSED_ROWS = sorted(_LIMITS_REFUSED)


def _checked_through(dotted, attr, reason):
    """The name a wrapped row's limits check is actually installed on.

    A ``"direct"`` row is checked on its own name. A ``"via X"`` row is
    checked on ``X``, written as one dotted name whose last segment is the
    attribute: pyepics spells a write three ways and all three reach the
    network through ``epics.ca.put``, so that one name is what the guard
    rebinds and the only one an assertion can look at.
    """
    if reason.startswith("via "):
        through = reason.split()[1]
        assert "." in through, (
            f"_LIMITS_WRAPPED[{dotted}.{attr}] says {reason!r}; the word after "
            f'"via" has to be the dotted name the check is installed on, and '
            f"{through!r} carries no attribute to look at"
        )
        owner, _, through_attr = through.rpartition(".")
        return owner, through_attr
    return dotted, attr


#: How to call one refused row: a receiver of its own class, arguments the
#: unrefused spelling accepts, and the words the refusal answers with. Keyed by
#: class, because every refused attribute on a class refuses the same way.
_REFUSAL_CALLS = {
    "p4p.client.raw.Context": (lambda cls: cls(), ("TEST:MAG:SP", None), "cannot be approved"),
    "p4p.client.thread.Context": (lambda cls: cls(), ("TEST:MAG:SP", None), "cannot be approved"),
    "p4p.client.asyncio.Context": (lambda cls: cls(), ("TEST:MAG:SP", None), "cannot be approved"),
    "p4p.client.cothread.Context": (
        lambda cls: cls(),
        ("TEST:MAG:SP", None),
        "cannot be approved",
    ),
    "pvaccess.Channel": (
        lambda cls: cls("TEST:MAG:SP"),
        (['{"value": 1.0}'],),
        "cannot be limits-checked",
    ),
    "tango.DeviceProxy": (
        lambda cls: cls("sys/tg_test/1"),
        ("On",),
        "refused in a limits-checked run",
    ),
    "tango.Connection": (lambda cls: cls(), ("On",), "refused in a limits-checked run"),
    "tango.Group": (
        lambda cls: cls("all"),
        ("current", 1.0),
        "refused in a limits-checked run",
    ),
}


@pytest.mark.parametrize(
    ("dotted", "attr"),
    _WRAPPED_ROWS,
    ids=[f"{dotted}.{attr}" for dotted, attr in _WRAPPED_ROWS],
)
def test_every_wrapped_row_is_rebound_by_the_block(monkeypatch, dotted, attr):
    """Every wrapped row names a Python name the block really re-points.

    A limits check reaches a write only by replacing the object the client
    library holds. The behaviour tests above prove a guard is somewhere on the
    path for the spellings they drive; this one proves the rebinding itself,
    mechanically and for the whole bucket, so a row added to the table with no
    matching install cannot claim a check it never gets.
    """
    reason = _LIMITS_WRAPPED[(dotted, attr)]
    assert reason.startswith(("direct", "via ")), (
        f"_LIMITS_WRAPPED[{dotted}.{attr}] says {reason!r}; a row says either "
        '"direct" or "via <dotted.attribute>", which is what names the object '
        "this assertion watches"
    )
    owner_dotted, owner_attr = _checked_through(dotted, attr, reason)

    _install_client_fakes(monkeypatch)
    owner = _resolve_target(owner_dotted)
    assert owner is not None, (
        f"{owner_dotted} has no stand-in in this suite, so {dotted}.{attr} "
        "would be asserted against nothing"
    )
    assert hasattr(owner, owner_attr), (
        f"the {owner_dotted} stand-in carries no {owner_attr}, so there is "
        "nothing here for the block to rebind"
    )
    before = getattr(owner, owner_attr)

    _run_monkeypatch(monkeypatch)

    assert getattr(owner, owner_attr) is not before, (
        f"{dotted}.{attr} is in _LIMITS_WRAPPED, but the block left "
        f"{owner_dotted}.{owner_attr} pointing at the client's own function"
    )


@pytest.mark.parametrize(
    ("dotted", "attr"),
    _REFUSED_ROWS,
    ids=[f"{dotted}.{attr}" for dotted, attr in _REFUSED_ROWS],
)
def test_every_refused_row_refuses_its_call(monkeypatch, dotted, attr):
    """Every refused row raises instead of reaching the client underneath it.

    The same call is made once before the block runs. A stand-in that raised
    on its own — a signature the arguments do not fit — would otherwise look
    exactly like a refusal.
    """
    assert dotted in _REFUSAL_CALLS, (
        f"{dotted} is in _LIMITS_REFUSED but _REFUSAL_CALLS has no entry for "
        "it: add a receiver factory for the class, the arguments its "
        "unrefused spelling accepts, and the words its refusal answers with"
    )
    receiver_of, args, refusal = _REFUSAL_CALLS[dotted]

    _install_client_fakes(monkeypatch)
    cls = _resolve_target(dotted)
    assert cls is not None, f"{dotted} has no stand-in in this suite"
    getattr(receiver_of(cls), attr)(*args)

    _run_monkeypatch(monkeypatch)

    with pytest.raises((RuntimeError, ValueError)) as refused:
        getattr(receiver_of(cls), attr)(*args)
    assert refusal in str(refused.value), (
        f"{dotted}.{attr} raised {refused.value!r}, which does not read as "
        "the refusal its bucket entry promises"
    )
