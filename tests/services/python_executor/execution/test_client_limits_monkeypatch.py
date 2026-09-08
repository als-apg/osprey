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
    ChannelLimitsConfig,
    LimitsValidator,
)
from osprey.errors import ChannelLimitsViolationError
from osprey.services.python_executor.execution.wrapper import ExecutionWrapper
from osprey.services.python_executor.write_surface import _CLIENT_WRITE_TARGETS

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


def _install_fake_tango(monkeypatch, writes=None, commands=None, group_calls=None):
    """Inject a ``tango`` whose DeviceProxy and Group record what reaches them.

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
    mod = _fake_module("tango")

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

        def write_attributes(self, name_val):
            writes.extend(list(name_val))
            return "written"

        def read_attribute(self, attr):
            return 1.0

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


def _install_fake_caproto(monkeypatch, writes=None):
    """Inject a fake ``caproto`` with both write entry points."""
    writes = [] if writes is None else writes
    caproto_mod = _fake_module("caproto")
    sync_mod = _fake_module("caproto.sync")
    sync_client = _fake_module("caproto.sync.client")
    threading_mod = _fake_module("caproto.threading")
    threading_client = _fake_module("caproto.threading.client")

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


def _run_monkeypatch(monkeypatch):
    """Execute the generated monkeypatch block against fakes; return its stdout."""
    for name, installer in _CLIENT_FAKES:
        if _needs_fake(name):
            installer(monkeypatch)

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
    assert "wrapped on DeviceProxy: write_attribute(), write_attributes()" in out
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
