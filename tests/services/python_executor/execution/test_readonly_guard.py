"""Tests for the readonly-run guard in the generated execution wrapper.

A script submitted with ``execution_mode="readonly"`` must be unable to write
to the control system *at runtime*, however the write is spelled. The
pre-execution regex only sees the standard spellings, so the wrapper installs
refusing replacements for every direct-library write entry point before the
user code runs. Late binding makes this spelling-independent: an alias such as
``from epics import caput as _w`` resolves to the refusing function because the
patch is already in place when the alias is bound.

Like the limits monkeypatch tests, most of these execute the generated *source
text* against fake ``epics``/``p4p`` modules injected into ``sys.modules``. A
fake proves the guard patches the name the table spells; it cannot prove the
guard reached the object a real library defines that write on, so the last test
in the file runs the guard in a subprocess against whatever is installed.
"""

import asyncio
import importlib
import json
import platform
import re
import subprocess
import sys
from types import ModuleType

import pytest

from osprey.services.python_executor.execution import wrapper as wrapper_module
from osprey.services.python_executor.execution.wrapper import (
    _READONLY_WRITE_TARGETS,
    READONLY_REFUSAL,
    READONLY_REFUSAL_MARKER,
    ExecutionWrapper,
)

pytestmark = pytest.mark.unit

# The refusal text contains literal parentheses; escape it for pytest.raises.
_REFUSAL = re.escape(READONLY_REFUSAL)


def _resolve_target(dotted):
    """Mirror of the resolver the guard emits, for the restore fixture below.

    Deliberately a copy rather than a shared import: the emitted guard has to
    be self-contained — it runs before the user code in a subprocess that may
    not be able to import OSPREY at all — so there is no function to share.
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
    """Undo the guard's patches after every test in this module.

    These tests exec the guard's source *in the test process*, which is the
    only practical way to assert on its behaviour. The guard patches objects in
    ``sys.modules``, and some of those — ``os``, ``subprocess``, ``ctypes`` —
    are the same objects pytest and the rest of the suite use. Without this,
    the first test to run leaves ``subprocess.run`` refusing for the remainder
    of the session, and unrelated tests fail with a readonly refusal.

    Snapshotting happens before the fake ``epics``/``p4p`` modules are injected,
    so it captures the real objects; restoring writes back to those same objects
    and is therefore unaffected by whatever ``sys.modules`` held in between.
    """
    saved = []
    for dotted, attrs in _READONLY_WRITE_TARGETS:
        target = _resolve_target(dotted)
        if target is None:
            continue
        for attr in attrs:
            if not hasattr(target, attr):
                continue
            original = getattr(target, attr)
            saved.append((target, attr, original))
            # The guard patches the second spelling of a re-exported write as
            # well — the module that defined it, reached by following the
            # original back through ``__module__``/``__name__`` — so snapshot
            # it on the same rule the guard patches it on, or it stays
            # refusing for the rest of the session.
            home = sys.modules.get(getattr(original, "__module__", ""))
            name = getattr(original, "__name__", None)
            if home is not None and isinstance(name, str):
                if getattr(home, name, None) is original:
                    saved.append((home, name, original))
    yield
    for target, attr, value in saved:
        setattr(target, attr, value)


class _RecordingContext:
    def __init__(self, *args, **kwargs):
        self.puts = []
        self.rpcs = []

    def put(self, name, values, request=None, timeout=5.0, **kwargs):
        self.puts.append((name, values))
        return "put-done"

    def rpc(self, name, value=None, request=None, timeout=5.0):
        self.rpcs.append((name, value))
        return "rpc-done"


class _AsyncRecordingContext(_RecordingContext):
    async def put(self, name, values, request=None, timeout=5.0, **kwargs):
        self.puts.append((name, values))
        return "put-done"


def _install_fake_epics(monkeypatch):
    mod = ModuleType("epics")
    ca = ModuleType("epics.ca")
    writes: list = []

    def ca_put(chid, value, **kwargs):
        writes.append(("ca.put", chid, value))
        return 1

    def caput(pvname, value, wait=False, timeout=60, **kwargs):
        writes.append(("caput", pvname, value))
        return 1

    class PV:
        def __init__(self, pvname):
            self.pvname = pvname

        def put(self, value, wait=False, timeout=60, **kwargs):
            writes.append(("PV.put", self.pvname, value))
            return 1

    ca.put = ca_put
    mod.ca = ca
    mod.caput = caput
    mod.PV = PV
    mod._writes = writes
    monkeypatch.setitem(sys.modules, "epics", mod)
    monkeypatch.setitem(sys.modules, "epics.ca", ca)
    return mod


def _install_fake_p4p(monkeypatch):
    p4p_mod = ModuleType("p4p")
    client_mod = ModuleType("p4p.client")
    p4p_mod.client = client_mod
    monkeypatch.setitem(sys.modules, "p4p", p4p_mod)
    monkeypatch.setitem(sys.modules, "p4p.client", client_mod)
    classes = {}
    for flavor, base in (("thread", _RecordingContext), ("asyncio", _AsyncRecordingContext)):
        flavor_mod = ModuleType(f"p4p.client.{flavor}")
        ctx_cls = type(f"{flavor.capitalize()}Context", (base,), {})
        flavor_mod.Context = ctx_cls
        classes[flavor] = ctx_cls
        setattr(client_mod, flavor, flavor_mod)
        monkeypatch.setitem(sys.modules, f"p4p.client.{flavor}", flavor_mod)
    monkeypatch.setitem(sys.modules, "p4p.client.cothread", None)
    return classes


def _run_guard(execution_mode):
    source = ExecutionWrapper(execution_mode=execution_mode)._get_readonly_guard()
    namespace: dict = {}
    exec(source, namespace)
    return source


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def test_readwrite_emits_no_guard():
    assert ExecutionWrapper(execution_mode="readwrite")._get_readonly_guard() == ""


def test_default_mode_is_readonly():
    """Fail closed: a wrapper built without a mode guards like a readonly run."""
    assert ExecutionWrapper()._get_readonly_guard() != ""


def test_guard_precedes_user_code_in_full_wrapper():
    wrapped = ExecutionWrapper(execution_mode="readonly").create_wrapper("print('hi')")
    assert wrapped.index(READONLY_REFUSAL) < wrapped.index("print('hi')")


def test_guard_is_independent_of_limits_validator():
    """The guard must exist with limits checking off — that is the case it protects."""
    wrapper = ExecutionWrapper(limits_validator=None, execution_mode="readonly")
    assert wrapper._get_limits_checking_monkeypatch() == ""
    assert READONLY_REFUSAL in wrapper._get_readonly_guard()


# ---------------------------------------------------------------------------
# pyepics
# ---------------------------------------------------------------------------


def test_readonly_refuses_epics_caput(monkeypatch):
    epics = _install_fake_epics(monkeypatch)
    _run_guard("readonly")
    with pytest.raises(RuntimeError, match=_REFUSAL):
        epics.caput("SR:MAG:QF:01:CURRENT:SP", 150)
    assert epics._writes == []


def test_readonly_refuses_aliased_caput(monkeypatch):
    """The regex-evading spelling lands on the same refusing function."""
    _install_fake_epics(monkeypatch)
    _run_guard("readonly")
    from epics import caput as _w  # bound AFTER the guard, as in a real run

    with pytest.raises(RuntimeError, match=_REFUSAL):
        _w("SR:MAG:QF:01:CURRENT:SP", 150)


def test_readonly_refuses_getattr_caput(monkeypatch):
    epics = _install_fake_epics(monkeypatch)
    _run_guard("readonly")
    with pytest.raises(RuntimeError, match=_REFUSAL):
        getattr(epics, "ca" + "put")("SR:MAG:QF:01:CURRENT:SP", 150)


def test_readonly_refuses_pv_put(monkeypatch):
    epics = _install_fake_epics(monkeypatch)
    _run_guard("readonly")
    with pytest.raises(RuntimeError, match=_REFUSAL):
        epics.PV("SR:MAG:QF:01:CURRENT:SP").put(150)
    assert epics._writes == []


def test_readonly_refuses_low_level_ca_put(monkeypatch):
    """``epics.ca.put`` is what caput/PV.put bottom out in; it is refused too."""
    epics = _install_fake_epics(monkeypatch)
    _run_guard("readonly")
    with pytest.raises(RuntimeError, match=_REFUSAL):
        epics.ca.put(12345, 150)


def test_readwrite_leaves_epics_untouched(monkeypatch):
    epics = _install_fake_epics(monkeypatch)
    _run_guard("readwrite")
    assert epics.caput("SR:MAG:QF:01:CURRENT:SP", 150) == 1
    assert epics._writes == [("caput", "SR:MAG:QF:01:CURRENT:SP", 150)]


def test_readonly_without_epics_installed_is_quiet(monkeypatch):
    monkeypatch.setitem(sys.modules, "epics", None)
    monkeypatch.setitem(sys.modules, "epics.ca", None)
    _install_fake_p4p(monkeypatch)
    _run_guard("readonly")  # must not raise


# ---------------------------------------------------------------------------
# p4p
# ---------------------------------------------------------------------------


def test_readonly_refuses_p4p_thread_put_and_rpc(monkeypatch):
    _install_fake_epics(monkeypatch)
    classes = _install_fake_p4p(monkeypatch)
    _run_guard("readonly")
    ctxt = classes["thread"]("pva")
    with pytest.raises(RuntimeError, match=_REFUSAL):
        ctxt.put("SR:MAG:QF:01:CURRENT:SP", 150)
    with pytest.raises(RuntimeError, match=_REFUSAL):
        ctxt.rpc("SR:SVC:ORBIT", {})
    assert ctxt.puts == [] and ctxt.rpcs == []


def test_readonly_refuses_p4p_asyncio_put(monkeypatch):
    _install_fake_epics(monkeypatch)
    classes = _install_fake_p4p(monkeypatch)
    _run_guard("readonly")
    ctxt = classes["asyncio"]("pva")
    with pytest.raises(RuntimeError, match=_REFUSAL):
        result = ctxt.put("SR:MAG:QF:01:CURRENT:SP", 150)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    assert ctxt.puts == []


def test_readonly_without_p4p_installed_is_quiet(monkeypatch):
    _install_fake_epics(monkeypatch)
    for name in (
        "p4p",
        "p4p.client",
        "p4p.client.thread",
        "p4p.client.asyncio",
        "p4p.client.cothread",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    _run_guard("readonly")  # must not raise


# ---------------------------------------------------------------------------
# caproto, pvaPy, Tango
#
# These three had a static import denial and nothing behind it, which
# ``importlib.import_module("caproto.sync.client")`` walked straight past —
# the AST check sees no ``ast.Import`` node to match. Patching the resolved
# object closes that, exactly as it already did for pyepics.
# ---------------------------------------------------------------------------


def _install_fake_caproto(monkeypatch):
    caproto_mod = ModuleType("caproto")
    sync_mod = ModuleType("caproto.sync")
    sync_client = ModuleType("caproto.sync.client")
    threading_mod = ModuleType("caproto.threading")
    threading_client = ModuleType("caproto.threading.client")
    writes: list = []

    def write(pv_name, data, **kwargs):
        writes.append((pv_name, data))
        return 1

    class PV:
        def __init__(self, name):
            self.name = name

        def write(self, data, **kwargs):
            writes.append((self.name, data))
            return 1

    sync_client.write = write
    threading_client.PV = PV
    caproto_mod.sync = sync_mod
    caproto_mod.threading = threading_mod
    sync_mod.client = sync_client
    threading_mod.client = threading_client
    caproto_mod._writes = writes
    for name, mod in (
        ("caproto", caproto_mod),
        ("caproto.sync", sync_mod),
        ("caproto.sync.client", sync_client),
        ("caproto.threading", threading_mod),
        ("caproto.threading.client", threading_client),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return caproto_mod


def test_readonly_refuses_caproto_sync_write(monkeypatch):
    caproto = _install_fake_caproto(monkeypatch)
    _run_guard("readonly")
    with pytest.raises(RuntimeError, match=_REFUSAL):
        caproto.sync.client.write("SR:MAG:QF:01:CURRENT:SP", 150)
    assert caproto._writes == []


def test_readonly_refuses_caproto_via_importlib(monkeypatch):
    """The spelling the AST import denylist cannot see."""
    caproto = _install_fake_caproto(monkeypatch)
    _run_guard("readonly")
    client = importlib.import_module("caproto.sync.client")
    with pytest.raises(RuntimeError, match=_REFUSAL):
        client.write("SR:MAG:QF:01:CURRENT:SP", 150)
    assert caproto._writes == []


def test_readonly_refuses_caproto_threading_pv_write(monkeypatch):
    caproto = _install_fake_caproto(monkeypatch)
    _run_guard("readonly")
    pv = caproto.threading.client.PV("SR:MAG:QF:01:CURRENT:SP")
    with pytest.raises(RuntimeError, match=_REFUSAL):
        pv.write(150)
    assert caproto._writes == []


def test_readonly_refuses_pvaccess_typed_setters(monkeypatch):
    """pvaPy spells one setter per type, plus three writes off the prefix.

    ``put``/``putDouble``/… are swept wholesale, and ``asyncPut``,
    ``parsePut`` and ``parsePutGet`` are swept with them: they are writes
    whose names simply do not begin with ``put``.
    """
    mod = ModuleType("pvaccess")
    writes: list = []

    class Channel:
        def __init__(self, name):
            self.name = name

        def put(self, value):
            writes.append(("put", value))

        def putDouble(self, value):  # noqa: N802 — pvaPy's own spelling
            writes.append(("putDouble", value))

        def asyncPut(self, value, callback=None):  # noqa: N802 — pvaPy's own spelling
            writes.append(("asyncPut", value))

        def parsePut(self, args):  # noqa: N802 — pvaPy's own spelling
            writes.append(("parsePut", args))

        def get(self):
            return 1.0

    mod.Channel = Channel
    monkeypatch.setitem(sys.modules, "pvaccess", mod)
    _run_guard("readonly")

    channel = mod.Channel("SR:MAG:QF:01:CURRENT:SP")
    with pytest.raises(RuntimeError, match=_REFUSAL):
        channel.put(150)
    with pytest.raises(RuntimeError, match=_REFUSAL):
        channel.putDouble(150.0)
    # The three writes pvaPy spells outside the ``put`` prefix are the same
    # write to the machine, and are refused with it.
    with pytest.raises(RuntimeError, match=_REFUSAL):
        channel.asyncPut(150.0, lambda _r: None)
    with pytest.raises(RuntimeError, match=_REFUSAL):
        channel.parsePut(["value=150.0"])
    assert writes == []
    assert channel.get() == 1.0, "reads must survive the guard untouched"


def test_readonly_refuses_tango_write_attribute(monkeypatch):
    mod = ModuleType("tango")
    writes: list = []

    class DeviceProxy:
        def __init__(self, name):
            self.name = name

        def write_attribute(self, attr, value):
            writes.append((attr, value))

        def command_inout(self, command, arg=None):
            writes.append((command, arg))

        def read_attribute(self, attr):
            return 1.0

    mod.DeviceProxy = DeviceProxy
    monkeypatch.setitem(sys.modules, "tango", mod)
    _run_guard("readonly")

    proxy = mod.DeviceProxy("sys/tg_test/1")
    with pytest.raises(RuntimeError, match=_REFUSAL):
        proxy.write_attribute("current", 150)
    with pytest.raises(RuntimeError, match=_REFUSAL):
        proxy.command_inout("TurnOn")
    assert writes == []
    assert proxy.read_attribute("current") == 1.0, "reads must survive the guard untouched"


def test_readonly_refuses_doocs4py_set(monkeypatch):
    """The DOOCS connector's own client is guarded like every other client.

    A DOOCS deployment is guaranteed to have ``doocs4py`` importable, so a
    readonly script naming it is not a hypothetical route to the machine — it
    is the route the shipped connector itself writes through.
    """
    mod = ModuleType("doocs4py")
    writes: list = []

    def _set(address, value):
        writes.append((address, value))

    def _get(address):
        return 1.0

    mod.set = _set
    mod.get = _get
    monkeypatch.setitem(sys.modules, "doocs4py", mod)
    _run_guard("readonly")

    with pytest.raises(RuntimeError, match=_REFUSAL):
        mod.set("FACILITY/MAGNET/H1/CURRENT.SP", 150)
    assert writes == []
    assert mod.get("FACILITY/MAGNET/H1/CURRENT.RBV") == 1.0, (
        "reads must survive the guard untouched"
    )


def test_readonly_refuses_aioca_caput(monkeypatch):
    """aioca is in every OSPREY environment via ``ophyd-async[ca]``.

    Unlike pyepics it needs no facility-specific install, so it is the Channel
    Access client a readonly script is most likely to actually find.
    """
    mod = ModuleType("aioca")
    writes: list = []

    async def caput(pv, value, **kwargs):
        writes.append((pv, value))

    async def caget(pv, **kwargs):
        return 1.0

    mod.caput = caput
    mod.caget = caget
    monkeypatch.setitem(sys.modules, "aioca", mod)
    _run_guard("readonly")

    with pytest.raises(RuntimeError, match=_REFUSAL):
        mod.caput("SR:MAG:QF:01:CURRENT:SP", 150)
    assert writes == []
    assert asyncio.run(mod.caget("SR:MAG:QF:01:CURRENT")) == 1.0, (
        "reads must survive the guard untouched"
    )


def test_every_client_package_is_also_denied_at_import():
    """The runtime guard and the static denylist name the same libraries.

    They are two halves of one gate — the denylist stops the import, the guard
    stops the call — and a client present in only one of them is a client a
    readonly script can still reach. Deriving both from one table is what makes
    that impossible; this pins the derivation.
    """
    from osprey.services.python_executor.analysis.safety_checks import (
        _READONLY_DENIED_IMPORTS,
    )
    from osprey.services.python_executor.write_surface import _CLIENT_WRITE_TARGETS

    packages = {dotted.split(".")[0] for dotted, _attrs in _CLIENT_WRITE_TARGETS}
    assert packages <= set(_READONLY_DENIED_IMPORTS)
    assert {"doocs4py", "aioca"} <= packages


def test_framework_write_targets_stay_importable():
    """Acquisition frameworks refuse their writes but are not import-denied.

    ophyd-async and Bluesky are document and analysis libraries as much as they
    are hardware drivers; denying the import would refuse a readonly script
    that only reads a catalog. Their write entry points are in the runtime
    guard instead.
    """
    from osprey.services.python_executor.analysis.safety_checks import (
        _READONLY_DENIED_IMPORTS,
    )
    from osprey.services.python_executor.write_surface import _FRAMEWORK_WRITE_TARGETS

    packages = {dotted.split(".")[0] for dotted, _attrs in _FRAMEWORK_WRITE_TARGETS}
    assert packages == {"ophyd_async", "bluesky"}
    assert not packages & set(_READONLY_DENIED_IMPORTS)


# ---------------------------------------------------------------------------
# Routes out of Python
#
# The issue names two explicitly: a shelled-out ``caput`` and ``ctypes``.
# Neither touches a control-system client package, so no import-time or
# call-site check can see them; both are refused at the point of use.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda: __import__("subprocess").run(["caput", "PV", "1"]), id="subprocess-run"
        ),
        pytest.param(
            lambda: __import__("subprocess").Popen(["caput", "PV", "1"]), id="subprocess-Popen"
        ),
        pytest.param(
            lambda: __import__("subprocess").check_output(["caput", "PV", "1"]),
            id="subprocess-check_output",
        ),
        pytest.param(lambda: __import__("os").system("caput PV 1"), id="os-system"),
        pytest.param(lambda: __import__("os").popen("caput PV 1"), id="os-popen"),
        pytest.param(
            lambda: __import__("os").execvp("caput", ["caput", "PV", "1"]), id="os-execvp"
        ),
        pytest.param(
            lambda: __import__("os").posix_spawn("/usr/bin/caput", ["caput", "PV", "1"], {}),
            id="os-posix_spawn",
        ),
        pytest.param(lambda: __import__("posix").system("caput PV 1"), id="posix-system"),
        pytest.param(lambda: __import__("ctypes").CDLL("libca.so"), id="ctypes-CDLL"),
        pytest.param(
            lambda: __import__("ctypes").cdll.LoadLibrary("libca.so"), id="ctypes-LoadLibrary"
        ),
        pytest.param(lambda: __import__("ctypes").cdll.libca, id="ctypes-cdll-attribute"),
    ],
)
def test_readonly_refuses_routes_out_of_python(call):
    _run_guard("readonly")
    with pytest.raises(RuntimeError, match=_REFUSAL):
        call()


def test_readonly_prewarms_platform_processor(monkeypatch):
    """An innocent stdlib metadata lookup must survive the subprocess refusal.

    CPython resolves ``platform.uname().processor`` lazily by shelling out to
    ``uname -p`` on first read, and h5py reads it while ``import at``
    initialises its type layer — so a cold cache under the guard killed the
    import of a pure-simulation library. The guard resolves it before patching
    subprocess; with the cache forced cold here, a guard without the pre-warm
    raises the refusal out of this lookup.
    """
    monkeypatch.setattr(platform, "_uname_cache", None)
    _run_guard("readonly")
    assert isinstance(platform.processor(), str)


def test_readonly_leaves_fork_alone():
    """Forking cannot run a new program; the exec half of fork+exec is refused."""
    import os

    _run_guard("readonly")
    assert os.fork.__name__ != "_osprey_readonly_refuse"


def test_readwrite_leaves_the_escape_hatches_alone():
    import ctypes
    import os
    import subprocess

    _run_guard("readwrite")
    assert subprocess.run.__name__ != "_osprey_readonly_refuse"
    assert os.system.__name__ != "_osprey_readonly_refuse"
    assert ctypes.CDLL.__name__ != "_osprey_readonly_refuse"


def test_guard_is_silent_when_optional_libraries_are_absent(capsys, monkeypatch):
    """Most of the table is absent on any given deployment — that must not print.

    A warning per missing target would land on the stdout of every readonly
    run, which is the agent's own output channel.
    """
    for name in ("epics", "p4p", "caproto", "pvaccess", "tango", "PyTango"):
        monkeypatch.setitem(sys.modules, name, None)
    _run_guard("readonly")
    assert capsys.readouterr().out == ""


# The defining module both tests below use. It must stay OUT of the table: a
# row naming it would patch it directly, and the assertions would then observe
# the row instead of the generic defining-module step they exist to pin down.
_HOME = "aioca._impl"


def test_readonly_refuses_the_module_that_defines_a_reexported_write(monkeypatch):
    """A re-exported write has two spellings, and both have to refuse.

    A package re-exports what a private module defines, and the two names are
    one function object: patching only the attribute the table names leaves
    ``from aioca._impl import caput`` writing to the machine. So the guard
    follows each original attribute back to the module that defined it and
    refuses there too — generically, for the defining modules no row can
    enumerate (PyTango's ``tango.device_proxy`` among them).
    """
    assert not any(dotted == _HOME for dotted, _ in _READONLY_WRITE_TARGETS), (
        f"{_HOME} must stay absent from the write table, or this test passes "
        "on the row rather than on the defining-module step"
    )
    package = ModuleType("aioca")
    impl = ModuleType(_HOME)
    writes: list = []

    async def caput(pv, value, **kwargs):
        writes.append((pv, value))

    caput.__module__ = _HOME
    impl.caput = caput
    package.caput = caput
    package._impl = impl
    monkeypatch.setitem(sys.modules, "aioca", package)
    monkeypatch.setitem(sys.modules, _HOME, impl)

    _run_guard("readonly")

    with pytest.raises(RuntimeError, match=_REFUSAL):
        package.caput("SR:MAG:QF:01:CURRENT:SP", 150)
    with pytest.raises(RuntimeError, match=_REFUSAL):
        impl.caput("SR:MAG:QF:01:CURRENT:SP", 150)
    assert writes == []


def test_defining_module_step_needs_identity_not_a_name_match(monkeypatch):
    """The step follows the object, never the name.

    ``__module__``/``__name__`` are metadata a decorator or a rebind can leave
    pointing at a module that holds something else entirely under that name.
    Patching on a name match alone would silently replace an unrelated
    attribute — including a read — so the step fires only when the defining
    module still holds this exact object.
    """
    assert not any(dotted == _HOME for dotted, _ in _READONLY_WRITE_TARGETS), (
        f"{_HOME} must stay absent from the write table, or this test observes "
        "the row rather than the defining-module step"
    )
    package = ModuleType("aioca")
    impl = ModuleType(_HOME)
    reads: list = []

    async def caput(pv, value, **kwargs):
        pass

    async def unrelated(pv, **kwargs):
        reads.append(pv)
        return 1.0

    # Same ``__name__``/``__module__`` as the patched attribute, different
    # object: the module's ``caput`` is not the one the table reached.
    caput.__module__ = _HOME
    unrelated.__name__ = "caput"
    unrelated.__module__ = _HOME
    impl.caput = unrelated
    package.caput = caput
    monkeypatch.setitem(sys.modules, "aioca", package)
    monkeypatch.setitem(sys.modules, _HOME, impl)

    _run_guard("readonly")

    with pytest.raises(RuntimeError, match=_REFUSAL):
        package.caput("SR:MAG:QF:01:CURRENT:SP", 150)
    assert impl.caput is unrelated
    assert asyncio.run(impl.caput("SR:MAG:QF:01:CURRENT")) == 1.0
    assert reads == ["SR:MAG:QF:01:CURRENT"]


def test_defining_module_failure_does_not_skip_the_rest_of_the_row(capsys, monkeypatch):
    """A failing defining-module step costs its own attribute, nothing more.

    The step is secondary — the attribute the table names already refuses when
    it runs — but it touches metadata a library controls, so it can raise:
    an unhashable ``__module__``, or a PEP 562 module ``__getattr__`` that
    raises something other than ``AttributeError``. If that escaped to the row
    handler, every LATER attribute of the row would be left unpatched, which
    for a row like ``os`` means ``execv`` still spawning after ``system`` was
    caught.
    """
    module = ModuleType("osprey_fake_client")

    def write_a(*args, **kwargs):
        return "wrote a"

    def write_b(*args, **kwargs):
        return "wrote b"

    # Unhashable ``__module__``: ``sys.modules.get`` raises TypeError.
    write_a.__module__ = ["not", "a", "name"]
    module.write_a = write_a
    module.write_b = write_b
    monkeypatch.setitem(sys.modules, "osprey_fake_client", module)
    monkeypatch.setattr(
        wrapper_module,
        "_READONLY_WRITE_TARGETS",
        (("osprey_fake_client", ("write_a", "write_b")),),
    )

    _run_guard("readonly")

    with pytest.raises(RuntimeError, match=_REFUSAL):
        module.write_a()
    with pytest.raises(RuntimeError, match=_REFUSAL):
        # The attribute AFTER the one whose step raised must still refuse.
        module.write_b()
    warning = capsys.readouterr().out
    assert "osprey_fake_client.write_a" in warning, (
        "the operator has to be told which attribute's step failed"
    )


# ---------------------------------------------------------------------------
# The installed libraries
#
# Everything above runs the guard's source in this process against fakes, which
# is what makes a spelling assertion readable — but a fake cannot tell us
# whether the guard reaches the object a REAL client library defines its write
# on. aioca re-exports ``caput`` from ``aioca._catools``; every p4p client
# flavour inherits ``put`` from ``p4p.client.raw.Context``; ophyd-async defines
# ``set`` on a base of ``SignalRW``. A guard that patches only the name the
# table spells leaves each of those originals reachable, and no fake would
# notice.
#
# So this runs in a real interpreter against whatever is installed: snapshot
# every resolvable row first, exec the guard, then assert the original survives
# nowhere — not at its defining module, and not in any base class that still
# holds it. A subprocess rather than this process because the guard patches
# ``os``, ``subprocess`` and ``ctypes`` for real, and because two of the checks
# have to call a write for real.
# ---------------------------------------------------------------------------

#: Last line of the probe's stdout, so anything the guard itself printed — it
#: warns about a row it could not patch — stays distinguishable from the report.
_REPORT_MARKER = "@@REPORT@@"

_INSTALLED_LIBRARY_PROBE = """
# Snapshot the write surface as the installed libraries actually define it,
# install the guard, then report what survived. A report rather than bare
# assertions, so the parent test can also check the run had teeth: a probe that
# silently checked nothing must not read as a pass.
#
# argv[1] is the guard source; argv[2] is the substring every refusal carries.
import importlib
import json
import pathlib
import sys

from osprey.services.python_executor.write_surface import (
    _CLIENT_WRITE_TARGETS,
    _FRAMEWORK_WRITE_TARGETS,
)

TABLE = _CLIENT_WRITE_TARGETS + _FRAMEWORK_WRITE_TARGETS
REFUSE = "_osprey_readonly_refuse"
REPORT_MARKER = "@@REPORT@@"
MARKER = sys.argv[2]

# Py_TPFLAGS_IMMUTABLETYPE. A C-extension type carrying it refuses setattr, so
# the guard cannot patch it and the MRO walk must not demand that it did.
IMMUTABLE_TYPE = 1 << 8


def resolve(dotted):
    # The guard's own resolver, restated: the emitted guard has to be
    # self-contained in the subprocess it runs in, so there is none to share.
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


def refusing(value):
    return getattr(value, "__name__", None) == REFUSE


# --- before the guard ------------------------------------------------------
# Only attributes that exist NOW are checked afterwards. The guard skips an
# attribute a library does not have, and a row may name a client flavour that
# is not installed; neither is a finding.
snapshot = []
absent = []
for dotted, attrs in TABLE:
    obj = resolve(dotted)
    if obj is None:
        absent.append(dotted)
        continue
    for attr in attrs:
        if not hasattr(obj, attr):
            continue
        original = getattr(obj, attr)
        try:
            home = sys.modules.get(getattr(original, "__module__", None))
        except TypeError:
            # An unhashable ``__module__``; the guard tolerates it too.
            home = None
        name = getattr(original, "__name__", None)
        # Conditional: most originals are not module-level names at all
        # (``epics.PV.put`` is a method), and a name that resolves to a
        # DIFFERENT object is not this write's definition.
        defines_it = (
            home is not None
            and isinstance(name, str)
            and getattr(home, name, None) is original
        )
        bases = []
        if isinstance(obj, type):
            # ``__mro__[:-1]`` drops ``object``, which defines none of these.
            bases = [base for base in obj.__mro__[:-1] if attr in vars(base)]
        snapshot.append(
            {
                "dotted": dotted,
                "attr": attr,
                "home": home,
                "name": name,
                "defines_it": defines_it,
                "bases": bases,
            }
        )

# --- the guard -------------------------------------------------------------
exec(compile(pathlib.Path(sys.argv[1]).read_text(), "<osprey-readonly-guard>", "exec"), globals())

# --- after the guard -------------------------------------------------------
failures = []
checked_home = []
checked_mro = []
skipped_mro = []

for row in snapshot:
    label = row["dotted"] + "." + row["attr"]
    if row["defines_it"]:
        home_label = row["home"].__name__ + "." + row["name"]
        checked_home.append(home_label)
        if not refusing(getattr(row["home"], row["name"], None)):
            failures.append(
                "(a) " + label + ": the module that defines it, " + home_label
                + ", still holds the original write"
            )
    for base in row["bases"]:
        base_label = base.__module__ + "." + base.__qualname__ + "." + row["attr"]
        if base.__flags__ & IMMUTABLE_TYPE:
            # A C-extension type refuses setattr. That floor is stated in
            # write_surface's docstring and covered by the import denylist.
            skipped_mro.append(base_label + " (immutable C type)")
            continue
        if getattr(base, "_is_protocol", False):
            # A typing.Protocol member is a stub nobody calls; patching it
            # would say nothing about the class that implements it.
            skipped_mro.append(base_label + " (typing.Protocol)")
            continue
        checked_mro.append(base_label)
        if not refusing(vars(base).get(row["attr"])):
            failures.append(
                "(b) " + label + ": base class " + base_label
                + " still holds the original write"
            )

# --- the rows this hotfix exists for ---------------------------------------
NAMED = (
    ("aioca._catools", "caput"),
    ("p4p.client.raw.Context", "put"),
    ("epicscorelibs.ca.cadef", "ca_array_put"),
    ("bluesky.run_engine.RunEngine", "__call__"),
    ("ophyd_async.core.SignalRW", "set"),
)
checked_named = []
for dotted, attr in NAMED:
    obj = resolve(dotted)
    if obj is None:
        failures.append("(c) " + dotted + " did not resolve in the probe")
        continue
    checked_named.append(dotted + "." + attr)
    if not refusing(getattr(obj, attr, None)):
        failures.append("(c) " + dotted + "." + attr + " is not the refusing function")


def expect_refusal(label, call):
    # A real call, not an identity check: the two frameworks reach their write
    # through a dunder and through a base class, which is exactly where an
    # identity check on the spelled name passes while the write still runs.
    try:
        call()
    except RuntimeError as exc:
        if MARKER not in str(exc):
            failures.append(
                "(d) " + label + " raised a RuntimeError that is not the refusal: " + str(exc)
            )
        return
    except BaseException as exc:
        failures.append(
            "(d) " + label + " raised " + type(exc).__name__ + " instead of refusing: " + str(exc)
        )
        return
    failures.append("(d) " + label + " returned instead of refusing")


import bluesky.run_engine

# ``__new__`` without ``__init__``: running a plan is what has to refuse, and a
# fully built RunEngine would open an event loop to find that out.
expect_refusal(
    "RunEngine(plan)",
    lambda: bluesky.run_engine.RunEngine.__new__(bluesky.run_engine.RunEngine)([]),
)

import ophyd_async.core

expect_refusal(
    "soft_signal_rw(float, 0.0).set(1.0)",
    lambda: ophyd_async.core.soft_signal_rw(float, 0.0).set(1.0),
)

# --- the read path the guard must NOT close --------------------------------
# ``p4p.client.raw`` subclasses the immutable C operation type at import time
# and hands the SAME object out for get as for put, so a row naming either of
# those two names refuses every PVAccess read — the path readonly mode exists
# to keep open. Pinned here: a get against a PV nobody serves has to time out,
# and only the put has to refuse.
import p4p.client.thread

context = p4p.client.thread.Context("pva")
try:
    try:
        context.get("OSPREY:READONLY:PROBE:NO:SUCH:PV", timeout=0.3)
        read_outcome = "returned a value"
    except TimeoutError:
        read_outcome = "TimeoutError"
    except BaseException as exc:
        read_outcome = type(exc).__name__ + ": " + str(exc)
    if read_outcome != "TimeoutError":
        failures.append(
            "(read) p4p Context.get under the guard: " + read_outcome
            + " -- a readonly run must still be able to read PVAccess"
        )
    expect_refusal(
        "p4p thread Context.put",
        lambda: context.put("OSPREY:READONLY:PROBE:NO:SUCH:PV", 1.0, timeout=0.3),
    )
finally:
    context.close()

print(
    REPORT_MARKER
    + json.dumps(
        {
            "failures": failures,
            "absent": absent,
            "checked_home": checked_home,
            "checked_mro": checked_mro,
            "skipped_mro": skipped_mro,
            "checked_named": checked_named,
            "attributes": [row["dotted"] + "." + row["attr"] for row in snapshot],
        }
    )
)
"""

#: Appended AFTER the guard source, in a cold interpreter. Bluesky's import
#: chain reads ``platform.uname().processor``, which CPython answers by
#: shelling out on first read — under a guard that refuses spawning and forgot
#: to pre-warm it, importing Bluesky at all would fail. So this checks the
#: import the guard has to leave working, in the one ordering where it breaks.
_COLD_IMPORT_PROBE = """
import importlib

run_engine = importlib.import_module("bluesky.run_engine")
assert run_engine.RunEngine.__call__.__name__ == "_osprey_readonly_refuse", (
    "a framework imported after the guard must still refuse its write"
)
print("@@COLD-OK@@")
"""


def _run_probe(tmp_path, name, source, *args):
    """Run *source* in a real interpreter and return its stdout."""
    script = tmp_path / name
    script.write_text(source)
    result = subprocess.run(
        [sys.executable, str(script), *args], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, (
        f"{name} exited {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return result.stdout


def test_guard_holds_against_installed_libraries(tmp_path):
    """Every installed write the table names refuses at the object defining it.

    The in-process tests above prove the guard patches the name each row
    spells. This proves the harder half against the real libraries: that the
    original is not still reachable one level down — at the private module a
    package re-exports it from, or in a base class the spelled class inherits
    it from. Both were open on the pre-fix table, and neither is visible to a
    fake.

    Rows for libraries that are not installed contribute nothing, which is the
    ordinary case; the named rows are guarded by ``importorskip`` so a missing
    library skips loudly instead of passing quietly.
    """
    for library in ("aioca", "p4p", "epicscorelibs", "bluesky", "ophyd_async"):
        pytest.importorskip(library, reason=f"{library} is not installed in this environment")

    guard_path = tmp_path / "readonly_guard.py"
    guard_path.write_text(ExecutionWrapper(execution_mode="readonly")._get_readonly_guard())

    stdout = _run_probe(
        tmp_path,
        "probe_installed_libraries.py",
        _INSTALLED_LIBRARY_PROBE,
        str(guard_path),
        READONLY_REFUSAL_MARKER,
    )

    assert _REPORT_MARKER in stdout, f"the probe produced no report:\n{stdout}"
    noise, _, payload = stdout.partition(_REPORT_MARKER)
    assert noise.strip() == "", (
        f"the guard warned about a row it could not patch against the installed libraries:\n{noise}"
    )
    report = json.loads(payload)

    assert report["failures"] == [], "\n".join(["the guard did not hold:", *report["failures"]])

    # --- the run has to have had teeth --------------------------------------
    # Every check above is a loop over what resolved, so all of them pass
    # vacuously if nothing did. The rows this hotfix was written for are named
    # here: the libraries are importorskip'd above, so their absence from the
    # report is a broken probe rather than a thin environment.
    assert set(report["checked_named"]) == {
        "aioca._catools.caput",
        "p4p.client.raw.Context.put",
        "epicscorelibs.ca.cadef.ca_array_put",
        "bluesky.run_engine.RunEngine.__call__",
        "ophyd_async.core.SignalRW.set",
    }
    assert "aioca._catools.caput" in report["checked_home"], (
        "the defining-module check must have run for aioca, the re-export that "
        "reached the machine before this fix"
    )
    assert "p4p.client.raw.Context.put" in report["checked_mro"], (
        "the MRO check must have run for the p4p base every client flavour inherits its put from"
    )
    assert "ophyd_async.core._signal.SignalW.set" in report["checked_mro"], (
        "the MRO check must have run for the ophyd-async base SignalRW inherits set from"
    )

    # --- and the skips have to be the two named reasons ---------------------
    # A skip is how a check stops being a check, so an unexplained one is the
    # quiet way this test would lose its teeth.
    for skipped in report["skipped_mro"]:
        assert skipped.endswith(("(immutable C type)", "(typing.Protocol)")), (
            f"a base was skipped for an unnamed reason: {skipped}"
        )
    assert any(s.startswith("p4p._p4p.SharedPV.") for s in report["skipped_mro"]), (
        "p4p's immutable C base is the floor write_surface documents; it has to "
        "be reached and skipped, not silently absent"
    )
    assert any(s.startswith("bluesky.protocols.Movable.set") for s in report["skipped_mro"]), (
        "the Movable protocol stub has to be reached and skipped"
    )

    # --- the guard must not break the import --------------------------------
    cold_stdout = _run_probe(
        tmp_path, "probe_cold_import.py", guard_path.read_text() + _COLD_IMPORT_PROBE
    )
    assert cold_stdout.strip() == "@@COLD-OK@@", (
        f"importing Bluesky under the guard is not clean:\n{cold_stdout}"
    )
