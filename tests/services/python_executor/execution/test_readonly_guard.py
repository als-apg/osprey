"""Tests for the readonly-run guard in the generated execution wrapper.

A script submitted with ``execution_mode="readonly"`` must be unable to write
to the control system *at runtime*, however the write is spelled. The
pre-execution regex only sees the standard spellings, so the wrapper installs
refusing replacements for every direct-library write entry point before the
user code runs. Late binding makes this spelling-independent: an alias such as
``from epics import caput as _w`` resolves to the refusing function because the
patch is already in place when the alias is bound.

Like the limits monkeypatch tests, these execute the generated *source text*
against fake ``epics``/``p4p`` modules injected into ``sys.modules``.
"""

import asyncio
import importlib
import re
import sys
from types import ModuleType

import pytest

from osprey.services.python_executor.execution.wrapper import (
    _READONLY_WRITE_TARGETS,
    READONLY_REFUSAL,
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
            if hasattr(target, attr):
                saved.append((target, attr, getattr(target, attr)))
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
    """pvaPy spells one setter per type; the ``put`` prefix is swept wholesale."""
    mod = ModuleType("pvaccess")
    writes: list = []

    class Channel:
        def __init__(self, name):
            self.name = name

        def put(self, value):
            writes.append(("put", value))

        def putDouble(self, value):  # noqa: N802 — pvaPy's own spelling
            writes.append(("putDouble", value))

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
