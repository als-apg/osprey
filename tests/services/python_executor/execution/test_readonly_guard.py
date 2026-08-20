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
import re
import sys
from types import ModuleType

import pytest

from osprey.services.python_executor.execution.wrapper import (
    READONLY_REFUSAL,
    ExecutionWrapper,
)

pytestmark = pytest.mark.unit

# The refusal text contains literal parentheses; escape it for pytest.raises.
_REFUSAL = re.escape(READONLY_REFUSAL)


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
