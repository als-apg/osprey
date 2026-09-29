"""Armed mode of the raw-put block: a client put passes only through the door.

``install("armed", ...)`` refuses a raw control-system client put unless a
connector opened the write door around it, and optionally refuses the rpc and
Tango command rows outright. It takes its tables as literals, exactly as the
readonly mode does, and touches nothing it is not passed: no escape route, no
``ctypes`` loader, no process-spawning call.

Every fake here is built as a real module object registered in ``sys.modules``
under a name no installed library uses, so the defining-module step the engine
takes resolves to the fake itself and never reaches an installed client or a
test module. The fixture from :mod:`tests.runtime._patch_restore` restores
whatever the block patched, and drops its import-hook finder.
"""

from __future__ import annotations

import ast
import asyncio
import ctypes
import inspect
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

from osprey.runtime import raw_put_block
from osprey.services.python_executor.execution.wrapper import READONLY_REFUSAL_MARKER
from osprey_connectors.control_system.write_door import open_door
from osprey_connectors.errors import (
    RAW_CLIENT_WRITE_MARKER,
    ChannelWriteBlockedError,
)
from tests.runtime._patch_restore import patch_restore, restore_patches  # noqa: F401

P4P_RPC_TEXT = "rpc is not mediated and cannot be approved — use the supervised write path"
TANGO_COMMAND_TEXT = (
    "Tango command refused in a limits-checked run: a command carries no value to bound"
)

_FAKE = "osprey_fake_armed"


def _fake_module(monkeypatch, restore_patches, name: str, source: str) -> ModuleType:  # noqa: F811
    """Build module *name* from *source*, register it, and track it for restore."""
    module = ModuleType(name)
    module.__file__ = f"<{name}>"
    exec(compile(textwrap.dedent(source), module.__file__, "exec"), vars(module))
    monkeypatch.setitem(sys.modules, name, module)
    restore_patches(module)
    for value in list(vars(module).values()):
        if isinstance(value, type):
            restore_patches(value)
    return module


def _install(**overrides):
    contract = {
        "blocked_targets": (),
        "rpc_targets": (),
        "refuse_rpc": False,
        "marker": RAW_CLIENT_WRITE_MARKER,
        "rpc_refusals": {},
    }
    contract.update(overrides)
    raw_put_block.install("armed", **contract)


@pytest.fixture
def client(monkeypatch, restore_patches):  # noqa: F811
    """A fake client: a module-level put, a PV class, an async put and an rpc."""
    return _fake_module(
        monkeypatch,
        restore_patches,
        _FAKE,
        """
        calls = []

        def caput(pvname, value, wait=False):
            calls.append((pvname, value))
            return "put-ok"

        async def acaput(pvname, value):
            calls.append((pvname, value))
            return "aput-ok"

        def read(pvname):
            return 1.0

        class PV:
            def __init__(self, pvname):
                self.pvname = pvname

            def put(self, value):
                calls.append((self.pvname, value))
                return "pv-ok"

            def get(self):
                return 2.0

            @staticmethod
            def static_put(pvname, value):
                calls.append((pvname, value))
                return "static-ok"

        class Context:
            name = "pva"

            def put(self, name, values):
                calls.append((name, values))
                return "ctx-ok"

            def rpc(self, name, value):
                return "rpc-ok"

        class Server:
            def post(self, value):
                return "post-ok"
        """,
    )


# --- blocked rows -----------------------------------------------------------


def test_blocked_put_refuses_with_raw_client_write_when_door_closed(client, caplog):
    _install(blocked_targets=((_FAKE, ("caput",)),))

    with caplog.at_level(logging.WARNING), pytest.raises(ChannelWriteBlockedError) as info:
        client.caput("SR:QF:SETPOINT", 3.0)

    assert info.value.reason == "RAW_CLIENT_WRITE"
    assert info.value.channel_address == "SR:QF:SETPOINT"
    assert RAW_CLIENT_WRITE_MARKER in str(info.value)
    assert "osprey.runtime.write_channel" in str(info.value)
    assert client.calls == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(RAW_CLIENT_WRITE_MARKER in r.getMessage() for r in warnings), (
        "a refusal swallowed by a callback must still be visible in the log"
    )


def test_blocked_put_passes_through_open_door(client):
    _install(blocked_targets=((_FAKE, ("caput",)),))

    with open_door():
        assert client.caput("SR:QF:SETPOINT", 3.0, wait=True) == "put-ok"

    assert client.calls == [("SR:QF:SETPOINT", 3.0)]
    with pytest.raises(ChannelWriteBlockedError):
        client.caput("SR:QF:SETPOINT", 4.0)


def test_blocked_method_derives_address_from_the_object(client):
    _install(blocked_targets=((f"{_FAKE}.PV", ("put",)),))

    with pytest.raises(ChannelWriteBlockedError) as info:
        client.PV("SR:BPM1").put(1.0)
    assert info.value.channel_address == "SR:BPM1"
    assert client.PV("SR:BPM1").get() == 2.0, "an unlisted attribute stays as it was"

    with open_door():
        assert client.PV("SR:BPM1").put(1.0) == "pv-ok"


def test_blocked_async_put_stays_a_coroutine_function(client):
    _install(blocked_targets=((_FAKE, ("acaput",)),))

    assert inspect.iscoroutinefunction(client.acaput)
    with pytest.raises(ChannelWriteBlockedError):
        asyncio.run(client.acaput("SR:A", 1))

    async def _through_door():
        with open_door():
            return await client.acaput("SR:A", 1)

    assert asyncio.run(_through_door()) == "aput-ok"


def test_blocked_put_passes_through_to_thread_but_not_a_bare_thread(client):
    import threading

    _install(blocked_targets=((_FAKE, ("caput",)),))

    async def _connector_put():
        with open_door():
            return await asyncio.to_thread(client.caput, "SR:T", 1)

    assert asyncio.run(_connector_put()) == "put-ok"

    errors: list[BaseException] = []

    def _raw():
        try:
            client.caput("SR:T", 2)
        except BaseException as error:
            errors.append(error)

    with open_door():
        worker = threading.Thread(target=_raw)
        worker.start()
        worker.join()
    assert len(errors) == 1 and isinstance(errors[0], ChannelWriteBlockedError)


def test_staticmethod_row_keeps_its_descriptor(client):
    _install(blocked_targets=((f"{_FAKE}.PV", ("static_put",)),))

    assert isinstance(inspect.getattr_static(client.PV, "static_put"), staticmethod)
    with open_door():
        assert client.PV("X").static_put("SR:S", 1) == "static-ok"
    with pytest.raises(ChannelWriteBlockedError) as info:
        client.PV.static_put("SR:S", 1)
    assert info.value.channel_address == "SR:S"


def test_reexported_put_refuses_under_both_spellings(monkeypatch, restore_patches):  # noqa: F811
    home = _fake_module(
        monkeypatch,
        restore_patches,
        f"{_FAKE}_home",
        """
        def caput(pvname, value):
            return "ok"
        """,
    )
    facade = _fake_module(monkeypatch, restore_patches, f"{_FAKE}_facade", "")
    facade.caput = home.caput

    _install(blocked_targets=((f"{_FAKE}_facade", ("caput",)),))

    for spelling in (facade.caput, home.caput):
        with pytest.raises(ChannelWriteBlockedError):
            spelling("SR:R", 1)
    with open_door():
        assert home.caput("SR:R", 1) == "ok"


def test_second_install_does_not_wrap_twice(client):
    rows = ((_FAKE, ("caput",)),)
    _install(blocked_targets=rows)
    first = client.caput
    _install(blocked_targets=rows)
    assert client.caput is first


def test_deferred_module_is_patched_when_imported(monkeypatch, tmp_path, restore_patches):  # noqa: F811
    package = f"{_FAKE}_late"
    (tmp_path / f"{package}.py").write_text("def caput(pvname, value):\n    return 'late-ok'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, package, raising=False)

    _install(blocked_targets=((package, ("caput",)),))
    module = __import__(package)
    restore_patches(module)

    with pytest.raises(ChannelWriteBlockedError):
        module.caput("SR:L", 1)
    with open_door():
        assert module.caput("SR:L", 1) == "late-ok"


def test_pvaccess_sweep_uses_the_armed_replacement(monkeypatch, restore_patches):  # noqa: F811
    pvaccess = _fake_module(
        monkeypatch,
        restore_patches,
        "pvaccess",
        """
        class Channel:
            def __init__(self, name):
                self._name = name

            def getName(self):
                return self._name

            def put(self, value):
                return "put-ok"

            def putDouble(self, value):
                return "double-ok"

            def parsePutGet(self, value):
                return "parse-ok"

            def get(self):
                return 3.0
        """,
    )
    _install(blocked_targets=(("pvaccess.Channel", ("put",)),))

    channel = pvaccess.Channel("SR:PVA")
    for verb in ("put", "putDouble", "parsePutGet"):
        with pytest.raises(ChannelWriteBlockedError) as info:
            getattr(channel, verb)(1.0)
        assert info.value.channel_address == "SR:PVA"
    with open_door():
        assert channel.putDouble(1.0) == "double-ok"
    assert channel.get() == 3.0


# --- rpc rows ---------------------------------------------------------------


def test_rpc_rows_refused_with_their_texts_when_refuse_rpc(monkeypatch, restore_patches, client):  # noqa: F811
    tango = _fake_module(
        monkeypatch,
        restore_patches,
        f"{_FAKE}_tango",
        """
        class DeviceProxy:
            def command_inout(self, name, arg=None):
                return "cmd-ok"
        """,
    )
    _install(
        rpc_targets=(
            (f"{_FAKE}.Context", ("rpc",)),
            (f"{_FAKE}_tango.DeviceProxy", ("command_inout",)),
        ),
        refuse_rpc=True,
        rpc_refusals={_FAKE: P4P_RPC_TEXT, f"{_FAKE}_tango": TANGO_COMMAND_TEXT},
    )

    with open_door():  # an rpc refusal does not consult the door
        with pytest.raises(RuntimeError) as p4p_info:
            client.Context().rpc("SR:RPC", 1)
        with pytest.raises(RuntimeError) as tango_info:
            tango.DeviceProxy().command_inout("Init")
    assert str(p4p_info.value) == P4P_RPC_TEXT
    assert str(tango_info.value) == TANGO_COMMAND_TEXT
    assert not isinstance(p4p_info.value, ChannelWriteBlockedError)


def test_rpc_refusal_prefers_the_longest_matching_key(client):
    _install(
        rpc_targets=((f"{_FAKE}.Context", ("rpc",)),),
        refuse_rpc=True,
        rpc_refusals={_FAKE: "outer", f"{_FAKE}.Context": "inner"},
    )
    with pytest.raises(RuntimeError, match="^inner$"):
        client.Context().rpc("SR:RPC", 1)


def test_rpc_rows_untouched_when_refuse_rpc_is_false(client):
    original = client.Context.rpc
    _install(
        rpc_targets=((f"{_FAKE}.Context", ("rpc",)),),
        refuse_rpc=False,
        rpc_refusals={_FAKE: P4P_RPC_TEXT},
    )
    assert client.Context.rpc is original
    assert client.Context().rpc("SR:RPC", 1) == "rpc-ok"


def test_rpc_row_without_a_refusal_text_fails_install(client):
    original = client.Context.rpc
    with pytest.raises(ValueError, match="rpc"):
        _install(
            rpc_targets=((f"{_FAKE}.Context", ("rpc",)),),
            refuse_rpc=True,
            rpc_refusals={"somewhere_else": "text"},
        )
    assert client.Context.rpc is original


def test_put_and_rpc_on_one_class_both_apply(client):
    _install(
        blocked_targets=((f"{_FAKE}.Context", ("put",)),),
        rpc_targets=((f"{_FAKE}.Context", ("rpc",)),),
        refuse_rpc=True,
        rpc_refusals={_FAKE: P4P_RPC_TEXT},
    )
    with pytest.raises(ChannelWriteBlockedError) as info:
        client.Context().put("SR:P4P", 1)
    assert info.value.channel_address == "SR:P4P"
    with pytest.raises(RuntimeError, match="rpc is not mediated"):
        client.Context().rpc("SR:P4P", 1)


# --- what armed mode leaves alone -------------------------------------------


def _escape_route_identities() -> dict[str, object]:
    return {
        "subprocess.Popen": subprocess.Popen,
        "subprocess.run": subprocess.run,
        "subprocess.call": subprocess.call,
        "subprocess.check_output": subprocess.check_output,
        "os.system": os.system,
        "os.popen": os.popen,
        "os.execv": os.execv,
        "ctypes.CDLL": ctypes.CDLL,
        "ctypes.PyDLL": ctypes.PyDLL,
        "ctypes.LibraryLoader.LoadLibrary": ctypes.LibraryLoader.LoadLibrary,
        "ctypes.LibraryLoader.__getattr__": ctypes.LibraryLoader.__getattr__,
    }


@pytest.mark.usefixtures("client")
def test_escape_routes_and_ctypes_are_untouched():
    before = _escape_route_identities()
    _install(
        blocked_targets=((_FAKE, ("caput",)), (f"{_FAKE}.PV", ("put",))),
        rpc_targets=((f"{_FAKE}.Context", ("rpc",)),),
        refuse_rpc=True,
        rpc_refusals={_FAKE: P4P_RPC_TEXT},
    )
    after = _escape_route_identities()
    changed = [name for name in before if after[name] is not before[name]]
    assert changed == [], f"armed mode patched {changed}"


def test_unlisted_rows_pass(client):
    _install(blocked_targets=((_FAKE, ("caput",)),))
    assert client.read("SR:X") == 1.0
    assert client.Server().post(1) == "post-ok"


def test_no_refusal_carries_the_readonly_marker(client):
    _install(
        blocked_targets=((_FAKE, ("caput",)),),
        rpc_targets=((f"{_FAKE}.Context", ("rpc",)),),
        refuse_rpc=True,
        rpc_refusals={_FAKE: P4P_RPC_TEXT},
    )
    messages = []
    for call in (lambda: client.caput("SR:M", 1), lambda: client.Context().rpc("SR:M", 1)):
        with pytest.raises(Exception) as info:
            call()
        messages.append(str(info.value))
    assert all(READONLY_REFUSAL_MARKER not in message for message in messages), messages


# --- without osprey_connectors ----------------------------------------------


def _hide_connectors(monkeypatch):
    for name in list(sys.modules):
        if name == "osprey_connectors" or name.startswith("osprey_connectors."):
            monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "osprey_connectors", None)


def test_without_connectors_refusal_is_a_stdlib_runtime_error(client, monkeypatch):
    _install(blocked_targets=((_FAKE, ("caput",)),))

    # Scoped: the suite's own teardown imports osprey_connectors.
    with monkeypatch.context() as hidden, pytest.raises(RuntimeError) as info:
        _hide_connectors(hidden)
        client.caput("SR:NC", 1)
    assert type(info.value) is RuntimeError
    assert RAW_CLIENT_WRITE_MARKER in str(info.value)
    assert "SR:NC" in str(info.value)
    assert client.calls == []


def test_source_runs_in_a_private_namespace_without_osprey(client, monkeypatch):
    """The executor's spelling: exec the source into a dict, then call install."""
    source = Path(raw_put_block.__file__).read_text(encoding="utf-8")
    # Scoped: the suite's own teardown imports osprey_connectors.
    with monkeypatch.context() as hidden:
        for name in list(sys.modules):
            if name == "osprey" or name.startswith("osprey."):
                hidden.setitem(sys.modules, name, None)
        _hide_connectors(hidden)

        namespace: dict = {}
        exec(compile(source, "osprey/runtime/raw_put_block.py", "exec"), namespace)
        namespace["install"](
            "armed",
            blocked_targets=((_FAKE, ("caput",)),),
            rpc_targets=(),
            refuse_rpc=False,
            marker="custom marker text",
            rpc_refusals={},
        )
        with pytest.raises(RuntimeError, match="custom marker text") as info:
            client.caput("SR:NS", 1)
    assert type(info.value) is RuntimeError


def test_module_scope_imports_neither_write_surface_nor_errors():
    tree = ast.parse(Path(raw_put_block.__file__).read_text(encoding="utf-8"))
    top_level = [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)]
    names = set()
    for node in top_level:
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
        else:
            names.update(alias.name for alias in node.names)
    assert not any(name.startswith(("osprey", "write_surface")) for name in names), names
    source = Path(raw_put_block.__file__).read_text(encoding="utf-8")
    assert "write_surface" not in "".join(
        line for line in source.splitlines() if line.lstrip().startswith(("import", "from"))
    )


# --- contract ---------------------------------------------------------------


def test_armed_rejects_the_readonly_keywords():
    with pytest.raises(TypeError):
        raw_put_block.install("armed", eager_targets=(), deferred_targets=(), refusal="x")


def test_readonly_rejects_the_armed_keywords():
    with pytest.raises(TypeError):
        raw_put_block.install(
            "readonly",
            blocked_targets=(),
            rpc_targets=(),
            refuse_rpc=False,
            marker="x",
            rpc_refusals={},
        )


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="unknown raw-put block mode"):
        raw_put_block.install("readwrite", blocked_targets=())


@pytest.mark.parametrize("marker", ["", None])
def test_armed_requires_a_marker(marker):
    with pytest.raises(ValueError, match="marker"):
        _install(marker=marker)


# --- address derivation -----------------------------------------------------


class _Named:
    def __init__(self, **attrs):
        vars(self).update(attrs)


def test_address_from_epics_ca_chid(monkeypatch):
    ca = ModuleType("epics.ca")
    ca.name = lambda chid: {7: "SR:CHID"}[chid]
    monkeypatch.setitem(sys.modules, "epics.ca", ca)
    derive = raw_put_block._derive_address
    assert derive("epics.ca", "put", (7, 1.0), {}) == "SR:CHID"
    assert derive("epics.ca", "sg_put", ("gid", 7, 1.0), {}) == "SR:CHID"


def test_address_from_cadef_chid(monkeypatch):
    cadef = ModuleType("epicscorelibs.ca.cadef")
    cadef.ca_name = lambda chid: b"SR:CADEF"
    monkeypatch.setitem(sys.modules, "epicscorelibs.ca.cadef", cadef)
    derive = raw_put_block._derive_address
    assert derive("epicscorelibs.ca.cadef", "ca_array_put", (6, 1, 9, None), {}) == "SR:CADEF"
    assert (
        derive("epicscorelibs.ca.cadef", "ca_array_put_callback", (6, 1, 9, None, None, None), {})
        == "SR:CADEF"
    )


@pytest.mark.parametrize(
    ("dotted", "attr", "args", "kwargs", "expected"),
    [
        ("epics", "caput", ("SR:A", 1), {}, "SR:A"),
        ("epics", "caput", (), {"pvname": "SR:KW", "value": 1}, "SR:KW"),
        ("epics", "caput_many", (["SR:A", "SR:B"], [1, 2]), {}, "SR:A, SR:B"),
        ("epics.PV", "put", (_Named(pvname="SR:PV"), 1), {}, "SR:PV"),
        ("aioca", "caput", ("SR:AIO", 1), {}, "SR:AIO"),
        ("p4p.client.thread.Context", "put", (_Named(name="pva"), "SR:P4P", 1), {}, "SR:P4P"),
        (
            "p4p.client.thread.Context",
            "put",
            (_Named(name="pva"), ["SR:X", "SR:Y"], [1, 2]),
            {},
            "SR:X, SR:Y",
        ),
        ("caproto.sync.client", "write", ("SR:CAP", 1), {}, "SR:CAP"),
        ("caproto.threading.client.PV", "write", (_Named(name="SR:CPV"), 1), {}, "SR:CPV"),
        (
            "caproto.threading.client.Batch",
            "write",
            (_Named(), _Named(name="SR:BAT"), 1),
            {},
            "SR:BAT",
        ),
        ("pvaccess.Channel", "put", (_Named(getName=lambda: "SR:PVA"), 1), {}, "SR:PVA"),
        ("doocs4py", "set", ("XFEL/MAG/Q1/CURRENT", 1), {}, "XFEL/MAG/Q1/CURRENT"),
        (
            "tango.DeviceProxy",
            "write_attribute",
            (_Named(dev_name=lambda: "sys/mag/1"), "current", 1),
            {},
            "sys/mag/1/current",
        ),
        (
            "tango.DeviceProxy",
            "write_attribute",
            (_Named(dev_name=lambda: "sys/mag/1"), _Named(name="voltage"), 1),
            {},
            "sys/mag/1/voltage",
        ),
        (
            "tango.AttributeProxy",
            "write",
            (_Named(name=lambda: "sys/mag/1/current"), 1),
            {},
            "sys/mag/1/current",
        ),
        (
            "tango.Group",
            "write_attribute",
            (_Named(get_name=lambda: "magnets"), "current", 1),
            {},
            "magnets/current",
        ),
        ("epics", "caput", (), {}, "<unknown>"),
        ("epics", "caput", (object(), 1), {}, "<unknown>"),
        ("epics.PV", "put", (object(), 1), {}, "<unknown>"),
        ("epics.ca", "put", (7, 1), {}, "<unknown>"),
    ],
)
def test_address_derivation(dotted, attr, args, kwargs, expected, monkeypatch):
    monkeypatch.delitem(sys.modules, "epics.ca", raising=False)
    assert raw_put_block._derive_address(dotted, attr, args, kwargs) == expected
