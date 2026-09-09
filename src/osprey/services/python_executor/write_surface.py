"""The one table of Python entry points that reach a control system.

Every gate that has to answer "is this a control-system write?" for *Python
source* reads this module rather than spelling a client library's name itself:

* the readonly runtime guard emitted into the execution subprocess
  (:mod:`osprey.services.python_executor.execution.wrapper`) patches every
  entry point here with a refusing function;
* the readonly import denylist
  (:mod:`osprey.services.python_executor.analysis.safety_checks`) is derived
  from :data:`_CLIENT_WRITE_TARGETS` — the two can no longer drift apart,
  because there is only one producer;
* the write-surface section of ``docs/source/architecture/python-executor.rst``
  is written from it.

The dotted target is resolved by importing its longest importable prefix and
then walking attributes, so a module (``epics``), a module attribute
(``epics.ca``) and a class (``p4p.client.thread.Context``) are all spelled the
same way. Attributes that do not exist on the resolved object are skipped,
which is what makes listing several client flavours free: an uninstalled or
older library simply contributes nothing. Over-listing is therefore safe and
under-listing is not.

Patching the object in ``sys.modules`` — rather than inspecting the source —
is what makes the guard immune to spelling. ``importlib.import_module("epics")``,
``from epics import caput as _w`` and ``getattr(epics, "ca" + "put")`` all end
up holding the refusing function, because they all resolve through the one
module object the guard mutates.

The floor of that technique is the Python name. An object that cannot be
re-pointed from Python is out of its reach: the immutable C-extension type
``p4p._p4p.ClientOperation`` is one such floor, and it deliberately has **no
row** in the table below. The type refuses ``setattr``, so it cannot be
patched; the two names that hand it out cannot be patched either without
breaking reads, because ``p4p/client/raw.py`` subclasses the type at
import time and resolves the subclass — the *same* object — for ``get`` as well
as for ``put`` and ``rpc``. Refusing those names would refuse every PVAccess
read, which is the path readonly mode exists to keep open. That route is
covered instead by refusing the ``p4p`` import outright in a readonly run and by
the ``p4p.client.*.Context`` ``put``/``rpc`` rows, which refuse the write calls
in any process that already holds the library. So readonly enforcement for
Channel Access and PVAccess rests on the Python layer named below together with
the import denylist, the connector's own refusal of ``write_channel`` and the
read-only network posture — not on the C objects themselves.

PyTango has a floor of the same shape. Its ``Group`` is a Python class wrapping
a C++ group it keeps at ``self._Group__group``, and that C++ class is reachable
from the module namespace as ``tango.group._RealGroup`` — so
``tango.group._RealGroup("all").write_attribute_asynch(...)`` writes past the
Python-level ``Group`` refusals. It has no row for the same reason
``p4p._p4p.ClientOperation`` has none: it is a C-extension type carrying the
group's reads as well as its writes, so it cannot be re-pointed from Python and
refusing it would refuse reading a group at all. A readonly run is covered
because ``tango`` cannot be imported there; a limits-checked run's group
refusals sit on the Python class only.

The table is split three ways because the three groups are answered
differently by the *static* import check:

* :data:`_CLIENT_WRITE_TARGETS` — control-system client libraries. A readonly
  run may not even import one: reads go through ``read_channel()``, so an
  import is only ever a route around the write gates.
* :data:`_FRAMEWORK_WRITE_TARGETS` — acquisition frameworks that can drive
  hardware but are also ordinary document/analysis libraries. Their *writes*
  refuse at runtime; their import stays allowed, which is why they are kept
  out of the denylist.
* :data:`_ESCAPE_ROUTES` — routes out of Python that reach a control system
  without importing a client at all.

Refusing a write is one question; letting an *approved* write through only
within the channel's limits is another, and the second is answered for fewer
entry points than the first. :data:`_LIMITS_WRAPPED`, :data:`_LIMITS_REFUSED`
and :data:`_LIMITS_UNWRAPPABLE` partition :data:`_CLIENT_WRITE_TARGETS` by what
a limits-checked run does with each entry point, so "which client writes are
bounded" has a written answer instead of being read out of the generated
monkeypatch. ``tests/services/python_executor/
execution/test_limits_parity.py`` holds the partition to the table.
"""

#: Control-system client libraries, as ``(dotted target, attributes)``. Also
#: the producer of :data:`READONLY_DENIED_IMPORTS`: a readonly run refuses to
#: import any top-level package named here.
_CLIENT_WRITE_TARGETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # --- EPICS Channel Access (pyepics) ---
    ("epics", ("caput", "caput_many")),
    ("epics.PV", ("put",)),
    ("epics.ca", ("put",)),
    # --- EPICS Channel Access (aioca). Not an optional extra: ophyd-async's
    # [ca] backend pulls it into every environment OSPREY builds, so a readonly
    # script can reach ``aioca.caput`` with nothing else installed. The name on
    # the package is a re-export of ``aioca._catools.caput``, so the defining
    # module is listed too: patching only ``aioca`` leaves
    # ``from aioca._catools import caput`` reaching the machine.
    ("aioca", ("caput",)),
    ("aioca._catools", ("caput",)),
    # --- Channel Access (epicscorelibs). The ctypes binding that aioca loads
    # ``libca`` through, so it is a client route in its own right. Listing it
    # also puts ``epicscorelibs`` in :data:`READONLY_DENIED_IMPORTS`, which is
    # what closes the route in a readonly run: the import is refused, so the
    # binding is never reached. A limits-checked run has no bound to put on it
    # — the call carries a value already marshalled into C memory and an
    # opaque channel handle — which is why both names sit in
    # :data:`_LIMITS_UNWRAPPABLE`. pyepics is unaffected: its
    # ``epics.ca.libca`` is ``None`` until ``initialize_libca()`` runs, and
    # that call loads a handle of its own rather than this one.
    ("epicscorelibs.ca.cadef", ("ca_array_put", "ca_array_put_callback")),
    # --- PVAccess (p4p): one client Context per concurrency flavour, plus the
    # server-side SharedPV, which puts values on the wire when it is opened or
    # posted to.
    ("p4p.client.raw.Context", ("put", "rpc")),
    ("p4p.client.thread.Context", ("put", "rpc")),
    ("p4p.client.asyncio.Context", ("put", "rpc")),
    ("p4p.client.cothread.Context", ("put", "rpc")),
    ("p4p.server.raw.SharedPV", ("post", "open")),
    ("p4p.server.thread.SharedPV", ("post", "open")),
    ("p4p.server.asyncio.SharedPV", ("post", "open")),
    # --- Channel Access (caproto) ---
    ("caproto.sync.client", ("write", "read_write_read")),
    ("caproto.threading.client.PV", ("write",)),
    ("caproto.threading.client.Batch", ("write",)),
    ("caproto.asyncio.client.PV", ("write",)),
    # --- PVAccess (pvaPy). Its ``Channel`` carries one typed setter per scalar
    # and array type, so the guards sweep the ``put``, ``asyncPut`` and
    # ``parsePut`` prefixes dynamically rather than enumerating the setters
    # here. One name per family is listed so the table still names the library
    # and so a test that puts back what a guard patched covers all three.
    ("pvaccess.Channel", ("put", "putGet", "asyncPut", "parsePut", "parsePutGet")),
    # --- DOOCS (doocs4py). The client the shipped DOOCS connector writes
    # through (``osprey_connectors.control_system.doocs_connector``), so a
    # readonly script on a DOOCS deployment can reach the machine with the one
    # library that deployment is guaranteed to have.
    ("doocs4py", ("set",)),
    # --- Tango. ``command_inout`` is included because a Tango command is an
    # action on the device, not a read — refusing it is the readonly reading.
    # ``PyTango`` is the legacy alias for the same package; when both import,
    # they resolve to the same class object and the second patch is a no-op.
    (
        "tango.DeviceProxy",
        (
            "write_attribute",
            "write_attributes",
            "write_attribute_asynch",
            "write_attributes_asynch",
            "write_read_attribute",
            "write_read_attributes",
            "put_property",
            "command_inout",
            "command_inout_asynch",
        ),
    ),
    # ``Connection`` is where PyTango DEFINES both command spellings;
    # ``DeviceProxy`` only inherits them. Patching the subclass alone installs
    # a shadow and leaves ``tango.Connection.command_inout(proxy, "On")``
    # reaching the device, so the definer carries its own row.
    ("tango.Connection", ("command_inout", "command_inout_asynch")),
    ("tango.AttributeProxy", ("write", "write_asynch", "write_read")),
    # ``Group`` is not a ``Connection`` subclass. It is a separate class with
    # its own commands and its own attribute writes, and a group write fans one
    # value out to every device the group matched.
    (
        "tango.Group",
        (
            "command_inout",
            "command_inout_asynch",
            "write_attribute",
            "write_attribute_asynch",
        ),
    ),
    (
        "PyTango.DeviceProxy",
        (
            "write_attribute",
            "write_attributes",
            "write_read_attribute",
            "command_inout",
        ),
    ),
)

#: Acquisition frameworks that can drive hardware through a client below them.
#: Deliberately NOT in :data:`READONLY_DENIED_IMPORTS`: both are also ordinary
#: document and analysis libraries, and a readonly script reading a Tiled
#: catalog or introspecting a device tree has a legitimate reason to import
#: them. What it may not do is *move* anything, so the write entry points
#: refuse at runtime.
_FRAMEWORK_WRITE_TARGETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # ophyd-async signals: ``set`` is the one method that puts a value on a
    # device, whichever backend is under it. The write-capable signal classes
    # are listed rather than the base ``Signal``, so a read-only ``SignalR``
    # keeps working.
    ("ophyd_async.core.SignalW", ("set",)),
    ("ophyd_async.core.SignalRW", ("set",)),
    ("ophyd_async.core.SignalX", ("trigger",)),
    # Running a plan is how the Bluesky stack moves hardware. Patching
    # ``__call__`` on the class is enough: Python resolves a dunder on the
    # type, so ``RE(plan)`` on any instance lands here.
    ("bluesky.RunEngine", ("__call__",)),
    ("bluesky.run_engine.RunEngine", ("__call__",)),
)

#: Routes out of Python. A readonly run has no legitimate use for these:
#: ``import subprocess`` is already refused in every mode by the static import
#: check, so anything reaching the process-spawning surface at runtime got
#: there by an evasion. ``os.fork`` is deliberately absent — forking alone
#: cannot run a new program, and refusing it would break ordinary
#: multiprocessing for no security gain, since the exec half of every
#: fork+exec is refused here.
_ESCAPE_ROUTES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "subprocess",
        (
            "run",
            "Popen",
            "call",
            "check_call",
            "check_output",
            "getoutput",
            "getstatusoutput",
        ),
    ),
    ("_posixsubprocess", ("fork_exec",)),
    # ``os`` re-exports these from ``posix``; patching only ``os`` would leave
    # ``import posix; posix.system(...)`` open, so both modules are swept.
    (
        "os",
        (
            "system",
            "popen",
            "execl",
            "execle",
            "execlp",
            "execlpe",
            "execv",
            "execve",
            "execvp",
            "execvpe",
            "spawnl",
            "spawnle",
            "spawnlp",
            "spawnlpe",
            "spawnv",
            "spawnve",
            "spawnvp",
            "spawnvpe",
            "posix_spawn",
            "posix_spawnp",
        ),
    ),
    (
        "posix",
        (
            "system",
            "popen",
            "execv",
            "execve",
            "posix_spawn",
            "posix_spawnp",
        ),
    ),
    # --- Loading a shared library sidesteps every Python-level guard above:
    # ``ctypes.CDLL("libca")`` reaches Channel Access without importing a
    # single client package. ``LibraryLoader.__getattr__`` is patched too,
    # because ``ctypes.cdll.libca`` never goes through ``CDLL`` by that name.
    ("ctypes", ("CDLL", "PyDLL", "WinDLL", "OleDLL")),
    ("ctypes.LibraryLoader", ("LoadLibrary", "__getattr__")),
)

#: Every entry point a readonly run refuses. The canonical machine-readable
#: answer to "what counts as a control-system write from Python" — the docs
#: list is written from it, and a library added above needs no other change to
#: be enforced.
_READONLY_WRITE_TARGETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    _CLIENT_WRITE_TARGETS + _FRAMEWORK_WRITE_TARGETS + _ESCAPE_ROUTES
)

#: Top-level packages a readonly run may not import, derived from
#: :data:`_CLIENT_WRITE_TARGETS` so the denylist cannot fall behind the guard.
READONLY_DENIED_IMPORTS: frozenset[str] = frozenset(
    dotted.split(".")[0] for dotted, _attrs in _CLIENT_WRITE_TARGETS
)

#: What a *limits-checked* run does with each client entry point, wrapped in
#: :data:`_LIMITS_WRAPPED`: the value is how the check is reached, either
#: ``"direct"`` — the guard patches this very name — or the name it is checked
#: through. The three buckets below partition :data:`_CLIENT_WRITE_TARGETS`
#: exactly, and are keyed by the canonical ``tango`` spelling: a
#: ``PyTango.DeviceProxy`` row resolves to the same class object, so its fate is
#: the ``tango.DeviceProxy`` row's fate and it carries no separate entry.
_LIMITS_WRAPPED: dict[tuple[str, str], str] = {
    ("epics", "caput"): "via epics.ca.put — caput() puts through a PV",
    ("epics", "caput_many"): "via epics.ca.put — one PV.put() per pair",
    ("epics.PV", "put"): "via epics.ca.put",
    ("epics.ca", "put"): (
        "direct — the choke point every pyepics spelling reaches; the channel "
        "is asked for by name because a chid is opaque"
    ),
    ("aioca", "caput"): "direct — the package re-export is rebound to the checked coroutine",
    ("aioca._catools", "caput"): (
        "direct — the defining module is rebound too, which is also what gets "
        "the array form checked: it writes each pair through this global"
    ),
    ("p4p.client.raw.Context", "put"): (
        "direct, but only for a put whose receiver IS the raw Context. A "
        "subclass receiver is forwarded unvalidated — that is a flavour "
        "Context re-entering through super().put after its own check, and "
        "equally a subclass a script writes itself, which is therefore checked "
        "nowhere"
    ),
    ("p4p.client.thread.Context", "put"): "direct",
    ("p4p.client.asyncio.Context", "put"): "direct",
    ("p4p.client.cothread.Context", "put"): "direct",
    ("caproto.sync.client", "write"): "direct",
    ("caproto.sync.client", "read_write_read"): (
        "direct — it drives the channel with the value the plain write drives, so it "
        "is bounded the same way and the reads around it are left alone"
    ),
    ("caproto.threading.client.PV", "write"): "direct — the PV knows its own channel",
    ("caproto.threading.client.Batch", "write"): (
        "direct — bounded under the name of the PV the call carries, since a batch is "
        "handed the PV to drive rather than being bound to one"
    ),
    ("caproto.asyncio.client.PV", "write"): (
        "direct — a coroutine guard, so a max_step channel is read through the PV's "
        "own await before the check rather than through a reader handed to it"
    ),
    ("pvaccess.Channel", "put"): "direct — swept by the put prefix",
    ("pvaccess.Channel", "putGet"): "direct — swept by the put prefix",
    ("pvaccess.Channel", "asyncPut"): "direct — swept by the asyncPut prefix",
    ("doocs4py", "set"): "direct",
    ("tango.DeviceProxy", "write_attribute"): (
        "direct — the channel is rebuilt as dev_name()/attribute"
    ),
    ("tango.DeviceProxy", "write_attributes"): "direct — one check per (attribute, value) pair",
    ("tango.DeviceProxy", "write_attribute_asynch"): (
        "direct — the asynchronous spelling drives the same device with the same "
        "value, and is checked against the same rebuilt address"
    ),
    ("tango.DeviceProxy", "write_attributes_asynch"): (
        "direct — one check per (attribute, value) pair, taken positionally or under "
        "either keyword PyTango names the pairs by"
    ),
    ("tango.DeviceProxy", "write_read_attribute"): (
        "direct — reading the attribute back afterwards does not change the value "
        "written, so it is bounded like the plain write"
    ),
    ("tango.DeviceProxy", "write_read_attributes"): (
        "direct — one check per (attribute, value) pair, taken positionally or under "
        "either keyword PyTango names the pairs by"
    ),
    ("tango.AttributeProxy", "write"): (
        "direct — the address is rebuilt from the device proxy behind the attribute "
        "and the attribute's own name; the call forwards to a wrapped DeviceProxy "
        "spelling, so the bound is applied twice and the outer check is the one left "
        "if the DeviceProxy install fails"
    ),
    ("tango.AttributeProxy", "write_asynch"): (
        "direct — the same rebuilt address as the plain attribute write"
    ),
    ("tango.AttributeProxy", "write_read"): (
        "direct — the same rebuilt address as the plain attribute write"
    ),
}

#: Entry points a limits-checked run refuses outright, in range or not, with
#: the reason no bound can be put on them.
_LIMITS_REFUSED: dict[tuple[str, str], str] = {
    ("p4p.client.raw.Context", "rpc"): "an rpc payload is arbitrary; limits cannot apply",
    ("p4p.client.thread.Context", "rpc"): "an rpc payload is arbitrary; limits cannot apply",
    ("p4p.client.asyncio.Context", "rpc"): "an rpc payload is arbitrary; limits cannot apply",
    ("p4p.client.cothread.Context", "rpc"): "an rpc payload is arbitrary; limits cannot apply",
    ("pvaccess.Channel", "parsePut"): (
        "a list of JSON strings parsed against the channel's own structure — "
        "nothing says which string carries the field the limits are about"
    ),
    ("pvaccess.Channel", "parsePutGet"): (
        "a list of JSON strings parsed against the channel's own structure — "
        "nothing says which string carries the field the limits are about"
    ),
    ("tango.DeviceProxy", "command_inout"): (
        "a command is an action on the device, not a channel write: no address "
        "to look limits up under and no number to bound"
    ),
    ("tango.DeviceProxy", "command_inout_asynch"): (
        "a command is an action on the device, not a channel write: no address "
        "to look limits up under and no number to bound"
    ),
    ("tango.Connection", "command_inout"): (
        "the class that DEFINES the command; refusing here closes the unbound "
        "spelling and every other Connection subclass with it"
    ),
    ("tango.Connection", "command_inout_asynch"): (
        "the class that DEFINES the command; refusing here closes the unbound "
        "spelling and every other Connection subclass with it"
    ),
    ("tango.Group", "command_inout"): "a group command is an action on many devices",
    ("tango.Group", "command_inout_asynch"): "a group command is an action on many devices",
    ("tango.Group", "write_attribute"): (
        "a group write fans one value out to every device the group matched: "
        "no single channel to bound it under and no one device to step against"
    ),
    ("tango.Group", "write_attribute_asynch"): (
        "a group write fans one value out to every device the group matched: "
        "no single channel to bound it under and no one device to step against"
    ),
}

#: Entry points a limits check cannot be attached to at all. Listed so the
#: partition stays honest about them: a readonly run still refuses every one.
_LIMITS_UNWRAPPABLE: dict[tuple[str, str], str] = {
    ("epicscorelibs.ca.cadef", "ca_array_put"): (
        "the call carries a value already marshalled into C memory and an "
        "opaque channel handle: no number left to bound and no channel name "
        "to look limits up under"
    ),
    ("epicscorelibs.ca.cadef", "ca_array_put_callback"): (
        "the call carries a value already marshalled into C memory and an "
        "opaque channel handle: no number left to bound and no channel name "
        "to look limits up under"
    ),
    ("p4p.server.raw.SharedPV", "post"): "server side — serves a PV, writes no device",
    ("p4p.server.raw.SharedPV", "open"): "server side — serves a PV, writes no device",
    ("p4p.server.thread.SharedPV", "post"): "server side — serves a PV, writes no device",
    ("p4p.server.thread.SharedPV", "open"): "server side — serves a PV, writes no device",
    ("p4p.server.asyncio.SharedPV", "post"): "server side — serves a PV, writes no device",
    ("p4p.server.asyncio.SharedPV", "open"): "server side — serves a PV, writes no device",
    ("tango.DeviceProxy", "put_property"): "writes the Tango database, not a channel",
}

__all__ = [
    "READONLY_DENIED_IMPORTS",
    "_CLIENT_WRITE_TARGETS",
    "_ESCAPE_ROUTES",
    "_FRAMEWORK_WRITE_TARGETS",
    "_LIMITS_REFUSED",
    "_LIMITS_UNWRAPPABLE",
    "_LIMITS_WRAPPED",
    "_READONLY_WRITE_TARGETS",
]
