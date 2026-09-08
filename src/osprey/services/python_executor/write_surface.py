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

The floor of that technique is the Python name. A handle that has already left
Python cannot be re-pointed: ``cadef.libca['ca_array_put']`` fetches a fresh
function pointer out of the shared library on every call. The immutable
C-extension type ``p4p._p4p.ClientOperation`` is the same kind of floor, and it
deliberately has **no row** in the table below. The type refuses ``setattr``, so
it cannot be patched; the two names that hand it out cannot be patched either
without breaking reads, because ``p4p/client/raw.py`` subclasses the type at
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
    # the half that holds: the module attributes are only re-pointable until
    # something re-fetches the function pointer out of ``libca`` itself.
    # pyepics is unaffected — its ``epics.ca.libca`` is ``None`` until
    # ``initialize_libca()`` runs.
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

__all__ = [
    "READONLY_DENIED_IMPORTS",
    "_CLIENT_WRITE_TARGETS",
    "_ESCAPE_ROUTES",
    "_FRAMEWORK_WRITE_TARGETS",
    "_READONLY_WRITE_TARGETS",
]
