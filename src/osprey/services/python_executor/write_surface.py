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
    ("epics.ca", ("put", "put_complete")),
    # --- EPICS Channel Access (aioca). Not an optional extra: ophyd-async's
    # [ca] backend pulls it into every environment OSPREY builds, so a readonly
    # script can reach ``aioca.caput`` with nothing else installed.
    ("aioca", ("caput", "caput_many")),
    # --- PVAccess (p4p): one client Context per concurrency flavour, plus the
    # server-side SharedPV, which puts values on the wire when it is opened or
    # posted to.
    ("p4p.client.thread.Context", ("put", "rpc")),
    ("p4p.client.asyncio.Context", ("put", "rpc")),
    ("p4p.client.cothread.Context", ("put", "rpc")),
    ("p4p.server.raw.SharedPV", ("post", "open")),
    ("p4p.server.thread.SharedPV", ("post", "open")),
    ("p4p.server.asyncio.SharedPV", ("post", "open")),
    # --- Channel Access (caproto) ---
    ("caproto.sync.client", ("write", "write_read")),
    ("caproto.threading.client.PV", ("write", "write_all")),
    ("caproto.threading.client.Batch", ("write",)),
    ("caproto.asyncio.client.PV", ("write",)),
    # --- PVAccess (pvaPy). Its ``Channel`` carries one typed setter per scalar
    # and array type, so the ``put`` prefix is swept dynamically by the guard
    # rather than enumerated here; ``put`` itself is listed so the table still
    # names the library.
    ("pvaccess.Channel", ("put", "putGet")),
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
            "write_pipe",
            "put_property",
            "command_inout",
            "command_inout_asynch",
        ),
    ),
    ("tango.AttributeProxy", ("write", "write_asynch", "write_read")),
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
    ("ophyd_async.core.SignalX", ("trigger", "execute")),
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
