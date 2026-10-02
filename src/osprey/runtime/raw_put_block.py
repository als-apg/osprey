"""Refuse raw control-system writes inside a Python process.

The block has two modes, sharing one patching engine that takes a replacement
per mode.

``install("armed", ...)`` is the block of a run with writes on. A write reaches
the machine through a connector, which checks and records it, so every raw
client put a table names is replaced by one that calls through only while a
connector holds the write door open (``osprey_connectors``' ``write_door``,
imported on each call) and otherwise logs a warning and raises
``ChannelWriteBlockedError`` with reason ``RAW_CLIENT_WRITE``, naming the
channel the call was aimed at. Where ``osprey_connectors`` is not importable
there is no connector and no door, and the refusal is a stdlib ``RuntimeError``
carrying the marker the caller passed. The rpc and Tango command rows are
refused with their own texts when the caller asks, and left alone otherwise.
A provider row answers per channel: pvaPy's ``Channel`` speaks Channel Access
as well as PVAccess, so its constructor is wrapped to record the provider each
channel was opened on, and a put is refused like any raw put unless the channel
is a PVAccess one, which calls through untouched. Armed mode patches exactly
the rows it is passed: no escape route, no process spawning and no ``ctypes``
loader, all of which the readonly mode closes.

``install("readonly", ...)`` replaces every write entry point a table names with
a function that refuses, and closes every route out of Python that could reach
one. It is the runtime half of a readonly run: the pre-execution regex sees only
the standard spellings of a write, and ``from epics import caput as _w`` evades
it, so the refusal has to be in place before the user code runs.

This module is stdlib-only at module scope and importing it has no side effects.
Both properties are load-bearing: the executor embeds this file's *source* into
the script it runs and executes it in a private namespace, so the guard holds in
an interpreter where ``osprey`` itself cannot be imported, and no name it
defines leaks into the user code. The tables arrive as literal arguments for the
same reason — nothing here reads them from ``osprey``.

``install`` is re-entrant. A second call in the same process adds a second
finder that skips the first, wraps no loader twice, and refuses what the first
already refused; an armed replacement is recognised and never wrapped again.

The table covers three kinds of route: the control-system client libraries
themselves, the process-spawning surface that could shell out to ``caput``, and
``ctypes``, which reaches Channel Access without importing any client package
at all. The ``ctypes`` rows are gated rather than refused outright, because a
readonly run reads through ``osprey.runtime`` and its EPICS connector is
pyepics, which reaches Channel Access by loading ``libca`` through ``ctypes``: a
load is let through only while pyepics' own ``initialize_libca`` is on the
stack, and the handle it gets back has its put entry points refused, so the raw
route through the library is closed like every other spelling. Every other load
refuses, pyepics present or not.

Each patched attribute is also followed back to the module that defined it, so
a write a package merely re-exports refuses under both of its spellings. That
step covers the defining modules no table can enumerate — PyTango defines its
writes in ``tango.device_proxy`` and ``tango.connection`` and binds them onto
``DeviceProxy``, and every binding has its own such layout.

The two halves of the table are patched at different moments. The eager rows
(clients and escape routes) are resolved and refused at install, because a
readonly script may not import a client at all and because a client module may
load a shared library while it executes. The deferred rows (acquisition
frameworks) are refused when their module is imported, through a
``sys.meta_path`` finder that wraps the located loader, because those are the
rows a readonly script is allowed to import and a script that imports neither
should not pay for them. A framework already imported at install is patched on
the spot.
"""

import importlib
import inspect
import logging
import platform
import sys
import weakref
from collections.abc import Callable, Iterable, Mapping
from typing import Any

#: Rows whose attributes are shared-library loaders. They are gated, not refused.
_GATED_LOADER_ROWS = ("ctypes", "ctypes.LibraryLoader")

#: Put entry points closed on the handle a permitted ``libca`` load returns.
_HANDLE_PUT_SYMBOLS = ("ca_array_put", "ca_array_put_callback", "ca_sg_array_put")

#: Attribute every finder and loader this module installs carries, so a second
#: install recognises the first and neither delegates to the other.
_GUARD_ATTR = "_osprey_readonly_guard"

#: Attribute every replacement armed mode installs carries, so a second install,
#: a re-export or the pvaPy sweep finds it already in place and leaves it.
_ARMED_ATTR = "_osprey_armed_block"

#: Logger armed refusals are reported under. Spelled out, not ``__name__``: the
#: executor runs this source in a namespace that has no module name.
_LOGGER_NAME = "osprey.runtime.raw_put_block"

Rows = Iterable[tuple[str, Iterable[str]]]

#: ``replacement(dotted, attr, original, owner)`` -> what replaces *original*.
Replacement = Callable[[str, str, Any, Any], Any]


def _pyepics_is_loading(_sys: Any = sys, _max_depth: int = 8) -> bool:
    """True while ``epics.ca.initialize_libca`` is a near caller.

    Matched by code object, not by name, so a same-named function opens nothing.
    """
    _code = getattr(
        getattr(_sys.modules.get("epics.ca"), "initialize_libca", None),
        "__code__",
        None,
    )
    if _code is None:
        return False
    # Frame 0 is this function, frame 1 the gated loader.
    _frame = _sys._getframe(2)
    for _ in range(_max_depth):
        if _frame is None:
            return False
        if _frame.f_code is _code:
            return True
        _frame = _frame.f_back
    return False


def _resolve(dotted: str) -> Any:
    """Import the longest importable prefix of *dotted*, then walk attributes.

    One spelling for modules, module attributes and classes alike. Returns None
    when the target is not present, which is the ordinary case for most of the
    table — an uninstalled library, or an optional flavour of an installed one.
    That case has to stay SILENT: it is true on every ordinary deployment, and a
    warning per absent target would print on every readonly run.
    """
    parts = dotted.split(".")
    for _cut in range(len(parts), 0, -1):
        fullname = ".".join(parts[:_cut])
        try:
            module = importlib.import_module(fullname)
        except ImportError:
            continue
        # The longest importable prefix decides: a parent without the child
        # means the target does not exist here either, so no shorter prefix is
        # tried.
        return _walk(module, fullname, dotted)
    return None


def _walk(module: Any, fullname: str, dotted: str) -> Any:
    """Walk the rest of *dotted* off *module*; None when it is not there."""
    obj = module
    for _attr in dotted.split(".")[len(fullname.split(".")) :]:
        try:
            obj = getattr(obj, _attr)
        except AttributeError:
            return None
    return obj


def _as_address(value: Any) -> str | None:
    """A channel name, or a comma-joined list of them; None for anything else."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value or None
    if isinstance(value, list | tuple) and value:
        names = [_as_address(item) for item in value]
        if all(isinstance(name, str) for name in names):
            return ", ".join(names)  # type: ignore[arg-type]
    return None


def _chid_name(module_name: str, function: str, chid: Any) -> Any:
    """Ask the already-imported client module to name a channel handle."""
    namer = getattr(sys.modules.get(module_name), function, None)
    return namer(chid) if callable(namer) else None


def _call_or_value(value: Any) -> Any:
    return value() if callable(value) else value


def _find_channel(dotted: str, attr: str, args: tuple, kwargs: dict) -> Any:
    """The channel a refused client call names, spelled the way its client does."""
    root = dotted.split(".")[0]
    if dotted == "epics.ca":
        # put(chid, value, ...); sg_put(gid, chid, value, ...)
        chid = kwargs.get("chid", args[1] if attr == "sg_put" else args[0])
        return _chid_name("epics.ca", "name", chid)
    if dotted == "epicscorelibs.ca.cadef":
        # ca_array_put[_callback](type, count, chid, value, ...)
        return _chid_name(dotted, "ca_name", args[2])
    if dotted == "epics.PV":
        return args[0].pvname
    if root == "p4p":
        # Context.put(self, name, values, ...); the Context's own ``name`` is
        # the provider, not a channel.
        return kwargs.get("name", args[1] if len(args) > 1 else None)
    if dotted == "pvaccess.Channel":
        return args[0].getName()
    if dotted == "pvaccess.CaIoc":
        # putField/dbpf(name, value); iocInit and start name no record.
        return args[1] if len(args) > 1 else None
    if root == "pvaccess":
        # A MultiChannel keeps the names it was built from to itself.
        return None
    if root == "caproto":
        if dotted.endswith(".PV"):
            return args[0].name
        if dotted.endswith(".Batch"):
            return args[1].name
    if root in ("tango", "PyTango"):
        proxy = args[0]
        if dotted.endswith("AttributeProxy"):
            return _call_or_value(proxy.name)
        owner = proxy.dev_name() if hasattr(proxy, "dev_name") else proxy.get_name()
        target = args[1] if len(args) > 1 else kwargs.get("attr_name")
        attribute = target if isinstance(target, str) else getattr(target, "name", None)
        return f"{owner}/{attribute}" if isinstance(attribute, str) else owner
    # Module-level puts (pyepics caput, aioca, caproto's sync client, doocs4py)
    # take the channel first. Anything else: a bound object that names its
    # channel, else the first argument spelled like one.
    if args and not _as_address(args[0]):
        pvname = getattr(args[0], "pvname", None)
        if _as_address(pvname):
            return pvname
        for candidate in args[1:]:
            if _as_address(candidate):
                return candidate
    if args:
        return args[0]
    for keyword in ("pvname", "pv", "pv_name", "pvs", "name", "address"):
        if keyword in kwargs:
            return kwargs[keyword]
    return None


def _derive_address(dotted: str, attr: str, args: tuple, kwargs: dict) -> str:
    """The channel a refused put was aimed at, or ``<unknown>``.

    Never raises: the refusal must happen whatever the arguments look like.
    """
    try:
        found = _find_channel(dotted, attr, args, kwargs)
    except Exception:
        found = None
    return _as_address(found) or "<unknown>"


def _door_is_open() -> bool:
    """Whether a connector opened the write door around this call.

    Imported on each call, not at install: this module must stay importable,
    and installable, where ``osprey_connectors`` is not. Without it no
    connector exists to open the door, so the door is closed.
    """
    try:
        from osprey_connectors.control_system.write_door import door_is_open
    except Exception:
        return False
    return bool(door_is_open())


def _raw_write_refusal(marker: str, address: str) -> Exception:
    """The error a refused raw put raises.

    ``ChannelWriteBlockedError(RAW_CLIENT_WRITE)`` where ``osprey_connectors``
    is importable; a stdlib ``RuntimeError`` carrying *marker* where it is not,
    so a run without the package still refuses, recognisably.
    """
    try:
        from osprey_connectors.errors import (
            ChannelWriteBlockedError,
            raw_client_write_message,
        )
    except Exception:
        return RuntimeError(
            f"{marker}: write to '{address}' bypassed the reference monitor. "
            "Use osprey.runtime.write_channel(address, value) or "
            "osprey.runtime.write_channels({address: value, ...}) so limits and "
            "approval apply."
        )
    return ChannelWriteBlockedError(address, "RAW_CLIENT_WRITE", raw_client_write_message(address))


def _refusal_text(texts: Mapping[str, str], dotted: str, attr: str) -> str | None:
    """The text keyed by the longest dotted prefix of ``dotted.attr``."""
    parts = f"{dotted}.{attr}".split(".")
    for cut in range(len(parts), 0, -1):
        text = texts.get(".".join(parts[:cut]))
        if text:
            return text
    return None


def _preserving_descriptor(
    owner: Any, attr: str, original: Any, build: Callable[[Any], Any]
) -> Any:
    """Build a replacement for *original* that binds the way it did.

    A ``staticmethod`` or ``classmethod`` looked up on its class comes back as
    a plain or bound function; setting a plain function in its place would
    bind the instance as a first argument it never took. Such a row is
    rebuilt around the underlying function and re-wrapped in its descriptor.
    """
    if isinstance(owner, type):
        try:
            static = inspect.getattr_static(owner, attr)
        except AttributeError:
            static = None
        if isinstance(static, staticmethod | classmethod):
            if getattr(static.__func__, _ARMED_ATTR, False):
                return static
            return type(static)(build(static.__func__))
    if getattr(original, _ARMED_ATTR, False):
        return original
    return build(original)


def _patch_rows(
    label: str,
    eager_targets: Rows,
    deferred_targets: Rows,
    replacement: Replacement,
) -> None:
    """Replace every attribute the rows name, now or when its module imports.

    *replacement* is called as ``replacement(dotted, attr, original, owner)``
    for the attribute itself, for its defining module's spelling and for every
    verb of the pvaPy sweep, and returns what goes in the original's place.
    """

    def _patch_row(dotted: str, attrs: Iterable[str], obj: Any) -> None:
        """Replace every attribute of one row on the object it resolved to."""
        try:
            for attr in attrs:
                if not hasattr(obj, attr):
                    continue
                original = getattr(obj, attr)
                setattr(obj, attr, replacement(dotted, attr, original, obj))
                # A re-export leaves a second spelling behind that no table can
                # name for every binding. Follow the original back to the module
                # that defined it and replace it there too. The identity check is
                # what makes this safe to run generically: ``__module__`` and
                # ``__name__`` are metadata a decorator or a rebind can leave
                # pointing at a module holding something else entirely, and
                # replacing an attribute on a name match alone could silently
                # refuse an unrelated read.
                try:
                    home = sys.modules.get(getattr(original, "__module__", None))  # type: ignore[arg-type]
                    name = getattr(original, "__name__", None)
                    if (
                        home is not None
                        and isinstance(name, str)
                        and getattr(home, name, None) is original
                    ):
                        setattr(home, name, replacement(dotted, attr, original, home))
                except Exception as home_error:
                    # Secondary, best-effort step: the attribute the table names
                    # is already replaced. A failure here — an unhashable
                    # ``__module__``, a module ``__getattr__`` that raises — must
                    # name the attribute and let the REST of the row be patched,
                    # so it is caught here rather than at the row level.
                    print(f"⚠️  {label} ({dotted}.{attr}) defining-module step failed: {home_error}")
            # pvaPy spells one typed setter per scalar and array type
            # (putDouble, putScalarArray, ...). Enumerating them would go stale
            # against the binding; the prefix will not. Three writes sit outside
            # that prefix — asyncPut, parsePut and parsePutGet — and they reach
            # the machine exactly as the rest do, so they are swept with them.
            if dotted == "pvaccess.Channel":
                for attr in dir(obj):
                    if attr.startswith(("put", "asyncPut", "parsePut")):
                        setattr(obj, attr, replacement(dotted, attr, getattr(obj, attr), obj))
        except Exception as guard_error:
            # A target that cannot be patched must not stop the ones after it,
            # and the operator needs to know which one.
            print(f"⚠️  {label} ({dotted}) failed: {guard_error}")

    # The eager rows are resolved before the loader is gated, in table order,
    # because a client module may load a shared library while it executes:
    # ``epicscorelibs.ca.cadef`` loads ``libca`` through ``ctypes.CDLL`` in its
    # module body, and that row precedes the ``ctypes`` rows.
    for dotted, attrs in eager_targets:
        try:
            obj = _resolve(dotted)
        except Exception as guard_error:
            print(f"⚠️  {label} ({dotted}) failed: {guard_error}")
            continue
        if obj is None:
            continue
        _patch_row(dotted, attrs, obj)

    # The deferred rows wait for the script's own import. Each row is indexed
    # under every module prefix of its dotted name, so it is reconsidered as
    # each module on its path finishes executing; a key that never names a
    # module simply never fires.
    rows_by_module: dict[str, list[tuple[str, Iterable[str]]]] = {}
    for dotted, attrs in deferred_targets:
        parts = dotted.split(".")
        for cut in range(1, len(parts) + 1):
            rows_by_module.setdefault(".".join(parts[:cut]), []).append((dotted, attrs))

    def _patch_module(fullname: str, module: Any) -> None:
        for dotted, attrs in rows_by_module.get(fullname, ()):
            obj = _walk(module, fullname, dotted)
            if obj is not None:
                _patch_row(dotted, attrs, obj)

    class _OspreyGuardLoader:
        """Run the located loader, then replace the writes its module defines."""

        _osprey_readonly_guard = True

        def __init__(self, loader: Any) -> None:
            self._osprey_loader = loader

        def create_module(self, spec: Any) -> Any:
            return self._osprey_loader.create_module(spec)

        def exec_module(self, module: Any) -> None:
            self._osprey_loader.exec_module(module)
            try:
                _patch_module(module.__name__, module)
            except Exception as error:
                print(f"⚠️  {label} ({module.__name__}) failed: {error}")

        def __getattr__(self, name: str) -> Any:
            # Everything else a loader answers (get_code, is_package,
            # get_source, ...) is the located loader's answer.
            if name == "_osprey_loader":
                raise AttributeError(name)
            return getattr(self._osprey_loader, name)

    class _OspreyGuardFinder:
        """Wrap the loader of every module a deferred write target lives in."""

        _osprey_readonly_guard = True

        def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
            if fullname not in rows_by_module:
                return None
            for finder in sys.meta_path:
                # Every guard finder is skipped, not only this one: two guards
                # delegating to each other would never return.
                if getattr(finder, _GUARD_ATTR, False):
                    continue
                find_spec = getattr(finder, "find_spec", None)
                if find_spec is None:
                    continue
                spec = find_spec(fullname, path, target)
                if spec is not None:
                    break
            else:
                return None
            loader = spec.loader
            # A namespace package or a legacy loader carries no write target's
            # definition; wrapping it would change how the module is created.
            if (
                loader is not None
                and hasattr(loader, "exec_module")
                and not getattr(loader, _GUARD_ATTR, False)
            ):
                spec.loader = _OspreyGuardLoader(loader)
            return spec

    sys.meta_path.insert(0, _OspreyGuardFinder())  # type: ignore[arg-type]
    for name in list(sys.modules):
        if name in rows_by_module:
            module = sys.modules.get(name)
            if module is not None:
                _patch_module(name, module)


def _install_readonly(
    *,
    eager_targets: Rows,
    deferred_targets: Rows,
    refusal: str,
) -> None:
    """Refuse every write entry point in the tables, and every escape route."""
    # CPython resolves ``platform.uname().processor`` lazily, by shelling out to
    # ``uname -p`` on first read — and h5py reads it while ``import at``
    # initialises its type layer, so the subprocess refusal below would kill the
    # import of a pure-simulation library. Resolve it once now, while spawning
    # is still allowed; the cached value answers every later lookup without
    # touching subprocess.
    platform.processor()

    def _osprey_readonly_refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(refusal)

    def _gate_loader(_original: Callable[..., Any]) -> Callable[..., Any]:
        def _osprey_gated_load(*_args: Any, **_kwargs: Any) -> Any:
            if not _pyepics_is_loading():
                _osprey_readonly_refuse()
            _handle = _original(*_args, **_kwargs)
            try:
                for _symbol in _HANDLE_PUT_SYMBOLS:
                    setattr(_handle, _symbol, _osprey_readonly_refuse)
            except Exception:
                # A handle whose put symbols cannot be closed is not handed out
                # at all.
                _osprey_readonly_refuse()
            return _handle

        return _osprey_gated_load

    def _readonly_replacement(dotted: str, _attr: str, original: Any, _owner: Any) -> Any:
        if dotted in _GATED_LOADER_ROWS:
            return _gate_loader(original)
        return _osprey_readonly_refuse

    _patch_rows("readonly guard", eager_targets, deferred_targets, _readonly_replacement)


def _opened_on_pva(pva: Any, args: tuple) -> bool:
    """Whether a pvaPy constructor call that succeeded opened PVAccess.

    pvaPy takes the provider as an optional second positional argument, and
    no keyword, and defaults to PVAccess without one. *pva* is the binding's
    own ``PVA``, captured when the constructor was patched, so a script that
    rebinds ``pvaccess.PVA`` changes nothing here. The type is compared
    exactly because the binding accepts a subclass of its provider enum, and a
    subclass holding ``CA`` can claim to equal anything.
    """
    if len(args) == 1:
        return True
    return len(args) == 2 and type(args[1]) is type(pva) and args[1] == pva


def _install_armed(
    *,
    blocked_targets: Rows,
    rpc_targets: Rows,
    refuse_rpc: bool,
    marker: str,
    rpc_refusals: Mapping[str, str],
    ca_provider_targets: Rows = (),
) -> None:
    """Refuse raw client puts the door did not let through; optionally rpc."""
    if not isinstance(marker, str) or not marker:
        raise ValueError("armed mode needs a non-empty refusal marker")
    blocked = tuple((dotted, tuple(attrs)) for dotted, attrs in blocked_targets)
    rpc = tuple((dotted, tuple(attrs)) for dotted, attrs in rpc_targets) if refuse_rpc else ()

    # Every rpc row must know its refusal before anything is patched: a row
    # left without one would be a refusal that says nothing.
    rpc_texts: dict[tuple[str, str], str] = {}
    for dotted, attrs in rpc:
        for attr in attrs:
            text = _refusal_text(rpc_refusals, dotted, attr)
            if text is None:
                raise ValueError(f"armed mode has no rpc refusal text for {dotted}.{attr}")
            rpc_texts[(dotted, attr)] = text
    # A provider row is patched with its constructor, which is what records the
    # provider each channel was opened on.
    provider = tuple((dotted, ("__init__", *attrs)) for dotted, attrs in ca_provider_targets)
    provider_owners = {dotted for dotted, _attrs in provider}
    blocked_rows = {(dotted, attr) for dotted, attrs in blocked for attr in attrs}
    blocked_owners = {dotted for dotted, _attrs in blocked}
    logger = logging.getLogger(_LOGGER_NAME)

    # Channels opened on PVAccess, keyed by ``id`` and held weakly, so an entry
    # leaves with its channel. A lookup checks the entry IS the channel asked
    # about: keying by identity rather than by the channel itself means a
    # subclass's ``__eq__``/``__hash__`` cannot make one channel answer for
    # another.
    pva_channels: Any = weakref.WeakValueDictionary()

    def _is_pva(channel: Any) -> bool:
        return pva_channels.get(id(channel)) is channel

    def _recording_init(dotted: str, original: Any) -> Any:
        # Read now, while the client module is being patched and before the
        # script holds it.
        pva = getattr(sys.modules.get(dotted.split(".")[0]), "PVA", None)

        def _osprey_armed_init(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Forgotten first: a channel constructed again is trusted only for
            # the provider the new call names, and not at all if it fails.
            pva_channels.pop(id(self), None)
            result = original(self, *args, **kwargs)
            if _opened_on_pva(pva, args):
                pva_channels[id(self)] = self
            return result

        return _osprey_armed_init

    def _blocked(dotted: str, attr: str, original: Any) -> Any:
        def _refuse(args: tuple, kwargs: dict) -> Exception:
            error = _raw_write_refusal(marker, _derive_address(dotted, attr, args, kwargs))
            # Logged before it is raised: a refusal raised inside a client
            # callback can be swallowed by the client, and must still be seen.
            logger.warning("%s", error)
            return error

        if inspect.iscoroutinefunction(original):

            async def _osprey_armed_put(*args: Any, **kwargs: Any) -> Any:
                if _door_is_open():
                    return await original(*args, **kwargs)
                raise _refuse(args, kwargs)

        else:

            def _osprey_armed_put(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
                if _door_is_open():
                    return original(*args, **kwargs)
                raise _refuse(args, kwargs)

        return _osprey_armed_put

    def _provider_gated(dotted: str, attr: str, original: Any) -> Any:
        refuse = _blocked(dotted, attr, original)

        def _osprey_armed_put(*args: Any, **kwargs: Any) -> Any:
            if args and _is_pva(args[0]):
                return original(*args, **kwargs)
            return refuse(*args, **kwargs)

        return _osprey_armed_put

    def _refused_rpc(text: str) -> Any:
        def _osprey_armed_rpc_refuse(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError(text)

        return _osprey_armed_rpc_refuse

    def _armed_replacement(dotted: str, attr: str, original: Any, owner: Any) -> Any:
        rpc_text = rpc_texts.get((dotted, attr))
        by_provider = dotted in provider_owners
        # The owner fallback is the pvaPy sweep's: its verbs are blocked with
        # the row that named the class.
        if (
            rpc_text is None
            and not by_provider
            and (dotted, attr) not in blocked_rows
            and dotted not in blocked_owners
        ):
            return original

        def marked(function: Any) -> Any:
            if rpc_text is not None:
                new = _refused_rpc(rpc_text)
            elif by_provider and attr == "__init__":
                new = _recording_init(dotted, function)
            elif by_provider:
                new = _provider_gated(dotted, attr, function)
            else:
                new = _blocked(dotted, attr, function)
            setattr(new, _ARMED_ATTR, True)
            for meta in ("__name__", "__qualname__", "__doc__"):
                try:
                    setattr(new, meta, getattr(function, meta))
                except Exception:
                    pass
            return new

        return _preserving_descriptor(owner, attr, original, marked)

    # Every row is deferred: an armed script may import any client, so the
    # rows are replaced as their module finishes importing (or now, when it
    # already has), and a client the script never imports is never loaded.
    _patch_rows("armed raw-put block", (), blocked + rpc + provider, _armed_replacement)


def install(mode: str, **contract: Any) -> None:
    """Install the raw-put block in *mode*, in this process.

    Every row is a ``(dotted, attrs)`` pair. Failing to patch one target prints
    a warning naming it and moves on to the next; an absent target is silent.

    ``"readonly"`` takes ``eager_targets``, ``deferred_targets`` and
    ``refusal``. *eager_targets* are resolved and refused now; *deferred_targets*
    are refused when their module is imported (or now, when it already is). A
    refused call raises ``RuntimeError(refusal)``.

    ``"armed"`` takes ``blocked_targets``, ``rpc_targets``, ``refuse_rpc``,
    ``marker``, ``rpc_refusals`` and ``ca_provider_targets``. A blocked row
    calls through while a connector holds the write door open; otherwise it
    logs a warning and raises ``ChannelWriteBlockedError`` with reason
    ``RAW_CLIENT_WRITE`` — or, where
    ``osprey_connectors`` is not importable, ``RuntimeError`` carrying
    *marker*. With *refuse_rpc*, each rpc row raises ``RuntimeError`` with the
    text *rpc_refusals* keys under the longest dotted prefix of ``dotted.attr``;
    without it the rpc rows are left alone. An optional ``ca_provider_targets``
    names pvaPy-shaped classes whose puts are refused like a blocked row unless
    the channel was constructed on PVAccess, which calls through. Nothing else
    is touched.
    """
    if mode == "readonly":
        _install_readonly(**contract)
    elif mode == "armed":
        _install_armed(**contract)
    else:
        raise ValueError(f"unknown raw-put block mode: {mode!r}")
