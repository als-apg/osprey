"""Renderer for the runtime filesystem guard emitted into executed scripts.

PURPOSE: One emitter, two callers, two postures. The visualization sandbox
(``mcp_server/workspace/execution/sandbox_executor.py``) needs an **allowlist**
— nothing is reachable unless a root says so. The python executor
(``services/python_executor/execution/wrapper.py``) needs a **denylist** —
everything is reachable except writes into the render zone and the profile
source set. Those are the same guard with the verdict inverted, so they are
rendered from the same function rather than maintained as two divergent
inline blocks.

Why *emitted source* and not an importable guard: the guard runs in a spawned
child that must not depend on osprey being importable there. Everything below
the ``render_fs_guard`` return is therefore self-contained text — it imports
only ``builtins``/``io``/``os``/``shutil``/``pathlib`` and closes over nothing
from the parent. Roots are baked in as repr'd string tuples, resolved *by the
caller*: the parent knows the project layout, the child must never re-derive
it (a child that resolved its own roots could be pointed at different ones by
the very code the guard is meant to contain).

Two postures, spelled precisely:

``default_deny=True`` (sandbox)
    Reads are allowed under ``bypass_prefixes`` (the Python environment) and
    under ``read_roots`` (the project tree — readable data, refused writes).
    Reads *and* writes are allowed under ``permitted_roots``. Everything else
    raises ``PermissionError``.

``default_deny=False`` (executor)
    Everything is allowed except a write landing inside ``protected_roots``,
    which raises ``PermissionError``. ``permitted_roots`` carves exceptions
    *out of* the protected set, which is how the wrapper keeps writing its own
    metadata and artifacts under a protected parent.

``write_modes_only_targets`` exists for one specific reason: CPython's
``pathlib`` routes through ``io.open``, not ``builtins.open``. Patching
``io.open`` outright would change pathlib *read* behaviour, which the
characterization suite pins. Patching it for write modes only closes
``Path.write_text`` while leaving ``Path.read_text`` exactly as it is today.

LIMIT — defense in depth, not a security boundary
-------------------------------------------------
This guard runs **inside** the child interpreter, in the same module namespace
as the code it is guarding, and it leaves its own restore handle in that
namespace. Code that knows the guard is there can simply take it down::

    _restore_patched_targets()       # the guard's own uninstall, in scope
    open('bui' + 'ld/x', 'w')        # now completely unguarded

That is not a defect to be patched around — nothing rendered *into* the child
can be hidden from the child. It is pinned by a characterization test
(``tests/services/python_executor/test_fs_guard.py::TestTamperLimit``) so the
stance stays a documented fact rather than an assumption someone later mistakes
for a boundary.

What this guard *is* for is the honestly-spelled-around path: the concatenated,
computed or ``os.path``-assembled target that the static pre-execution walker
cannot see, and the ordinary accident where analysis code writes its output one
directory too high. What stops code that is deliberately attacking the boundary
is the operating system — the container's privilege split, where the render
zone and the profile sources are owned by a different user than the one running
agent code. Widening the patch set below raises the cost of an accident and
closes routes reached by mistake; it does not make the guard a sandbox.
"""

import textwrap
from collections.abc import Iterable
from pathlib import Path

# ``open``/``io.open`` mode characters that mean "this call may modify the
# file". Anything else ('r', 'rb', 'rt') is a read. Same set the static walker
# in ``path_policy`` uses, deliberately: a mode the pre-check calls a write
# must not be a read at runtime.
_WRITE_MODE_CHARS = "wax+"

# How each patchable entry point names the paths it may write.
#   "open"     -> (file, mode=...) — write-ness comes from the mode string
#   "os_open"  -> (path, flags)    — write-ness comes from the O_* flags
#   otherwise  -> always a write; the tuple gives (position, keyword aliases)
#                 for every argument naming a path the call may create,
#                 overwrite, or delete. Copy *sources* are reads and are not
#                 listed; ``move``/``rename``/``replace`` list both sides
#                 because they remove the original too. ``symlink``/``link``
#                 list ``dst`` only: the link entry being created is the write,
#                 while the source it points at is neither read nor touched.
_PATH_ARGS: dict[str, tuple[tuple[int, tuple[str, ...]], ...]] = {
    "os.truncate": ((0, ("path",)),),
    "os.remove": ((0, ("path",)),),
    "os.unlink": ((0, ("path",)),),
    "os.makedirs": ((0, ("name",)),),
    "os.mkdir": ((0, ("path",)),),
    "os.rmdir": ((0, ("path",)),),
    "os.removedirs": ((0, ("name",)),),
    "os.symlink": ((1, ("dst",)),),
    "os.link": ((1, ("dst",)),),
    "shutil.rmtree": ((0, ("path",)),),
    "os.rename": ((0, ("src",)), (1, ("dst",))),
    "os.replace": ((0, ("src",)), (1, ("dst",))),
    "shutil.move": ((0, ("src",)), (1, ("dst",))),
    "shutil.copy": ((1, ("dst",)),),
    "shutil.copy2": ((1, ("dst",)),),
    "shutil.copyfile": ((1, ("dst",)),),
}

# Entry points whose write-ness is decided by an argument rather than by being
# a write at all. Only these can be patched for write modes only.
_MODE_BEARING_TARGETS = ("builtins.open", "io.open")
_FLAG_BEARING_TARGETS = ("os.open",)

#: Every entry point the executor patches. Wide on purpose: ``builtins.open``
#: alone leaves ``os.remove``, ``shutil.rmtree`` and pathlib as open routes
#: into the render zone. Directory removal (``os.rmdir``/``os.removedirs``) and
#: link creation (``os.symlink``/``os.link``) are here for the same reason —
#: dropping ``build/`` and planting a new entry inside it are both writes to the
#: render zone, whichever call spells them. Absentees are skipped at install
#: time, so naming a platform-specific entry point here costs nothing.
EXECUTOR_PATCH_TARGETS: tuple[str, ...] = (
    "builtins.open",
    "io.open",
    "os.open",
    "os.truncate",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "os.removedirs",
    "os.rename",
    "os.replace",
    "os.makedirs",
    "os.mkdir",
    "os.symlink",
    "os.link",
    "shutil.copy",
    "shutil.copy2",
    "shutil.copyfile",
    "shutil.move",
    "shutil.rmtree",
)

#: What the visualization sandbox patches outright. ``io.open`` is *not* here
#: — it belongs in ``write_modes_only_targets`` so pathlib reads keep passing.
SANDBOX_PATCH_TARGETS: tuple[str, ...] = ("builtins.open",)

#: The sandbox's write-modes-only set: closes ``Path.write_text`` without
#: touching ``Path.read_text``.
SANDBOX_WRITE_MODES_ONLY_TARGETS: tuple[str, ...] = ("io.open",)

_SUPPORTED_TARGETS: tuple[str, ...] = (
    *_MODE_BEARING_TARGETS,
    *_FLAG_BEARING_TARGETS,
    *_PATH_ARGS,
)

_MODULE_ALIASES = {
    "builtins": "_osprey_builtins",
    "io": "_osprey_io",
    "os": "_osprey_os",
    "shutil": "_osprey_shutil",
}

# Refusal wording. The sandbox's two message shapes — a ``Sandbox:`` prefix
# with a "write denied" or "access denied" fragment — are pinned by
# tests/mcp_server/test_sandbox_guard_characterization.py, so they are
# reproduced here character-for-character up to the trailing explanation.
# The prefix is a parameter because the executor's refusals have to carry a
# marker its audit path scans for.
DEFAULT_ALLOWLIST_PREFIX = "Sandbox:"
DEFAULT_DENYLIST_PREFIX = "Protected path:"

_WRITE_DETAIL_ALLOWLIST = "Writes are only allowed in permitted directories."
_ACCESS_DETAIL_ALLOWLIST = "Only permitted and read-only directories are allowed."
_WRITE_DETAIL_DENYLIST = "This location is protected and cannot be modified by executed code."


def _root_tuple(roots: Iterable[str | Path]) -> tuple[str, ...]:
    """Normalize *roots* to a repr-able tuple of strings.

    No ``resolve()`` here on purpose: the caller resolves in the parent, where
    the real layout is known. Resolving a second time would be harmless for a
    good caller and would silently paper over a bad one.
    """
    return tuple(str(root) for root in roots)


def render_fs_guard(
    *,
    default_deny: bool,
    permitted_roots: Iterable[str | Path],
    protected_roots: Iterable[str | Path],
    read_roots: Iterable[str | Path],
    bypass_prefixes: Iterable[str] = (),
    patch_targets: Iterable[str] = EXECUTOR_PATCH_TARGETS,
    write_modes_only_targets: Iterable[str] = (),
    refusal_prefix: str | None = None,
) -> str:
    """Render the filesystem guard as source code to embed in a child script.

    :param default_deny: ``True`` for the sandbox's allowlist posture,
        ``False`` for the executor's denylist posture.
    :param permitted_roots: Read *and* write allowed. Checked first in both
        postures, so it also carves exceptions out of ``protected_roots``.
    :param protected_roots: Denylist posture only — writes landing here are
        refused.
    :param read_roots: Allowlist posture only — readable, writes refused.
    :param bypass_prefixes: Markers matched by *containment* against the
        resolved path; a hit allows the call in any mode. The sandbox passes
        ``('site-packages', 'lib/python', sys.prefix)``, which mixes substring
        markers with a true prefix — containment subsumes both, and the only
        paths it additionally admits embed the venv prefix mid-path.
    :param patch_targets: Dotted names to rebind outright.
    :param write_modes_only_targets: Mode-bearing dotted names to rebind for
        write modes only; reads pass through untouched.
    :param refusal_prefix: Leading text on every refusal. Defaults to
        ``Sandbox:`` under ``default_deny`` and ``Protected path:`` otherwise.
    :returns: Left-aligned Python source. Indent it yourself
        (``textwrap.indent``) if it lands inside a block.
    :raises ValueError: On an unknown target, a non-mode-bearing name in
        ``write_modes_only_targets``, a name in both target sets, or an empty
        ``refusal_prefix``.
    """
    patch_targets = tuple(patch_targets)
    write_modes_only_targets = tuple(write_modes_only_targets)

    unknown = [
        t for t in (*patch_targets, *write_modes_only_targets) if t not in _SUPPORTED_TARGETS
    ]
    if unknown:
        raise ValueError(
            f"unsupported fs guard target(s): {unknown!r}. Supported: {list(_SUPPORTED_TARGETS)!r}"
        )
    not_mode_bearing = [t for t in write_modes_only_targets if t not in _MODE_BEARING_TARGETS]
    if not_mode_bearing:
        raise ValueError(
            f"write_modes_only_targets may only name mode-bearing entry points "
            f"{list(_MODE_BEARING_TARGETS)!r}; got {not_mode_bearing!r}. Everything else "
            f"({', '.join(sorted(_PATH_ARGS))}, os.open) is a write by construction."
        )
    both = [t for t in write_modes_only_targets if t in patch_targets]
    if both:
        raise ValueError(
            f"target(s) {both!r} appear in both patch_targets and write_modes_only_targets; "
            f"a name can only be rebound once."
        )

    prefix = refusal_prefix
    if prefix is None:
        prefix = DEFAULT_ALLOWLIST_PREFIX if default_deny else DEFAULT_DENYLIST_PREFIX
    if not prefix.strip():
        raise ValueError("refusal_prefix must be non-empty — refusals are matched by their prefix")

    write_detail = _WRITE_DETAIL_ALLOWLIST if default_deny else _WRITE_DETAIL_DENYLIST

    # (dotted, kind, path-arg spec, write-modes-only) — one row per rebound
    # name, consumed by a single loop in the emitted code.
    table: list[tuple[str, str, tuple[tuple[int, tuple[str, ...]], ...], bool]] = []
    for dotted in (*patch_targets, *write_modes_only_targets):
        write_only = dotted in write_modes_only_targets
        if dotted in _MODE_BEARING_TARGETS:
            kind = "open"
        elif dotted in _FLAG_BEARING_TARGETS:
            kind = "os_open"
        else:
            kind = "paths"
        table.append((dotted, kind, _PATH_ARGS.get(dotted, ()), write_only))

    modules_literal = ", ".join(f'"{name}": {alias}' for name, alias in _MODULE_ALIASES.items())

    return textwrap.dedent(
        f'''
        # --- OSPREY filesystem guard (generated by render_fs_guard) ----------
        # Self-contained: imports nothing from osprey, because the child that
        # runs this may not have osprey importable at all.
        import builtins as _osprey_builtins
        import io as _osprey_io
        import os as _osprey_os
        import shutil as _osprey_shutil
        from pathlib import Path as _OspreyGuardPath

        _OSPREY_FS_DEFAULT_DENY = {default_deny!r}
        _OSPREY_FS_PERMITTED = {_root_tuple(permitted_roots)!r}
        _OSPREY_FS_PROTECTED = {_root_tuple(protected_roots)!r}
        _OSPREY_FS_READ_ROOTS = {_root_tuple(read_roots)!r}
        _OSPREY_FS_BYPASS = {tuple(bypass_prefixes)!r}
        _OSPREY_FS_PREFIX = {prefix!r}
        _OSPREY_FS_WRITE_DETAIL = {write_detail!r}
        _OSPREY_FS_ACCESS_DETAIL = {_ACCESS_DETAIL_ALLOWLIST!r}
        _OSPREY_FS_WRITE_MODE_CHARS = {_WRITE_MODE_CHARS!r}
        _OSPREY_FS_TARGETS = {tuple(table)!r}
        _OSPREY_FS_MODULES = {{{modules_literal}}}
        # A write for os.open purposes: any flag that can create, extend or
        # shorten the file. O_EXCL implies O_CREAT and needs no entry.
        _OSPREY_FS_OS_WRITE_FLAGS = (
            _osprey_os.O_WRONLY
            | _osprey_os.O_RDWR
            | _osprey_os.O_CREAT
            | _osprey_os.O_APPEND
            | _osprey_os.O_TRUNC
        )

        # Originals, captured BEFORE the corresponding name is rebound. This is
        # also the restore table: a name absent here was never patched.
        _osprey_fs_originals = {{}}


        def _osprey_fs_is_write_mode(mode):
            """True when an open() mode string may modify the file."""
            try:
                mode = str(mode)
            except Exception:
                # An unreadable mode is treated as a write: the guard errs
                # toward refusing, never toward letting an unknown call past.
                return True
            return any(_ch in mode for _ch in _OSPREY_FS_WRITE_MODE_CHARS)


        def _osprey_fs_under(path, roots):
            for _root in roots:
                try:
                    path.relative_to(_root)
                    return True
                except ValueError:
                    continue
            return False


        def _osprey_fs_check(candidate, is_write):
            """Refuse *candidate* or return quietly. The whole policy is here."""
            try:
                path = _OspreyGuardPath(candidate).resolve()
            except (TypeError, ValueError):
                # Not a filesystem path — an already-open file descriptor, for
                # instance. There is no path to judge, so there is nothing to
                # refuse; the call goes through to the real function.
                return
            path_str = str(path)
            for _marker in _OSPREY_FS_BYPASS:
                if _marker and _marker in path_str:
                    return
            if _osprey_fs_under(path, _OSPREY_FS_PERMITTED):
                return
            if _OSPREY_FS_DEFAULT_DENY:
                if _osprey_fs_under(path, _OSPREY_FS_READ_ROOTS):
                    if is_write:
                        raise PermissionError(
                            f"{{_OSPREY_FS_PREFIX}} write denied for '{{path}}'. "
                            f"{{_OSPREY_FS_WRITE_DETAIL}}"
                        )
                    return
                raise PermissionError(
                    f"{{_OSPREY_FS_PREFIX}} access denied for '{{path}}'. "
                    f"{{_OSPREY_FS_ACCESS_DETAIL}}"
                )
            if is_write and _osprey_fs_under(path, _OSPREY_FS_PROTECTED):
                raise PermissionError(
                    f"{{_OSPREY_FS_PREFIX}} write denied for '{{path}}'. "
                    f"{{_OSPREY_FS_WRITE_DETAIL}}"
                )


        def _osprey_fs_capture(dotted):
            _module_name, _attr = dotted.rsplit(".", 1)
            _original = getattr(_OSPREY_FS_MODULES[_module_name], _attr)
            _osprey_fs_originals[dotted] = _original
            return _original


        def _osprey_fs_wrap_open(dotted, write_modes_only):
            _original = _osprey_fs_capture(dotted)

            def _osprey_guarded_open(file, mode="r", *args, **kwargs):
                _is_write = _osprey_fs_is_write_mode(mode)
                if _is_write or not write_modes_only:
                    _osprey_fs_check(file, _is_write)
                return _original(file, mode, *args, **kwargs)

            return _osprey_guarded_open


        def _osprey_fs_wrap_os_open(dotted):
            _original = _osprey_fs_capture(dotted)

            def _osprey_guarded_os_open(path, flags, *args, **kwargs):
                _osprey_fs_check(path, bool(flags & _OSPREY_FS_OS_WRITE_FLAGS))
                return _original(path, flags, *args, **kwargs)

            return _osprey_guarded_os_open


        def _osprey_fs_wrap_paths(dotted, spec):
            _original = _osprey_fs_capture(dotted)

            def _osprey_guarded_paths(*args, **kwargs):
                for _position, _names in spec:
                    if len(args) > _position:
                        _osprey_fs_check(args[_position], True)
                        continue
                    for _name in _names:
                        if _name in kwargs:
                            _osprey_fs_check(kwargs[_name], True)
                            break
                return _original(*args, **kwargs)

            return _osprey_guarded_paths


        def _install_patched_targets():
            for _dotted, _kind, _spec, _write_only in _OSPREY_FS_TARGETS:
                _module_name, _attr = _dotted.rsplit(".", 1)
                _module = _OSPREY_FS_MODULES[_module_name]
                if not hasattr(_module, _attr):
                    # Platform-specific absentee (os.truncate on some
                    # systems). Nothing to patch, nothing to restore.
                    continue
                if _kind == "open":
                    _replacement = _osprey_fs_wrap_open(_dotted, _write_only)
                elif _kind == "os_open":
                    _replacement = _osprey_fs_wrap_os_open(_dotted)
                else:
                    _replacement = _osprey_fs_wrap_paths(_dotted, _spec)
                setattr(_module, _attr, _replacement)


        def _restore_patched_targets():
            """Put every rebound name back. Idempotent, and safe in a finally."""
            for _dotted, _original in list(_osprey_fs_originals.items()):
                _module_name, _attr = _dotted.rsplit(".", 1)
                setattr(_OSPREY_FS_MODULES[_module_name], _attr, _original)
            _osprey_fs_originals.clear()


        _install_patched_targets()
        # --- end OSPREY filesystem guard ------------------------------------
        '''
    ).lstrip("\n")
