"""Static path policy for user-submitted code — mode-independent write refusal.

PURPOSE: Catch *literal* write intent aimed at the render zone or the profile
source set before the code is ever spawned, so the agent gets a readable
refusal instead of a runtime traceback. This is a pre-check, not the
enforcement boundary: the boundary is the emitted runtime guard
(``fs_guard.render_fs_guard``), which patches the filesystem entry points in
the child process and therefore sees the paths this walker cannot.

Two things are deliberately *deferred* to that runtime guard rather than
guessed at here:

* **Dynamic paths.** ``open(target, 'w')`` carries no literal to resolve. No
  issue is emitted — a static walker that guessed would either refuse
  legitimate work or give false comfort.
* **Dynamic modes.** ``open('build/config.yml', mode)`` may well be a read.
  The runtime guard refuses it if it turns out to be a write, so nothing is
  lost by staying quiet.

Matching is by *function name* (``.rmtree``, ``.makedirs``, ``os.unlink``),
not by the module the caller happened to bind, so ``import shutil as sh`` is
still seen. That breadth is safe because an issue also requires the resolved
literal path to land inside a protected root; aliasing games that defeat name
matching entirely (``getattr(shutil, 'rmtree')``) fall through to the runtime
guard like any other dynamic call. The exception is the names ordinary
non-path types share — ``remove``, ``replace``, ``rename``, ``copy``,
``move`` — which a list or a string answers to as readily as ``os`` does:
those are matched only when the receiver is known to be ``os``, ``shutil``
(``posix``, or an alias any of them was imported under), or a ``Path(...)``
expression, so ``names.remove('profile.yml')`` stays clean.

A relative literal is resolved against the child's working directory, which
the caller passes in; existence is never required. The calls that remove or
relocate what a path names — the deletes, both sides of a rename, replace or
move, a link's destination — are also refused when the literal *contains* a
protected root, as the runtime guard refuses them.

The policy is applied in **every** execution mode. Readonly and readwrite runs
alike may not write into the render zone or the profile sources — that is the
self-change boundary, not a control-system write gate.

The walker carries two checks that are *not* about paths, both aimed at code
that takes the runtime guard down rather than spelling a path around it. The
guard cannot police either one — nothing rendered into the child can be hidden
from the child, and neither attempt calls a patched entry point on the way — so
a static layer, here, before the child is spawned, is the only one that can
refuse them:

* **Naming the guard's own identifiers** (``_restore_patched_targets``,
  ``_osprey_fs_check``, the ``_OSPREY_FS_*`` tables).
* **Reloading a module the guard patched** (``importlib.reload(os)``). A reload
  re-executes the module and restores its original attributes, taking the
  guard's rebindings with them — a plainer spelling of the same disarm.

Unlike the path checks these do not defer dynamic spellings, because a string
literal handed to ``getattr`` or to ``sys.modules[...]`` is the normal way both
are attempted rather than an edge case. What neither check reaches is a binding
it cannot see the name of — ``import os as _o; reload(_o)`` reads as a reload of
something unguarded. That is the honest limit of a static layer: it raises the
cost of a disarm, it does not make one impossible. The boundary that holds
against code deliberately attacking it is the operating system's — the
container's privilege split, where the render zone and the profile sources
belong to a different user than the one running agent code.
"""

import ast
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

from osprey.services.python_executor.execution.fs_guard import GUARDED_MODULES

# ``open``/``Path.open`` mode characters that mean "this call may modify the
# file". Anything else ('r', 'rb', 'rt') is a read and is never an issue.
_WRITE_MODE_CHARS = frozenset("wax+")

# Write-intent call names, mapped to ``(position, contains)`` per argument that
# names a path the call may create, overwrite, or delete. Sources of a copy are
# reads and are not listed; ``move`` and ``rename``/``replace`` list both sides
# because they remove the original as well; ``symlink``/``link`` list the link
# being created. ``contains`` marks a call that removes or relocates what the
# path names, so a literal holding a protected root is refused too.
_PATH_ARG_POSITIONS: dict[str, tuple[tuple[int, bool], ...]] = {
    # shutil
    "copy": ((1, False),),
    "copy2": ((1, False),),
    "copyfile": ((1, False),),
    "move": ((0, True), (1, True)),
    "rmtree": ((0, True),),
    # os
    "remove": ((0, True),),
    "unlink": ((0, True),),
    "rmdir": ((0, True),),
    "removedirs": ((0, True),),
    "rename": ((0, True), (1, True)),
    "replace": ((0, True), (1, True)),
    "symlink": ((1, True),),
    "link": ((1, True),),
    "makedirs": ((0, False),),
    "mkdir": ((0, False),),
    "truncate": ((0, False),),
}

# Call names that lists, strings, dicts and other non-path types answer to as
# well (``list.remove``, ``str.replace``, ``dict.copy``). Matched only when the
# receiver is known to be a path module or a path object.
_SHARED_CALL_NAMES = frozenset({"remove", "replace", "rename", "copy", "move"})

# The modules whose functions take paths under the shared names above.
_PATH_MODULES = frozenset({"os", "shutil", "posix"})

# ``pathlib`` constructors: a call to one of these is a path object.
_PATH_CONSTRUCTORS = frozenset({"Path", "PurePath", "PosixPath", "WindowsPath"})

# ``Path`` methods that act on the receiver's own path, by ``contains``: the
# deletes and relocations remove what it names, ``mkdir``/``touch`` create it.
_PATH_RECEIVER_METHODS: dict[str, bool] = {
    "unlink": True,
    "rmdir": True,
    "rename": True,
    "replace": True,
    "move": True,
    "mkdir": False,
    "touch": False,
}

# ``Path`` methods whose first argument is a destination path, by ``contains``.
_PATH_DESTINATION_METHODS: dict[str, bool] = {
    "rename": True,
    "replace": True,
    "move": True,
    "copy": False,
}

#: One write target of a call: ``(literal path, call spelling, contains)``.
_WriteTarget = tuple[str, str, bool]

# ``pathlib.Path`` methods that write unconditionally — no mode to inspect.
_PATH_WRITE_METHODS = frozenset({"write_text", "write_bytes"})

# Receivers for which ``.open(...)`` has the builtin signature (path, mode).
_BUILTIN_OPEN_RECEIVERS = frozenset({"io"})

# Identifier prefixes owned by the runtime sandbox guard. ``fs_guard``'s
# preamble defines the ``_OSPREY_FS_*`` policy tables and the ``_osprey_fs_*``
# helpers that read them; nothing a user submits has a reason to name one.
# Matched as a *prefix of a whole identifier* and case-sensitively, so the
# guard's real names (``_osprey_fs_check``, ``_OSPREY_FS_PROTECTED``) are all
# covered while an ordinary variable that merely contains the text
# (``my_osprey_fsx``) is not.
_GUARD_IDENTIFIER_PREFIXES = ("_osprey_fs", "_OSPREY_FS")

# Guard identifiers that share no prefix with the others; matched exactly.
_GUARD_IDENTIFIER_NAMES = frozenset({"_restore_patched_targets"})

# The call that puts a patched module back the way it was. Matched by bare
# name like the write-intent calls above, so ``importlib.reload``, a bare
# ``reload`` imported from importlib, and the legacy ``imp.reload`` all read
# the same. A bare ``reload`` call is only an issue when its arguments name a
# module the guard actually patches, which is what keeps an unrelated
# ``config.reload()`` clean.
_RELOAD_CALL_NAME = "reload"
_GUARDED_MODULE_NAMES = frozenset(GUARDED_MODULES)


def path_policy_issues(
    code: str,
    *,
    protected_roots: Iterable[str | os.PathLike[str]],
    working_directory: str | os.PathLike[str],
    permitted_roots: Iterable[str | os.PathLike[str]] = (),
) -> list[str]:
    """Return an issue per literal write aimed at a protected root.

    Args:
        code: The user-submitted source to analyse.
        protected_roots: Absolute paths (files or directories) that user code
            may not write into, in any execution mode. Resolved by the caller
            in the parent process — never re-derived here.
        working_directory: The directory the child runs in; a relative
            literal is resolved against it, as the child would resolve it.
        permitted_roots: Absolute paths carved back out of the protected set
            (the agent's own data zone, the execution folder). A path inside
            one of these is clean even when it also sits inside a protected
            root.

    Returns:
        Human-readable issue strings; empty means nothing statically
        objectionable was found. Syntax errors are the business of
        ``check_syntax`` — this walker stays quiet on them.
    """
    protected = tuple(_resolve_root(root) for root in protected_roots)
    permitted = tuple(_resolve_root(root) for root in permitted_roots)
    cwd = _resolve_root(working_directory)

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    # Guard tampering is independent of the root sets — there is no path to
    # compare, and a deployment that protects nothing still owns its guard.
    issues: list[str] = _guard_tamper_issues(tree) + _reload_tamper_issues(tree)
    if not protected:
        return issues

    receivers = _path_module_names(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for raw, spelling, contains in _write_targets(node, receivers):
            path = _anchor(raw, cwd)
            if any(path.is_relative_to(root) for root in permitted):
                continue
            hit = _protected_hit(path, protected)
            if hit is not None:
                issues.append(
                    f"Write to '{raw}' via {spelling} is not allowed: "
                    f"the path is inside the protected location '{hit}'"
                )
                continue
            held = _contained_root(path, protected) if contains else None
            if held is not None:
                issues.append(
                    f"Write to '{raw}' via {spelling} is not allowed: "
                    f"the path contains the protected location '{held}'"
                )
    return issues


def _path_module_names(tree: ast.AST) -> dict[str, str | None]:
    """Names that are known to reach a path module, and what each one is.

    The module names themselves and every alias one of them is imported under
    (``import shutil as sh``) map to ``None``; every function imported out of
    one maps to the function's own name (``from os import replace as swap``
    maps ``swap`` to ``replace``). A shared call name spelled through any of
    them is a path call, and the same name on anything else is not.
    """
    names: dict[str, str | None] = dict.fromkeys(_PATH_MODULES)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _PATH_MODULES:
                    names[alias.asname or alias.name] = None
        elif isinstance(node, ast.ImportFrom) and node.module in _PATH_MODULES:
            for alias in node.names:
                names[alias.asname or alias.name] = alias.name
    return names


def _guard_tamper_issues(tree: ast.AST) -> list[str]:
    """Return an issue per reference to a sandbox-guard identifier.

    One issue per distinct name, in source order, so a script that touches the
    guard three ways is reported three ways rather than once per occurrence.
    """
    issues: list[str] = []
    seen: set[str] = set()
    for node in ast.walk(tree):
        for name in _identifiers(node):
            if name in seen or not _is_guard_identifier(name):
                continue
            seen.add(name)
            issues.append(
                f"Reference to '{name}' is not allowed: it names an internal of the "
                "sandbox guard, and reading, rebinding or restoring the guard from "
                "user code is tampering with the boundary that constrains that code"
            )
    return issues


def _reload_tamper_issues(tree: ast.AST) -> list[str]:
    """Return an issue per guarded module a ``reload`` call names.

    One issue per distinct module, so a script that reloads ``os`` twice is
    reported once and a script that reloads ``os`` and ``io`` is reported
    twice.
    """
    issues: list[str] = []
    seen: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _called_name(node.func) != _RELOAD_CALL_NAME:
            continue
        for module in _reloaded_guarded_modules(node):
            if module in seen:
                continue
            seen.add(module)
            issues.append(
                f"Reloading the module '{module}' is not allowed: a reload re-executes "
                "the module and restores its original attributes, which takes the runtime "
                "filesystem guard's patches down with them, and disarming the guard from "
                "user code is tampering with the boundary that constrains that code"
            )
    return issues


def _reloaded_guarded_modules(node: ast.Call) -> list[str]:
    """Guarded module names spelled anywhere inside a reload call's arguments.

    The whole argument subtree is inspected rather than just a bare ``Name``,
    so ``reload(sys.modules['os'])`` is read the same as ``reload(os)`` — the
    string constant is the module name either way.
    """
    names: list[str] = []
    for arg in (*node.args, *(keyword.value for keyword in node.keywords)):
        for inner in ast.walk(arg):
            for name in _identifiers(inner):
                if name in _GUARDED_MODULE_NAMES and name not in names:
                    names.append(name)
    return names


def _is_guard_identifier(name: str) -> bool:
    """True when *name* is one the emitted sandbox guard owns.

    Case-sensitive, and anchored at the start of the identifier: the guard's
    names all begin with one of :data:`_GUARD_IDENTIFIER_PREFIXES`, while a
    variable that merely embeds the text (``my_osprey_fsx``) does not.
    """
    return name in _GUARD_IDENTIFIER_NAMES or name.startswith(_GUARD_IDENTIFIER_PREFIXES)


def _identifiers(node: ast.AST) -> tuple[str, ...]:
    """Every identifier-shaped string one AST node spells.

    Covers the binding and reference forms (``Name``, ``Attribute``, defs,
    parameters, ``global``/``nonlocal``, imports, ``except ... as``) *and*
    string constants that are themselves identifiers — ``getattr(m, '_osprey_fs_check')``
    and ``globals()['_restore_patched_targets']`` are the ordinary spellings of
    this attempt, not exotic ones. Requiring ``str.isidentifier()`` keeps prose
    and docstrings that happen to mention a guard name out of the match.
    """
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        return (node.attr,)
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return (node.name,)
    if isinstance(node, ast.arg):
        return (node.arg,)
    if isinstance(node, ast.keyword):
        return (node.arg,) if node.arg else ()
    if isinstance(node, ast.Global | ast.Nonlocal):
        return tuple(node.names)
    if isinstance(node, ast.alias):
        return tuple(part for part in (node.name, node.asname) if part)
    if isinstance(node, ast.ExceptHandler):
        return (node.name,) if node.name else ()
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,) if node.value.isidentifier() else ()
    return ()


def _write_targets(node: ast.Call, receivers: dict[str, str | None]) -> list[_WriteTarget]:
    """Return ``(literal path, call spelling, contains)`` per write target of one call.

    *receivers* are the names known to call a path module
    (:func:`_path_module_names`); ``contains`` marks a target the call removes
    or relocates.
    """
    func = node.func
    name = _called_name(func)
    if name is None:
        return []
    if isinstance(func, ast.Name) and (imported := receivers.get(func.id)):
        name = imported

    # open(...) / io.open(...) — path first, mode second.
    builtin_open = isinstance(func, ast.Name) and name == "open"
    module_open = (
        isinstance(func, ast.Attribute)
        and name == "open"
        and isinstance(func.value, ast.Name)
        and func.value.id in _BUILTIN_OPEN_RECEIVERS
    )
    if builtin_open or module_open:
        if not _has_write_mode(node, mode_index=1):
            return []
        target = _positional(node, 0)
        for keyword in node.keywords:
            if keyword.arg == "file":  # open(file='...', mode='w')
                target = keyword.value
        path = _literal_path(target)
        return [(path, "open()", False)] if path else []

    if isinstance(func, ast.Attribute):
        receiver = _literal_path(func.value)
        if receiver:
            # Path('...').write_text(...) / .write_bytes(...)
            if name in _PATH_WRITE_METHODS:
                return [(receiver, f"Path.{name}()", False)]
            # Path('...').open('w') — mode is the first argument here.
            if name == "open":
                if not _has_write_mode(node, mode_index=0):
                    return []
                return [(receiver, "Path.open()", False)]
        if _is_path_object(func.value):
            return _path_method_targets(node, name, receiver)

    if name in _SHARED_CALL_NAMES and not _is_path_module_call(func, receivers):
        return []
    positions = _PATH_ARG_POSITIONS.get(name)
    if positions is None:
        return []
    targets: list[_WriteTarget] = []
    for index, contains in positions:
        path = _literal_path(_positional(node, index))
        if path:
            targets.append((path, f"{name}()", contains))
    return targets


def _path_method_targets(node: ast.Call, name: str, receiver: str | None) -> list[_WriteTarget]:
    """Write targets of a method called on a ``Path(...)`` expression."""
    targets: list[_WriteTarget] = []
    if receiver and name in _PATH_RECEIVER_METHODS:
        targets.append((receiver, f"Path.{name}()", _PATH_RECEIVER_METHODS[name]))
    if name in _PATH_DESTINATION_METHODS:
        destination = _literal_path(_positional(node, 0))
        for keyword in node.keywords:
            if keyword.arg in {"target", "dst"}:
                destination = _literal_path(keyword.value)
        if destination:
            targets.append((destination, f"Path.{name}()", _PATH_DESTINATION_METHODS[name]))
    return targets


def _is_path_object(node: ast.expr) -> bool:
    """True for a ``Path(...)`` call or a ``/`` join with one on the left.

    A bare string is not a path object: ``'a,b'.replace(...)`` is a string
    method, whatever the string spells.
    """
    if isinstance(node, ast.Call):
        return _called_name(node.func) in _PATH_CONSTRUCTORS
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _is_path_object(node.left)
    return False


def _is_path_module_call(func: ast.expr, receivers: dict[str, str | None]) -> bool:
    """True when *func* is a path module's function, by a name known to be one."""
    if isinstance(func, ast.Attribute):
        return isinstance(func.value, ast.Name) and func.value.id in receivers
    return isinstance(func, ast.Name) and func.id in receivers


def _called_name(func: ast.expr) -> str | None:
    """Return the bare name a call spells, ignoring the module it is bound to."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _positional(node: ast.Call, index: int) -> ast.expr | None:
    """Return positional argument ``index``, or None (starred args included)."""
    plain = [arg for arg in node.args if not isinstance(arg, ast.Starred)]
    if len(plain) != len(node.args):
        return None  # a *args splat makes positions unknowable
    return plain[index] if index < len(plain) else None


def _has_write_mode(node: ast.Call, *, mode_index: int) -> bool:
    """True when the call's mode literal implies modification.

    A missing mode means the default 'r'. A *non-literal* mode is deferred to
    the runtime guard, which sees the real value.
    """
    mode_node: ast.expr | None = _positional(node, mode_index)
    for keyword in node.keywords:
        if keyword.arg == "mode":
            mode_node = keyword.value
    if mode_node is None:
        return False
    if not (isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str)):
        return False
    return bool(_WRITE_MODE_CHARS & set(mode_node.value))


def _literal_path(node: ast.expr | None) -> str | None:
    """Resolve a path expression to its literal string, or None if dynamic.

    Understands the spellings that carry a literal all the way through:
    ``'a/b'``, ``Path('a', 'b')``, ``pathlib.Path('a')`` and ``Path('a') / 'b'``.
    """
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Call):
        if _called_name(node.func) in _PATH_CONSTRUCTORS:
            parts = [_literal_path(arg) for arg in node.args]
            if parts and all(part is not None for part in parts):
                return os.path.join(*parts)  # type: ignore[arg-type]
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _literal_path(node.left)
        right = _literal_path(node.right)
        if left is not None and right is not None:
            return os.path.join(left, right)
    return None


def _resolve_root(root: str | os.PathLike[str]) -> Path:
    return Path(root).expanduser().resolve()


def _anchor(raw: str, cwd: Path) -> Path:
    """Resolve the literal *raw* the way the child would, from *cwd*."""
    return (cwd / os.path.expanduser(raw)).resolve()


def _protected_hit(path: Path, protected: Sequence[Path]) -> Path | None:
    """Return the protected root *path* lands in, or None."""
    for root in protected:
        if path.is_relative_to(root):
            return root
    return None


def _contained_root(path: Path, protected: Sequence[Path]) -> Path | None:
    """Return the first protected root that lies inside *path*, or None."""
    for root in protected:
        if root.is_relative_to(path):
            return root
    return None
