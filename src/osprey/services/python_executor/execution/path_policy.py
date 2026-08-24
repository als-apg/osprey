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

Matching is by *function name* (``.rmtree``, ``.copy``, ``os.remove``), not by
the module the caller happened to bind, so ``import shutil as sh`` is still
seen. That breadth is safe because an issue also requires the resolved literal
path to land inside a protected root; aliasing games that defeat name matching
entirely (``getattr(shutil, 'rmtree')``) fall through to the runtime guard like
any other dynamic call.

The policy is applied in **every** execution mode. Readonly and readwrite runs
alike may not write into the render zone or the profile sources — that is the
self-change boundary, not a control-system write gate.

The walker carries one check that is *not* about paths: user code naming the
emitted guard's own identifiers. The guard cannot police attempts to unpatch
it — code that has already reached ``_restore_patched_targets`` is past the
boundary — so the one layer that can refuse that is a static one, here, before
the child is ever spawned. Unlike the path checks it does not defer dynamic
spellings, because a string literal handed to ``getattr`` is the normal way
this is attempted rather than an edge case.
"""

import ast
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

# ``open``/``Path.open`` mode characters that mean "this call may modify the
# file". Anything else ('r', 'rb', 'rt') is a read and is never an issue.
_WRITE_MODE_CHARS = frozenset("wax+")

# Write-intent call names, mapped to the argument positions that name a path
# the call may create, overwrite, or delete. Sources of a copy are reads and
# are not listed; ``move`` and ``rename``/``replace`` list both sides because
# they remove the original as well.
_PATH_ARG_POSITIONS: dict[str, tuple[int, ...]] = {
    # shutil
    "copy": (1,),
    "copy2": (1,),
    "copyfile": (1,),
    "move": (0, 1),
    "rmtree": (0,),
    # os
    "remove": (0,),
    "unlink": (0,),
    "rename": (0, 1),
    "replace": (0, 1),
    "makedirs": (0,),
    "mkdir": (0,),
    "truncate": (0,),
}

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


def path_policy_issues(
    code: str,
    *,
    protected_roots: Iterable[str | os.PathLike[str]],
    permitted_roots: Iterable[str | os.PathLike[str]] = (),
) -> list[str]:
    """Return an issue per literal write aimed at a protected root.

    Args:
        code: The user-submitted source to analyse.
        protected_roots: Absolute paths (files or directories) that user code
            may not write into, in any execution mode. Resolved by the caller
            in the parent process — never re-derived here.
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

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    # Guard tampering is independent of the root sets — there is no path to
    # compare, and a deployment that protects nothing still owns its guard.
    issues: list[str] = _guard_tamper_issues(tree)
    if not protected:
        return issues

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for raw, spelling in _write_targets(node):
            hit = _protected_hit(raw, protected, permitted)
            if hit is not None:
                issues.append(
                    f"Write to '{raw}' via {spelling} is not allowed: "
                    f"the path is inside the protected location '{hit}'"
                )
    return issues


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


def _write_targets(node: ast.Call) -> list[tuple[str, str]]:
    """Return ``(literal path, call spelling)`` per write target of one call."""
    func = node.func
    name = _called_name(func)
    if name is None:
        return []

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
        return [(path, "open()")] if path else []

    if isinstance(func, ast.Attribute):
        receiver = _literal_path(func.value)
        if receiver:
            # Path('...').write_text(...) / .write_bytes(...)
            if name in _PATH_WRITE_METHODS:
                return [(receiver, f"Path.{name}()")]
            # Path('...').open('w') — mode is the first argument here.
            if name == "open":
                if not _has_write_mode(node, mode_index=0):
                    return []
                return [(receiver, "Path.open()")]

    positions = _PATH_ARG_POSITIONS.get(name)
    if positions is None:
        return []
    targets: list[tuple[str, str]] = []
    for index in positions:
        path = _literal_path(_positional(node, index))
        if path:
            targets.append((path, f"{name}()"))
    return targets


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
        if _called_name(node.func) in {"Path", "PurePath", "PosixPath", "WindowsPath"}:
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


def _protected_hit(raw: str, protected: Sequence[Path], permitted: Sequence[Path]) -> Path | None:
    """Return the protected root ``raw`` lands in, or None if it is clean."""
    if any(_is_inside(raw, root) for root in permitted):
        return None
    for root in protected:
        if _is_inside(raw, root):
            return root
    return None


def _is_inside(raw: str, root: Path) -> bool:
    """True when the literal path ``raw`` names ``root`` or something under it.

    Absolute paths are compared after resolution; existence is never required.
    A *relative* path has no cwd to resolve against here — the child process
    supplies that — so it is matched structurally instead: it is inside ``root``
    when some trailing run of ``root``'s components prefixes it, which is what
    ``build/config.yml`` means for a render zone at ``<project>/build``. A
    relative path that climbs out with ``..`` is undecidable and is left to the
    runtime guard.
    """
    candidate = os.path.expanduser(raw)
    if os.path.isabs(candidate):
        return Path(candidate).resolve().is_relative_to(root)

    parts = Path(os.path.normpath(candidate)).parts
    if not parts or ".." in parts:
        return False
    root_parts = root.parts
    for length in range(1, len(root_parts) + 1):
        tail = root_parts[-length:]
        if tail[0] == root.anchor:
            break  # a relative path can never match root's anchor
        if len(tail) <= len(parts) and parts[: len(tail)] == tail:
            return True
    return False
