"""Audit: test modules must not mutate process state at import time.

Import-time mutations are the one class of cross-test pollution a fixture
cannot undo — they run during collection, in whichever worker happens to
import the module, before any fixture of any scope gets a chance to snapshot
what they overwrote. This audit walks every module under ``tests/`` and
asserts that the set of import-time writes to ``os.environ``, ``sys.path``,
``sys.modules``, and sockets is *exactly* the whitelist below.

The whitelist is deliberately exact rather than a minimum: a new import-time
mutation fails this test, and removing a whitelisted one fails it too, so the
list cannot rot into a list of things that used to be true. Each whitelisted
site carries an inline ``# import-time required because ...`` comment at the
mutation itself explaining why a fixture cannot do the job.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]

#: ``path relative to tests/`` -> the kinds of import-time mutation allowed there.
WHITELIST: dict[str, set[str]] = {
    # libca latches EPICS_CA_* at C-library init, on the first import of
    # epics/softioc anywhere in the process; the ports are picked from free
    # ones at the same moment for the same reason.
    "va/test_record_factory.py": {"os.environ", "socket"},
    "va/test_apply_fault.py": {"os.environ", "socket"},
    # docs/source/_ext is not an installed package, and the extension under
    # test is imported at module top level.
    "documentation/test_workflow_autodoc.py": {"sys.path"},
    # scripts/ is not a package: these scripts are loaded by path and
    # registered in sys.modules so their dataclasses resolve cls.__module__.
    "benchmark/test_matrix.py": {"sys.modules"},
    "benchmark/test_matrix_dashboard.py": {"sys.modules"},
    "va/e2e/conftest.py": {"sys.modules"},
}

#: Marker every whitelisted site must carry, next to the mutation.
JUSTIFICATION_MARKER = "# import-time required because"

_MAPPING_MUTATORS = frozenset({"setdefault", "update", "pop", "clear", "popitem"})
_LIST_MUTATORS = frozenset(
    {"insert", "append", "extend", "remove", "pop", "clear", "sort", "reverse"}
)
_SOCKET_OPENERS = frozenset({"socket", "create_connection", "create_server", "socketpair"})
_PORT_HELPERS = frozenset({"free_port", "_free_port"})


def _dotted(node: ast.AST) -> str:
    """Render ``a.b.c`` attribute chains; empty string for anything else."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _target_kind(node: ast.AST) -> str | None:
    """Kind of guarded object a subscript/attribute assignment target names."""
    if isinstance(node, ast.Subscript):
        node = node.value
    dotted = _dotted(node)
    return dotted if dotted in {"os.environ", "sys.path", "sys.modules"} else None


def _node_kinds(node: ast.AST) -> set[str]:
    """Kinds of guarded state a single node writes."""
    if isinstance(node, ast.Assign | ast.Delete):
        kinds = {_target_kind(target) for target in node.targets}
    elif isinstance(node, ast.AugAssign | ast.AnnAssign):
        kinds = {_target_kind(node.target)}
    elif isinstance(node, ast.Call):
        kinds = {_call_kind(node)}
    else:
        return set()
    kinds.discard(None)
    return kinds  # type: ignore[return-value]


def _call_kind(call: ast.Call) -> str | None:
    """Kind of guarded state a call mutates, if any."""
    if isinstance(call.func, ast.Name):
        return "socket" if call.func.id in _PORT_HELPERS else None

    if not isinstance(call.func, ast.Attribute):
        return None

    receiver = _dotted(call.func.value)
    attr = call.func.attr

    if receiver in {"os.environ", "sys.modules"} and attr in _MAPPING_MUTATORS:
        return receiver
    if receiver == "sys.path" and attr in _LIST_MUTATORS:
        return "sys.path"
    if receiver == "os" and attr in {"putenv", "unsetenv"}:
        return "os.environ"
    if receiver == "socket" and attr in _SOCKET_OPENERS:
        return "socket"
    return None


def _is_main_guard(node: ast.AST) -> bool:
    """True for ``if __name__ == "__main__"`` blocks (and variants with more
    conditions), whose bodies never run under pytest's import."""
    if not isinstance(node, ast.If):
        return False
    names = {sub.id for sub in ast.walk(node.test) if isinstance(sub, ast.Name)}
    constants = {sub.value for sub in ast.walk(node.test) if isinstance(sub, ast.Constant)}
    return "__name__" in names and "__main__" in constants


def _executed_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """Yield the nodes that run when ``node`` itself runs.

    Recursion stops at every function boundary: defining a function does not
    execute its body, so only decorators and default arguments are reached.
    Class bodies, control flow, and comprehensions all execute in place and
    are followed. ``__main__`` guards are skipped — pytest imports the module,
    it never runs it as a script.
    """
    for child in ast.iter_child_nodes(node):
        if _is_main_guard(child):
            continue
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            deferred = [*getattr(child, "decorator_list", []), *child.args.defaults]
            for evaluated in deferred:
                yield evaluated
                yield from _executed_nodes(evaluated)
            continue
        yield child
        yield from _executed_nodes(child)


def import_time_mutations(source: str) -> set[str]:
    """Kinds of guarded state a module writes while being imported.

    Calls to functions defined in the same module are followed, since calling
    one at import time executes its body then (this is how
    ``test_matrix_dashboard.py`` registers its ``sys.modules`` entry).
    """
    tree = ast.parse(source)
    local_functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }

    found: set[str] = set()
    entered: set[str] = set()
    pending: list[ast.AST] = [tree]
    while pending:
        for node in _executed_nodes(pending.pop()):
            found |= _node_kinds(node)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                name = node.func.id
                if name in local_functions and name not in entered:
                    entered.add(name)
                    pending.append(local_functions[name])
    return found


def _test_modules() -> list[Path]:
    return sorted(p for p in TESTS_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def test_import_time_mutations_match_whitelist():
    """No test module mutates guarded state at import time off-whitelist."""
    actual: dict[str, set[str]] = {}
    for path in _test_modules():
        mutations = import_time_mutations(path.read_text())
        if mutations:
            actual[path.relative_to(TESTS_ROOT).as_posix()] = mutations

    unlisted = {
        path: sorted(kinds) for path, kinds in actual.items() if WHITELIST.get(path) != kinds
    }
    stale = {path: sorted(kinds) for path, kinds in WHITELIST.items() if actual.get(path) != kinds}
    assert actual == WHITELIST, (
        "Import-time mutation audit does not match the whitelist.\n"
        f"Found but not whitelisted (or different kinds): {unlisted}\n"
        f"Whitelisted but not found (or different kinds): {stale}\n"
        "Move the mutation into a fixture, or — if it genuinely cannot wait for "
        "one — add it here with an inline justification comment at the site."
    )


def test_whitelisted_sites_carry_justification():
    """Every whitelisted module explains its exemption at the mutation site."""
    missing = [
        relative_path
        for relative_path in WHITELIST
        if JUSTIFICATION_MARKER not in (TESTS_ROOT / relative_path).read_text()
    ]
    assert not missing, (
        f"Whitelisted modules without a '{JUSTIFICATION_MARKER} ...' comment: {missing}\n"
        "Add the comment directly above the mutation, saying what forces it to run at "
        "import time rather than in a fixture."
    )
