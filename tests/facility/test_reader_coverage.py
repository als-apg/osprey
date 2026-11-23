"""No module parses ``facility.json`` unchecked.

The invariant is held at the call sites, so a reader that takes the facility file
as a parameter cannot hide behind it:

* **Names.** Every expression in ``src/`` (outside ``osprey.facility.panels``,
  which carries its own chain) that names the facility document is found: a
  ``facility_file_path(...)`` call, ``FACILITY_FILE`` under any import alias or
  re-export, the ``"facility.json"`` literal or a path spelling ending in it, and
  ``osprey.apps.sdk.facility_file()``. Names are resolved by import origin, never
  by bare spelling, so a local ``load_facility`` is not a checked entry point.
* **Flow.** Inside its function, each use of such a name, or of a variable bound
  to one, is on a whitelist: an argument to a checked entry point
  (:data:`CHECKED_ENTRY_POINTS`); ``.stat()``/``.exists()``/``.is_file()``/
  ``.is_dir()``/``.parent``/``.name``; ``str()``, ``os.fspath``, an f-string or a
  logging argument (each only formats the path); a comparison; composition into
  another path (``root / name``, ``Path(name)``), whose result is followed the
  same way; or an argument to a same-module function, whose parameter is then
  followed through that function. A :data:`PATH_ONLY` function may also return
  it, store it, write to it, or hand it to a constructor (a dataclass field).
  Any read (``.read_text``/``.read_bytes``/``.open``/``open()``) fails, and so
  does handing it to any other function.
* **Entry points.** Each checked entry point reaches a header check
  (:data:`HEADER_CHECKS`) or another checked entry point, following calls into
  same-module helpers and constructors. The panel engine's chain,
  ``FacilityDocument.from_file`` to its ``_check_header``, is held the same way,
  and ``load_graph`` must reach it.
* **Roster.** Every function in ``src/`` that calls a checked entry point is a
  roster entry (``tests/facility/_reader_roster.py``) or on
  :data:`TRANSITIVE_OK`; every roster entry names a function that exists, one of
  the three channels, and a ``/3`` test node id that collects and is neither
  skipped nor expected to fail.

The scan functions take the source root as an argument, so
``test_reader_coverage_detects`` can run them on a planted tree.
"""

from __future__ import annotations

import ast
import functools
import json
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tests.facility._reader_roster import CHANNELS, ROSTER

REPO = Path(__file__).resolve().parents[2]
SRC = (REPO / "src", REPO / "packages" / "osprey-facility" / "src")

#: The panel engine reads the document through its own chain; its modules are
#: not scanned for names, and its chain is held by the entry-point check.
PANELS = "osprey.facility.panels"

#: The functions that read the facility document and check its header before
#: any other use. Handing them the document's path is always allowed.
CHECKED_ENTRY_POINTS: frozenset[str] = frozenset(
    {
        "osprey_facility.reading.load_facility",
        "osprey.facility.views.facts.parsed_facility_file",
        "osprey_facility.records.FacilityRecords.from_file",
        "osprey.utils.facility.facility_identity",
        "osprey.utils.facility.facility_name",
        "osprey.facility.views.facts.read_facts",
        "osprey.facility.panels.model.load_graph",
    }
)

#: The header checks an entry point's body must reach.
HEADER_CHECKS: frozenset[str] = frozenset(
    {"osprey_facility.reading.check_header", "osprey_facility.reading.load_facility"}
)

#: The panel engine's own chain: ``load_graph`` reaches the reader, the reader
#: reaches the check.
PANELS_READER = "osprey.facility.panels.document.FacilityDocument.from_file"
PANELS_CHECK = "osprey.facility.panels.document._check_header"

#: Where ``FACILITY_FILE`` is defined, and the modules that re-export it. Any alias
#: resolves to one of these; re-exports elsewhere are followed through the index.
NAME_CONSTANTS: frozenset[str] = frozenset(
    {
        "osprey_facility.FACILITY_FILE",
        "osprey_facility.render.FACILITY_FILE",
        "osprey.facility.panels.document.FACILITY_FILE",
        "osprey.facility.panels.FACILITY_FILE",
    }
)

#: Functions whose call names the facility document, and their public spelling.
NAME_PRODUCERS: frozenset[str] = frozenset(
    {
        "osprey.channel_roster.sources.facility_file_path",
        "osprey.apps.sdk._facts.facility_file",
        "osprey.apps.sdk.facility_file",
    }
)

#: The functions (or, for a module-level use, the module) that hold the facility
#: document's path as a path only, and so may return it, store it, write to it
#: or hand it to a constructor. Never a read. An addition is a plan deviation.
PATH_ONLY: frozenset[str] = frozenset(
    {
        # The writer: names the file it writes and returns it in the written list.
        "osprey.facility.views.outputs.render_facility_outputs",
        # Writes the facility file the writer names, or copies an earlier render's
        # over it; reads nothing of it (plan deviation: split from the writer).
        "osprey_facility.render.write_facility_file",
        # Hashes the file a block at a time against the build's own bytes; parses nothing.
        "osprey_facility.render.serialised_file_document",
        # Names the facility file as the roster source of the records it already holds.
        "osprey.facility.views.bluesky.write_bluesky_view",
        # Same, for the device document of records already read (plan deviation).
        "osprey.facility.views.bluesky.bluesky_document",
        # Resolves the built facility file and hands it to load_graph, which checks it.
        "osprey.cli.facility_cmd._panel_graph",
        # The public SDK spelling of the render's facility file: returns the path.
        "osprey.apps.sdk._facts.facility_file",
        # The reserved-file pattern that keeps a profile from writing over facility.json.
        "osprey.cli.profile_conventions",
        # The framework's spelling of the render's facility file: returns the path (plan deviation).
        "osprey.channel_roster.sources.facility_file_path",
        # Puts that path into the roster source it resolves, a dataclass field (plan deviation).
        "osprey.channel_roster.sources.resolve_roster_source",
        # Returns and memoizes the path it has read through load_facility (plan deviation).
        "osprey.runtime._facility_channels",
    }
)

#: Functions that call a checked entry point but are not readers of their own:
#: each with the reason it needs no roster line.
TRANSITIVE_OK: frozenset[str] = frozenset(
    {
        # The app lifespan nested in _create_lifespan, whose roster line covers it.
        "osprey.interfaces.channel_finder.app._create_lifespan.lifespan",
    }
)

_READS = frozenset({"read_text", "read_bytes", "open"})
_WRITES = frozenset({"write_text", "write_bytes", "unlink", "replace", "rename"})
_ACCESSORS = frozenset({"parent", "name", "stem", "suffix", "parents"})
_PROBES = frozenset({"stat", "exists", "is_file", "is_dir"})
_FORMATTERS = frozenset({"builtins.str", "builtins.repr", "os.fspath"})
_PATH_TYPES = frozenset({"pathlib.Path", "pathlib.PurePath"})
_LOG_METHODS = frozenset({"debug", "info", "warning", "error", "exception", "critical", "log"})
_LOGGER = re.compile(r"_?(log|logger|LOG|LOGGER)$")
#: A path spelling of the facility file: the bare name, or a path ending in it.
_LITERAL = re.compile(r"(?:[\w.-]+/)*facility\.json")

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


# --- the index ---------------------------------------------------------------------------


def _module_of(src: Path, path: Path) -> str:
    parts = list(path.relative_to(src).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


@dataclass
class Scope:
    """A function (or a module or class body) and what its names are bound to."""

    qualname: str
    module: str
    node: ast.AST
    parent: Scope | None
    is_package: bool = False
    #: Names bound by an import or a def in this scope, to their qualified origin.
    bound: dict[str, str] = field(default_factory=dict)
    #: Names bound any other way (parameters, assignments): local, unresolved.
    local: set[str] = field(default_factory=set)
    #: Names declared ``global`` here.
    globals_: set[str] = field(default_factory=set)
    #: The class this function is a method of, when it is one.
    cls: str | None = None
    params: list[str] = field(default_factory=list)


def _own_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """Every node of ``node``'s body, not descending into nested functions or classes."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        yield child
        if not isinstance(child, (*_FUNCTIONS, ast.ClassDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(child))


def _import_origin(scope: Scope, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    package = scope.module.split(".")
    if not scope.is_package:
        package = package[:-1]
    if node.level > 1:
        package = package[: len(package) - (node.level - 1)]
    return ".".join([*package, node.module] if node.module else package)


def _bind(scope: Scope) -> None:
    node = scope.node
    if isinstance(node, _FUNCTIONS):
        args = node.args
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            scope.local.add(arg.arg)
            scope.params.append(arg.arg)
        for arg in (args.vararg, args.kwarg):
            if arg is not None:
                scope.local.add(arg.arg)
    for child in _own_nodes(node):
        if isinstance(child, ast.Import):
            for alias in child.names:
                if alias.asname:
                    scope.bound[alias.asname] = alias.name
                else:
                    top = alias.name.split(".")[0]
                    scope.bound[top] = top
        elif isinstance(child, ast.ImportFrom):
            origin = _import_origin(scope, child)
            for alias in child.names:
                scope.bound[alias.asname or alias.name] = f"{origin}.{alias.name}"
        elif isinstance(child, ast.Global):
            scope.globals_.update(child.names)
        elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            scope.local.add(child.id)
        elif isinstance(child, ast.arg):
            scope.local.add(child.arg)
    for child in _defs(node):
        scope.bound[child.name] = f"{scope.qualname}.{child.name}"
    scope.local -= scope.bound.keys()


def _defs(node: ast.AST) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
    for child in _own_nodes(node):
        if isinstance(child, (*_FUNCTIONS, ast.ClassDef)):
            yield child


class Index:
    """Every module of a source tree, its scopes and their bindings."""

    def __init__(self, src: Path | tuple[Path, ...]) -> None:
        roots = (src,) if isinstance(src, Path) else tuple(src)
        self.src = roots
        self.modules: dict[str, Scope] = {}
        self.scopes: dict[str, Scope] = {}
        #: Each node's parent, and each node's innermost enclosing scope.
        self.parent: dict[ast.AST, ast.AST] = {}
        self.scope_of: dict[ast.AST, Scope] = {}
        for root in roots:
            for path in sorted(root.rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                module = _module_of(root, path)
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                scope = Scope(module, module, tree, None, is_package=path.name == "__init__.py")
                self.modules[module] = scope
                self._add(scope)

    def _add(self, scope: Scope) -> None:
        self.scopes[scope.qualname] = scope
        _bind(scope)
        for child in ast.iter_child_nodes(scope.node):
            self.parent[child] = scope.node
        for node in _own_nodes(scope.node):
            self.scope_of[node] = scope
            for child in ast.iter_child_nodes(node):
                self.parent[child] = node
        for child in _defs(scope.node):
            inner = Scope(
                f"{scope.qualname}.{child.name}",
                scope.module,
                child,
                scope,
                is_package=scope.is_package,
                cls=scope.qualname if isinstance(scope.node, ast.ClassDef) else None,
            )
            self._add(inner)

    # --- resolution ----------------------------------------------------------------------

    def canonical(self, qualified: str, depth: int = 0) -> str:
        """``qualified`` with every re-export followed to where the name is defined."""
        if depth > 20 or qualified in self.scopes:
            return qualified
        parts = qualified.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            module = ".".join(parts[:cut])
            if module in self.modules:
                head, rest = parts[cut], parts[cut + 1 :]
                origin = self.modules[module].bound.get(head)
                if origin is None or origin == f"{module}.{head}":
                    return qualified
                return self.canonical(".".join([origin, *rest]), depth + 1)
        return qualified

    def resolve(self, node: ast.AST, scope: Scope) -> str | None:
        """The qualified name ``node`` (a name or attribute chain) refers to, if known."""
        if isinstance(node, ast.Name):
            current: Scope | None = scope
            while current is not None:
                if isinstance(current.node, ast.ClassDef) and current is not scope:
                    current = current.parent
                    continue
                if node.id in current.bound:
                    return self.canonical(current.bound[node.id])
                if node.id in current.local:
                    if current.cls and current.params[:1] == [node.id]:
                        return f"<{current.cls}>"
                    return None
                current = current.parent
            return f"builtins.{node.id}"
        if isinstance(node, ast.Attribute):
            base = self.resolve(node.value, scope)
            if base is None:
                return None
            if base.startswith("<"):
                base = base[1:-1]
            return self.canonical(f"{base}.{node.attr}")
        return None

    def callee(self, call: ast.Call, scope: Scope) -> str | None:
        """What ``call`` calls; a class's ``__init__`` when it calls a class."""
        name = self.resolve(call.func, scope)
        if name is None:
            return None
        if name.startswith("<"):
            name = name[1:-1]
        target = self.scopes.get(name)
        if target is not None and isinstance(target.node, ast.ClassDef):
            return f"{name}.__init__"
        return name

    def is_class(self, name: str | None) -> bool:
        if name is None:
            return False
        if name.endswith(".__init__"):
            return True
        target = self.scopes.get(name)
        if target is not None:
            return isinstance(target.node, ast.ClassDef)
        return name.rsplit(".", 1)[-1][:1].isupper()


@functools.cache
def index(src: Path | tuple[Path, ...]) -> Index:
    return Index(src)


# --- names -------------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class Problem:
    """One use of the facility document's name outside the whitelist."""

    scope: str
    line: int
    detail: str

    def __str__(self) -> str:
        return f"{self.scope}:{self.line}: {self.detail}"


def _is_docstring(node: ast.Constant, idx: Index) -> bool:
    parent = idx.parent.get(node)
    if not isinstance(parent, ast.Expr):
        return False
    holder = idx.parent.get(parent)
    body = getattr(holder, "body", None)
    return isinstance(body, list) and bool(body) and body[0] is parent


def _is_definition(node: ast.AST, idx: Index) -> bool:
    """The literal of ``FACILITY_FILE = "facility.json"`` itself."""
    parent = idx.parent.get(node)
    if not isinstance(parent, ast.Assign) or parent.value is not node:
        return False
    scope = idx.scope_of[node]
    return any(
        isinstance(target, ast.Name) and f"{scope.qualname}.{target.id}" in NAME_CONSTANTS
        for target in parent.targets
    )


def name_sites(src: Path | tuple[Path, ...]) -> list[tuple[Scope, ast.AST, str]]:
    """Each expression in ``src`` naming the facility document, with its scope."""
    idx = index(src)
    sites = []
    for node, scope in idx.scope_of.items():
        if scope.module == PANELS or scope.module.startswith(PANELS + "."):
            continue
        what = None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if (
                _LITERAL.fullmatch(node.value)
                and not _is_docstring(node, idx)
                and not _is_definition(node, idx)
            ):
                what = repr(node.value)
        elif isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
            if idx.resolve(node, scope) in NAME_CONSTANTS:
                what = "FACILITY_FILE"
        elif isinstance(node, ast.Call):
            if idx.resolve(node.func, scope) in NAME_PRODUCERS:
                what = f"{idx.resolve(node.func, scope)}()"
        if what is not None:
            sites.append((scope, node, what))
    return sorted(sites, key=lambda site: (site[0].qualname, site[1].lineno))


# --- flow --------------------------------------------------------------------------------


class Flow:
    """Follows each use of a name through its function and the helpers it reaches."""

    def __init__(self, idx: Index) -> None:
        self.idx = idx
        self.seen: set[tuple[str, str, bool]] = set()
        self.problems: list[Problem] = []

    def fail(self, scope: Scope, node: ast.AST, detail: str) -> None:
        self.problems.append(Problem(scope.qualname, getattr(node, "lineno", 0), detail))

    def variable(self, scope: Scope, name: str, path_only: bool) -> None:
        """Follow every load of local ``name`` in ``scope``, nested functions included."""
        if (scope.qualname, name, path_only) in self.seen:
            return
        self.seen.add((scope.qualname, name, path_only))
        for node in ast.walk(scope.node):
            if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load):
                self.use(self.idx.scope_of[node], node, path_only)

    def helper(self, scope: Scope, call: ast.Call, node: ast.AST, target: Scope, path_only: bool):
        """Follow ``node``, an argument of ``call``, into the parameter of ``target``."""
        params = target.params
        if target.cls is not None and params and not _is_static(target.node):
            params = params[1:]
        if node in call.args:
            position = call.args.index(node)
            if position >= len(params):
                self.fail(scope, node, f"passed as *args to {target.qualname}")
                return
            name = params[position]
        else:
            name = next(kw.arg for kw in call.keywords if kw.value is node)
        if name is None or name not in target.params:
            self.fail(scope, node, f"passed as **kwargs to {target.qualname}")
            return
        self.variable(target, name, path_only or target.qualname in PATH_ONLY)

    def use(self, scope: Scope, expr: ast.AST, path_only: bool) -> None:
        """Check one use of ``expr``, which carries the facility document's path."""
        idx = self.idx
        path_only = path_only or scope.qualname in PATH_ONLY
        node = expr
        contained = False
        while True:
            parent = idx.parent.get(node)
            if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div):
                node = parent
                continue
            if isinstance(parent, (ast.Tuple, ast.List, ast.Set, ast.IfExp, ast.BoolOp)):
                if isinstance(parent, ast.IfExp) and parent.test is node:
                    return
                contained = contained or isinstance(parent, (ast.Tuple, ast.List, ast.Set))
                node = parent
                continue
            if isinstance(parent, ast.Starred):
                node = parent
                continue
            if isinstance(parent, ast.keyword):
                call = idx.parent[parent]
                assert isinstance(call, ast.Call)
                self.argument(scope, call, parent.value, node, path_only)
                return
            if isinstance(parent, ast.Call):
                if parent.func is node:
                    self.fail(scope, node, "called")
                    return
                callee = idx.callee(parent, scope)
                if callee in _PATH_TYPES:
                    node = parent
                    continue
                self.argument(scope, parent, node, node, path_only)
                return
            if isinstance(parent, ast.Attribute):
                attr = parent.attr
                call = idx.parent.get(parent)
                called = isinstance(call, ast.Call) and call.func is parent
                if attr == "open" and called and path_only and _writes_only(call):
                    pass
                elif attr in _READS:
                    self.fail(scope, parent, f"read with .{attr}()")
                elif attr in _ACCESSORS and not called:
                    pass
                elif attr in _PROBES and called:
                    pass
                elif attr in _WRITES and called and path_only:
                    pass
                else:
                    self.fail(scope, parent, f"used as .{attr}")
                return
            if isinstance(parent, (ast.FormattedValue, ast.Compare)):
                return
            if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
                if contained:
                    if not path_only:
                        self.fail(scope, parent, "stored in a container")
                    return
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in scope.globals_:
                        self.variable(scope, target.id, path_only)
                    elif not path_only:
                        self.fail(scope, parent, "stored")
                return
            if isinstance(parent, (ast.Return, ast.Yield, ast.YieldFrom)):
                if not path_only:
                    self.fail(scope, parent, "returned")
                return
            if isinstance(parent, ast.Expr):
                return
            self.fail(scope, node, f"used in a {type(parent).__name__}")
            return

    def argument(
        self, scope: Scope, call: ast.Call, arg: ast.AST, carried: ast.AST, path_only: bool
    ) -> None:
        """``carried`` reaches ``call`` as its argument ``arg``."""
        idx = self.idx
        callee = idx.callee(call, scope)
        if callee in CHECKED_ENTRY_POINTS or callee in PATH_ONLY:
            return
        if callee in _FORMATTERS:
            return
        func = call.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in _LOG_METHODS
            and isinstance(func.value, ast.Name)
            and _LOGGER.match(func.value.id)
        ):
            return
        if callee == "builtins.open":
            self.fail(scope, call, "read with open()")
            return
        target = idx.scopes.get(callee or "")
        if (
            target is not None
            and isinstance(target.node, _FUNCTIONS)
            and target.module == scope.module
            and arg is carried
        ):
            self.helper(scope, call, arg, target, path_only)
            return
        if path_only and idx.is_class(callee):
            return
        self.fail(
            scope, call, f"passed to {callee or ast.unparse(func)}, not a checked entry point"
        )


def _writes_only(call: ast.Call) -> bool:
    """Whether ``call`` (a ``.open(...)``) opens for writing alone, never reading."""
    modes = [*call.args[:1], *(kw.value for kw in call.keywords if kw.arg == "mode")]
    return bool(modes) and all(
        isinstance(mode, ast.Constant)
        and isinstance(mode.value, str)
        and "r" not in mode.value
        and "+" not in mode.value
        for mode in modes
    )


def _is_static(node: ast.AST) -> bool:
    return any(
        isinstance(dec, ast.Name) and dec.id == "staticmethod"
        for dec in getattr(node, "decorator_list", ())
    )


def flow_problems(src: Path | tuple[Path, ...]) -> list[Problem]:
    """Every use of the facility document's name in ``src`` outside the whitelist."""
    flow = Flow(index(src))
    for scope, node, _ in name_sites(src):
        flow.use(scope, node, False)
    return sorted(set(flow.problems))


# --- entry points and callers ------------------------------------------------------------


def calls(src: Path | tuple[Path, ...], qualname: str) -> Iterator[str]:
    """Every callee ``qualname``'s body calls, nested functions included."""
    idx = index(src)
    scope = idx.scopes[qualname]
    for node in ast.walk(scope.node):
        if isinstance(node, ast.Call):
            callee = idx.callee(node, idx.scope_of[node])
            if callee is not None:
                yield callee


def reaches(src: Path | tuple[Path, ...], qualname: str, targets: frozenset[str]) -> bool:
    """Whether ``qualname`` calls one of ``targets``, through same-module helpers."""
    idx = index(src)
    seen: set[str] = set()
    todo = [qualname]
    module = idx.scopes[qualname].module
    while todo:
        current = todo.pop()
        if current in seen or current not in idx.scopes:
            continue
        seen.add(current)
        for callee in calls(src, current):
            if callee in targets and callee != qualname:
                return True
            if callee in idx.scopes and idx.scopes[callee].module == module:
                todo.append(callee)
    return False


def callers(src: Path | tuple[Path, ...], targets: frozenset[str]) -> set[str]:
    """Every function in ``src`` whose own body calls one of ``targets``."""
    idx = index(src)
    found = set()
    for node, scope in idx.scope_of.items():
        if isinstance(node, ast.Call) and isinstance(scope.node, _FUNCTIONS):
            if idx.callee(node, scope) in targets:
                found.add(scope.qualname)
    return found


# --- the tests ---------------------------------------------------------------------------


def test_the_scan_finds_the_names_it_is_meant_to_find():
    """The control: each kind of name is found in today's tree, so an empty result is real."""
    found = {(scope.qualname, what) for scope, _, what in name_sites(SRC)}
    kinds = {what for _, what in found}
    assert "FACILITY_FILE" in kinds
    assert "'facility.json'" in kinds
    assert "osprey.channel_roster.sources.facility_file_path()" in kinds
    assert ("osprey.apps.sdk._facts.facility_file", "FACILITY_FILE") in found
    # Resolved through osprey_facility.render's re-export.
    assert ("osprey.apps.sdk.testing._deployment.RenderedTree.read", "FACILITY_FILE") in found


def test_no_module_uses_the_facility_document_s_name_unchecked():
    problems = flow_problems(SRC)
    assert not problems, "the facility document is used outside the whitelist:\n" + "\n".join(
        map(str, problems)
    )


def test_path_only_is_the_ruled_list():
    assert PATH_ONLY == frozenset(
        {
            "osprey.facility.views.outputs.render_facility_outputs",
            "osprey_facility.render.write_facility_file",
            "osprey_facility.render.serialised_file_document",
            "osprey.facility.views.bluesky.write_bluesky_view",
            "osprey.facility.views.bluesky.bluesky_document",
            "osprey.cli.facility_cmd._panel_graph",
            "osprey.apps.sdk._facts.facility_file",
            "osprey.cli.profile_conventions",
            "osprey.channel_roster.sources.facility_file_path",
            "osprey.channel_roster.sources.resolve_roster_source",
            "osprey.runtime._facility_channels",
        }
    )
    idx = index(SRC)
    assert PATH_ONLY <= idx.scopes.keys(), sorted(PATH_ONLY - idx.scopes.keys())


@pytest.mark.parametrize("entry", sorted(CHECKED_ENTRY_POINTS))
def test_each_checked_entry_point_checks_the_header(entry: str):
    idx = index(SRC)
    assert entry in idx.scopes, f"{entry} is not in the scanned source any more"
    targets = HEADER_CHECKS | CHECKED_ENTRY_POINTS | {PANELS_READER}
    assert reaches(SRC, entry, frozenset(targets)), f"{entry} reaches no header check"


def test_the_panel_engine_reads_through_its_own_check():
    assert reaches(SRC, PANELS_READER, frozenset({PANELS_CHECK}))
    assert reaches(SRC, "osprey.facility.panels.model.load_graph", frozenset({PANELS_READER}))


def test_every_caller_of_a_checked_entry_point_is_on_the_roster():
    rostered = {entry.function for entry in ROSTER}
    found = callers(SRC, CHECKED_ENTRY_POINTS)
    missing = found - rostered - TRANSITIVE_OK - CHECKED_ENTRY_POINTS
    assert not missing, f"call a checked entry point but have no roster line: {sorted(missing)}"
    stale = TRANSITIVE_OK - found
    assert not stale, f"on TRANSITIVE_OK but call no checked entry point: {sorted(stale)}"


def test_every_roster_entry_names_a_function_and_a_channel():
    idx = index(SRC)
    for entry in ROSTER:
        assert entry.channel in CHANNELS, entry
        assert entry.function in idx.scopes, f"{entry.function} is not in the scanned source"


_COLLECT = """
import json
import sys

import pytest


class Record:
    def __init__(self):
        self.items = {}

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, items):
        for item in items:
            marks = {mark.name for mark in item.iter_markers()}
            skipif = any(
                mark.args and mark.args[0] is True for mark in item.iter_markers("skipif")
            )
            self.items[item.nodeid] = sorted(marks | ({"skipif"} if skipif else set()))


record = Record()
args = [*sys.argv[2:], "--collect-only", "-q", "-o", "addopts=", "-p", "no:cacheprovider"]
code = pytest.main(args, plugins=[record])
with open(sys.argv[1], "w", encoding="utf-8") as out:
    json.dump(record.items, out)
sys.exit(code)
"""


def test_every_roster_test_collects_and_runs(tmp_path: Path):
    """Each roster entry's ``/3`` test is collected, and is not skipped or an xfail."""
    files = sorted({entry.test.split("::", 1)[0] for entry in ROSTER})
    script = tmp_path / "collect.py"
    script.write_text(_COLLECT, encoding="utf-8")
    out = tmp_path / "items.json"
    result = subprocess.run(
        [sys.executable, str(script), str(out), *files],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.is_file(), result.stdout + result.stderr
    items: dict[str, list[str]] = json.loads(out.read_text(encoding="utf-8"))
    for entry in ROSTER:
        matching = [
            node for node in items if node == entry.test or node.startswith(entry.test + "[")
        ]
        assert matching, f"{entry.function}: {entry.test} is not collected"
        for node in matching:
            inert = {"skip", "skipif", "xfail"} & set(items[node])
            assert not inert, f"{entry.function}: {node} is marked {sorted(inert)}"
