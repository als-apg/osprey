"""Finding hand-spelt layout paths in shipped Python: the scanner the spelling guards share.

A layout declaration is the one place a tree's paths are spelt; every other
module reads them from it. The guards over ``src/`` and ``packages/*/src``
hand this scanner the spellings their tree guards and compare what it finds
with the sites still allowed to spell one. It reads code, not prose:

- every string constant and every f-string, except a docstring (any string
  that stands alone as a statement). A guarded spelling matches anywhere
  inside one when bounded by characters that cannot continue a path segment,
  so it is found inside a sentence as well as inside a path;
- a join: the consecutive constant operands of a ``/`` chain (the arguments
  of a ``Path(...)`` operand included), of ``Path(...)`` and its pure
  variants, of ``.joinpath(...)`` and of ``os.path.join(...)``, joined with
  ``/`` before matching. In a join, a single operand equal to a
  guarded directory name is a match;
- a constant whose whole value is a guarded directory name, as the value of an
  assignment, a parameter default or an element of a tuple, list or set. As a
  subscript, a dict key, a comparison operand or a call argument it is an
  identity (a document key, a declaration lookup), not a spelling;
- a declaration key: a constant whose whole value is one of the *keys* a guard
  passes is an identity as a dict key or as an argument of a declaration
  lookup (:data:`LOOKUPS`), whatever spelling it equals. A row's key and its
  path are equal while the path has not moved; the key does not move with it.

A guarded spelling ending in ``/`` names a directory: its name is a guarded
directory name, and the name next to a ``/`` on either side of it
(``data/facility/scenarios``) is the same spelling.

Interface::

    sites = scan(REPO_ROOT, GUARDED, exempt={"path/of/the/declaration.py"}, keys=KEYS)
    problems = compare(sites, ALLOWED)   # [] when the tree matches the allow-list

``ALLOWED`` maps ``(file, constant text)`` to ``(count, category)``; the
category is one of :data:`CATEGORIES`. :func:`scan_source` reads one file's
source for a self-check.

The leading underscore keeps the module out of pytest collection.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

#: The trees whose Python ships.
SOURCE_ROOTS = ("src", "packages/*/src")

#: A copy in a package or shipped script that cannot import the declaration,
#: held equal to it by a test.
PINNED_COPY = "pinned copy"
#: A render path, config key or hash label whose value equals a source spelling.
EQUAL_TO_SOURCE = "render path, config key or hash label equal to a source spelling"
#: A file generated from the schema and compared byte for byte with a regeneration.
GENERATED_SCHEMA = "generated schema file"
#: A template or preset tree, which spells the layout it lays down.
TEMPLATE_TREE = "template tree"
#: A public test helper, which builds facility trees for apps' tests.
PUBLIC_TEST_HELPER = "public test helper"

#: Why a site may still spell a guarded path. Nothing else is a reason.
CATEGORIES = frozenset(
    {PINNED_COPY, EQUAL_TO_SOURCE, GENERATED_SCHEMA, TEMPLATE_TREE, PUBLIC_TEST_HELPER}
)

#: The declaration lookups, whose arguments equal to a declared key are identities.
LOOKUPS = frozenset(
    {"entry", "child", "display_path", "child_display_path", "child_path", "convention_source"}
)

#: The callables whose arguments are joined like the operands of a ``/`` chain.
_PATH_CALLS = frozenset({"Path", "PurePath", "PurePosixPath", "PureWindowsPath", "PosixPath"})

#: What can continue a path segment, on either side of a match.
_SEGMENT = r"[\w-]"


@dataclass(frozen=True)
class Site:
    """One place a guarded spelling is spelt.

    Attributes:
        file: The file, POSIX, relative to the repository root.
        line: The line the constant starts on.
        text: The constant's text: a string's value, an f-string with ``{}``
            for each replacement, or a join's operands joined with ``/``.
        spelling: The guarded spelling it holds.
    """

    file: str
    line: int
    text: str
    spelling: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: {self.spelling!r} in {self.text!r}"


def _directory_names(guarded: Iterable[str]) -> frozenset[str]:
    return frozenset(spelling[:-1] for spelling in guarded if spelling.endswith("/"))


def _pattern(spelling: str) -> re.Pattern[str]:
    """The bounded match of *spelling*, and of a directory name next to a ``/``."""
    before = rf"(?<!{_SEGMENT})(?<!\.)"
    after = rf"(?!{_SEGMENT})(?!\.{_SEGMENT})"
    if spelling.endswith("/"):
        name = re.escape(spelling[:-1])
        return re.compile(rf"{before}{name}/|/{name}{after}")
    return re.compile(rf"{before}{re.escape(spelling)}{after}")


class _Reader(ast.NodeVisitor):
    """Collects the sites of one module."""

    def __init__(self, file: str, guarded: Collection[str], keys: Collection[str] = ()) -> None:
        self.file = file
        self.keys = frozenset(keys)
        self.patterns = {spelling: _pattern(spelling) for spelling in guarded}
        self.directories = _directory_names(guarded)
        self.sites: list[Site] = []
        self._read: set[int] = set()

    # --- what a site is ------------------------------------------------------

    def _match(self, line: int, text: str) -> None:
        for spelling, pattern in self.patterns.items():
            for _ in pattern.finditer(text):
                self.sites.append(Site(self.file, line, text, spelling))

    def _whole(self, node: ast.AST) -> None:
        """A constant that is exactly a guarded directory name."""
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in self.directories
            and id(node) not in self._read
        ):
            self._read.add(id(node))
            self.sites.append(Site(self.file, node.lineno, node.value, f"{node.value}/"))

    def _join(self, operands: list[ast.expr]) -> None:
        run: list[ast.Constant] = []
        for operand in [*operands, None]:
            if (
                isinstance(operand, ast.Constant)
                and isinstance(operand.value, str)
                and id(operand) not in self._read
            ):
                run.append(operand)
                continue
            if run:
                for constant in run:
                    self._read.add(id(constant))
                text = "/".join(constant.value for constant in run)
                if len(run) == 1 and text in self.directories:
                    self.sites.append(Site(self.file, run[0].lineno, text, f"{text}/"))
                else:
                    self._match(run[0].lineno, text)
                run = []

    # --- the visitors ----------------------------------------------------------

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            self._read.add(id(node.value))
            return
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Div):
            self._join(list(_division_operands(node)))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_path_call(node.func):
            self._join(list(node.args))
        lookup = _callee_name(node.func) in LOOKUPS
        for arg in [*node.args, *(keyword.value for keyword in node.keywords)]:
            self._mark_identity(arg)
            if lookup:
                self._mark_key(arg)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._whole(node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._whole(node.value)
        self.generic_visit(node)

    def visit_arguments(self, node: ast.arguments) -> None:
        for default in [*node.defaults, *node.kw_defaults]:
            if default is not None:
                self._whole(default)
        self.generic_visit(node)

    def _visit_collection(self, node: ast.Tuple | ast.List | ast.Set) -> None:
        for element in node.elts:
            self._whole(element)
        self.generic_visit(node)

    visit_Tuple = visit_List = visit_Set = _visit_collection

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self._mark_identity(node.slice)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key in node.keys:
            if key is not None:
                self._mark_identity(key)
                self._mark_key(key)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for operand in [node.left, *node.comparators]:
            self._mark_identity(operand)
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        if id(node) in self._read:
            return
        self._read.add(id(node))
        text = "".join(
            value.value if isinstance(value, ast.Constant) else "{}" for value in node.values
        )
        for value in node.values:
            if isinstance(value, ast.Constant):
                self._read.add(id(value))
        self._match(node.lineno, text)
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                self.visit(value.value)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and id(node) not in self._read:
            self._read.add(id(node))
            self._match(node.lineno, node.value)

    def _mark_identity(self, node: ast.AST) -> None:
        """A bare directory name here, or in a collection here, is an identity."""
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            for element in node.elts:
                self._mark_identity(element)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in self.directories
        ):
            self._read.add(id(node))

    def _mark_key(self, node: ast.AST) -> None:
        """A constant equal to a declared key here is an identity."""
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in self.keys
        ):
            self._read.add(id(node))


def _callee_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _division_operands(node: ast.expr) -> Iterator[ast.expr]:
    """The operands of a ``/`` chain, left to right.

    A ``Path(...)`` operand contributes its own arguments, so
    ``Path("a") / "b"`` is the one join ``a/b``.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        yield from _division_operands(node.left)
        yield from _division_operands(node.right)
    elif isinstance(node, ast.Call) and _is_path_call(node.func) and not node.keywords:
        yield from node.args
    else:
        yield node


def _is_path_call(func: ast.expr) -> bool:
    """``Path(...)``, ``pathlib.PurePosixPath(...)``, ``x.joinpath(...)``, ``os.path.join(...)``."""
    if isinstance(func, ast.Name):
        return func.id in _PATH_CALLS
    if isinstance(func, ast.Attribute):
        if func.attr in _PATH_CALLS or func.attr == "joinpath":
            return True
        return func.attr == "join" and ast.unparse(func.value) in {"os.path", "posixpath", "path"}
    return False


def scan_source(
    source: str, file: str, guarded: Collection[str], keys: Collection[str] = ()
) -> list[Site]:
    """Every site of *guarded* in one module's *source*, in source order.

    A node's rule runs before its children are read, so a constant a join, a
    value or an identity has decided is not read again on its own.
    """
    reader = _Reader(file, guarded, keys)
    reader.visit(ast.parse(source))
    return sorted(reader.sites, key=lambda site: (site.line, site.text, site.spelling))


def scan(
    repo_root: Path,
    guarded: Collection[str],
    *,
    roots: Iterable[str] = SOURCE_ROOTS,
    exempt: Collection[str] = (),
    keys: Collection[str] = (),
) -> list[Site]:
    """Every site of *guarded* in the Python under *roots*, except the *exempt* files.

    A constant equal to one of *keys* is a declaration key where an identity can stand.
    """
    sites: list[Site] = []
    for pattern in roots:
        for root in sorted(repo_root.glob(pattern)):
            for path in sorted(root.rglob("*.py")):
                file = path.relative_to(repo_root).as_posix()
                if file in exempt:
                    continue
                sites.extend(scan_source(path.read_text(encoding="utf-8"), file, guarded, keys))
    return sites


def compare(sites: Iterable[Site], allowed: Mapping[tuple[str, str], tuple[int, str]]) -> list[str]:
    """What differs between the *sites* found and the *allowed* ones, one line each.

    A site not allowed, an allowed site found a different number of times or
    not at all, and an allowed site with no category of :data:`CATEGORIES`
    each make a line; an empty list means the tree matches.
    """
    found: Counter[tuple[str, str]] = Counter()
    where: dict[tuple[str, str], list[Site]] = {}
    for site in sites:
        key = (site.file, site.text)
        found[key] += 1
        where.setdefault(key, []).append(site)
    lines: list[str] = []
    for key, count in sorted(found.items()):
        if key not in allowed:
            lines.extend(f"spelt by hand: {site}" for site in where[key])
            continue
        expected, _ = allowed[key]
        if count != expected:
            lines.append(f"allowed {expected} time(s), found {count}: {key[0]}: {key[1]!r}")
    for key, (_, category) in sorted(allowed.items()):
        if key not in found:
            lines.append(f"allowed and no longer found: {key[0]}: {key[1]!r}")
        if category not in CATEGORIES:
            lines.append(f"allowed with no category of decision: {key[0]}: {category!r}")
    return lines
