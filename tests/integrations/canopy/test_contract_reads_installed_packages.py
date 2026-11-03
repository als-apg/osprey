"""No CANOPY contract module reads a path above an installed CANOPY package.

The contract runs against CANOPY installed at the pin, from any kind of
install. A package's own directory is part of every install; whatever sits
above it (a checkout's ``tests/``, its repository root) exists only beside an
editable one. A module here that needs a CANOPY file outside the packages
ships its own copy under ``fixtures/``.

These tests parse the directory's sources and import no CANOPY package, so
they run in every environment.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest
from tests._canopy_contract import CANOPY_PACKAGES

HERE = Path(__file__).parent

#: Path methods that return the same location, so they add no level.
_SAME_PLACE = ("resolve", "absolute")

#: ``.parent`` once is the package's own directory; one more leaves the install.
_LEVELS_ALLOWED = 1

#: Nodes whose body binds names of its own.
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _is_canopy(dotted: str) -> bool:
    return dotted.split(".", 1)[0] in CANOPY_PACKAGES


def _canopy_names(tree: ast.AST) -> set[str]:
    """Every name *tree* binds by importing a CANOPY package or a part of one."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_canopy(alias.name):
                    names.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and _is_canopy(node.module):
                names.update(alias.asname or alias.name for alias in node.names)
    return names


def _root_name(node: ast.expr) -> str | None:
    """The name an attribute chain starts from: ``a`` for ``a.b.c``."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _is_package_file_path(node: ast.expr, names: set[str]) -> bool:
    """Whether *node* is ``Path(<name>.__file__)`` for a CANOPY *name*."""
    if not (isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords):
        return False
    called = node.func
    called_name = called.attr if isinstance(called, ast.Attribute) else getattr(called, "id", None)
    argument = node.args[0]
    return (
        called_name == "Path"
        and isinstance(argument, ast.Attribute)
        and argument.attr == "__file__"
        and _root_name(argument.value) in names
    )


def _levels_above_package_file(
    node: ast.expr, names: set[str], bound: dict[str, int] | None = None
) -> int | None:
    """How many directories *node* climbs from a CANOPY package's file.

    ``None`` when *node* is not a climb from ``Path(<name>.__file__)``. A
    ``parents`` index that is not a literal integer counts as leaving the
    package, because nothing bounds it. *bound* maps a local name to the
    levels its value already climbs; a climb from that name adds them.
    """
    bound = bound or {}
    levels = 0
    while True:
        if _is_package_file_path(node, names):
            return levels
        if isinstance(node, ast.Name) and node.id in bound:
            return levels + bound[node.id]
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            levels += 1
            node = node.value
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "parents"
        ):
            index = node.slice
            if isinstance(index, ast.Constant) and type(index.value) is int and index.value >= 0:
                levels += index.value + 1
            else:
                levels += _LEVELS_ALLOWED + 1
            node = node.value.value
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _SAME_PLACE
        ):
            node = node.func.value
        else:
            return None


def _own_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Every node below *scope* that belongs to it rather than to a nested scope.

    A nested scope's header (decorators, defaults, bases) belongs to *scope*;
    its body belongs to the nested scope.
    """
    pending = list(scope.body if isinstance(scope, _SCOPES) else ast.iter_child_nodes(scope))
    while pending:
        node = pending.pop()
        yield node
        children = list(ast.iter_child_nodes(node))
        if isinstance(node, _SCOPES):
            body = {id(statement) for statement in node.body}
            children = [child for child in children if id(child) not in body]
        pending.extend(children)


def _bindings(nodes: list[ast.AST], names: set[str]) -> dict[str, int]:
    """Each name *nodes* assign a climb from a CANOPY package's file, with its levels.

    A name bound more than once keeps its highest levels.
    """
    bound: dict[str, int] = {}
    for node in nodes:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        levels = _levels_above_package_file(value, names)
        if levels is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                bound[target.id] = max(levels, bound.get(target.id, levels))
    return bound


def lines_leaving_the_packages(source: str) -> list[int]:
    """The lines of *source* that derive a path above an installed CANOPY package.

    A name bound in the same scope to a climb from a package's file counts its
    value's levels wherever that scope climbs from it.
    """
    tree = ast.parse(source)
    names = _canopy_names(tree)
    lines: set[int] = set()
    for scope in [tree, *(node for node in ast.walk(tree) if isinstance(node, _SCOPES))]:
        nodes = list(_own_nodes(scope))
        bound = _bindings(nodes, names)
        for node in nodes:
            if isinstance(node, ast.expr):
                levels = _levels_above_package_file(node, names, bound)
                if levels is not None and levels > _LEVELS_ALLOWED:
                    lines.add(node.lineno)
    return sorted(lines)


def offenders(directory: Path) -> Iterator[str]:
    """``<file>:<line>`` for every offending line of every module below *directory*."""
    for path in sorted(directory.rglob("*.py")):
        for line in lines_leaving_the_packages(path.read_text(encoding="utf-8")):
            yield f"{path.relative_to(directory).as_posix()}:{line}"


def test_every_contract_module_reads_only_inside_the_installed_packages():
    found = list(offenders(HERE))
    assert not found, (
        "a path above an installed CANOPY package exists only beside an editable install; "
        "ship the file under fixtures/ instead: " + ", ".join(found)
    )


def test_the_guard_names_a_planted_checkout_root(tmp_path):
    (tmp_path / "test_planted.py").write_text(
        "from pathlib import Path\n"
        "\n"
        "import canopy\n"
        "\n"
        "CANOPY_ROOT = Path(canopy.__file__).resolve().parents[2]\n",
        encoding="utf-8",
    )
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "test_clean.py").write_text("import canopy\n", encoding="utf-8")

    assert list(offenders(tmp_path)) == ["test_planted.py:5"]


@pytest.mark.parametrize(
    "source",
    [
        "import canopy\nPath(canopy.__file__).parents[1]\n",
        "import canopy\nPath(canopy.__file__).parent.parent\n",
        "import canopy\nPath(canopy.__file__).resolve().parent.resolve().parent\n",
        "import canopy\nPath(canopy.__file__).parent.parents[0]\n",
        "import canopy\nPath(canopy.__file__).parents[depth]\n",
        "import canopy\nimport pathlib\npathlib.Path(canopy.__file__).parents[2]\n",
        "import canopy.main\nPath(canopy.main.__file__).parents[2]\n",
        "import canopy_mcp.server as server\nPath(server.__file__).parents[1]\n",
        "from canopy_mcp import server\nPath(server.__file__).parent.parent\n",
        "from pydmconverter import cli as entry\nPath(entry.__file__).absolute().parents[3]\n",
        "def root():\n    import pydmconverter\n    return Path(pydmconverter.__file__).parents[1]\n",
        "import canopy\nroot = Path(canopy.__file__)\nroot.parents[2]\n",
        "import canopy\nroot = Path(canopy.__file__).resolve()\nroot.parent.parent\n",
        "import canopy\nhere = Path(canopy.__file__).parent\nhere.parent\n",
        "import canopy\ndef f():\n    top: Path = Path(canopy.__file__)\n    return top.parents[1]\n",
    ],
)
def test_a_climb_out_of_a_package_is_named(source):
    assert lines_leaving_the_packages(source) == [len(source.splitlines())]


@pytest.mark.parametrize(
    "source",
    [
        "import canopy  # noqa: F401\n",
        "import canopy\nPath(canopy.__file__)\n",
        "import canopy_mcp\nPath(canopy_mcp.__file__).parent\n",
        "import canopy_mcp\nPath(canopy_mcp.__file__).resolve().parents[0]\n",
        "import canopy\nPath(__file__).resolve().parents[3]\n",
        "import canopy\nimport osprey\nPath(osprey.__file__).parent.parent\n",
        "import canopy_tools\nPath(canopy_tools.__file__).parents[2]\n",
        "import canopy\nroot = Path(canopy.__file__).resolve()\nroot.parent\n",
        "import canopy\ndef f():\n    root = Path(canopy.__file__)\ndef g(root):\n    return root.parents[2]\n",
    ],
)
def test_a_path_inside_a_package_or_outside_canopy_is_accepted(source):
    assert lines_leaving_the_packages(source) == []
