"""The application import boundary, checked over a package's source.

Application code reaches OSPREY through :data:`ALLOWED_MODULES` and nothing else. The
check reads source by path, so it covers what a ``pyproject.toml`` contract cannot name:
import-linter refuses a listed module that is absent from its graph, and a deployment's
package is not in it. It needs no development dependency, because ``osprey app check``
runs it in a deployment repository.

Every import statement counts, including ones inside functions and under
``if TYPE_CHECKING:``; an ``importlib.import_module("...")`` string does not. A package's
own modules and third-party imports are outside the rule.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

from osprey.apps.errors import Violation

ALLOWED_MODULES = ("osprey.apps.sdk", "osprey.facility.kinds", "osprey_connectors")
"""The modules application code may import, with their public submodules."""

_FRAMEWORK_ROOTS = frozenset(module.split(".")[0] for module in ALLOWED_MODULES)
_RULE_MESSAGE = (
    "application code imports OSPREY only through osprey.apps.sdk, osprey.facility.kinds "
    "and public osprey_connectors modules"
)


def _is_allowed(imported: str) -> bool:
    """Whether a dotted name is an allowed module or a public name under one."""
    if any(segment.startswith("_") for segment in imported.split(".")):
        return False
    return any(imported == root or imported.startswith(f"{root}.") for root in ALLOWED_MODULES)


def _absolute_imports(tree: ast.AST) -> Iterator[tuple[str, int]]:
    """Yield each absolute import's dotted target and line, ``module.name`` for a ``from``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            for alias in node.names:
                target = node.module if alias.name == "*" else f"{node.module}.{alias.name}"
                yield target, node.lineno


def _check_source(path: Path, package: Path, app: str) -> Iterator[Violation]:
    """Yield the violations of one source file of *package*."""
    relative = path.relative_to(package.parent)
    source = relative.as_posix()
    try:
        tree = ast.parse(path.read_bytes(), filename=source)
    except (OSError, SyntaxError, ValueError) as exc:
        yield Violation(app, "", "unreadable_source", f"cannot be checked: {exc}", source=source)
        return

    for imported, line in _absolute_imports(tree):
        if imported.split(".")[0] in _FRAMEWORK_ROOTS and not _is_allowed(imported):
            yield Violation(app, imported, "import_boundary", _RULE_MESSAGE, f"{source}:{line}")

    depth = len(relative.parts) - 1
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > depth:
            yield Violation(
                app,
                "." * node.level + (node.module or ""),
                "relative_import_escape",
                "a relative import stays inside the application's package; " + _RULE_MESSAGE,
                f"{source}:{node.lineno}",
            )


def check_package(package: Path, *, app: str = "") -> list[Violation]:
    """Check every Python source file under *package* against the import boundary.

    Args:
        package: The package's directory.
        app: The application the violations belong to, when the caller knows it.

    Returns:
        Every violation, by file and then in source order. A violation's ``field`` is the
        refused import and its ``source`` the file and line, relative to the package's
        parent directory.

    Raises:
        ValueError: *package* is not a directory, which would otherwise pass unchecked.
    """
    if not package.is_dir():
        raise ValueError(f"{package} is not a package directory")
    return [
        violation
        for path in sorted(package.rglob("*.py"))
        for violation in _check_source(path, package, app)
    ]
