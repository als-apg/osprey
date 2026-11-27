"""The application import boundary, checked over a package's source.

Application code reaches OSPREY through :data:`ALLOWED_MODULES` and nothing else. The
check reads source by path, so it covers what a ``pyproject.toml`` contract cannot name:
import-linter refuses a listed module that is absent from its graph, and a deployment's
package is not in it. It needs no development dependency, because ``osprey app check``
runs it in a deployment repository.

Every import statement counts, including ones inside functions and under
``if TYPE_CHECKING:``; an ``importlib.import_module("...")`` string does not. A package's
own modules and third-party imports are outside the rule.

An allowed import must also resolve: ``import_missing_module`` names a module that does
not exist and ``import_missing_name`` a name its module does not define. Resolving imports
the framework module, never application code, relative imports or third-party modules; an
import that fails for any reason other than a missing module is skipped. No application package lives
under ``osprey``, ``osprey_facility`` or ``osprey_connectors``: an import under any of them
is OSPREY's, whatever package holds it. Of ``osprey_facility`` only ``kinds`` is open to an
application; its records, readers and every other module are refused.

:func:`check_apps` runs the check over every package of a deployment's applications; both
``osprey build`` and ``osprey app check`` call it, so they refuse the same imports.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.util
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from osprey.apps.errors import Violation
from osprey.apps.origin import AppOrigin, origin_of

if TYPE_CHECKING:
    from osprey.apps.manifest import AppManifest

ALLOWED_MODULES = ("osprey.apps.sdk", "osprey_facility.kinds", "osprey_connectors")
"""The modules application code may import, with their public submodules."""

_FRAMEWORK_ROOTS = frozenset(module.split(".")[0] for module in ALLOWED_MODULES)
_RULE_MESSAGE = (
    "application code imports OSPREY only through osprey.apps.sdk, osprey_facility.kinds "
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


def _framework_imports(tree: ast.AST) -> Iterator[tuple[str, str | None, int]]:
    """Yield each absolute import as ``(module, name, line)``; ``name`` is ``None`` for
    ``import X`` and a star import."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, None, node.lineno
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            for alias in node.names:
                yield node.module, None if alias.name == "*" else alias.name, node.lineno


def _absent(module: str) -> bool | None:
    """Whether *module* does not exist: ``True`` missing, ``False`` present, ``None``
    when resolving it failed for another reason."""
    try:
        return importlib.util.find_spec(module) is None
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if module == missing or module.startswith(f"{missing}."):
            return True
        return None
    except Exception:  # any other failure skips the check
        return None


class _Resolver:
    """Resolve framework imports once per module and name within one check."""

    def __init__(self) -> None:
        self._absent: dict[str, bool | None] = {}
        self._missing_name: dict[tuple[str, str], bool] = {}

    def module_absent(self, module: str) -> bool | None:
        if module not in self._absent:
            self._absent[module] = _absent(module)
        return self._absent[module]

    def name_missing(self, module: str, name: str) -> bool:
        key = (module, name)
        if key not in self._missing_name:
            self._missing_name[key] = self._resolve_name(module, name)
        return self._missing_name[key]

    def _resolve_name(self, module: str, name: str) -> bool:
        try:
            imported = importlib.import_module(module)
        except Exception:  # any failure to import skips the check
            return False
        if hasattr(imported, name):
            return False
        return self.module_absent(f"{module}.{name}") is True


def _resolution_violations(
    tree: ast.AST, source: str, app: str, resolver: _Resolver
) -> Iterator[Violation]:
    """Yield an allowed framework import that names a missing module or name."""
    for module, name, line in _framework_imports(tree):
        target = module if name is None else f"{module}.{name}"
        if module.split(".")[0] not in _FRAMEWORK_ROOTS or not _is_allowed(target):
            continue
        absent = resolver.module_absent(module)
        if absent is None:
            continue
        if absent:
            yield Violation(
                app,
                module,
                "import_missing_module",
                "no such module; " + _RULE_MESSAGE,
                f"{source}:{line}",
            )
        elif name is not None and resolver.name_missing(module, name):
            yield Violation(
                app,
                target,
                "import_missing_name",
                f"{module} has no name {name!r}; " + _RULE_MESSAGE,
                f"{source}:{line}",
            )


def _check_source(path: Path, package: Path, app: str, resolver: _Resolver) -> Iterator[Violation]:
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

    yield from _resolution_violations(tree, source, app, resolver)

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
            A relative import is measured from *package*.

    Returns:
        Every violation, by file and then in source order. A violation's ``field`` is the
        refused import and its ``source`` the file and line, relative to the package's
        parent directory.

    Raises:
        ValueError: *package* is not a directory, which would otherwise pass unchecked.
    """
    if not package.is_dir():
        raise ValueError(f"{package} is not a package directory")
    resolver = _Resolver()
    return [
        violation
        for path in sorted(package.rglob("*.py"))
        for violation in _check_source(path, package, app, resolver)
    ]


@dataclasses.dataclass(frozen=True)
class CheckedApp:
    """One application the per-application checks judge.

    Attributes:
        name: The application's name, which its violations carry.
        origin: Where its directory is; a report names a file through
            :meth:`~osprey.apps.origin.AppOrigin.display`.
    """

    name: str
    origin: AppOrigin

    @property
    def directory(self) -> Path:
        """The application's directory."""
        return self.origin.directory


def checked_apps(manifests: Iterable[AppManifest], root: Path) -> list[CheckedApp]:
    """Each of *manifests* with the directory it was read from, under deployment *root*."""
    return [CheckedApp(manifest.name, origin_of(manifest, root)) for manifest in manifests]


def check_apps(apps: Iterable[CheckedApp]) -> tuple[list[Violation], list[str]]:
    """The import boundary over each importable package of each of *apps*.

    A deployment application's packages are its top-level directories holding an
    ``__init__.py``; an installed application's directory is itself its package.

    Returns:
        The violations, by application, package, file and source order, each naming its
        file through the application's origin; and the notes for the caller to print,
        one per deployment application with no Python package.
    """
    found: list[Violation] = []
    notes: list[str] = []
    for checked in apps:
        origin = checked.origin
        if origin.installed:
            found += [
                dataclasses.replace(
                    violation,
                    source=origin.display(_below(violation.source, checked.directory.name)),
                )
                for violation in check_package(checked.directory, app=checked.name)
            ]
            continue
        packages = _packages(checked.directory)
        if not packages:
            notes.append(f"{origin.display()}/: no Python package, no import boundary to check")
        found += [
            dataclasses.replace(violation, source=origin.display(violation.source or ""))
            for package in packages
            for violation in check_package(package, app=checked.name)
        ]
    return found, notes


def _below(source: str | None, top: str) -> str:
    """*source*, a path under the directory *top*, relative to that directory."""
    return (source or "").removeprefix(f"{top}/")


def _packages(app_dir: Path) -> list[Path]:
    """The top-level directories of an application that hold an ``__init__.py``."""
    return [
        entry
        for entry in sorted(app_dir.iterdir(), key=lambda path: path.name)
        if not entry.name.startswith(".")
        and entry.name != "__pycache__"
        and (entry / "__init__.py").is_file()
    ]
