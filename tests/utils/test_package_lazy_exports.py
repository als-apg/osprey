"""Guards the export maps behind every lazily-exporting package under ``src/osprey``.

A lazily resolved name fails at the first attribute access that reaches it,
never at import: a name its module no longer defines is not an import error and
not a collection error, but an ``AttributeError`` raised in whichever process
touches that attribute first. The map behind the hook is therefore checked here
rather than discovered by whichever caller gets there first.
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterator
from importlib import import_module, invalidate_caches
from pathlib import Path

import pytest

#: Every package whose public names are resolved through a ``_LAZY_EXPORTS`` map.
LAZY_EXPORT_PACKAGES = (
    "osprey",
    "osprey.dispatch",
    "osprey.health",
    "osprey.interfaces.web_terminal",
    "osprey.services.ariel_search.database",
    "osprey.services.channel_finder.graph_index",
    "osprey.services.facility_knowledge.ontology_compiler",
    "osprey.services.virtual_accelerator.ioc",
    "osprey.services.virtual_accelerator.model",
)

#: Packages whose module ``__getattr__`` aliases submodules rather than the
#: symbols a module defines. There is no defining module to check a name
#: against — the target *is* the name — and the submodule aliased first imports
#: a Channel Access server extension published for one platform.
SUBMODULE_ALIAS_PACKAGES = frozenset({"osprey.services.virtual_accelerator.serving"})

_SRC = Path(__file__).resolve().parents[2] / "src"

_SYNTHETIC_PACKAGE = "mispointed_lazy_exports_pkg"


def _resolve(package: str, name: str) -> tuple[object, object]:
    """Read *name* from *package* and from the module its map names for it.

    Args:
        package: Dotted name of a package carrying a ``_LAZY_EXPORTS`` map.
        name: Public name to read from both.

    Returns:
        The object the package hands back and the object its named module holds.
    """
    pkg = import_module(package)
    module = import_module(pkg._LAZY_EXPORTS[name], package)
    return getattr(pkg, name), getattr(module, name)


def _map_entries() -> list[tuple[str, str]]:
    """Return every ``(package, mapped name)`` pair across the roster."""
    return [
        (package, name)
        for package in LAZY_EXPORT_PACKAGES
        for name in import_module(package)._LAZY_EXPORTS
    ]


def _public_names() -> list[tuple[str, str]]:
    """Return every ``(package, name in __all__)`` pair across the roster."""
    return [
        (package, name)
        for package in LAZY_EXPORT_PACKAGES
        for name in import_module(package).__all__
    ]


def _packages_defining_a_module_getattr() -> set[str]:
    """Return the dotted name of every package whose ``__init__`` defines ``__getattr__``.

    The source is parsed, never imported, so the walk costs nothing but the
    files it reads.

    Returns:
        Dotted package names, as they would be imported.
    """
    found = set()
    for path in sorted((_SRC / "osprey").rglob("__init__.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(
            isinstance(node, ast.FunctionDef) and node.name == "__getattr__" for node in tree.body
        ):
            found.add(".".join(path.relative_to(_SRC).parent.parts))
    return found


def _assigns_a_lazy_export_map(package: str) -> bool:
    """Report whether *package*'s ``__init__`` assigns ``_LAZY_EXPORTS`` at module level."""
    path = _SRC.joinpath(*package.split(".")) / "__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(target, ast.Name) and target.id == "_LAZY_EXPORTS" for target in targets):
            return True
    return False


@pytest.fixture
def mispointed_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Install a package whose map points a name at a module that lacks it.

    Yields:
        The importable name of the synthetic package.
    """
    package_dir = tmp_path / _SYNTHETIC_PACKAGE
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(
        "from typing import Any\n"
        "\n"
        '_LAZY_EXPORTS: dict[str, str] = {"thing": ".home"}\n'
        "\n"
        "\n"
        "def __getattr__(name: str) -> Any:\n"
        "    module_name = _LAZY_EXPORTS.get(name)\n"
        "    if module_name is None:\n"
        '        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")\n'
        "    from importlib import import_module\n"
        "\n"
        "    value = getattr(import_module(module_name, __name__), name)\n"
        "    globals()[name] = value\n"
        "    return value\n",
        encoding="utf-8",
    )
    (package_dir / "home.py").write_text("something_else = object()\n", encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    invalidate_caches()
    try:
        yield _SYNTHETIC_PACKAGE
    finally:
        for name in [
            name
            for name in sys.modules
            if name == _SYNTHETIC_PACKAGE or name.startswith(f"{_SYNTHETIC_PACKAGE}.")
        ]:
            del sys.modules[name]


@pytest.mark.parametrize(("package", "name"), _map_entries())
def test_every_entry_resolves_to_the_object_its_module_holds(package: str, name: str):
    """The package hands back the very object its named module holds."""
    from_package, from_module = _resolve(package, name)

    assert from_package is from_module


@pytest.mark.parametrize(("package", "name"), _public_names())
def test_every_public_name_is_reachable(package: str, name: str):
    """Every name in ``__all__`` is bound eagerly or carried by the map."""
    assert getattr(import_module(package), name) is not None


@pytest.mark.parametrize("package", LAZY_EXPORT_PACKAGES)
def test_an_unknown_name_raises_attribute_error(package: str):
    """The hook refuses a name the map does not carry, naming package and name."""
    expected = re.escape(f"module {package!r} has no attribute 'no_such_export'")

    with pytest.raises(AttributeError, match=expected):
        import_module(package).no_such_export


def test_the_check_fails_on_a_mispointed_entry(mispointed_package: str):
    """A map entry whose module lacks the name is what this file exists to catch."""
    with pytest.raises(AttributeError):
        _resolve(mispointed_package, "thing")


def test_every_lazily_exporting_package_declares_a_map():
    """Every package resolving names through a hook is on one of the two lists."""
    found = _packages_defining_a_module_getattr()

    assert set(LAZY_EXPORT_PACKAGES) | SUBMODULE_ALIAS_PACKAGES == found
    for package in LAZY_EXPORT_PACKAGES:
        assert _assigns_a_lazy_export_map(package), f"{package} declares no _LAZY_EXPORTS map"
