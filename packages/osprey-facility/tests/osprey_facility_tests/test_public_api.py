"""The package's public API: the modules the README declares, and what they list.

The README's ``Public API`` table names every public module with the consumer
it serves. Each listed module imports, lists in ``__all__`` only names it has,
and lists every public name it defines. The schema accessor hands out both
shipped schema files, and the MML exporter accessor the shipped script.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
from importlib import metadata
from pathlib import Path
from types import ModuleType

import pytest

from osprey_facility.layers.mml import importer
from osprey_facility.schema import published_schema

#: The package's root directory: the README sits there.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]

#: A row of the README's public-module table.
_ROW = re.compile(r"^\|\s*`(osprey_facility(?:\.\w+)*)`\s*\|\s*(.+?)\s*\|\s*$")

#: The consumers a public module may serve (the framework job names its modules).
CONSUMERS = ("fork/tool", "kind plug-in", "app", "framework job: ")

#: Public modules the move left without an ``__all__``. Only shrinks.
WITHOUT_ALL = frozenset({"osprey_facility.pairing", "osprey_facility.relations"})

#: Public names a module defines and does not list, as the move left them. Only shrinks.
UNLISTED = frozenset(
    {
        "osprey_facility.kinds.CHROMATICITY_MONITOR",
        "osprey_facility.kinds.CRM",
        "osprey_facility.kinds.DISPERSION",
        "osprey_facility.kinds.ORM",
        "osprey_facility.kinds.PLAN_VALUES",
        "osprey_facility.kinds.TRM",
        "osprey_facility.kinds.block_admitted",
        "osprey_facility.kinds.judged_here",
        "osprey_facility.kinds.listed_kinds",
        "osprey_facility.layers.mml.mapping.DERIVED",
        "osprey_facility.layers.mml.mapping.ENGINE_BY_TYPE",
        "osprey_facility.layers.mml.mapping.IMPORTED",
        "osprey_facility.layers.mml.mapping.MONITOR_FIELD",
        "osprey_facility.layers.mml.mapping.SETPOINT_FIELD",
        "osprey_facility.layers.mml.seed.NAMED_ADDRESSES",
        "osprey_facility.records.DEFAULT_GROUP_KIND",
        "osprey_facility.references.ReferenceKind",
        "osprey_facility.render.serialised_document",
        "osprey_facility.render.serialised_file_document",
        "osprey_facility.sources.read_yaml",
    }
)


def public_modules() -> dict[str, str]:
    """Each public module the README's table declares, with its consumer."""
    text = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    return {match[1]: match[2] for line in text.splitlines() if (match := _ROW.match(line.strip()))}


PUBLIC = public_modules()


def _defined(module: ModuleType) -> set[str]:
    """The public names *module*'s own source binds at top level (imports excluded)."""
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {name for name in names if not name.startswith("_")}


def test_the_readme_declares_the_public_modules() -> None:
    assert "osprey_facility.schema" in PUBLIC
    assert "osprey_facility.layers.mml.importer" in PUBLIC
    assert "osprey_facility.kinds" in PUBLIC
    for module, consumer in PUBLIC.items():
        assert consumer.startswith(CONSUMERS), f"{module}: {consumer!r} names no consumer"


@pytest.mark.parametrize("name", sorted(PUBLIC))
def test_a_public_module_has_every_name_it_lists(name: str) -> None:
    module = importlib.import_module(name)
    listed = getattr(module, "__all__", None)
    if name in WITHOUT_ALL:
        assert listed is None, f"{name} lists its names now; remove it from WITHOUT_ALL"
        return
    assert listed is not None, f"{name} is public and has no __all__"
    missing = [entry for entry in listed if not hasattr(module, entry)]
    assert missing == []


@pytest.mark.parametrize("name", sorted(PUBLIC.keys() - WITHOUT_ALL))
def test_a_public_module_lists_every_public_name_it_defines(name: str) -> None:
    module = importlib.import_module(name)
    unlisted = {f"{name}.{entry}" for entry in _defined(module) - set(module.__all__)}
    owed = {entry for entry in UNLISTED if entry.rpartition(".")[0] == name}
    assert unlisted - owed == set()
    assert owed - unlisted == set(), "listed now; remove from UNLISTED"


def test_the_package_declares_its_version() -> None:
    package = importlib.import_module("osprey_facility")
    assert "__version__" in package.__all__
    try:
        installed = metadata.version("osprey-facility")
    except metadata.PackageNotFoundError:  # pragma: no cover — source tree without an install
        installed = "0.0.0+unbuilt"
    assert package.__version__ == installed
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    assert "`osprey_facility.__version__`" in readme, "the README does not declare the version"


def test_the_accessor_returns_the_current_schema() -> None:
    schema = published_schema()
    assert schema.name == "facility.schema.json"
    assert schema.is_file()
    assert isinstance(json.loads(schema.read_text(encoding="utf-8")), dict)


def test_the_accessor_returns_a_version_floor() -> None:
    schema = published_schema(4)
    assert schema.name == "facility-4.schema.json"
    assert schema.is_file()
    assert isinstance(json.loads(schema.read_text(encoding="utf-8")), dict)


@pytest.mark.parametrize("version", [0, 3, 999, -4, "4", 4.0, True])
def test_the_accessor_refuses_an_unknown_version(version: object) -> None:
    with pytest.raises(LookupError, match="no JSON Schema"):
        published_schema(version)  # type: ignore[arg-type]


def test_the_exporter_accessor_returns_the_shipped_script() -> None:
    shipped = Path(importer.__file__).with_name("mml_export.m").read_text(encoding="utf-8")
    script = importer.exporter_script()
    assert script == shipped
    assert "function" in script
