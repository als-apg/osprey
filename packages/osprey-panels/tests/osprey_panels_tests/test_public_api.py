"""The package's public API: the modules the README declares, and what they list.

The README's ``Public API`` table names every public module with the one
consumer it serves. Each listed module imports, lists in ``__all__`` only names
it has, and lists every public name it defines. The command's module lists the
entry point the ``osprey-panels`` console script names, and the Screen-IR module
hands out the vendored schema.
"""

from __future__ import annotations

import ast
import importlib
import re
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

from osprey_panels import screen_ir
from osprey_panels_tests._harness._paths import PACKAGE_ROOT, PACKAGE_SRC

#: A row of the README's public-module table.
_ROW = re.compile(r"^\|\s*`(osprey_panels(?:\.\w+)*)`\s*\|\s*(.+?)\s*\|\s*$")

#: The consumers a public module may serve (a contract test names its file).
CONSUMERS = ("fork/tool", "command", "contract test: ")

#: How many public names each module defines and does not list, as the move
#: left them. Only shrinks: a module that defines a new public name lists it in
#: ``__all__`` or makes it private, and one that lists or privatises an owed name
#: lowers its count here. Counted, not named: the engine modules the contract
#: test reads carry several hundred such names, most of them drawing constants.
UNLISTED: dict[str, int] = {
    "osprey_panels.cli": 3,
    "osprey_panels.describe": 65,
    "osprey_panels.layout": 294,
    "osprey_panels.model": 6,
    "osprey_panels.symbols": 42,
    "osprey_panels.view": 6,
}


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
    assert "osprey_panels" in PUBLIC
    assert "osprey_panels.cli" in PUBLIC
    assert "osprey_panels.document" in PUBLIC
    assert "osprey_panels.screen_ir" in PUBLIC
    for module, consumer in PUBLIC.items():
        assert consumer.startswith(CONSUMERS), f"{module}: {consumer!r} names no consumer"


@pytest.mark.parametrize(
    ("module", "consumer"),
    sorted((m, c) for m, c in PUBLIC.items() if c.startswith("contract test: ")),
)
def test_a_contract_test_consumer_names_one_test_file(module: str, consumer: str) -> None:
    named = re.fullmatch(r"contract test: `([^`]+\.py)`", consumer)
    assert named, f"{module}: {consumer!r} names no single test file"
    assert named[1].startswith("tests/"), f"{module}: {named[1]} is not a root test"


@pytest.mark.parametrize("name", sorted(PUBLIC))
def test_a_public_module_has_every_name_it_lists(name: str) -> None:
    module = importlib.import_module(name)
    listed = getattr(module, "__all__", None)
    assert listed is not None, f"{name} is public and has no __all__"
    missing = [entry for entry in listed if not hasattr(module, entry)]
    assert missing == []


@pytest.mark.parametrize("name", sorted(PUBLIC))
def test_a_public_module_lists_every_public_name_it_defines(name: str) -> None:
    module = importlib.import_module(name)
    unlisted = sorted(_defined(module) - set(module.__all__))
    owed = UNLISTED.get(name, 0)
    assert len(unlisted) <= owed, (
        f"{name} defines {len(unlisted)} public names it does not list, {owed} owed; "
        f"list the new ones in __all__ or make them private: {unlisted}"
    )
    assert len(unlisted) == owed, f"{name} owes {len(unlisted)} now; lower UNLISTED to it"


def test_every_unlisted_entry_names_a_public_module() -> None:
    stray = set(UNLISTED) - set(PUBLIC)
    assert stray == set(), "not a public module; remove from UNLISTED"


def test_the_console_script_names_a_listed_entry_point() -> None:
    pyproject = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    target = pyproject["project"]["scripts"]["osprey-panels"]
    module, _, attribute = target.partition(":")
    assert module in PUBLIC, f"the console script names {module}, which is not public"
    assert attribute in importlib.import_module(module).__all__


def test_the_schema_accessor_returns_the_vendored_schema() -> None:
    schema = screen_ir.schema()
    vendored = PACKAGE_SRC / "osprey_panels" / "screen-ir.schema.json"
    assert vendored.is_file()
    assert isinstance(schema, dict)
    assert schema.get("$schema"), "the vendored file is not a JSON Schema"
