"""The connectors package's public API is exactly what its ``__all__`` lists say.

``control_system``, ``archiver`` and ``ipc`` gather underscore-private
implementation modules behind one ``__init__``. The other public modules of the
package, the ``simulation`` package's among them, are their own API and list it
in their own ``__all__``. A name is public when a list names it and private when it starts
with an underscore; nothing is in between. Which modules the framework may
import is held by ``lint-imports`` (``[tool.importlinter]`` in the root
``pyproject.toml``); which names it may take from them is held here.
"""

from __future__ import annotations

import ast
import functools
import importlib
import importlib.util
import inspect
import os
import re
import subprocess
import sys
import tomllib
import typing
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "packages" / "osprey-connectors" / "src"
ROOT = PACKAGE / "osprey_connectors"
FRAMEWORK = REPO / "src" / "osprey"

NAMESPACES = (
    "osprey_connectors.control_system",
    "osprey_connectors.archiver",
    "osprey_connectors.ipc",
)

#: The namespaces that resolve some names on first access (PEP 562).
LAZY_NAMESPACES = ("osprey_connectors.control_system", "osprey_connectors.ipc")

#: The public modules that are their own API.
MODULES = (
    tuple(
        f"osprey_connectors.{path.stem}"
        for path in sorted(ROOT.glob("*.py"))
        if not path.stem.startswith("_")
    )
    + ("osprey_connectors.control_system.write_door", "osprey_connectors.simulation")
    + tuple(
        f"osprey_connectors.simulation.{path.stem}"
        for path in sorted((ROOT / "simulation").glob("*.py"))
        if not path.stem.startswith("_")
    )
)

#: A module's own logger is module plumbing, not API, and is conventionally
#: spelled without an underscore.
NOT_API = frozenset({"logger"})


def _source(module: str) -> Path:
    base = PACKAGE.joinpath(*module.split("."))
    return base / "__init__.py" if base.is_dir() else base.with_suffix(".py")


def _defined_names(module: str) -> set[str]:
    """Underscore-free names a module defines at top level (not ones it imports)."""
    names: set[str] = set()
    for node in ast.parse(_source(module).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {name for name in names if not name.startswith("_")} - NOT_API


def _public_owner(obj: object) -> bool:
    """Whether some public module or namespace lists *obj* under its own name."""
    name = getattr(obj, "__name__", None)
    for module in NAMESPACES + MODULES:
        loaded = importlib.import_module(module)
        if name in loaded.__all__ and getattr(loaded, name) is obj:
            return True
    return False


# -- Every listed name is there, and every defined name is listed ------------


@pytest.mark.parametrize("module", NAMESPACES + MODULES)
def test_every_listed_name_is_there(module: str):
    loaded = importlib.import_module(module)

    assert loaded.__all__, f"{module} lists nothing"
    missing = [name for name in loaded.__all__ if not hasattr(loaded, name)]
    assert not missing, f"{module} lists names it does not define: {missing}"


@pytest.mark.parametrize("module", MODULES)
def test_every_name_a_public_module_defines_is_listed_or_private(module: str):
    # An underscore-free name left out of the list is neither: callers take it
    # as public, and PLC2701 cannot tell them otherwise.
    unlisted = sorted(_defined_names(module) - set(importlib.import_module(module).__all__))

    assert not unlisted, f"{module}: list these in __all__ or give them an underscore: {unlisted}"


@pytest.mark.parametrize("namespace", LAZY_NAMESPACES)
def test_a_lazy_namespace_maps_what_it_lists(namespace: str):
    loaded = importlib.import_module(namespace)
    tree = ast.parse(_source(namespace).read_text(encoding="utf-8"))
    type_checking = {
        alias.asname or alias.name: stmt.module
        for node in tree.body
        if isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING"
        for stmt in node.body
        if isinstance(stmt, ast.ImportFrom)
        for alias in stmt.names
    }

    assert set(loaded._LAZY_EXPORTS) <= set(loaded.__all__)
    assert type_checking == loaded._LAZY_EXPORTS
    for name, home in loaded._LAZY_EXPORTS.items():
        assert getattr(loaded, name) is getattr(importlib.import_module(home), name)


# -- The framework takes only listed names -----------------------------------


@functools.cache
def _framework_imports() -> tuple[tuple[str, str, str], ...]:
    """``(file:line, module, name)`` for every name src/osprey takes from the package.

    Two tests read the same scan, and the tree does not change while the suite
    runs, so it is made once.
    """
    found: list[tuple[str, str, str]] = []
    for path in sorted(FRAMEWORK.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        # Every name the scan reports is reached through an import of the
        # package, and an import spells the package's name in the source (a
        # module path is identifiers, which have no escape sequence). Most of
        # the framework imports nothing from it, and parsing is what the scan
        # spends its time on.
        if "osprey_connectors" not in source:
            continue
        tree = ast.parse(source)
        aliases: dict[str, str] = {}
        where = path.relative_to(REPO)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if not node.module.startswith("osprey_connectors"):
                    continue
                for alias in node.names:
                    full = f"{node.module}.{alias.name}"
                    if importlib.util.find_spec(node.module) and _is_module(full):
                        aliases[alias.asname or alias.name] = full
                    else:
                        found.append((f"{where}:{node.lineno}", node.module, alias.name))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("osprey_connectors") and alias.asname:
                        aliases[alias.asname] = alias.name
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                module = aliases.get(node.value.id)
                if module is not None and not _is_module(f"{module}.{node.attr}"):
                    found.append((f"{where}:{node.lineno}", module, node.attr))
    return tuple(found)


def _is_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


@pytest.mark.skipif(not FRAMEWORK.is_dir(), reason="reads the framework's source tree")
def test_the_framework_takes_only_listed_names():
    # A module this tree does not have is a spelling an older release used,
    # imported as a fallback (the rendered limits hook's, for one); the import
    # contracts and the hook's own tests answer for it.
    unlisted = [
        f"{where}: {module}.{name}"
        for where, module, name in _framework_imports()
        if _is_module(module)
        and name not in getattr(importlib.import_module(module), "__all__", ())
    ]

    assert not unlisted, "the framework uses names its module does not list:\n" + "\n".join(
        unlisted
    )


def test_the_scan_sees_the_framework_imports():
    # A scan that finds nothing would pass the test above vacuously.
    assert len(_framework_imports()) > 300


# -- Public signatures name only public types -------------------------------

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _annotation_names(module: object, annotation: object) -> set[object]:
    """The package objects an annotation names, resolved in *module*'s namespace."""
    if annotation is inspect.Parameter.empty:
        return set()
    found: set[object] = set()
    if isinstance(annotation, str):
        scope = {**vars(module), **_type_checking_names(module)}
        for identifier in _IDENTIFIER.findall(annotation):
            obj = scope.get(identifier)
            if obj is not None:
                found.add(obj)
    else:
        if inspect.isclass(annotation):
            found.add(annotation)
        for arg in typing.get_args(annotation):
            found |= _annotation_names(module, arg)
    return {
        obj
        for obj in found
        if inspect.isclass(obj)
        and isinstance(getattr(obj, "__module__", None), str)
        and obj.__module__.startswith("osprey_connectors")
    }


@functools.cache
def _type_checking_names(module: object) -> dict[str, object]:
    """The names *module* imports under ``if TYPE_CHECKING:``, resolved.

    Every string annotation of the module asks for them, and parsing its source
    again for each one was most of what the signature test spent.
    """
    names: dict[str, object] = {}
    source = inspect.getsourcefile(module)  # type: ignore[arg-type]
    if source is None:
        return names
    for node in ast.parse(Path(source).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING":
            for stmt in node.body:
                if isinstance(stmt, ast.ImportFrom) and stmt.module:
                    imported = importlib.import_module(stmt.module)
                    for alias in stmt.names:
                        names[alias.asname or alias.name] = getattr(imported, alias.name)
    return names


def _public_callables() -> list[tuple[str, object, object]]:
    """``(label, defining module, callable)`` for every public function, class and method."""
    found: list[tuple[str, object, object]] = []
    for module_name in NAMESPACES + MODULES:
        loaded = importlib.import_module(module_name)
        for name in loaded.__all__:
            obj = getattr(loaded, name)
            home = sys.modules.get(getattr(obj, "__module__", ""), loaded)
            if inspect.isclass(obj):
                for member_name, member in vars(obj).items():
                    if member_name.startswith("_") and member_name != "__init__":
                        continue
                    func = member.fget if isinstance(member, property) else member
                    func = getattr(func, "__func__", func)
                    if inspect.isfunction(func):
                        found.append((f"{module_name}.{name}.{member_name}", home, func))
            elif inspect.isfunction(obj):
                found.append((f"{module_name}.{name}", home, obj))
    return found


def test_public_signatures_name_only_public_types():
    private = []
    for label, home, func in _public_callables():
        signature = inspect.signature(func)
        annotations = [p.annotation for p in signature.parameters.values()]
        annotations.append(signature.return_annotation)
        for annotation in annotations:
            for obj in _annotation_names(home, annotation):
                if not _public_owner(obj):
                    private.append(f"{label}: {obj.__module__}.{obj.__qualname__}")

    assert not private, "public signatures name types no module lists:\n" + "\n".join(
        sorted(set(private))
    )


# -- What the namespaces load ------------------------------------------------


def _loaded_after(statement: str) -> list[str]:
    code = f"import sys; {statement}; print('\\n'.join(sorted(sys.modules)))"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(PACKAGE)),
        timeout=60,
        check=True,
    )
    return result.stdout.split()


def test_importing_the_ipc_namespace_loads_only_the_codec():
    # The connector-host child runs as `python -m osprey_connectors.ipc._host`;
    # a package that imported `_host` itself would load it twice. The codec and
    # its exception registry are pure and are imported eagerly.
    loaded = _loaded_after("import osprey_connectors.ipc")

    assert [m for m in loaded if m.startswith("osprey_connectors.ipc._")] == [
        "osprey_connectors.ipc._exceptions",
        "osprey_connectors.ipc._frames",
    ]


def test_the_write_door_does_not_load_the_limits_validator():
    # The raw-put guard reads the door inside executor sandboxes and kernels;
    # the validator would bring its logging and YAML stack into every one.
    loaded = _loaded_after("import osprey_connectors.control_system.write_door")

    assert "osprey_connectors.control_system._limits_validator" not in loaded
    assert "rich" not in loaded


# -- Layout the import contracts rely on -------------------------------------


def test_every_package_directory_is_a_package():
    # import-linter builds its graph from packages; a directory of modules with
    # no __init__ would be invisible to every contract.
    missing = sorted(
        str(directory.relative_to(PACKAGE))
        for directory in {path.parent for path in ROOT.rglob("*.py")}
        if "__pycache__" not in directory.parts and not (directory / "__init__.py").is_file()
    )

    assert not missing, f"directories without an __init__.py: {missing}"


def test_every_private_top_level_module_is_protected():
    # The modules of the namespaces are covered by `<namespace>.*`; a private
    # module at the package's top level has to be named in the contract.
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    contract = next(
        c for c in config["tool"]["importlinter"]["contracts"] if c["id"] == "connectors-public-api"
    )
    # `_version` is written by the build and gitignored, so it is absent on a
    # fresh clone, and import-linter refuses a listed module the graph lacks.
    # Nothing imports it; the package answers its version from its metadata.
    private = {
        f"osprey_connectors.{path.stem}"
        for path in ROOT.glob("_*.py")
        if path.stem not in {"__init__", "_version"}
    }

    assert private <= set(contract["protected_modules"]), sorted(
        private - set(contract["protected_modules"])
    )


def test_every_top_level_module_has_a_layer():
    # `connectors-layers` orders the package's top-level modules; one left out
    # of every layer is held to nothing, so an upward import from it, or into
    # it, passes unseen. A module added to the package has to be given a layer.
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    contract = next(
        c for c in config["tool"]["importlinter"]["contracts"] if c["id"] == "connectors-layers"
    )
    layered = {
        name.strip()
        for layer in contract["layers"]
        for name in re.split(r"[|:]", layer)
        if name.strip()
    }
    # `_version` is written by the build and gitignored (see above); nothing
    # imports it.
    top_level = {
        f"osprey_connectors.{path.stem}"
        for path in ROOT.glob("*.py")
        if path.stem not in {"__init__", "_version"}
    } | {f"osprey_connectors.{path.parent.name}" for path in ROOT.glob("*/__init__.py")}

    assert top_level <= layered, sorted(top_level - layered)
