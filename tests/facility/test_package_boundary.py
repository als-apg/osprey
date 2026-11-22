"""The facility description and the panel engine keep to the imports of their packages.

``osprey.facility`` will leave the framework as two distributions: the description
(``osprey-facility``) and the panel engine (``osprey-panels``). Which module goes where is
written once, in the import-linter contracts of ``pyproject.toml``; this module reads the
sides from there and checks the two edge classes import-linter cannot see:

* **A1, third-party imports.** import-linter only follows imports inside its root packages,
  so an import of any other distribution is invisible to it. Each package may import the
  standard library, the root packages import-linter already checks, and its own ruled
  dependency set (:data:`DESCRIPTION_DEPENDENCIES`, :data:`PANELS_DEPENDENCIES`).
* **A2, dynamic reaches.** ``importlib.import_module("...")``,
  ``importlib.resources.files("...")`` and a path climbed from ``__file__`` reach a module
  without an import statement. Such a reach is held to the contract row it crosses, the
  same rule a static import of that module would meet. The whole ``src/osprey`` tree is
  scanned, since framework modules reach the two packages too; a file is parsed only when
  its text spells one of the reaches, which keeps the scan fast.

Today's edges are ratchets (:data:`THIRD_PARTY_RATCHET`, :data:`DYNAMIC_RATCHET`), each
entry naming its ruling and the epic phase that removes it. A new edge fails, and so does
an entry whose edge is gone: the lists only shrink.

The completeness half keeps the contracts themselves whole:

* every module under ``src/osprey`` (bar the build-written ``osprey._version``) is on
  exactly one side, read with the contracts' ``as_packages = false`` semantics. A
  top-level ``osprey.*`` wildcard would be skipped by import-linter's overlap rule, so
  the framework is listed module by module, and a new module or package fails until a
  list names it;
* the framework list is the same in the three contracts that spell it;
* every ``ignore_imports`` entry of the facility contracts sits under a ``# phase N``
  header naming an epic phase (2-6), read from the raw text since TOML drops comments,
  and none uses a wildcard;
* the public modules ``facility-public-api`` declares in its comment exist, are
  description modules and are not protected, and together with ``protected_modules``
  they cover the description exactly once.

The scan functions take the source root as an argument, so the detection tests can run
them on a temporary tree.
"""

from __future__ import annotations

import ast
import functools
import os
import re
import sys
import tomllib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"

#: Third-party import roots the description may use (ruling 3), each with its
#: distribution. The mml import layer additionally gets :data:`MML_DEPENDENCIES`.
DESCRIPTION_DEPENDENCIES: Mapping[str, str] = {
    "pydantic": "pydantic (ruling 3)",
    "yaml": "PyYAML (ruling 3)",
    "packaging": "packaging (ruling 3)",
    "numpy": "numpy (ruling 3)",
}

#: The ``mml`` extra of the description (ruling 3), for ``osprey.facility.layers.mml``.
MML_DEPENDENCIES: Mapping[str, str] = {
    "scipy": "scipy (ruling 3, mml extra)",
    "at": "accelerator-toolbox (ruling 3, mml extra)",
}

#: The description modules the ``mml`` extra serves.
MML_PREFIX = "osprey.facility.layers.mml"

#: Third-party import roots the panel engine may use: the standard library plus the
#: validator of its pinned Screen-IR schema (ruling 8).
PANELS_DEPENDENCIES: Mapping[str, str] = {
    "jsonschema": "jsonschema (ruling 8)",
}

#: Today's A1 edges, ``importer -> import root``, with ruling and removing phase.
#: The click edges go when the description raises plain errors and the framework's
#: command line turns them into click ones (ruling 3).
_CLICK = "ruling 3: the description raises plain errors, the CLI adapts them; phase 3"

THIRD_PARTY_RATCHET: Mapping[str, str] = {
    "osprey.facility.errors -> click": _CLICK,
    "osprey.facility.layers -> click": _CLICK,
    "osprey.facility.layers.list -> click": _CLICK,
    "osprey.facility.layers.mml.importer -> click": _CLICK,
    "osprey.facility.layers.mml.loaders.json_any -> click": _CLICK,
    "osprey.facility.layers.mml.loaders.mat -> click": _CLICK,
    "osprey.facility.layers.mml.mapping -> click": _CLICK,
    "osprey.facility.layers.mml.systems -> click": _CLICK,
    "osprey.facility.validate -> click": _CLICK,
    "osprey.facility.kinds -> pyaml_cs_osprey": "ruling 3: the kind name becomes the "
    "description's own; phase 3",
}

#: Today's A2 edges, ``importer -> target [spelling]``, with row, ruling and removing phase.
DYNAMIC_RATCHET: Mapping[str, str] = {
    "osprey.facility.panels.emit -> osprey.templates.apps.demo [__file__]": (
        "S4, ruling 7: the caller passes the theme file in; phase 5"
    ),
    "osprey.cli.facility_cmd -> osprey.facility.layers.mml [files]": (
        "S6, ruling 5: the framework uses the description's public modules; phase 4"
    ),
}

#: Substrings one of which a file must contain to be parsed for A2.
_DYNAMIC_SPELLINGS = ("import_module", "__import__", "files(", "__file__")

DESCRIPTION, PANELS, FRAMEWORK = ("description", "panels", "framework")


# --- the sides, read from the contracts ------------------------------------------------


def _matches(module: str, pattern: str) -> bool:
    """Whether ``module`` is named by a contract entry (``as_packages = false``)."""
    if pattern.endswith(".**"):
        return module.startswith(pattern[:-2])
    if pattern.endswith(".*"):
        base = pattern[:-1]
        return module.startswith(base) and "." not in module[len(base) :]
    return module == pattern


def _named(module: str, patterns: Iterable[str]) -> bool:
    return any(_matches(module, pattern) for pattern in patterns)


@dataclass(frozen=True)
class Sides:
    """The module lists of the facility contracts.

    Attributes:
        description: Modules of the description (``facility-description-standalone``).
        panels: Modules of the panel engine (``panels-document-only``).
        framework: Modules of the framework (``framework-without-panels``).
        non_public: Description modules the framework may not reach
            (``facility-public-api``).
        roots: import-linter's root packages, whose static imports it checks itself.
    """

    description: tuple[str, ...]
    panels: tuple[str, ...]
    framework: tuple[str, ...]
    non_public: tuple[str, ...]
    roots: tuple[str, ...]

    def naming(self, module: str) -> list[str]:
        """Every side whose contract list names ``module``; one, when the lists are whole."""
        return [
            side
            for side, patterns in (
                (DESCRIPTION, self.description),
                (PANELS, self.panels),
                (FRAMEWORK, self.framework),
            )
            if _named(module, patterns)
        ]

    def of(self, module: str) -> str | None:
        """The side ``module`` is on, or None when it is on none."""
        return next(iter(self.naming(module)), None)


PYPROJECT = REPO / "pyproject.toml"


def _importlinter(pyproject: Path = PYPROJECT) -> dict:
    """The ``[tool.importlinter]`` table of ``pyproject``."""
    return tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["importlinter"]


def _contracts(config: dict | None = None) -> dict[str, dict]:
    """The import-linter contracts by id, of ``config`` or else of ``pyproject.toml``."""
    return {contract["id"]: contract for contract in (config or _importlinter())["contracts"]}


def read_sides(pyproject: Path = PYPROJECT) -> Sides:
    """The sides as the contracts in ``pyproject`` list them."""
    config = _importlinter(pyproject)
    contracts = _contracts(config)
    wanted = (
        "facility-description-standalone",
        "panels-document-only",
        "framework-without-panels",
        "facility-public-api",
    )
    missing = [cid for cid in wanted if cid not in contracts]
    assert not missing, f"pyproject.toml has no import-linter contract {missing}"
    description = contracts["facility-description-standalone"]
    return Sides(
        description=tuple(description["source_modules"]),
        panels=tuple(contracts["panels-document-only"]["source_modules"]),
        framework=tuple(contracts["framework-without-panels"]["source_modules"]),
        non_public=tuple(contracts["facility-public-api"]["protected_modules"]),
        roots=tuple(config["root_packages"]),
    )


# --- reading a module --------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class Reach:
    """One place a module reaches another.

    Attributes:
        module: The module that reaches.
        line: The line it does so on.
        target: The module it reaches; for an import statement the module imported from.
        how: ``import`` for an import statement, else the dynamic spelling.
    """

    module: str
    line: int
    target: str
    how: str = "import"

    @property
    def edge(self) -> str:
        """The ratchet key of this reach."""
        if self.how == "import":
            return f"{self.module} -> {self.target.split('.')[0]}"
        return f"{self.module} -> {self.target} [{self.how}]"

    def __str__(self) -> str:
        return f"{self.module}:{self.line} -> {self.target}  [{self.how}]"


def _module_of(src: Path, path: Path) -> str | None:
    """The dotted name of the deepest module or package ``path`` lies in, under ``src``."""
    try:
        parts = list(path.relative_to(src).parts)
    except ValueError:
        return None
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
        if parts[-1] == "__init__":
            parts.pop()
        elif (src.joinpath(*parts).with_suffix(".py")).is_file():
            return ".".join(parts)
    while parts and not (src.joinpath(*parts) / "__init__.py").is_file():
        parts.pop()
    return ".".join(parts) or None


def _climbed(node: ast.AST) -> int:
    """Directories ``node`` climbs from ``__file__``, or 0 when it is no such path.

    ``.parent`` climbs one, ``.parents[k]`` climbs ``k + 1`` and ``os.path.dirname``
    one; ``Path(__file__).parent`` is the module's own directory.
    """
    levels = 0
    while True:
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            levels += 1
            node = node.value
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "parents"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
        ):
            levels += node.slice.value + 1
            node = node.value.value
        elif (
            isinstance(node, ast.Call)
            and ast.unparse(node.func).rsplit(".", 1)[-1] == "dirname"
            and len(node.args) == 1
        ):
            levels += 1
            node = node.args[0]
        else:
            break
    if not levels:
        return 0
    named = any(isinstance(n, ast.Name) and n.id == "__file__" for n in ast.walk(node))
    return levels if named else 0


def _climb(node: ast.AST) -> tuple[int, list[str], bool] | None:
    """A path climbed from ``__file__``: levels, the constant parts joined on, and
    whether a non-constant part ended the walk down."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        parts = [node.right]
        base: ast.AST = node.left
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "joinpath"
    ):
        parts = list(node.args)
        base = node.func.value
    else:
        levels = _climbed(node)
        return (levels, [], False) if levels else None
    inner = _climb(base)
    if inner is None:
        return None
    levels, tail, open_ = inner
    for part in parts:
        if open_ or not (isinstance(part, ast.Constant) and isinstance(part.value, str)):
            open_ = True
            continue
        tail = [*tail, *part.value.split("/")]
    return levels, tail, open_


class _Reaches(ast.NodeVisitor):
    """Collects the reaches of one module, import statements included or not."""

    def __init__(self, src: Path, path: Path, module: str, text: str, *, imports: bool) -> None:
        self.src = src
        self.path = path
        self.module = module
        self.package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        self.imports = imports
        self.climbs = "__file__" in text
        self.found: list[Reach] = []
        self._seen: set[int] = set()

    def visit_Import(self, node: ast.Import) -> None:
        if self.imports:
            for alias in node.names:
                self.found.append(Reach(self.module, node.lineno, alias.name))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self.imports and not node.level and node.module:
            self.found.append(Reach(self.module, node.lineno, node.module))

    def visit_Call(self, node: ast.Call) -> None:
        called = ast.unparse(node.func).rsplit(".", 1)[-1]
        first = node.args[0] if node.args else None
        if (
            called in ("files", "import_module", "__import__")
            and isinstance(first, ast.Constant)
            and isinstance(first.value, str)
        ):
            target = first.value
            if called == "import_module" and target.startswith("."):
                package = node.args[1] if len(node.args) > 1 else None
                if not (isinstance(package, ast.Constant) and isinstance(package.value, str)):
                    package = None
                base = package.value if package else self.package
                level = len(target) - len(target.lstrip("."))
                head = base.split(".")[: len(base.split(".")) - (level - 1)]
                target = ".".join([*head, *([target.lstrip(".")] if target.strip(".") else [])])
            how = "import_module" if called == "__import__" else called
            self.found.append(Reach(self.module, node.lineno, target, how))
        self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        if (
            self.climbs
            and isinstance(node, (ast.Attribute, ast.Subscript, ast.BinOp, ast.Call))
            and id(node) not in self._seen
        ):
            climb = _climb(node)
            if climb is not None:
                # The outermost expression is visited first and reaches furthest.
                self._seen.update(id(n) for n in ast.walk(node))
                levels, tail, _ = climb
                where = Path(os.path.normpath(self.path.parents[levels - 1].joinpath(*tail)))
                target = _module_of(self.src, where)
                if target is not None:
                    line = getattr(node, "lineno", 0)
                    self.found.append(Reach(self.module, line, target, "__file__"))
        super().generic_visit(node)


@functools.cache
def scan(src: Path, sides: Sides) -> tuple[tuple[str | None, Reach], ...]:
    """Every reach under ``src/osprey`` relevant to A1 or A2, with the importer's side.

    A description or panel-engine module is read whole; any other module only when its
    text spells a dynamic reach, and then for its dynamic reaches alone.
    """
    return tuple(_scan(src, sides))


def _module_files(src: Path) -> Iterator[tuple[Path, str]]:
    """Each source file under ``src/osprey`` with the module it lies in, in path order."""
    for path in sorted((src / "osprey").rglob("*.py")):
        if "__pycache__" not in path.parts and (module := _module_of(src, path)):
            yield path, module


def _scan(src: Path, sides: Sides) -> Iterator[tuple[str | None, Reach]]:
    for path, module in _module_files(src):
        side = sides.of(module)
        whole = side in (DESCRIPTION, PANELS)
        text = path.read_text(encoding="utf-8")
        if not whole and not any(spelling in text for spelling in _DYNAMIC_SPELLINGS):
            continue
        visitor = _Reaches(src, path, module, text, imports=whole)
        visitor.visit(ast.parse(text, filename=str(path)))
        for reach in visitor.found:
            yield side, reach


# --- the rules ---------------------------------------------------------------------------


def allowed_third_party(module: str) -> Mapping[str, str]:
    """The third-party import roots ``module``'s package allows besides the stdlib."""
    if module == MML_PREFIX or module.startswith(MML_PREFIX + "."):
        return {**DESCRIPTION_DEPENDENCIES, **MML_DEPENDENCIES}
    return DESCRIPTION_DEPENDENCIES


def third_party_edges(src: Path, sides: Sides) -> dict[str, list[Reach]]:
    """A1: imports of a description or panel-engine module outside its dependency set."""
    stdlib = sys.stdlib_module_names
    found: dict[str, list[Reach]] = {}
    for side, reach in scan(src, sides):
        if reach.how != "import" or side not in (DESCRIPTION, PANELS):
            continue
        root = reach.target.split(".")[0]
        if root in stdlib or root in sides.roots:
            continue
        allowed = PANELS_DEPENDENCIES if side == PANELS else allowed_third_party(reach.module)
        if root not in allowed:
            found.setdefault(reach.edge, []).append(reach)
    return found


def crossed_row(side: str | None, target: str, sides: Sides) -> str | None:
    """The contract row a reach from ``side`` to ``target`` breaks, or None."""
    to = sides.of(target)
    connectors = target.split(".")[0] == "osprey_connectors"
    if side == DESCRIPTION and to == FRAMEWORK:
        return "S1"
    if side == DESCRIPTION and to == PANELS:
        return "S2"
    if side == PANELS and to == DESCRIPTION:
        return "S3"
    if side == PANELS and (to == FRAMEWORK or connectors):
        return "S4"
    if side == FRAMEWORK and to == PANELS:
        return "S5"
    if side == FRAMEWORK and to == DESCRIPTION and _named(target, sides.non_public):
        return "S6"
    return None


def dynamic_edges(src: Path, sides: Sides) -> dict[str, list[Reach]]:
    """A2: dynamic reaches anywhere under ``src/osprey`` that cross a contract row."""
    found: dict[str, list[Reach]] = {}
    for side, reach in scan(src, sides):
        if reach.how != "import" and crossed_row(side, reach.target, sides):
            found.setdefault(reach.edge, []).append(reach)
    return found


def _report(title: str, edges: Mapping[str, list[Reach]], keys: Iterable[str]) -> str:
    lines = [f"{title}:"]
    for key in sorted(keys):
        lines.append(f"  {key}")
        lines.extend(f"    {reach}" for reach in sorted(edges.get(key, ())))
    return "\n".join(lines)


# --- the tests ---------------------------------------------------------------------------


def test_every_contract_side_names_some_module() -> None:
    sides = read_sides()
    assert sides.description and sides.panels and sides.framework and sides.non_public


def test_the_two_packages_import_only_their_dependency_sets() -> None:
    """A description or panel-engine import of another distribution is ruled (A1)."""
    edges = third_party_edges(SRC, read_sides())
    new = edges.keys() - THIRD_PARTY_RATCHET.keys()
    assert not new, _report(
        "Imports outside the package's dependency set (rule out, or name the "
        "dependency in DESCRIPTION_DEPENDENCIES / PANELS_DEPENDENCIES with its ruling)",
        edges,
        new,
    )


def test_every_third_party_ratchet_entry_is_still_an_edge() -> None:
    edges = third_party_edges(SRC, read_sides())
    stale = THIRD_PARTY_RATCHET.keys() - edges.keys()
    assert not stale, (
        f"THIRD_PARTY_RATCHET entries whose edge is gone, delete them: {sorted(stale)}"
    )


def test_no_dynamic_reach_crosses_a_contract_row() -> None:
    """``import_module``, ``resources.files`` and ``__file__`` climbs meet the contracts (A2)."""
    sides = read_sides()
    edges = dynamic_edges(SRC, sides)
    new = edges.keys() - DYNAMIC_RATCHET.keys()
    rows = {
        key: crossed_row(sides.of(edges[key][0].module), edges[key][0].target, sides) for key in new
    }
    assert not new, _report(
        "Dynamic reaches crossing a contract row (row per edge: "
        + ", ".join(f"{key} = {row}" for key, row in sorted(rows.items()))
        + ")",
        edges,
        new,
    )


def test_every_dynamic_ratchet_entry_is_still_an_edge() -> None:
    edges = dynamic_edges(SRC, read_sides())
    stale = DYNAMIC_RATCHET.keys() - edges.keys()
    assert not stale, f"DYNAMIC_RATCHET entries whose edge is gone, delete them: {sorted(stale)}"


def test_every_ratchet_entry_names_its_ruling_and_phase() -> None:
    for key, why in {**THIRD_PARTY_RATCHET, **DYNAMIC_RATCHET}.items():
        assert "ruling" in why and "phase" in why, f"{key!r}: {why!r}"


# --- completeness ------------------------------------------------------------------------

#: The import-linter contracts that encode the facility boundaries (rows S1-S6).
FACILITY_CONTRACTS = (
    "facility-description-standalone",
    "facility-description-without-panels",
    "panels-document-only",
    "panels-standalone",
    "framework-without-panels",
    "facility-public-api",
)

#: The epic phases a ratchet entry may name: phase 1 adds the entries, 2-6 remove them.
EPIC_PHASES = range(2, 7)

#: Modules no contract names: ``osprey._version`` is written by the build and missing on
#: a fresh clone, so import-linter would refuse it (the policy comment of the contracts).
UNLISTED = frozenset({"osprey._version"})

#: The contract lists that spell the framework, where a new framework module goes.
FRAMEWORK_LISTS = (
    "framework-without-panels.source_modules",
    "facility-description-standalone.forbidden_modules",
    "panels-standalone.forbidden_modules",
)

_PHASE = re.compile(r"#\s*phase\s+(\d+)\b")
_ENTRY = re.compile(r'^\s*"([^"]*)"\s*,?\s*(#.*)?$')


def modules(src: Path) -> list[str]:
    """Every module and package under ``src/osprey``, by dotted name."""
    return sorted({module for _, module in _module_files(src)})


def misclassified(src: Path, sides: Sides) -> dict[str, list[str]]:
    """Modules on no side or on several, with the sides that name them."""
    found: dict[str, list[str]] = {}
    for module in modules(src):
        if module in UNLISTED:
            continue
        named = sides.naming(module)
        if len(named) != 1:
            found[module] = named
    return found


def contract_blocks(text: str) -> dict[str, list[str]]:
    """The raw lines of each import-linter contract in ``pyproject.toml`` text, by id."""
    blocks: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("["):
            current = None
            if line.strip() == "[[tool.importlinter.contracts]]":
                current = []
        if current is None:
            continue
        current.append(line)
        if match := re.match(r'^id\s*=\s*"([^"]+)"', line):
            blocks[match.group(1)] = current
    return blocks


def ignore_phases(block: list[str]) -> list[tuple[str, int | None]]:
    """Each ``ignore_imports`` entry of a raw contract block with the phase it sits under.

    A ``# phase N`` comment sets the phase for the entries after it; a comment that
    names no phase continues the header above it.
    """
    entries: list[tuple[str, int | None]] = []
    inside = False
    phase: int | None = None
    for line in block:
        stripped = line.strip()
        if not inside:
            inside = stripped.startswith("ignore_imports") and stripped.endswith("[")
            continue
        if stripped == "]":
            break
        if stripped.startswith("#"):
            if match := _PHASE.match(stripped):
                phase = int(match.group(1))
            continue
        if match := _ENTRY.match(line):
            entries.append((match.group(1), phase))
        elif stripped:
            entries.append((stripped, None))
    return entries


def contract_comment(block: list[str]) -> str:
    """The comment lines of a raw contract block, joined into one text."""
    return " ".join(
        line.strip().lstrip("#").strip() for line in block if line.strip().startswith("#")
    )


def declared_public(comment: str) -> list[str]:
    """The public description modules a ``facility-public-api`` comment declares.

    The declaration runs from ``Public:`` to the sentence that hands over to the list;
    names are relative to ``osprey.facility`` unless spelled from ``osprey.``, and a
    possessive (``layers.mml``'s ``family``) makes its module the base of the names
    after it, up to the next ``;``.
    """
    match = re.search(r"Public:(.*?)The modules listed below", comment)
    assert match, "facility-public-api's comment declares no public set ('Public: ...')"
    public: list[str] = []
    for clause in match.group(1).split(";"):
        base = "osprey.facility"
        for name, possessive in re.findall(r"`([^`]+)`('s)?", clause):
            module = name if name.startswith("osprey.") else f"{base}.{name}"
            if possessive:
                base = module
            else:
                public.append(module)
    return public


def _pyproject_text() -> str:
    return PYPROJECT.read_text(encoding="utf-8")


def _declared_public() -> list[str]:
    """The public modules ``facility-public-api`` declares in ``pyproject.toml``."""
    return declared_public(
        contract_comment(contract_blocks(_pyproject_text())["facility-public-api"])
    )


def test_every_module_is_on_exactly_one_side() -> None:
    """A new module, a new top-level package included, fails until a contract lists it."""
    wrong = misclassified(SRC, read_sides())
    nowhere = sorted(module for module, named in wrong.items() if not named)
    several = {module: named for module, named in wrong.items() if named}
    assert not wrong, (
        "Modules the facility contracts place on no side or on several; each module is "
        "on exactly one. A framework module or package goes into each of "
        + ", ".join(FRAMEWORK_LISTS)
        + " (a package as 'X' plus 'X.**'); a description or panel-engine module "
        "into that side's lists.\n"
        f"  on no side: {nowhere}\n"
        f"  on several: {several}"
    )


def test_the_framework_list_is_the_same_in_every_contract_that_spells_it() -> None:
    contracts = _contracts()
    framework = contracts["framework-without-panels"]["source_modules"]
    for spelled in FRAMEWORK_LISTS[1:]:
        cid, key = spelled.split(".")
        missing = [module for module in framework if module not in contracts[cid][key]]
        assert not missing, f"{spelled} lacks framework modules {missing}"


def test_the_contracts_name_only_existing_modules() -> None:
    """A list entry that names no module is a typo or a module since moved."""
    existing = modules(SRC)
    contracts = _contracts()
    stale = {
        f"{cid}.{key}": [
            pattern
            for pattern in contracts[cid].get(key, ())
            if pattern.split(".")[0] == "osprey"
            and not any(_matches(module, pattern) for module in existing)
        ]
        for cid in FACILITY_CONTRACTS
        for key in ("source_modules", "forbidden_modules", "protected_modules")
    }
    stale = {where: patterns for where, patterns in stale.items() if patterns}
    assert not stale, f"Contract entries naming no module under src/osprey: {stale}"


def test_every_ignored_import_sits_under_an_epic_phase_header() -> None:
    blocks = contract_blocks(_pyproject_text())
    contracts = _contracts()
    for cid in FACILITY_CONTRACTS:
        assert cid in blocks, f"pyproject.toml has no import-linter contract {cid!r}"
        entries = ignore_phases(blocks[cid])
        assert [entry for entry, _ in entries] == contracts[cid].get("ignore_imports", []), (
            f"{cid}: the raw ignore_imports lines do not read as the parsed list; keep one "
            "quoted entry per line"
        )
        unphased = [entry for entry, phase in entries if phase not in EPIC_PHASES]
        assert not unphased, (
            f"{cid}: ignore_imports entries under no '# phase N' header naming an epic "
            f"phase ({EPIC_PHASES.start}-{EPIC_PHASES.stop - 1}): {unphased}"
        )


def test_no_ignored_import_uses_a_wildcard() -> None:
    """One importer -> imported pair per entry, so a removed edge leaves one stale entry."""
    contracts = _contracts()
    wild = {
        cid: [entry for entry in contracts[cid].get("ignore_imports", []) if "*" in entry]
        for cid in FACILITY_CONTRACTS
    }
    wild = {cid: entries for cid, entries in wild.items() if entries}
    assert not wild, f"ignore_imports entries with a wildcard: {wild}"


def test_the_declared_public_set_names_existing_description_modules() -> None:
    sides = read_sides()
    public = _declared_public()
    existing = set(modules(SRC))
    missing = [module for module in public if module not in existing]
    assert not missing, f"facility-public-api declares modules that do not exist: {missing}"
    elsewhere = [module for module in public if sides.of(module) != DESCRIPTION]
    assert not elsewhere, f"facility-public-api declares non-description modules: {elsewhere}"
    protected = [module for module in public if _named(module, sides.non_public)]
    assert not protected, f"facility-public-api declares protected modules public: {protected}"


def test_every_description_module_is_public_or_protected() -> None:
    sides = read_sides()
    public = set(_declared_public())
    description = [module for module in modules(SRC) if sides.of(module) == DESCRIPTION]
    undeclared = [
        module
        for module in description
        if module not in public and not _named(module, sides.non_public)
    ]
    assert not undeclared, (
        "Description modules neither declared public in facility-public-api's comment nor "
        f"listed in its protected_modules: {undeclared}"
    )
