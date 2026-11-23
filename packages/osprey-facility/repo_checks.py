"""The facility package's checks: its public names, its wheel, and its standalone lane.

``scripts/check.py`` collects this module after the framework's. The package
declares its public modules in the README's ``Public API`` table, each with the
consumer it serves; every other module is private. The rule that keeps that API
narrow is declared here, beside the package: a name a public module lists in
``__all__`` has a user outside that module, or is a declared extension point (a
name a kind plug-in, or a fork or tool reading facility.json without the
framework, uses from outside this tree). The CI lane that installs the package
alone, with no framework, is declared here too.

Standard library only, for the reason ``scripts/check.py`` gives.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from pathlib import Path

import check_kit as kit
from check_kit import BOTH, CLEAN, FAILED_EXIT, ROOT, Check, Lane

#: This module, as the lines about its own sets name it.
OWN_FILE = "packages/osprey-facility/repo_checks.py"

#: The package's import root.
PACKAGE = "osprey_facility"

#: The package's source root, relative to the repository root.
PACKAGE_SRC = "packages/osprey-facility/src"

#: The file whose ``Public API`` table declares the public modules.
README = "packages/osprey-facility/README.md"

#: The trees whose ``*.py`` files count as users of a listed name. Tests,
#: scripts and docs do not ship, so they are not users.
USER_ROOTS = ("src", "packages/*/src")

#: A row of the README's public-module table: ``| `osprey_facility.x` | consumer |``.
_ROW = re.compile(r"^\|\s*`(osprey_facility(?:\.\w+)*)`\s*\|\s*(.+?)\s*\|\s*$")

#: Names that are public with no user in the tree, each with the reason.
EXTENSION_POINTS: Mapping[str, str] = {
    "osprey_facility.kinds.CHROMATICITY_MONITOR_NAME": "a kind plug-in names a built-in kind",
    "osprey_facility.kinds.CRM_FILE": "a kind plug-in names a built-in kind's view file",
    "osprey_facility.kinds.E_DELTA_CAP": "a kind plug-in's measure plan reads the energy cap",
    "osprey_facility.kinds.E_DELTA_ORBIT": "a kind plug-in's measure plan reads the orbit step",
    "osprey_facility.kinds.FIT_STRATEGIES": "a kind plug-in's measure plan names its fit",
    "osprey_facility.kinds.KindDeclaration": "a kind plug-in declares its kind in this type",
    "osprey_facility.kinds.KindFactsError": "a kind plug-in's facts writer raises it",
    "osprey_facility.kinds.KindSettingsError": "a kind plug-in's settings hook raises it",
    "osprey_facility.kinds.KindStop": "a kind plug-in's build stop answers in this type",
    "osprey_facility.kinds.KindTool": "a kind plug-in names its pyAML tool in this type",
    "osprey_facility.kinds.MEMBER_SORTS": "a kind plug-in states its member sorts from it",
    "osprey_facility.kinds.REQUIREMENTS": "a kind plug-in states each member's requirement",
    "osprey_facility.kinds.RESULT_SHAPES": "a kind plug-in's measure plan names its result",
    "osprey_facility.kinds.SHIPPED_SORTS": "a kind plug-in reads the members pyAML ships",
    "osprey_facility.kinds.STEP_FACTORIES": "a kind plug-in's measure plan names its steps",
    "osprey_facility.kinds.TRM_FILE": "a kind plug-in names a built-in kind's view file",
    "osprey_facility.kinds.ViewSection": "a kind plug-in's view writer answers in this type",
    "osprey_facility.kinds.block_problems": "a kind plug-in checks its measurement block",
    "osprey_facility.kinds.member_sorts": "a kind plug-in reads a declaration's member sorts",
    "osprey_facility.kinds.plan_problems": "a kind plug-in checks its measure plan",
    "osprey_facility.schema.Facility": (
        "a fork or tool validates a facility file against the generated model"
    ),
    "osprey_facility.schema.core": "a fork or tool reads the generated model's classes",
    "osprey_facility.schema.published_schema": (
        "a fork or tool reads the published JSON Schema from the installed package"
    ),
}


#: The debt the move into the package left, which only shrinks: each entry is a
#: public module with no literal ``__all__`` (named by its module) or a name a
#: public module lists that nothing outside it uses (named dotted). The check
#: passes them and fails on a stale one; narrow a module's ``__all__``, or give
#: it one, and remove its entries.
UNNARROWED: frozenset[str] = frozenset(
    {
        "osprey_facility.build.build_facility",
        "osprey_facility.errors.KINDS",
        "osprey_facility.errors.WARNING_KINDS",
        "osprey_facility.kinds.BUILTIN_DISTRIBUTION",
        "osprey_facility.layers.list.COLUMNS",
        "osprey_facility.layers.list.LAYER_DIR",
        "osprey_facility.layers.mml.importer.ENGINE",
        "osprey_facility.layers.mml.importer.LAYER_DIR",
        "osprey_facility.layers.mml.importer.MODEL_SUFFIX",
        "osprey_facility.layers.mml.importer.TRANSPORT",
        "osprey_facility.layers.mml.importer.check_decks",
        "osprey_facility.layers.mml.importer.read_exports",
        "osprey_facility.layers.mml.importer.write_records",
        "osprey_facility.layers.mml.mapping.Branch",
        "osprey_facility.layers.mml.mapping.CALIBRATION_KINDS",
        "osprey_facility.layers.mml.mapping.DIRECTION_VALUES",
        "osprey_facility.layers.mml.mapping.Direction",
        "osprey_facility.layers.mml.mapping.Family",
        "osprey_facility.layers.mml.mapping.Field",
        "osprey_facility.layers.mml.mapping.FieldRole",
        "osprey_facility.layers.mml.mapping.Identity",
        "osprey_facility.layers.mml.mapping.PlaceRules",
        "osprey_facility.layers.mml.mapping.ROWS_BEYOND_KIND",
        "osprey_facility.layers.mml.mapping.SHARED_KIND",
        "osprey_facility.layers.mml.mapping.SectorList",
        "osprey_facility.layers.mml.mapping.SharedAnswer",
        "osprey_facility.layers.mml.mapping.UNBOUND_KIND",
        "osprey_facility.layers.mml.mapping.draft_mapping",
        "osprey_facility.layers.mml.mapping.draft_text",
        "osprey_facility.layers.mml.mapping.judgment_key",
        "osprey_facility.layers.mml.mapping.parse_mapping",
        "osprey_facility.layers.mml.mapping.require_decided",
        "osprey_facility.layers.mml.mapping.undecided_slots",
        "osprey_facility.layers.mml.response_check.Block",
        "osprey_facility.layers.mml.response_check.Entry",
        "osprey_facility.layers.mml.response_check.FLOOR_FRACTION",
        "osprey_facility.layers.mml.response_check.Figures",
        "osprey_facility.layers.mml.response_check.LeftOut",
        "osprey_facility.layers.mml.response_check.MEASURED_RATIO",
        "osprey_facility.layers.mml.response_check.MEASURED_SIGN",
        "osprey_facility.layers.mml.response_check.MODEL_INSIDE_BAND",
        "osprey_facility.layers.mml.response_check.MODEL_ORIGIN",
        "osprey_facility.layers.mml.response_check.ModelCheck",
        "osprey_facility.layers.mml.response_check.POLARITY_SHARE",
        "osprey_facility.layers.mml.response_check.TOLERANCE_FRACTION",
        "osprey_facility.layers.mml.response_check.Verdict",
        "osprey_facility.layers.mml.response_check.banded",
        "osprey_facility.layers.mml.response_check.compare",
        "osprey_facility.layers.mml.response_check.figures",
        "osprey_facility.layers.mml.response_check.judge",
        "osprey_facility.layers.mml.seed.CLASSES_FILE",
        "osprey_facility.layers.mml.seed.READOUT_FILE",
        "osprey_facility.layers.mml.seed.Seeded",
        "osprey_facility.layers.mml.seed.TUNING",
        "osprey_facility.layers.mml.seed.band",
        "osprey_facility.layers.relations.LAYER_FILES",
        "osprey_facility.layers.relations.RECORD_FILES",
        "osprey_facility.layers.relations.RESERVED_LAYERS",
        "osprey_facility.layers.relations.TABLE_FILES",
        "osprey_facility.pairing",
        "osprey_facility.reading.READS",
        "osprey_facility.records.FacilityRecord",
        "osprey_facility.records.RecordKind",
        "osprey_facility.relations",
        "osprey_facility.scenarios.ScenarioLogEntry",
        "osprey_facility.scenarios.check_scenario_sessions",
        "osprey_facility.sessions.ARTIFACT_FORMS",
        "osprey_facility.sessions.ResolvedArtifact",
        "osprey_facility.sessions.ScenarioSession",
        "osprey_facility.sessions.SessionPart",
        "osprey_facility.sources.CLAIM_BASES",
        "osprey_facility.sources.CLAIM_WORDS",
        "osprey_facility.sources.LAYER_OWN_FILES",
        "osprey_facility.sources.LoadResult",
        "osprey_facility.sources.SourceRecord",
        "osprey_facility.sources.slot_names",
        "osprey_facility.validate.STAGES",
        "osprey_facility.validate.StageReport",
        "osprey_facility.validate.check_records",
        "osprey_facility.validate.check_references",
        "osprey_facility.validate.check_schema",
        "osprey_facility.validate.ordered_slots",
        "osprey_facility.validate.schema_document",
        "osprey_facility.validate.sort_errors",
        "osprey_facility.validate.tolerance_bound",
    }
)


def public_modules(root: Path) -> dict[str, str]:
    """Each public module the README's table at *root* declares, with its consumer."""
    text = (root / README).read_text(encoding="utf-8")
    return {match[1]: match[2] for line in text.splitlines() if (match := _ROW.match(line.strip()))}


def _module_name(relative: Path) -> str:
    """The dotted module a file under the package source root defines."""
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _module_file(source_root: Path, module: str) -> Path | None:
    """The file that defines *module* under *source_root*, or ``None``."""
    base = source_root.joinpath(*module.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


def _listed(tree: ast.Module) -> list[tuple[str, int]] | None:
    """The names a module's literal top-level ``__all__`` lists, with their lines."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            if not isinstance(node.value, (ast.List, ast.Tuple)):
                return None
            return [
                (item.value, item.lineno)
                for item in node.value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            ]
    return None


def _uses(tree: ast.Module, package: str | None) -> set[tuple[str, str]]:
    """``(module, name)`` for every name of the facility package *tree* uses.

    *package* is the dotted package the file belongs to when it is under the
    package source root, which is what a relative import resolves against.
    """
    uses: set[tuple[str, str]] = set()
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                if package is None:
                    continue
                parts = package.split(".")
                parts = parts[: len(parts) - (node.level - 1)]
                module = ".".join([*parts, *([node.module] if node.module else [])])
            else:
                module = node.module or ""
            if module.split(".")[0] != PACKAGE:
                continue
            for alias in node.names:
                uses.add((module, alias.name))
                bound[alias.asname or alias.name] = f"{module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] != PACKAGE:
                    continue
                if alias.asname:
                    bound[alias.asname] = alias.name
                else:
                    bound[PACKAGE] = PACKAGE
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        chain: list[str] = []
        value: ast.expr = node
        while isinstance(value, ast.Attribute):
            chain.append(value.attr)
            value = value.value
        if not (isinstance(value, ast.Name) and value.id in bound):
            continue
        module = bound[value.id]
        for attr in reversed(chain):
            uses.add((module, attr))
            module = f"{module}.{attr}"
    return uses


def stray_exports(
    root: Path,
    extension_points: Mapping[str, str] | None = None,
    unnarrowed: frozenset[str] | None = None,
) -> list[str]:
    """One line per listed name in the tree at *root* that nothing outside its module uses.

    A declared public module with no file or no literal ``__all__`` is a line,
    and so is a stale entry of *extension_points* (no ``__all__`` lists it) or
    of *unnarrowed* (the debt it names is gone).
    """
    declared = EXTENSION_POINTS if extension_points is None else extension_points
    debt = UNNARROWED if unnarrowed is None else unnarrowed
    owed: set[str] = set()
    source_root = root / PACKAGE_SRC
    lines: list[str] = []
    exports: dict[str, tuple[Path, list[tuple[str, int]]]] = {}
    for module in sorted(public_modules(root)):
        path = _module_file(source_root, module)
        if path is None:
            lines.append(f"{README}: {module} is a public module and no file defines it")
            continue
        names = _listed(ast.parse(path.read_text(encoding="utf-8")))
        if names is not None:
            exports[module] = (path, names)
            continue
        owed.add(module)
        if module not in debt:
            lines.append(
                f"{path.relative_to(root).as_posix()}: {module} is a public module and has "
                "no literal __all__; list its public names"
            )
    users: dict[tuple[str, str], set[Path]] = {}
    for pattern in USER_ROOTS:
        for tree_root in sorted(root.glob(pattern)):
            for path in sorted(tree_root.rglob("*.py")):
                source = path.read_text(encoding="utf-8")
                inside = path.is_relative_to(source_root)
                if not inside and PACKAGE not in source:
                    continue
                package = None
                if inside:
                    module = _module_name(path.relative_to(source_root))
                    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
                for use in _uses(ast.parse(source), package):
                    users.setdefault(use, set()).add(path)
    listed: set[str] = set()
    for module, (path, names) in sorted(exports.items()):
        where = path.relative_to(root).as_posix()
        for name, line in names:
            dotted = f"{module}.{name}"
            listed.add(dotted)
            if dotted in declared:
                continue
            if not users.get((module, name), set()) - {path}:
                owed.add(dotted)
                if dotted in debt:
                    continue
                lines.append(
                    f"{where}:{line}: {dotted} is listed in __all__ and nothing outside its "
                    "module uses it; make it private or declare it in EXTENSION_POINTS"
                )
    lines.extend(
        f"{OWN_FILE}: {dotted} is in EXTENSION_POINTS and no __all__ lists it; "
        "remove it from EXTENSION_POINTS"
        for dotted in sorted(set(declared) - listed)
    )
    lines.extend(
        f"{OWN_FILE}: {dotted} is in UNNARROWED and is no longer owed; remove it from UNNARROWED"
        for dotted in sorted(debt - owed)
    )
    return lines


def _check_public_names(_context: kit.Context) -> int:
    lines = stray_exports(ROOT)
    for line in lines:
        kit.say(f"facility-public-names: {line}")
    return FAILED_EXIT if lines else CLEAN


CHECKS: tuple[Check, ...] = (
    # Implemented in Python: an AST scan of the trees whose code ships.
    Check(
        "facility-public-names",
        "static",
        BOTH,
        (),
        scope=("packages/*/src/**.py", "src/**.py", README, OWN_FILE),
        builtin=_check_public_names,
    ),
    # The wheel-contents tests read a built wheel, which the sweep does not
    # build, so the sweep deselects their marker; the full tier runs them alone,
    # each building the wheel when OSPREY_FACILITY_WHEEL names none, as the
    # standalone CI job does with the wheel it builds once.
    Check(
        "facility-wheel-build",
        "tests",
        kit.FULL,
        (*kit.UV_RUN, "pytest", "packages/osprey-facility/tests/", "-m", "wheel_build"),
        scope=("packages/osprey-facility/**", "pyproject.toml", "uv.lock"),
        needs=("free-space", "uv", "install-current"),
        bound_s=600,
        report="junit-pytest",
        tier_flags={"full": ("--tb=short",)},
        exec_flags=True,
    ),
)

LANES: tuple[Lane, ...] = (
    Lane(
        "osprey-facility-standalone",
        "installs the facility package alone",
        ("packages/osprey-facility/**", "pyproject.toml", "uv.lock"),
    ),
)
