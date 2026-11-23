"""The panels package's checks: its public names, its wheel, and its standalone lane.

``scripts/check.py`` collects this module after the framework's. The package
declares its public modules in the README's ``Public API`` table, each with the
consumer it serves; every other module is private. The rule that keeps that API
narrow is declared here, beside the package: a name a public module lists in
``__all__`` has a user outside that module, or is a declared extension point (a
name a fork or tool driving the engine without the framework, or the command's
entry point, uses from outside this tree). The CI lane that installs the
package alone, with no framework and no facility package, is declared here too.

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
OWN_FILE = "packages/osprey-panels/repo_checks.py"

#: The package's import root.
PACKAGE = "osprey_panels"

#: The package's source root, relative to the repository root.
PACKAGE_SRC = "packages/osprey-panels/src"

#: The file whose ``Public API`` table declares the public modules.
README = "packages/osprey-panels/README.md"

#: The trees whose ``*.py`` files count as users of a listed name. Tests,
#: scripts and docs do not ship, so they are not users.
USER_ROOTS = ("src", "packages/*/src")

#: A row of the README's public-module table: ``| `osprey_panels.x` | consumer |``.
_ROW = re.compile(r"^\|\s*`(osprey_panels(?:\.\w+)*)`\s*\|\s*(.+?)\s*\|\s*$")

#: Names that are public with no user in the tree, each with the reason.
EXTENSION_POINTS: Mapping[str, str] = {
    "osprey_panels.Channel": "a fork or tool reads the panel graph",
    "osprey_panels.Citation": "a fork or tool reads the panel graph",
    "osprey_panels.Edge": "a fork or tool reads the panel graph",
    "osprey_panels.FacilityDocument": (
        "a fork or tool reads the facility file through the engine's reader"
    ),
    "osprey_panels.FacilityDocumentError": (
        "a fork or tool reads the facility file through the engine's reader"
    ),
    "osprey_panels.HealthRule": "a fork or tool reads the panel rule tables",
    "osprey_panels.Pair": "a fork or tool reads the panel graph",
    "osprey_panels.PanelGraph": "a fork or tool reads the panel graph",
    "osprey_panels.PanelTableError": "a fork or tool reads the panel rule tables",
    "osprey_panels.PanelTables": "a fork or tool reads the panel rule tables",
    "osprey_panels.Pose": "a fork or tool reads the panel rule tables",
    "osprey_panels.State": "a fork or tool turns channel values into states",
    "osprey_panels.Target": "a fork or tool reads what a view contains",
    "osprey_panels.Thing": "a fork or tool reads the panel graph",
    "osprey_panels.View": "a fork or tool reads what a view contains",
    "osprey_panels.channel_list": "a fork or tool says what a screen shows, without layout",
    "osprey_panels.channel_state": "a fork or tool turns channel values into states",
    "osprey_panels.checkout": "a fork or tool says what a screen shows, without layout",
    "osprey_panels.describe": "a fork or tool says what a screen shows, without layout",
    "osprey_panels.device_card": "a fork or tool says what a screen shows, without layout",
    "osprey_panels.document.READS": (
        "the contract test holds the facility header to the headers the engine reads"
    ),
    "osprey_panels.health_rollup": "a fork or tool says what a screen shows, without layout",
    "osprey_panels.layout.aggregates": (
        "the Canopy contract test counts a screen's server-side aggregates"
    ),
    "osprey_panels.layout.engine_length": (
        "the Canopy contract test measures a ladder's engine length"
    ),
    "osprey_panels.layout.needs": (
        "the Canopy contract test reads what a screen needs of the runtime"
    ),
    "osprey_panels.load_graph": "a fork or tool reads the panel graph",
    "osprey_panels.place_view": "a fork or tool says what a screen shows, without layout",
    "osprey_panels.rack_elevation": "a fork or tool says what a screen shows, without layout",
    "osprey_panels.read_tables": "a fork or tool reads the panel rule tables",
    "osprey_panels.rollup": "a fork or tool turns channel values into states",
    "osprey_panels.row_matches": "a fork or tool turns channel values into states",
    "osprey_panels.rule_channels": "a fork or tool turns channel values into states",
    "osprey_panels.screen_ir.schema": (
        "a fork or tool validates a screen against the vendored Screen-IR schema"
    ),
    "osprey_panels.set_overview": "a fork or tool says what a screen shows, without layout",
    "osprey_panels.supported": "a fork or tool turns channel values into states",
    "osprey_panels.targets": "a fork or tool says what a screen shows, without layout",
    "osprey_panels.unsupported": "a fork or tool turns channel values into states",
    "osprey_panels.worst": "a fork or tool turns channel values into states",
}

#: The debt the move into the package left, which only shrinks: each entry is a
#: public module with no literal ``__all__`` (named by its module) or a name a
#: public module lists that nothing outside it uses (named dotted). The check
#: passes them and fails on a stale one; narrow a module's ``__all__``, or give
#: it one, and remove its entries.
UNNARROWED: frozenset[str] = frozenset(
    {
        "osprey_panels.describe.CARD_SECTIONS",
        "osprey_panels.describe.GENERIC_SYMBOL",
        "osprey_panels.describe.HOP_CONTESTED",
        "osprey_panels.describe.HOP_NOT_STATED",
        "osprey_panels.describe.NO_SYSTEM",
        "osprey_panels.describe.PLACE_RELATION",
        "osprey_panels.describe.RACK_ORDER",
        "osprey_panels.layout.FORMAT_REQUESTS",
        "osprey_panels.layout.Ladder",
        "osprey_panels.layout.MIN_STRIP_SCALE",
        "osprey_panels.layout.WIDGETS",
        "osprey_panels.layout.compiled_rules",
        "osprey_panels.layout.light_rule",
        "osprey_panels.layout.metric_formula",
        "osprey_panels.layout.picture_labels",
        "osprey_panels.layout.stem_levels",
        "osprey_panels.layout.strip_labels",
        "osprey_panels.layout.text_width",
        "osprey_panels.layout.unexpressed",
        "osprey_panels.symbols.AGGREGATE_MARK",
        "osprey_panels.symbols.ALARM_ROLES",
        "osprey_panels.symbols.APPEARANCES",
        "osprey_panels.symbols.BADGE_LETTERS",
        "osprey_panels.symbols.DISAGREE_MARK",
        "osprey_panels.symbols.STATE_DISTANCE",
        "osprey_panels.symbols.UNKNOWN_MARK",
        "osprey_panels.symbols.colour_distance",
        "osprey_panels.symbols.role",
        "osprey_panels.symbols.styled",
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
    """``(module, name)`` for every name of the panels package *tree* uses.

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
        kit.say(f"panels-public-names: {line}")
    return FAILED_EXIT if lines else CLEAN


CHECKS: tuple[Check, ...] = (
    # Implemented in Python: an AST scan of the trees whose code ships.
    Check(
        "panels-public-names",
        "static",
        BOTH,
        (),
        scope=("packages/*/src/**.py", "src/**.py", README, OWN_FILE),
        builtin=_check_public_names,
    ),
    # The wheel-contents tests read a built wheel, which the sweep does not
    # build, so the sweep deselects their marker; the full tier runs them alone,
    # each building the wheel when OSPREY_PANELS_WHEEL names none, as the
    # standalone CI job does with the wheel it builds once.
    Check(
        "panels-wheel-build",
        "tests",
        kit.FULL,
        (*kit.UV_RUN, "pytest", "packages/osprey-panels/tests/", "-m", "wheel_build"),
        scope=("packages/osprey-panels/**", "pyproject.toml", "uv.lock"),
        needs=("free-space", "uv", "install-current"),
        bound_s=600,
        report="junit-pytest",
        tier_flags={"full": ("--tb=short",)},
        exec_flags=True,
    ),
)

LANES: tuple[Lane, ...] = (
    Lane(
        "osprey-panels-standalone",
        "installs the panels package alone",
        ("packages/osprey-panels/**", "pyproject.toml", "uv.lock"),
    ),
)
