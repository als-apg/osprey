"""The connectors package's checks: its public names, and the lanes that install it alone.

``scripts/check.py`` collects this module after the framework's. The package
releases on its own tags and is installed without the framework, so the lanes
that prove that are declared here, beside the package, not in the framework's
module. So is the rule that keeps its API narrow: a name a public module lists
in ``__all__`` has a user outside that module, or is a declared extension point.

Standard library only, for the reason ``scripts/check.py`` gives.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

import check_kit as kit
from check_kit import BOTH, CLEAN, FAILED_EXIT, ROOT, Check, Lane

#: This module, as the lines about its own sets name it.
OWN_FILE = "packages/osprey-connectors/repo_checks.py"

#: The package's source root, relative to the repository root.
PACKAGE_SRC = "packages/osprey-connectors/src"

#: The trees whose ``*.py`` files count as users of a listed name. Tests,
#: scripts and docs do not ship, so they are not users.
USER_ROOTS = ("src", "packages/*/src")

#: Names that are public with no user in the tree, each with the reason.
EXTENSION_POINTS: Mapping[str, str] = {
    "osprey_connectors.alarm_log.AlarmLog": "an out-of-tree alarm-log backend subclasses it",
    "osprey_connectors.alarm_state.AlarmState": (
        "an out-of-tree alarm-state backend subclasses it"
    ),
    "osprey_connectors.archiver.ArchiveExtent": ("an archiver's store_extent answers in this type"),
    "osprey_connectors.archiver.ArchiverMetadata": (
        "an archiver's get_metadata answers in this type"
    ),
    "osprey_connectors.connection.ConnectionSettings": "a connector author declares its settings",
    "osprey_connectors.connection.Credential": "a connector author declares its credentials",
    "osprey_connectors.connection.Login": "a connector author declares how it logs in",
    "osprey_connectors.connection.LoginKind": "a connector author names its login kind",
    "osprey_connectors.directory.ChannelDirectory": (
        "an out-of-tree directory backend subclasses it"
    ),
    "osprey_connectors.factory.BuiltinConnector": (
        "the framework reads each built-in's fields in this type"
    ),
    "osprey_connectors.honesty.ArchiverPairing": (
        "every archiver-pairing judgement answers in this type"
    ),
    "osprey_connectors.ipc.ConnectorHostError": "a caller catches every host failure by its base",
    "osprey_connectors.ipc.ConnectorHostLostError": "a caller tells a lost host from the others",
    "osprey_connectors.ipc.ConnectorHostStartError": (
        "a caller tells a failed start from the others"
    ),
    "osprey_connectors.ipc.ConnectorHostUnresponsiveError": (
        "a caller tells an unresponsive host from the others"
    ),
    "osprey_connectors.ipc.ConnectorHostWriteTimeoutError": (
        "a caller tells a timed-out write from the others"
    ),
    "osprey_connectors.ipc.FrameDecodeError": "a caller tells a bad frame read from the others",
    "osprey_connectors.ipc.FrameEncodeError": "a caller tells a bad frame write from the others",
    "osprey_connectors.ipc.GATEWAYS_ROWS": (
        "a caller names the verdict that a target declares gateway rows"
    ),
    "osprey_connectors.ipc.LiveCollision": "a live-collision check answers in this type",
    "osprey_connectors.ipc.PooledConnector": (
        "the pool hands every caller its connector in this type"
    ),
    "osprey_connectors.ipc.READONLY": "a caller asks the pool for a readonly child by this value",
    "osprey_connectors.put_log.PutLog": "an out-of-tree put-log backend subclasses it",
    "osprey_connectors.recovery.Recovery": "a recovery answers what it did in this type",
    "osprey_connectors.recovery.Supervised": "a supervisor's target meets this protocol",
    "osprey_connectors.simulation.archive.FingerprintComparison": (
        "a fingerprint comparison answers in this type"
    ),
    "osprey_connectors.simulation.archive.SeedReport": (
        "a seeding run reports, and reports progress, in this type"
    ),
    "osprey_connectors.simulation.composite.WriteRefusedError": (
        "a caller reads which channel a refused write names"
    ),
    "osprey_connectors.simulation.logbook.PlotSeries": "a plot spec lists its series in this type",
    "osprey_connectors.simulation.series.BURST_WINDOW_TAUS": (
        "the framework's archive rewrite reads the burst window"
    ),
    "osprey_connectors.simulation.series.loss_refill_bound": (
        "the framework's archive rewrite reads the loss refill bound"
    ),
    "osprey_connectors.simulation.state.Overlap": "a composition check answers in this type",
    "osprey_connectors.simulation.view.Binding": "a view lists its wiring records in this type",
    "osprey_connectors.simulation.view.COMMAND_MARK": "a word of the wiring vocabulary",
    "osprey_connectors.simulation.view.Channel": "a view answers a channel in this type",
    "osprey_connectors.simulation.view.ENGINE_ROLES": "a word of the wiring vocabulary",
    "osprey_connectors.simulation.view.MARKS": "a word of the wiring vocabulary",
    "osprey_connectors.simulation.view.OUTPUT": "a word of the wiring vocabulary",
    "osprey_connectors.simulation.view.PERIODIC": "a word of the wiring vocabulary",
    "osprey_connectors.simulation.view.PLANES": "a word of the wiring vocabulary",
    "osprey_connectors.simulation.view.READBACK": "a word of the wiring vocabulary",
    "osprey_connectors.simulation.view.REFRESH": "a word of the wiring vocabulary",
    "osprey_connectors.types.LIMITS_MODE_EXCLUSIVE": (
        "a word of the limits-mode vocabulary a config states"
    ),
    "osprey_connectors.types.SERVED": "a word of the serving vocabulary a config states",
    "osprey_connectors.write_authorization.Approver": "a grant says who approved it in this type",
    "osprey_connectors.write_authorization.ChannelRange": (
        "a grant and a request bound each channel in this type"
    ),
    "osprey_connectors.write_authorization.GrantStanding": (
        "every grant listing answers in this type"
    ),
    "osprey_connectors.write_authorization.Ledger": (
        "a lease and a grant listing return the holder's ledger in this type"
    ),
    "osprey_connectors.write_authorization.LedgerStep": (
        "a ledger lists its spent steps in this type"
    ),
    "osprey_connectors.write_authorization.Revocation": (
        "a revoked grant says whose stop ended it in this type"
    ),
    "osprey_connectors.write_target.UNRESOLVED": (
        "a caller tells an unplaced write by this sentinel"
    ),
    "osprey_connectors.write_target.Unresolved": "an unplaced write answers in this type",
}


def _module_name(relative: Path) -> str:
    """The dotted module a file under the package source root defines."""
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


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


def _exports(source_root: Path) -> dict[str, tuple[Path, list[tuple[str, int]]]]:
    """Every public module of the package under *source_root* with a literal ``__all__``.

    A module is private when any segment of its path starts with an
    underscore; ``__init__.py`` stands for its package.
    """
    found: dict[str, tuple[Path, list[tuple[str, int]]]] = {}
    for path in sorted((source_root / "osprey_connectors").rglob("*.py")):
        relative = path.relative_to(source_root)
        segments = [*relative.parts[:-1], relative.stem]
        if relative.name == "__init__.py":
            segments.pop()
        if any(segment.startswith("_") for segment in segments):
            continue
        listed = _listed(ast.parse(path.read_text(encoding="utf-8")))
        if listed is not None:
            found[_module_name(relative)] = (path, listed)
    return found


def _uses(tree: ast.Module, package: str | None) -> set[tuple[str, str]]:
    """``(module, name)`` for every name of the connectors package *tree* uses.

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
            if module.split(".")[0] != "osprey_connectors":
                continue
            for alias in node.names:
                uses.add((module, alias.name))
                bound[alias.asname or alias.name] = f"{module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] != "osprey_connectors":
                    continue
                if alias.asname:
                    bound[alias.asname] = alias.name
                else:
                    bound["osprey_connectors"] = "osprey_connectors"
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


def stray_exports(root: Path, extension_points: Mapping[str, str] | None = None) -> list[str]:
    """One line per listed name in the tree at *root* that nothing outside its module uses.

    A stale entry of *extension_points* (no ``__all__`` lists it) is a line too.
    """
    declared = EXTENSION_POINTS if extension_points is None else extension_points
    source_root = root / PACKAGE_SRC
    exports = _exports(source_root)
    users: dict[tuple[str, str], set[Path]] = {}
    for pattern in USER_ROOTS:
        for tree_root in sorted(root.glob(pattern)):
            for path in sorted(tree_root.rglob("*.py")):
                source = path.read_text(encoding="utf-8")
                inside = path.is_relative_to(source_root)
                if not inside and "osprey_connectors" not in source:
                    continue
                package = None
                if inside:
                    module = _module_name(path.relative_to(source_root))
                    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
                for use in _uses(ast.parse(source), package):
                    users.setdefault(use, set()).add(path)
    lines: list[str] = []
    listed: set[str] = set()
    for module, (path, names) in sorted(exports.items()):
        where = path.relative_to(root).as_posix()
        for name, line in names:
            dotted = f"{module}.{name}"
            listed.add(dotted)
            if dotted in declared:
                continue
            if not users.get((module, name), set()) - {path}:
                lines.append(
                    f"{where}:{line}: {dotted} is listed in __all__ and nothing outside its "
                    "module uses it; make it private or declare it in EXTENSION_POINTS"
                )
    lines.extend(
        f"{OWN_FILE}: {dotted} is in EXTENSION_POINTS and no __all__ lists it; "
        "remove it from EXTENSION_POINTS"
        for dotted in sorted(set(declared) - listed)
    )
    return lines


def _check_public_names(_context: kit.Context) -> int:
    lines = stray_exports(ROOT)
    for line in lines:
        kit.say(f"connectors-public-names: {line}")
    return FAILED_EXIT if lines else CLEAN


CHECKS: tuple[Check, ...] = (
    # Implemented in Python: an AST scan of the trees whose code ships.
    Check(
        "connectors-public-names",
        "static",
        BOTH,
        (),
        scope=("packages/*/src/**.py", "src/**.py", OWN_FILE),
        builtin=_check_public_names,
    ),
)

LANES: tuple[Lane, ...] = (
    Lane(
        "connectors-standalone",
        "installs the connectors alone",
        ("packages/osprey-connectors/**", "pyproject.toml", "uv.lock"),
    ),
    Lane(
        "connectors-compat-plan",
        "reads the release tags",
        ("scripts/ci/connectors_compat_matrix.py", "pyproject.toml"),
    ),
    Lane(
        "connectors-compat",
        "installs released frameworks",
        (
            "packages/osprey-connectors/**",
            "scripts/ci/connectors_compat_lane.py",
            "scripts/ci/connectors_compat_matrix.py",
        ),
    ),
)
