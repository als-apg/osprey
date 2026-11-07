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
from collections.abc import Mapping, Set
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
    "osprey_connectors.ipc.ConnectorHostError": "a caller catches every host failure by its base",
    "osprey_connectors.ipc.ConnectorHostLostError": "a caller tells a lost host from the others",
    "osprey_connectors.ipc.ConnectorHostStartError": "a caller tells a failed start from the others",
    "osprey_connectors.ipc.ConnectorHostUnresponsiveError": (
        "a caller tells an unresponsive host from the others"
    ),
    "osprey_connectors.ipc.ConnectorHostWriteTimeoutError": (
        "a caller tells a timed-out write from the others"
    ),
    "osprey_connectors.ipc.FrameDecodeError": "a caller tells a bad frame read from the others",
    "osprey_connectors.ipc.FrameEncodeError": "a caller tells a bad frame write from the others",
    "osprey_connectors.connection.ConnectionSettings": "a connector author declares its settings",
    "osprey_connectors.connection.Credential": "a connector author declares its credentials",
    "osprey_connectors.connection.Login": "a connector author declares how it logs in",
    "osprey_connectors.connection.LoginKind": "a connector author names its login kind",
    "osprey_connectors.pv_directory.contract.AlarmLog": (
        "an out-of-tree alarm-log backend subclasses it"
    ),
    "osprey_connectors.pv_directory.contract.AlarmState": (
        "an out-of-tree alarm-state backend subclasses it"
    ),
    "osprey_connectors.pv_directory.contract.PVDirectory": (
        "an out-of-tree directory backend subclasses it"
    ),
    "osprey_connectors.pv_directory.contract.PutLog": (
        "an out-of-tree put-log backend subclasses it"
    ),
    "osprey_connectors.simulation.view.OUTPUT": "a word of the wiring vocabulary",
    "osprey_connectors.simulation.view.PERIODIC": "a word of the wiring vocabulary",
}

#: Interim: the listed names that had no user when this check was added. Each
#: leaves this set when it is made private or declared in EXTENSION_POINTS; the
#: set only shrinks, and an entry that gains a user or leaves ``__all__`` fails.
UNUSED_BASELINE: frozenset[str] = frozenset(
    {
        "osprey_connectors.archive_targets.ADDRESS_SETTINGS",
        "osprey_connectors.archive_targets.ARCHIVER_SECTION_KEY",
        "osprey_connectors.archive_targets.DEFAULT_ADDRESS_SETTING",
        "osprey_connectors.archive_targets.NO_TARGETS",
        "osprey_connectors.archive_targets.STORE_ARCHIVERS",
        "osprey_connectors.archive_targets.STORE_TARGETS",
        "osprey_connectors.archiver.ArchiverMetadata",
        "osprey_connectors.archiver.Processing",
        "osprey_connectors.archiver.aggregate_series",
        "osprey_connectors.archiver.channel_address",
        "osprey_connectors.archiver.decimate_raw",
        "osprey_connectors.archiver.long_frame",
        "osprey_connectors.archiver.reject_non_numeric",
        "osprey_connectors.archiver.require_datetime",
        "osprey_connectors.archiver.resolve_processing",
        "osprey_connectors.archiver.to_utc",
        "osprey_connectors.control_context.REACHABLE_STATES",
        "osprey_connectors.control_context.REPORT_APPLIED",
        "osprey_connectors.control_context.REPORT_APPLYING",
        "osprey_connectors.control_context.REPORT_FAILED",
        "osprey_connectors.control_context.SCHEMA_VERSION",
        "osprey_connectors.control_context.WRITES_STATEMENT_KEYS",
        "osprey_connectors.control_context.is_foreign_process",
        "osprey_connectors.control_context.live_report_payloads",
        "osprey_connectors.control_context.live_statement_reports",
        "osprey_connectors.control_context.parse_posture",
        "osprey_connectors.control_context.parse_report",
        "osprey_connectors.control_context.posture_mirror",
        "osprey_connectors.control_context.report_path_under",
        "osprey_connectors.control_context.state_dir",
        "osprey_connectors.factory.BuiltinConnector",
        "osprey_connectors.honesty.ArchiverPairing",
        "osprey_connectors.ipc.DEFAULT_PVA_PORT",
        "osprey_connectors.ipc.LiveCollision",
        "osprey_connectors.ipc.MODE_ADDR_LIST",
        "osprey_connectors.ipc.PooledConnector",
        "osprey_connectors.ipc.READONLY",
        "osprey_connectors.ipc.ROLE_PVA",
        "osprey_connectors.ipc.SETTLE_TIMEOUT_S",
        "osprey_connectors.ipc.verify_child_report",
        "osprey_connectors.posture_store.AGENT_DATA_ROOT_ENV_VAR",
        "osprey_connectors.posture_store.CONTROL_TREE_MARKER_NAME",
        "osprey_connectors.posture_store.LAUNCH_POSTURE_ALL_TARGETS",
        "osprey_connectors.posture_store.LOGIN_TIERS_KEY",
        "osprey_connectors.posture_store.VALID_POSTURES",
        "osprey_connectors.posture_store.agent_data_root",
        "osprey_connectors.posture_store.invalidate_cache",
        "osprey_connectors.recovery.Origin",
        "osprey_connectors.recovery.Recovery",
        "osprey_connectors.recovery.Supervised",
        "osprey_connectors.session_hold.EXCERPT_LIMIT",
        "osprey_connectors.session_hold.SCHEMA_VERSION",
        "osprey_connectors.settle.POLL_INTERVAL_S",
        "osprey_connectors.simulation.ACTIVE_SCENARIOS_FILENAME",
        "osprey_connectors.simulation.DEFAULT_SCENARIO",
        "osprey_connectors.simulation.Overlap",
        "osprey_connectors.simulation.STATUS_NO_DECK",
        "osprey_connectors.simulation.archive.FingerprintComparison",
        "osprey_connectors.simulation.archive.SeedReport",
        "osprey_connectors.simulation.channel_faults.DISCONNECTED",
        "osprey_connectors.simulation.channel_faults.FROZEN",
        "osprey_connectors.simulation.coerce",
        "osprey_connectors.simulation.composed_set",
        "osprey_connectors.simulation.composite.ENGINE_GROUP",
        "osprey_connectors.simulation.composite.INSTANCES",
        "osprey_connectors.simulation.composite.STATUS_NO_DECK",
        "osprey_connectors.simulation.composite.UDF",
        "osprey_connectors.simulation.composite.WriteRefusedError",
        "osprey_connectors.simulation.logbook.PlotSeries",
        "osprey_connectors.simulation.overlap_record",
        "osprey_connectors.simulation.parse_active_state",
        "osprey_connectors.simulation.pulse.AMPLITUDE_DRIFT_FRACTION",
        "osprey_connectors.simulation.pulse.DRIFT_PERIOD_S",
        "osprey_connectors.simulation.pulse.EDGE_JITTER_SAMPLES",
        "osprey_connectors.simulation.pulse.FALL_TAU_SAMPLES",
        "osprey_connectors.simulation.pulse.FLAT_TOP_DROOP",
        "osprey_connectors.simulation.pulse.FLAT_TOP_SAMPLES",
        "osprey_connectors.simulation.pulse.MAX_TIMING_DRIFT_SAMPLES",
        "osprey_connectors.simulation.pulse.NOISE_FRACTION",
        "osprey_connectors.simulation.pulse.RINGING_AMPLITUDE_FRACTION",
        "osprey_connectors.simulation.pulse.RINGING_COMPONENTS",
        "osprey_connectors.simulation.pulse.RINGING_DAMPING_SAMPLES",
        "osprey_connectors.simulation.pulse.RINGING_LONGEST_PERIOD_SAMPLES",
        "osprey_connectors.simulation.pulse.RINGING_PERIOD_RATIO",
        "osprey_connectors.simulation.pulse.RISE_SAMPLES",
        "osprey_connectors.simulation.pulse.RISE_START_SAMPLE",
        "osprey_connectors.simulation.pulse.SHAPE_CONSTANT_NAMES",
        "osprey_connectors.simulation.pulse.TRACE_POINTS",
        "osprey_connectors.simulation.pulse.shape_version",
        "osprey_connectors.simulation.read_active_state",
        "osprey_connectors.simulation.resolve_active_scenarios",
        "osprey_connectors.simulation.scenario_targets",
        "osprey_connectors.simulation.topoff.DECAY_FORM",
        "osprey_connectors.simulation.topoff.LIFETIME_KEY",
        "osprey_connectors.simulation.topoff.REFILL_PHASE",
        "osprey_connectors.simulation.topoff.SECONDS_PER_HOUR",
        "osprey_connectors.simulation.topoff.SHAPE_CONSTANT_NAMES",
        "osprey_connectors.simulation.topoff.TOPOFF_KEY",
        "osprey_connectors.simulation.topoff.shape",
        "osprey_connectors.simulation.topoff.shape_version",
        "osprey_connectors.simulation.validate_composition",
        "osprey_connectors.pv_directory.targets.has_simulator",
        "osprey_connectors.pv_directory.targets.pv_directory_targets",
        "osprey_connectors.simulation.values.VALUE_TYPES",
        "osprey_connectors.simulation.values.WAVEFORM",
        "osprey_connectors.simulation.view.Binding",
        "osprey_connectors.simulation.view.NO_VIEW_MESSAGE",
        "osprey_connectors.simulation.view.PLANES",
        "osprey_connectors.simulation.view.REFRESH",
        "osprey_connectors.simulation.view.ROLES",
        "osprey_connectors.simulation.write_active_state",
        "osprey_connectors.standin.ARCHIVER_RECORDER_SERVICE",
        "osprey_connectors.standin.DEPLOYED_SERVICES_KEY",
        "osprey_connectors.standin.archive_belongs_to_standin",
        "osprey_connectors.types.CLI_ARCHIVER_TYPES",
        "osprey_connectors.types.LIMITS_MODE_EXCLUSIVE",
        "osprey_connectors.types.SERVED",
        "osprey_connectors.types.TRANSPORT_DOOCS",
        "osprey_connectors.types.TRANSPORT_TANGO",
        "osprey_connectors.write_authorization.APPROVAL_SURFACES",
        "osprey_connectors.write_authorization.Approver",
        "osprey_connectors.write_authorization.COMPLETIONS",
        "osprey_connectors.write_authorization.ChannelRange",
        "osprey_connectors.write_authorization.GrantStanding",
        "osprey_connectors.write_authorization.LEDGER_DIR_ENV_VAR",
        "osprey_connectors.write_authorization.Ledger",
        "osprey_connectors.write_authorization.LedgerStep",
        "osprey_connectors.write_authorization.PRUNE_AFTER_S",
        "osprey_connectors.write_authorization.Revocation",
        "osprey_connectors.write_authorization.SCHEMA_VERSION",
        "osprey_connectors.write_authorization.SURFACES",
        "osprey_connectors.write_authorization.auth_overlap",
        "osprey_connectors.write_authorization.authorization_verdict",
        "osprey_connectors.write_authorization.grant_in_force",
        "osprey_connectors.write_authorization.issue_grant",
        "osprey_connectors.write_authorization.parse_grant",
        "osprey_connectors.write_authorization.parse_ledger",
        "osprey_connectors.write_authorization.parse_request",
        "osprey_connectors.write_authorization.pid_namespace",
        "osprey_connectors.write_authorization.process_start_time",
        "osprey_connectors.write_authorization.read_ledger",
        "osprey_connectors.write_authorization.self_issued_grants",
        "osprey_connectors.write_authorization.spend_step",
        "osprey_connectors.write_target.FALLBACK_WRITE_TOOLS",
        "osprey_connectors.write_target.UNRESOLVED",
        "osprey_connectors.write_target.Unresolved",
    }
)


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


def stray_exports(
    root: Path,
    extension_points: Mapping[str, str] | None = None,
    baseline: Set[str] | None = None,
) -> list[str]:
    """One line per listed name in the tree at *root* that nothing outside its module uses.

    A stale entry of *extension_points* (no ``__all__`` lists it) or of
    *baseline* (it has gained a user, or no ``__all__`` lists it) is a line too.
    """
    declared = EXTENSION_POINTS if extension_points is None else extension_points
    interim = UNUSED_BASELINE if baseline is None else baseline
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
            used = bool(users.get((module, name), set()) - {path})
            if dotted in declared:
                continue
            if dotted in interim:
                if used:
                    lines.append(
                        f"{where}:{line}: {dotted} is in UNUSED_BASELINE and has a user "
                        "outside its module; remove it from UNUSED_BASELINE"
                    )
                continue
            if not used:
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
        f"{OWN_FILE}: {dotted} is in UNUSED_BASELINE and no __all__ lists it; "
        "remove it from UNUSED_BASELINE"
        for dotted in sorted(set(interim) - listed)
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
