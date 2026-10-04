"""The scenario readers of ``osprey.simulation.apply`` read the simulator view.

The logbook, the anchor and the archiver events of the active set come from the
render's ``data/simulator/`` view and the active-scenarios state file, never
from the simulation engine or its machine model.
"""

from __future__ import annotations

import ast
from pathlib import Path

import osprey.simulation.apply as apply_module

#: The engine and machine modules, under both of their spellings.
FORBIDDEN_MODULES = frozenset(
    {
        "osprey.simulation.engine",
        "osprey.simulation.machine",
        "osprey_connectors.simulation.engine",
        "osprey_connectors.simulation.machine",
    }
)

#: The functions that read the view.
VIEW_READERS = ("active_logbook_entries", "persisted_scenario_anchor", "active_archiver_events")


def _tree() -> ast.Module:
    return ast.parse(Path(apply_module.__file__).read_text(encoding="utf-8"))


def _imports_forbidden(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        return node.module in FORBIDDEN_MODULES
    if isinstance(node, ast.Import):
        return any(alias.name in FORBIDDEN_MODULES for alias in node.names)
    return False


def _forbidden_names(tree: ast.Module) -> set[str]:
    """Every name the module binds by importing from an engine or machine module."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if _imports_forbidden(node):
            assert isinstance(node, ast.Import | ast.ImportFrom)
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return names


def test_logbook_anchor_and_archiver_events_bodies_call_no_engine_or_machine():
    tree = _tree()
    forbidden = _forbidden_names(tree)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in VIEW_READERS
    }
    assert sorted(functions) == sorted(VIEW_READERS)

    offences: list[str] = []
    for name, function in functions.items():
        for node in ast.walk(function):
            if _imports_forbidden(node):
                offences.append(f"{name}:{node.lineno} imports an engine or machine module")
            elif isinstance(node, ast.Name) and node.id in forbidden:
                offences.append(f"{name}:{node.lineno} uses {node.id}")

    assert offences == []


def test_logbook_anchor_and_archiver_events_check_sees_both_module_spellings():
    """The check is live: a body naming the engine under either spelling is caught."""
    for module in sorted(FORBIDDEN_MODULES):
        source = (
            f"from {module} import SimulationEngine\n"
            "def active_logbook_entries(config, project_dir):\n"
            "    return SimulationEngine\n"
        )
        tree = ast.parse(source)

        assert _forbidden_names(tree) == {"SimulationEngine"}
