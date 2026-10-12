"""The engine plug-in contract: what a plug-in module states and what it may declare."""

from __future__ import annotations

import ast
import subprocess
import sys
import types
from importlib.metadata import entry_points
from pathlib import Path

import pytest

import osprey
from osprey.simulation.engines import ENTRY_POINT_GROUP, EnginePlugin

#: The engine-neutral modules that ask an engine for a solve.
SOLVE_CALLERS = (
    "simulation/response.py",
    "facility/layers/mml/response_check.py",
)


def _function(*args, **kwargs):
    return None


def test_the_pyat_plug_in_satisfies_the_contract():
    module = entry_points(group=ENTRY_POINT_GROUP)["pyat"].load()
    assert isinstance(module, EnginePlugin)


def test_a_module_missing_build_does_not():
    module = types.ModuleType("four_functions")
    for name in ("locate", "prepare", "start_values", "describe"):
        setattr(module, name, _function)
    assert not isinstance(module, EnginePlugin)


def test_importing_the_contract_loads_no_engine_library():
    code = (
        "import sys\n"
        "import osprey.simulation.engines\n"
        "loaded = [m for m in ('at', 'lume', 'lume_pyat', 'numpy') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


def test_pyat_declares_the_orbit_solve_error():
    pytest.importorskip("lume_pyat")
    from lume_pyat.exceptions import OrbitSolveError

    from osprey.simulation.engines import pyat

    assert pyat.SOLVE_ERRORS == (OrbitSolveError,)


def _engine_import(module: str) -> bool:
    if module in ("lume_pyat", "at"):
        return True
    return module.startswith(("lume_pyat.", "at.", "osprey.simulation.engines.pyat"))


@pytest.mark.parametrize("relative", SOLVE_CALLERS)
def test_the_engine_neutral_solve_callers_import_no_engine(relative: str):
    path = Path(osprey.__file__).parent / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.append(node.module)
    assert [module for module in imported if _engine_import(module)] == []
