"""The engine plug-in contract: what a plug-in module states and what it may declare."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import types
from importlib.metadata import entry_points
from pathlib import Path

import pytest

import osprey
from osprey.simulation.engines import (
    ENTRY_POINT_GROUP,
    OPTIONAL_NAMES,
    TIME_CONTRACT,
    EnginePlugin,
)

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


#: The names every plug-in module states; the time contract adds none.
REQUIRED_NAMES = ("locate", "prepare", "start_values", "describe", "build")


def _registered_engines() -> list[tuple[str, types.ModuleType]]:
    return [(point.name, point.load()) for point in entry_points(group=ENTRY_POINT_GROUP)]


def _documented_names() -> list[str]:
    """The first-column names of the optional-name table in the protocol's docstring."""
    lines = (EnginePlugin.__doc__ or "").splitlines()
    rules = [index for index, line in enumerate(lines) if line.strip().startswith("=====")]
    assert len(rules) == 3, "the docstring states one simple table"
    rule = lines[rules[0]]
    indent = len(rule) - len(rule.lstrip())
    width = len(rule.split()[0])
    names = []
    for line in lines[rules[1] + 1 : rules[2]]:
        cell = line[indent : indent + width]
        match = re.match(r"``([A-Za-z_]+)", cell)
        if match:
            names.append(match.group(1))
    return names


def test_the_time_contract_does_not_grow_the_required_names():
    module = types.ModuleType("five_functions")
    for name in REQUIRED_NAMES:
        setattr(module, name, _function)
    assert isinstance(module, EnginePlugin)
    for name in TIME_CONTRACT + ("deckless_start_values",):
        assert not hasattr(EnginePlugin, name)


def test_the_time_contract_names_are_optional_names():
    assert TIME_CONTRACT == ("advance", "next_deadline", "emitted", "client_write")
    assert set(TIME_CONTRACT) <= set(OPTIONAL_NAMES)
    assert "deckless_start_values" in OPTIONAL_NAMES
    assert not set(REQUIRED_NAMES) & set(OPTIONAL_NAMES)


def test_the_documented_optional_names_match_the_module():
    documented = _documented_names()
    assert len(documented) == len(set(documented))
    assert sorted(documented) == sorted(OPTIONAL_NAMES)


@pytest.mark.parametrize("name", TIME_CONTRACT)
def test_every_registered_engine_omits_or_provides_each_time_contract_name(name: str):
    engines = _registered_engines()
    assert engines, "at least one engine is registered"
    for engine, module in engines:
        value = getattr(module, name, None)
        assert value is None or callable(value), f"{engine}.{name} is not callable"


def test_every_registered_engine_omits_or_sets_the_deckless_flag():
    for engine, module in _registered_engines():
        flag = getattr(module, "deckless_start_values", None)
        assert flag is None or isinstance(flag, bool), f"{engine}.deckless_start_values"


def test_an_engine_without_the_time_contract_still_satisfies_the_contract():
    module = types.ModuleType("lattice_only")
    for name in REQUIRED_NAMES:
        setattr(module, name, _function)
    assert isinstance(module, EnginePlugin)
    assert all(getattr(module, name, None) is None for name in TIME_CONTRACT)


def test_a_time_driven_engine_satisfies_the_contract():
    module = types.ModuleType("time_driven")
    for name in REQUIRED_NAMES + TIME_CONTRACT:
        setattr(module, name, _function)
    module.deckless_start_values = True
    assert isinstance(module, EnginePlugin)


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


def test_pyat_declares_its_request_refusal():
    pytest.importorskip("lume_pyat")
    from osprey.simulation.engines import pyat

    assert pyat.REFUSAL_ERRORS == (pyat.RequestRefused,)
    assert issubclass(pyat.RequestRefused, ValueError)


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
