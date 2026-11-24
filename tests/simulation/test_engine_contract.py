"""The framework's engine plug-ins against the engine contract of ``osprey_connectors``.

The contract itself -- the group, the protocol, the optional names and the
calibration math -- is held by the connectors suite. These tests hold what the
framework ships: its registered engines satisfy the contract, and
``osprey.simulation.engines`` holds implementations only.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
import types
from importlib.metadata import entry_points
from pathlib import Path

import pytest

import osprey
import osprey.simulation.engines
from osprey_connectors.simulation.engines import (
    ENGINE_GROUP,
    OPTIONAL_NAMES,
    TIME_CONTRACT,
    EnginePlugin,
    FacilityDefectError,
)
from osprey_facility.errors import FacilityBuildError

#: The engine-neutral modules that ask an engine for a solve.
SOLVE_CALLERS = (
    "osprey/simulation/response.py",
    "osprey_facility/layers/mml/response_check.py",
)

#: The engines the framework registers.
FRAMEWORK_ENGINES = {
    "pyat": "osprey.simulation.engines.pyat",
    "injection": "osprey.simulation.engines.injection",
}


def _registered_engines() -> list[tuple[str, types.ModuleType]]:
    return [
        (point.name, point.load())
        for point in entry_points(group=ENGINE_GROUP)
        if point.name in FRAMEWORK_ENGINES
    ]


def test_the_framework_engines_register_under_the_contract_group():
    found = {point.name: point.value for point in entry_points(group=ENGINE_GROUP)}
    for name, module in FRAMEWORK_ENGINES.items():
        assert found.get(name) == module


@pytest.mark.parametrize("name", sorted(FRAMEWORK_ENGINES))
def test_each_framework_engine_satisfies_the_contract(name: str):
    module = entry_points(group=ENGINE_GROUP)[name].load()
    assert isinstance(module, EnginePlugin)


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


def test_every_optional_value_a_registered_engine_states_is_a_tuple_of_exceptions():
    for engine, module in _registered_engines():
        for name in (n for n in OPTIONAL_NAMES if n.endswith("_ERRORS")):
            value = getattr(module, name, None)
            if value is None:
                continue
            assert isinstance(value, tuple), f"{engine}.{name}"
            assert all(issubclass(item, Exception) for item in value), f"{engine}.{name}"


def test_the_package_holds_implementations_only():
    """The contract's names live in ``osprey_connectors``, never under ``osprey.``."""
    for name in ("ENTRY_POINT_GROUP", "OPTIONAL_NAMES", "TIME_CONTRACT", "EnginePlugin"):
        assert not hasattr(osprey.simulation.engines, name), name
    package = Path(osprey.simulation.engines.__file__).parent
    assert not (package / "calibration.py").exists()


def test_a_build_stop_is_a_facility_defect():
    """An engine raising a build stop on a request refuses it as a facility-file defect."""
    assert issubclass(FacilityBuildError, FacilityDefectError)


def test_importing_the_package_loads_no_engine_library():
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
    package, _, rest = relative.partition("/")
    path = Path(str(importlib.import_module(package).__file__)).parent / rest
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.append(node.module)
    assert [module for module in imported if _engine_import(module)] == []
