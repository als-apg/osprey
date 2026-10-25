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


#: The commands a wiring record naming no element can give the pyAT engine.
PYAT_COMMANDS = frozenset({"arm", "inject", "clear_stored_charge"})


def _command_record(attribute: str) -> dict[str, object]:
    return {
        "address": f"CMD:{attribute}",
        "direction": "write",
        "engine": {"attribute": attribute},
        "default": 0,
        "value_type": "int",
    }


def test_pyat_classifies_exactly_its_commands():
    pytest.importorskip("lume_pyat")
    from osprey.simulation.engines import pyat

    assert pyat.COMMAND_ATTRIBUTES == PYAT_COMMANDS


@pytest.mark.parametrize("attribute", sorted(PYAT_COMMANDS))
def test_a_command_record_is_described_as_a_command(attribute: str):
    pytest.importorskip("lume_pyat")
    from osprey.simulation.engines import pyat

    description = pyat.describe(_command_record(attribute))
    assert description[pyat.COMMAND_MARK] is True
    assert description["role"] == "setpoint"


def test_a_command_attribute_on_an_element_is_no_command():
    pytest.importorskip("lume_pyat")
    from osprey.simulation.engines import pyat

    record = {**_command_record("clear_stored_charge"), "element": "BPM_01"}
    assert not pyat._is_command(record)


#: The global settings a wiring record naming no element can give the pyAT
#: engine: one model-wide engine value each, held by the model.
PYAT_GLOBALS = frozenset({"turns_tracked"})


def _global_record(attribute: str, direction: str = "write") -> dict[str, object]:
    return {
        "address": f"GLOBAL:{attribute}",
        "direction": direction,
        "engine": {"attribute": attribute},
        "value_type": "int",
    }


def test_pyat_classifies_exactly_its_global_settings():
    pytest.importorskip("lume_pyat")
    from osprey.simulation.engines import pyat

    assert pyat.GLOBAL_ATTRIBUTES == PYAT_GLOBALS
    assert not pyat.GLOBAL_ATTRIBUTES & pyat.COMMAND_ATTRIBUTES
    assert not pyat.GLOBAL_ATTRIBUTES & set(pyat.OPTICS_ATTRIBUTES)
    assert not pyat.GLOBAL_ATTRIBUTES & set(pyat.DECK_PROPERTIES)


@pytest.mark.parametrize("attribute", sorted(PYAT_GLOBALS))
@pytest.mark.parametrize(("direction", "role"), [("write", "setpoint"), ("read", "readback")])
def test_a_global_record_is_described_as_a_global_setting(
    attribute: str, direction: str, role: str
):
    pytest.importorskip("lume_pyat")
    from osprey.simulation.engines import pyat

    description = pyat.describe(_global_record(attribute, direction))
    assert description == {
        "role": role,
        "plane": None,
        "refresh": "pass",
        "kind": pyat.GLOBAL_KIND,
    }
    assert pyat.COMMAND_MARK not in description


#: The attribute a read naming no element reads the pyAT engine's stored
#: charge by, served as a current.
PYAT_STORED_CURRENT = "stored_current"


def _stored_current_record(**extra: object) -> dict[str, object]:
    return {
        "address": "BEAM:CURRENT",
        "direction": "read",
        "engine": {"attribute": PYAT_STORED_CURRENT},
        **extra,
    }


def test_pyat_reads_its_stored_charge_by_exactly_one_attribute():
    pytest.importorskip("lume_pyat")
    from osprey.simulation.engines import pyat

    assert pyat.STORED_CURRENT_ATTRIBUTE == PYAT_STORED_CURRENT
    assert PYAT_STORED_CURRENT not in pyat.COMMAND_ATTRIBUTES
    assert PYAT_STORED_CURRENT not in pyat.GLOBAL_ATTRIBUTES
    assert PYAT_STORED_CURRENT not in pyat.OPTICS_ATTRIBUTES
    assert PYAT_STORED_CURRENT not in pyat.DECK_PROPERTIES
    assert PYAT_STORED_CURRENT != pyat.ARMED_ATTRIBUTE


def test_a_stored_current_record_is_an_output_the_periodic_pass_publishes():
    pytest.importorskip("lume_pyat")
    from osprey.simulation.engines import pyat

    description = pyat.describe(_stored_current_record())
    assert description == {"role": "output", "plane": None, "refresh": "periodic", "kind": None}


def test_a_stored_current_attribute_on_an_element_is_no_stored_current():
    pytest.importorskip("lume_pyat")
    from osprey.simulation.engines import pyat

    assert pyat._is_stored_current(_stored_current_record())
    assert not pyat._is_stored_current(_stored_current_record(element="BPM_01"))
    assert not pyat._is_stored_current(_stored_current_record(direction="write"))


@pytest.mark.parametrize("attribute", sorted(PYAT_GLOBALS))
def test_a_global_attribute_on_an_element_is_no_global_setting(attribute: str):
    pytest.importorskip("lume_pyat")
    from osprey.simulation.engines import pyat

    record = {**_global_record(attribute), "element": "BPM_01"}
    assert not pyat._is_global(record)
    assert pyat.describe(record)["kind"] != pyat.GLOBAL_KIND
