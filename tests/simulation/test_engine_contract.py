"""The engine plug-in contract: what a plug-in module states and what it may declare."""

from __future__ import annotations

import subprocess
import sys
import types
from importlib.metadata import entry_points

from osprey.simulation.engines import ENTRY_POINT_GROUP, EnginePlugin


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
