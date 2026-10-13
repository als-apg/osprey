"""Guard that the framework never loads the optimizer's search stack.

xopt, torch and their kin are the ``optimizer`` extra, installed only in the
service's image. ``osprey`` and its CLI must load without them, and so must the
optimizer's own light modules (models, environment files), which phase 2's MCP
server will import. The check runs in a subprocess: once a module is in the
pytest process's ``sys.modules`` the observation is worthless.
"""

import os
import subprocess
import sys
from pathlib import Path

CHECKOUT_SRC = Path(__file__).resolve().parents[3] / "src"

#: The ``optimizer`` extra's heavy roots.
FORBIDDEN_ROOTS = ("xopt", "torch", "botorch", "gpytorch", "polars")

LIGHT_MODULES = (
    "osprey",
    "osprey.cli.main",
    "osprey.services.optimizer",
    "osprey.services.optimizer.models.api",
    "osprey.services.optimizer.models.config",
    "osprey.services.optimizer.environments.spec",
    "osprey.services.optimizer.environments.connector_env",
)


def test_osprey_and_the_optimizers_light_modules_load_without_the_search_stack() -> None:
    code = (
        "import importlib, sys\n"
        f"for name in {LIGHT_MODULES!r}:\n"
        "    importlib.import_module(name)\n"
        f"roots = {FORBIDDEN_ROOTS!r}\n"
        "loaded = sorted({name.split('.')[0] for name in sys.modules if name.split('.')[0] in roots})\n"
        "print(loaded if loaded else 'CLEAN')\n"
    )
    pythonpath = os.pathsep.join(p for p in (str(CHECKOUT_SRC), os.environ.get("PYTHONPATH")) if p)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=pythonpath),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "CLEAN", result.stdout
