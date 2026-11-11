"""Guard the two import boundaries around the optimizer.

The framework does not need the optimizer: ``osprey`` and its CLI load
without ``osprey_optimizer`` at all. And the optimizer's base layer (models
and environment files, which the framework imports) loads without the
``service`` extra: xopt, torch and their kin are installed only in the
optimizer's image. Each check runs in a fresh subprocess: once a module is in
the pytest process's ``sys.modules`` the observation is worthless.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKOUT_PATHS = (
    REPO_ROOT / "src",
    REPO_ROOT / "packages" / "osprey-optimizer" / "src",
)

#: The ``service`` extra's roots that nothing in the base may load. pandas and
#: numpy are absent on purpose: osprey-connectors, a base dependency, has them.
SERVICE_ROOTS = ("xopt", "torch", "botorch", "gpytorch", "polars", "redis", "fastapi", "sklearn")

FRAMEWORK_MODULES = ("osprey", "osprey.cli.main")

BASE_MODULES = (
    "osprey_optimizer",
    "osprey_optimizer.models",
    "osprey_optimizer.models.api",
    "osprey_optimizer.models.config",
    "osprey_optimizer.environments.spec",
    "osprey_optimizer.environments.connector_env",
)


def _roots_loaded_after_importing(modules: tuple[str, ...], roots: tuple[str, ...]) -> str:
    code = (
        "import importlib, sys\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        f"roots = {roots!r}\n"
        "loaded = sorted({name.split('.')[0] for name in sys.modules if name.split('.')[0] in roots})\n"
        "print(loaded if loaded else 'CLEAN')\n"
    )
    pythonpath = os.pathsep.join(
        p for p in (*map(str, CHECKOUT_PATHS), os.environ.get("PYTHONPATH")) if p
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=pythonpath),
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_osprey_loads_without_the_optimizer() -> None:
    roots = ("osprey_optimizer", *SERVICE_ROOTS)
    assert _roots_loaded_after_importing(FRAMEWORK_MODULES, roots) == "CLEAN"


def test_the_optimizers_base_loads_without_the_service_extra() -> None:
    assert _roots_loaded_after_importing(BASE_MODULES, SERVICE_ROOTS) == "CLEAN"
