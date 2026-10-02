"""Guard that the auth sidecar's import closure holds no deployment or CLI module.

The sidecar is a long-running service. The deployment layer imports the
sidecar's leaf modules and never the other way round, so loading the app and
every route module must not reach ``osprey.deployment``, whose render path loads
the CLI's output helpers and their prompt library. The check runs in a
subprocess on purpose: once a module is in the pytest process's ``sys.modules``
the observation is worthless, so the target modules are never imported here.
"""

import os
import subprocess
import sys
from pathlib import Path

CHECKOUT_SRC = Path(__file__).resolve().parents[3] / "src"

FORBIDDEN_ROOTS = (
    #: The deployment layer: compose render, credential provisioning, image builds.
    "osprey.deployment",
    #: The interactive command line and its output helpers.
    "osprey.cli",
    #: The prompt library the CLI's styles load.
    "questionary",
    #: The terminal toolkit underneath ``questionary``.
    "prompt_toolkit",
)


def test_sidecar_import_closure_holds_no_cli_or_deployment_modules() -> None:
    """Loading the app and every route module leaves every forbidden root unloaded."""
    # Arrange
    assert CHECKOUT_SRC.is_dir(), f"checkout src/ not found at {CHECKOUT_SRC}"
    code = (
        "import importlib, pkgutil, sys\n"
        "import osprey.services.auth_sidecar.app as app\n"
        "import osprey.services.auth_sidecar.routes as routes\n"
        "for info in pkgutil.iter_modules(routes.__path__):\n"
        "    importlib.import_module(f'{routes.__name__}.{info.name}')\n"
        f"src = {str(CHECKOUT_SRC) + os.sep!r}\n"
        "assert app.__file__.startswith(src), (app.__file__, src)\n"
        f"roots = {FORBIDDEN_ROOTS!r}\n"
        "loaded = sorted(\n"
        "    name for name in sys.modules\n"
        "    if any(name == root or name.startswith(root + '.') for root in roots)\n"
        ")\n"
        "print(loaded if loaded else 'CLEAN')\n"
    )
    pythonpath = os.pathsep.join(p for p in (str(CHECKOUT_SRC), os.environ.get("PYTHONPATH")) if p)

    # Act
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=pythonpath),
    )

    # Assert
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "CLEAN", result.stdout
