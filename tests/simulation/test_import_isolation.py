"""Guard that importing the model modules does not drag in EPICS IOC deps.

``softioc``/``cothread`` must stay behind the virtual-accelerator entry point:
importing the model layer in a plain venv has to work without them. The check runs in a subprocess on purpose — once a module is imported into
the pytest host process the observation is worthless, so the target modules are
never imported here.
"""

import os
import subprocess
import sys
from pathlib import Path

CHECKOUT_SRC = Path(__file__).resolve().parents[2] / "src"


def _assert_subprocess_clean(code: str) -> None:
    """Run `code` in a fresh interpreter and require it to print CLEAN."""
    pythonpath = os.pathsep.join(p for p in (str(CHECKOUT_SRC), os.environ.get("PYTHONPATH")) if p)
    env = dict(os.environ, PYTHONPATH=pythonpath)
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "CLEAN" in r.stdout


def test_va_model_import_is_softioc_clean():
    """The VA's LUME model layer must not drag in the EPICS IOC deps either.

    The pyat engine's model and its variable classes are the facility adapter
    over the ``lume-pyat`` package, and "softioc-free" is a shipping property
    rather than a preference: the package it binds to has no softioc to
    import, so a stray IOC import on this side would make the adapter
    unloadable anywhere the serving layer is not also installed. Both modules
    are checked -- the model imports the variables, but a variable module
    imported alone must be clean as well.

    The ``__file__`` assertions pin the checkout: resolving these modules to
    site-packages would mean the test is examining an installed OSPREY instead
    of the tree under test.
    """
    assert CHECKOUT_SRC.is_dir(), f"checkout src/ not found at {CHECKOUT_SRC}"
    code = (
        "import osprey.simulation.engines.pyat_variables as variables;"
        "import osprey.simulation.engines.pyat_model as model;"
        "import sys;"
        f"src={str(CHECKOUT_SRC) + os.sep!r};"
        "assert variables.__file__.startswith(src), (variables.__file__, src);"
        "assert model.__file__.startswith(src), (model.__file__, src);"
        "bad=sorted(m for m in sys.modules if m.split('.')[0] in ('softioc','cothread'));"
        "assert not bad, bad;"
        "print('CLEAN')"
    )
    _assert_subprocess_clean(code)
