"""Guard that importing the lattice modules does not drag in EPICS IOC deps.

``softioc``/``cothread`` must stay behind the virtual-accelerator entry point:
importing :mod:`osprey.simulation.lattice` in a plain venv has to work without
them. The check runs in a subprocess on purpose — once a module is imported into
the pytest host process the observation is worthless, so the target modules are
never imported here.
"""

import os
import subprocess
import sys
from pathlib import Path

CHECKOUT_SRC = Path(__file__).resolve().parents[2] / "src"


def test_lattice_import_is_plain_venv_clean():
    assert CHECKOUT_SRC.is_dir(), f"checkout src/ not found at {CHECKOUT_SRC}"
    code = (
        "import osprey.simulation.lattice, osprey.simulation.facility_spec, sys;"
        f"src={str(CHECKOUT_SRC) + os.sep!r};"
        "origin=osprey.simulation.lattice.__file__;"
        "assert origin.startswith(src), (origin, src);"
        "bad=sorted(m for m in sys.modules if m.split('.')[0] in ('softioc','cothread'));"
        "assert not bad, bad;"
        "print('CLEAN')"
    )
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
