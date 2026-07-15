import os
import subprocess
import sys


def test_lattice_import_is_plain_venv_clean():
    worktree_src = "/Users/thellert/LBL/ML/osprey/.claude/worktrees/one-facility/src"
    code = (
        "import osprey.simulation.lattice, osprey.simulation.facility_spec, sys;"
        "bad=sorted(m for m in sys.modules if m.split('.')[0] in ('softioc','cothread'));"
        "assert not bad, bad;"
        "print('CLEAN')"
    )
    env = dict(os.environ, PYTHONPATH=worktree_src)
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "CLEAN" in r.stdout
