"""Run functions of the shipped qmd entrypoint under ``sh`` against a stub ``qmd``.

The entrypoint ends with a bare ``main "$@"`` call. Dropping that line turns the
file into a library a test can source and call one function of, with the real
shell code and a stub ``qmd`` first on PATH in place of the binary the image has.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ENTRYPOINT = Path(__file__).resolve().parents[2] / "src/osprey/templates/services/qmd/entrypoint.sh"


def run_entrypoint(
    tmp_path: Path, *, stub: str, script: str, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Source the entrypoint minus ``main`` and run *script* after it under ``sh``.

    *stub* is the body of the ``qmd`` put first on PATH. *env* adds to a minimal
    environment that sets PATH, a HOME under *tmp_path*, and a qmd port.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    qmd = bin_dir / "qmd"
    qmd.write_text(stub)
    qmd.chmod(0o755)

    lines = ENTRYPOINT.read_text().splitlines()
    assert lines[-1] == 'main "$@"', "the entrypoint no longer ends with its main call"
    library = tmp_path / "entrypoint-lib.sh"
    library.write_text("\n".join(lines[:-1]) + "\n")

    full_env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "OSPREY_QMD_PORT": "1",
        **env,
    }
    return subprocess.run(
        ["sh", "-c", f'. "{library}"; {script}'],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
