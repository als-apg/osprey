"""Run functions of the shipped qmd entrypoint under ``sh`` against a stub ``qmd``.

The entrypoint ends with a bare ``main "$@"`` call. Dropping that line turns the
file into a library a test can source and call one function of, with the real
shell code and a stub ``qmd`` first on PATH in place of the binary the image has.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
from pathlib import Path

ENTRYPOINT = Path(__file__).resolve().parents[2] / "src/osprey/templates/services/qmd/entrypoint.sh"


def _prepare(
    tmp_path: Path, *, stub: str, script: str, env: dict[str, str]
) -> tuple[str, dict[str, str]]:
    """Write the stub and the library; return the ``-c`` command and its environment."""
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
    return f'. "{library}"; {script}', full_env


def run_entrypoint(
    tmp_path: Path, *, stub: str, script: str, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Source the entrypoint minus ``main`` and run *script* after it under ``sh``.

    *stub* is the body of the ``qmd`` put first on PATH. *env* adds to a minimal
    environment that sets PATH, a HOME under *tmp_path*, and a qmd port.
    """
    command, full_env = _prepare(tmp_path, stub=stub, script=script, env=env)
    return subprocess.run(
        ["sh", "-c", command],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def run_entrypoint_for_leftovers(
    tmp_path: Path, *, shell: str, stub: str, script: str, env: dict[str, str]
) -> tuple[int | None, str, list[str]]:
    """Run *script* as :func:`run_entrypoint` does, under *shell*, and list what outlives it.

    The shell leads a process group of its own, which every process it starts
    inherits and keeps after it is orphaned. Once the shell has exited, any live
    member of that group is a process the script left behind. Output goes to a
    file, so a leftover holding it open cannot stall the run. The group is
    killed before returning, which reaches only processes this run started.

    Returns the shell's exit status (``None`` if it was still running after 60
    seconds), its combined output, and one ``ps`` line per leftover.
    """
    command, full_env = _prepare(tmp_path, stub=stub, script=script, env=env)
    output = tmp_path / "output"
    with output.open("w") as sink:
        proc = subprocess.Popen(
            [shell, "-c", command],
            env=full_env,
            stdout=sink,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        try:
            returncode: int | None = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            returncode = None
        table = subprocess.run(
            ["ps", "-A", "-o", "pid=,pgid=,stat=,args="],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        leftovers = [
            line.strip()
            for line in table.splitlines()
            if line.split()[1] == str(proc.pid)
            and line.split()[0] != str(proc.pid)
            and not line.split()[2].startswith("Z")
        ]
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    return returncode, output.read_text(), leftovers
