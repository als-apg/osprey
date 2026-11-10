"""Tests for the stranded-stack cleanup ci.yml's stack lanes run with ``if: always()``.

The script runs on a failed or cancelled job, so it must remove every resource
it owes by exact name and must never fail the job itself. A fake ``docker`` on
``PATH`` records the calls it makes.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "cleanup_stacks.sh"


def _fake_docker(bin_dir: Path, log: Path, *, exit_code: int) -> None:
    """A ``docker`` that logs its arguments, lists one volume, and exits *exit_code*."""
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> "{log}"\n'
        'if [[ "$1 $2" == "volume ls" ]]; then echo "${5#name=^}vol"; fi\n'
        f"exit {exit_code}\n"
    )
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR)


def _run(tmp_path: Path, *projects: str, exit_code: int = 0) -> tuple[int, list[str]]:
    log = tmp_path / "calls.log"
    log.touch()
    _fake_docker(tmp_path, log, exit_code=exit_code)
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}
    done = subprocess.run([str(_SCRIPT), *projects], env=env, capture_output=True, check=False)
    return done.returncode, log.read_text().splitlines()


def _owed(project: str) -> list[str]:
    return [
        f"compose -p {project} down",
        f"rm -f {project}-agent-record",
        f"rm -f {project}-agent-record-clickhouse",
        f"volume ls -q --filter name=^{project}_",
        f"volume rm -f {project}_vol",
    ]


def test_the_script_is_executable() -> None:
    """ci.yml calls it by path, not through ``bash``."""
    assert os.access(_SCRIPT, os.X_OK)


def test_every_project_is_removed_by_exact_name(tmp_path: Path) -> None:
    code, calls = _run(tmp_path, "queue-e2e", "suproj")
    assert code == 0
    assert calls == [*_owed("queue-e2e"), *_owed("suproj")]


@pytest.mark.parametrize("exit_code", [1, 125])
def test_a_failing_docker_never_fails_the_cleanup(tmp_path: Path, exit_code: int) -> None:
    """Nothing to remove (or no daemon) must not turn a cleanup step red."""
    code, calls = _run(tmp_path, "queue-e2e", "suproj", exit_code=exit_code)
    assert code == 0
    assert [c for c in calls if c.startswith("compose")] == [
        "compose -p queue-e2e down",
        "compose -p suproj down",
    ]


def test_no_docker_on_the_path_never_fails_the_cleanup(tmp_path: Path) -> None:
    """A PATH holding only what the script needs besides docker."""
    for tool in ("bash", "xargs"):
        found = shutil.which(tool)
        assert found is not None, tool
        (tmp_path / tool).symlink_to(found)
    env = {**os.environ, "PATH": str(tmp_path)}
    done = subprocess.run([str(_SCRIPT), "queue-e2e"], env=env, capture_output=True, check=False)
    assert done.returncode == 0, done.stderr
    assert b"docker" in done.stderr, "docker was found after all; the test proves nothing"


def test_no_project_removes_nothing(tmp_path: Path) -> None:
    assert _run(tmp_path) == (0, [])
