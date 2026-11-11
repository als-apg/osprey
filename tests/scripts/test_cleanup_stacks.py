"""Tests for the stranded-stack cleanup ci.yml's stack lanes run with ``if: always()``.

The script runs on a failed or cancelled job, so it must remove every resource
it owes by exact name or its project's own prefix, and must never fail the job
itself. A fake ``docker`` and ``podman`` on ``PATH`` record the calls it makes.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "cleanup_stacks.sh"


def _fake_engine(bin_dir: Path, name: str, *, exit_code: int) -> Path:
    """A container CLI called *name* that logs its arguments and exits *exit_code*.

    A listing answers one entry under its name filter, and ``images`` its reference.
    """
    log = bin_dir / f"{name}.log"
    log.touch()
    engine = bin_dir / name
    engine.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> "{log}"\n'
        'case "$1 $2" in\n'
        '  "volume ls") echo "${5#name=^}vol" ;;\n'
        '  "ps -aq") echo "${4#name=^}c" ;;\n'
        '  "images --format") echo "$4" ;;\n'
        "esac\n"
        f"exit {exit_code}\n"
    )
    engine.chmod(engine.stat().st_mode | stat.S_IXUSR)
    return log


def _run_engines(
    tmp_path: Path, *args: str, exit_code: int = 0, engine: str | None = None
) -> tuple[int, dict[str, list[str]]]:
    logs = {
        name: _fake_engine(tmp_path, name, exit_code=exit_code) for name in ("docker", "podman")
    }
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}
    env.pop("ENGINE", None)
    if engine is not None:
        env["ENGINE"] = engine
    done = subprocess.run([str(_SCRIPT), *args], env=env, capture_output=True, check=False)
    return done.returncode, {name: log.read_text().splitlines() for name, log in logs.items()}


def _run(tmp_path: Path, *args: str, exit_code: int = 0) -> tuple[int, list[str]]:
    code, calls = _run_engines(tmp_path, *args, exit_code=exit_code)
    assert calls["podman"] == []
    return code, calls["docker"]


def _owed(project: str) -> list[str]:
    return [
        f"compose -p {project} down",
        f"rm -f {project}-agent-record",
        f"rm -f {project}-agent-record-clickhouse",
        f"volume ls -q --filter name=^{project}_",
        f"volume rm -f {project}_vol",
    ]


def _swept(project: str) -> list[str]:
    """``_owed`` with ``--sweep``: the containers and volumes named ``<project>-*`` too."""
    owed = _owed(project)
    return [
        *owed[:3],
        f"ps -aq --filter name=^{project}-",
        f"rm -f {project}-c",
        *owed[3:],
        f"volume ls -q --filter name=^{project}-",
        f"volume rm -f {project}-vol",
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


def test_engine_selects_the_container_cli(tmp_path: Path) -> None:
    """``ENGINE=podman`` runs every removal through podman and none through docker."""
    code, calls = _run_engines(tmp_path, "--sweep", "lane", engine="podman")
    assert code == 0
    assert calls == {"docker": [], "podman": _swept("lane")}


def test_sweep_removes_the_project_prefix_too(tmp_path: Path) -> None:
    code, calls = _run(tmp_path, "--sweep", "queue-e2e", "suproj")
    assert code == 0
    assert calls == [*_swept("queue-e2e"), *_swept("suproj")]


def test_images_are_removed_by_reference_after_every_project(tmp_path: Path) -> None:
    """Last, since a surviving container pins its image; an exact tag and a glob alike."""
    code, calls = _run(tmp_path, "--image", "lane:local", "--image", "lane-*", "lane")
    assert code == 0
    fmt = "{{.Repository}}:{{.Tag}}"
    assert calls == [
        *_owed("lane"),
        f"images --format {fmt} lane:local",
        "rmi -f lane:local",
        f"images --format {fmt} lane-*",
        "rmi -f lane-*",
    ]


def test_a_failing_image_removal_never_fails_the_cleanup(tmp_path: Path) -> None:
    code, calls = _run(tmp_path, "--sweep", "--image", "lane:local", "lane", exit_code=1)
    assert code == 0
    assert "rmi -f lane:local" in calls


def test_a_trailing_image_option_without_a_reference_removes_no_image(tmp_path: Path) -> None:
    code, calls = _run(tmp_path, "--image")
    assert (code, calls) == (0, [])
