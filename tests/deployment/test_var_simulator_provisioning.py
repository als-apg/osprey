"""Host-side provisioning of the simulator's model-log directories.

``var/simulator/`` and ``var/simulator/standin/`` are bound read-write into
every container that runs the composite, each container appending under its
own uid. They are created before compose runs, setgid and group-writable, so
the containers share one group and none of them finds a root-owned source.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import sysconfig
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from osprey.deployment.compose_generator import (
    _ensure_agent_data_structure,
    ensure_simulator_log_dirs,
    simulated_target_configured,
)
from osprey_connectors.simulation.composite import Composite
from tests._container_support import docker_cli_unavailable_reason

REPO_ROOT = Path(__file__).resolve().parents[2]
PURELIB = Path(sysconfig.get_paths()["purelib"])


def _assert_shared(path: Path) -> None:
    mode = path.stat().st_mode
    assert path.is_dir()
    assert mode & stat.S_ISGID
    assert mode & stat.S_IWGRP


def test_both_log_directories_are_provisioned_setgid_and_group_writable(
    tmp_path: Path,
) -> None:
    gid = ensure_simulator_log_dirs(tmp_path)

    _assert_shared(tmp_path / "var" / "simulator")
    _assert_shared(tmp_path / "var" / "simulator" / "standin")
    assert gid == (tmp_path / "var" / "simulator").stat().st_gid


def _config(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    return {"project_root": str(tmp_path), **extra}


@pytest.mark.parametrize(
    "extra",
    [
        {"deployed_services": ["virtual_accelerator"]},
        {
            "control_system": {
                "type": "virtual_accelerator",
                "connector": {"virtual_accelerator": {"serving": "in_process"}},
            }
        },
        {
            "control_system": {
                "type": "epics",
                "connector": {"epics": {"gateways": {}}, "virtual_accelerator": {"port": 5064}},
            }
        },
    ],
    ids=["va-deployed", "in-process", "va-target-beside-live"],
)
def test_a_deployment_with_a_simulated_target_provisions_them(
    tmp_path: Path, extra: dict[str, Any]
) -> None:
    _ensure_agent_data_structure(_config(tmp_path, **extra))

    _assert_shared(tmp_path / "var" / "simulator")
    _assert_shared(tmp_path / "var" / "simulator" / "standin")


def test_a_live_only_deployment_provisions_none(tmp_path: Path) -> None:
    config = _config(
        tmp_path, control_system={"type": "epics", "connector": {"epics": {"gateways": {}}}}
    )

    assert not simulated_target_configured(config)
    _ensure_agent_data_structure(config)

    assert not (tmp_path / "var" / "simulator").exists()


def _append(log_dir: Path, instance: str) -> None:
    """Append one record through the composite's own log writer."""
    writer = SimpleNamespace(_instance=instance, _log_dir=log_dir, _log_line=Composite._log_line)
    Composite._log(writer, "M", {"event": "noticed"})  # type: ignore[arg-type]


def _container_venue(platform: str, in_ci: bool, docker_unavailable: str | None) -> tuple[str, str]:
    """Whether the two-uid case runs here, skips, or fails, and why.

    The case proves a group-permission property only a second uid can observe.
    On a Linux CI runner a missing engine is the lane being wrong about itself,
    so it fails instead of skipping.
    """
    if platform != "linux":
        return ("skip", "the writers run this host's Linux venv inside a Linux container")
    if docker_unavailable and in_ci:
        return ("fail", f"a Linux CI runner must run the two-uid case: {docker_unavailable}")
    if docker_unavailable:
        return ("skip", docker_unavailable)
    return ("run", "")


@pytest.mark.parametrize(
    ("platform", "in_ci", "docker_unavailable", "verdict"),
    [
        ("darwin", True, None, "skip"),
        ("linux", True, "docker CLI not on PATH", "fail"),
        ("linux", False, "docker CLI not on PATH", "skip"),
        ("linux", True, None, "run"),
    ],
)
def test_the_two_uid_case_runs_skips_or_fails_by_venue(
    platform: str, in_ci: bool, docker_unavailable: str | None, verdict: str
) -> None:
    assert _container_venue(platform, in_ci, docker_unavailable)[0] == verdict


#: Run inside the container: the composite's own writer, under the container
#: umask, with the host venv's site-packages (and its .pth entries) on the path.
WRITER = textwrap.dedent(
    """
    import os
    import site
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    os.umask(0o022)
    site.addsitedir(sys.argv[1])

    from osprey_connectors.simulation.composite import Composite

    writer = SimpleNamespace(
        _instance=sys.argv[2], _log_dir=Path("/simulator-log"), _log_line=Composite._log_line
    )
    Composite._log(writer, "M", {"event": "noticed"})
    """
)


def _append_in_container(
    log_dir: Path, uid: int, gid: int, instance: str
) -> subprocess.CompletedProcess[str]:
    """Append one record from a container running as ``uid:gid``.

    The repository and the host venv's site-packages are mounted read-only at
    their host paths, so the editable installs' path entries stay valid; the
    image's CPython minor matches the venv's, which compiled wheels require.
    """
    image = f"python:{sys.version_info.major}.{sys.version_info.minor}-slim"
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            f"{uid}:{gid}",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-v",
            f"{log_dir}:/simulator-log",
            "-v",
            f"{REPO_ROOT}:{REPO_ROOT}:ro",
            "-v",
            f"{PURELIB}:{PURELIB}:ro",
            image,
            "python",
            "-c",
            WRITER,
            str(PURELIB),
            instance,
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_two_writers_append_to_one_log_under_the_container_umask(tmp_path: Path) -> None:
    ensure_simulator_log_dirs(tmp_path)
    log_dir = tmp_path / "var" / "simulator"
    previous = os.umask(0o022)
    try:
        _append(log_dir, "virtual_accelerator")
        _append(log_dir, "inprocess")
    finally:
        os.umask(previous)

    log = log_dir / "M.log"
    assert stat.S_IMODE(log.stat().st_mode) == 0o664
    assert log.stat().st_gid == log_dir.stat().st_gid
    assert len(log.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.xdist_group("docker")
def test_two_uids_in_one_group_append_to_one_log_from_two_containers(tmp_path: Path) -> None:
    """Two containers under different uids, sharing only the directory's group,
    both append to one model log through the composite's own writer."""
    platform = sys.platform
    unavailable = docker_cli_unavailable_reason() if platform == "linux" else None
    verdict, reason = _container_venue(platform, bool(os.environ.get("CI")), unavailable)
    if verdict == "skip":
        pytest.skip(reason)
    if verdict == "fail":
        pytest.fail(reason)

    gid = ensure_simulator_log_dirs(tmp_path)
    assert gid is not None
    log_dir = tmp_path / "var" / "simulator"
    for uid, instance in ((60001, "virtual_accelerator"), (60002, "inprocess")):
        result = _append_in_container(log_dir, uid, gid, instance)
        assert result.returncode == 0, result.stderr
        assert "cannot append" not in result.stderr + result.stdout

    log = log_dir / "M.log"
    info = log.stat()
    assert info.st_uid == 60001
    assert stat.S_IMODE(info.st_mode) == 0o664
    assert info.st_gid == gid
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, lines
    assert len([line for line in lines if "virtual_accelerator" in line]) == 1, lines
    assert len([line for line in lines if "inprocess" in line]) == 1, lines
