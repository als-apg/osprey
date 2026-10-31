"""The VA gate script refuses a runtime that cannot build its image before staging.

The virtual accelerator image is linux/amd64 only, and its Containerfile's
``FROM --platform`` pin is honoured by BuildKit alone. The script takes its
runtime from ``osprey.deployment.runtime_helper --build``, the decision
``osprey up`` makes, so a docker without BuildKit stops before the build
context is staged, naming how to enable it, and a docker pinned on a host where
it is podman's docker emulation stops naming the setting to change. Every
builder is asked for linux/amd64 on the command line.

These run the real script against a stub ``docker`` on ``PATH`` and read the
real build command; no image is built.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.skipif(shutil.which("bash") is None, reason="the gate script is a bash script"),
    pytest.mark.skipif(
        not (REPO_ROOT / ".venv/bin/python").exists(),
        reason="the gate script decides its runtime with the worktree venv's python",
    ),
]
SCRIPT = REPO_ROOT / "scripts/va/build_and_boot_check.sh"
STAGING_LINE = "Staging minimal build context"

_STUB = """#!/bin/sh
echo "$@" >> "$STUB_LOG"
if [ "$1" = "--version" ]; then
    printf '%s\\n' "$STUB_VERSION"
    exit 0
fi
if [ "$1" = "buildx" ]; then
    exit "$STUB_BUILDX_RC"
fi
if [ "$1 $2" = "compose version" ] || [ "$1" = "ps" ]; then
    exit 0
fi
if [ "$1" = "build" ]; then
    exit 1
fi
exit 0
"""


@pytest.fixture
def stub_docker(tmp_path: Path) -> Path:
    """An executable ``docker`` that logs every call, one line per call."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "docker"
    stub.write_text(_STUB)
    stub.chmod(0o755)
    return stub


def _run(
    tmp_path: Path,
    stub: Path,
    *,
    buildx_rc: int,
    buildkit: str | None = None,
    version: str = "",
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    log = tmp_path / "stub.log"
    log.touch()
    env = {k: v for k, v in os.environ.items() if k not in ("DOCKER_BUILDKIT", "PATH")}
    env.update(
        PATH=f"{stub.parent}{os.pathsep}{os.environ.get('PATH', '')}",
        CONTAINER_RUNTIME="docker",
        TMPDIR=str(tmp_path),
        STUB_LOG=str(log),
        STUB_BUILDX_RC=str(buildx_rc),
        STUB_VERSION=version,
    )
    if buildkit is not None:
        env["DOCKER_BUILDKIT"] = buildkit
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result, log.read_text().splitlines()


def test_docker_without_buildx_stops_before_staging_and_names_how_to_enable_it(
    tmp_path: Path, stub_docker: Path
) -> None:
    result, calls = _run(tmp_path, stub_docker, buildx_rc=1)

    assert result.returncode == 1
    assert "docker-buildx-plugin" in result.stderr
    assert "DOCKER_BUILDKIT=1" in result.stderr
    assert STAGING_LINE not in result.stdout
    assert "buildx version" in calls
    assert not any(call.startswith("build ") for call in calls)


def test_docker_with_buildkit_switched_off_stops_before_staging(
    tmp_path: Path, stub_docker: Path
) -> None:
    result, calls = _run(tmp_path, stub_docker, buildx_rc=0, buildkit="0")

    assert result.returncode == 1
    assert "legacy builder" in result.stderr
    assert "Unset DOCKER_BUILDKIT" in result.stderr
    assert STAGING_LINE not in result.stdout
    assert not any(call.startswith("build ") for call in calls)


def test_docker_that_is_podman_emulation_is_refused_when_pinned(
    tmp_path: Path, stub_docker: Path
) -> None:
    result, calls = _run(tmp_path, stub_docker, buildx_rc=1, version="podman version 5.0.0")

    assert result.returncode == 1
    assert "container_runtime: podman" in result.stderr
    assert "CONTAINER_RUNTIME=podman" in result.stderr
    assert "BuildKit" not in result.stderr
    assert STAGING_LINE not in result.stdout
    assert calls == ["--version"]


@pytest.mark.parametrize(
    ("buildx_rc", "buildkit"),
    [(0, None), (1, "1")],
    ids=["buildx-present", "buildkit-env"],
)
def test_docker_with_buildkit_passes_the_check(
    tmp_path: Path, stub_docker: Path, buildx_rc: int, buildkit: str | None
) -> None:
    result, _ = _run(tmp_path, stub_docker, buildx_rc=buildx_rc, buildkit=buildkit)

    # The stub refuses `build`, so the run stops later; only the check is judged.
    assert STAGING_LINE in result.stdout
    assert "FATAL: docker has no BuildKit" not in result.stderr
    assert "legacy builder" not in result.stderr


def test_the_image_build_asks_for_linux_amd64() -> None:
    text = SCRIPT.read_text().replace("\\\n", " ")
    builds = [
        line.strip() for line in text.splitlines() if line.strip().startswith('"${RUNTIME}" build')
    ]

    assert len(builds) == 1
    assert "--platform linux/amd64" in builds[0]
