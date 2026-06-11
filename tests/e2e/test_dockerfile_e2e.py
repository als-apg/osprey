"""E2E: build and boot the generated reference Dockerfile.

Renders a hello-world project, runs a real ``docker build`` on the
Dockerfile that ``osprey build`` generated, and smoke-tests the image:

- ``osprey --version`` works (OSPREY installed and importable),
- the runtime user is the non-root ``osprey`` user,
- ``claude --version`` works as that user (canary for the /root/.local
  permission-traversal chain the native installer requires),
- ``config.yml`` inside the image points at the container path (the
  ``osprey claude regen --runtime-root`` build step healed the host paths —
  the host build here uses ``--skip-deps``, which records the host
  interpreter, so this genuinely exercises the relocation path).

Set ``OSPREY_E2E_PIP_SPEC`` to override which OSPREY gets installed inside
the image. The image default is the PyPI release, which only works once a
release containing ``regen --runtime-root`` is published — CI pins the PR
head SHA instead so the image tests the branch under review.

Skipped entirely when docker is unavailable. Excluded from the main
e2e-tests CI job (runs in its own dockerfile-e2e job).
"""

import os
import platform
import shutil
import subprocess
import uuid

import pytest
from click.testing import CliRunner

from osprey.cli.main import cli


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = [
    pytest.mark.dockerbuild,
    pytest.mark.skipif(not _docker_available(), reason="docker binary or daemon not available"),
]

BUILD_TIMEOUT = 1800  # cold image build downloads base layers + pip deps
RUN_TIMEOUT = 300

# OSPREY's dependency chain (accelerator-toolbox) ships linux/amd64 wheels
# only; on arm64 hosts (Apple Silicon) a native build would need a compiler
# the slim base lacks. Build/run amd64 under emulation instead — it matches
# the real deployment targets.
_PLATFORM_ARGS = (
    ["--platform", "linux/amd64"] if platform.machine().lower() in ("arm64", "aarch64") else []
)


def _docker_run(tag: str, *cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "run", "--rm", *_PLATFORM_ARGS, tag, *cmd],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT,
    )


def test_generated_dockerfile_builds_and_boots(tmp_path):
    project_name = "dockere2e"
    result = CliRunner().invoke(
        cli,
        [
            "build",
            project_name,
            "--preset",
            "hello-world",
            "--skip-deps",
            "--skip-lifecycle",
            "-o",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    project = tmp_path / project_name
    assert (project / "Dockerfile").exists()

    tag = f"osprey-dockerfile-e2e:{uuid.uuid4().hex[:8]}"
    build_cmd = ["docker", "build", *_PLATFORM_ARGS, "-t", tag]
    pip_spec = os.environ.get("OSPREY_E2E_PIP_SPEC")
    if pip_spec:
        build_cmd += ["--build-arg", f"OSPREY_PIP_SPEC={pip_spec}"]
    build_cmd.append(".")

    try:
        build = subprocess.run(
            build_cmd, cwd=project, capture_output=True, text=True, timeout=BUILD_TIMEOUT
        )
        assert build.returncode == 0, (
            f"docker build failed:\n--- stdout ---\n{build.stdout[-4000:]}"
            f"\n--- stderr ---\n{build.stderr[-4000:]}"
        )

        # OSPREY installed and importable
        version = _docker_run(tag, "osprey", "--version")
        assert version.returncode == 0, version.stderr

        # Non-root runtime user
        whoami = _docker_run(tag, "whoami")
        assert whoami.stdout.strip() == "osprey"

        # Claude Code callable as the non-root user (permission-chain canary)
        claude = _docker_run(tag, "claude", "--version")
        assert claude.returncode == 0, (
            f"claude --version failed as non-root user — the /root/.local "
            f"traversal chain is broken:\n{claude.stderr}"
        )

        # regen --runtime-root healed the recorded host paths
        config = _docker_run(tag, "cat", f"/app/{project_name}/config.yml")
        assert config.returncode == 0, config.stderr
        assert f"project_root: /app/{project_name}" in config.stdout
        assert str(tmp_path) not in config.stdout, "host build path leaked into the image config"

        # .dockerignore did its job: no secrets/host state in the image
        env_check = _docker_run(tag, "sh", "-c", f"test ! -e /app/{project_name}/.env && echo OK")
        assert "OK" in env_check.stdout, ".env must never enter the image"
    finally:
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
