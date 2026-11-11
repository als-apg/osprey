"""The image the generated reference Dockerfile builds, shared by its two e2e modules.

``test_dockerfile_e2e.py`` proves the image without a model (``dockerfile-e2e``,
every same-repo PR); ``test_dockerfile_agent_e2e.py`` drives one real agent turn
through it (``dockerfile-agent-e2e``, a model-spending lane). Both build the
image the same way, from here, so the turn is served by the image the smoke
tests inspect.

The image is built with ``--set provider=als-apg`` so the in-image
``config.yml`` resolves to the provider CI can reach; LLM credentials are
injected at ``docker run`` time (never baked into the image).

Set ``OSPREY_E2E_PIP_SPEC`` to override which OSPREY gets installed inside the
image. The image default is the PyPI release; CI pins the PR head SHA instead so
the image tests the branch under review.
"""

from __future__ import annotations

import os
import platform
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.main import cli
from osprey.port_layout import default_port
from osprey.utils.workspace import container_image_context
from tests._container_support import docker_cli_unavailable_reason_once

_DOCKER_UNAVAILABLE = docker_cli_unavailable_reason_once()

#: Each module's ``pytestmark``: both need docker and skip without it.
MODULE_MARKS = [
    pytest.mark.dockerbuild,
    pytest.mark.skipif(_DOCKER_UNAVAILABLE is not None, reason=_DOCKER_UNAVAILABLE or ""),
]

#: The port the project image serves on INSIDE the container: the ``web`` slot
#: of the layout, which ``Dockerfile.j2`` renders into its ``EXPOSE`` line. The
#: image is built from a config that never moves ``deployment.port_base``, so
#: the layout's default base is the right one to derive it at.
WEB_SLOT = default_port("web")

BUILD_TIMEOUT = 1800  # cold image build downloads base layers + pip deps
RUN_TIMEOUT = 300
PROJECT_NAME = "dockere2e"

# OSPREY's dependency chain (accelerator-toolbox) ships linux/amd64 wheels
# only; on arm64 hosts (Apple Silicon) a native build would need a compiler
# the slim base lacks. Build/run amd64 under emulation instead — it matches
# the real deployment targets.
PLATFORM_ARGS = (
    ["--platform", "linux/amd64"] if platform.machine().lower() in ("arm64", "aarch64") else []
)


def docker_run(tag: str, *cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "run", "--rm", *PLATFORM_ARGS, tag, *cmd],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT,
    )


def init_and_build_repo(out_dir: Path) -> Path:
    """Init a hello-world deployment repo pinned to als-apg, and build it.

    ``--no-git`` because nothing here reads history and the tmp tree may sit
    inside another repository; ``--skip-deps`` because the image installs its
    own dependencies from ``OSPREY_PIP_SPEC`` and the host venv would only slow
    the render down. Returns the repo root.
    """
    repo = out_dir / PROJECT_NAME
    runner = CliRunner()
    init = runner.invoke(
        cli,
        [
            "init",
            str(repo),
            "--preset",
            "hello-world",
            "--set",
            "provider=als-apg",
            "--set",
            "model=claude-haiku-5-5",
            "--no-git",
        ],
    )
    assert init.exit_code == 0, init.output
    build = runner.invoke(cli, ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert build.exit_code == 0, build.output
    return repo


def image_context(repo: Path) -> Path:
    """The directory ``docker build`` runs against for this repo's image.

    Resolved through the production helper, not spelled here: the context is
    what lands in the image, so it is what decides the container's layout, and a
    test that picked its own would prove the layout of something the deploy path
    never builds. It is a deployment repo the build rendered against the
    ``/app/<project>`` path a container sees, so its ``Dockerfile`` is one level
    down, in the render.
    """
    context = container_image_context(repo, PROJECT_NAME)
    assert (context / "build" / "Dockerfile").is_file(), f"no rendered Dockerfile under {context}"
    return context


def build_image(context, tag: str, *, progress_plain: bool = False) -> subprocess.CompletedProcess:
    """Run ``docker build`` on the generated Dockerfile; return the completed process.

    Passes the same ``com.osprey.project`` label ``osprey up`` stamps via
    ``_project_image_build_cmd``, so label assertions here cover the shape the
    real build path produces. ``DOCKER_BUILDKIT=1`` pins the BuildKit builder
    for every build so layer-cache semantics (and ``--progress=plain`` output,
    used by the cache-hit test) are uniform across docker versions.
    """
    build_cmd = [
        "docker",
        "build",
        *PLATFORM_ARGS,
        "-f",
        str(Path(context) / "build" / "Dockerfile"),
        "-t",
        tag,
        "--label",
        f"com.osprey.project={PROJECT_NAME}",
    ]
    if progress_plain:
        build_cmd += ["--progress=plain"]
    pip_spec = os.environ.get("OSPREY_E2E_PIP_SPEC")
    if pip_spec:
        build_cmd += ["--build-arg", f"OSPREY_PIP_SPEC={pip_spec}"]
    build_cmd.append(".")
    build = subprocess.run(
        build_cmd,
        cwd=context,
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT,
        env={**os.environ, "DOCKER_BUILDKIT": "1"},
    )
    assert build.returncode == 0, (
        f"docker build failed:\n--- stdout ---\n{build.stdout[-4000:]}"
        f"\n--- stderr ---\n{build.stderr[-4000:]}"
    )
    return build


def built_image(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, Path, str, Path]]:
    """Build the repo + its reference image; yield ``(tag, repo, project, out_dir)``.

    The body of each module's module-scoped ``built_image`` fixture; the image
    is removed when the module is done with it.
    """
    out_dir = tmp_path_factory.mktemp("dockerfile-e2e")
    repo = init_and_build_repo(out_dir)
    tag = f"osprey-dockerfile-e2e:{uuid.uuid4().hex[:8]}"
    try:
        build_image(image_context(repo), tag)
        yield tag, repo, PROJECT_NAME, out_dir
    finally:
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True, timeout=120)
