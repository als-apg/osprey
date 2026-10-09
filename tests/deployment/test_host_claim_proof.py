"""Real-runtime proof that the ownership check reads what compose really stamps.

:func:`container_ownership.host_claim` decides whether this checkout may act on
the containers that carry its compose project name, by the ``com.osprey.repo-id``
label OSPREY bakes into every service and the working-directory label compose
itself stamps. The unit tests feed it hand-written rows; this module stands a
project up with real compose and checks that the probe reads the labels docker
actually writes — the partition, the recorded path, and the moved-checkout
case where that path is gone.

Kept OUT of ``tests/e2e/`` for the reason ``test_env_digest_recreate_proof.py``
gives: module-level skip when docker is unavailable, exact-named teardown.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import pytest

from osprey.deployment.compose_generator import REPO_ID_LABEL
from osprey.deployment.container_ownership import host_claim
from osprey.deployment.reset import RuntimeProbe
from tests._container_support import docker_cli_unavailable_reason

_DOCKER_UNAVAILABLE = docker_cli_unavailable_reason()

pytestmark = [
    pytest.mark.dockerbuild,
    pytest.mark.skipif(_DOCKER_UNAVAILABLE is not None, reason=_DOCKER_UNAVAILABLE or ""),
]

# Already pulled by this directory's other dockerbuild tests; `sleep` keeps the
# container running with no ports and no config of its own.
_IMAGE = "nginx:1.31-alpine"

_THEIRS = "0123456789ab"


def _compose_up(workdir: Path, project: str, repo_id: str) -> list[str]:
    """Bring one labelled service up from *workdir*; return the base command."""
    compose_file = workdir / "docker-compose.yml"
    compose_file.write_text(
        "services:\n"
        "  keep:\n"
        f"    image: {_IMAGE}\n"
        '    command: ["sleep", "600"]\n'
        '    volumes: ["data:/data"]\n'
        "    labels:\n"
        f'      {REPO_ID_LABEL}: "{repo_id}"\n'
        "volumes:\n"
        "  data:\n",
        encoding="utf-8",
    )
    base = ["docker", "compose", "-p", project, "-f", str(compose_file)]
    subprocess.run([*base, "up", "-d"], check=True, capture_output=True, text=True, timeout=300)
    return base


def test_the_probe_partitions_real_containers_by_the_baked_label(tmp_path):
    token = uuid.uuid4().hex[:8]
    project = f"ospreyclaim-{token}"
    workdir = tmp_path / "their-checkout"
    workdir.mkdir()
    base = _compose_up(workdir, project, _THEIRS)
    try:
        probe = RuntimeProbe("docker")

        theirs = host_claim(project, "a1b2c3d4e5f6", probe=probe)
        assert [r.name for r in theirs.foreign] == [f"{project}-keep-1"]
        assert not theirs.containers.ours and not theirs.containers.unidentified
        # Compose's own working-directory label is the recorded path.
        assert theirs.foreign[0].recorded_path == str(workdir)
        assert theirs.held_elsewhere and not theirs.other_copy_gone
        # The volume is listed for the message and never partitioned.
        assert [v.name for v in theirs.volumes] == [f"{project}_data"]

        mine = host_claim(project, _THEIRS, probe=probe)
        assert not mine.foreign and [r.name for r in mine.containers.ours] == [f"{project}-keep-1"]
    finally:
        subprocess.run([*base, "down", "-v"], capture_output=True, text=True, timeout=300)


def test_a_moved_checkout_is_the_other_copy_gone_case(tmp_path):
    token = uuid.uuid4().hex[:8]
    project = f"ospreyclaim-{token}"
    workdir = tmp_path / "moved-away"
    workdir.mkdir()
    base = _compose_up(workdir, project, _THEIRS)
    try:
        # The compose file stays readable for teardown; the recorded directory
        # is what moves. Rename it so the path compose stamped no longer exists.
        compose_text = (workdir / "docker-compose.yml").read_text(encoding="utf-8")
        workdir.rename(tmp_path / "elsewhere")
        (tmp_path / "docker-compose.yml").write_text(compose_text, encoding="utf-8")
        base = ["docker", "compose", "-p", project, "-f", str(tmp_path / "docker-compose.yml")]

        claim = host_claim(project, "a1b2c3d4e5f6", probe=RuntimeProbe("docker"))
        assert claim.held_elsewhere
        assert claim.other_copy_gone
    finally:
        subprocess.run([*base, "down", "-v"], capture_output=True, text=True, timeout=300)
