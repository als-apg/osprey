"""Real-compose proof that renamed web containers are reconciled, not stranded.

Web-terminal containers are named on the compose project
(``<project>-web-<user>``, ``<project>-nginx``); an earlier scheme named them
on ``facility.prefix``. Compose identifies a service's container by its project
and service labels, so the first ``up`` under the new names recreates every
service still in the compose file. A terminal whose user has left the roster is
in no compose file any more, and only OSPREY's own orphan reconcile
(:func:`lifecycle.remove_orphan_terminals`) removes it — by its labels, since
its name belongs to the old scheme.

This module stands a minimal stack up under the old names, re-renders it under
the new names with one user dropped, runs ``up -d`` plus the reconcile the way
``osprey up`` does, and checks that no container carrying an old name is left.

Kept OUT of ``tests/e2e/`` for the reason ``test_env_digest_recreate_proof.py``
gives: module-level skip when docker is unavailable, exact-named teardown.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import pytest

from osprey.deployment.web_terminals import lifecycle
from osprey.deployment.web_terminals.naming import web_container_name
from tests._container_support import docker_cli_unavailable_reason

_DOCKER_UNAVAILABLE = docker_cli_unavailable_reason()

pytestmark = [
    pytest.mark.dockerbuild,
    pytest.mark.skipif(_DOCKER_UNAVAILABLE is not None, reason=_DOCKER_UNAVAILABLE or ""),
]

# Already pulled by this directory's other dockerbuild tests; `sleep` keeps the
# container running with no ports and no config of its own.
_IMAGE = "nginx:1.31-alpine"


def _render(compose_file: Path, names: dict[str, str]) -> None:
    """Write a compose file with one ``sleep`` service per ``{service: container_name}``."""
    lines = ["services:"]
    for service, container_name in names.items():
        lines += [
            f"  {service}:",
            f"    image: {_IMAGE}",
            '    command: ["sleep", "600"]',
            f"    container_name: {container_name}",
        ]
    compose_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _running_names() -> set[str]:
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def test_first_up_under_project_names_leaves_no_old_named_container(tmp_path):
    token = uuid.uuid4().hex[:8]
    project = f"ospreywf-rename-{token}"
    prefix = f"oldpfx{token}"
    compose_file = tmp_path / "docker-compose.web.yml"
    base = ["docker", "compose", "-p", project, "-f", str(compose_file)]

    old_names = {
        "nginx": f"{prefix}-nginx",
        "web-alice": f"{prefix}-web-alice",
        "web-eve": f"{prefix}-web-eve",
    }
    new_names = {
        "nginx": f"{project}-nginx",
        "web-alice": web_container_name(project, "alice"),
    }
    config = {
        "project_name": project,
        "container_runtime": "docker",
        "modules": {"web_terminals": {"enabled": True, "users": ["alice"]}},
    }

    try:
        _render(compose_file, old_names)
        subprocess.run(base + ["up", "-d"], capture_output=True, timeout=180, check=True)
        assert set(old_names.values()) <= _running_names()

        # The upgrade: eve has left the roster, every survivor is renamed. The
        # web stack's `up` never passes --remove-orphans (it shares the compose
        # project with the services stack), exactly as here.
        _render(compose_file, new_names)
        subprocess.run(base + ["up", "-d"], capture_output=True, timeout=180, check=True)
        removed = lifecycle.remove_orphan_terminals(config)

        names = _running_names()
        assert removed == {"eve": old_names["web-eve"]}
        assert not {name for name in names if name.startswith(f"{prefix}-")}, names
        assert set(new_names.values()) <= names
    finally:
        subprocess.run(base + ["down", "--timeout", "5"], capture_output=True, timeout=120)
        for name in (*old_names.values(), *new_names.values()):
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)
