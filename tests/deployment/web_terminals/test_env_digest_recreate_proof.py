"""Real-compose proof of the `.env.auth` digest-label recreate mechanism.

The auth sidecar reads its credentials from ``env_file: .env.auth``, whose
content compose bakes into the container at CREATION time. A content-only edit
changes no service definition, so podman-compose leaves the old container
running — the deployed fix is to render a sha256 digest of the file into the
service definition as a label (``render.AUTH_ENV_DIGEST_LABEL``), turning the
edit into a definition change. This module proves the mechanism against a real
``docker compose``: same minimal service shape (``env_file`` + digest label),
digest computed by the same :func:`auth_env_digest` helper the deploy path
uses, and the re-render + re-``up`` cycle a redeploy performs. Because docker
compose (unlike podman-compose) also hashes ``env_file`` content on its own,
the end-to-end edit phase alone cannot attribute its recreate to the label —
the dedicated label-only phase is what pins the label as a sufficient trigger.

Kept OUT of ``tests/e2e/`` deliberately: files there carrying ``dockerbuild``
need their own CI job plus an ``--ignore`` in the shared e2e lane (see
``test_ci_workflow_wiring.py``). The precedent for real-docker tests in the
unit lane is ``test_nginx_validate.py``/``test_auth_serving.py`` in this same
directory — module-level skip when docker is unavailable, exact-named teardown.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from osprey.deployment.web_terminals.artifacts import auth_env_digest
from osprey.deployment.web_terminals.auth_credentials import AUTH_ENV_FILENAME
from osprey.deployment.web_terminals.render import AUTH_ENV_DIGEST_LABEL


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = [
    pytest.mark.dockerbuild,
    pytest.mark.skipif(not _docker_available(), reason="docker not available"),
]

# Already pulled by this directory's other dockerbuild tests, so no extra image
# cost; `sleep` exists in the alpine base and keeps the container running with
# no ports and no config of its own.
_IMAGE = "nginx:1.27-alpine"

_ENV_VAR = "OSPREY_AUTH_SESSION_SECRET"  # variable NAME only; test values are dummies


def _render(project_dir: Path, digest_override: str | None = None) -> None:
    """The redeploy's render step, minimally: digest `.env.auth`, stamp the label.

    ``digest_override`` substitutes a hand-chosen label value for the real
    digest — the label-only phase's tool for changing the service definition
    while leaving ``.env.auth`` (and everything else) byte-identical.
    """
    digest = digest_override if digest_override is not None else auth_env_digest(project_dir)
    (project_dir / "docker-compose.yml").write_text(
        "services:\n"
        "  auth:\n"
        f"    image: {_IMAGE}\n"
        '    command: ["sleep", "600"]\n'
        f"    env_file: {AUTH_ENV_FILENAME}\n"
        "    labels:\n"
        f'      {AUTH_ENV_DIGEST_LABEL}: "{digest}"\n',
        encoding="utf-8",
    )


def _container_id(base: list[str]) -> str:
    result = subprocess.run(
        base + ["ps", "-q", "auth"], capture_output=True, text=True, timeout=60, check=True
    )
    container_id = result.stdout.strip()
    assert container_id, "auth container not running"
    return container_id


def _baked_env_value(container_id: str) -> str:
    """The env_file value compose baked in at creation, straight from inspect."""
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{range .Config.Env}}{{println .}}{{end}}", container_id],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith(f"{_ENV_VAR}="):
            return line.split("=", 1)[1]
    raise AssertionError(f"{_ENV_VAR} not present in container env")


def test_env_auth_edit_recreates_the_service_through_the_digest_label(tmp_path):
    """Four real `up -d` runs: create, re-up after a hand-edit (must recreate
    and bake the NEW value), re-up with nothing changed (must NOT recreate),
    re-up after a LABEL-ONLY change (must recreate).

    The last phase is what earns "through the digest label" in this test's
    name: docker compose hashes env_file content into its own config hash, so
    on docker the hand-edit phase would recreate even with no label at all —
    podman-compose (the implementation the label exists for) is the one that
    does not. Changing only the label, with `.env.auth` untouched, isolates a
    label change as a sufficient recreate trigger on its own."""
    project = f"ospreywf-envdigest-{uuid.uuid4().hex[:8]}"
    env_auth = tmp_path / AUTH_ENV_FILENAME
    base = ["docker", "compose", "-p", project, "-f", str(tmp_path / "docker-compose.yml")]

    def _up() -> None:
        subprocess.run(base + ["up", "-d"], capture_output=True, timeout=120, check=True)

    try:
        env_auth.write_text(f"{_ENV_VAR}=first-value\n", encoding="utf-8")
        _render(tmp_path)
        _up()
        first_id = _container_id(base)
        assert _baked_env_value(first_id) == "first-value"

        # The documented operator workflow: edit the file by hand, redeploy.
        env_auth.write_text(f"{_ENV_VAR}=second-value\n", encoding="utf-8")
        _render(tmp_path)
        _up()
        second_id = _container_id(base)
        assert second_id != first_id, "label change did not recreate the service"
        assert _baked_env_value(second_id) == "second-value"

        # Negative control: an unchanged redeploy must leave the container
        # alone — this is what proves the recreates in this test are
        # definition-driven, not unconditional.
        _render(tmp_path)
        _up()
        assert _container_id(base) == second_id

        # Label-only phase: same `.env.auth` bytes, hand-altered label value.
        # This is the recreate docker's own env_file hashing CANNOT explain,
        # so it attributes the trigger to the label itself — the property the
        # deploy path relies on for podman-compose.
        _render(tmp_path, digest_override="0" * 64)
        _up()
        third_id = _container_id(base)
        assert third_id != second_id, "a label-only definition change did not recreate"
        # The env was untouched, so the recreated container bakes the same
        # value — the bounce came from the label, not from a file change.
        assert _baked_env_value(third_id) == "second-value"
    finally:
        subprocess.run(base + ["down", "--timeout", "5"], capture_output=True, timeout=120)
