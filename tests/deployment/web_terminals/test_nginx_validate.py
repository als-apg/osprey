"""Durable `nginx -t` validation of the rendered nginx fragment (Task 1.3).

`test_render.py`'s seam tests are string-match assertions on the Jinja output —
cheap and fast, but they can't catch an nginx syntax error the strings happen to
satisfy. This module closes that gap for real by actually invoking `nginx -t`
inside `nginx:1.27-alpine` (the base image the deployed stack uses) against both
renders relevant to the gated auth/TLS seam (Task 1.3):

- the default render (`auth.method` unset -> "none", `tls.enabled` unset -> False):
  the seam is fully inert, matching Phase 1's plain-http/no-auth posture.
- an enabled render (`tls.enabled: true` with a real self-signed cert/key pair
  generated fresh per test run, `auth.method` set to a stub value): the seam this
  task actually renders (`listen 443 ssl` + `ssl_certificate*` + `auth_request` +
  the fail-closed internal target).

This is the OC-1/M1 gap closer: "validated by running nginx, not string-matched"
needs to be true in CI, not just as a one-off manual check.

Skipped entirely when docker (or openssl) is unavailable, mirroring
`tests/e2e/test_dockerfile_e2e.py`'s `_docker_available()` pattern.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from osprey.deployment.web_terminals.render import render_web_terminals

_BASE_PORTS = {"web": 9091, "artifact": 9291, "ariel": 9391, "lattice": 9491}
_NGINX_IMAGE = "nginx:1.27-alpine"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = [
    pytest.mark.dockerbuild,
    pytest.mark.skipif(
        not (_docker_available() and shutil.which("openssl")),
        reason="docker (or openssl, for the self-signed cert fixture) not available",
    ),
]


def _config(tls: dict | None = None, auth: dict | None = None) -> dict:
    web_terminals: dict = {
        "enabled": True,
        "nginx_port": 9080,
        "web_base_port": _BASE_PORTS["web"],
        "artifact_base_port": _BASE_PORTS["artifact"],
        "ariel_base_port": _BASE_PORTS["ariel"],
        "lattice_base_port": _BASE_PORTS["lattice"],
        "users": ["alice", "bob"],
    }
    if tls is not None:
        web_terminals["tls"] = tls
    if auth is not None:
        web_terminals["auth"] = auth
    return {
        "facility": {"name": "Demo Light Source", "prefix": "dls", "timezone": "UTC"},
        "registry": {"url": "git.dls.example.org:5050/physics/production/dls-profiles"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "modules": {"web_terminals": web_terminals},
    }


def _generate_self_signed_cert(certs_dir: Path) -> tuple[Path, Path]:
    """Generate a throwaway self-signed cert/key pair for the enabled TLS render."""
    cert_path = certs_dir / "dls.crt"
    key_path = certs_dir / "dls.key"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key_path),
            "-out",
            str(cert_path),
            "-days",
            "1",
            "-subj",
            "/CN=dls-deploy.dls.example.org",
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    return cert_path, key_path


def _run_nginx_t(conf_dir: Path, certs_dir: Path | None) -> subprocess.CompletedProcess:
    mounts = [
        "-v",
        f"{conf_dir}/default.conf:/etc/nginx/conf.d/default.conf:ro",
    ]
    if certs_dir is not None:
        mounts += ["-v", f"{certs_dir}:/etc/nginx/certs:ro"]
    return subprocess.run(
        ["docker", "run", "--rm", *mounts, _NGINX_IMAGE, "nginx", "-t"],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_default_gated_off_render_passes_nginx_t() -> None:
    """C1: the default (auth none / tls off) render is not just inert by
    string-match — it's a config `nginx -t` actually accepts."""
    # Arrange
    artifacts = render_web_terminals(_config())

    with tempfile.TemporaryDirectory() as tmp:
        conf_dir = Path(tmp)
        (conf_dir / "default.conf").write_text(artifacts["nginx/nginx.conf"])

        # Act
        result = _run_nginx_t(conf_dir, certs_dir=None)

    # Assert
    assert result.returncode == 0, result.stderr


def test_enabled_tls_and_auth_render_passes_nginx_t() -> None:
    """C2: the enabled render (tls.enabled + a real self-signed cert/key,
    auth.method set to a stub) — the seam this task actually renders
    (`listen 443 ssl` + `ssl_certificate*` + `auth_request` + the fail-closed
    internal target) — is validated by running nginx, not string-matched."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conf_dir = tmp_path / "conf"
        certs_dir = tmp_path / "certs"
        conf_dir.mkdir()
        certs_dir.mkdir()

        cert_path, key_path = _generate_self_signed_cert(certs_dir)

        # Arrange — cert/key paths must match where they'll be mounted inside
        # the container (/etc/nginx/certs/...), not the host tmp path.
        artifacts = render_web_terminals(
            _config(
                tls={
                    "enabled": True,
                    "cert": f"/etc/nginx/certs/{cert_path.name}",
                    "key": f"/etc/nginx/certs/{key_path.name}",
                },
                auth={"method": "oauth2_proxy"},
            )
        )
        (conf_dir / "default.conf").write_text(artifacts["nginx/nginx.conf"])

        # Act
        result = _run_nginx_t(conf_dir, certs_dir=certs_dir)

    # Assert
    assert result.returncode == 0, result.stderr
