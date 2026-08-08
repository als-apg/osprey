"""Durable `nginx -t` validation of the rendered nginx fragment.

`test_render.py`'s and `test_auth_tls_seams.py`'s assertions are string matches
on the Jinja output — cheap and fast, but they can't catch an nginx syntax error
the strings happen to satisfy. This module closes that gap for real by actually
invoking `nginx -t` inside `nginx:1.27-alpine` (the base image the deployed stack
uses) against each render shape the auth/TLS seams can produce:

- the default render (`auth.method` unset -> "none", `tls.enabled` unset -> False):
  the seam is fully inert, matching the pre-auth plain-http posture.
- the fully enabled render (`tls.enabled: true` with a real self-signed cert/key
  pair generated fresh per test run, plus `auth.method: password`): two server
  blocks, the plain port's `return 301 https://…`, `listen 443 ssl` +
  `ssl_certificate*`, the two `map`s, a per-user `auth_request` and `internal`
  verify target, the shared named 401 handler, and the public `/auth/` prefix.
- the cleartext-auth render (`auth.method: password` with
  `allow_insecure_http`, the shape a facility behind its own TLS terminator
  deploys): the same auth surface inside a *single* server block.

Every one of these tests asserts the constructs it means to validate are
actually present in the rendered fragment before handing it to nginx — a
template that stopped emitting the auth surface would otherwise still "pass"
`nginx -t` and report a vacuous green.

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


def _config(
    tls: dict | None = None, auth: dict | None = None, users: list[str] | None = None
) -> dict:
    web_terminals: dict = {
        "enabled": True,
        "nginx_port": 9080,
        "web_base_port": _BASE_PORTS["web"],
        "artifact_base_port": _BASE_PORTS["artifact"],
        "ariel_base_port": _BASE_PORTS["ariel"],
        "lattice_base_port": _BASE_PORTS["lattice"],
        "users": ["alice", "bob"] if users is None else users,
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


def _directives(conf: str) -> str:
    """`conf` with its comment lines dropped.

    The template's prose names `auth_request`, `/verify` and `$request_uri` at
    length, so counting or absence assertions have to ignore it.
    """
    return "\n".join(line for line in conf.splitlines() if not line.lstrip().startswith("#"))


def _assert_auth_surface_present(nginx_conf: str, users: list[str]) -> None:
    """Every construct the enabled renders exist to hand nginx is really there.

    Without this, a template that stopped emitting the auth surface would still
    produce a fragment `nginx -t` accepts, and these tests would report a green
    that validated nothing about authentication.
    """
    directives = _directives(nginx_conf)
    assert directives.count("auth_request ") == len(users)
    for user in users:
        assert f"auth_request /_osprey_auth/{user};" in directives
        assert f"location = /_osprey_auth/{user} {{" in directives
        assert f"/verify?user={user};" in directives
    assert directives.count("internal;") == len(users)
    assert "map $uri $osprey_auth_next {" in directives
    assert "map $http_upgrade$http_accept $osprey_auth_wants_login_page {" in directives
    assert "error_page 401 = @osprey_auth_401;" in directives
    assert "location @osprey_auth_401 {" in directives
    assert "location /auth/ {" in directives
    assert "return 200;" not in nginx_conf


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
    """C2: the fully enabled render (tls.enabled + a real self-signed cert/key,
    auth.method: password) — two server blocks, the plain port's 301, the ssl
    listener and cert pair, the two maps, a per-user `auth_request` and
    `internal` verify target, the shared named 401 handler and the public
    `/auth/` prefix — is validated by running nginx, not string-matched.

    Several of those constructs are ones a fragment can get syntactically wrong
    in ways no string match notices: a named location referenced by `error_page`
    but never defined, a `map` at the wrong context, or an `if` block nginx
    rejects where it stands.
    """
    users = ["alice", "bob"]

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
                users=users,
                tls={
                    "enabled": True,
                    "cert": f"/etc/nginx/certs/{cert_path.name}",
                    "key": f"/etc/nginx/certs/{key_path.name}",
                },
                auth={"method": "password"},
            )
        )
        nginx_conf = artifacts["nginx/nginx.conf"]
        # Guard against a vacuous green: nginx would happily accept a fragment
        # that had quietly stopped emitting the whole gated surface.
        _assert_auth_surface_present(nginx_conf, users)
        assert "listen 443 ssl;" in nginx_conf
        assert f"ssl_certificate /etc/nginx/certs/{cert_path.name};" in nginx_conf
        assert f"ssl_certificate_key /etc/nginx/certs/{key_path.name};" in nginx_conf
        assert "return 301 https://$host$request_uri;" in nginx_conf

        (conf_dir / "default.conf").write_text(nginx_conf)

        # Act
        result = _run_nginx_t(conf_dir, certs_dir=certs_dir)

    # Assert
    assert result.returncode == 0, result.stderr


def test_cleartext_auth_render_passes_nginx_t() -> None:
    """C3: the auth-without-TLS render (`allow_insecure_http`, the shape a
    facility behind its own TLS terminator deploys) puts the same gated surface
    inside a SINGLE server block — a different nesting of the same directives,
    and one nothing else runs nginx against."""
    users = ["alice", "bob"]

    # Arrange
    artifacts = render_web_terminals(
        _config(users=users, auth={"method": "password", "allow_insecure_http": True})
    )
    nginx_conf = artifacts["nginx/nginx.conf"]
    _assert_auth_surface_present(nginx_conf, users)
    assert "absolute_redirect off;" in nginx_conf
    assert "listen 443 ssl;" not in nginx_conf

    with tempfile.TemporaryDirectory() as tmp:
        conf_dir = Path(tmp)
        (conf_dir / "default.conf").write_text(nginx_conf)

        # Act
        result = _run_nginx_t(conf_dir, certs_dir=None)

    # Assert
    assert result.returncode == 0, result.stderr
