"""Tests for the render-and-write seam shared by bring-up and the lifecycle verbs."""

from __future__ import annotations

import hashlib

import pytest
import yaml

from osprey.deployment.web_terminals.artifacts import auth_env_digest, write_web_terminal_artifacts
from osprey.deployment.web_terminals.auth_credentials import AUTH_ENV_FILENAME
from osprey.deployment.web_terminals.render import AUTH_ENV_DIGEST_LABEL


def _config(users):
    return {
        "facility": {"prefix": "als", "name": "ALS"},
        "registry": {"url": "registry.example.org"},
        "deploy": {"fqdn": "deploy.example.org"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "nginx_port": 8080,
                "web_base_port": 9000,
                "artifact_base_port": 9100,
                "ariel_base_port": 9200,
                "lattice_base_port": 9300,
                "users": users,
            }
        },
    }


def test_write_web_terminal_artifacts_writes_three_files_under_dest(tmp_path):
    written = write_web_terminal_artifacts(_config(["alice", "bob"]), tmp_path)

    names = {p.relative_to(tmp_path).as_posix() for p in written}
    assert names == {
        "docker-compose.web.yml",
        "nginx/nginx.conf",
        "nginx/landing.html",
    }
    for path in written:
        assert path.is_file()
        assert path.read_text(encoding="utf-8")  # non-empty


def test_write_web_terminal_artifacts_creates_nginx_parent_dir(tmp_path):
    write_web_terminal_artifacts(_config(["alice"]), tmp_path)
    assert (tmp_path / "nginx").is_dir()
    assert (tmp_path / "nginx" / "nginx.conf").is_file()


def test_write_web_terminal_artifacts_defaults_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    written = write_web_terminal_artifacts(_config(["alice"]))
    assert (tmp_path / "docker-compose.web.yml").is_file()
    # Written paths are relative to the CWD default.
    assert any(p.name == "docker-compose.web.yml" for p in written)


def test_write_web_terminal_artifacts_reflects_object_form_users(tmp_path):
    """Object-form users with explicit indices render into the compose overlay."""
    write_web_terminal_artifacts(
        _config([{"name": "alice", "index": 0}, {"name": "bob", "index": 1}]), tmp_path
    )
    compose = (tmp_path / "docker-compose.web.yml").read_text(encoding="utf-8")
    assert "web-alice" in compose
    assert "web-bob" in compose


def test_write_web_terminal_artifacts_propagates_render_valueerror(tmp_path):
    """An unrenderable config (TLS enabled without cert/key) surfaces as ValueError."""
    config = _config(["alice"])
    config["modules"]["web_terminals"]["tls"] = {"enabled": True}
    with pytest.raises(ValueError):
        write_web_terminal_artifacts(config, tmp_path)


# ---------------------------------------------------------------------------
# The .env.auth digest: this seam is where file content meets the rendered
# sidecar definition, so it is where the digest is computed
# ---------------------------------------------------------------------------


def _auth_config(users):
    """The base config with authentication on (no TLS, so opt into HTTP)."""
    config = _config(users)
    config["modules"]["web_terminals"]["auth"] = {
        "method": "password",
        "allow_insecure_http": True,
    }
    return config


def _rendered_auth_service(dest) -> dict:
    return yaml.safe_load((dest / "docker-compose.web.yml").read_text(encoding="utf-8"))[
        "services"
    ]["auth"]


def test_write_stamps_the_auth_sidecar_with_the_env_auth_content_digest(tmp_path):
    """The label is the sha256 of the file's exact bytes under dest_dir — the
    same directory compose resolves `env_file: .env.auth` against, so the
    digest is a faithful stand-in for what the sidecar will actually read."""
    content = b"OSPREY_AUTH_SESSION_SECRET=abc123\n"
    (tmp_path / AUTH_ENV_FILENAME).write_bytes(content)

    write_web_terminal_artifacts(_auth_config(["alice"]), tmp_path)

    auth = _rendered_auth_service(tmp_path)
    assert auth["labels"][AUTH_ENV_DIGEST_LABEL] == hashlib.sha256(content).hexdigest()


def test_hand_edit_of_env_auth_changes_the_rendered_sidecar_definition(tmp_path):
    """THE BUG this label exists to fix: an operator hand-appends OIDC client
    credentials to `.env.auth` (the documented workflow) and redeploys. The
    mint is idempotent, so nothing else about the deploy changes — the re-render
    itself must change the sidecar's service definition, because a definition
    change is the only recreate trigger every compose implementation honours.
    An unchanged file must keep the render byte-identical (no-op redeploys
    recreate nothing)."""
    config = _auth_config(["alice"])
    env_auth = tmp_path / AUTH_ENV_FILENAME
    env_auth.write_text("OSPREY_AUTH_SESSION_SECRET=abc123\n", encoding="utf-8")

    write_web_terminal_artifacts(config, tmp_path)
    before = _rendered_auth_service(tmp_path)
    compose_before = (tmp_path / "docker-compose.web.yml").read_bytes()

    # No-op redeploy first: same file, byte-identical render.
    write_web_terminal_artifacts(config, tmp_path)
    assert (tmp_path / "docker-compose.web.yml").read_bytes() == compose_before

    # The hand-edit, exactly as documented for OIDC deployments.
    with env_auth.open("a", encoding="utf-8") as handle:
        handle.write("OSPREY_AUTH_OIDC_CLIENT_SECRET=idp-issued-secret\n")
    write_web_terminal_artifacts(config, tmp_path)
    after = _rendered_auth_service(tmp_path)

    assert before != after
    assert before["labels"][AUTH_ENV_DIGEST_LABEL] != after["labels"][AUTH_ENV_DIGEST_LABEL]


def test_missing_env_auth_digests_the_empty_string_instead_of_crashing(tmp_path):
    """A render from a root with no `.env.auth` yet (e.g. re-rendering artifacts
    outside a full deploy) must never crash — it stamps the digest of empty
    content, which the first real deploy's re-render then supersedes."""
    write_web_terminal_artifacts(_auth_config(["alice"]), tmp_path)

    auth = _rendered_auth_service(tmp_path)
    assert auth["labels"][AUTH_ENV_DIGEST_LABEL] == hashlib.sha256(b"").hexdigest()


def test_auth_env_digest_reads_the_file_under_the_given_root(tmp_path):
    """The helper the compose-level proof reuses: digest of the exact bytes,
    empty-content sentinel when the file is absent."""
    assert auth_env_digest(tmp_path) == hashlib.sha256(b"").hexdigest()

    (tmp_path / AUTH_ENV_FILENAME).write_bytes(b"A=1\n")
    assert auth_env_digest(tmp_path) == hashlib.sha256(b"A=1\n").hexdigest()
