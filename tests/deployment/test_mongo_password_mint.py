"""Unit tests for the archiver Mongo root password in the token-mint map.

Mirrors ``test_postgres_password_mint.py``'s coverage for the ``mongodb``
deployed-service entry in ``_SERVICE_TOKEN_VARS`` (container_lifecycle.py):
``osprey up`` mints a strong ``MONGO_ROOT_PASSWORD`` into the project
``.env``, which the mongodb compose template reads as
MONGO_INITDB_ROOT_PASSWORD and the agent's archiver connector block names
(``archiver.mongodb_archiver.password_env``) rather than carrying a value.

What makes this store different from the ARIEL one is the *number* of readers
the single value has to satisfy: the container that initializes the volume, the
archiver_recorder writing samples in-network, and the connector reading them
back. The registry entries are what make those three the same value, so the
tests assert the invariant end to end — a mint that reaches the ``.env``, and a
``.env`` value that every registry agrees on.
"""

from __future__ import annotations

import subprocess

import pytest

from osprey.deployment import container_lifecycle
from osprey.deployment.host_ports import _remedy_for_service
from osprey.deployment.service_tokens import _generate_token, _validate_var


@pytest.fixture
def captured_argv(monkeypatch, tmp_path):
    """Patch deploy_up's collaborators for a project with only 'mongodb' deployed."""
    captured: dict = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: ({"deployed_services": ["mongodb"]}, ["docker-compose.yml"]),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )

    def _fake_run(cmd, env=None, check=False, **kwargs):
        captured["cmd"] = cmd
        # run_captured re-wraps this result to carry its spool path, so the
        # stand-in has to be a real completed process, and it passes redirection
        # kwargs this ignores.
        return subprocess.CompletedProcess(list(cmd), 0)

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)
    return captured


@pytest.fixture
def _clean_token_env(monkeypatch):
    monkeypatch.delenv("MONGO_ROOT_PASSWORD", raising=False)


def _parse_env(tmp_path):
    from osprey.utils.dotenv import parse_dotenv_file

    path = tmp_path / ".env"
    return parse_dotenv_file(path) if path.is_file() else {}


def test_mongodb_deploy_mints_mongo_root_password(captured_argv, _clean_token_env, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=False)

    env = _parse_env(tmp_path)
    assert env.get("MONGO_ROOT_PASSWORD")
    # token_hex(32) → 64 hex chars, 256 bits of entropy.
    assert len(env["MONGO_ROOT_PASSWORD"]) == 64
    # A mongodb-only deploy must not mint unrelated service tokens.
    assert "ARIEL_DB_PASSWORD" not in env
    assert "ZO_ROOT_USER_PASSWORD" not in env


def test_mongo_root_password_generator_is_uri_safe_every_time():
    """Every mint must survive being substituted into a mongodb:// URI.

    The connector passes the password to ``MongoClient`` as a keyword argument,
    but the recorder, the seeder, and any operator reaching the store by hand
    may spell the same credential as a URI, where an unescaped reserved
    character silently reshapes the authority component instead of erroring.
    The hex alphabet is what makes all of those spellings equivalent.
    """
    for _ in range(50):
        value = _generate_token("MONGO_ROOT_PASSWORD")
        assert value.isalnum()
        assert _validate_var("MONGO_ROOT_PASSWORD", value)


def test_mongodb_mint_is_idempotent(captured_argv, _clean_token_env, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    first = _parse_env(tmp_path)
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    second = _parse_env(tmp_path)

    assert first["MONGO_ROOT_PASSWORD"] == second["MONGO_ROOT_PASSWORD"]
    assert (tmp_path / ".env").read_text().count("MONGO_ROOT_PASSWORD=") == 1


def test_existing_mongo_root_password_is_preserved(captured_argv, _clean_token_env, tmp_path):
    """The volume that adopted the value outlives any later mint."""
    (tmp_path / ".env").write_text("MONGO_ROOT_PASSWORD=preexistingvalue\n", encoding="utf-8")
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    assert _parse_env(tmp_path)["MONGO_ROOT_PASSWORD"] == "preexistingvalue"


@pytest.mark.parametrize("bad", ["p@ssword", "pass word", "a/b", "user:pw", "q?x", "frag#ment"])
def test_operator_password_with_reserved_char_is_rejected(
    captured_argv, _clean_token_env, tmp_path, bad
):
    """Refuse at the deploy boundary rather than let a URI reader mis-parse it."""
    (tmp_path / ".env").write_text(f"MONGO_ROOT_PASSWORD={bad}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="MONGO_ROOT_PASSWORD"):
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)


def test_rejection_never_echoes_the_password(captured_argv, _clean_token_env, tmp_path):
    (tmp_path / ".env").write_text("MONGO_ROOT_PASSWORD=p@ssword\n", encoding="utf-8")
    with pytest.raises(RuntimeError) as excinfo:
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    assert "p@ssword" not in str(excinfo.value)


class TestRegistryAgreement:
    """The registries that make one minted value reach all three readers."""

    def test_the_token_var_matches_what_the_compose_template_reads(self):
        """The template's ``${MONGO_ROOT_PASSWORD:-...}`` and the mint map are one contract."""
        from osprey.deployment.container_lifecycle import _SERVICE_TOKEN_VARS

        template = _mongodb_template_text()
        (var,) = _SERVICE_TOKEN_VARS["mongodb"]
        assert f"${{{var}:-" in template

    def test_the_minted_var_is_registered_as_volume_initialized(self):
        """A missing entry makes the fresh-mint warning silently never fire.

        The store adopts ``MONGO_ROOT_PASSWORD`` only when it initializes a
        fresh data volume, so a newly minted value beside a surviving volume is
        a credential mismatch nothing else names — the registry entry is what
        makes ``_ensure_service_tokens`` warn at the mint. The template
        declaring a named volume is the fact that earns the entry its place.
        """
        from osprey.deployment.container_lifecycle import _VOLUME_INITIALIZED_VARS

        assert _VOLUME_INITIALIZED_VARS.get("MONGO_ROOT_PASSWORD") == "mongodb"
        declared = _declared_top_level_volumes(_mongodb_template_text())
        assert "archiver_mongodb_data" in declared, (
            "the volume the store initializes /data/db on must be a declared "
            "named volume, or the fresh-mint warning guards nothing"
        )

    def test_the_port_remedy_names_the_knob_the_profile_block_emits(self):
        """The generic fallback would be ``services.mongodb.port``, which is wrong."""
        assert _remedy_for_service("mongodb") == "services.mongodb.port_host"


def _mongodb_template_text() -> str:
    """The mongodb compose template as written, jinja expressions intact."""
    from pathlib import Path

    import osprey

    path = (
        Path(osprey.__file__).parent
        / "templates"
        / "services"
        / "mongodb"
        / "docker-compose.yml.j2"
    )
    return path.read_text(encoding="utf-8")


def _declared_top_level_volumes(template: str) -> set[str]:
    """Bare volume names from the template's top-level ``volumes:`` mapping.

    Scanned textually rather than with ``yaml.safe_load``: the service body
    carries jinja expressions (``${VAR:-{{ … }}}``) that are not valid YAML, so
    parsing the file whole would fail for reasons unrelated to what is asserted.
    The top-level block is a plain list of keys, which reads unambiguously from
    indentation alone.
    """
    names: set[str] = set()
    in_block = False
    for line in template.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[:1].isspace():
            in_block = line.rstrip() == "volumes:"
            continue
        if in_block:
            names.add(line.strip().rstrip(":"))
    return names
