"""Unit tests for the ARIEL Postgres password in the token-mint map.

Mirrors ``test_bluesky_token_mint.py``'s coverage for the ``postgresql``
deployed-service entry in ``_SERVICE_TOKEN_VARS`` (container_lifecycle.py):
``osprey up`` mints a strong ``ARIEL_DB_PASSWORD`` into the project
``.env``, which the postgresql compose template reads as POSTGRES_PASSWORD and
the ariel DSN (``ariel.database.uri``) references as
``${ARIEL_DB_PASSWORD:-ariel}`` — a single source of truth replacing the old
shared ``ariel``/``ariel`` default on fresh volumes.
"""

from __future__ import annotations

import subprocess

import pytest

from osprey.deployment import container_lifecycle
from osprey.deployment.service_tokens import _generate_token, _validate_var


@pytest.fixture
def captured_argv(monkeypatch, tmp_path):
    """Patch deploy_up's collaborators for a project with only 'postgresql' deployed."""
    captured: dict = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: ({"deployed_services": ["postgresql"]}, ["docker-compose.yml"]),
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
    monkeypatch.delenv("ARIEL_DB_PASSWORD", raising=False)
    monkeypatch.delenv("ARIEL_DSN", raising=False)


def _parse_env(tmp_path):
    from osprey.utils.dotenv import parse_dotenv_file

    path = tmp_path / ".env"
    return parse_dotenv_file(path) if path.is_file() else {}


def test_postgresql_deploy_mints_ariel_db_password(captured_argv, _clean_token_env, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=False)

    env = _parse_env(tmp_path)
    assert env.get("ARIEL_DB_PASSWORD")
    # token_hex(32) → 64 hex chars, 256 bits of entropy.
    assert len(env["ARIEL_DB_PASSWORD"]) == 64
    # A postgres-only deploy must not mint unrelated service tokens.
    assert "ZO_ROOT_USER_PASSWORD" not in env
    assert "BLUESKY_LAUNCH_TOKEN" not in env


@pytest.mark.parametrize("execution_method", ["local", "subprocess"])
def test_ariel_db_password_mints_under_writes_enabled_and_subprocess_execution(
    captured_argv, _clean_token_env, monkeypatch, tmp_path, execution_method
):
    """ARIEL_DB_PASSWORD mints with writes enabled and Python running as a subprocess.

    The ``captured_argv`` fixture's config carries no ``control_system`` or
    ``execution`` section, so on its own it never exercises this combination —
    the one a since-deleted deploy-time guard used to withhold tokens under
    (``writes_enabled: true`` plus the ``local`` spelling of the subprocess
    backend). Minting is unconditional for every var a deployed service
    declares, and this password in particular is not an arming credential at
    all: it is the Postgres superuser secret the ARIEL store initializes with,
    and withholding it leaves the store on the shared ``ariel``/``ariel``
    default. Assert it for ARIEL_DB_PASSWORD specifically rather than trusting a
    bluesky-scoped test to generalize across the token map.

    Both spellings are covered: ``local`` is the legacy name for the same
    subprocess backend (see ``osprey.utils.config.resolve_execution_method``)
    and is the value the deleted guard keyed on.
    """
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (
            {
                "deployed_services": ["postgresql"],
                "control_system": {"writes_enabled": True},
                "execution": {"execution_method": execution_method},
            },
            ["docker-compose.yml"],
        ),
    )

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=False)

    env = _parse_env(tmp_path)
    assert env.get("ARIEL_DB_PASSWORD")
    assert len(env["ARIEL_DB_PASSWORD"]) == 64
    assert _validate_var("ARIEL_DB_PASSWORD", env["ARIEL_DB_PASSWORD"])


def test_ariel_db_password_generator_is_uri_safe_every_time():
    """The value lands unescaped in the DSN's password slot, so every mint
    must be free of URI-reserved characters — guaranteed by the hex alphabet."""
    for _ in range(50):
        value = _generate_token("ARIEL_DB_PASSWORD")
        assert value.isalnum()
        assert _validate_var("ARIEL_DB_PASSWORD", value)


def test_postgresql_mint_is_idempotent(captured_argv, _clean_token_env, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    first = _parse_env(tmp_path)
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    second = _parse_env(tmp_path)

    assert first["ARIEL_DB_PASSWORD"] == second["ARIEL_DB_PASSWORD"]
    assert (tmp_path / ".env").read_text().count("ARIEL_DB_PASSWORD=") == 1


def test_existing_ariel_db_password_is_preserved(captured_argv, _clean_token_env, tmp_path):
    (tmp_path / ".env").write_text("ARIEL_DB_PASSWORD=preexistingvalue\n", encoding="utf-8")
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    assert _parse_env(tmp_path)["ARIEL_DB_PASSWORD"] == "preexistingvalue"


def test_operator_password_with_reserved_char_is_rejected(
    captured_argv, _clean_token_env, tmp_path
):
    """An operator-supplied password containing a URI-reserved character would
    silently reshape the ariel DSN — refuse at the deploy boundary instead."""
    (tmp_path / ".env").write_text("ARIEL_DB_PASSWORD=p@ssword\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ARIEL_DB_PASSWORD"):
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
