"""The postgresql service ships a SELECT-only role for the agent's SQL tool.

The raw-SQL tool is auto-allowed and, until this role existed, ran on the same
connection ingestion writes with — the superuser the official image creates.
``BEGIN READ ONLY`` stops writes but not ``pg_read_file`` or any other
superuser-only function, and an allowlist over query text can never enumerate
them, so the privileges the connection holds are the line that has to hold.

The role is created by an init script the entrypoint runs while it initializes
a fresh data volume, mounted from the service's build context. These tests
render the packaged templates the way ``osprey up`` does and pin the two halves
that have to agree: the mount and the password passthrough in the compose
document, and what the script actually grants.
"""

from __future__ import annotations

import subprocess
from importlib import resources

import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

from osprey.port_layout import DEFAULT_PORT_BASE, layout_ports

COMPOSE_TEMPLATE = "services/postgresql/docker-compose.yml.j2"
INITDB_TEMPLATE = "services/postgresql/10-readonly-role.sh.j2"

#: Where the entrypoint looks for the script, and the build path the compose
#: document binds into it.
INITDB_TARGET = "/docker-entrypoint-initdb.d/10-readonly-role.sh"
INITDB_SOURCE = "./build/services/postgresql/10-readonly-role.sh"


def _env() -> Environment:
    templates_root = resources.files("osprey").joinpath("templates")
    return Environment(loader=FileSystemLoader(str(templates_root)), autoescape=False)


def _context(**postgresql_block: object) -> dict:
    return {
        "services": {"postgresql": postgresql_block},
        "deployment": {},
        "system": {"timezone": "UTC"},
        "osprey_labels": {
            "project_name": "probe-test",
            "repo_id": "0123456789ab",
            "project_root": "/r/probe-test",
        },
        "osprey_ports": layout_ports(DEFAULT_PORT_BASE),
    }


def _render_compose(**postgresql_block: object) -> dict:
    rendered = _env().get_template(COMPOSE_TEMPLATE).render(**_context(**postgresql_block))
    return yaml.safe_load(rendered)


def _render_initdb(**postgresql_block: object) -> str:
    """The init script as the build renders it beside the compose document."""
    return _env().get_template(INITDB_TEMPLATE).render(**_context(**postgresql_block))


def test_compose_mounts_the_init_script_the_entrypoint_runs():
    """The rendered script is bound read-only where initdb looks for it."""
    volumes = _render_compose()["services"]["postgresql"]["volumes"]
    assert f"{INITDB_SOURCE}:{INITDB_TARGET}:ro" in volumes


def test_compose_passes_the_readonly_password_into_the_container():
    """The script reads the secret from the environment, so it has to arrive."""
    environment = _render_compose()["services"]["postgresql"]["environment"]
    assert environment["ARIEL_DB_READONLY_PASSWORD"] == "${ARIEL_DB_READONLY_PASSWORD:-ariel_ro}"


def test_default_render_names_the_ariel_ro_role():
    script = _render_initdb()
    assert 'ro_role="ariel_ro"' in script
    assert 'owner="ariel"' in script


@pytest.mark.parametrize("username", ["logbook", "erf_ariel"])
def test_role_name_is_derived_from_a_custom_username(username: str):
    """A facility that renames the Postgres user gets a matching ``_ro`` role.

    The suffix is derived from ``services.postgresql.username``, the one place
    the owner is declared, so nothing has to be renamed in a second file.
    """
    script = _render_initdb(username=username, database_name=f"{username}_db")
    assert f'ro_role="{username}_ro"' in script
    assert f'owner="{username}"' in script
    assert f'db_name="{username}_db"' in script


def test_script_reads_the_password_from_the_environment():
    """Rendered at build time, so the secret cannot be baked in: the script
    reads it from the variable the compose service passes through, with the
    same fallback the compose document carries."""
    script = _render_initdb()
    assert "${ARIEL_DB_READONLY_PASSWORD:-ariel_ro}" in script


def test_role_is_created_without_any_superuser_privilege():
    script = _render_initdb()
    assert "NOSUPERUSER" in script
    assert "NOCREATEDB" in script
    assert "NOCREATEROLE" in script
    assert "NOINHERIT" in script


def test_the_only_grant_is_select():
    """Read the logbook, nothing else — including tables created later.

    ``ON ALL TABLES`` covers what exists when the script runs and ``ALTER
    DEFAULT PRIVILEGES`` covers what the owner creates afterwards, which is how
    a ``text_embeddings_*`` table an enhancement module adds becomes readable
    without a second grant.
    """
    script = _render_initdb()
    assert "GRANT SELECT ON ALL TABLES IN SCHEMA public" in script
    assert "ALTER DEFAULT PRIVILEGES FOR ROLE" in script
    assert "GRANT SELECT ON TABLES" in script

    granted = [line for line in script.splitlines() if line.strip().startswith("GRANT ")]
    for line in granted:
        assert "SELECT" in line or "CONNECT" in line or "USAGE" in line, line
    assert not any(
        word in script for word in ("GRANT INSERT", "GRANT UPDATE", "GRANT DELETE", "GRANT ALL")
    )


# ---------------------------------------------------------------------------
# The init script is SOURCED, not executed
#
# The official entrypoint executes a file under /docker-entrypoint-initdb.d/
# only when it carries an executable bit, and sources it otherwise. The build
# writes the rendered script without one and the bind mount is read-only, so
# the sourced path is the one that runs — which makes every `set` in the file
# a change to the entrypoint's own shell unless it is scoped.
# ---------------------------------------------------------------------------


def _write_script(tmp_path, **postgresql_block: object):
    script = tmp_path / "10-readonly-role.sh"
    script.write_text(_render_initdb(**postgresql_block), encoding="utf-8")
    return script


def test_the_rendered_script_is_valid_shell(tmp_path):
    script = _write_script(tmp_path)
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0


def test_sourcing_it_leaves_the_callers_shell_options_alone(tmp_path):
    """`set -u` under an entrypoint that does not run with it would abort the
    initialization on the first unset variable the entrypoint reads next."""
    script = _write_script(tmp_path)
    # The entrypoint's own options, and a psql that succeeds without a server.
    (tmp_path / "psql").write_text("#!/usr/bin/env bash\ncat >/dev/null\n", encoding="utf-8")
    (tmp_path / "psql").chmod(0o755)

    probe = subprocess.run(
        ["bash", "-c", f'set -Eeo pipefail; . "{script}"; echo "OPTIONS:$-"'],
        env={
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "POSTGRES_USER": "ariel",
            "POSTGRES_DB": "ariel",
        },
        capture_output=True,
        text=True,
    )

    assert probe.returncode == 0, probe.stderr
    (options,) = [
        line.removeprefix("OPTIONS:")
        for line in probe.stdout.splitlines()
        if line.startswith("OPTIONS:")
    ]
    assert "u" not in options
