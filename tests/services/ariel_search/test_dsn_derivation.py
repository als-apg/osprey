"""The ARIEL DSN is derived from the Postgres the project actually runs.

``ariel.database.uri`` used to be rendered into every project as a hardcoded
``postgresql://ariel:…@localhost:5432/ariel`` — a second, silent copy of facts
already declared under ``services.postgresql``.  Moving the database port there
left the DSN pointing at the old one, and nothing said so.

The templates no longer render the key at all: with ``uri`` unset the DSN is
derived from ``services.postgresql`` at load time, so a port move is a one-place
edit.  An explicit ``uri`` still wins verbatim — that is how a project points at
a database it does not run — and a legacy ``connection_string`` keeps working,
loudly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml

import osprey.templates
from osprey.services.ariel_search.config import ARIELConfig, resolve_ariel_dsn

DSN_LOGGER = "osprey.services.ariel_search.config"

TEMPLATE_ROOT = Path(osprey.templates.__file__).parent
ARIEL_TEMPLATES = {
    "control_assistant": "apps/control_assistant/config.yml.j2",
    "ariel_standalone": "apps/ariel_standalone/config.yml.j2",
}


def _render_template(relative_path: str) -> str:
    """Render a shipped app config template with a representative context."""
    from jinja2 import ChainableUndefined, Environment, FileSystemLoader

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_ROOT)),
        undefined=ChainableUndefined,
        keep_trailing_newline=True,
    )
    return env.get_template(relative_path).render(
        project_name="demo",
        facility_name="Demo Facility",
        default_provider="anthropic",
        default_model="claude-haiku-4-5-20251001",
        channel_finder_mode="in_context",
        default_pipeline="in_context",
        enable_in_context=True,
        enable_hierarchical=False,
        enable_middle_layer=False,
        channel_finder_tools=[],
        project_root="/tmp/demo",
    )


def _load_rendered(relative_path: str) -> dict[str, Any]:
    """Render a template and parse it the way the config loader would."""
    return yaml.safe_load(_render_template(relative_path))


@pytest.fixture(autouse=True)
def no_ambient_db_password(monkeypatch):
    """Pin the password default: a developer's own ARIEL_DB_PASSWORD must not decide."""
    monkeypatch.delenv("ARIEL_DB_PASSWORD", raising=False)


@pytest.fixture
def rearmed_connection_string_warning(monkeypatch):
    """Re-arm the once-per-process guard, and un-arm it again after the test.

    ``monkeypatch`` restores the flag on teardown, so a test that trips the
    warning cannot silence it for whatever runs next in the same session.
    """
    from osprey.services.ariel_search import config as config_module

    monkeypatch.setattr(config_module, "_connection_string_warned", False)


# ---------------------------------------------------------------------------
# The shipped templates
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(ARIEL_TEMPLATES))
def test_templates_render_no_ariel_database_uri(name: str) -> None:
    """No shipped template writes the DSN out — there is nothing to drift."""
    text = _render_template(ARIEL_TEMPLATES[name])
    config = _load_rendered(ARIEL_TEMPLATES[name])

    assert config["ariel"].get("database") is None, (
        f"{name} still renders an ariel.database block: {config['ariel']['database']!r} — "
        "the DSN is derived from services.postgresql, not written out"
    )
    assert "#   uri: postgresql://" in text, (
        f"{name} lost the commented external-database override example — a project "
        "pointing at a database it does not run needs to see how"
    )


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(ARIEL_TEMPLATES))
def test_templates_declare_what_the_dsn_derives_from(name: str) -> None:
    """Deriving is only honest if the source keys are actually rendered."""
    postgresql = _load_rendered(ARIEL_TEMPLATES[name])["services"]["postgresql"]

    assert postgresql["username"] == "ariel"
    assert postgresql["database_name"] == "ariel"
    assert postgresql["port_host"] == 5432


# ---------------------------------------------------------------------------
# Derivation follows the Postgres block
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(ARIEL_TEMPLATES))
def test_dsn_follows_a_post_render_port_host_edit(name: str) -> None:
    """Editing services.postgresql after render moves the DSN with it.

    This is the whole point of the change: a rendered project that reassigns
    the host port (5432 already taken, say) gets an ARIEL connection that
    follows, without a second edit to a duplicated DSN.
    """
    config = _load_rendered(ARIEL_TEMPLATES[name])

    as_rendered = ARIELConfig.from_dict(config["ariel"], config["services"]["postgresql"])
    assert as_rendered.database.uri == "postgresql://ariel:ariel@localhost:5432/ariel"

    config["services"]["postgresql"]["port_host"] = 15432

    after_edit = ARIELConfig.from_dict(config["ariel"], config["services"]["postgresql"])
    assert after_edit.database.uri == "postgresql://ariel:ariel@localhost:15432/ariel"


@pytest.mark.unit
def test_dsn_follows_username_and_database_name() -> None:
    """Every derived field comes from services.postgresql, not just the port."""
    uri = resolve_ariel_dsn(
        {"search_modules": {"keyword": {"enabled": True}}},
        {"username": "logbook", "database_name": "olog", "port_host": 6543},
    )

    assert uri == "postgresql://logbook:ariel@localhost:6543/olog"


@pytest.mark.unit
def test_password_reads_the_env_the_compose_service_reads(monkeypatch) -> None:
    """The DSN password is the same ARIEL_DB_PASSWORD Postgres was started with."""
    monkeypatch.setenv("ARIEL_DB_PASSWORD", "password-from-dotenv")

    uri = resolve_ariel_dsn({}, {"username": "ariel", "database_name": "ariel"})

    assert uri == "postgresql://ariel:password-from-dotenv@localhost:5432/ariel"


@pytest.mark.unit
def test_missing_services_block_falls_back_to_shipped_defaults(monkeypatch) -> None:
    """A config with no services block still yields a connectable local DSN."""
    monkeypatch.delenv("ARIEL_DB_PASSWORD", raising=False)

    assert resolve_ariel_dsn({}, None) == "postgresql://ariel:ariel@localhost:5432/ariel"
    assert (
        resolve_ariel_dsn({"database": {}}, {}) == "postgresql://ariel:ariel@localhost:5432/ariel"
    )


# ---------------------------------------------------------------------------
# Explicit uri wins
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_explicit_uri_wins_untouched() -> None:
    """An explicit DSN is passed through verbatim, services block or not."""
    external = "postgresql://reader:pw@logbook-db.example.org:5433/facility_olog"

    config = ARIELConfig.from_dict(
        {"database": {"uri": external}},
        {"username": "ariel", "database_name": "ariel", "port_host": 15432},
    )

    assert config.database.uri == external


@pytest.mark.unit
def test_explicit_uri_ignores_a_port_host_edit() -> None:
    """A project that wrote its own DSN keeps it when the local Postgres moves."""
    ariel_section = {"database": {"uri": "postgresql://ariel:pw@localhost:5432/ariel"}}
    services = {"username": "ariel", "database_name": "ariel", "port_host": 5432}

    services["port_host"] = 15432

    assert resolve_ariel_dsn(ariel_section, services) == (
        "postgresql://ariel:pw@localhost:5432/ariel"
    )


# ---------------------------------------------------------------------------
# The web interface: derive first, then rewrite the host for the container
# ---------------------------------------------------------------------------


def _write_project_config(tmp_path: Path, ariel: dict[str, Any], port_host: int) -> Path:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "ariel": ariel,
                "services": {
                    "postgresql": {
                        "username": "ariel",
                        "database_name": "ariel",
                        "port_host": port_host,
                    }
                },
            },
            sort_keys=False,
        )
    )
    return config_path


@pytest.mark.unit
def test_web_interface_derives_the_dsn(tmp_path, monkeypatch) -> None:
    """The web app reads the same derived DSN as every other entry point."""
    from osprey.interfaces.ariel.app import load_ariel_config

    monkeypatch.delenv("ARIEL_DATABASE_HOST", raising=False)
    config_path = _write_project_config(tmp_path, {"search_modules": {}}, port_host=15432)

    ariel_config = load_ariel_config(config_path)

    assert ariel_config["database"]["uri"] == "postgresql://ariel:ariel@localhost:15432/ariel"


@pytest.mark.unit
def test_container_host_override_still_reaches_a_derived_dsn(tmp_path, monkeypatch) -> None:
    """ARIEL_DATABASE_HOST rewrites the derived host, not just a written-out one.

    The web interface runs in a container, where `localhost` is the container
    itself.  The override that repoints it at the Postgres service has to see a
    resolved DSN — so the derivation happens first, and only then the rewrite.
    """
    from osprey.interfaces.ariel.app import load_ariel_config

    monkeypatch.setenv("ARIEL_DATABASE_HOST", "postgresql-service")
    config_path = _write_project_config(tmp_path, {"search_modules": {}}, port_host=15432)

    ariel_config = load_ariel_config(config_path)

    assert ariel_config["database"]["uri"] == (
        "postgresql://ariel:ariel@postgresql-service:15432/ariel"
    )


# ---------------------------------------------------------------------------
# Legacy connection_string — honored, and said out loud once
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_legacy_connection_string_is_honored_as_the_uri(
    caplog, rearmed_connection_string_warning
) -> None:
    """The retired alias still reaches the database it names.

    The shim that copied ``connection_string`` into ``uri`` lived in the MCP
    server alone, so only that one entry point honored it.  Deriving in the
    shared path could have silently redirected such a project at a *different*
    database — the legacy value wins over the derived DSN everywhere instead.
    """
    legacy = "postgresql://ariel:pw@logbook-db.example.org:5432/ariel"

    with caplog.at_level(logging.WARNING, logger=DSN_LOGGER):
        config = ARIELConfig.from_dict(
            {"database": {"connection_string": legacy}},
            {"username": "other", "database_name": "other", "port_host": 15432},
        )

    assert config.database.uri == legacy

    warnings_emitted = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings_emitted) == 1, warnings_emitted
    assert "ariel.database.connection_string" in warnings_emitted[0]
    assert "ariel.database.uri" in warnings_emitted[0], (
        f"the warning must name the key that replaced the alias, got: {warnings_emitted[0]}"
    )


@pytest.mark.unit
def test_legacy_connection_string_warns_once_per_process(
    caplog, rearmed_connection_string_warning
) -> None:
    """A CLI that parses the config repeatedly says it once, not once per parse."""
    ariel_section = {"database": {"connection_string": "postgresql://ariel:pw@db:5432/ariel"}}

    with caplog.at_level(logging.WARNING, logger=DSN_LOGGER):
        for _ in range(3):
            ARIELConfig.from_dict(ariel_section, {})

    warnings_emitted = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and "connection_string" in r.getMessage()
    ]
    assert len(warnings_emitted) == 1, (
        f"expected exactly one warning across three parses, got {len(warnings_emitted)}"
    )


@pytest.mark.unit
def test_deriving_the_dsn_stays_silent(caplog, rearmed_connection_string_warning) -> None:
    """Deriving is the normal path — it is not a deprecation event."""
    with caplog.at_level(logging.DEBUG, logger="osprey"):
        ARIELConfig.from_dict({}, {"username": "ariel", "database_name": "ariel"})

    noisy = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not noisy, "deriving the DSN emitted: " + "; ".join(r.getMessage() for r in noisy)
