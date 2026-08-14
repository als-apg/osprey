"""A deploy that runs ARIEL's Postgres also makes it usable.

Before this, ``osprey up`` started the ARIEL store and stopped there: nothing on
the deploy path ever created the schema, so a first bring-up left a database with
no tables, an ARIEL panel that reported the database unavailable, and an `ariel`
MCP server whose every tool failed on a missing relation. Migrations existed only
behind commands an operator had to know to run (``osprey ariel migrate`` /
``quickstart``), which is knowledge a turn-key preset cannot assume.

Staging ARIEL is therefore the logbook counterpart of ``_stage_archiver_store``:
start the store, wait for it, create the schema, and — on a first bring-up only —
write the narrative the active scenarios document.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.deployment import container_lifecycle

ARIEL_CONFIG: dict = {
    "project_name": "demo",
    "deployed_services": ["postgresql"],
    "services": {"postgresql": {"username": "ariel", "database_name": "ariel", "port_host": 5432}},
    "ariel": {"search_modules": {"keyword": {"enabled": True}}},
}


@pytest.fixture
def ariel_stubs(monkeypatch, tmp_path):
    """Reduce ``_stage_ariel_store`` to its compose call and its two ARIEL calls."""
    state: dict = {"cmds": [], "migrated": [], "seeded": [], "waited": []}

    (tmp_path / ".env").write_text("ARIEL_DB_PASSWORD=s3cret\n", encoding="utf-8")

    monkeypatch.setattr(container_lifecycle, "runtime_env", lambda config, env: dict(env))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(
        container_lifecycle, "compose_base_cmd", lambda *a, **k: ["docker", "compose"]
    )
    monkeypatch.setattr(container_lifecycle, "_env_file_args", lambda root=None: [])
    monkeypatch.setattr(
        container_lifecycle,
        "run_captured",
        lambda cmd, **kwargs: state["cmds"].append(list(cmd)),
    )
    monkeypatch.setattr(
        container_lifecycle,
        "_wait_for_ariel_store",
        lambda ariel_config, deadline: state["waited"].append(ariel_config),
    )
    monkeypatch.setattr(
        container_lifecycle,
        "_migrate_ariel_store",
        lambda ariel_config: state["migrated"].append(ariel_config),
    )

    from osprey.simulation import apply as apply_mod

    monkeypatch.setattr(
        apply_mod,
        "seed_active_logbook",
        lambda config, project_dir, ariel_config: state["seeded"].append(ariel_config) or 3,
    )
    return state


def _stage(config: dict, project_dir: Path) -> None:
    container_lifecycle._stage_ariel_store(config, ["compose.yml"], {}, project_dir)


def test_the_store_is_started_on_its_own(ariel_stubs, tmp_path):
    """Started ahead of the stack, like the archiver store: the schema has to
    exist before anything that reads it comes up."""
    # Act
    _stage(ARIEL_CONFIG, tmp_path)

    # Assert
    assert ariel_stubs["cmds"] == [["docker", "compose", "up", "-d", "postgresql"]]


def test_the_schema_is_created_before_anything_reads_it(ariel_stubs, tmp_path):
    """The defect this staging exists for: a store with no tables."""
    # Act
    _stage(ARIEL_CONFIG, tmp_path)

    # Assert
    assert len(ariel_stubs["migrated"]) == 1


def test_migrations_authenticate_with_the_project_dotenv_password(
    ariel_stubs, tmp_path, monkeypatch
):
    """Read from the project's own `.env` by name -- never from whatever the
    ambient environment happens to export for another deployment's database."""
    # Arrange
    monkeypatch.setenv("ARIEL_DB_PASSWORD", "someone-elses-database")

    # Act
    _stage(ARIEL_CONFIG, tmp_path)

    # Assert
    dsn = ariel_stubs["migrated"][0]["database"]["uri"]
    assert "s3cret" in dsn
    assert "someone-elses-database" not in dsn


def test_the_active_narrative_is_seeded_after_the_schema(ariel_stubs, tmp_path):
    """Order matters: seeding writes into tables the migration creates."""
    # Act
    _stage(ARIEL_CONFIG, tmp_path)

    # Assert
    assert len(ariel_stubs["seeded"]) == 1
    assert ariel_stubs["seeded"][0] == ariel_stubs["migrated"][0]


def test_a_store_the_project_does_not_run_is_left_alone(ariel_stubs, tmp_path):
    """A project pointed at a Postgres someone else runs must never have its
    schema or its contents rewritten by a local deploy."""
    # Arrange
    config = {**ARIEL_CONFIG, "deployed_services": []}

    # Act
    _stage(config, tmp_path)

    # Assert
    assert ariel_stubs["cmds"] == []
    assert ariel_stubs["migrated"] == []


def test_a_project_without_ariel_stages_nothing(ariel_stubs, tmp_path):
    """The store can be deployed for something else entirely; with no `ariel:`
    section there is no schema this deploy knows how to create."""
    # Arrange
    config = {key: value for key, value in ARIEL_CONFIG.items() if key != "ariel"}

    # Act
    _stage(config, tmp_path)

    # Assert
    assert ariel_stubs["migrated"] == []


def test_an_unreachable_store_warns_and_leaves_the_deploy_standing(
    ariel_stubs, tmp_path, monkeypatch, caplog
):
    """The logbook is one panel of many. A database that will not answer must be
    said out loud, naming the command that finishes the job -- but a control-room
    deploy whose scans and channels are fine is not aborted over its search tab."""

    # Arrange
    def _boom(ariel_config, deadline):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(container_lifecycle, "_wait_for_ariel_store", _boom)

    # Act
    with caplog.at_level("WARNING"):
        _stage(ARIEL_CONFIG, tmp_path)

    # Assert
    assert "osprey ariel quickstart" in caplog.text
    assert ariel_stubs["migrated"] == []
