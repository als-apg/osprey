"""DB-backed contract for ``apply_scenarios`` logbook seeding (Postgres-gated).

Exercises the full apply path end to end against a real database: compose the
active scenarios, purge the logbook, reseed from the bundles' relative-timestamp
entries against a fixed anchor, and assert the DB holds exactly the expected
entries with timestamps pinned to the documented time-of-day. Uses the shared
``database_url`` fixture (skips when no Postgres is available).
"""

from __future__ import annotations

import asyncio
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from osprey.simulation.apply import apply_scenarios

# xdist_group("docker"): pins every container-starting test file onto one worker, so
# a run has a single testcontainers session and a single ryuk reaper -- concurrent
# reaper starts race the Docker daemon's port mapper. It also serializes the shared
# database: the session ``database_url`` fixture prefers a running dev Postgres with
# ONE shared ``ariel_test`` database over a per-worker container, so parallel workers
# would otherwise collide on migrations/seed/truncate.
pytestmark = [pytest.mark.xdist_group("docker")]

TEMPLATE_SIM = (
    Path(__file__).resolve().parents[3]
    / "src/osprey/templates/apps/control_assistant/data/simulation"
)
# Fixed apply-time anchor T0 so resolved timestamps are deterministic.
T0 = datetime(2026, 6, 13, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _restore_schema_after(integration_ariel_config, database_url):
    """Re-run migrations after each test to undo apply's full-DB purge.

    ``apply_scenarios`` purges the logbook (truncates entries, drops the
    ``text_embeddings_*`` tables). Tests in this session share one database and
    migrate only once (the session ``_migrations_applied`` guard), so without
    this restore an embedding table dropped here would never be recreated and
    later embedding integration tests would fail. Migrations are idempotent.
    """
    yield

    async def _remigrate() -> None:
        from osprey.services.ariel_search.config import DatabaseConfig
        from osprey.services.ariel_search.database import (
            create_connection_pool,
            run_migrations,
        )

        pool = await create_connection_pool(DatabaseConfig(uri=database_url))
        try:
            await run_migrations(pool, integration_ariel_config)
        finally:
            await pool.close()

    asyncio.run(_remigrate())


def _make_project(tmp_path: Path, database_url: str) -> Path:
    """Stage a minimal sim-backed project pointing ARIEL at the test DB."""
    sim_dst = tmp_path / "data" / "simulation"
    sim_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEMPLATE_SIM, sim_dst)
    config = {
        "control_system": {
            "connector": {"mock": {"simulation_file": "data/simulation/machine.json"}}
        },
        "ariel": {"database": {"uri": database_url}},
    }
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config))
    return tmp_path


async def _fetch(database_url: str) -> dict[str, datetime]:
    """Return ``{entry_id: timestamp}`` with timestamps normalized to UTC.

    ``timestamptz`` columns read back in the DB session's local timezone, which
    would shift the wall-clock date; normalize to UTC so assertions compare the
    instant we stored, not its local rendering.
    """
    from osprey.services.ariel_search.config import DatabaseConfig
    from osprey.services.ariel_search.database import create_connection_pool

    pool = await create_connection_pool(DatabaseConfig(uri=database_url))
    try:
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute("SELECT entry_id, timestamp FROM enhanced_entries")
            rows = await cur.fetchall()
    finally:
        await pool.close()
    return {entry_id: ts.astimezone(UTC) for entry_id, ts in rows}


def test_apply_seeds_exactly_the_active_entries(tmp_path, database_url):
    project = _make_project(tmp_path, database_url)

    result = apply_scenarios(project, ["rf-thermal"], now=T0)
    assert result.active == ("nominal", "rf-thermal")
    assert result.purged is True
    assert result.logbook_seeded == 28  # 25 ambient + 3 incident

    entries = asyncio.run(_fetch(database_url))
    assert len(entries) == 28
    assert {"DEMO-001", "DEMO-025", "DEMO-026", "DEMO-027", "DEMO-028"} <= set(entries)


def test_resolved_timestamps_pin_time_of_day(tmp_path, database_url):
    project = _make_project(tmp_path, database_url)
    apply_scenarios(project, ["rf-thermal"], now=T0)
    entries = asyncio.run(_fetch(database_url))

    # DEMO-026: days_ago=4 at 03:20:00 -> 2026-06-09 03:20 UTC.
    demo026 = entries["DEMO-026"]
    assert (demo026.year, demo026.month, demo026.day) == (2026, 6, 9)
    assert (demo026.hour, demo026.minute) == (3, 20)
    # DEMO-028 (newest) lands at now-2d.
    assert entries["DEMO-028"].day == 11


def test_purge_replaces_prior_narrative(tmp_path, database_url):
    project = _make_project(tmp_path, database_url)

    apply_scenarios(project, ["rf-thermal"], now=T0)
    assert "DEMO-026" in asyncio.run(_fetch(database_url))

    # Re-apply with a telemetry-only scenario: the incident arc must be purged.
    result = apply_scenarios(project, ["vacuum-burst"], now=T0)
    assert result.logbook_seeded == 25
    entries = asyncio.run(_fetch(database_url))
    assert len(entries) == 25
    assert "DEMO-026" not in entries


def test_reapply_is_idempotent(tmp_path, database_url):
    project = _make_project(tmp_path, database_url)

    first = apply_scenarios(project, ["rf-thermal"], now=T0)
    second = apply_scenarios(project, ["rf-thermal"], now=T0)
    assert first.logbook_seeded == second.logbook_seeded == 28

    entries = asyncio.run(_fetch(database_url))
    assert len(entries) == 28  # no duplication across re-applies


async def _embedding_tables(database_url: str) -> set[str]:
    """Every text and image embedding table currently in the store."""
    from osprey.services.ariel_search.config import DatabaseConfig
    from osprey.services.ariel_search.database import create_connection_pool
    from osprey.services.ariel_search.database.repository import image_embedding_table_names

    pool = await create_connection_pool(DatabaseConfig(uri=database_url))
    try:
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name LIKE 'text_embeddings_%'"
            )
            tables = {row[0] for row in await cur.fetchall()}
            tables.update(await image_embedding_table_names(cur))
    finally:
        await pool.close()
    return tables


def test_apply_leaves_the_embedding_tables_in_place(tmp_path, database_url):
    """The purge inside apply drops every embedding table; apply migrates again
    afterwards, so vector and picture search work on the reseeded logbook without
    a manual ``osprey ariel migrate`` (a running ingest watcher never recreates
    them on its own)."""
    from osprey.services.ariel_search.database.migrations import image_table_name
    from tests.services.ariel_search.llama_stub import MODEL as IMAGE_MODEL

    project = _make_project(tmp_path, database_url)
    config = yaml.safe_load((project / "config.yml").read_text())
    config["ariel"]["enhancement_modules"] = {
        "text_embedding": {
            "enabled": True,
            "models": [{"name": "nomic-embed-text", "dimension": 768}],
        },
        "image_embedding": {
            "enabled": True,
            "provider": {"name": "llama-cpp", "base_url": "http://127.0.0.1:9"},
            "model": IMAGE_MODEL,
            "dimensions": 1024,
        },
    }
    (project / "config.yml").write_text(yaml.safe_dump(config))

    apply_scenarios(project, ["rf-thermal"], now=T0)

    tables = asyncio.run(_embedding_tables(database_url))
    assert image_table_name(IMAGE_MODEL, 1024) in tables
    assert any(table.startswith("text_embeddings_") for table in tables), tables


async def _pictures(database_url: str, entry_id: str) -> tuple[list, list, list]:
    """``(entry attachments JSONB, copy rows, renditions)`` for one seeded entry."""
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database import create_connection_pool
    from osprey.services.ariel_search.database.repository import ARIELRepository

    config = ARIELConfig.from_dict({"database": {"uri": database_url}})
    pool = await create_connection_pool(config.database)
    try:
        repository = ARIELRepository(pool, config)
        entry = await repository.get_entry(entry_id)
        rows = await repository.get_copy_rows(entry_id)
        renditions = [await repository.get_rendition(row["attachment_id"]) for row in rows]
    finally:
        await pool.close()
    assert entry is not None, f"{entry_id} was not seeded"
    return list(entry["attachments"]), rows, renditions


def test_seeded_pictures_are_copied_with_a_viewable_rendition(tmp_path, database_url):
    """A bundle entry's picture is in the store, linked on the entry and viewable as
    soon as apply returns -- no enhancement pass runs in between."""
    project = _make_project(tmp_path, database_url)
    apply_scenarios(project, ["rf-thermal"], now=T0)

    items, rows, renditions = asyncio.run(_pictures(database_url, "DEMO-027"))

    assert len(items) == len(rows) == 1
    (row,) = rows
    assert row["copy_status"] == "copied", row
    assert items[0]["url"] == f"/api/attachments/{row['attachment_id']}"
    assert items[0]["filename"] == "cavity_temperatures_week.png"
    assert renditions[0] is not None

    bare, bare_rows, _ = asyncio.run(_pictures(database_url, "DEMO-026"))
    assert bare == [] and bare_rows == []


def test_reapply_replaces_pictures_rather_than_piling_them_up(tmp_path, database_url):
    project = _make_project(tmp_path, database_url)
    apply_scenarios(project, ["rf-thermal"], now=T0)
    apply_scenarios(project, ["rf-thermal"], now=T0)

    items, rows, _ = asyncio.run(_pictures(database_url, "DEMO-027"))
    assert len(items) == len(rows) == 1


def test_a_demo_narrative_seeds_every_scenario_with_its_pictures(tmp_path, database_url):
    """The standalone path: no simulation, every bundle's narrative, viewable pictures."""
    from osprey.services.ariel_search.cli_operations import run_migrate
    from osprey.simulation.apply import seed_active_logbook

    shutil.copytree(TEMPLATE_SIM / "scenarios", tmp_path / "data" / "logbook_seed")
    ariel = {"database": {"uri": database_url}, "demo_narrative": "data/logbook_seed"}
    config = {"ariel": ariel}
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config))
    asyncio.run(run_migrate(ariel))
    from osprey.services.ariel_search.cli_operations import execute_purge

    asyncio.run(execute_purge(ariel, embeddings_only=False))

    seeded = seed_active_logbook(config, tmp_path, ariel)

    assert seeded == 29
    assert len(asyncio.run(_fetch(database_url))) == 29
    for entry_id in ("DEMO-011", "DEMO-027", "DEMO-031"):
        items, rows, renditions = asyncio.run(_pictures(database_url, entry_id))
        assert len(items) == len(rows) == 1, entry_id
        assert rows[0]["copy_status"] == "copied"
        assert renditions[0] is not None
    # A second deploy finds the logbook full and adds nothing.
    assert seed_active_logbook(config, tmp_path, ariel) == 0
