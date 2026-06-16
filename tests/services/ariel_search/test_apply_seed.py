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
