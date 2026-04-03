"""Import Middle Layer JSON into a DuckDB database.

Reads the MML JSON file via MiddleLayerDatabase (reusing all existing
flattening logic), then exports the data into DuckDB tables with a
full-text search index.

Usage:
    python -m osprey.services.channel_finder.databases.duckdb_import \
        --json data/middle_layer.json --output data/middle_layer.duckdb

The import is idempotent: rows with source='mml' are replaced on each
run, while source='runtime' rows (added by agents at runtime) are
preserved across rebuilds.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from osprey.services.channel_finder.databases.middle_layer import MiddleLayerDatabase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS systems (
    name        TEXT PRIMARY KEY,
    description TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS families (
    system      TEXT NOT NULL,
    name        TEXT NOT NULL,
    description TEXT DEFAULT '',
    PRIMARY KEY (system, name)
);

CREATE TABLE IF NOT EXISTS channels (
    channel_name TEXT PRIMARY KEY,
    system       TEXT NOT NULL,
    family       TEXT NOT NULL,
    field        TEXT DEFAULT '',
    subfield     TEXT DEFAULT '',
    description  TEXT DEFAULT '',
    units        TEXT DEFAULT '',
    data_type    TEXT DEFAULT '',
    mode         TEXT DEFAULT '',
    member_of    TEXT DEFAULT '',
    source       TEXT DEFAULT 'mml',
    updated_at   TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS device_map (
    system       TEXT NOT NULL,
    family       TEXT NOT NULL,
    device_index INTEGER NOT NULL,
    sector       INTEGER,
    device       INTEGER,
    common_name  TEXT DEFAULT '',
    PRIMARY KEY (system, family, device_index)
);
"""


def _install_fts(con: duckdb.DuckDBPyConnection) -> None:
    """Install and load the FTS extension, preferring a bundled local copy."""
    bundled = Path("data/duckdb_extensions/fts.duckdb_extension")
    if bundled.exists():
        logger.info("Installing FTS extension from bundled file: %s", bundled)
        con.execute(f"INSTALL '{bundled.resolve()}'")
    else:
        logger.info("No bundled FTS extension found, downloading from repository")
        proxy = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY", "")
        if proxy:
            con.execute(f"SET http_proxy = '{proxy}'")
        con.execute("INSTALL fts")
    con.execute("LOAD fts")


def _create_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create tables if they don't already exist."""
    _install_fts(con)
    for stmt in _SCHEMA_SQL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            con.execute(stmt)


def _import_systems(con: duckdb.DuckDBPyConnection, db: MiddleLayerDatabase) -> int:
    """Import system-level data. Returns row count."""
    systems = db.list_systems()
    if not systems:
        return 0
    con.execute("DELETE FROM systems")
    con.executemany(
        "INSERT INTO systems (name, description) VALUES (?, ?)",
        [(s["name"], s["description"]) for s in systems],
    )
    return len(systems)


def _import_families(con: duckdb.DuckDBPyConnection, db: MiddleLayerDatabase) -> int:
    """Import family-level data. Returns row count."""
    rows = []
    for system_info in db.list_systems():
        system = system_info["name"]
        for fam in db.list_families(system):
            rows.append((system, fam["name"], fam["description"]))
    if not rows:
        return 0
    con.execute("DELETE FROM families")
    con.executemany(
        "INSERT INTO families (system, name, description) VALUES (?, ?, ?)",
        rows,
    )
    return len(rows)


def _import_channels(con: duckdb.DuckDBPyConnection, db: MiddleLayerDatabase) -> int:
    """Import channels from the flattened channel_map. Returns row count."""
    now = datetime.now(timezone.utc)

    # Only delete MML-sourced rows, preserving runtime additions
    con.execute("DELETE FROM channels WHERE source = 'mml'")

    rows = []
    for ch_name, meta in db.channel_map.items():
        subfield = meta.get("subfield")
        if isinstance(subfield, list):
            subfield = ":".join(subfield) if subfield else ""
        elif subfield is None:
            subfield = ""

        member_of = meta.get("MemberOf", "")
        if isinstance(member_of, list):
            member_of = ", ".join(str(m) for m in member_of)

        rows.append((
            ch_name,
            meta.get("system", ""),
            meta.get("family", ""),
            meta.get("field", ""),
            subfield,
            meta.get("Description", meta.get("description", "")),
            meta.get("Units", meta.get("HWUnits", "")),
            meta.get("DataType", ""),
            meta.get("Mode", ""),
            member_of,
            "mml",
            now,
        ))

    if rows:
        con.executemany(
            """INSERT INTO channels
               (channel_name, system, family, field, subfield, description,
                units, data_type, mode, member_of, source, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
    return len(rows)


def _import_device_map(con: duckdb.DuckDBPyConnection, db: MiddleLayerDatabase) -> int:
    """Import device maps (setup/DeviceList + CommonNames). Returns row count."""
    con.execute("DELETE FROM device_map")
    rows = []

    for system_info in db.list_systems():
        system = system_info["name"]
        for fam_info in db.list_families(system):
            family = fam_info["name"]
            family_data = db.data.get(system, {}).get(family, {})
            setup = family_data.get("setup", {})
            if not isinstance(setup, dict):
                continue
            device_list = setup.get("DeviceList")
            if not device_list:
                continue
            common_names = setup.get("CommonNames") or []

            for idx, entry in enumerate(device_list):
                if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                    continue
                sector, device = entry[0], entry[1]
                cname = common_names[idx] if idx < len(common_names) else ""
                rows.append((system, family, idx, sector, device, cname))

    if rows:
        con.executemany(
            """INSERT INTO device_map
               (system, family, device_index, sector, device, common_name)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
    return len(rows)


def _create_fts_index(con: duckdb.DuckDBPyConnection) -> None:
    """Create full-text search index on channels table."""
    # Drop existing FTS index if present (overwrite=1)
    con.execute(
        "PRAGMA create_fts_index('channels', 'channel_name', "
        "'channel_name', 'description', 'system', 'family', 'field', "
        "overwrite=1)"
    )


def import_to_duckdb(json_path: str, output_path: str) -> dict:
    """Import a Middle Layer JSON file into a DuckDB database.

    Args:
        json_path: Path to the MML JSON file.
        output_path: Path for the output .duckdb file.

    Returns:
        Dict with import statistics.
    """
    logger.info("Loading MML database from %s", json_path)
    db = MiddleLayerDatabase(json_path)

    logger.info("Opening DuckDB at %s", output_path)
    con = duckdb.connect(output_path)

    try:
        _create_schema(con)

        n_systems = _import_systems(con, db)
        n_families = _import_families(con, db)
        n_channels = _import_channels(con, db)
        n_devices = _import_device_map(con, db)
        _create_fts_index(con)

        stats = {
            "systems": n_systems,
            "families": n_families,
            "channels": n_channels,
            "device_map_entries": n_devices,
            "json_path": json_path,
            "duckdb_path": output_path,
        }
        logger.info("Import complete: %s", stats)
        return stats
    finally:
        con.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import Middle Layer JSON into DuckDB",
    )
    parser.add_argument(
        "--json", required=True, help="Path to middle_layer.json",
    )
    parser.add_argument(
        "--output", required=True, help="Path for output .duckdb file",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    stats = import_to_duckdb(args.json, args.output)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
