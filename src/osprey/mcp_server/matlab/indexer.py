"""JSON → SQLite FTS5 indexer for MATLAB Middle Layer.

Reads the matlab_dependencies.json file (nodes[] and edges[]) and builds a
searchable SQLite database with full-text search via FTS5.
"""

import json
import logging
import os
import sqlite3
import time
from pathlib import Path

from osprey.mcp_server.matlab.db import get_connection

logger = logging.getLogger("osprey.mcp_server.matlab.indexer")

# --- Schema -------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS functions (
    function_name TEXT PRIMARY KEY,
    file_path TEXT,
    docstring TEXT,
    source_code TEXT,
    group_name TEXT,
    type TEXT,
    in_degree INTEGER DEFAULT 0,
    out_degree INTEGER DEFAULT 0
);

CREATE VIRTUAL TABLE IF NOT EXISTS functions_fts USING fts5(
    function_name, docstring, source_code, group_name,
    content=functions, content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS dependencies (
    caller TEXT NOT NULL REFERENCES functions(function_name),
    callee TEXT NOT NULL REFERENCES functions(function_name),
    PRIMARY KEY (caller, callee)
);

CREATE INDEX IF NOT EXISTS idx_group_name ON functions(group_name);
CREATE INDEX IF NOT EXISTS idx_type ON functions(type);
CREATE INDEX IF NOT EXISTS idx_in_degree ON functions(in_degree);
CREATE INDEX IF NOT EXISTS idx_out_degree ON functions(out_degree);

CREATE TABLE IF NOT EXISTS stats (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

# FTS triggers for keeping the index in sync with the content table
FTS_TRIGGERS_SQL = """
CREATE TRIGGER IF NOT EXISTS functions_ai AFTER INSERT ON functions BEGIN
    INSERT INTO functions_fts(rowid, function_name, docstring, source_code, group_name)
    VALUES (new.rowid, new.function_name, new.docstring, new.source_code, new.group_name);
END;

CREATE TRIGGER IF NOT EXISTS functions_ad AFTER DELETE ON functions BEGIN
    INSERT INTO functions_fts(functions_fts, rowid, function_name, docstring, source_code, group_name)
    VALUES ('delete', old.rowid, old.function_name, old.docstring, old.source_code, old.group_name);
END;

CREATE TRIGGER IF NOT EXISTS functions_au AFTER UPDATE ON functions BEGIN
    INSERT INTO functions_fts(functions_fts, rowid, function_name, docstring, source_code, group_name)
    VALUES ('delete', old.rowid, old.function_name, old.docstring, old.source_code, old.group_name);
    INSERT INTO functions_fts(rowid, function_name, docstring, source_code, group_name)
    VALUES (new.rowid, new.function_name, new.docstring, new.source_code, new.group_name);
END;
"""

# --- Helpers ------------------------------------------------------------------

INSERT_SQL = """
INSERT OR REPLACE INTO functions (
    function_name, file_path, docstring, source_code,
    group_name, type, in_degree, out_degree
) VALUES (
    :function_name, :file_path, :docstring, :source_code,
    :group_name, :type, :in_degree, :out_degree
)
"""

EDGE_INSERT_SQL = """
INSERT OR IGNORE INTO dependencies (caller, callee)
VALUES (?, ?)
"""


def _read_source_code(file_path: str | None) -> str | None:
    """Read source code from a .m file, returning None if unavailable."""
    if not file_path:
        return None
    path = Path(file_path)
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _parse_node(node: dict) -> dict:
    """Parse a single node from matlab_dependencies.json into a row dict."""
    function_name = node.get("id", "").strip()
    file_path = node.get("file_path", "")
    source_code = _read_source_code(file_path)

    return {
        "function_name": function_name,
        "file_path": file_path,
        "docstring": node.get("docstring", ""),
        "source_code": source_code,
        "group_name": node.get("group", ""),
        "type": node.get("type", ""),
        "in_degree": node.get("in_degree", 0) or 0,
        "out_degree": node.get("out_degree", 0) or 0,
    }


# --- Stats materialization ----------------------------------------------------


def _materialize_stats(conn: sqlite3.Connection) -> None:
    """Compute and store aggregate statistics."""
    stats = {}

    row = conn.execute("SELECT COUNT(*) as c FROM functions").fetchone()
    stats["total_functions"] = str(row["c"])

    row = conn.execute("SELECT COUNT(*) as c FROM dependencies").fetchone()
    stats["total_edges"] = str(row["c"])

    # Group breakdown
    rows = conn.execute(
        "SELECT group_name, COUNT(*) as c FROM functions "
        "WHERE group_name != '' GROUP BY group_name ORDER BY c DESC"
    ).fetchall()
    stats["groups"] = json.dumps({r["group_name"]: r["c"] for r in rows})

    # Type breakdown
    rows = conn.execute(
        "SELECT type, COUNT(*) as c FROM functions WHERE type != '' GROUP BY type ORDER BY c DESC"
    ).fetchall()
    stats["types"] = json.dumps({r["type"]: r["c"] for r in rows})

    # Top 10 most-called (highest in_degree)
    rows = conn.execute(
        "SELECT function_name, in_degree FROM functions ORDER BY in_degree DESC LIMIT 10"
    ).fetchall()
    stats["top_called"] = json.dumps({r["function_name"]: r["in_degree"] for r in rows})

    # Top 10 most-dependent (highest out_degree)
    rows = conn.execute(
        "SELECT function_name, out_degree FROM functions ORDER BY out_degree DESC LIMIT 10"
    ).fetchall()
    stats["top_dependent"] = json.dumps({r["function_name"]: r["out_degree"] for r in rows})

    for key, value in stats.items():
        conn.execute(
            "INSERT OR REPLACE INTO stats (key, value) VALUES (?, ?)",
            (key, value),
        )
    conn.commit()


# --- Main indexer -------------------------------------------------------------


def build_index(
    data_file: str | Path,
    db_path: str | Path | None = None,
    batch_size: int = 500,
) -> Path:
    """Build the SQLite FTS5 index from matlab_dependencies.json.

    Args:
        data_file: Path to matlab_dependencies.json.
        db_path: Output database path. Defaults to ``get_db_path()``.
        batch_size: Number of rows per INSERT batch.

    Returns:
        Path to the created database.
    """
    from osprey.mcp_server.matlab.db import get_db_path

    data_file = Path(data_file)
    if not data_file.is_file():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    db_path = Path(db_path) if db_path else get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building index: %s → %s", data_file, db_path)

    # Load JSON
    with open(data_file) as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    total_nodes = len(nodes)
    total_edges = len(edges)
    logger.info("Found %d nodes and %d edges", total_nodes, total_edges)

    conn = get_connection(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.executescript(FTS_TRIGGERS_SQL)

    t0 = time.time()

    # --- Index functions ---
    indexed = 0
    skipped = 0
    batch: list[dict] = []

    for i, node in enumerate(nodes):
        row = _parse_node(node)
        if not row["function_name"]:
            skipped += 1
            continue

        batch.append(row)
        if len(batch) >= batch_size:
            conn.executemany(INSERT_SQL, batch)
            conn.commit()
            indexed += len(batch)
            batch.clear()

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                "Functions: %d/%d (%.0f/sec) — indexed=%d skipped=%d",
                i + 1,
                total_nodes,
                rate,
                indexed,
                skipped,
            )

    # Flush remaining batch
    if batch:
        conn.executemany(INSERT_SQL, batch)
        conn.commit()
        indexed += len(batch)

    logger.info("Indexed %d functions, skipped %d", indexed, skipped)

    # --- Index dependencies (filter self-loops) ---
    edge_batch: list[tuple[str, str]] = []
    edge_count = 0
    self_loops = 0

    for edge in edges:
        caller = edge.get("source", "")
        callee = edge.get("target", "")
        if not caller or not callee:
            continue
        if caller == callee:
            self_loops += 1
            continue
        edge_batch.append((caller, callee))
        if len(edge_batch) >= batch_size:
            conn.executemany(EDGE_INSERT_SQL, edge_batch)
            conn.commit()
            edge_count += len(edge_batch)
            edge_batch.clear()

    if edge_batch:
        conn.executemany(EDGE_INSERT_SQL, edge_batch)
        conn.commit()
        edge_count += len(edge_batch)

    logger.info("Indexed %d edges, filtered %d self-loops", edge_count, self_loops)

    # Rebuild FTS index for optimal ranking
    logger.info("Rebuilding FTS index...")
    conn.execute("INSERT INTO functions_fts(functions_fts) VALUES ('rebuild')")
    conn.commit()

    # Materialize stats
    logger.info("Computing statistics...")
    _materialize_stats(conn)

    elapsed = time.time() - t0
    logger.info(
        "Indexing complete in %.1fs: %d functions, %d edges (DB: %s)",
        elapsed,
        indexed,
        edge_count,
        db_path,
    )

    # Log DB size
    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    logger.info("Database size: %.1f MB", db_size_mb)

    conn.close()
    return db_path
