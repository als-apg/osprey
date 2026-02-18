"""JSON → SQLite FTS5 indexer for AccelPapers.

Reads structured JSON files from INSPIRE downloads and builds a searchable
SQLite database with full-text search via FTS5.
"""

import json
import logging
import os
import sqlite3
import time
from pathlib import Path

from osprey.mcp_server.accelpapers.db import get_connection

logger = logging.getLogger("osprey.mcp_server.accelpapers.indexer")

# --- Schema -------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    texkey TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    year INTEGER,
    earliest_date TEXT,
    conference TEXT,
    first_author TEXT,
    all_authors TEXT,
    affiliations TEXT,
    keywords TEXT,
    doi TEXT,
    citation_count INTEGER DEFAULT 0,
    paper_id TEXT,
    document_type TEXT,
    inspire_url TEXT,
    pdf_url TEXT,
    arxiv_id TEXT,
    journal_title TEXT,
    publisher TEXT,
    num_pages INTEGER,
    full_text TEXT,
    json_path TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    title, abstract, all_authors, keywords, full_text,
    content=papers, content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE INDEX IF NOT EXISTS idx_conference ON papers(conference);
CREATE INDEX IF NOT EXISTS idx_year ON papers(year);
CREATE INDEX IF NOT EXISTS idx_first_author ON papers(first_author);
CREATE INDEX IF NOT EXISTS idx_citation_count ON papers(citation_count);
CREATE INDEX IF NOT EXISTS idx_document_type ON papers(document_type);

CREATE TABLE IF NOT EXISTS stats (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

# FTS triggers for keeping the index in sync with the content table
FTS_TRIGGERS_SQL = """
CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
    INSERT INTO papers_fts(rowid, title, abstract, all_authors, keywords, full_text)
    VALUES (new.rowid, new.title, new.abstract, new.all_authors, new.keywords, new.full_text);
END;

CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, title, abstract, all_authors, keywords, full_text)
    VALUES ('delete', old.rowid, old.title, old.abstract, old.all_authors, old.keywords, old.full_text);
END;

CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, title, abstract, all_authors, keywords, full_text)
    VALUES ('delete', old.rowid, old.title, old.abstract, old.all_authors, old.keywords, old.full_text);
    INSERT INTO papers_fts(rowid, title, abstract, all_authors, keywords, full_text)
    VALUES (new.rowid, new.title, new.abstract, new.all_authors, new.keywords, new.full_text);
END;
"""


# --- Helpers ------------------------------------------------------------------


def normalize_year(raw: str | int | None) -> int | None:
    """Extract a 4-digit year from various formats."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # Take first 4 digits
    digits = "".join(c for c in s[:10] if c.isdigit())
    if len(digits) >= 4:
        year = int(digits[:4])
        if 1900 <= year <= 2100:
            return year
    return None


def normalize_num_pages(raw: str | int | None) -> int | None:
    """Extract page count from string or int."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    digits = "".join(c for c in s if c.isdigit())
    if digits:
        return int(digits)
    return None


def extract_full_text(content: dict | None) -> str:
    """Extract concatenated full text from content.sections dict.

    ``content.sections`` is a dict keyed by string numbers, each with
    ``full_text`` (list of strings) and optionally ``clean_text``.
    """
    if not content or "sections" not in content:
        return ""
    sections = content["sections"]
    if not isinstance(sections, dict):
        return ""
    parts = []
    for _key in sorted(sections.keys(), key=lambda k: int(k) if k.isdigit() else 0):
        section = sections[_key]
        title = section.get("title", "")
        if title:
            parts.append(title)
        full_text = section.get("full_text", [])
        if isinstance(full_text, list):
            parts.extend(str(t) for t in full_text if t)
        elif isinstance(full_text, str):
            parts.append(full_text)
    return "\n".join(parts)


def get_conference(data: dict, subdir_name: str) -> str:
    """Determine conference name from paper data or parent directory."""
    conf = data.get("conf_acronym", "").strip()
    if conf:
        return conf
    # Fall back to subdir name if it looks like a conference
    if subdir_name and not subdir_name.startswith("ARXIV") and subdir_name != "book-chapter":
        return subdir_name
    return ""


def build_all_authors(data: dict) -> str:
    """Build a semicolon-separated author list."""
    authors = []
    first = data.get("first_author_full_name", "")
    if first:
        authors.append(first)
    others = data.get("other_authors_full_names", [])
    if isinstance(others, list):
        authors.extend(str(a) for a in others if a)
    return "; ".join(authors)


def _extract_affiliations(data: dict) -> str:
    """Extract affiliations from paper data."""
    affs = []
    first_aff = data.get("first_author_affiliations", "")
    if first_aff:
        affs.append(first_aff)
    # Also check metadata.authors for more affiliations
    meta_authors = data.get("metadata", {}).get("authors", [])
    for author in meta_authors:
        for aff in author.get("affiliations", []):
            val = aff.get("value", "")
            if val and val not in affs:
                affs.append(val)
    return "; ".join(affs)


def _parse_paper(data: dict, json_path: str, subdir_name: str) -> dict | None:
    """Parse a single paper JSON into a row dict. Returns None if unusable."""
    texkey = data.get("texkey", "").strip()
    title = data.get("title", "").strip()
    if not texkey or not title:
        return None

    keywords = data.get("keywords", [])
    kw_str = "; ".join(str(k) for k in keywords) if isinstance(keywords, list) else ""

    return {
        "texkey": texkey,
        "title": title,
        "abstract": data.get("abstract_value", ""),
        "year": normalize_year(data.get("year")),
        "earliest_date": data.get("earliest_date", ""),
        "conference": get_conference(data, subdir_name),
        "first_author": data.get("first_author_full_name", ""),
        "all_authors": build_all_authors(data),
        "affiliations": _extract_affiliations(data),
        "keywords": kw_str,
        "doi": data.get("doi", ""),
        "citation_count": data.get("citation_count", 0) or 0,
        "paper_id": data.get("paper_id", ""),
        "document_type": data.get("document_type", ""),
        "inspire_url": data.get("inspireHEP_url", ""),
        "pdf_url": data.get("pdf_url", ""),
        "arxiv_id": data.get("arxiv_eprints", ""),
        "journal_title": data.get("journal_title", ""),
        "publisher": data.get("publisher", ""),
        "num_pages": normalize_num_pages(data.get("number_of_pages")),
        "full_text": extract_full_text(data.get("content")),
        "json_path": json_path,
    }


# --- Main indexer -------------------------------------------------------------

INSERT_SQL = """
INSERT OR REPLACE INTO papers (
    texkey, title, abstract, year, earliest_date, conference,
    first_author, all_authors, affiliations, keywords, doi,
    citation_count, paper_id, document_type, inspire_url,
    pdf_url, arxiv_id, journal_title, publisher, num_pages,
    full_text, json_path
) VALUES (
    :texkey, :title, :abstract, :year, :earliest_date, :conference,
    :first_author, :all_authors, :affiliations, :keywords, :doi,
    :citation_count, :paper_id, :document_type, :inspire_url,
    :pdf_url, :arxiv_id, :journal_title, :publisher, :num_pages,
    :full_text, :json_path
)
"""


def _collect_json_files(data_dir: Path) -> list[tuple[Path, str]]:
    """Collect all JSON files with their parent subdir name."""
    files = []
    for entry in sorted(data_dir.iterdir()):
        if entry.is_file() and entry.suffix == ".json":
            files.append((entry, ""))
        elif entry.is_dir():
            subdir_name = entry.name
            for f in sorted(entry.iterdir()):
                if f.is_file() and f.suffix == ".json":
                    files.append((f, subdir_name))
    return files


def _materialize_stats(conn: sqlite3.Connection) -> None:
    """Compute and store aggregate statistics."""
    stats = {}

    row = conn.execute("SELECT COUNT(*) as c FROM papers").fetchone()
    stats["total_papers"] = str(row["c"])

    row = conn.execute(
        "SELECT MIN(year) as mn, MAX(year) as mx FROM papers WHERE year IS NOT NULL"
    ).fetchone()
    stats["year_min"] = str(row["mn"]) if row["mn"] else ""
    stats["year_max"] = str(row["mx"]) if row["mx"] else ""

    row = conn.execute("SELECT COUNT(DISTINCT conference) as c FROM papers WHERE conference != ''").fetchone()
    stats["num_conferences"] = str(row["c"])

    row = conn.execute("SELECT COUNT(DISTINCT first_author) as c FROM papers WHERE first_author != ''").fetchone()
    stats["num_authors"] = str(row["c"])

    row = conn.execute(
        "SELECT SUM(citation_count) as s FROM papers WHERE citation_count > 0"
    ).fetchone()
    stats["total_citations"] = str(row["s"]) if row["s"] else "0"

    # Document type breakdown
    rows = conn.execute(
        "SELECT document_type, COUNT(*) as c FROM papers GROUP BY document_type ORDER BY c DESC"
    ).fetchall()
    stats["document_types"] = json.dumps({r["document_type"]: r["c"] for r in rows})

    # Top 10 conferences
    rows = conn.execute(
        "SELECT conference, COUNT(*) as c FROM papers WHERE conference != '' "
        "GROUP BY conference ORDER BY c DESC LIMIT 10"
    ).fetchall()
    stats["top_conferences"] = json.dumps({r["conference"]: r["c"] for r in rows})

    for key, value in stats.items():
        conn.execute(
            "INSERT OR REPLACE INTO stats (key, value) VALUES (?, ?)",
            (key, value),
        )
    conn.commit()


def build_index(data_dir: str | Path, db_path: str | Path | None = None, batch_size: int = 500) -> Path:
    """Build the SQLite FTS5 index from INSPIRE JSON files.

    Args:
        data_dir: Directory containing JSON files (possibly in subdirectories).
        db_path: Output database path. Defaults to ``get_db_path()``.
        batch_size: Number of rows per INSERT batch.

    Returns:
        Path to the created database.
    """
    from osprey.mcp_server.accelpapers.db import get_db_path

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    db_path = Path(db_path) if db_path else get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building index: %s → %s", data_dir, db_path)

    conn = get_connection(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.executescript(FTS_TRIGGERS_SQL)

    json_files = _collect_json_files(data_dir)
    total = len(json_files)
    logger.info("Found %d JSON files to index", total)

    indexed = 0
    skipped = 0
    errors = 0
    batch: list[dict] = []
    t0 = time.time()

    for i, (fpath, subdir_name) in enumerate(json_files):
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", fpath.name, exc)
            errors += 1
            continue

        if not isinstance(data, dict):
            skipped += 1
            continue

        row = _parse_paper(data, str(fpath), subdir_name)
        if row is None:
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
                "Progress: %d/%d (%.0f files/sec) — indexed=%d skipped=%d errors=%d",
                i + 1, total, rate, indexed, skipped, errors,
            )

    # Flush remaining batch
    if batch:
        conn.executemany(INSERT_SQL, batch)
        conn.commit()
        indexed += len(batch)

    # Rebuild FTS index for optimal ranking
    logger.info("Rebuilding FTS index...")
    conn.execute("INSERT INTO papers_fts(papers_fts) VALUES ('rebuild')")
    conn.commit()

    # Materialize stats
    logger.info("Computing statistics...")
    _materialize_stats(conn)

    elapsed = time.time() - t0
    logger.info(
        "Indexing complete in %.1fs: %d indexed, %d skipped, %d errors (DB: %s)",
        elapsed, indexed, skipped, errors, db_path,
    )

    # Log DB size
    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    logger.info("Database size: %.1f MB", db_size_mb)

    conn.close()
    return db_path
