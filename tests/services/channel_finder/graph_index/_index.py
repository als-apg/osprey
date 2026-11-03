"""Helpers for the graph-index suites: a build's ``meta`` and a read-only query."""

from __future__ import annotations

from pathlib import Path

import duckdb

from osprey.services.channel_finder.graph_index.builder import CALLER_META_KEYS, ParsedCorpus


def corpus_meta(parsed: ParsedCorpus, **overrides: object) -> dict:
    """The ``meta`` mapping a corpus build states for *parsed*."""
    values = {
        "corpus_sha256": "c" * 64,
        "corpus_filename": "facility.ttl",
        "binding_count": len(parsed.binding_rows),
        "device_count": len({row.device_uri for row in parsed.binding_rows}),
        "class_count": len(parsed.class_rows),
        "signal_count": parsed.signal_count,
        "section_count": len(parsed.section_codes),
    }
    values.update(overrides)
    assert set(values) == set(CALLER_META_KEYS)
    return values


def read_rows(index_path: Path, sql: str) -> list[tuple]:
    """Run ``sql`` against a fresh read-only connection, as the reader will."""
    con = duckdb.connect(str(index_path), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()
