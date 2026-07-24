"""Shared helper for DuckDB full-text-search extension setup.

Kept free of heavy imports so MCP tool modules can import it at load time.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

_BUNDLED_FTS = Path("data/duckdb_extensions/fts.duckdb_extension")


def ensure_fts(con: duckdb.DuckDBPyConnection) -> None:
    """Load the FTS extension, installing it first if this machine lacks it.

    Tries a plain LOAD first (the common case once installed); on failure
    installs from a bundled local copy when present, otherwise downloads
    from the extension repository, then loads.
    """
    try:
        con.execute("LOAD fts")
        return
    except duckdb.Error:
        logger.info("FTS extension not installed yet, installing")
    if _BUNDLED_FTS.exists():
        logger.info("Installing FTS extension from bundled file: %s", _BUNDLED_FTS)
        con.execute(f"INSTALL '{_BUNDLED_FTS.resolve()}'")
    else:
        logger.info("No bundled FTS extension found, downloading from repository")
        proxy = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY", "")
        if proxy:
            con.execute(f"SET http_proxy = '{proxy}'")
        con.execute("INSTALL fts")
    con.execute("LOAD fts")
