"""Shared helper for DuckDB full-text-search extension setup.

Kept free of heavy imports so MCP tool modules can import it at load time.
"""

from __future__ import annotations

import logging
import os

import duckdb

logger = logging.getLogger(__name__)


def ensure_fts(con: duckdb.DuckDBPyConnection) -> None:
    """Load the FTS extension, installing it first if this machine lacks it.

    Tries a plain LOAD first (the common case once installed); on failure
    downloads the extension from the DuckDB extension repository, through the
    proxy named by ``http_proxy`` / ``HTTP_PROXY`` when one is set, then loads.

    Raises:
        RuntimeError: The extension is not installed and the download failed.
    """
    try:
        con.execute("LOAD fts")
        return
    except duckdb.Error:
        logger.info("FTS extension not installed yet, downloading from repository")
    proxy = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY", "")
    if proxy:
        con.execute(f"SET http_proxy = '{proxy}'")
    try:
        con.execute("INSTALL fts")
    except duckdb.Error as exc:
        raise RuntimeError(
            "The DuckDB full-text-search extension is not installed and could not be "
            "downloaded. This host needs network access to the DuckDB extension "
            "repository, or a proxy set in http_proxy / HTTP_PROXY."
        ) from exc
    con.execute("LOAD fts")
