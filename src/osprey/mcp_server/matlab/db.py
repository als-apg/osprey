"""Database connection and path management for MATLAB MML."""

import os
import sqlite3
from pathlib import Path

DEFAULT_DB_DIR = Path.home() / ".matlab-mml"
DEFAULT_DB_NAME = "mml.db"


def get_db_path() -> Path:
    """Return the SQLite database path.

    Resolution order:
      1. ``MATLAB_MML_DB`` environment variable
      2. ``~/.matlab-mml/mml.db``
    """
    env = os.environ.get("MATLAB_MML_DB")
    if env:
        return Path(env)
    return DEFAULT_DB_DIR / DEFAULT_DB_NAME


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode and Row factory.

    Args:
        db_path: Explicit path. Falls back to ``get_db_path()`` if None.

    Returns:
        Configured sqlite3.Connection.
    """
    path = db_path or get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn
