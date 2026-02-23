"""
Abstract base class for all database implementations.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path


class DatabaseWriteError(Exception):
    """Raised on invalid database write operations."""

    def __init__(self, message: str, error_type: str = "write_error"):
        super().__init__(message)
        self.error_type = error_type


class BaseDatabase(ABC):
    """Abstract base class for all channel databases."""

    def __init__(self, db_path: str):
        """
        Initialize database.

        Args:
            db_path: Path to database file
        """
        self.db_path = db_path
        self.load_database()

    @abstractmethod
    def load_database(self):
        """Load database from file."""
        pass

    @abstractmethod
    def _serialize(self) -> dict | list:
        """Serialize in-memory state back to JSON-compatible structure.

        This is the inverse of load_database(): it returns the data
        that should be written to the database file on disk.
        """
        pass

    def _persist(self) -> None:
        """Write current in-memory state to disk atomically."""
        self._atomic_write(Path(self.db_path), self._serialize())

    @staticmethod
    def _atomic_write(path: Path, data: dict | list) -> None:
        """Write JSON atomically: backup current file, write via temp, os.replace.

        Creates a ``.json.bak`` backup of the existing file before overwriting.
        """
        path = Path(path)

        # Create backup if file exists
        if path.exists():
            bak = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, bak)

        # Write to temp file in same directory, then atomic replace
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @abstractmethod
    def get_channel(self, channel_name: str) -> dict | None:
        """
        Get channel information by name.

        Args:
            channel_name: Channel name to lookup

        Returns:
            Channel dict or None if not found
        """
        pass

    @abstractmethod
    def get_all_channels(self) -> list[dict]:
        """
        Get all channels in the database.

        Returns:
            List of channel dictionaries
        """
        pass

    @abstractmethod
    def validate_channel(self, channel_name: str) -> bool:
        """
        Check if a channel exists.

        Args:
            channel_name: Channel name to validate

        Returns:
            True if channel exists, False otherwise
        """
        pass

    @abstractmethod
    def get_statistics(self) -> dict:
        """
        Get database statistics.

        Returns:
            Dict with statistics (total_channels, etc.)
        """
        pass
