"""Tests for BaseDatabase persistence infrastructure (_atomic_write)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.channel_finder.core.base_database import BaseDatabase


class TestAtomicWrite:
    """Tests for atomic file write with backup."""

    def test_atomic_write_creates_backup(self, tmp_path: Path):
        f = tmp_path / "test.json"
        f.write_text('{"old": true}')

        BaseDatabase._atomic_write(f, {"new": True})

        assert f.exists()
        assert json.loads(f.read_text()) == {"new": True}
        bak = tmp_path / "test.json.bak"
        assert bak.exists()
        assert json.loads(bak.read_text()) == {"old": True}

    def test_atomic_write_new_file(self, tmp_path: Path):
        f = tmp_path / "new.json"
        BaseDatabase._atomic_write(f, {"created": True})

        assert f.exists()
        assert json.loads(f.read_text()) == {"created": True}
        # No backup for new file
        assert not (tmp_path / "new.json.bak").exists()

    def test_atomic_write_preserves_on_error(self, tmp_path: Path):
        f = tmp_path / "safe.json"
        original = {"safe": True}
        f.write_text(json.dumps(original))

        # Force an error during write by providing an unserializable object
        class BadObj:
            pass

        with pytest.raises(TypeError):
            BaseDatabase._atomic_write(f, BadObj())

        # Original file untouched
        assert json.loads(f.read_text()) == original

    def test_atomic_write_writes_list(self, tmp_path: Path):
        f = tmp_path / "list.json"
        data = [{"channel": "ch1"}, {"channel": "ch2"}]
        BaseDatabase._atomic_write(f, data)

        assert json.loads(f.read_text()) == data

    def test_atomic_write_trailing_newline(self, tmp_path: Path):
        f = tmp_path / "newline.json"
        BaseDatabase._atomic_write(f, {"key": "val"})

        assert f.read_text().endswith("\n")
