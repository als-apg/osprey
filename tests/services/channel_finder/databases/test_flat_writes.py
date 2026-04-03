"""Tests for flat ChannelDatabase write methods."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.channel_finder.core.base_database import DatabaseWriteError
from osprey.services.channel_finder.databases.flat import ChannelDatabase


@pytest.fixture()
def flat_db(tmp_path: Path) -> ChannelDatabase:
    """Create a flat database with two channels."""
    db_path = tmp_path / "flat.json"
    data = [
        {"channel": "SR:BPM:01:X", "address": "SR:BPM:01:X", "description": "BPM X"},
        {"channel": "SR:BPM:01:Y", "address": "SR:BPM:01:Y", "description": "BPM Y"},
    ]
    db_path.write_text(json.dumps(data, indent=2))
    return ChannelDatabase(str(db_path))


class TestFlatSerialize:
    """Tests for _serialize()."""

    def test_serialize_returns_channels_list(self, flat_db: ChannelDatabase):
        result = flat_db._serialize()
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["channel"] == "SR:BPM:01:X"


class TestFlatAddChannel:
    """Tests for add_channel()."""

    def test_add_channel(self, flat_db: ChannelDatabase):
        result = flat_db.add_channel("SR:MAG:01", address="SR:MAG:01", description="Magnet 1")

        assert result["success"] is True
        assert result["channel"] == "SR:MAG:01"
        # In-memory state updated
        assert flat_db.validate_channel("SR:MAG:01")
        assert len(flat_db.channels) == 3
        # Persisted to disk
        on_disk = json.loads(Path(flat_db.db_path).read_text())
        names = [ch["channel"] for ch in on_disk]
        assert "SR:MAG:01" in names

    def test_add_channel_default_address(self, flat_db: ChannelDatabase):
        flat_db.add_channel("TEST:CH")
        entry = flat_db.get_channel("TEST:CH")
        assert entry["address"] == "TEST:CH"

    def test_add_duplicate_raises(self, flat_db: ChannelDatabase):
        with pytest.raises(DatabaseWriteError, match="already exists"):
            flat_db.add_channel("SR:BPM:01:X")


class TestFlatDeleteChannel:
    """Tests for delete_channel()."""

    def test_delete_channel(self, flat_db: ChannelDatabase):
        result = flat_db.delete_channel("SR:BPM:01:X")

        assert result["success"] is True
        assert not flat_db.validate_channel("SR:BPM:01:X")
        assert flat_db.validate_channel("SR:BPM:01:Y")
        assert len(flat_db.channels) == 1
        # Persisted
        on_disk = json.loads(Path(flat_db.db_path).read_text())
        assert len(on_disk) == 1

    def test_delete_nonexistent_raises(self, flat_db: ChannelDatabase):
        with pytest.raises(DatabaseWriteError, match="not found"):
            flat_db.delete_channel("NONEXISTENT")


class TestFlatUpdateChannel:
    """Tests for update_channel()."""

    def test_update_description(self, flat_db: ChannelDatabase):
        result = flat_db.update_channel("SR:BPM:01:X", new_description="Updated")

        assert result["success"] is True
        entry = flat_db.get_channel("SR:BPM:01:X")
        assert entry["description"] == "Updated"
        # Persisted
        on_disk = json.loads(Path(flat_db.db_path).read_text())
        ch = next(c for c in on_disk if c["channel"] == "SR:BPM:01:X")
        assert ch["description"] == "Updated"

    def test_update_address(self, flat_db: ChannelDatabase):
        flat_db.update_channel("SR:BPM:01:X", new_address="NEW:ADDR")
        entry = flat_db.get_channel("SR:BPM:01:X")
        assert entry["address"] == "NEW:ADDR"

    def test_update_both(self, flat_db: ChannelDatabase):
        flat_db.update_channel("SR:BPM:01:X", new_description="New", new_address="New:Addr")
        entry = flat_db.get_channel("SR:BPM:01:X")
        assert entry["description"] == "New"
        assert entry["address"] == "New:Addr"

    def test_update_nonexistent_raises(self, flat_db: ChannelDatabase):
        with pytest.raises(DatabaseWriteError, match="not found"):
            flat_db.update_channel("NONEXISTENT", new_description="nope")
