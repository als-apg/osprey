"""Tests for MiddleLayerDatabase write methods."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.channel_finder.core.base_database import DatabaseWriteError
from osprey.services.channel_finder.databases.middle_layer import MiddleLayerDatabase


@pytest.fixture()
def ml_db(tmp_path: Path) -> MiddleLayerDatabase:
    """Create a middle layer database instance."""
    db_path = tmp_path / "ml.json"
    data = {
        "SR": {
            "_description": "Storage Ring",
            "BPM": {
                "_description": "BPM family",
                "Monitor": {
                    "ChannelNames": ["SR01:BPM:X", "SR01:BPM:Y"],
                    "DataType": "double",
                },
                "Setpoint": {
                    "X": {
                        "ChannelNames": ["SR01:BPM:XSet"],
                    },
                },
                "setup": {"DeviceList": [[1, 1]]},
            },
        },
    }
    db_path.write_text(json.dumps(data, indent=2))
    return MiddleLayerDatabase(str(db_path))


class TestMLSerialize:
    def test_serialize_returns_full_tree(self, ml_db: MiddleLayerDatabase):
        result = ml_db._serialize()
        assert isinstance(result, dict)
        assert "SR" in result
        assert "BPM" in result["SR"]


class TestMLAddFamily:
    def test_add_family(self, ml_db: MiddleLayerDatabase):
        result = ml_db.add_family("SR", "CORR", description="Corrector")

        assert result["success"] is True
        assert "CORR" in ml_db.data["SR"]
        assert ml_db.data["SR"]["CORR"]["_description"] == "Corrector"
        # Persisted
        on_disk = json.loads(Path(ml_db.db_path).read_text())
        assert "CORR" in on_disk["SR"]

    def test_add_duplicate_raises(self, ml_db: MiddleLayerDatabase):
        with pytest.raises(DatabaseWriteError, match="already exists"):
            ml_db.add_family("SR", "BPM")


class TestMLDeleteFamily:
    def test_delete_family_cascades(self, ml_db: MiddleLayerDatabase):
        result = ml_db.delete_family("SR", "BPM")

        assert result["success"] is True
        assert result["affected_channels"] >= 3
        assert "BPM" not in ml_db.data["SR"]
        # Channel map rebuilt
        assert not ml_db.validate_channel("SR01:BPM:X")

    def test_delete_nonexistent_raises(self, ml_db: MiddleLayerDatabase):
        with pytest.raises(DatabaseWriteError, match="not found"):
            ml_db.delete_family("SR", "NONEXISTENT")


class TestMLAddChannel:
    def test_add_channel(self, ml_db: MiddleLayerDatabase):
        result = ml_db.add_channel("SR", "BPM", "Monitor", "SR02:BPM:X")

        assert result["success"] is True
        assert "SR02:BPM:X" in ml_db.data["SR"]["BPM"]["Monitor"]["ChannelNames"]
        assert ml_db.validate_channel("SR02:BPM:X")

    def test_add_channel_creates_field(self, ml_db: MiddleLayerDatabase):
        result = ml_db.add_channel("SR", "BPM", "NewField", "SR:BPM:NEW")

        assert result["success"] is True
        assert ml_db.data["SR"]["BPM"]["NewField"]["ChannelNames"] == ["SR:BPM:NEW"]

    def test_add_duplicate_channel_raises(self, ml_db: MiddleLayerDatabase):
        with pytest.raises(DatabaseWriteError, match="already exists"):
            ml_db.add_channel("SR", "BPM", "Monitor", "SR01:BPM:X")


class TestMLDeleteChannel:
    def test_delete_channel(self, ml_db: MiddleLayerDatabase):
        result = ml_db.delete_channel("SR", "BPM", "Monitor", "SR01:BPM:X")

        assert result["success"] is True
        assert "SR01:BPM:X" not in ml_db.data["SR"]["BPM"]["Monitor"]["ChannelNames"]
        assert not ml_db.validate_channel("SR01:BPM:X")

    def test_delete_nonexistent_raises(self, ml_db: MiddleLayerDatabase):
        with pytest.raises(DatabaseWriteError, match="not found"):
            ml_db.delete_channel("SR", "BPM", "Monitor", "NONEXISTENT")


class TestMLCountFamilyChannels:
    def test_count(self, ml_db: MiddleLayerDatabase):
        count = ml_db.count_family_channels("SR", "BPM")
        assert count >= 3  # SR01:BPM:X, SR01:BPM:Y, SR01:BPM:XSet
