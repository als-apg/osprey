"""
Tests for Middle Layer React Agent Pipeline

Tests the MML-style channel finder with React agent and database query tools.
"""

import pytest

from osprey.services.channel_finder.databases.middle_layer import (
    MiddleLayerDatabase,
)

# === Database Tests ===


def test_middle_layer_database_load(sample_middle_layer_db_path) -> None:
    """Test that middle layer database loads correctly."""
    db = MiddleLayerDatabase(sample_middle_layer_db_path)

    # Check database loaded
    assert db.data is not None
    assert "SR" in db.data
    assert "BPM" in db.data["SR"]

    # Check channel map built
    assert len(db.channel_map) > 0
    assert "SR01C:BPM1:X" in db.channel_map


def test_middle_layer_database_list_systems(sample_middle_layer_db_path) -> None:
    """Test listing systems with descriptions."""
    db = MiddleLayerDatabase(sample_middle_layer_db_path)

    systems = db.list_systems()
    system_names = [s["name"] for s in systems]
    assert "SR" in system_names
    assert "BR" in system_names
    assert "BTS" in system_names

    # Check structure
    for system in systems:
        assert "name" in system
        assert "description" in system
        assert isinstance(system["description"], str)  # Can be empty string


def test_middle_layer_database_list_families(sample_middle_layer_db_path) -> None:
    """Test listing families in a system with descriptions."""
    db = MiddleLayerDatabase(sample_middle_layer_db_path)

    families = db.list_families("SR")
    family_names = [f["name"] for f in families]
    assert "BPM" in family_names
    assert "HCM" in family_names
    assert "VCM" in family_names
    assert "DCCT" in family_names

    # Check structure
    for family in families:
        assert "name" in family
        assert "description" in family
        assert isinstance(family["description"], str)  # Can be empty string


def test_middle_layer_database_inspect_fields(sample_middle_layer_db_path) -> None:
    """Test inspecting field structure with descriptions."""
    db = MiddleLayerDatabase(sample_middle_layer_db_path)

    # Top-level fields
    fields = db.inspect_fields("SR", "BPM")
    assert "Monitor" in fields
    assert "Setpoint" in fields
    assert "setup" in fields

    # Check structure - fields now return dicts with type and description
    for _field_name, field_info in fields.items():
        assert "type" in field_info
        assert "description" in field_info
        assert isinstance(field_info["description"], str)  # Can be empty string

    # Subfields
    subfields = db.inspect_fields("SR", "BPM", "Setpoint")
    assert "X" in subfields
    assert "Y" in subfields

    # Check subfield structure
    for _subfield_name, subfield_info in subfields.items():
        assert "type" in subfield_info
        assert "description" in subfield_info


def test_middle_layer_database_list_channel_names(sample_middle_layer_db_path) -> None:
    """Test retrieving channel names."""
    db = MiddleLayerDatabase(sample_middle_layer_db_path)

    # Simple field
    channels = db.list_channel_names("SR", "BPM", "Monitor")
    assert "SR01C:BPM1:X" in channels
    assert "SR01C:BPM1:Y" in channels
    assert len(channels) == 8

    # Subfield
    channels = db.list_channel_names("SR", "BPM", "Setpoint", subfield="X")
    assert "SR01C:BPM1:XSet" in channels
    assert "SR01C:BPM2:XSet" in channels
    assert len(channels) == 4


def test_middle_layer_database_sector_filtering(sample_middle_layer_db_path) -> None:
    """Test filtering by sectors."""
    db = MiddleLayerDatabase(sample_middle_layer_db_path)

    # Filter to sector 1 only - use Setpoint/X which has one channel per device
    channels = db.list_channel_names("SR", "BPM", "Setpoint", subfield="X", sectors=[1])
    assert "SR01C:BPM1:XSet" in channels
    assert "SR01C:BPM2:XSet" in channels
    # Should not include sector 2
    assert "SR02C:BPM1:XSet" not in channels
    assert len(channels) == 2


def test_middle_layer_database_device_filtering(sample_middle_layer_db_path) -> None:
    """Test filtering by devices."""
    db = MiddleLayerDatabase(sample_middle_layer_db_path)

    # Filter to device 1 only - use Setpoint/X which has one channel per device
    channels = db.list_channel_names("SR", "BPM", "Setpoint", subfield="X", devices=[1])
    assert "SR01C:BPM1:XSet" in channels
    assert "SR02C:BPM1:XSet" in channels
    # Should not include device 2
    assert "SR01C:BPM2:XSet" not in channels
    assert len(channels) == 2


def test_middle_layer_database_combined_filtering(sample_middle_layer_db_path) -> None:
    """Test filtering by both sectors and devices."""
    db = MiddleLayerDatabase(sample_middle_layer_db_path)

    # Filter to sector 1, device 1 only - use Setpoint/X which has one channel per device
    channels = db.list_channel_names(
        "SR", "BPM", "Setpoint", subfield="X", sectors=[1], devices=[1]
    )
    assert "SR01C:BPM1:XSet" in channels
    assert len(channels) == 1


def test_middle_layer_database_validation(sample_middle_layer_db_path) -> None:
    """Test channel validation."""
    db = MiddleLayerDatabase(sample_middle_layer_db_path)

    # Valid channels
    assert db.validate_channel("SR01C:BPM1:X") is True
    assert db.validate_channel("SR:DCCT:Current") is True

    # Invalid channels
    assert db.validate_channel("INVALID:PV") is False
    assert db.validate_channel("SR01C:BPM999:X") is False


def test_middle_layer_database_get_channel(sample_middle_layer_db_path) -> None:
    """Test getting channel metadata."""
    db = MiddleLayerDatabase(sample_middle_layer_db_path)

    channel = db.get_channel("SR01C:BPM1:X")
    assert channel is not None
    assert channel["channel"] == "SR01C:BPM1:X"
    assert channel["address"] == "SR01C:BPM1:X"
    assert channel["system"] == "SR"
    assert channel["family"] == "BPM"
    assert channel["field"] == "Monitor"


def test_middle_layer_database_statistics(sample_middle_layer_db_path) -> None:
    """Test database statistics."""
    db = MiddleLayerDatabase(sample_middle_layer_db_path)

    stats = db.get_statistics()
    assert stats["format"] == "middle_layer"
    assert stats["total_channels"] > 0
    assert stats["systems"] == 3  # SR, BR, BTS
    assert stats["families"] > 0


def test_middle_layer_database_error_handling(sample_middle_layer_db_path) -> None:
    """Test error handling for invalid queries."""
    db = MiddleLayerDatabase(sample_middle_layer_db_path)

    # Invalid system
    with pytest.raises(ValueError, match="System 'INVALID' not found"):
        db.list_families("INVALID")

    # Invalid family
    with pytest.raises(ValueError, match="Family 'INVALID' not found"):
        db.inspect_fields("SR", "INVALID")

    # Invalid field
    with pytest.raises(ValueError, match="Field 'INVALID' not found"):
        db.list_channel_names("SR", "BPM", "INVALID")

    # Invalid subfield
    with pytest.raises(ValueError, match="Subfield 'INVALID' not found"):
        db.list_channel_names("SR", "BPM", "Setpoint", subfield="INVALID")


# === Fixtures ===


def test_middle_layer_database_descriptions_optional(sample_middle_layer_db_path) -> None:
    """Test that descriptions are optional and system works without them."""
    db = MiddleLayerDatabase(sample_middle_layer_db_path)

    # VCM has no descriptions in test data - should still work
    families = db.list_families("SR")
    vcm_family = [f for f in families if f["name"] == "VCM"][0]
    assert vcm_family["description"] == ""  # Empty string, not missing

    # Fields without descriptions should also work
    fields = db.inspect_fields("SR", "VCM")
    for field_info in fields.values():
        assert "description" in field_info
        # Can be empty string


def test_middle_layer_database_descriptions_present(sample_middle_layer_db_path) -> None:
    """Test that descriptions are extracted when present."""
    db = MiddleLayerDatabase(sample_middle_layer_db_path)

    # BPM has descriptions in test data
    families = db.list_families("SR")
    bpm_family = [f for f in families if f["name"] == "BPM"][0]
    assert bpm_family["description"] != ""  # Should have description
    assert "position" in bpm_family["description"].lower()  # Content check

    # Check field-level descriptions
    fields = db.inspect_fields("SR", "BPM")
    monitor_field = fields.get("Monitor")
    assert monitor_field is not None
    assert monitor_field["description"] != ""  # Should have description


@pytest.fixture
def sample_middle_layer_db_path(tmp_path) -> str:
    """Create a sample middle layer database for testing."""
    db_file = tmp_path / "test_middle_layer.json"

    # Create a minimal test database
    import json

    test_data = {
        "SR": {
            "_description": "Storage Ring - main synchrotron light source",
            "BPM": {
                "_description": "Beam Position Monitors - measure beam X/Y position",
                "Monitor": {
                    "_description": "Position readback values",
                    "ChannelNames": [
                        "SR01C:BPM1:X",
                        "SR01C:BPM1:Y",
                        "SR01C:BPM2:X",
                        "SR01C:BPM2:Y",
                        "SR02C:BPM1:X",
                        "SR02C:BPM1:Y",
                        "SR02C:BPM2:X",
                        "SR02C:BPM2:Y",
                    ],
                },
                "Setpoint": {
                    "X": {
                        "ChannelNames": [
                            "SR01C:BPM1:XSet",
                            "SR01C:BPM2:XSet",
                            "SR02C:BPM1:XSet",
                            "SR02C:BPM2:XSet",
                        ]
                    },
                    "Y": {
                        "ChannelNames": [
                            "SR01C:BPM1:YSet",
                            "SR01C:BPM2:YSet",
                            "SR02C:BPM1:YSet",
                            "SR02C:BPM2:YSet",
                        ]
                    },
                },
                "setup": {
                    "CommonNames": ["BPM 1-1", "BPM 1-2", "BPM 2-1", "BPM 2-2"],
                    "DeviceList": [[1, 1], [1, 2], [2, 1], [2, 2]],
                },
            },
            "HCM": {
                "_description": "Horizontal Corrector Magnets",
                "Monitor": {
                    "_description": "Current readback in Amperes",
                    "ChannelNames": [
                        "SR01C:HCM1:Current",
                        "SR01C:HCM2:Current",
                    ],
                },
                "Setpoint": {
                    "ChannelNames": [
                        "SR01C:HCM1:SetCurrent",
                        "SR01C:HCM2:SetCurrent",
                    ]
                },
                "setup": {
                    "CommonNames": ["H Corrector 1", "H Corrector 2"],
                    "DeviceList": [[1, 1], [1, 2]],
                },
            },
            "VCM": {
                "Monitor": {
                    "ChannelNames": [
                        "SR01C:VCM1:Current",
                        "SR01C:VCM2:Current",
                    ]
                },
                "Setpoint": {
                    "ChannelNames": [
                        "SR01C:VCM1:SetCurrent",
                        "SR01C:VCM2:SetCurrent",
                    ]
                },
                "setup": {
                    "CommonNames": ["V Corrector 1", "V Corrector 2"],
                },
            },
            "DCCT": {
                "Monitor": {"ChannelNames": ["SR:DCCT:Current"]},
                "setup": {"CommonNames": ["Beam Current Monitor"]},
            },
        },
        "BR": {
            "_description": "Booster Ring",
            "BPM": {
                "Monitor": {"ChannelNames": ["BR:BPM1:X", "BR:BPM1:Y"]},
                "setup": {"CommonNames": ["BR BPM 1"]},
            },
            "DCCT": {
                "Monitor": {"ChannelNames": ["BR:DCCT:Current"]},
                "setup": {"CommonNames": ["BR Beam Current"]},
            },
        },
        "BTS": {
            "BPM": {
                "Monitor": {"ChannelNames": ["BTS:BPM1:X", "BTS:BPM1:Y"]},
                "setup": {"CommonNames": ["BTS BPM 1"]},
            }
        },
    }

    with open(db_file, "w") as f:
        json.dump(test_data, f, indent=2)

    return str(db_file)
