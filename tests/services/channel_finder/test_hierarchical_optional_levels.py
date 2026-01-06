"""
Tests for hierarchical channel finder with optional/unused levels support.

Tests Phase 1 feature: Allow hierarchy levels that don't appear in naming pattern.
"""

import json
import tempfile
from pathlib import Path

import pytest
from src.osprey.templates.apps.control_assistant.services.channel_finder.databases.hierarchical import (
    HierarchicalChannelDatabase,
)


@pytest.fixture
def jlab_style_db_content():
    """
    JLab-style database: semantic hierarchy with PV names at leaf.

    Hierarchy has 5 levels but naming pattern only uses {pv}.
    Tests navigation-only levels.
    """
    return {
        "hierarchy": {
            "levels": [
                {"name": "system", "type": "tree"},
                {"name": "family", "type": "tree"},
                {"name": "sector", "type": "tree"},
                {"name": "device", "type": "tree"},
                {"name": "pv", "type": "tree"},
            ],
            "naming_pattern": "{pv}",
        },
        "tree": {
            "Magnets": {
                "_description": "Magnet system",
                "Quads": {
                    "_description": "Quadrupole magnets",
                    "1L": {
                        "_description": "North Linac",
                        "MQ1L01": {
                            "_description": "Quad device 1",
                            "MQ1L01.S": {"_description": "Setpoint"},
                            "MQ1L01M": {"_description": "Readback"},
                        },
                    },
                },
            },
            "Diagnostics": {
                "_description": "Diagnostic system",
                "BPMs": {
                    "_description": "Beam position monitors",
                    "1L": {
                        "_description": "North Linac",
                        "IPM1L01": {
                            "_description": "BPM device 1",
                            "IPM1L01X": {"_description": "X position"},
                            "IPM1L01Y": {"_description": "Y position"},
                        },
                    },
                },
            },
        },
    }


@pytest.fixture
def partial_pattern_db_content():
    """
    Database with some levels used in pattern, some not.

    Tests mixed usage of hierarchy levels.
    """
    return {
        "hierarchy": {
            "levels": [
                {"name": "system", "type": "tree"},
                {"name": "subsystem", "type": "tree"},
                {"name": "location", "type": "tree"},
                {"name": "device", "type": "instances"},
                {"name": "signal", "type": "tree"},
            ],
            "naming_pattern": "{system}-{subsystem}:{device}:{signal}",
        },
        "tree": {
            "RF": {
                "_description": "RF system",
                "Cavities": {
                    "_description": "RF cavities",
                    "Hall-A": {
                        "_description": "Hall A location (not in pattern)",
                        "DEVICE": {
                            "_expansion": {
                                "_type": "range",
                                "_pattern": "CAV{:02d}",
                                "_range": [1, 2],
                            },
                            "Voltage": {"_description": "Cavity voltage"},
                            "Phase": {"_description": "Cavity phase"},
                        },
                    },
                },
            }
        },
    }


@pytest.fixture
def jlab_db(jlab_style_db_content):
    """Create temporary JLab-style database."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(jlab_style_db_content, f)
        db_path = f.name

    db = HierarchicalChannelDatabase(db_path)
    yield db

    # Cleanup
    Path(db_path).unlink()


@pytest.fixture
def partial_pattern_db(partial_pattern_db_content):
    """Create temporary partial-pattern database."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(partial_pattern_db_content, f)
        db_path = f.name

    db = HierarchicalChannelDatabase(db_path)
    yield db

    # Cleanup
    Path(db_path).unlink()


class TestOptionalLevelsValidation:
    """Test validation logic for optional/unused levels."""

    def test_jlab_style_loads_successfully(self, jlab_db):
        """JLab-style database should load without validation errors."""
        assert jlab_db is not None
        assert jlab_db.naming_pattern == "{pv}"
        assert len(jlab_db.hierarchy_levels) == 5

    def test_partial_pattern_loads_successfully(self, partial_pattern_db):
        """Partial pattern database should load without validation errors."""
        assert partial_pattern_db is not None
        assert partial_pattern_db.naming_pattern == "{system}-{subsystem}:{device}:{signal}"
        assert len(partial_pattern_db.hierarchy_levels) == 5

    def test_undefined_level_in_pattern_raises_error(self):
        """Pattern referencing undefined level should raise error."""
        invalid_content = {
            "hierarchy": {
                "levels": [{"name": "system", "type": "tree"}, {"name": "device", "type": "tree"}],
                "naming_pattern": "{system}:{undefined_level}",
            },
            "tree": {"SYS": {"_description": "System", "DEV": {"_description": "Device"}}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_content, f)
            db_path = f.name

        try:
            with pytest.raises(ValueError, match="undefined hierarchy levels"):
                HierarchicalChannelDatabase(db_path)
        finally:
            Path(db_path).unlink()


class TestGetPatternLevels:
    """Test _get_pattern_levels helper method."""

    def test_jlab_pattern_levels(self, jlab_db):
        """JLab pattern should extract only 'pv' level."""
        pattern_levels = jlab_db._get_pattern_levels()
        assert pattern_levels == ["pv"]

    def test_partial_pattern_levels(self, partial_pattern_db):
        """Partial pattern should extract correct subset."""
        pattern_levels = partial_pattern_db._get_pattern_levels()
        # Should be in hierarchy order, not pattern order
        assert set(pattern_levels) == {"system", "subsystem", "device", "signal"}
        # Verify 'location' is NOT in pattern levels
        assert "location" not in pattern_levels

    def test_pattern_levels_order(self, partial_pattern_db):
        """Pattern levels should be ordered by hierarchy, not pattern appearance."""
        pattern_levels = partial_pattern_db._get_pattern_levels()
        # Hierarchy order: system, subsystem, location, device, signal
        # Pattern uses: system, subsystem, device, signal (skips location)
        # Should return in hierarchy order
        assert pattern_levels == ["system", "subsystem", "device", "signal"]


class TestChannelBuilding:
    """Test channel name building with optional levels."""

    def test_jlab_channel_building(self, jlab_db):
        """JLab-style should build PV names directly."""
        selections = {
            "system": "Magnets",
            "family": "Quads",
            "sector": "1L",
            "device": "MQ1L01",
            "pv": ["MQ1L01.S", "MQ1L01M"],
        }

        channels = jlab_db.build_channels_from_selections(selections)
        assert set(channels) == {"MQ1L01.S", "MQ1L01M"}

    def test_partial_pattern_channel_building(self, partial_pattern_db):
        """Partial pattern should use only pattern levels."""
        selections = {
            "system": "RF",
            "subsystem": "Cavities",
            "location": "Hall-A",  # Not in pattern - should be ignored
            "device": ["CAV01", "CAV02"],
            "signal": "Voltage",
        }

        channels = partial_pattern_db.build_channels_from_selections(selections)
        # Pattern: {system}-{subsystem}:{device}:{signal}
        # location is ignored
        assert set(channels) == {"RF-Cavities:CAV01:Voltage", "RF-Cavities:CAV02:Voltage"}

    def test_channel_map_generation_jlab(self, jlab_db):
        """Channel map should generate correct PV names for JLab style."""
        channel_map = jlab_db.channel_map

        # Should have 4 channels total
        assert len(channel_map) == 4

        # Verify specific channels exist
        assert "MQ1L01.S" in channel_map
        assert "MQ1L01M" in channel_map
        assert "IPM1L01X" in channel_map
        assert "IPM1L01Y" in channel_map

        # Verify path is complete (includes all hierarchy levels)
        mq_path = channel_map["MQ1L01.S"]["path"]
        assert mq_path["system"] == "Magnets"
        assert mq_path["family"] == "Quads"
        assert mq_path["sector"] == "1L"
        assert mq_path["device"] == "MQ1L01"
        assert mq_path["pv"] == "MQ1L01.S"

    def test_channel_map_generation_partial(self, partial_pattern_db):
        """Channel map should generate correct names with partial pattern."""
        channel_map = partial_pattern_db.channel_map

        # Should have 4 channels (2 devices × 2 signals)
        assert len(channel_map) == 4

        # Verify channels use pattern format
        assert "RF-Cavities:CAV01:Voltage" in channel_map
        assert "RF-Cavities:CAV01:Phase" in channel_map
        assert "RF-Cavities:CAV02:Voltage" in channel_map
        assert "RF-Cavities:CAV02:Phase" in channel_map

        # Verify path includes location even though it's not in pattern
        cav_path = channel_map["RF-Cavities:CAV01:Voltage"]["path"]
        assert cav_path["system"] == "RF"
        assert cav_path["subsystem"] == "Cavities"
        assert cav_path["location"] == "Hall-A"  # Preserved in path
        assert cav_path["device"] == "CAV01"
        assert cav_path["signal"] == "Voltage"


class TestChannelValidation:
    """Test channel validation with optional levels."""

    def test_jlab_validate_existing_channel(self, jlab_db):
        """Should validate channels that exist."""
        assert jlab_db.validate_channel("MQ1L01.S") is True
        assert jlab_db.validate_channel("IPM1L01X") is True

    def test_jlab_validate_nonexistent_channel(self, jlab_db):
        """Should reject channels that don't exist."""
        assert jlab_db.validate_channel("MQ1L01.FAKE") is False
        assert jlab_db.validate_channel("Magnets:Quads:MQ1L01.S") is False

    def test_partial_validate_channels(self, partial_pattern_db):
        """Should validate channels built from pattern."""
        assert partial_pattern_db.validate_channel("RF-Cavities:CAV01:Voltage") is True
        assert partial_pattern_db.validate_channel("RF-Cavities:CAV01:FAKE") is False


class TestNavigationWithOptionalLevels:
    """Test hierarchy navigation with optional levels."""

    def test_get_options_at_level_jlab(self, jlab_db):
        """Navigation should work through all levels even if not in pattern."""
        # System level
        options = jlab_db.get_options_at_level("system", {})
        assert len(options) == 2
        assert any(opt["name"] == "Magnets" for opt in options)
        assert any(opt["name"] == "Diagnostics" for opt in options)

        # Family level (after selecting system)
        options = jlab_db.get_options_at_level("family", {"system": "Magnets"})
        assert len(options) == 1
        assert options[0]["name"] == "Quads"

        # Sector level
        options = jlab_db.get_options_at_level("sector", {"system": "Magnets", "family": "Quads"})
        assert len(options) == 1
        assert options[0]["name"] == "1L"

        # Device level
        options = jlab_db.get_options_at_level(
            "device", {"system": "Magnets", "family": "Quads", "sector": "1L"}
        )
        assert len(options) == 1
        assert options[0]["name"] == "MQ1L01"

        # PV level (leaf)
        options = jlab_db.get_options_at_level(
            "pv", {"system": "Magnets", "family": "Quads", "sector": "1L", "device": "MQ1L01"}
        )
        assert len(options) == 2
        assert any(opt["name"] == "MQ1L01.S" for opt in options)
        assert any(opt["name"] == "MQ1L01M" for opt in options)

    def test_get_options_partial_pattern(self, partial_pattern_db):
        """Navigation should include location level even though not in pattern."""
        options = partial_pattern_db.get_options_at_level(
            "location", {"system": "RF", "subsystem": "Cavities"}
        )
        assert len(options) == 1
        assert options[0]["name"] == "Hall-A"


class TestStatistics:
    """Test statistics reporting with optional levels."""

    def test_jlab_statistics(self, jlab_db):
        """Statistics should report all channels correctly."""
        stats = jlab_db.get_statistics()
        assert stats["total_channels"] == 4
        assert stats["hierarchy_levels"] == ["system", "family", "sector", "device", "pv"]

    def test_partial_statistics(self, partial_pattern_db):
        """Statistics should work with partial patterns."""
        stats = partial_pattern_db.get_statistics()
        assert stats["total_channels"] == 4
        assert stats["hierarchy_levels"] == ["system", "subsystem", "location", "device", "signal"]


class TestChannelPartFeature:
    """Test _channel_part decoupling of tree keys from naming components."""

    def test_channel_part_override_basic(self):
        """Tree key can be different from channel name component."""
        content = {
            "hierarchy": {
                "levels": [{"name": "system", "type": "tree"}, {"name": "signal", "type": "tree"}],
                "naming_pattern": "{system}:{signal}",
            },
            "tree": {
                "Magnets": {
                    "_channel_part": "MAG",
                    "_description": "Magnet system",
                    "Current": {"_channel_part": "I", "_description": "Current signal"},
                    "Voltage": {"_channel_part": "V", "_description": "Voltage signal"},
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            assert len(db.channel_map) == 2
            # Channel names use _channel_part, not tree keys
            assert "MAG:I" in db.channel_map
            assert "MAG:V" in db.channel_map
            # Tree keys not used in channel names
            assert "Magnets:Current" not in db.channel_map
        finally:
            Path(db_path).unlink()

    def test_channel_part_mixed_usage(self):
        """Can mix _channel_part and default tree key behavior."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "device", "type": "tree"},
                    {"name": "signal", "type": "tree"},
                ],
                "naming_pattern": "{system}:{device}:{signal}",
            },
            "tree": {
                "MAG": {
                    "_description": "Uses tree key MAG",
                    "Dipole": {
                        "_channel_part": "DIP",  # Override
                        "_description": "Uses _channel_part DIP",
                        "Current": {"_description": "Uses tree key Current"},
                    },
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            assert len(db.channel_map) == 1
            # Mixed: MAG (tree key), DIP (_channel_part), Current (tree key)
            assert "MAG:DIP:Current" in db.channel_map
        finally:
            Path(db_path).unlink()

    def test_channel_part_empty_string_skips(self):
        """Empty string _channel_part means navigation-only (not in name)."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "category", "type": "tree"},
                    {"name": "system", "type": "tree"},
                    {"name": "signal", "type": "tree"},
                ],
                "naming_pattern": "{system}:{signal}",  # category not in pattern
            },
            "tree": {
                "PowerSupplies": {
                    "_channel_part": "",  # Navigation only
                    "_description": "Power supply category",
                    "MAG": {
                        "_description": "Magnet power supplies",
                        "Current": {"_description": "Current"},
                    },
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            assert len(db.channel_map) == 1
            # Category level navigated but not in name
            assert "MAG:Current" in db.channel_map

            # Path includes all levels
            channel_data = db.channel_map["MAG:Current"]
            assert channel_data["path"]["category"] == ""
            assert channel_data["path"]["system"] == "MAG"
            assert channel_data["path"]["signal"] == "Current"
        finally:
            Path(db_path).unlink()

    def test_channel_part_with_instances(self):
        """_channel_part works with instance expansion."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "device", "type": "instances"},
                    {"name": "signal", "type": "tree"},
                ],
                "naming_pattern": "{system}:{device}:{signal}",
            },
            "tree": {
                "Magnets": {
                    "_channel_part": "MAG",
                    "DEVICE": {
                        "_expansion": {"_type": "range", "_pattern": "D{:02d}", "_range": [1, 2]},
                        "Current": {"_description": "Current"},
                    },
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            assert len(db.channel_map) == 2
            assert "MAG:D01:Current" in db.channel_map
            assert "MAG:D02:Current" in db.channel_map
        finally:
            Path(db_path).unlink()

    def test_jlab_pattern_with_channel_part(self):
        """JLab pattern: friendly tree keys + _channel_part for actual PV names."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "family", "type": "tree"},
                    {"name": "location", "type": "tree"},
                    {"name": "pv", "type": "tree"},
                ],
                "naming_pattern": "{pv}",
            },
            "tree": {
                "Magnets": {
                    "_channel_part": "",  # Navigation only
                    "Skew Quads": {
                        "_channel_part": "",  # Navigation only
                        "North Linac": {
                            "_channel_part": "",  # Navigation only
                            "MQS1L02.S": {"_channel_part": "MQS1L02.S"},  # Actual PV name
                            "MQS1L02M": {"_channel_part": "MQS1L02M"},
                        },
                    },
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            assert len(db.channel_map) == 2
            assert "MQS1L02.S" in db.channel_map
            assert "MQS1L02M" in db.channel_map

            # Verify navigation path preserved
            path = db.channel_map["MQS1L02.S"]["path"]
            assert path["system"] == ""
            assert path["family"] == ""
            assert path["location"] == ""
            assert path["pv"] == "MQS1L02.S"
        finally:
            Path(db_path).unlink()

    def test_no_channel_part_uses_tree_key(self):
        """Databases without _channel_part use tree keys (backward compatible)."""
        content = {
            "hierarchy": {
                "levels": [{"name": "system", "type": "tree"}, {"name": "signal", "type": "tree"}],
                "naming_pattern": "{system}:{signal}",
            },
            "tree": {"MAG": {"Current": {}, "Voltage": {}}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            assert len(db.channel_map) == 2
            # Uses tree keys when no _channel_part specified
            assert "MAG:Current" in db.channel_map
            assert "MAG:Voltage" in db.channel_map
        finally:
            Path(db_path).unlink()


class TestBackwardCompatibility:
    """Ensure backward compatibility with existing databases."""

    def test_all_levels_in_pattern_still_works(self):
        """Traditional pattern using all levels should work as before."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "device", "type": "instances"},
                    {"name": "field", "type": "tree"},
                ],
                "naming_pattern": "{system}:{device}:{field}",
            },
            "tree": {
                "MAG": {
                    "_description": "Magnets",
                    "DEVICE": {
                        "_expansion": {"_type": "range", "_pattern": "D{:02d}", "_range": [1, 2]},
                        "Current": {"_description": "Current"},
                        "Voltage": {"_description": "Voltage"},
                    },
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            assert len(db.channel_map) == 4
            assert "MAG:D01:Current" in db.channel_map
            assert "MAG:D02:Voltage" in db.channel_map
        finally:
            Path(db_path).unlink()


class TestAutomaticLeafDetection:
    """Test automatic leaf detection for nodes without children."""

    def test_childless_node_automatic_leaf(self):
        """Nodes without children are automatically detected as leaves (no _is_leaf needed)."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "signal", "type": "tree"},
                    {"name": "suffix", "type": "tree", "optional": True},
                ],
                "naming_pattern": "{system}:{signal}_{suffix}",
            },
            "tree": {
                "MAG": {
                    "CURRENT": {
                        "_is_leaf": True,
                        "_description": "Base signal (has children, needs _is_leaf)",
                        "RB": {"_description": "Readback (no children, automatic leaf)"},
                        "SP": {"_description": "Setpoint (no children, automatic leaf)"},
                    }
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            # Should generate 3 channels: base + RB + SP (all without explicit _is_leaf on RB/SP)
            assert len(db.channel_map) == 3
            assert "MAG:CURRENT_RB" in db.channel_map
            assert "MAG:CURRENT_SP" in db.channel_map
        finally:
            Path(db_path).unlink()

    def test_only_metadata_keys_makes_automatic_leaf(self):
        """Nodes with only _metadata keys (starting with _) are automatic leaves."""
        content = {
            "hierarchy": {
                "levels": [{"name": "system", "type": "tree"}, {"name": "signal", "type": "tree"}],
                "naming_pattern": "{system}:{signal}",
            },
            "tree": {
                "MAG": {
                    "CURRENT": {
                        "_description": "Has only metadata - automatic leaf",
                        "_example": "MAG:CURRENT",
                        "_units": "Amps",
                    }
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            # Should generate channel without explicit _is_leaf
            assert len(db.channel_map) == 1
            assert "MAG:CURRENT" in db.channel_map
        finally:
            Path(db_path).unlink()


class TestExplicitIsLeafMarker:
    """Test explicit _is_leaf marker for nodes with children that are also leaves."""

    def test_basic_is_leaf_marker(self):
        """Node with _is_leaf=true generates channel even with remaining levels."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "device", "type": "tree"},
                    {"name": "signal", "type": "tree", "optional": True},
                ],
                "naming_pattern": "{system}:{device}:{signal}",
            },
            "tree": {
                "MAG": {"DEV01": {"_is_leaf": True, "_description": "Device is itself a channel"}}
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            # Should generate channel at device level (signal level optional and empty)
            assert len(db.channel_map) == 1
            assert "MAG:DEV01:" in db.channel_map or "MAG:DEV01" in db.channel_map
        finally:
            Path(db_path).unlink()

    def test_leaf_with_children(self):
        """Node with _is_leaf can also have children (suffix pattern)."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "signal", "type": "tree"},
                    {"name": "suffix", "type": "tree", "optional": True},
                ],
                "naming_pattern": "{system}:{signal}_{suffix}",
            },
            "tree": {
                "MAG": {
                    "CURRENT": {
                        "_is_leaf": True,
                        "_description": "Base current signal (explicit _is_leaf needed - has children)",
                        "RB": {"_description": "Readback (automatic leaf - no _is_leaf needed)"},
                        "SP": {"_description": "Setpoint (automatic leaf - no _is_leaf needed)"},
                    }
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            # Should generate 3 channels: base + RB + SP
            assert len(db.channel_map) == 3

            # Base signal (cleaned from "MAG:CURRENT_")
            base_channels = [
                ch for ch in db.channel_map.keys() if ch.endswith("CURRENT") or ch == "MAG:CURRENT_"
            ]
            assert len(base_channels) >= 1

            # Suffixed signals
            assert "MAG:CURRENT_RB" in db.channel_map
            assert "MAG:CURRENT_SP" in db.channel_map
        finally:
            Path(db_path).unlink()

    def test_optional_subdevice_direct_path(self):
        """Some signals skip optional subdevice level."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "device", "type": "tree"},
                    {"name": "subdevice", "type": "tree", "optional": True},
                    {"name": "signal", "type": "tree"},
                ],
                "naming_pattern": "{system}:{device}:{subdevice}:{signal}",
            },
            "tree": {
                "SYS": {
                    "DEV": {
                        "DIRECT_SIGNAL": {
                            "_description": "Signal without subdevice (automatic leaf)"
                        },
                        "SUBDEV": {
                            "_description": "Subdevice",
                            "SUB_SIGNAL": {
                                "_description": "Signal from subdevice (automatic leaf)"
                            },
                        },
                    }
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            assert len(db.channel_map) == 2

            # Direct signal (cleaned from "SYS:DEV::DIRECT_SIGNAL")
            direct = [ch for ch in db.channel_map.keys() if "DIRECT_SIGNAL" in ch]
            assert len(direct) == 1
            # Should NOT have double colons
            assert "::" not in direct[0]

            # Subdevice signal (full path)
            assert "SYS:DEV:SUBDEV:SUB_SIGNAL" in db.channel_map
        finally:
            Path(db_path).unlink()


class TestOptionalLevelsSeparatorCleanup:
    """Test separator cleanup when optional levels are skipped."""

    def test_double_colon_cleanup(self):
        """Double colons from skipped level should be cleaned to single."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "a", "type": "tree"},
                    {"name": "b", "type": "tree", "optional": True},
                    {"name": "c", "type": "tree"},
                ],
                "naming_pattern": "{a}:{b}:{c}",
            },
            "tree": {"A": {"C": {"_is_leaf": True}}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            assert len(db.channel_map) == 1
            channel = list(db.channel_map.keys())[0]
            # Should be "A:C" not "A::C"
            assert channel == "A:C"
            assert "::" not in channel
        finally:
            Path(db_path).unlink()

    def test_trailing_separator_cleanup(self):
        """Trailing separators from missing suffix should be removed."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "a", "type": "tree"},
                    {"name": "b", "type": "tree"},
                    {"name": "c", "type": "tree", "optional": True},
                ],
                "naming_pattern": "{a}:{b}_{c}",
            },
            "tree": {"A": {"B": {"_is_leaf": True}}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            assert len(db.channel_map) == 1
            channel = list(db.channel_map.keys())[0]
            # Should be "A:B" not "A:B_"
            assert channel == "A:B"
            assert not channel.endswith("_")
        finally:
            Path(db_path).unlink()

    def test_multiple_optional_levels_cleanup(self):
        """Multiple consecutive optional levels should clean up properly."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "sys", "type": "tree"},
                    {"name": "opt1", "type": "tree", "optional": True},
                    {"name": "opt2", "type": "tree", "optional": True},
                    {"name": "sig", "type": "tree"},
                ],
                "naming_pattern": "{sys}:{opt1}:{opt2}:{sig}",
            },
            "tree": {"S": {"SIG": {"_is_leaf": True}}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            assert len(db.channel_map) == 1
            channel = list(db.channel_map.keys())[0]
            # Should be "S:SIG" not "S:::SIG"
            assert channel == "S:SIG"
            assert ":::" not in channel
            assert "::" not in channel
        finally:
            Path(db_path).unlink()


class TestOptionalLevelsEdgeCases:
    """Test edge cases and complex scenarios."""

    def test_optional_level_with_instances(self):
        """Optional suffix level with instance expansion before it."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "device", "type": "instances"},
                    {"name": "signal", "type": "tree"},
                    {"name": "suffix", "type": "instances", "optional": True},
                ],
                "naming_pattern": "{system}:{device}:{signal}_{suffix}",
            },
            "tree": {
                "SYS": {
                    "DEVICE": {
                        "_expansion": {"_type": "range", "_pattern": "D{:02d}", "_range": [1, 2]},
                        "SIG1": {"_description": "Signal without suffix (automatic leaf)"},
                        "SIG2": {
                            "_description": "Signal with suffixes (no base)",
                            "SUFFIX": {"_expansion": {"_type": "list", "_instances": ["RB", "SP"]}},
                        },
                    }
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            # 2 devices × (1 sig without suffix + 1 sig with 2 suffixes) = 2 × 3 = 6 channels
            assert len(db.channel_map) == 6

            # Signals without suffix (trailing _ should be cleaned)
            sig1 = [ch for ch in db.channel_map.keys() if "SIG1" in ch]
            assert len(sig1) == 2
            for ch in sig1:
                assert not ch.endswith("_")  # Cleaned

            # Signals with suffix
            sig2_rb = [ch for ch in db.channel_map.keys() if "SIG2_RB" in ch]
            assert len(sig2_rb) == 2
            sig2_sp = [ch for ch in db.channel_map.keys() if "SIG2_SP" in ch]
            assert len(sig2_sp) == 2
        finally:
            Path(db_path).unlink()

    def test_validation_optional_not_in_pattern_fails(self):
        """Optional level must be in naming pattern."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "optional_level", "type": "tree", "optional": True},
                    {"name": "signal", "type": "tree"},
                ],
                "naming_pattern": "{system}:{signal}",  # Missing optional_level!
            },
            "tree": {"SYS": {"SIG": {}}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            with pytest.raises(ValueError, match="Optional level .* must appear in naming_pattern"):
                HierarchicalChannelDatabase(db_path)
        finally:
            Path(db_path).unlink()

    def test_mixed_optional_and_required(self):
        """Mix of optional and required levels in realistic scenario."""
        content = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "subsystem", "type": "tree"},
                    {"name": "device", "type": "instances"},
                    {"name": "subdevice", "type": "tree", "optional": True},
                    {"name": "signal", "type": "tree"},
                    {"name": "suffix", "type": "tree", "optional": True},
                ],
                "naming_pattern": "{system}-{subsystem}:{device}:{subdevice}:{signal}_{suffix}",
            },
            "tree": {
                "SYS": {
                    "SUB": {
                        "DEVICE": {
                            "_expansion": {"_type": "list", "_instances": ["D01"]},
                            "SIG": {
                                "_is_leaf": True,
                                "_description": "Base signal (explicit _is_leaf - has children)",
                                "RB": {"_description": "Readback (automatic leaf)"},
                            },
                        }
                    }
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            # Should have 2 channels: base + RB
            assert len(db.channel_map) == 2

            channels = list(db.channel_map.keys())
            # Both should be cleaned (no :: or trailing _)
            for ch in channels:
                assert "::" not in ch
                assert not (ch.endswith("_") and not ch.endswith("_RB"))
        finally:
            Path(db_path).unlink()


class TestDirectSignalsAtOptionalLevels:
    """Test that optional levels show both containers and direct signals (leaf nodes).

    This tests the behavior where at an optional level like 'subdevice', the database
    presents BOTH:
    - Container nodes (subdevices like PSU, ADC, etc.)
    - Leaf nodes (direct signals like Heartbeat, Status, etc.)

    This allows the LLM to naturally select either a subdevice to navigate deeper,
    or a direct signal that skips the optional level entirely.
    """

    @pytest.fixture
    def direct_signals_db_content(self):
        """Database with direct signals and subdevices at same level."""
        return {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "subsystem", "type": "tree"},
                    {"name": "device", "type": "instances"},
                    {"name": "subdevice", "type": "tree", "optional": True},
                    {"name": "signal", "type": "tree"},
                    {"name": "suffix", "type": "tree", "optional": True},
                ],
                "naming_pattern": "{system}:{subsystem}:{device}:{subdevice}:{signal}:{suffix}",
            },
            "tree": {
                "CTRL": {
                    "_description": "Control System",
                    "MAIN": {
                        "_description": "Main Control",
                        "DEVICE": {
                            "_expansion": {
                                "_type": "range",
                                "_pattern": "MC-{:02d}",
                                "_range": [1, 2],
                            },
                            "_description": "Main control devices",
                            # Direct signals (no subdevice)
                            "Status": {"_description": "Device status"},
                            "Heartbeat": {"_description": "Device heartbeat"},
                            # Subdevice with signals
                            "PSU": {
                                "_description": "Power Supply",
                                "Voltage": {"_description": "Output voltage"},
                                "Current": {"_description": "Output current"},
                            },
                            # Subdevice with signals
                            "ADC": {
                                "_description": "ADC",
                                "Value": {"_description": "ADC value"},
                            },
                        },
                    },
                },
            },
        }

    @pytest.fixture
    def direct_signals_db(self, direct_signals_db_content):
        """Create temporary database with direct signals."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(direct_signals_db_content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            yield db
        finally:
            Path(db_path).unlink()

    def test_optional_level_shows_both_containers_and_leaves(self, direct_signals_db):
        """Optional level should show BOTH subdevices (containers) AND direct signals (leaves)."""
        # Navigate to subdevice level
        selections = {"system": "CTRL", "subsystem": "MAIN", "device": "MC-01"}
        options = direct_signals_db.get_options_at_level("subdevice", selections)

        # Should include BOTH containers and leaf nodes
        option_names = {opt["name"] for opt in options}

        # Containers (subdevices)
        assert "PSU" in option_names, "Should include PSU subdevice container"
        assert "ADC" in option_names, "Should include ADC subdevice container"

        # Leaf nodes (direct signals)
        assert "Status" in option_names, "Should include Status direct signal"
        assert "Heartbeat" in option_names, "Should include Heartbeat direct signal"

        # Should have all 4 options
        assert len(options) == 4

    def test_direct_signal_channels_generated_correctly(self, direct_signals_db):
        """Direct signals should generate channels that skip the optional subdevice level."""
        # Build channels for direct signal
        selections = {
            "system": "CTRL",
            "subsystem": "MAIN",
            "device": "MC-01",
            "signal": "Heartbeat",
        }
        channels = direct_signals_db.build_channels_from_selections(selections)

        # Should generate channel without subdevice
        assert len(channels) == 1
        assert channels[0] == "CTRL:MAIN:MC-01:Heartbeat"

    def test_subdevice_signal_channels_generated_correctly(self, direct_signals_db):
        """Subdevice signals should generate channels with subdevice included."""
        # Build channels for subdevice signal
        selections = {
            "system": "CTRL",
            "subsystem": "MAIN",
            "device": "MC-01",
            "subdevice": "PSU",
            "signal": "Voltage",
        }
        channels = direct_signals_db.build_channels_from_selections(selections)

        # Should generate channel WITH subdevice
        assert len(channels) == 1
        assert channels[0] == "CTRL:MAIN:MC-01:PSU:Voltage"

    def test_all_channels_validate(self, direct_signals_db):
        """All generated channels (direct and subdevice) should validate."""
        # Direct signals
        assert direct_signals_db.validate_channel("CTRL:MAIN:MC-01:Status")
        assert direct_signals_db.validate_channel("CTRL:MAIN:MC-01:Heartbeat")
        assert direct_signals_db.validate_channel("CTRL:MAIN:MC-02:Status")
        assert direct_signals_db.validate_channel("CTRL:MAIN:MC-02:Heartbeat")

        # Subdevice signals
        assert direct_signals_db.validate_channel("CTRL:MAIN:MC-01:PSU:Voltage")
        assert direct_signals_db.validate_channel("CTRL:MAIN:MC-01:PSU:Current")
        assert direct_signals_db.validate_channel("CTRL:MAIN:MC-01:ADC:Value")
        assert direct_signals_db.validate_channel("CTRL:MAIN:MC-02:PSU:Voltage")

    def test_channel_count_includes_both_direct_and_subdevice(self, direct_signals_db):
        """Total channel count should include both direct signals and subdevice signals."""
        # 2 devices * (2 direct signals + 2 PSU signals + 1 ADC signal) = 2 * 5 = 10
        assert len(direct_signals_db.channel_map) == 10

    def test_get_channel_returns_correct_data(self, direct_signals_db):
        """Getting channel data should work for both direct and subdevice signals."""
        # Direct signal
        status_data = direct_signals_db.get_channel("CTRL:MAIN:MC-01:Status")
        assert status_data is not None
        assert status_data["address"] == "CTRL:MAIN:MC-01:Status"

        # Subdevice signal
        psu_data = direct_signals_db.get_channel("CTRL:MAIN:MC-01:PSU:Voltage")
        assert psu_data is not None
        assert psu_data["address"] == "CTRL:MAIN:MC-01:PSU:Voltage"


class TestExpansionAtOptionalLevel:
    """
    Test expansion behavior at optional tree levels.

    BUG: When a node with _expansion is defined at an optional tree level,
    the base container name should NOT appear as a selectable option.
    Only the expanded instances should be presented.
    """

    @pytest.fixture
    def expansion_at_optional_db_content(self):
        """
        Database with expansion at optional tree level.

        Structure:
        - system: CTRL (tree)
        - subsystem: MAIN (tree)
        - device: MC-01, MC-02 (instances)
        - subdevice: PSU, ADC, MOTOR, CH-1, CH-2 (optional tree with expansion)
          - PSU, ADC, MOTOR: regular containers (no expansion)
          - CH: container with expansion to CH-1, CH-2
        - signal: varies by subdevice (tree)
        - suffix: RB, SP (optional tree)
        """
        return {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "subsystem", "type": "tree"},
                    {"name": "device", "type": "instances"},
                    {"name": "subdevice", "type": "tree", "optional": True},
                    {"name": "signal", "type": "tree"},
                    {"name": "suffix", "type": "tree", "optional": True},
                ],
                "naming_pattern": "{system}:{subsystem}:{device}:{subdevice}:{signal}:{suffix}",
            },
            "tree": {
                "CTRL": {
                    "_description": "Control System",
                    "MAIN": {
                        "_description": "Main Subsystem",
                        "DEVICE": {
                            "_expansion": {
                                "_type": "range",
                                "_pattern": "MC-{:02d}",
                                "_range": [1, 2],
                            },
                            "_description": "Main control devices",
                            # Regular subdevices (no expansion)
                            "PSU": {
                                "_description": "Power Supply",
                                "Voltage": {"_description": "Output voltage"},
                                "Current": {"_description": "Output current"},
                            },
                            "ADC": {
                                "_description": "ADC",
                                "Value": {
                                    "_separator": "_",
                                    "_is_leaf": True,
                                    "_description": "ADC value",
                                    "RB": {"_description": "Readback"},
                                    "SP": {"_description": "Setpoint"},
                                },
                            },
                            "MOTOR": {
                                "_description": "Motor",
                                "Position": {"_description": "Motor position"},
                            },
                            # Subdevice with expansion (BUG LOCATION)
                            "CH": {
                                "_expansion": {
                                    "_type": "range",
                                    "_pattern": "CH-{}",
                                    "_range": [1, 2],
                                },
                                "_description": "Hardware channel modules",
                                "Input": {"_description": "Input value"},
                                "Gain": {
                                    "_separator": "_",
                                    "_is_leaf": True,
                                    "_description": "Gain amplification",
                                    "RB": {"_description": "Gain readback"},
                                    "SP": {"_description": "Gain setpoint"},
                                },
                            },
                        },
                    },
                },
            },
        }

    @pytest.fixture
    def expansion_at_optional_db(self, expansion_at_optional_db_content):
        """Create temporary database with expansion at optional level."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(expansion_at_optional_db_content, f)
            db_path = f.name

        try:
            db = HierarchicalChannelDatabase(db_path)
            yield db
        finally:
            Path(db_path).unlink()

    def test_expansion_at_optional_level_not_included_as_option(self, expansion_at_optional_db):
        """
        BUG TEST: Base container with _expansion should NOT appear in options.

        At the optional 'subdevice' level, we should see:
        - PSU (regular container) ✓
        - ADC (regular container) ✓
        - MOTOR (regular container) ✓
        - CH-1 (expanded instance) ✓
        - CH-2 (expanded instance) ✓

        We should NOT see:
        - CH (base container) ✗
        """
        selections = {"system": "CTRL", "subsystem": "MAIN", "device": "MC-01"}
        options = expansion_at_optional_db.get_options_at_level("subdevice", selections)

        option_names = {opt["name"] for opt in options}

        # Regular containers should appear
        assert "PSU" in option_names, "PSU (regular container) should appear"
        assert "ADC" in option_names, "ADC (regular container) should appear"
        assert "MOTOR" in option_names, "MOTOR (regular container) should appear"

        # Base container with expansion should NOT appear
        assert "CH" not in option_names, (
            "BUG: Base container 'CH' should NOT appear as selectable option. "
            "Only expanded instances CH-1 and CH-2 should be selectable."
        )

        # Expanded instances SHOULD appear
        assert "CH-1" in option_names, "CH-1 (expanded instance) should appear"
        assert "CH-2" in option_names, "CH-2 (expanded instance) should appear"

        # Total count should be 5: PSU, ADC, MOTOR, CH-1, CH-2
        assert len(options) == 5, f"Expected 5 options, got {len(options)}: {option_names}"

    def test_expansion_at_optional_level_builds_valid_channels(self, expansion_at_optional_db):
        """
        Expanded instances should generate valid channels in the channel map.

        Note: This test verifies that the channels exist in the channel map,
        which is populated during database initialization and correctly applies
        separator overrides.
        """
        # Verify CH-1 channels exist and validate
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-01:CH-1:Gain_RB") is True
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-01:CH-1:Gain_SP") is True
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-01:CH-1:Input") is True

        # Verify CH-2 channels exist and validate
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-01:CH-2:Gain_RB") is True
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-01:CH-2:Gain_SP") is True
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-01:CH-2:Input") is True

        # Verify both devices (MC-01 and MC-02)
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-02:CH-1:Gain_RB") is True
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-02:CH-2:Gain_RB") is True

    def test_base_container_name_does_not_build_valid_channels(self, expansion_at_optional_db):
        """
        Using base container name 'CH' should NOT build valid channels.

        This test demonstrates the bug: if the LLM incorrectly selects 'CH',
        it will build an invalid channel name.
        """
        # Try to use CH (base container) instead of CH-1/CH-2
        selections = {
            "system": "CTRL",
            "subsystem": "MAIN",
            "device": "MC-01",
            "subdevice": "CH",  # ❌ Should not be selectable
            "signal": "Gain",
            "suffix": "RB",
        }
        channels = expansion_at_optional_db.build_channels_from_selections(selections)

        # The invalid channel CTRL:MAIN:MC-01:CH:Gain_RB should NOT validate
        # (Even if build_channels returns it, validation should fail)
        if channels:
            # If the build process returns something, it should NOT validate
            assert expansion_at_optional_db.validate_channel(channels[0]) is False, (
                f"Channel {channels[0]} should NOT validate (uses base container 'CH')"
            )

        # Explicitly verify the specific invalid channel
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-01:CH:Gain_RB") is False

    def test_expansion_preserves_regular_containers(self, expansion_at_optional_db):
        """Regular containers without expansion should work normally."""
        # Select PSU (regular container, no expansion)
        selections = {
            "system": "CTRL",
            "subsystem": "MAIN",
            "device": "MC-01",
            "subdevice": "PSU",
            "signal": "Voltage",
        }
        channels = expansion_at_optional_db.build_channels_from_selections(selections)

        assert len(channels) == 1
        assert channels[0] == "CTRL:MAIN:MC-01:PSU:Voltage"
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-01:PSU:Voltage") is True

    def test_all_expanded_channels_exist_in_channel_map(self, expansion_at_optional_db):
        """All expanded channels should be generated and in channel map."""
        # CH-1 channels
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-01:CH-1:Input")
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-01:CH-1:Gain")
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-01:CH-1:Gain_RB")
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-01:CH-1:Gain_SP")

        # CH-2 channels
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-01:CH-2:Input")
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-01:CH-2:Gain")
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-01:CH-2:Gain_RB")
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-01:CH-2:Gain_SP")

        # Both devices (MC-01 and MC-02)
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-02:CH-1:Gain_RB")
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-02:CH-2:Gain_RB")

    def test_base_container_channels_do_not_exist(self, expansion_at_optional_db):
        """Channels using base container 'CH' should NOT exist in channel map."""
        # These should all be invalid
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-01:CH:Input") is False
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-01:CH:Gain") is False
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-01:CH:Gain_RB") is False
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-01:CH:Gain_SP") is False
        assert expansion_at_optional_db.validate_channel("CTRL:MAIN:MC-02:CH:Gain_RB") is False

    def test_channel_count_uses_expanded_instances(self, expansion_at_optional_db):
        """Total channel count should use expanded instances, not base container."""
        # Expected channels per device:
        # - PSU: Voltage, Current = 2
        # - ADC: Value, Value_RB, Value_SP = 3
        # - MOTOR: Position = 1
        # - CH-1: Input, Gain, Gain_RB, Gain_SP = 4
        # - CH-2: Input, Gain, Gain_RB, Gain_SP = 4
        # Total per device: 2 + 3 + 1 + 4 + 4 = 14
        # Total for 2 devices: 14 * 2 = 28
        expected_count = 28

        actual_count = len(expansion_at_optional_db.channel_map)
        assert actual_count == expected_count, (
            f"Expected {expected_count} channels (using expanded instances), got {actual_count}"
        )

    def test_build_channels_from_selections_with_expanded_instance_preserves_separator_override(
        self, expansion_at_optional_db
    ):
        """
        REGRESSION TEST: build_channels_from_selections should preserve separator overrides
        when given an expanded instance like 'CH-1'.

        Context:
        - The optional_levels.json database has 'CH' with expansion to CH-1, CH-2
        - Inside the 'CH' container, 'Gain' has _separator: "_" (use underscore, not colon)
        - The channel map correctly creates: CTRL:MAIN:MC-01:CH-1:Gain_RB ✅
        - But build_channels_from_selections with subdevice: "CH-1" used to build:
          CTRL:MAIN:MC-01:CH-1:Gain:RB ❌ (wrong - uses colon instead of underscore)

        Root cause:
        - When navigating the tree with "CH-1", the method can't find that literal key
          (only "CH" exists with the expansion definition)
        - This caused it to lose access to separator override metadata

        This test verifies the fix: expanded instances should correctly resolve to their
        container node and preserve all separator overrides.
        """
        # Build channels using the expanded instance directly
        selections = {
            "system": "CTRL",
            "subsystem": "MAIN",
            "device": "MC-01",
            "subdevice": "CH-1",  # Expanded instance (tree only has "CH")
            "signal": "Gain",
            "suffix": "RB",
        }

        channels = expansion_at_optional_db.build_channels_from_selections(selections)

        # Should build the channel with the underscore separator (not colon)
        assert len(channels) == 1
        channel = channels[0]

        # The separator override from Gain's _separator: "_" should be applied
        expected = "CTRL:MAIN:MC-01:CH-1:Gain_RB"  # Underscore, not colon!
        assert channel == expected, (
            f"Separator override not applied correctly.\n"
            f"Expected: {expected}\n"
            f"Got:      {channel}\n"
            f"The 'Gain' node has _separator: '_', so RB should connect with underscore"
        )

        # Also verify with CH-2
        selections["subdevice"] = "CH-2"
        channels = expansion_at_optional_db.build_channels_from_selections(selections)
        assert channels[0] == "CTRL:MAIN:MC-01:CH-2:Gain_RB"
