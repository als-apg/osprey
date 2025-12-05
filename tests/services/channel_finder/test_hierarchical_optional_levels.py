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
