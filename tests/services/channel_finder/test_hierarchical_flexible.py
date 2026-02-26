"""
Unit tests for flexible hierarchical channel database.

Tests the new hierarchy_config system that supports arbitrary mixing of
instance levels (expand_here) and tree levels (categories).
"""

import json
import tempfile
from pathlib import Path

import pytest

from osprey.services.channel_finder.databases.hierarchical import (
    HierarchicalChannelDatabase,
)


# Test helper function to create temporary database files
def create_temp_database(db_dict: dict) -> str:
    """Create a temporary JSON database file."""
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(db_dict, temp_file)
    temp_file.flush()
    temp_file.close()
    return temp_file.name


class TestBackwardCompatibility:
    """Test that existing databases without hierarchy_config still work."""

    def test_legacy_database_loads(self):
        """Legacy database without hierarchy_config loads successfully."""
        legacy_db = {
            "hierarchy_definition": ["system", "family", "device", "field", "subfield"],
            "naming_pattern": "{system}:{family}[{device}]:{field}:{subfield}",
            "tree": {
                "MAG": {
                    "DIPOLE": {
                        "devices": {"_type": "range", "_pattern": "B{:02d}", "_range": [1, 3]},
                        "fields": {
                            "CURRENT": {
                                "subfields": {
                                    "SP": {"_description": "Setpoint"},
                                    "RB": {"_description": "Readback"},
                                }
                            }
                        },
                    }
                }
            },
        }

        db_path = create_temp_database(legacy_db)

        try:
            db = HierarchicalChannelDatabase(db_path)

            # Should auto-infer config
            assert hasattr(db, "hierarchy_config")
            assert "levels" in db.hierarchy_config

            # Should build channels correctly
            assert len(db.channel_map) == 6  # 3 devices × 2 subfields

            # Should have inferred correct structure
            assert db.hierarchy_config["levels"]["system"]["type"] == "tree"
            assert db.hierarchy_config["levels"]["family"]["type"] == "tree"
            assert db.hierarchy_config["levels"]["device"]["type"] == "container"

        finally:
            Path(db_path).unlink()

    def test_legacy_navigation_works(self):
        """Legacy database navigation still functions correctly."""
        legacy_db = {
            "hierarchy_definition": ["system", "family", "device", "field", "subfield"],
            "naming_pattern": "{system}:{family}[{device}]:{field}:{subfield}",
            "tree": {
                "MAG": {
                    "DIPOLE": {
                        "devices": {"_type": "range", "_pattern": "B{:02d}", "_range": [1, 2]},
                        "fields": {"CURRENT": {"subfields": {"SP": {"_description": "Setpoint"}}}},
                    }
                }
            },
        }

        db_path = create_temp_database(legacy_db)

        try:
            db = HierarchicalChannelDatabase(db_path)

            # Test navigation at each level
            systems = db.get_options_at_level("system", {})
            assert len(systems) == 1
            assert systems[0]["name"] == "MAG"

            families = db.get_options_at_level("family", {"system": "MAG"})
            assert len(families) == 1
            assert families[0]["name"] == "DIPOLE"

            devices = db.get_options_at_level("device", {"system": "MAG", "family": "DIPOLE"})
            assert len(devices) == 2
            assert devices[0]["name"] == "B01"
            assert devices[1]["name"] == "B02"

        finally:
            Path(db_path).unlink()


class TestMixedInstanceTree:
    """Test Instance → Category → Instance → Category pattern."""

    def test_production_line_structure(self):
        """Test LINE[instances] → STATION[tree] → PARAMETER[tree]."""
        production_db = {
            "hierarchy_definition": ["line", "station", "parameter"],
            "naming_pattern": "LINE{line}:{station}:{parameter}",
            "hierarchy_config": {
                "levels": {
                    "line": {"type": "instances"},
                    "station": {"type": "tree"},
                    "parameter": {"type": "tree"},
                }
            },
            "tree": {
                "LINE": {
                    "_expansion": {"_type": "range", "_pattern": "{}", "_range": [1, 3]},
                    "ASSEMBLY": {
                        "SPEED": {"_description": "Line speed"},
                        "STATUS": {"_description": "Status"},
                    },
                    "INSPECTION": {
                        "PASS_COUNT": {"_description": "Passed"},
                        "FAIL_COUNT": {"_description": "Failed"},
                    },
                }
            },
        }

        db_path = create_temp_database(production_db)

        try:
            db = HierarchicalChannelDatabase(db_path)

            # Check channel count: 3 lines × 2 stations × (2+2) parameters = 12
            assert len(db.channel_map) == 12

            # Test navigation - line level should generate instances
            lines = db.get_options_at_level("line", {})
            assert len(lines) == 3
            assert lines[0]["name"] == "1"
            assert lines[1]["name"] == "2"
            assert lines[2]["name"] == "3"

            # Station level should show tree categories (same for all lines)
            stations = db.get_options_at_level("station", {"line": "1"})
            assert len(stations) == 2
            assert stations[0]["name"] == "ASSEMBLY"
            assert stations[1]["name"] == "INSPECTION"

            # Parameter level depends on station
            params_assembly = db.get_options_at_level(
                "parameter", {"line": "1", "station": "ASSEMBLY"}
            )
            assert len(params_assembly) == 2
            assert {p["name"] for p in params_assembly} == {"SPEED", "STATUS"}

            params_inspection = db.get_options_at_level(
                "parameter", {"line": "1", "station": "INSPECTION"}
            )
            assert len(params_inspection) == 2
            assert {p["name"] for p in params_inspection} == {"PASS_COUNT", "FAIL_COUNT"}

            # Test channel generation
            channels = db.build_channels_from_selections(
                {"line": ["1", "2"], "station": "ASSEMBLY", "parameter": "SPEED"}
            )
            assert len(channels) == 2
            assert "LINE1:ASSEMBLY:SPEED" in channels
            assert "LINE2:ASSEMBLY:SPEED" in channels

        finally:
            Path(db_path).unlink()

    def test_building_management_structure(self):
        """Test SECTOR[instances] → BUILDING[tree] → FLOOR[instances] → ROOM[instances] → EQUIPMENT[tree]."""
        building_db = {
            "hierarchy_definition": ["sector", "building", "floor", "room", "equipment"],
            "naming_pattern": "S{sector}:{building}:F{floor}:R{room}:{equipment}",
            "hierarchy_config": {
                "levels": {
                    "sector": {"type": "instances"},
                    "building": {"type": "tree"},
                    "floor": {"type": "instances"},
                    "room": {"type": "instances"},
                    "equipment": {"type": "tree"},
                }
            },
            "tree": {
                "SECTOR": {
                    "_expansion": {"_type": "range", "_pattern": "{:02d}", "_range": [1, 2]},
                    "MAIN_BUILDING": {
                        "_description": "Main building",
                        "FLOOR": {
                            "_expansion": {"_type": "range", "_pattern": "{}", "_range": [1, 2]},
                            "ROOM": {
                                "_expansion": {
                                    "_type": "range",
                                    "_pattern": "{:03d}",
                                    "_range": [101, 102],
                                },
                                "HVAC": {"_description": "Climate control"},
                                "LIGHTING": {"_description": "Lighting"},
                            },
                        },
                    },
                }
            },
        }

        db_path = create_temp_database(building_db)

        try:
            db = HierarchicalChannelDatabase(db_path)

            # Check channel count: 2 sectors × 1 building × 2 floors × 2 rooms × 2 equipment = 16
            assert len(db.channel_map) == 16

            # Test that instance levels don't change tree position
            sectors = db.get_options_at_level("sector", {})
            assert len(sectors) == 2

            # Building level - should be same for any sector
            buildings_s01 = db.get_options_at_level("building", {"sector": "01"})
            buildings_s02 = db.get_options_at_level("building", {"sector": "02"})
            assert buildings_s01 == buildings_s02
            assert len(buildings_s01) == 1
            assert buildings_s01[0]["name"] == "MAIN_BUILDING"

            # Floor level - also same for any sector (because it's after building)
            floors = db.get_options_at_level("floor", {"sector": "01", "building": "MAIN_BUILDING"})
            assert len(floors) == 2
            assert floors[0]["name"] == "1"
            assert floors[1]["name"] == "2"

            # Room level - consecutive instance level
            rooms = db.get_options_at_level(
                "room", {"sector": "01", "building": "MAIN_BUILDING", "floor": "1"}
            )
            assert len(rooms) == 2
            assert rooms[0]["name"] == "101"
            assert rooms[1]["name"] == "102"

            # Equipment level
            equipment = db.get_options_at_level(
                "equipment",
                {"sector": "01", "building": "MAIN_BUILDING", "floor": "1", "room": "101"},
            )
            assert len(equipment) == 2
            assert {e["name"] for e in equipment} == {"HVAC", "LIGHTING"}

            # Verify specific channel exists
            assert db.validate_channel("S01:MAIN_BUILDING:F1:R101:HVAC")
            assert db.validate_channel("S02:MAIN_BUILDING:F2:R102:LIGHTING")

        finally:
            Path(db_path).unlink()


class TestMultipleConsecutiveInstances:
    """Test multiple instance levels in a row."""

    def test_consecutive_instances_validation(self):
        """Consecutive instance levels must be properly nested."""
        # INCORRECT structure - siblings instead of nested
        bad_db = {
            "hierarchy_definition": ["floor", "room"],
            "naming_pattern": "F{floor}:R{room}",
            "hierarchy_config": {
                "levels": {"floor": {"type": "instances"}, "room": {"type": "instances"}}
            },
            "tree": {
                "FLOOR": {"_expansion": {"_type": "range", "_pattern": "{}", "_range": [1, 2]}},
                "ROOM": {
                    "_expansion": {"_type": "range", "_pattern": "{:03d}", "_range": [101, 102]}
                },
            },
        }

        db_path = create_temp_database(bad_db)

        try:
            with pytest.raises(ValueError, match="Consecutive instance levels"):
                HierarchicalChannelDatabase(db_path)
        finally:
            Path(db_path).unlink()

    def test_consecutive_instances_correct_nesting(self):
        """Correctly nested consecutive instances work properly."""
        good_db = {
            "hierarchy_definition": ["floor", "room", "equipment"],
            "naming_pattern": "F{floor}:R{room}:{equipment}",
            "hierarchy_config": {
                "levels": {
                    "floor": {"type": "instances"},
                    "room": {"type": "instances"},
                    "equipment": {"type": "tree"},
                }
            },
            "tree": {
                "FLOOR": {
                    "_expansion": {"_type": "range", "_pattern": "{}", "_range": [1, 2]},
                    "ROOM": {
                        "_expansion": {
                            "_type": "range",
                            "_pattern": "{:03d}",
                            "_range": [101, 102],
                        },
                        "HVAC": {"_description": "Climate"},
                        "LIGHTS": {"_description": "Lighting"},
                    },
                }
            },
        }

        db_path = create_temp_database(good_db)

        try:
            db = HierarchicalChannelDatabase(db_path)

            # Should create: 2 floors × 2 rooms × 2 equipment = 8 channels
            assert len(db.channel_map) == 8

            # Verify specific channels
            assert db.validate_channel("F1:R101:HVAC")
            assert db.validate_channel("F2:R102:LIGHTS")

        finally:
            Path(db_path).unlink()


class TestInstanceFirstLevel:
    """Test instance level at the first position."""

    def test_instance_at_root(self):
        """Instance level at first position works correctly."""
        db_dict = {
            "hierarchy_definition": ["line", "station"],
            "naming_pattern": "LINE{line}:{station}",
            "hierarchy_config": {
                "levels": {"line": {"type": "instances"}, "station": {"type": "tree"}}
            },
            "tree": {
                "LINE": {
                    "_expansion": {"_type": "list", "_instances": ["A", "B", "C"]},
                    "ASSEMBLY": {"_description": "Assembly station"},
                    "PACKAGING": {"_description": "Packaging station"},
                }
            },
        }

        db_path = create_temp_database(db_dict)

        try:
            db = HierarchicalChannelDatabase(db_path)

            # Should create: 3 lines × 2 stations = 6 channels
            assert len(db.channel_map) == 6

            # Test navigation at first level
            lines = db.get_options_at_level("line", {})
            assert len(lines) == 3
            assert {line["name"] for line in lines} == {"A", "B", "C"}

            # Station level should be same for all lines
            stations = db.get_options_at_level("station", {"line": "A"})
            assert len(stations) == 2
            assert {s["name"] for s in stations} == {"ASSEMBLY", "PACKAGING"}

        finally:
            Path(db_path).unlink()


class TestCartesianProduct:
    """Test channel generation creates correct combinations."""

    def test_cartesian_product_all_levels(self):
        """Channel generation works with any number of levels."""
        db_dict = {
            "hierarchy_definition": ["a", "b", "c"],
            "naming_pattern": "{a}:{b}:{c}",
            "hierarchy_config": {
                "levels": {"a": {"type": "tree"}, "b": {"type": "tree"}, "c": {"type": "tree"}}
            },
            "tree": {
                "A1": {
                    "B1": {
                        "C1": {"_description": "Option C1"},
                        "C2": {"_description": "Option C2"},
                    },
                    "B2": {"C3": {"_description": "Option C3"}},
                }
            },
        }

        db_path = create_temp_database(db_dict)

        try:
            db = HierarchicalChannelDatabase(db_path)

            # Test full cartesian product
            channels = db.build_channels_from_selections(
                {"a": "A1", "b": ["B1", "B2"], "c": ["C1", "C2", "C3"]}
            )

            # Should get all combinations: A1:B1:C1, A1:B1:C2, A1:B2:C1, A1:B2:C2, A1:B2:C3
            # But only A1:B1:C1, A1:B1:C2, A1:B2:C3 are valid in the tree
            # The build function doesn't validate - just generates
            assert len(channels) == 6

            # Test with instances
            channels2 = db.build_channels_from_selections({"a": ["A1"], "b": "B1", "c": "C1"})
            assert len(channels2) == 1
            assert channels2[0] == "A1:B1:C1"

        finally:
            Path(db_path).unlink()


class TestNavigationSkipsInstances:
    """Test tree navigation doesn't change position for instance levels."""

    def test_instance_level_stays_at_node(self):
        """Instance levels don't navigate - they expand in place."""
        db_dict = {
            "hierarchy_definition": ["sector", "building", "floor"],
            "naming_pattern": "S{sector}:{building}:F{floor}",
            "hierarchy_config": {
                "levels": {
                    "sector": {"type": "instances"},
                    "building": {"type": "tree"},
                    "floor": {"type": "tree"},
                }
            },
            "tree": {
                "SECTOR": {
                    "_expansion": {"_type": "range", "_pattern": "{:02d}", "_range": [1, 2]},
                    "BUILDING_A": {
                        "FLOOR_1": {"_description": "First floor"},
                        "FLOOR_2": {"_description": "Second floor"},
                    },
                    "BUILDING_B": {"FLOOR_G": {"_description": "Ground floor"}},
                }
            },
        }

        db_path = create_temp_database(db_dict)

        try:
            db = HierarchicalChannelDatabase(db_path)

            # Navigation to building should be same regardless of sector selection
            buildings_s01 = db.get_options_at_level("building", {"sector": "01"})
            buildings_s02 = db.get_options_at_level("building", {"sector": "02"})

            assert buildings_s01 == buildings_s02
            assert len(buildings_s01) == 2
            assert {b["name"] for b in buildings_s01} == {"BUILDING_A", "BUILDING_B"}

            # Floor options depend on building, not sector
            floors_a = db.get_options_at_level("floor", {"sector": "01", "building": "BUILDING_A"})
            floors_b = db.get_options_at_level("floor", {"sector": "02", "building": "BUILDING_B"})

            assert len(floors_a) == 2
            assert len(floors_b) == 1

        finally:
            Path(db_path).unlink()


class TestExpansionTypes:
    """Test different expansion types (range vs list)."""

    def test_range_expansion(self):
        """Range-based expansion with patterns."""
        db_dict = {
            "hierarchy_definition": ["device"],
            "naming_pattern": "{device}",
            "hierarchy_config": {"levels": {"device": {"type": "instances"}}},
            "tree": {
                "DEVICE": {
                    "_expansion": {"_type": "range", "_pattern": "DEV{:03d}", "_range": [5, 7]}
                }
            },
        }

        db_path = create_temp_database(db_dict)

        try:
            db = HierarchicalChannelDatabase(db_path)

            devices = db.get_options_at_level("device", {})
            assert len(devices) == 3
            assert devices[0]["name"] == "DEV005"
            assert devices[1]["name"] == "DEV006"
            assert devices[2]["name"] == "DEV007"

        finally:
            Path(db_path).unlink()

    def test_list_expansion(self):
        """List-based expansion with named instances."""
        db_dict = {
            "hierarchy_definition": ["server"],
            "naming_pattern": "{server}",
            "hierarchy_config": {"levels": {"server": {"type": "instances"}}},
            "tree": {
                "SERVER": {
                    "_expansion": {"_type": "list", "_instances": ["MAIN", "BACKUP", "TEST"]}
                }
            },
        }

        db_path = create_temp_database(db_dict)

        try:
            db = HierarchicalChannelDatabase(db_path)

            servers = db.get_options_at_level("server", {})
            assert len(servers) == 3
            assert {s["name"] for s in servers} == {"MAIN", "BACKUP", "TEST"}

        finally:
            Path(db_path).unlink()


class TestValidation:
    """Test validation of hierarchy configuration."""

    def test_missing_hierarchy_config_levels(self):
        """Missing 'levels' key in hierarchy_config."""
        bad_db = {
            "hierarchy_definition": ["level1"],
            "naming_pattern": "{level1}",
            "hierarchy_config": {},  # Missing 'levels'
            "tree": {},
        }

        db_path = create_temp_database(bad_db)

        try:
            with pytest.raises(ValueError, match="must contain 'levels' key"):
                HierarchicalChannelDatabase(db_path)
        finally:
            Path(db_path).unlink()

    def test_unconfigured_level(self):
        """All levels must be configured."""
        bad_db = {
            "hierarchy_definition": ["level1", "level2"],
            "naming_pattern": "{level1}:{level2}",
            "hierarchy_config": {
                "levels": {
                    "level1": {"type": "tree"}
                    # Missing level2!
                }
            },
            "tree": {},
        }

        db_path = create_temp_database(bad_db)

        try:
            with pytest.raises(ValueError, match="Level 'level2' not found"):
                HierarchicalChannelDatabase(db_path)
        finally:
            Path(db_path).unlink()

    def test_missing_structure_property(self):
        """Levels must have 'type' property."""
        bad_db = {
            "hierarchy_definition": ["level1"],
            "naming_pattern": "{level1}",
            "hierarchy_config": {"levels": {"level1": {}}},  # Missing type!
            "tree": {},
        }

        db_path = create_temp_database(bad_db)

        try:
            with pytest.raises(ValueError, match="missing required 'type' property"):
                HierarchicalChannelDatabase(db_path)
        finally:
            Path(db_path).unlink()

    def test_invalid_structure_value(self):
        """Type must be 'tree', 'instances', or 'container'."""
        bad_db = {
            "hierarchy_definition": ["level1"],
            "naming_pattern": "{level1}",
            "hierarchy_config": {"levels": {"level1": {"type": "invalid"}}},
            "tree": {},
        }

        db_path = create_temp_database(bad_db)

        try:
            with pytest.raises(ValueError, match="has invalid type"):
                HierarchicalChannelDatabase(db_path)
        finally:
            Path(db_path).unlink()

    def test_missing_expansion_definition(self):
        """Instance level must have _expansion definition."""
        bad_db = {
            "hierarchy_definition": ["device"],
            "naming_pattern": "{device}",
            "hierarchy_config": {"levels": {"device": {"type": "instances"}}},
            "tree": {"DEVICE": {"CHILD": {}}},  # Has child but missing _expansion!
        }

        db_path = create_temp_database(bad_db)

        try:
            with pytest.raises(ValueError, match="missing '_expansion' definition"):
                HierarchicalChannelDatabase(db_path)
        finally:
            Path(db_path).unlink()

    def test_invalid_expansion_type(self):
        """Expansion _type must be 'range' or 'list'."""
        bad_db = {
            "hierarchy_definition": ["device"],
            "naming_pattern": "{device}",
            "hierarchy_config": {"levels": {"device": {"type": "instances"}}},
            "tree": {"DEVICE": {"_expansion": {"_type": "invalid"}}},
        }

        db_path = create_temp_database(bad_db)

        try:
            with pytest.raises(ValueError, match="invalid '_type'"):
                HierarchicalChannelDatabase(db_path)
        finally:
            Path(db_path).unlink()

    def test_range_expansion_missing_fields(self):
        """Range expansion must have _pattern and _range."""
        # Missing _pattern
        bad_db1 = {
            "hierarchy_definition": ["device"],
            "naming_pattern": "{device}",
            "hierarchy_config": {"levels": {"device": {"type": "instances"}}},
            "tree": {"DEVICE": {"_expansion": {"_type": "range", "_range": [1, 10]}}},
        }

        db_path1 = create_temp_database(bad_db1)

        try:
            with pytest.raises(ValueError, match="requires '_pattern' field"):
                HierarchicalChannelDatabase(db_path1)
        finally:
            Path(db_path1).unlink()

        # Missing _range
        bad_db2 = {
            "hierarchy_definition": ["device"],
            "naming_pattern": "{device}",
            "hierarchy_config": {"levels": {"device": {"type": "instances"}}},
            "tree": {"DEVICE": {"_expansion": {"_type": "range", "_pattern": "{}"}}},
        }

        db_path2 = create_temp_database(bad_db2)

        try:
            with pytest.raises(ValueError, match="requires '_range' field"):
                HierarchicalChannelDatabase(db_path2)
        finally:
            Path(db_path2).unlink()

    def test_list_expansion_missing_instances(self):
        """List expansion must have _instances."""
        bad_db = {
            "hierarchy_definition": ["device"],
            "naming_pattern": "{device}",
            "hierarchy_config": {"levels": {"device": {"type": "instances"}}},
            "tree": {"DEVICE": {"_expansion": {"_type": "list"}}},
        }

        db_path = create_temp_database(bad_db)

        try:
            with pytest.raises(ValueError, match="requires '_instances' field"):
                HierarchicalChannelDatabase(db_path)
        finally:
            Path(db_path).unlink()


class TestEdgeCases:
    """Test edge cases and unusual configurations."""

    def test_all_instances_no_tree_levels(self):
        """Database with only instance levels (no categories)."""
        db_dict = {
            "hierarchy_definition": ["rack", "shelf", "slot"],
            "naming_pattern": "R{rack}:S{shelf}:SLOT{slot}",
            "hierarchy_config": {
                "levels": {
                    "rack": {"type": "instances"},
                    "shelf": {"type": "instances"},
                    "slot": {"type": "instances"},
                }
            },
            "tree": {
                "RACK": {
                    "_expansion": {"_type": "range", "_pattern": "{}", "_range": [1, 2]},
                    "SHELF": {
                        "_expansion": {"_type": "range", "_pattern": "{}", "_range": [1, 2]},
                        "SLOT": {
                            "_expansion": {"_type": "range", "_pattern": "{:02d}", "_range": [1, 3]}
                        },
                    },
                }
            },
        }

        db_path = create_temp_database(db_dict)

        try:
            db = HierarchicalChannelDatabase(db_path)

            # Should create: 2 racks × 2 shelves × 3 slots = 12 channels
            assert len(db.channel_map) == 12

            # Verify specific channels
            assert db.validate_channel("R1:S1:SLOT01")
            assert db.validate_channel("R2:S2:SLOT03")

        finally:
            Path(db_path).unlink()

    def test_all_tree_no_instances(self):
        """Database with only tree levels (no instances)."""
        db_dict = {
            "hierarchy_definition": ["building", "wing", "type"],
            "naming_pattern": "{building}:{wing}:{type}",
            "hierarchy_config": {
                "levels": {
                    "building": {"type": "tree"},
                    "wing": {"type": "tree"},
                    "type": {"type": "tree"},
                }
            },
            "tree": {
                "MAIN": {
                    "NORTH": {
                        "OFFICE": {"_description": "Office space"},
                        "LAB": {"_description": "Laboratory"},
                    },
                    "SOUTH": {"STORAGE": {"_description": "Storage area"}},
                }
            },
        }

        db_path = create_temp_database(db_dict)

        try:
            db = HierarchicalChannelDatabase(db_path)

            # Should create: 1 building × 2 wings × (2+1) types = 3 channels
            assert len(db.channel_map) == 3

            assert db.validate_channel("MAIN:NORTH:OFFICE")
            assert db.validate_channel("MAIN:NORTH:LAB")
            assert db.validate_channel("MAIN:SOUTH:STORAGE")

        finally:
            Path(db_path).unlink()

    def test_single_level_instance(self):
        """Single-level hierarchy with instance."""
        db_dict = {
            "hierarchy_definition": ["device"],
            "naming_pattern": "DEV_{device}",
            "hierarchy_config": {"levels": {"device": {"type": "instances"}}},
            "tree": {
                "DEVICE": {"_expansion": {"_type": "range", "_pattern": "{:02d}", "_range": [1, 3]}}
            },
        }

        db_path = create_temp_database(db_dict)

        try:
            db = HierarchicalChannelDatabase(db_path)

            assert len(db.channel_map) == 3
            assert db.validate_channel("DEV_01")
            assert db.validate_channel("DEV_02")
            assert db.validate_channel("DEV_03")

        finally:
            Path(db_path).unlink()

    def test_single_level_tree(self):
        """Single-level hierarchy with tree."""
        db_dict = {
            "hierarchy_definition": ["system"],
            "naming_pattern": "{system}",
            "hierarchy_config": {"levels": {"system": {"type": "tree"}}},
            "tree": {
                "SYSTEM_A": {"_description": "System A"},
                "SYSTEM_B": {"_description": "System B"},
            },
        }

        db_path = create_temp_database(db_dict)

        try:
            db = HierarchicalChannelDatabase(db_path)

            assert len(db.channel_map) == 2
            assert db.validate_channel("SYSTEM_A")
            assert db.validate_channel("SYSTEM_B")

        finally:
            Path(db_path).unlink()


class TestRangeValidation:
    """Test validation of range expansion parameters."""

    def test_invalid_range_format(self):
        """Range must be [start, end] list."""
        bad_db = {
            "hierarchy_definition": ["device"],
            "naming_pattern": "{device}",
            "hierarchy_config": {"levels": {"device": {"type": "instances"}}},
            "tree": {
                "DEVICE": {
                    "_expansion": {
                        "_type": "range",
                        "_pattern": "{}",
                        "_range": [1, 2, 3],  # Too many elements!
                    }
                }
            },
        }

        db_path = create_temp_database(bad_db)

        try:
            with pytest.raises(ValueError, match="must be \\[start, end\\] list"):
                HierarchicalChannelDatabase(db_path)
        finally:
            Path(db_path).unlink()

    def test_invalid_range_types(self):
        """Range start and end must be integers."""
        bad_db = {
            "hierarchy_definition": ["device"],
            "naming_pattern": "{device}",
            "hierarchy_config": {"levels": {"device": {"type": "instances"}}},
            "tree": {
                "DEVICE": {
                    "_expansion": {
                        "_type": "range",
                        "_pattern": "{}",
                        "_range": ["1", "10"],  # Strings instead of ints!
                    }
                }
            },
        }

        db_path = create_temp_database(bad_db)

        try:
            with pytest.raises(ValueError, match="start and end must be integers"):
                HierarchicalChannelDatabase(db_path)
        finally:
            Path(db_path).unlink()

    def test_invalid_range_order(self):
        """Range start must be <= end."""
        bad_db = {
            "hierarchy_definition": ["device"],
            "naming_pattern": "{device}",
            "hierarchy_config": {"levels": {"device": {"type": "instances"}}},
            "tree": {
                "DEVICE": {
                    "_expansion": {
                        "_type": "range",
                        "_pattern": "{}",
                        "_range": [10, 1],  # End < start!
                    }
                }
            },
        }

        db_path = create_temp_database(bad_db)

        try:
            with pytest.raises(ValueError, match="start must be <= end"):
                HierarchicalChannelDatabase(db_path)
        finally:
            Path(db_path).unlink()


class TestComplexHierarchies:
    """Test complex real-world hierarchies."""

    def test_deep_hierarchy(self):
        """Test deep hierarchy (8 levels)."""
        deep_db = {
            "hierarchy_definition": [
                "region",
                "site",
                "building",
                "floor",
                "room",
                "rack",
                "device",
                "channel",
            ],
            "naming_pattern": "{region}:{site}:{building}:F{floor}:R{room}:RACK{rack}:{device}:{channel}",
            "hierarchy_config": {
                "levels": {
                    "region": {"type": "tree"},
                    "site": {"type": "tree"},
                    "building": {"type": "tree"},
                    "floor": {"type": "instances"},
                    "room": {"type": "instances"},
                    "rack": {"type": "instances"},
                    "device": {"type": "tree"},
                    "channel": {"type": "tree"},
                }
            },
            "tree": {
                "WEST": {
                    "SITE_A": {
                        "BLDG_1": {
                            "FLOOR": {
                                "_expansion": {
                                    "_type": "range",
                                    "_pattern": "{}",
                                    "_range": [1, 2],
                                },
                                "ROOM": {
                                    "_expansion": {
                                        "_type": "range",
                                        "_pattern": "{:03d}",
                                        "_range": [101, 102],
                                    },
                                    "RACK": {
                                        "_expansion": {
                                            "_type": "range",
                                            "_pattern": "{}",
                                            "_range": [1, 2],
                                        },
                                        "SERVER": {
                                            "TEMP": {"_description": "Temperature"},
                                            "POWER": {"_description": "Power"},
                                        },
                                    },
                                },
                            }
                        }
                    }
                }
            },
        }

        db_path = create_temp_database(deep_db)

        try:
            db = HierarchicalChannelDatabase(db_path)

            # Should create: 1×1×1×2×2×2×1×2 = 16 channels
            assert len(db.channel_map) == 16

            # Verify specific channel
            assert db.validate_channel("WEST:SITE_A:BLDG_1:F1:R101:RACK1:SERVER:TEMP")

        finally:
            Path(db_path).unlink()


class TestNamingPatternValidation:
    """Test validation of naming_pattern against hierarchy levels."""

    def test_naming_pattern_matches_levels(self):
        """Valid naming pattern with all level names should load successfully."""
        good_db = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "device", "type": "instances"},
                    {"name": "field", "type": "tree"},
                ],
                "naming_pattern": "{system}:{device}:{field}",
            },
            "tree": {
                "SYS1": {
                    "DEVICE": {
                        "_expansion": {"_type": "range", "_pattern": "D{:02d}", "_range": [1, 2]},
                        "FIELD1": {"_description": "Field 1"},
                    }
                }
            },
        }

        db_path = create_temp_database(good_db)

        try:
            db = HierarchicalChannelDatabase(db_path)
            assert len(db.channel_map) == 2
        finally:
            Path(db_path).unlink()

    def test_naming_pattern_missing_level(self):
        """Naming pattern with subset of levels is now allowed (navigation-only levels)."""
        good_db = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "device", "type": "instances"},
                    {"name": "field", "type": "tree"},
                ],
                "naming_pattern": "{system}:{field}",  # Device level not in pattern (navigation-only)
            },
            "tree": {
                "SYS1": {
                    "DEVICE": {
                        "_expansion": {"_type": "range", "_pattern": "{:02d}", "_range": [1, 1]},
                        "FIELD1": {"_description": "Field 1"},
                    }
                }
            },
        }

        db_path = create_temp_database(good_db)

        try:
            # This should now work - device is navigation-only
            db = HierarchicalChannelDatabase(db_path)
            # Pattern should extract only system and field
            pattern_levels = db._get_pattern_levels()
            assert set(pattern_levels) == {"system", "field"}
        finally:
            Path(db_path).unlink()

    def test_naming_pattern_extra_level(self):
        """Naming pattern with extra undefined level should raise error."""
        bad_db = {
            "hierarchy": {
                "levels": [{"name": "system", "type": "tree"}, {"name": "field", "type": "tree"}],
                "naming_pattern": "{system}:{device}:{field}",  # Extra {device}!
            },
            "tree": {},
        }

        db_path = create_temp_database(bad_db)

        try:
            with pytest.raises(ValueError, match="undefined hierarchy levels"):
                HierarchicalChannelDatabase(db_path)
        finally:
            Path(db_path).unlink()

    def test_naming_pattern_wrong_level_name(self):
        """Naming pattern with typo in level name should raise error."""
        bad_db = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "device", "type": "instances"},
                    {"name": "field", "type": "tree"},
                ],
                "naming_pattern": "{system}:{devices}:{field}",  # Typo: devices vs device
            },
            "tree": {},
        }

        db_path = create_temp_database(bad_db)

        try:
            with pytest.raises(ValueError, match="undefined hierarchy levels"):
                HierarchicalChannelDatabase(db_path)
        finally:
            Path(db_path).unlink()

    def test_naming_pattern_different_order(self):
        """Naming pattern with different order is valid (order in pattern is what matters)."""
        good_db = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "device", "type": "instances"},
                    {"name": "field", "type": "tree"},
                ],
                "naming_pattern": "{field}-{system}-{device}",  # Different order, but all present
            },
            "tree": {
                "SYS1": {
                    "DEVICE": {
                        "_expansion": {"_type": "range", "_pattern": "{:02d}", "_range": [1, 2]},
                        "FIELD1": {"_description": "Field 1"},
                    }
                }
            },
        }

        db_path = create_temp_database(good_db)

        try:
            db = HierarchicalChannelDatabase(db_path)
            # Should create channels with pattern: FIELD1-SYS1-01, FIELD1-SYS1-02
            assert len(db.channel_map) == 2
            assert "FIELD1-SYS1-01" in db.channel_map
        finally:
            Path(db_path).unlink()

    def test_legacy_format_no_validation(self):
        """Legacy format (separate fields) doesn't validate naming pattern automatically."""
        # This would be out of sync and fails at channel building time (not at load time)
        legacy_db = {
            "hierarchy_definition": ["system", "device", "field"],
            "naming_pattern": "{system}:{wrong}:{field}",  # Out of sync!
            "hierarchy_config": {
                "levels": {
                    "system": {"type": "tree"},
                    "device": {"type": "instances"},
                    "field": {"type": "tree"},
                }
            },
            "tree": {
                "SYS1": {
                    "DEVICE": {
                        "_expansion": {"_type": "range", "_pattern": "{:02d}", "_range": [1, 1]},
                        "FIELD1": {"_description": "Field 1"},
                    }
                }
            },
        }

        db_path = create_temp_database(legacy_db)

        try:
            # Legacy format doesn't validate up-front, fails during channel map building
            with pytest.raises(KeyError, match="wrong"):
                HierarchicalChannelDatabase(db_path)
        finally:
            Path(db_path).unlink()
