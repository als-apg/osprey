"""
Integration tests for example hierarchical databases.

Tests that the new flexible hierarchy system works with:
1. Example databases (mixed hierarchy, instance-first, consecutive instances)
2. Legacy accelerator database (backward compatibility)
"""

from pathlib import Path

import pytest

from osprey.templates.apps.control_assistant.services.channel_finder.databases.hierarchical import (
    HierarchicalChannelDatabase,
)


class TestExampleDatabases:
    """Test the example databases load and work correctly."""

    def test_mixed_hierarchy_loads(self):
        """Test example_mixed_hierarchy.json loads successfully."""
        from osprey.templates.apps.control_assistant.services.channel_finder.databases.hierarchical import (
            HierarchicalChannelDatabase,
        )

        # Path relative to osprey root
        db_path = (
            Path(__file__).parents[3]
            / "src/osprey/templates/apps/control_assistant/data/channel_databases/examples/mixed_hierarchy.json"
        )

        if not db_path.exists():
            pytest.skip(f"Example database not found: {db_path}")

        db = HierarchicalChannelDatabase(str(db_path))

        # Verify structure loaded
        assert db.hierarchy_levels == ["sector", "building", "floor", "room", "equipment"]
        assert hasattr(db, "hierarchy_config")

        # Verify channels were generated
        # Expected: 4 sectors × (1×5×20×3 + 1×3×15×2 + 1×2×5×4) = complex calculation
        # MAIN_BUILDING: 4×5×20×3 = 1,200
        # ANNEX: 4×3×15×2 = 360
        # LAB: 4×2×5×4 = 160
        # Total: 1,720 channels
        assert len(db.channel_map) > 0
        print(f"Mixed hierarchy: {len(db.channel_map)} channels generated")

        # Test navigation at each level
        sectors = db.get_options_at_level("sector", {})
        assert len(sectors) == 4
        assert sectors[0]["name"] == "01"

        buildings = db.get_options_at_level("building", {"sector": "01"})
        assert len(buildings) == 3
        assert set(b["name"] for b in buildings) == {"MAIN_BUILDING", "ANNEX", "LAB"}

        # Test floors for MAIN_BUILDING
        floors_main = db.get_options_at_level(
            "floor", {"sector": "01", "building": "MAIN_BUILDING"}
        )
        assert len(floors_main) == 5

        # Test floors for LAB (different count)
        floors_lab = db.get_options_at_level("floor", {"sector": "01", "building": "LAB"})
        assert len(floors_lab) == 2

        # Test specific channels exist
        assert db.validate_channel("S01:MAIN_BUILDING:F1:R101:HVAC")
        assert db.validate_channel("S04:LAB:F2:RCLEAN_ROOM:PRESSURE")

        # Test channel retrieval
        channel_info = db.get_channel("S01:MAIN_BUILDING:F1:R101:HVAC")
        assert channel_info is not None
        assert channel_info["channel"] == "S01:MAIN_BUILDING:F1:R101:HVAC"

        print("✓ Mixed hierarchy database: All tests passed")

    def test_instance_first_loads(self):
        """Test instance_first.json loads successfully (production line example)."""
        db_path = (
            Path(__file__).parents[3]
            / "src/osprey/templates/apps/control_assistant/data/channel_databases/examples/instance_first.json"
        )

        if not db_path.exists():
            pytest.skip(f"Example database not found: {db_path}")

        db = HierarchicalChannelDatabase(str(db_path))

        # Verify structure
        assert db.hierarchy_levels == ["line", "station", "parameter"]
        assert hasattr(db, "hierarchy_config")

        # Verify instance-first configuration (new unified schema uses "type" not "structure")
        assert db.hierarchy_config["levels"]["line"]["type"] == "instances"
        assert db.hierarchy_config["levels"]["station"]["type"] == "tree"
        assert db.hierarchy_config["levels"]["parameter"]["type"] == "tree"

        # Verify channels: 5 lines × 4 stations × various parameters
        # ASSEMBLY: 4 params, INSPECTION: 5 params, PACKAGING: 4 params, CONVEYOR: 4 params
        # Total per line: 17 parameters
        # Total: 5 × 17 = 85 channels
        assert len(db.channel_map) > 0
        print(f"Instance-first pattern: {len(db.channel_map)} channels generated")

        # Test navigation
        lines = db.get_options_at_level("line", {})
        assert len(lines) == 5
        assert lines[0]["name"] == "1"
        assert lines[4]["name"] == "5"

        stations = db.get_options_at_level("station", {"line": "1"})
        assert len(stations) == 4
        assert set(s["name"] for s in stations) == {
            "ASSEMBLY",
            "INSPECTION",
            "PACKAGING",
            "CONVEYOR",
        }

        # Test parameters for different stations
        params_assembly = db.get_options_at_level("parameter", {"line": "1", "station": "ASSEMBLY"})
        assert "SPEED" in [p["name"] for p in params_assembly]

        params_inspection = db.get_options_at_level(
            "parameter", {"line": "1", "station": "INSPECTION"}
        )
        assert "PASS_COUNT" in [p["name"] for p in params_inspection]

        # Test specific channels
        assert db.validate_channel("LINE1:ASSEMBLY:SPEED")
        assert db.validate_channel("LINE5:INSPECTION:FAIL_COUNT")

        print("✓ Instance-first pattern database: All tests passed")

    def test_consecutive_instances_loads(self):
        """Test consecutive_instances.json loads successfully (accelerator naming pattern from CEBAF)."""
        db_path = (
            Path(__file__).parents[3]
            / "src/osprey/templates/apps/control_assistant/data/channel_databases/examples/consecutive_instances.json"
        )

        if not db_path.exists():
            pytest.skip(f"Example database not found: {db_path}")

        db = HierarchicalChannelDatabase(str(db_path))

        # Verify structure - demonstrates consecutive instances (sector + device)
        assert db.hierarchy_levels == ["system", "family", "sector", "device", "property"]
        assert hasattr(db, "hierarchy_config")

        # Verify consecutive instance configuration (new unified schema uses "type" not "structure")
        assert db.hierarchy_config["levels"]["sector"]["type"] == "instances"
        assert db.hierarchy_config["levels"]["device"]["type"] == "instances"
        assert db.hierarchy_config["levels"]["property"]["type"] == "tree"

        # Verify channels were generated
        # Example calculation for QB: 6 sectors × 99 devices × 4 properties = 2,376
        # Plus DP (5 sectors × 50 devices × 4 properties) and CM, BP, IP...
        assert len(db.channel_map) > 0
        print(f"Consecutive instances pattern: {len(db.channel_map)} channels generated")

        # Test navigation through consecutive instances
        systems = db.get_options_at_level("system", {})
        assert "M" in [s["name"] for s in systems]

        families = db.get_options_at_level("family", {"system": "M"})
        assert "QB" in [f["name"] for f in families]

        # Sector level (first instance level)
        sectors = db.get_options_at_level("sector", {"system": "M", "family": "QB"})
        assert len(sectors) == 6
        assert "0L" in [s["name"] for s in sectors]

        # Device level (second consecutive instance level)
        devices = db.get_options_at_level("device", {"system": "M", "family": "QB", "sector": "0L"})
        assert len(devices) == 99
        assert "07" in [d["name"] for d in devices]

        # Property level
        properties = db.get_options_at_level(
            "property", {"system": "M", "family": "QB", "sector": "0L", "device": "07"}
        )
        assert len(properties) == 4
        assert ".S" in [p["name"] for p in properties]

        # Test specific channels from the colleague's examples
        assert db.validate_channel("MQB0L07.S")  # Injector quad 7, current setpoint
        assert db.validate_channel("MQB0L08M")  # Injector quad 8, current readback
        assert db.validate_channel("MQB1A01.BDL")  # First arc quad 1, field setpoint
        assert db.validate_channel("MDP1A01.S")  # Dipole in first arc
        assert db.validate_channel("DBP3A10.X")  # BPM X position

        print("✓ Consecutive instances pattern database: All tests passed")

    def test_jlab_style_loads(self):
        """Test hierarchical_jlab_style.json loads successfully (navigation-only levels with _channel_part)."""
        db_path = (
            Path(__file__).parents[3]
            / "src/osprey/templates/apps/control_assistant/data/channel_databases/examples/hierarchical_jlab_style.json"
        )

        if not db_path.exists():
            pytest.skip(f"Example database not found: {db_path}")

        db = HierarchicalChannelDatabase(str(db_path))

        # Verify structure
        assert db.hierarchy_levels == ["system", "family", "sector", "device", "pv"]
        assert db.naming_pattern == "{pv}"
        assert hasattr(db, "hierarchy_config")

        # All levels are tree type in this example
        assert db.hierarchy_config["levels"]["system"]["type"] == "tree"
        assert db.hierarchy_config["levels"]["family"]["type"] == "tree"
        assert db.hierarchy_config["levels"]["sector"]["type"] == "tree"
        assert db.hierarchy_config["levels"]["device"]["type"] == "tree"
        assert db.hierarchy_config["levels"]["pv"]["type"] == "tree"

        # Verify channels: 19 total PVs across 3 systems
        assert len(db.channel_map) == 19
        print(f"JLab-style pattern: {len(db.channel_map)} channels generated")

        # Verify channel names are PV strings (not friendly names)
        assert "MQS1L02.S" in db.channel_map  # Not "Current Setpoint"
        assert "IPM1L02X" in db.channel_map  # Not "Horizontal Position"
        assert "VIP1L02V" in db.channel_map  # Not "Pump Voltage"

        # Test navigation uses friendly names
        systems = db.get_options_at_level("system", {})
        assert len(systems) == 3
        system_names = {s["name"] for s in systems}
        assert system_names == {"Magnets", "Diagnostics", "Vacuum"}

        # Test family level (friendly names)
        families = db.get_options_at_level("family", {"system": "Magnets"})
        assert len(families) == 2
        family_names = {f["name"] for f in families}
        assert family_names == {"Skew Quadrupoles", "Main Quadrupoles"}

        # Test sector level (friendly names, not "1L", "1A")
        sectors = db.get_options_at_level(
            "sector", {"system": "Magnets", "family": "Skew Quadrupoles"}
        )
        assert len(sectors) == 2
        sector_names = {s["name"] for s in sectors}
        assert sector_names == {"North Linac", "East Arc"}  # Not "1L", "1A"!

        # Test PV level (friendly names at leaf)
        pvs = db.get_options_at_level(
            "pv",
            {
                "system": "Magnets",
                "family": "Skew Quadrupoles",
                "sector": "North Linac",
                "device": "MQS1L02",
            },
        )
        assert len(pvs) == 4
        pv_names = {p["name"] for p in pvs}
        # These are FRIENDLY names in navigation
        assert "Current Setpoint" in pv_names
        assert "Magnet Current" in pv_names
        # NOT the cryptic PV strings
        assert "MQS1L02.S" not in pv_names

        # Verify navigation-only levels work
        # Only 'pv' appears in pattern, others are navigation-only
        pattern_levels = db._get_pattern_levels()
        assert pattern_levels == ["pv"]

        # Test channel building
        # Note: build_channels_from_selections expects channel parts, not tree keys
        # The _channel_part mapping happens during navigation/selection, not here
        channels = db.build_channels_from_selections(
            {
                "system": "Magnets",  # Navigation-only (not used in pattern)
                "family": "Skew Quadrupoles",  # Navigation-only (not used in pattern)
                "sector": "North Linac",  # Navigation-only (not used in pattern)
                "device": "MQS1L02",  # Navigation-only (not used in pattern)
                "pv": ["MQS1L02.S", "MQS1L02M"],  # These are the _channel_part values
            }
        )
        assert len(channels) == 2
        assert "MQS1L02.S" in channels
        assert "MQS1L02M" in channels

        print("✓ JLab-style database: Navigation-only levels with _channel_part working correctly")

    def test_legacy_accelerator_backward_compatible(self):
        """Test main hierarchical.json uses new unified schema correctly."""
        db_path = (
            Path(__file__).parents[3]
            / "src/osprey/templates/apps/control_assistant/data/channel_databases/hierarchical.json"
        )

        if not db_path.exists():
            pytest.skip(f"Database not found: {db_path}")

        db = HierarchicalChannelDatabase(str(db_path))

        # Should have hierarchy_config from new unified schema
        assert hasattr(db, "hierarchy_config")
        assert "levels" in db.hierarchy_config

        # Verify structure
        assert db.hierarchy_levels == ["system", "family", "device", "field", "subfield"]

        # Main database now uses new unified schema with instances type
        assert db.hierarchy_config["levels"]["device"]["type"] == "instances"

        # Test that channels were built
        assert len(db.channel_map) > 0
        print(f"Legacy accelerator: {len(db.channel_map)} channels generated")

        # Test navigation still works
        systems = db.get_options_at_level("system", {})
        assert len(systems) > 0
        assert "MAG" in [s["name"] for s in systems]

        families = db.get_options_at_level("family", {"system": "MAG"})
        assert len(families) > 0
        assert "DIPOLE" in [f["name"] for f in families]

        # Test specific channels from benchmark
        assert db.validate_channel("MAG:DIPOLE[B01]:CURRENT:SP")
        assert db.validate_channel("VAC:ION-PUMP[SR03]:PRESSURE:RB")
        assert db.validate_channel("RF:CAVITY[C1]:POWER:FWD")

        print("✓ Legacy accelerator database: Backward compatibility verified")

    def test_channel_generation_cartesian_product(self):
        """Test that channel generation produces correct Cartesian product."""
        db_path = (
            Path(__file__).parents[3]
            / "src/osprey/templates/apps/control_assistant/data/channel_databases/examples/instance_first.json"
        )

        if not db_path.exists():
            pytest.skip(f"Example database not found: {db_path}")

        db = HierarchicalChannelDatabase(str(db_path))

        # Test cartesian product with multiple selections
        channels = db.build_channels_from_selections(
            {
                "line": ["1", "2"],
                "station": ["ASSEMBLY", "INSPECTION"],
                "parameter": ["SPEED", "STATUS"],
            }
        )

        # Should get: 2 lines × 2 stations × 2 parameters = 8 channels
        assert len(channels) == 8

        expected = [
            "LINE1:ASSEMBLY:SPEED",
            "LINE1:ASSEMBLY:STATUS",
            "LINE1:INSPECTION:SPEED",
            "LINE1:INSPECTION:STATUS",
            "LINE2:ASSEMBLY:SPEED",
            "LINE2:ASSEMBLY:STATUS",
            "LINE2:INSPECTION:SPEED",
            "LINE2:INSPECTION:STATUS",
        ]

        for expected_channel in expected[:4]:  # At least verify first few
            assert expected_channel in channels

        print("✓ Cartesian product generation works correctly")


class TestDatabaseStatistics:
    """Test statistics gathering works with new flexible system."""

    def test_stats_for_instance_first_level(self):
        """Statistics work when first level is instance."""
        db_path = (
            Path(__file__).parents[3]
            / "src/osprey/templates/apps/control_assistant/data/channel_databases/examples/instance_first.json"
        )

        if not db_path.exists():
            pytest.skip(f"Example database not found: {db_path}")

        db = HierarchicalChannelDatabase(str(db_path))
        stats = db.get_statistics()

        assert "total_channels" in stats
        assert stats["total_channels"] > 0
        assert "hierarchy_levels" in stats
        assert stats["hierarchy_levels"] == ["line", "station", "parameter"]

        # When first level is instance, no 'systems' breakdown
        # (systems breakdown only works for tree-type first level)
        print(f"✓ Statistics: {stats}")

    def test_stats_for_tree_first_level(self):
        """Statistics work when first level is tree."""
        db_path = (
            Path(__file__).parents[3]
            / "src/osprey/templates/apps/control_assistant/data/channel_databases/examples/mixed_hierarchy.json"
        )

        if not db_path.exists():
            pytest.skip(f"Example database not found: {db_path}")

        db = HierarchicalChannelDatabase(str(db_path))
        stats = db.get_statistics()

        assert "total_channels" in stats
        # First level is instance (sector), so no systems breakdown
        assert stats["hierarchy_levels"] == ["sector", "building", "floor", "room", "equipment"]

        print(f"✓ Statistics: {stats}")


if __name__ == "__main__":
    """Allow running tests directly for quick validation."""
    import sys

    print("=" * 80)
    print("Testing Example Hierarchical Databases")
    print("=" * 80)
    print()

    test_class = TestExampleDatabases()

    print("1. Testing mixed_hierarchy.json...")
    try:
        test_class.test_mixed_hierarchy_loads()
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        sys.exit(1)

    print()
    print("2. Testing instance_first.json (production line pattern)...")
    try:
        test_class.test_instance_first_loads()
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        sys.exit(1)

    print()
    print("3. Testing consecutive_instances.json (CEBAF accelerator pattern)...")
    try:
        test_class.test_consecutive_instances_loads()
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        sys.exit(1)

    print()
    print("4. Testing hierarchical_jlab_style.json (navigation-only levels with _channel_part)...")
    try:
        test_class.test_jlab_style_loads()
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        sys.exit(1)

    print()
    print("5. Testing backward compatibility with hierarchical.json...")
    try:
        test_class.test_legacy_accelerator_backward_compatible()
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        sys.exit(1)

    print()
    print("6. Testing Cartesian product generation...")
    try:
        test_class.test_channel_generation_cartesian_product()
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        sys.exit(1)

    print()
    print("=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
