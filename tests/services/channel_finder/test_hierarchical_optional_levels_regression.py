"""
Regression tests for hierarchical channel finder optional levels navigation.

These tests verify correct behavior when navigating optional hierarchy levels,
particularly edge cases where signals exist directly at a level that has an
optional child level (e.g., signals that skip an optional subdevice level).

Key scenarios tested:
- Direct signals at device level should NOT appear as subdevice options
- build_channels_from_selections() correctly handles missing optional levels
- Separator cleanup is applied when optional levels are skipped
- Channel map generation produces correct channels for both paths

These tests serve as regression tests to ensure the optional level navigation
logic continues to work correctly.
"""

import pytest
from src.osprey.templates.apps.control_assistant.services.channel_finder.databases.hierarchical import (
    HierarchicalChannelDatabase,
)


@pytest.fixture
def optional_levels_db():
    """
    Load the actual optional_levels.json example database.

    This database has the hierarchy:
    - system (tree)
    - subsystem (tree)
    - device (instances)
    - subdevice (tree, optional)
    - signal (tree)
    - suffix (tree, optional)
    """
    db_path = "src/osprey/templates/apps/control_assistant/data/channel_databases/examples/optional_levels.json"
    return HierarchicalChannelDatabase(db_path)


class TestOptionalLevelNavigation:
    """
    Test correct navigation behavior with optional levels.

    Verifies that optional tree levels correctly distinguish between
    container nodes (which belong at the current level) and leaf/terminal
    nodes (which belong to the next level).
    """

    def test_direct_signals_not_in_subdevice_options(self, optional_levels_db):
        """
        Verify that direct signals don't appear as subdevice options.

        Database structure:
            CTRL → MAIN → MC-01 → Heartbeat (direct signal, no subdevice)
                                → Status (direct signal, no subdevice)
                                → PSU (actual subdevice) → Voltage/Current

        When querying at the optional subdevice level, only actual subdevices
        (nodes with children) should be returned, not leaf nodes that are
        direct signals.
        """
        # Navigate to device level
        selections_to_device = {
            "system": "CTRL",
            "subsystem": "MAIN",
            "device": "MC-01"
        }

        # Get options at subdevice level (the optional level after device)
        subdevice_options = optional_levels_db.get_options_at_level(
            "subdevice",
            selections_to_device
        )

        # Extract option names
        subdevice_names = [opt["name"] for opt in subdevice_options]

        print("\n=== Optional Level Navigation Test ===")
        print(f"Subdevice options returned: {subdevice_names}")
        print(f"Number of options: {len(subdevice_names)}")

        # Direct signals should NOT be in subdevice options
        assert "Heartbeat" not in subdevice_names, "Heartbeat should not appear as subdevice (it's a direct signal)"
        assert "Status" not in subdevice_names, "Status should not appear as subdevice (it's a direct signal)"
        assert "Mode" not in subdevice_names, "Mode should not appear as subdevice (it's a direct signal)"
        assert "Config" not in subdevice_names, "Config should not appear as subdevice (it's a direct signal)"

        # Only actual subdevices should be in the list
        assert "PSU" in subdevice_names, "PSU is a real subdevice, should be in list"
        assert "ADC" in subdevice_names, "ADC is a real subdevice, should be in list"
        assert "MOTOR" in subdevice_names, "MOTOR is a real subdevice, should be in list"
        assert "CH" in subdevice_names, "CH is a real subdevice, should be in list"

        # Verify only 4 subdevices are returned
        assert len(subdevice_names) == 4, f"Expected 4 subdevices, got {len(subdevice_names)}"

    def test_subdevice_vs_signal_distinction(self, optional_levels_db):
        """
        Verify correct distinction between subdevices and direct signals.

        At device level MC-01, there are:
        - DIRECT signals (no subdevice): Status, Heartbeat, Mode, Config
        - Subdevice nodes (have children): PSU, ADC, MOTOR, CH

        Only actual subdevices should appear as subdevice options.
        """
        selections_to_device = {
            "system": "CTRL",
            "subsystem": "MAIN",
            "device": "MC-01"
        }

        # Get subdevice options
        subdevice_options = optional_levels_db.get_options_at_level(
            "subdevice",
            selections_to_device
        )
        subdevice_names = [opt["name"] for opt in subdevice_options]

        # Expected: Only actual subdevices should appear
        expected_subdevices = {"PSU", "ADC", "MOTOR", "CH"}

        # NOT expected: Direct signals should NOT appear as subdevices
        unexpected_in_subdevices = {"Heartbeat", "Status", "Mode", "Config"}

        print("\n=== Subdevice vs Signal Distinction ===")
        print(f"Expected subdevices only: {expected_subdevices}")
        print(f"Should NOT be in subdevices: {unexpected_in_subdevices}")
        print(f"\nActual subdevice options: {set(subdevice_names)}")

        # Verify correct behavior
        wrong_nodes_appearing = unexpected_in_subdevices.intersection(set(subdevice_names))
        if wrong_nodes_appearing:
            print(f"\n🐛 REGRESSION: These signals are wrongly appearing as subdevices: {wrong_nodes_appearing}")
        else:
            print("\n✅ CORRECT: No signals appearing as subdevices")

        # Verify only actual subdevices appear
        assert set(subdevice_names) == expected_subdevices, f"Expected {expected_subdevices}, got {set(subdevice_names)}"

        # Verify no signals appear as subdevices
        assert len(wrong_nodes_appearing) == 0, f"Signals should not appear as subdevices, found: {wrong_nodes_appearing}"

    def test_channel_map_has_direct_signals(self, optional_levels_db):
        """
        Verify that channel map generation correctly handles direct signals.

        The channel_map should contain entries for both:
        - Direct signals that skip optional levels (e.g., CTRL:MAIN:MC-01:Heartbeat)
        - Signals accessed via optional levels (e.g., CTRL:MAIN:MC-01:PSU:Voltage)
        """
        channel_map = optional_levels_db.channel_map

        # These should exist (direct signals)
        expected_direct = [
            "CTRL:MAIN:MC-01:Heartbeat",
            "CTRL:MAIN:MC-02:Heartbeat",
            "CTRL:MAIN:MC-03:Heartbeat",
            "CTRL:MAIN:MC-01:Status",
            "CTRL:MAIN:MC-02:Status",
            "CTRL:MAIN:MC-03:Status",
        ]

        # These should exist (signals via subdevice)
        expected_via_subdevice = [
            "CTRL:MAIN:MC-01:PSU:Voltage",
            "CTRL:MAIN:MC-01:PSU:Current",
        ]

        print("\n=== Channel Map Verification ===")
        print(f"Total channels in map: {len(channel_map)}")

        # Check direct signals
        for channel in expected_direct:
            if channel in channel_map:
                print(f"✓ Found: {channel}")
            else:
                print(f"✗ Missing: {channel}")

        # Check subdevice signals
        for channel in expected_via_subdevice:
            if channel in channel_map:
                print(f"✓ Found: {channel}")
            else:
                print(f"✗ Missing: {channel}")

        # Verify all channels exist
        for channel in expected_direct:
            assert channel in channel_map, f"Channel map should contain direct signal: {channel}"

        for channel in expected_via_subdevice:
            assert channel in channel_map, f"Channel map should contain subdevice signal: {channel}"

    def test_signal_options_when_skipping_subdevice(self, optional_levels_db):
        """
        Verify behavior when getting signal options without selecting subdevice.

        When navigating directly to signal level (skipping optional subdevice),
        the system should return all direct children at that position.
        """
        selections_to_device = {
            "system": "CTRL",
            "subsystem": "MAIN",
            "device": "MC-01"
            # Note: NOT selecting subdevice
        }

        # Get signal options directly (skipping optional subdevice)
        signal_options = optional_levels_db.get_options_at_level(
            "signal",
            selections_to_device  # No subdevice selected
        )

        signal_names = [opt["name"] for opt in signal_options]

        print("\n=== Skipping Optional Subdevice Level ===")
        print(f"Signal options when subdevice not selected: {signal_names}")
        print(f"Number of options: {len(signal_names)}")

        # Verify we get all direct children (both direct signals and subdevices)
        assert len(signal_names) > 0, "Should return options when skipping optional level"


class TestChannelBuildingWithOptionalLevels:
    """
    Test build_channels_from_selections() with optional levels.

    Verifies that channel names are built correctly when optional levels
    are omitted from the selections dictionary.
    """

    def test_build_channel_skipping_optional_levels(self, optional_levels_db):
        """
        Verify channel building works when optional levels are omitted.

        When optional levels are not included in the selections dict,
        they should be treated as empty strings and separator cleanup
        should be applied.
        """
        # Selections that skip both optional levels (subdevice and suffix)
        selections = {
            "system": "CTRL",
            "subsystem": "MAIN",
            "device": "MC-01",
            # subdevice: not provided (optional, skipped)
            "signal": "Heartbeat",
            # suffix: not provided (optional, skipped)
        }

        channels = optional_levels_db.build_channels_from_selections(selections)

        print("\n=== Building Channel With Skipped Optional Levels ===")
        print(f"Selections: {selections}")
        print(f"Built channels: {channels}")

        # Should build channel correctly with optional levels omitted
        assert len(channels) == 1, "Should build exactly one channel, got " + str(len(channels))
        channel = channels[0]

        # Verify no separator artifacts
        assert "::" not in channel, f"Channel has double colon: {channel}"
        assert not channel.endswith(":"), f"Channel has trailing colon: {channel}"
        assert not channel.endswith("_"), f"Channel has trailing underscore: {channel}"

        # Verify correct channel name
        assert channel == "CTRL:MAIN:MC-01:Heartbeat", f"Expected 'CTRL:MAIN:MC-01:Heartbeat', got '{channel}'"

    def test_build_channel_with_one_optional_included(self, optional_levels_db):
        """
        Verify channel building with some optional levels included.

        When subdevice is provided but suffix is not, the channel
        should include the subdevice and apply cleanup to the suffix.
        """
        selections = {
            "system": "CTRL",
            "subsystem": "MAIN",
            "device": "MC-01",
            "subdevice": "PSU",
            "signal": "Voltage",
            # suffix: not provided (optional, skipped)
        }

        channels = optional_levels_db.build_channels_from_selections(selections)

        print("\n=== Building Channel With Partial Optional Levels ===")
        print(f"Selections: {selections}")
        print(f"Built channels: {channels}")

        # Should work with optional suffix omitted
        assert len(channels) == 1, f"Should build exactly one channel, got {len(channels)}"
        channel = channels[0]
        assert channel == "CTRL:MAIN:MC-01:PSU:Voltage", f"Expected 'CTRL:MAIN:MC-01:PSU:Voltage', got '{channel}'"

    def test_build_channel_with_explicit_empty_optionals(self, optional_levels_db):
        """
        Verify that explicit empty strings for optional levels are handled.

        Even when optional levels are explicitly provided as empty strings,
        separator cleanup should be applied automatically.
        """
        # Explicitly provide empty strings for optional levels
        selections = {
            "system": "CTRL",
            "subsystem": "MAIN",
            "device": "MC-01",
            "subdevice": "",  # Explicitly empty
            "signal": "Heartbeat",
            "suffix": "",  # Explicitly empty
        }

        channels = optional_levels_db.build_channels_from_selections(selections)

        print("\n=== Building With Explicit Empty Strings ===")
        print(f"Selections: {selections}")
        print(f"Built channels: {channels}")

        # Should produce clean channel without separator artifacts
        assert len(channels) == 1, f"Should build exactly one channel, got {len(channels)}"
        channel = channels[0]
        print(f"✓ Channel: '{channel}'")

        # Verify no separator artifacts (cleanup is applied automatically)
        assert "::" not in channel, f"Channel should not have double colons: '{channel}'"
        assert not channel.endswith(":"), f"Channel should not have trailing colon: '{channel}'"
        assert not channel.endswith("_"), f"Channel should not have trailing underscore: '{channel}'"

        # Verify correct channel name
        assert channel == "CTRL:MAIN:MC-01:Heartbeat", f"Expected 'CTRL:MAIN:MC-01:Heartbeat', got '{channel}'"
        print("✅ CORRECT: Separator cleanup applied automatically")


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    pytest.main([__file__, "-v", "-s"])

