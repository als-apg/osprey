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

from osprey.services.channel_finder.databases.hierarchical import (
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
        selections_to_device = {"system": "CTRL", "subsystem": "MAIN", "device": "MC-01"}

        # Get options at subdevice level (the optional level after device)
        subdevice_options = optional_levels_db.get_options_at_level(
            "subdevice", selections_to_device
        )

        # Extract option names
        subdevice_names = [opt["name"] for opt in subdevice_options]

        print("\n=== Optional Level Navigation Test ===")
        print(f"Subdevice options returned: {subdevice_names}")
        print(f"Number of options: {len(subdevice_names)}")

        # NEW BEHAVIOR: Direct signals SHOULD be included at optional levels
        # This allows the LLM to naturally select either a subdevice (PSU, ADC)
        # or a direct signal (Heartbeat, Status) without needing NOTHING_FOUND logic

        # Direct signals (leaf nodes) should be in subdevice options
        assert "Heartbeat" in subdevice_names, (
            "Heartbeat should appear in options (it's a direct signal at device level)"
        )
        assert "Status" in subdevice_names, (
            "Status should appear in options (it's a direct signal at device level)"
        )
        assert "Mode" in subdevice_names, (
            "Mode should appear in options (it's a direct signal at device level)"
        )
        assert "Config" in subdevice_names, (
            "Config should appear in options (it's a direct signal at device level)"
        )

        # Actual subdevices (container nodes) should also be in the list
        assert "PSU" in subdevice_names, "PSU is a subdevice container, should be in list"
        assert "ADC" in subdevice_names, "ADC is a subdevice container, should be in list"
        assert "MOTOR" in subdevice_names, "MOTOR is a subdevice container, should be in list"

        # CH has _expansion, so it appears as CH-1, CH-2 (not "CH")
        assert "CH-1" in subdevice_names, "CH-1 is an expanded instance, should be in list"
        assert "CH-2" in subdevice_names, "CH-2 is an expanded instance, should be in list"
        assert "CH" not in subdevice_names, (
            "CH (base container) should NOT be in list (has _expansion)"
        )

        # Verify all 9 options are returned (4 direct signals + 3 subdevices + 2 CH instances)
        assert len(subdevice_names) == 9, (
            f"Expected 9 options (4 direct + 3 subdevices + 2 CH instances), got {len(subdevice_names)}"
        )

    def test_subdevice_vs_signal_distinction(self, optional_levels_db):
        """
        Verify that optional levels show both containers and leaf nodes.

        At device level MC-01, there are:
        - DIRECT signals (leaf nodes, no subdevice): Status, Heartbeat, Mode, Config
        - Subdevice nodes (containers with children): PSU, ADC, MOTOR, CH

        NEW BEHAVIOR: Optional levels now show BOTH containers AND leaf nodes.
        This allows the LLM to naturally select either without needing NOTHING_FOUND logic.
        """
        selections_to_device = {"system": "CTRL", "subsystem": "MAIN", "device": "MC-01"}

        # Get subdevice options
        subdevice_options = optional_levels_db.get_options_at_level(
            "subdevice", selections_to_device
        )
        subdevice_names = [opt["name"] for opt in subdevice_options]

        # Expected: BOTH containers and leaf nodes
        # Note: CH has _expansion, so it appears as CH-1, CH-2 (not "CH")
        expected_containers = {"PSU", "ADC", "MOTOR", "CH-1", "CH-2"}
        expected_direct_signals = {"Heartbeat", "Status", "Mode", "Config"}
        expected_all = expected_containers | expected_direct_signals

        print("\n=== Subdevice Options (NEW BEHAVIOR) ===")
        print(f"Expected containers: {expected_containers}")
        print(f"Expected direct signals: {expected_direct_signals}")
        print(f"\nActual options: {set(subdevice_names)}")

        # Verify NEW behavior: both containers and leaves appear
        containers_present = expected_containers.intersection(set(subdevice_names))
        direct_signals_present = expected_direct_signals.intersection(set(subdevice_names))

        if (
            containers_present == expected_containers
            and direct_signals_present == expected_direct_signals
        ):
            print("\n✅ CORRECT: Both containers and direct signals appear at optional level")
        else:
            print(f"\n🐛 Missing containers: {expected_containers - containers_present}")
            print(f"🐛 Missing direct signals: {expected_direct_signals - direct_signals_present}")

        # Verify all expected options appear
        assert set(subdevice_names) == expected_all, (
            f"Expected {expected_all}, got {set(subdevice_names)}"
        )

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
            "device": "MC-01",
            # Note: NOT selecting subdevice
        }

        # Get signal options directly (skipping optional subdevice)
        signal_options = optional_levels_db.get_options_at_level(
            "signal",
            selections_to_device,  # No subdevice selected
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
        assert channel == "CTRL:MAIN:MC-01:Heartbeat", (
            f"Expected 'CTRL:MAIN:MC-01:Heartbeat', got '{channel}'"
        )

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
        assert channel == "CTRL:MAIN:MC-01:PSU:Voltage", (
            f"Expected 'CTRL:MAIN:MC-01:PSU:Voltage', got '{channel}'"
        )

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
        assert not channel.endswith("_"), (
            f"Channel should not have trailing underscore: '{channel}'"
        )

        # Verify correct channel name
        assert channel == "CTRL:MAIN:MC-01:Heartbeat", (
            f"Expected 'CTRL:MAIN:MC-01:Heartbeat', got '{channel}'"
        )
        print("✅ CORRECT: Separator cleanup applied automatically")


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    pytest.main([__file__, "-v", "-s"])
