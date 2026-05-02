"""Integration tests: generated databases load through pipeline database classes."""

from __future__ import annotations

import json

import pytest

from osprey.services.channel_finder.benchmarks.generator import (
    TEMPLATE_DB_PATH,
    TIER_1,
    TIER_3,
    expand_hierarchy,
    format_hierarchical,
    format_in_context,
    format_middle_layer,
)


@pytest.fixture(scope="module")
def generated_dbs(tmp_path_factory):
    """Generate all three databases once per module."""
    tmp = tmp_path_factory.mktemp("schema_compat")

    tree_data = json.loads(TEMPLATE_DB_PATH.read_text(encoding="utf-8"))
    channels = expand_hierarchy(tree_data)

    ic_data = format_in_context(channels, TIER_1)
    ic_path = tmp / "in_context.json"
    ic_path.write_text(json.dumps(ic_data))

    hier_data = format_hierarchical(tree_data, TIER_3)
    hier_path = tmp / "hierarchical.json"
    hier_path.write_text(json.dumps(hier_data))

    ml_data = format_middle_layer(channels, TIER_3)
    ml_path = tmp / "middle_layer.json"
    ml_path.write_text(json.dumps(ml_data))

    return {
        "in_context": (ic_path, ic_data),
        "hierarchical": (hier_path, hier_data),
        "middle_layer": (ml_path, ml_data),
    }


class TestInContextCompat:
    """In-context database loads via ChannelDatabase."""

    def test_load_and_get_channel(self, generated_dbs):
        from osprey.services.channel_finder.databases.flat import ChannelDatabase

        path, data = generated_dbs["in_context"]
        db = ChannelDatabase(str(path))
        db.load_database()

        # Alias lookup returns correct PV
        first = data["channels"][0]
        result = db.get_channel(first["channel"])
        assert result is not None
        assert result["address"] == first["address"]

    def test_statistics(self, generated_dbs):
        from osprey.services.channel_finder.databases.flat import ChannelDatabase

        path, _ = generated_dbs["in_context"]
        db = ChannelDatabase(str(path))
        db.load_database()
        stats = db.get_statistics()
        assert stats["total_channels"] == TIER_1.target_count


class TestHierarchicalCompat:
    """Hierarchical database loads via HierarchicalChannelDatabase."""

    def test_load_and_get_options(self, generated_dbs):
        from osprey.services.channel_finder.databases.hierarchical import (
            HierarchicalChannelDatabase,
        )

        path, _ = generated_dbs["hierarchical"]
        db = HierarchicalChannelDatabase(str(path))
        db.load_database()

        # get_options_at_level at root "ring" level should return rings
        options = db.get_options_at_level("ring", {})
        ring_names = [o["name"] for o in options]
        assert "SR" in ring_names
        assert "BR" in ring_names
        assert "BTS" in ring_names

    def test_build_channels(self, generated_dbs):
        from osprey.services.channel_finder.databases.hierarchical import (
            HierarchicalChannelDatabase,
        )

        path, _ = generated_dbs["hierarchical"]
        db = HierarchicalChannelDatabase(str(path))
        db.load_database()

        # Get a valid device instance from the expansion to use in selections
        device_options = db.get_options_at_level(
            "device",
            {"ring": "SR", "system": "MAG", "family": "DIPOLE"},
        )
        assert len(device_options) > 0
        first_device = device_options[0]["name"]

        # Build channels for a known path through the hierarchy
        channels = db.build_channels_from_selections(
            {
                "ring": "SR",
                "system": "MAG",
                "family": "DIPOLE",
                "device": first_device,
                "field": "CURRENT",
                "subfield": "SP",
            }
        )
        assert len(channels) > 0
        for ch in channels:
            assert ch.startswith("SR:MAG:DIPOLE:")

    def test_statistics(self, generated_dbs):
        from osprey.services.channel_finder.databases.hierarchical import (
            HierarchicalChannelDatabase,
        )

        path, _ = generated_dbs["hierarchical"]
        db = HierarchicalChannelDatabase(str(path))
        db.load_database()
        stats = db.get_statistics()
        assert stats["total_channels"] == TIER_3.target_count


class TestMiddleLayerCompat:
    """Middle-layer database loads via MiddleLayerDatabase."""

    def test_load_and_list_channel_names(self, generated_dbs):
        from osprey.services.channel_finder.databases.middle_layer import MiddleLayerDatabase

        path, _ = generated_dbs["middle_layer"]
        db = MiddleLayerDatabase(str(path))
        db.load_database()

        channels = db.list_channel_names("SR", "BPM", "POSITION", "X")
        assert len(channels) > 0

    def test_sector_filtering(self, generated_dbs):
        from osprey.services.channel_finder.databases.middle_layer import MiddleLayerDatabase

        path, _ = generated_dbs["middle_layer"]
        db = MiddleLayerDatabase(str(path))
        db.load_database()

        all_ch = db.list_channel_names("SR", "BPM", "POSITION", "X")
        filtered = db.list_channel_names("SR", "BPM", "POSITION", "X", sectors=[1])
        assert 0 < len(filtered) < len(all_ch)

    def test_get_common_names(self, generated_dbs):
        from osprey.services.channel_finder.databases.middle_layer import MiddleLayerDatabase

        path, _ = generated_dbs["middle_layer"]
        db = MiddleLayerDatabase(str(path))
        db.load_database()

        names = db.get_common_names("SR", "BPM")
        assert names is not None
        assert len(names) > 0

    def test_statistics(self, generated_dbs):
        from osprey.services.channel_finder.databases.middle_layer import MiddleLayerDatabase

        path, _ = generated_dbs["middle_layer"]
        db = MiddleLayerDatabase(str(path))
        db.load_database()
        stats = db.get_statistics()
        assert stats["total_channels"] == TIER_3.target_count
