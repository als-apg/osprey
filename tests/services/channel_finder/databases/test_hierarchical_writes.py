"""Tests for HierarchicalChannelDatabase write methods."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.channel_finder.core.base_database import DatabaseWriteError
from osprey.services.channel_finder.databases.hierarchical import (
    HierarchicalChannelDatabase,
)


@pytest.fixture()
def hier_db(tmp_path: Path) -> HierarchicalChannelDatabase:
    """Create a simple 3-level hierarchical database (all tree-type levels)."""
    db_path = tmp_path / "hier.json"
    data = {
        "hierarchy": {
            "levels": [
                {"name": "system", "type": "tree"},
                {"name": "device", "type": "tree"},
                {"name": "signal", "type": "tree"},
            ],
            "naming_pattern": "{system}:{device}:{signal}",
        },
        "tree": {
            "SR": {
                "_description": "Storage Ring",
                "BPM": {
                    "_description": "Beam Position Monitor",
                    "X": {},
                    "Y": {},
                },
                "QUAD": {
                    "_description": "Quadrupole",
                },
            },
            "BR": {
                "_description": "Booster Ring",
            },
        },
    }
    db_path.write_text(json.dumps(data, indent=2))
    return HierarchicalChannelDatabase(str(db_path))


@pytest.fixture()
def hier_db_with_instances(tmp_path: Path) -> HierarchicalChannelDatabase:
    """5-level DB with instance-type device level.

    Hierarchy: system(tree) -> family(tree) -> device(instances) -> field(tree) -> subfield(tree)
    """
    db_path = tmp_path / "hier_inst.json"
    data = {
        "hierarchy": {
            "levels": [
                {"name": "system", "type": "tree"},
                {"name": "family", "type": "tree"},
                {"name": "device", "type": "instances"},
                {"name": "field", "type": "tree"},
                {"name": "subfield", "type": "tree"},
            ],
            "naming_pattern": "{system}:{family}:{device}:{field}:{subfield}",
        },
        "tree": {
            "SR": {
                "_description": "Storage Ring",
                "BPM": {
                    "_description": "Beam Position Monitors",
                    "DEVICE": {
                        "_expansion": {
                            "_type": "range",
                            "_pattern": "B{:02d}",
                            "_range": [1, 4],
                        },
                        "X": {
                            "_description": "Horizontal position",
                            "Raw": {},
                            "Calibrated": {},
                        },
                        "Y": {
                            "_description": "Vertical position",
                        },
                    },
                },
                "QUAD": {
                    "_description": "Quadrupoles",
                    "DEVICE": {
                        "_expansion": {
                            "_type": "range",
                            "_pattern": "Q{:02d}",
                            "_range": [1, 6],
                        },
                        "Setpoint": {},
                        "Readback": {},
                    },
                },
            },
        },
    }
    db_path.write_text(json.dumps(data, indent=2))
    return HierarchicalChannelDatabase(str(db_path))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _disk_data(db: HierarchicalChannelDatabase) -> dict:
    """Read the on-disk JSON for the database."""
    return json.loads(Path(db.db_path).read_text())


# ---------------------------------------------------------------------------
# Serialize
# ---------------------------------------------------------------------------


class TestSerialize:
    def test_serialize_returns_raw_data(self, hier_db: HierarchicalChannelDatabase):
        result = hier_db._serialize()
        assert isinstance(result, dict)
        assert "tree" in result
        assert "hierarchy" in result

    def test_serialize_reflects_mutations(self, hier_db: HierarchicalChannelDatabase):
        hier_db.tree["NEW"] = {}
        result = hier_db._serialize()
        assert "NEW" in result["tree"]


# ---------------------------------------------------------------------------
# Simple tree CRUD (no instance levels)
# ---------------------------------------------------------------------------


class TestAddNode:
    def test_add_node_at_root(self, hier_db: HierarchicalChannelDatabase):
        result = hier_db.add_node(level="system", parent_selections={}, name="LI")

        assert result["success"] is True
        assert "LI" in hier_db.tree
        assert "LI" in _disk_data(hier_db)["tree"]

    def test_add_node_nested(self, hier_db: HierarchicalChannelDatabase):
        result = hier_db.add_node(
            level="device",
            parent_selections={"system": "SR"},
            name="CORR",
            description="Corrector",
        )

        assert result["success"] is True
        assert "CORR" in hier_db.tree["SR"]
        assert hier_db.tree["SR"]["CORR"]["_description"] == "Corrector"
        on_disk = _disk_data(hier_db)
        assert on_disk["tree"]["SR"]["CORR"]["_description"] == "Corrector"

    def test_add_duplicate_raises(self, hier_db: HierarchicalChannelDatabase):
        with pytest.raises(DatabaseWriteError, match="already exists"):
            hier_db.add_node(level="system", parent_selections={}, name="SR")


class TestEditNode:
    def test_rename(self, hier_db: HierarchicalChannelDatabase):
        result = hier_db.edit_node(
            level="device",
            selections={"system": "SR"},
            old_name="BPM",
            new_name="BPM_v2",
        )

        assert result["success"] is True
        assert "BPM_v2" in hier_db.tree["SR"]
        assert "BPM" not in hier_db.tree["SR"]
        # Children preserved
        assert "X" in hier_db.tree["SR"]["BPM_v2"]
        on_disk = _disk_data(hier_db)
        assert "BPM_v2" in on_disk["tree"]["SR"]

    def test_rename_nonexistent_raises(self, hier_db: HierarchicalChannelDatabase):
        with pytest.raises(DatabaseWriteError, match="not found"):
            hier_db.edit_node(
                level="device",
                selections={"system": "SR"},
                old_name="NONEXISTENT",
                new_name="NEW",
            )

    def test_rename_to_existing_raises(self, hier_db: HierarchicalChannelDatabase):
        with pytest.raises(DatabaseWriteError, match="already exists"):
            hier_db.edit_node(
                level="device",
                selections={"system": "SR"},
                old_name="BPM",
                new_name="QUAD",
            )

    def test_edit_description_only(self, hier_db: HierarchicalChannelDatabase):
        result = hier_db.edit_node(
            level="device",
            selections={"system": "SR"},
            old_name="BPM",
            description="Updated BPM description",
        )

        assert result["success"] is True
        assert "BPM" in hier_db.tree["SR"]
        assert hier_db.tree["SR"]["BPM"]["_description"] == "Updated BPM description"

    def test_edit_name_and_description(self, hier_db: HierarchicalChannelDatabase):
        result = hier_db.edit_node(
            level="device",
            selections={"system": "SR"},
            old_name="BPM",
            new_name="BPM_v2",
            description="New BPM",
        )

        assert result["success"] is True
        assert "BPM_v2" in hier_db.tree["SR"]
        assert hier_db.tree["SR"]["BPM_v2"]["_description"] == "New BPM"

    def test_edit_clear_description(self, hier_db: HierarchicalChannelDatabase):
        result = hier_db.edit_node(
            level="device",
            selections={"system": "SR"},
            old_name="BPM",
            description="",
        )

        assert result["success"] is True
        assert "_description" not in hier_db.tree["SR"]["BPM"]


class TestDeleteNode:
    def test_delete_node(self, hier_db: HierarchicalChannelDatabase):
        result = hier_db.delete_node(level="system", selections={}, name="BR")

        assert result["success"] is True
        assert "BR" not in hier_db.tree
        assert "BR" not in _disk_data(hier_db)["tree"]

    def test_delete_node_cascades(self, hier_db: HierarchicalChannelDatabase):
        result = hier_db.delete_node(level="device", selections={"system": "SR"}, name="BPM")

        assert result["success"] is True
        assert result["affected_channels"] > 0
        assert "BPM" not in hier_db.tree["SR"]

    def test_delete_nonexistent_raises(self, hier_db: HierarchicalChannelDatabase):
        with pytest.raises(DatabaseWriteError, match="not found"):
            hier_db.delete_node(level="system", selections={}, name="GONE")


class TestCountDescendants:
    def test_count_descendants(self, hier_db: HierarchicalChannelDatabase):
        impact = hier_db.count_descendants(level="device", selections={"system": "SR"}, name="BPM")

        assert impact["channels"] >= 2  # At least X and Y
        assert "signal" in impact


# ---------------------------------------------------------------------------
# Instance-level CRUD
# ---------------------------------------------------------------------------


class TestInstanceLevelAddNode:
    def test_add_field_through_instance_level(
        self, hier_db_with_instances: HierarchicalChannelDatabase
    ):
        result = hier_db_with_instances.add_node(
            level="field",
            parent_selections={"system": "SR", "family": "BPM", "device": "B01"},
            name="Sum",
            description="Sum signal",
        )

        assert result["success"] is True
        device_node = hier_db_with_instances.tree["SR"]["BPM"]["DEVICE"]
        assert "Sum" in device_node
        assert device_node["Sum"]["_description"] == "Sum signal"

    def test_add_subfield_through_instance_level(
        self, hier_db_with_instances: HierarchicalChannelDatabase
    ):
        result = hier_db_with_instances.add_node(
            level="subfield",
            parent_selections={
                "system": "SR",
                "family": "BPM",
                "device": "B01",
                "field": "X",
            },
            name="Filtered",
        )

        assert result["success"] is True
        x_node = hier_db_with_instances.tree["SR"]["BPM"]["DEVICE"]["X"]
        assert "Filtered" in x_node

    def test_different_instance_values_reach_same_node(
        self, hier_db_with_instances: HierarchicalChannelDatabase
    ):
        hier_db_with_instances.add_node(
            level="field",
            parent_selections={"system": "SR", "family": "BPM", "device": "B04"},
            name="Current",
        )

        device_node = hier_db_with_instances.tree["SR"]["BPM"]["DEVICE"]
        assert "Current" in device_node

    def test_add_duplicate_through_instance_raises(
        self, hier_db_with_instances: HierarchicalChannelDatabase
    ):
        with pytest.raises(DatabaseWriteError, match="already exists"):
            hier_db_with_instances.add_node(
                level="field",
                parent_selections={
                    "system": "SR",
                    "family": "BPM",
                    "device": "B01",
                },
                name="X",
            )

    def test_tree_level_crud_still_works(self, hier_db_with_instances: HierarchicalChannelDatabase):
        result = hier_db_with_instances.add_node(
            level="family",
            parent_selections={"system": "SR"},
            name="CORR",
            description="Corrector magnets",
        )

        assert result["success"] is True
        assert "CORR" in hier_db_with_instances.tree["SR"]


class TestInstanceLevelEditNode:
    def test_rename_field_through_instance_level(
        self, hier_db_with_instances: HierarchicalChannelDatabase
    ):
        result = hier_db_with_instances.edit_node(
            level="field",
            selections={"system": "SR", "family": "BPM", "device": "B01"},
            old_name="Y",
            new_name="Y_pos",
        )

        assert result["success"] is True
        device_node = hier_db_with_instances.tree["SR"]["BPM"]["DEVICE"]
        assert "Y_pos" in device_node
        assert "Y" not in device_node


class TestInstanceLevelDeleteNode:
    def test_delete_field_through_instance_level(
        self, hier_db_with_instances: HierarchicalChannelDatabase
    ):
        result = hier_db_with_instances.delete_node(
            level="field",
            selections={"system": "SR", "family": "BPM", "device": "B01"},
            name="Y",
        )

        assert result["success"] is True
        device_node = hier_db_with_instances.tree["SR"]["BPM"]["DEVICE"]
        assert "Y" not in device_node
        assert "X" in device_node


class TestInstanceLevelCountDescendants:
    def test_count_through_instance_level(
        self, hier_db_with_instances: HierarchicalChannelDatabase
    ):
        impact = hier_db_with_instances.count_descendants(
            level="field",
            selections={"system": "SR", "family": "BPM", "device": "B01"},
            name="X",
        )
        # X has children Raw and Calibrated
        assert impact["channels"] >= 2

    def test_expansion_aware_breakdown(self, hier_db_with_instances: HierarchicalChannelDatabase):
        impact = hier_db_with_instances.count_descendants(
            level="family",
            selections={"system": "SR"},
            name="QUAD",
        )

        # QUAD: 6 devices (expansion [1,6]), 2 fields (Setpoint, Readback)
        assert impact["device"] == 6
        assert impact["field"] == 2 * 6  # 2 fields per device x 6 devices
        assert impact["channels"] == 12  # 2 leaf fields x 6 devices


# ---------------------------------------------------------------------------
# Expansion get/edit
# ---------------------------------------------------------------------------


class TestGetExpansion:
    def test_get_expansion(self, hier_db_with_instances: HierarchicalChannelDatabase):
        result = hier_db_with_instances.get_expansion(
            level="device",
            selections={"system": "SR", "family": "BPM"},
        )

        assert result["expansion"]["pattern"] == "B{:02d}"
        assert result["expansion"]["range"] == [1, 4]

    def test_get_expansion_new_family_no_container(
        self, hier_db_with_instances: HierarchicalChannelDatabase
    ):
        hier_db_with_instances.add_node(
            level="family",
            parent_selections={"system": "SR"},
            name="NEWFAM",
            description="Brand new family",
        )
        result = hier_db_with_instances.get_expansion(
            level="device",
            selections={"system": "SR", "family": "NEWFAM"},
        )

        assert result["expansion"]["pattern"] == ""
        assert result["expansion"]["range"] == [1, 1]


class TestEditExpansion:
    def test_edit_expansion_range(self, hier_db_with_instances: HierarchicalChannelDatabase):
        result = hier_db_with_instances.edit_expansion(
            level="device",
            selections={"system": "SR", "family": "BPM"},
            range_start=1,
            range_end=8,
        )

        assert result["success"] is True
        assert result["expansion"]["range"] == [1, 8]
        assert result["expansion"]["pattern"] == "B{:02d}"
        on_disk = _disk_data(hier_db_with_instances)
        raw = on_disk["tree"]["SR"]["BPM"]["DEVICE"]["_expansion"]
        assert raw["_range"] == [1, 8]

    def test_edit_expansion_pattern(self, hier_db_with_instances: HierarchicalChannelDatabase):
        result = hier_db_with_instances.edit_expansion(
            level="device",
            selections={"system": "SR", "family": "BPM"},
            pattern="BPM{:03d}",
        )

        assert result["success"] is True
        assert result["expansion"]["pattern"] == "BPM{:03d}"
        on_disk = _disk_data(hier_db_with_instances)
        raw = on_disk["tree"]["SR"]["BPM"]["DEVICE"]["_expansion"]
        assert raw["_pattern"] == "BPM{:03d}"

    def test_edit_expansion_bad_path_raises(
        self, hier_db_with_instances: HierarchicalChannelDatabase
    ):
        with pytest.raises(DatabaseWriteError, match="not found"):
            hier_db_with_instances.edit_expansion(
                level="device",
                selections={"system": "SR", "family": "NONEXISTENT"},
            )

    def test_edit_expansion_no_container_tree_only(self, hier_db: HierarchicalChannelDatabase):
        with pytest.raises(DatabaseWriteError, match="No expansion container"):
            hier_db.edit_expansion(
                level="device",
                selections={"system": "SR"},
            )

    def test_edit_expansion_auto_creates_container(
        self, hier_db_with_instances: HierarchicalChannelDatabase
    ):
        hier_db_with_instances.add_node(
            level="family",
            parent_selections={"system": "SR"},
            name="SEXT",
            description="Sextupoles",
        )
        result = hier_db_with_instances.edit_expansion(
            level="device",
            selections={"system": "SR", "family": "SEXT"},
            pattern="S{:02d}",
            range_start=1,
            range_end=12,
        )

        assert result["success"] is True
        assert result["expansion"]["pattern"] == "S{:02d}"
        assert result["expansion"]["range"] == [1, 12]
        on_disk = _disk_data(hier_db_with_instances)
        raw = on_disk["tree"]["SR"]["SEXT"]["DEVICE"]["_expansion"]
        assert raw["_pattern"] == "S{:02d}"
        assert raw["_range"] == [1, 12]
        assert raw["_type"] == "range"


# ---------------------------------------------------------------------------
# Channel map rebuild
# ---------------------------------------------------------------------------


class TestChannelMapRebuild:
    def test_add_node_updates_channel_map(self, hier_db: HierarchicalChannelDatabase):
        assert not hier_db.validate_channel("SR:CORR:")
        hier_db.add_node(
            level="signal",
            parent_selections={"system": "SR", "device": "BPM"},
            name="Z",
        )
        assert hier_db.validate_channel("SR:BPM:Z")

    def test_delete_node_updates_channel_map(self, hier_db: HierarchicalChannelDatabase):
        assert hier_db.validate_channel("SR:BPM:X")
        hier_db.delete_node(
            level="signal",
            selections={"system": "SR", "device": "BPM"},
            name="X",
        )
        assert not hier_db.validate_channel("SR:BPM:X")
