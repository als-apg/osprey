"""Tests for Channel Finder database CRUD operations.

Uses tmp_path fixtures with real JSON files. No mocking needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from osprey.interfaces.channel_finder.database_crud import (
    CrudError,
    _atomic_write,
    _hier_level_info,
    _load_json,
    hier_add_node,
    hier_count_descendants,
    hier_delete_node,
    hier_edit_expansion,
    hier_edit_node,
    hier_get_expansion,
    ic_add_channel,
    ic_delete_channel,
    ic_update_channel,
    ml_add_channel,
    ml_add_family,
    ml_count_family_channels,
    ml_delete_channel,
    ml_delete_family,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def hier_db(tmp_path: Path) -> Path:
    """Create a minimal hierarchical database file."""
    db = tmp_path / "hier.json"
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
    db.write_text(json.dumps(data, indent=2))
    return db


@pytest.fixture()
def ml_db(tmp_path: Path) -> Path:
    """Create a minimal middle layer database file."""
    db = tmp_path / "ml.json"
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
    db.write_text(json.dumps(data, indent=2))
    return db


@pytest.fixture()
def ic_db(tmp_path: Path) -> Path:
    """Create a minimal in-context (flat) database file."""
    db = tmp_path / "ic.json"
    data = [
        {"channel": "SR:BPM:01:X", "address": "SR:BPM:01:X", "description": "BPM X"},
        {"channel": "SR:BPM:01:Y", "address": "SR:BPM:01:Y", "description": "BPM Y"},
    ]
    db.write_text(json.dumps(data, indent=2))
    return db


@pytest.fixture()
def ic_dict_db(tmp_path: Path) -> Path:
    """Create an in-context database in dict format."""
    db = tmp_path / "ic_dict.json"
    data = {
        "channels": [
            {"channel": "SR:MAG:01", "address": "SR:MAG:01", "description": "Magnet 1"},
        ]
    }
    db.write_text(json.dumps(data, indent=2))
    return db


# ---------------------------------------------------------------------------
# _reload_registry is a no-op in tests (registries not initialized)
# ---------------------------------------------------------------------------

_RELOAD_PATCH = patch("osprey.interfaces.channel_finder.database_crud._reload_registry")


# ---------------------------------------------------------------------------
# File I/O tests
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    """Tests for atomic file write with backup."""

    def test_atomic_write_creates_backup(self, tmp_path: Path):
        f = tmp_path / "test.json"
        f.write_text('{"old": true}')

        _atomic_write(f, {"new": True})

        assert f.exists()
        assert json.loads(f.read_text()) == {"new": True}
        bak = tmp_path / "test.json.bak"
        assert bak.exists()
        assert json.loads(bak.read_text()) == {"old": True}

    def test_atomic_write_new_file(self, tmp_path: Path):
        f = tmp_path / "new.json"
        _atomic_write(f, {"created": True})

        assert f.exists()
        assert json.loads(f.read_text()) == {"created": True}
        # No backup for new file
        assert not (tmp_path / "new.json.bak").exists()

    def test_atomic_write_preserves_on_error(self, tmp_path: Path):
        f = tmp_path / "safe.json"
        original = {"safe": True}
        f.write_text(json.dumps(original))

        # Force an error during write by providing an unserializable object
        class BadObj:
            pass

        with pytest.raises(TypeError):
            _atomic_write(f, BadObj())

        # Original file untouched
        assert json.loads(f.read_text()) == original


# ---------------------------------------------------------------------------
# Hierarchical CRUD tests
# ---------------------------------------------------------------------------


class TestHierarchicalCrud:
    """Tests for hierarchical database CRUD operations."""

    def test_hier_add_node_at_root(self, hier_db: Path):
        with _RELOAD_PATCH:
            result = hier_add_node(hier_db, level="system", parent_selections={}, name="LI")

        assert result["success"] is True
        data = _load_json(hier_db)
        assert "LI" in data["tree"]

    def test_hier_add_node_nested(self, hier_db: Path):
        with _RELOAD_PATCH:
            result = hier_add_node(
                hier_db,
                level="device",
                parent_selections={"system": "SR"},
                name="CORR",
                description="Corrector",
            )

        assert result["success"] is True
        data = _load_json(hier_db)
        assert "CORR" in data["tree"]["SR"]
        assert data["tree"]["SR"]["CORR"]["_description"] == "Corrector"

    def test_hier_add_duplicate_raises(self, hier_db: Path):
        with _RELOAD_PATCH, pytest.raises(CrudError, match="already exists"):
            hier_add_node(hier_db, level="system", parent_selections={}, name="SR")

    def test_hier_edit_node(self, hier_db: Path):
        with _RELOAD_PATCH:
            result = hier_edit_node(
                hier_db,
                level="device",
                selections={"system": "SR"},
                old_name="BPM",
                new_name="BPM_v2",
            )

        assert result["success"] is True
        data = _load_json(hier_db)
        assert "BPM_v2" in data["tree"]["SR"]
        assert "BPM" not in data["tree"]["SR"]
        # Children preserved
        assert "X" in data["tree"]["SR"]["BPM_v2"]

    def test_hier_rename_nonexistent_raises(self, hier_db: Path):
        with _RELOAD_PATCH, pytest.raises(CrudError, match="not found"):
            hier_edit_node(
                hier_db,
                level="device",
                selections={"system": "SR"},
                old_name="NONEXISTENT",
                new_name="NEW",
            )

    def test_hier_rename_to_existing_raises(self, hier_db: Path):
        with _RELOAD_PATCH, pytest.raises(CrudError, match="already exists"):
            hier_edit_node(
                hier_db,
                level="device",
                selections={"system": "SR"},
                old_name="BPM",
                new_name="QUAD",
            )

    def test_hier_edit_description_only(self, hier_db: Path):
        with _RELOAD_PATCH:
            result = hier_edit_node(
                hier_db,
                level="device",
                selections={"system": "SR"},
                old_name="BPM",
                description="Updated BPM description",
            )

        assert result["success"] is True
        data = _load_json(hier_db)
        assert "BPM" in data["tree"]["SR"]  # name unchanged
        assert data["tree"]["SR"]["BPM"]["_description"] == "Updated BPM description"

    def test_hier_edit_name_and_description(self, hier_db: Path):
        with _RELOAD_PATCH:
            result = hier_edit_node(
                hier_db,
                level="device",
                selections={"system": "SR"},
                old_name="BPM",
                new_name="BPM_v2",
                description="New BPM",
            )

        assert result["success"] is True
        data = _load_json(hier_db)
        assert "BPM_v2" in data["tree"]["SR"]
        assert "BPM" not in data["tree"]["SR"]
        assert data["tree"]["SR"]["BPM_v2"]["_description"] == "New BPM"

    def test_hier_edit_clear_description(self, hier_db: Path):
        with _RELOAD_PATCH:
            result = hier_edit_node(
                hier_db,
                level="device",
                selections={"system": "SR"},
                old_name="BPM",
                description="",
            )

        assert result["success"] is True
        data = _load_json(hier_db)
        assert "_description" not in data["tree"]["SR"]["BPM"]

    def test_hier_delete_node(self, hier_db: Path):
        with _RELOAD_PATCH:
            result = hier_delete_node(
                hier_db,
                level="system",
                selections={},
                name="BR",
            )

        assert result["success"] is True
        data = _load_json(hier_db)
        assert "BR" not in data["tree"]

    def test_hier_delete_node_cascades(self, hier_db: Path):
        with _RELOAD_PATCH:
            result = hier_delete_node(
                hier_db,
                level="device",
                selections={"system": "SR"},
                name="BPM",
            )

        assert result["success"] is True
        assert result["affected_channels"] > 0
        data = _load_json(hier_db)
        assert "BPM" not in data["tree"]["SR"]

    def test_hier_delete_nonexistent_raises(self, hier_db: Path):
        with _RELOAD_PATCH, pytest.raises(CrudError, match="not found"):
            hier_delete_node(hier_db, level="system", selections={}, name="GONE")

    def test_hier_count_descendants(self, hier_db: Path):
        with _RELOAD_PATCH:
            count = hier_count_descendants(
                hier_db,
                level="device",
                selections={"system": "SR"},
                name="BPM",
            )

        assert count >= 2  # At least X and Y


# ---------------------------------------------------------------------------
# Middle Layer CRUD tests
# ---------------------------------------------------------------------------


class TestMiddleLayerCrud:
    """Tests for middle layer database CRUD operations."""

    def test_ml_add_family(self, ml_db: Path):
        with _RELOAD_PATCH:
            result = ml_add_family(ml_db, system="SR", family="CORR", description="Corrector")

        assert result["success"] is True
        data = _load_json(ml_db)
        assert "CORR" in data["SR"]
        assert data["SR"]["CORR"]["_description"] == "Corrector"

    def test_ml_add_family_duplicate_raises(self, ml_db: Path):
        with _RELOAD_PATCH, pytest.raises(CrudError, match="already exists"):
            ml_add_family(ml_db, system="SR", family="BPM")

    def test_ml_delete_family_cascades(self, ml_db: Path):
        with _RELOAD_PATCH:
            result = ml_delete_family(ml_db, system="SR", family="BPM")

        assert result["success"] is True
        assert result["affected_channels"] >= 3  # X, Y, XSet
        data = _load_json(ml_db)
        assert "BPM" not in data["SR"]

    def test_ml_add_channel(self, ml_db: Path):
        with _RELOAD_PATCH:
            result = ml_add_channel(
                ml_db,
                system="SR",
                family="BPM",
                field="Monitor",
                channel_name="SR02:BPM:X",
            )

        assert result["success"] is True
        data = _load_json(ml_db)
        assert "SR02:BPM:X" in data["SR"]["BPM"]["Monitor"]["ChannelNames"]

    def test_ml_add_channel_creates_field(self, ml_db: Path):
        with _RELOAD_PATCH:
            result = ml_add_channel(
                ml_db,
                system="SR",
                family="BPM",
                field="NewField",
                channel_name="SR:BPM:NEW",
            )

        assert result["success"] is True
        data = _load_json(ml_db)
        assert data["SR"]["BPM"]["NewField"]["ChannelNames"] == ["SR:BPM:NEW"]

    def test_ml_delete_channel(self, ml_db: Path):
        with _RELOAD_PATCH:
            result = ml_delete_channel(
                ml_db,
                system="SR",
                family="BPM",
                field="Monitor",
                channel_name="SR01:BPM:X",
            )

        assert result["success"] is True
        data = _load_json(ml_db)
        assert "SR01:BPM:X" not in data["SR"]["BPM"]["Monitor"]["ChannelNames"]

    def test_ml_delete_channel_not_found_raises(self, ml_db: Path):
        with _RELOAD_PATCH, pytest.raises(CrudError, match="not found"):
            ml_delete_channel(
                ml_db,
                system="SR",
                family="BPM",
                field="Monitor",
                channel_name="NONEXISTENT",
            )

    def test_ml_count_family_channels(self, ml_db: Path):
        with _RELOAD_PATCH:
            count = ml_count_family_channels(ml_db, system="SR", family="BPM")

        assert count >= 3  # SR01:BPM:X, SR01:BPM:Y, SR01:BPM:XSet


# ---------------------------------------------------------------------------
# In-Context CRUD tests
# ---------------------------------------------------------------------------


class TestInContextCrud:
    """Tests for in-context database CRUD operations."""

    def test_ic_add_channel(self, ic_db: Path):
        with _RELOAD_PATCH:
            result = ic_add_channel(
                ic_db,
                channel="SR:MAG:01",
                address="SR:MAG:01",
                description="Magnet 1",
            )

        assert result["success"] is True
        data = _load_json(ic_db)
        names = [ch["channel"] for ch in data]
        assert "SR:MAG:01" in names

    def test_ic_add_duplicate_raises(self, ic_db: Path):
        with _RELOAD_PATCH, pytest.raises(CrudError, match="already exists"):
            ic_add_channel(ic_db, channel="SR:BPM:01:X")

    def test_ic_delete_channel(self, ic_db: Path):
        with _RELOAD_PATCH:
            result = ic_delete_channel(ic_db, channel="SR:BPM:01:X")

        assert result["success"] is True
        data = _load_json(ic_db)
        names = [ch["channel"] for ch in data]
        assert "SR:BPM:01:X" not in names
        assert "SR:BPM:01:Y" in names  # other channel preserved

    def test_ic_delete_nonexistent_raises(self, ic_db: Path):
        with _RELOAD_PATCH, pytest.raises(CrudError, match="not found"):
            ic_delete_channel(ic_db, channel="NONEXISTENT")

    def test_ic_update_channel(self, ic_db: Path):
        with _RELOAD_PATCH:
            result = ic_update_channel(
                ic_db,
                channel="SR:BPM:01:X",
                new_description="Updated description",
                new_address="NEW:ADDR",
            )

        assert result["success"] is True
        data = _load_json(ic_db)
        ch = next(c for c in data if c["channel"] == "SR:BPM:01:X")
        assert ch["description"] == "Updated description"
        assert ch["address"] == "NEW:ADDR"

    def test_ic_update_nonexistent_raises(self, ic_db: Path):
        with _RELOAD_PATCH, pytest.raises(CrudError, match="not found"):
            ic_update_channel(ic_db, channel="NONEXISTENT", new_description="nope")

    def test_ic_add_channel_dict_format(self, ic_dict_db: Path):
        with _RELOAD_PATCH:
            result = ic_add_channel(
                ic_dict_db,
                channel="SR:MAG:02",
                description="Magnet 2",
            )

        assert result["success"] is True
        data = _load_json(ic_dict_db)
        names = [ch["channel"] for ch in data["channels"]]
        assert "SR:MAG:02" in names


# ---------------------------------------------------------------------------
# Hierarchical CRUD with instance-type levels
# ---------------------------------------------------------------------------


@pytest.fixture()
def hier_db_with_instances(tmp_path: Path) -> Path:
    """Hierarchical DB matching production schema with instance-type device level.

    Hierarchy: system (tree) → family (tree) → device (instances) → field (tree) → subfield (tree)
    The device level uses _expansion to generate B01..B04. Fields/subfields live
    inside the DEVICE container alongside _expansion.
    """
    db = tmp_path / "hier_inst.json"
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
                            "pattern": "B{:02d}",
                            "range": [1, 4],
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
                            "pattern": "Q{:02d}",
                            "range": [1, 6],
                        },
                        "Setpoint": {},
                        "Readback": {},
                    },
                },
            },
        },
    }
    db.write_text(json.dumps(data, indent=2))
    return db


class TestHierarchicalCrudWithInstances:
    """Tests for CRUD operations that navigate through instance-type levels."""

    def test_level_info_extracts_types(self, hier_db_with_instances: Path):
        data = _load_json(hier_db_with_instances)
        names, types = _hier_level_info(data)
        assert names == ["system", "family", "device", "field", "subfield"]
        assert types["system"] == "tree"
        assert types["device"] == "instances"
        assert types["field"] == "tree"

    def test_add_field_through_instance_level(self, hier_db_with_instances: Path):
        """Adding a field when navigating through an instance-level device."""
        with _RELOAD_PATCH:
            result = hier_add_node(
                hier_db_with_instances,
                level="field",
                parent_selections={"system": "SR", "family": "BPM", "device": "B01"},
                name="Sum",
                description="Sum signal",
            )

        assert result["success"] is True
        data = _load_json(hier_db_with_instances)
        device_node = data["tree"]["SR"]["BPM"]["DEVICE"]
        assert "Sum" in device_node
        assert device_node["Sum"]["_description"] == "Sum signal"

    def test_add_subfield_through_instance_level(self, hier_db_with_instances: Path):
        """Adding a subfield when navigating through an instance-level device."""
        with _RELOAD_PATCH:
            result = hier_add_node(
                hier_db_with_instances,
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
        data = _load_json(hier_db_with_instances)
        x_node = data["tree"]["SR"]["BPM"]["DEVICE"]["X"]
        assert "Filtered" in x_node

    def test_different_instance_values_reach_same_node(self, hier_db_with_instances: Path):
        """B01 and B04 should both navigate to the same DEVICE container."""
        with _RELOAD_PATCH:
            hier_add_node(
                hier_db_with_instances,
                level="field",
                parent_selections={"system": "SR", "family": "BPM", "device": "B04"},
                name="Current",
            )

        data = _load_json(hier_db_with_instances)
        device_node = data["tree"]["SR"]["BPM"]["DEVICE"]
        assert "Current" in device_node

    def test_rename_field_through_instance_level(self, hier_db_with_instances: Path):
        with _RELOAD_PATCH:
            result = hier_edit_node(
                hier_db_with_instances,
                level="field",
                selections={"system": "SR", "family": "BPM", "device": "B01"},
                old_name="Y",
                new_name="Y_pos",
            )

        assert result["success"] is True
        data = _load_json(hier_db_with_instances)
        device_node = data["tree"]["SR"]["BPM"]["DEVICE"]
        assert "Y_pos" in device_node
        assert "Y" not in device_node

    def test_delete_field_through_instance_level(self, hier_db_with_instances: Path):
        with _RELOAD_PATCH:
            result = hier_delete_node(
                hier_db_with_instances,
                level="field",
                selections={"system": "SR", "family": "BPM", "device": "B01"},
                name="Y",
            )

        assert result["success"] is True
        data = _load_json(hier_db_with_instances)
        device_node = data["tree"]["SR"]["BPM"]["DEVICE"]
        assert "Y" not in device_node
        assert "X" in device_node  # other field preserved

    def test_count_descendants_through_instance_level(self, hier_db_with_instances: Path):
        with _RELOAD_PATCH:
            count = hier_count_descendants(
                hier_db_with_instances,
                level="field",
                selections={"system": "SR", "family": "BPM", "device": "B01"},
                name="X",
            )

        # X has children Raw and Calibrated (both empty dicts = leaf nodes)
        assert count >= 2

    def test_tree_level_crud_still_works(self, hier_db_with_instances: Path):
        """Regression: tree-level CRUD unaffected by instance-level support."""
        with _RELOAD_PATCH:
            result = hier_add_node(
                hier_db_with_instances,
                level="family",
                parent_selections={"system": "SR"},
                name="CORR",
                description="Corrector magnets",
            )

        assert result["success"] is True
        data = _load_json(hier_db_with_instances)
        assert "CORR" in data["tree"]["SR"]

    def test_add_duplicate_through_instance_raises(self, hier_db_with_instances: Path):
        with _RELOAD_PATCH, pytest.raises(CrudError, match="already exists"):
            hier_add_node(
                hier_db_with_instances,
                level="field",
                parent_selections={"system": "SR", "family": "BPM", "device": "B01"},
                name="X",  # already exists
            )

    # ---- Expansion editing tests ----

    def test_get_expansion(self, hier_db_with_instances: Path):
        with _RELOAD_PATCH:
            result = hier_get_expansion(
                hier_db_with_instances,
                level="device",
                selections={"system": "SR", "family": "BPM"},
            )

        assert result["expansion"]["pattern"] == "B{:02d}"
        assert result["expansion"]["range"] == [1, 4]

    def test_edit_expansion_range(self, hier_db_with_instances: Path):
        """Change range [1,4] to [1,8], verify JSON."""
        with _RELOAD_PATCH:
            result = hier_edit_expansion(
                hier_db_with_instances,
                level="device",
                selections={"system": "SR", "family": "BPM"},
                range_start=1,
                range_end=8,
            )

        assert result["success"] is True
        data = _load_json(hier_db_with_instances)
        expansion = data["tree"]["SR"]["BPM"]["DEVICE"]["_expansion"]
        assert expansion["range"] == [1, 8]
        # Pattern unchanged
        assert expansion["pattern"] == "B{:02d}"

    def test_edit_expansion_pattern(self, hier_db_with_instances: Path):
        """Change pattern, verify."""
        with _RELOAD_PATCH:
            result = hier_edit_expansion(
                hier_db_with_instances,
                level="device",
                selections={"system": "SR", "family": "BPM"},
                pattern="BPM{:03d}",
            )

        assert result["success"] is True
        data = _load_json(hier_db_with_instances)
        expansion = data["tree"]["SR"]["BPM"]["DEVICE"]["_expansion"]
        assert expansion["pattern"] == "BPM{:03d}"
        # Range unchanged
        assert expansion["range"] == [1, 4]

    def test_edit_expansion_not_found_bad_path(self, hier_db_with_instances: Path):
        """Bad parent path raises CrudError."""
        with _RELOAD_PATCH, pytest.raises(CrudError, match="not found"):
            hier_edit_expansion(
                hier_db_with_instances,
                level="device",
                selections={"system": "SR", "family": "NONEXISTENT"},
            )

    def test_edit_expansion_no_container(self, hier_db: Path):
        """Tree-only DB with no _expansion raises CrudError."""
        with _RELOAD_PATCH, pytest.raises(CrudError, match="No expansion container"):
            hier_edit_expansion(
                hier_db,
                level="device",
                selections={"system": "SR"},
            )
