"""Tests for the OSPREY type registry — single source of truth for type metadata."""

import json
import re

import pytest

from osprey.stores.type_registry import (
    ARTIFACT_TYPES,
    CATEGORIES,
    TOOL_TYPES,
    get_artifact_types,
    get_tool_types,
    load_categories_from_config,
    register_category,
    registry_to_api_dict,
    valid_category_keys,
)

HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


class TestTypeDefs:
    """Every TypeDef must have a non-empty label and a valid 6-digit hex colour."""

    @pytest.mark.parametrize("key,td", list(ARTIFACT_TYPES.items()), ids=list(ARTIFACT_TYPES))
    def test_artifact_type_fields(self, key, td):
        assert td.key == key
        assert td.label, f"artifact type {key!r} has empty label"
        assert HEX_RE.match(td.color), f"artifact type {key!r} has invalid colour {td.color!r}"

    @pytest.mark.parametrize("key,td", list(TOOL_TYPES.items()), ids=list(TOOL_TYPES))
    def test_tool_type_fields(self, key, td):
        assert td.key == key
        assert td.label, f"tool type {key!r} has empty label"
        assert HEX_RE.match(td.color), f"tool type {key!r} has invalid colour {td.color!r}"


class TestPublicAPI:
    def test_get_artifact_types_returns_copy(self):
        a = get_artifact_types()
        a["bogus"] = None  # mutate
        assert "bogus" not in ARTIFACT_TYPES

    def test_get_tool_types_returns_copy(self):
        t = get_tool_types()
        t["bogus"] = None
        assert "bogus" not in TOOL_TYPES

    def test_valid_category_keys_matches_categories(self):
        assert valid_category_keys() == set(CATEGORIES)


class TestRegistryToAPIDict:
    def test_json_serialisable(self):
        d = registry_to_api_dict()
        # Should not raise
        serialised = json.dumps(d)
        assert isinstance(serialised, str)

    def test_structure(self):
        d = registry_to_api_dict()
        assert set(d.keys()) == {"artifact_types", "tool_types", "categories"}
        for domain in d.values():
            for _key, info in domain.items():
                assert "label" in info
                assert "color" in info

    def test_all_artifact_types_present(self):
        d = registry_to_api_dict()
        assert set(d["artifact_types"]) == set(ARTIFACT_TYPES)

    def test_all_tool_types_present(self):
        d = registry_to_api_dict()
        assert set(d["tool_types"]) == set(TOOL_TYPES)


class TestKnownCategoryTypes:
    """Verify that common category strings are in the registry."""

    KNOWN_CATEGORIES = [
        "channel_values",
        "write_results",
        "code_output",
        "visualization",
        "dashboard",
        "document",
        "screenshot",
        "agent_response",
        "logbook_research",
        "search_results",
    ]

    @pytest.mark.parametrize("cat", KNOWN_CATEGORIES)
    def test_category_registered(self, cat):
        assert cat in valid_category_keys(), f"{cat!r} not in registry"


class TestBugFixes:
    """Regression: the bugs identified in the plan must stay fixed."""

    def test_code_output_in_categories(self):
        assert "code_output" in CATEGORIES

    def test_channel_list_not_in_categories(self):
        """Phantom type — no producer ever existed."""
        assert "channel_list" not in CATEGORIES


class TestRegisterCategory:
    """Tests for the runtime register_category() API."""

    def setup_method(self):
        """Snapshot CATEGORIES so we can restore after mutation."""
        self._snapshot = dict(CATEGORIES)

    def teardown_method(self):
        """Restore CATEGORIES to pre-test state."""
        CATEGORIES.clear()
        CATEGORIES.update(self._snapshot)

    def test_register_new_category(self):
        register_category("beam_diag", "Beam Diagnostics", "#f59e0b")
        assert "beam_diag" in CATEGORIES
        assert "beam_diag" in valid_category_keys()
        td = CATEGORIES["beam_diag"]
        assert td.key == "beam_diag"
        assert td.label == "Beam Diagnostics"
        assert td.color == "#f59e0b"

    def test_override_existing_category(self):
        original = CATEGORIES["archiver_data"]
        register_category("archiver_data", "EPICS Archiver", "#1e40af")
        td = CATEGORIES["archiver_data"]
        assert td.label == "EPICS Archiver"
        assert td.color == "#1e40af"
        assert td.label != original.label

    def test_invalid_color_raises(self):
        with pytest.raises(ValueError, match="must be #RRGGBB"):
            register_category("bad", "Bad", "red")

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            register_category("", "Label", "#aabbcc")

    def test_empty_label_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            register_category("key", "", "#aabbcc")


class TestLoadCategoriesFromConfig:
    """Tests for load_categories_from_config()."""

    def setup_method(self):
        self._snapshot = dict(CATEGORIES)

    def teardown_method(self):
        CATEGORIES.clear()
        CATEGORIES.update(self._snapshot)

    def test_loads_from_config(self, monkeypatch):
        """Config with a custom category registers it."""

        class _FakeConfigBuilder:
            def get(self, key, default=None):
                if key == "categories":
                    return {
                        "vacuum_status": {
                            "label": "Vacuum Status",
                            "color": "#ef4444",
                        }
                    }
                return default

        monkeypatch.setattr(
            "osprey.utils.config.get_config_builder",
            lambda **kw: _FakeConfigBuilder(),
        )

        count = load_categories_from_config()
        assert count == 1
        assert "vacuum_status" in CATEGORIES
        assert CATEGORIES["vacuum_status"].label == "Vacuum Status"

    def test_no_config_returns_zero(self, monkeypatch):
        """When config has no categories key, returns 0."""

        class _EmptyConfigBuilder:
            def get(self, key, default=None):
                return default

        monkeypatch.setattr(
            "osprey.utils.config.get_config_builder",
            lambda **kw: _EmptyConfigBuilder(),
        )

        count = load_categories_from_config()
        assert count == 0
