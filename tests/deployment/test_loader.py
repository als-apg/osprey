"""Tests for deployment configuration parameter loader.

This test module verifies the parameter loading system including YAML imports,
environment variable expansion, and hierarchical parameter access.

Iteration 2: Comprehensive tests for loader.py
Target: 0% → 60%+ coverage
"""

import os
from unittest.mock import patch

import pytest

from osprey.deployment.loader import (
    InvalidParam,
    Params,
    _deep_update_dict,
    _load_yaml,
    load_params,
)


class TestDeepUpdateDict:
    """Test deep dictionary merging functionality."""

    def test_simple_merge(self):
        """Test merging non-nested dictionaries."""
        source = {"a": 1, "b": 2}
        update = {"c": 3}
        _deep_update_dict(source, update)

        assert source == {"a": 1, "b": 2, "c": 3}

    def test_nested_merge(self):
        """Test merging nested dictionaries."""
        source = {"db": {"host": "localhost", "port": 5432}}
        update = {"db": {"name": "myapp"}}
        _deep_update_dict(source, update)

        assert source == {"db": {"host": "localhost", "port": 5432, "name": "myapp"}}

    def test_value_replacement(self):
        """Test that non-dict values replace dict values."""
        source = {"timeout": {"connect": 5, "read": 30}}
        update = {"timeout": 10}
        _deep_update_dict(source, update)

        assert source["timeout"] == 10

    def test_deep_nested_merge(self):
        """Test merging deeply nested dictionaries."""
        source = {"level1": {"level2": {"level3": {"value": 1}}}}
        update = {"level1": {"level2": {"level3": {"new_value": 2}}}}
        _deep_update_dict(source, update)

        assert source["level1"]["level2"]["level3"]["value"] == 1
        assert source["level1"]["level2"]["level3"]["new_value"] == 2

    def test_adds_new_top_level_keys(self):
        """Test adding new top-level keys."""
        source = {"existing": "value"}
        update = {"new_key": "new_value"}
        _deep_update_dict(source, update)

        assert source["new_key"] == "new_value"
        assert source["existing"] == "value"

    def test_overwrites_existing_values(self):
        """Test overwriting existing scalar values."""
        source = {"key": "old_value"}
        update = {"key": "new_value"}
        _deep_update_dict(source, update)

        assert source["key"] == "new_value"

    def test_empty_source_dict(self):
        """Test merging into empty source dictionary."""
        source = {}
        update = {"key": "value"}
        _deep_update_dict(source, update)

        assert source == {"key": "value"}

    def test_empty_update_dict(self):
        """Test merging empty update dictionary."""
        source = {"key": "value"}
        update = {}
        _deep_update_dict(source, update)

        assert source == {"key": "value"}


class TestLoadYAML:
    """Test YAML loading with import processing."""

    def test_load_simple_yaml(self, tmp_path):
        """Test loading a simple YAML file."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("key: value\nnum: 42")

        result = _load_yaml(str(config_file))

        assert result == {"key": "value", "num": 42}

    def test_load_yaml_with_import(self, tmp_path):
        """Test loading YAML with import directive."""
        base_file = tmp_path / "base.yml"
        base_file.write_text("base_key: base_value\nshared: base")

        main_file = tmp_path / "main.yml"
        main_file.write_text("import: base.yml\nmain_key: main_value\nshared: main")

        result = _load_yaml(str(main_file))

        assert result["base_key"] == "base_value"
        assert result["main_key"] == "main_value"
        assert result["shared"] == "main"  # Main overrides base

    def test_load_yaml_with_nested_import(self, tmp_path):
        """Test loading YAML with nested imports."""
        level1 = tmp_path / "level1.yml"
        level1.write_text("l1: value1")

        level2 = tmp_path / "level2.yml"
        level2.write_text("import: level1.yml\nl2: value2")

        level3 = tmp_path / "level3.yml"
        level3.write_text("import: level2.yml\nl3: value3")

        result = _load_yaml(str(level3))

        assert result["l1"] == "value1"
        assert result["l2"] == "value2"
        assert result["l3"] == "value3"

    def test_circular_import_detection(self, tmp_path):
        """Test that circular imports are detected and raise error."""
        file_a = tmp_path / "a.yml"
        file_b = tmp_path / "b.yml"

        file_a.write_text("import: b.yml\nvalue_a: a")
        file_b.write_text("import: a.yml\nvalue_b: b")

        with pytest.raises(ValueError, match="Circular import"):
            _load_yaml(str(file_a))

    def test_import_file_not_found(self, tmp_path):
        """Test error when imported file doesn't exist."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("import: nonexistent.yml\nkey: value")

        with pytest.raises(FileNotFoundError, match="not found"):
            _load_yaml(str(config_file))

    def test_import_must_be_string(self, tmp_path):
        """Test that import value must be a string."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("import: [list, of, files]\nkey: value")

        with pytest.raises(ValueError, match="must be a string"):
            _load_yaml(str(config_file))

    def test_import_relative_to_parent_directory(self, tmp_path):
        """Test that import paths are relative to parent file."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        base = tmp_path / "base.yml"
        base.write_text("base: value")

        child = subdir / "child.yml"
        child.write_text("import: ../base.yml\nchild: value")

        result = _load_yaml(str(child))

        assert result["base"] == "value"
        assert result["child"] == "value"


class TestLoadParams:
    """Test main load_params function."""

    def test_load_params_returns_params_object(self, tmp_path):
        """Test that load_params returns a Params object."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("key: value")

        params = load_params(str(config_file))

        assert isinstance(params, Params)

    def test_load_params_with_dict_data(self, tmp_path):
        """Test loading parameters from dictionary structure."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("database:\n  host: localhost\n  port: 5432")

        params = load_params(str(config_file))

        assert params.database.host == "localhost"
        assert params.database.port == 5432

    def test_load_params_with_import(self, tmp_path):
        """Test load_params with import directive."""
        base = tmp_path / "base.yml"
        base.write_text("timeout: 30")

        main = tmp_path / "main.yml"
        main.write_text("import: base.yml\nretries: 3")

        params = load_params(str(main))

        assert params.timeout == 30
        assert params.retries == 3


class TestParamsClass:
    """Test Params class functionality."""

    def test_params_initialization_with_dict(self):
        """Test creating Params from dictionary."""
        data = {"key": "value", "num": 42}
        params = Params(data, "root")

        assert params.key == "value"
        assert params.num == 42

    def test_params_initialization_with_list(self):
        """Test creating Params from list."""
        data = ["item1", "item2", "item3"]
        params = Params(data, "root")

        assert params[0] == "item1"
        assert params[1] == "item2"
        assert len(params) == 3

    def test_params_initialization_with_scalar(self):
        """Test creating Params from scalar value."""
        params = Params(42, "root")

        assert params() == 42

    def test_params_dot_notation_access(self):
        """Test accessing nested parameters with dot notation."""
        data = {"db": {"host": "localhost", "port": 5432}}
        params = Params(data, "root")

        assert params.db.host == "localhost"
        assert params.db.port == 5432

    def test_params_bracket_notation_access(self):
        """Test accessing parameters with bracket notation."""
        data = {"servers": ["web1", "web2", "api1"]}
        params = Params(data, "root")

        assert params["servers"][0] == "web1"
        assert params.servers[1] == "web2"

    def test_params_environment_variable_expansion(self):
        """Test environment variable expansion in parameters."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            data = {"path": "${TEST_VAR}/data"}
            params = Params(data, "root")

            assert params.path == "test_value/data"

    def test_params_is_valid(self):
        """Test is_valid method returns True for valid Params."""
        params = Params({"key": "value"}, "root")

        assert params.is_valid() is True

    def test_params_bool_evaluation(self):
        """Test boolean evaluation of Params."""
        empty_params = Params({}, "root")
        non_empty_params = Params({"key": "value"}, "root")

        assert not bool(empty_params)
        assert bool(non_empty_params)

    def test_params_len(self):
        """Test len() on Params."""
        list_params = Params([1, 2, 3], "root")
        dict_params = Params({"a": 1, "b": 2}, "root")

        assert len(list_params) == 3
        assert len(dict_params) == 2

    def test_params_contains(self):
        """Test 'in' operator on Params."""
        dict_params = Params({"key": "value"}, "root")
        list_params = Params([1, 2, 3], "root")

        assert "key" in dict_params
        assert "missing" not in dict_params
        assert 1 in list_params

    def test_params_keys(self):
        """Test keys() method on dict Params."""
        data = {"a": 1, "b": 2, "c": 3}
        params = Params(data, "root")

        keys = list(params.keys())
        assert set(keys) == {"a", "b", "c"}

    def test_params_values(self):
        """Test values() method on dict Params."""
        data = {"a": 1, "b": 2}
        params = Params(data, "root")

        values = list(params.values())
        assert 1 in values
        assert 2 in values

    def test_params_items(self):
        """Test items() method on dict Params."""
        data = {"a": 1, "b": 2}
        params = Params(data, "root")

        items = list(params.items())
        # items() returns tuples of (key, Params_object)
        keys = [k for k, v in items]
        assert "a" in keys
        assert "b" in keys

    def test_params_get_with_default(self):
        """Test get() method with default value."""
        params = Params({"existing": "value"}, "root")

        assert params.get("existing") == "value"
        assert params.get("missing", "default") == "default"

    def test_params_copy(self):
        """Test deep copying Params."""
        data = {"nested": {"value": 42}}
        params = Params(data, "root")

        copied = params.copy()

        assert copied["nested"]["value"] == 42
        assert copied is not data

    def test_params_equality(self):
        """Test equality comparison between Params objects."""
        data = {"key": "value"}
        params1 = Params(data, "root")
        params2 = Params(data, "root")
        params3 = Params({"other": "data"}, "root")

        assert params1 == params2
        assert params1 != params3

    def test_params_iteration(self):
        """Test iterating over Params."""
        list_data = [1, 2, 3]
        list_params = Params(list_data, "root")

        items = list(list_params)
        assert len(items) == 3

    def test_params_call_returns_raw_data(self):
        """Test calling Params() returns raw data."""
        data = {"key": "value"}
        params = Params(data, "root")

        # For dict params, calling gives access to dict operations
        assert isinstance(params(), dict)


class TestInvalidParam:
    """Test InvalidParam class functionality."""

    def test_invalid_param_creation(self):
        """Test creating InvalidParam objects."""
        invalid = InvalidParam("missing")

        assert invalid._name == "missing"
        assert invalid.is_valid() is False

    def test_invalid_param_is_valid(self):
        """Test is_valid returns False."""
        invalid = InvalidParam("missing")

        assert invalid.is_valid() is False

    def test_invalid_param_bool_evaluation(self):
        """Test InvalidParam evaluates to False."""
        invalid = InvalidParam("missing")

        assert not invalid
        assert bool(invalid) is False

    def test_invalid_param_continued_access(self):
        """Test that InvalidParam supports continued dot notation."""
        invalid = InvalidParam("missing")
        still_invalid = invalid.nested.deeply.buried

        assert isinstance(still_invalid, InvalidParam)
        assert still_invalid.is_valid() is False

    def test_invalid_param_bracket_access_raises(self):
        """Test that bracket notation on InvalidParam raises TypeError."""
        invalid = InvalidParam("missing")

        with pytest.raises(TypeError):
            _ = invalid["key"]

    def test_invalid_param_repr(self):
        """Test string representation of InvalidParam."""
        invalid = InvalidParam("missing", parent=Params({"data": "value"}, "root"))

        repr_str = repr(invalid)
        assert "InvalidParam" in repr_str
        assert "missing" in repr_str

    def test_invalid_param_get_path(self):
        """Test path tracking for InvalidParam."""
        root = Params({"data": "value"}, "root")
        invalid = InvalidParam("missing", parent=root)

        path = invalid.get_path()
        assert "missing" in path

    def test_params_returns_invalid_for_missing_key(self):
        """Test that Params returns InvalidParam for missing keys."""
        params = Params({"existing": "value"}, "root")

        result = params.missing_key

        assert isinstance(result, InvalidParam)
        assert result.is_valid() is False

    def test_invalid_param_chain_repr(self):
        """Test repr shows the first invalid parameter in chain."""
        params = Params({"level1": {"level2": "value"}}, "root")
        invalid = params.missing.deeply.nested

        repr_str = repr(invalid)
        assert "missing" in repr_str


class TestParamsEdgeCases:
    """Test edge cases and error conditions."""

    def test_params_keys_on_non_dict_returns_empty(self):
        """Test that keys() on non-dict params returns empty list."""
        params = Params([1, 2, 3], "root")

        keys = params.keys()

        assert keys == []

    def test_params_values_on_non_dict_returns_empty(self):
        """Test that values() on non-dict params returns empty list."""
        params = Params([1, 2, 3], "root")

        values = params.values()

        assert values == []

    def test_params_items_on_non_dict_returns_empty(self):
        """Test that items() on non-dict params returns empty list."""
        params = Params([1, 2, 3], "root")

        items = params.items()

        assert items == []

    def test_params_get_on_non_dict_returns_default(self):
        """Test that get() on non-dict params returns default."""
        params = Params([1, 2, 3], "root")

        result = params.get("key", "default")

        assert result == "default"

    def test_params_contains_on_scalar_returns_false(self):
        """Test that 'in' on scalar params returns False."""
        params = Params(42, "root")

        result = "anything" in params

        assert result is False

    def test_params_getitem_on_non_indexable_returns_invalid(self):
        """Test bracket notation on non-indexable params returns InvalidParam."""
        params = Params(42, "root")

        result = params["key"]

        assert isinstance(result, InvalidParam)

    def test_params_getattr_on_non_dict_raises(self):
        """Test dot notation on non-dict params raises TypeError."""
        params = Params([1, 2, 3], "root")

        with pytest.raises(TypeError, match="not a dict"):
            _ = params.missing_attr


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_nested_params_with_lists_and_dicts(self):
        """Test complex nested structure with mixed types."""
        data = {
            "servers": [{"name": "web1", "port": 8080}, {"name": "web2", "port": 8081}],
            "database": {
                "primary": {"host": "db1", "port": 5432},
                "replica": {"host": "db2", "port": 5432},
            },
        }
        params = Params(data, "root")

        assert params.servers[0].name == "web1"
        assert params.servers[1].port == 8081
        assert params.database.primary.host == "db1"
        assert params.database.replica.port == 5432

    def test_environment_expansion_in_nested_structure(self):
        """Test environment variable expansion in nested structures."""
        with patch.dict(os.environ, {"PROJECT_ROOT": "/home/user/project"}):
            data = {
                "paths": {
                    "root": "${PROJECT_ROOT}",
                    "data": "${PROJECT_ROOT}/data",
                    "logs": "${PROJECT_ROOT}/logs",
                }
            }
            params = Params(data, "root")

            assert params.paths.root == "/home/user/project"
            assert params.paths.data == "/home/user/project/data"
            assert params.paths.logs == "/home/user/project/logs"

    def test_full_workflow_with_import_and_env_vars(self, tmp_path):
        """Test complete workflow with imports and environment variables."""
        with patch.dict(os.environ, {"DB_HOST": "production.db"}):
            base = tmp_path / "base.yml"
            base.write_text("""
database:
  host: ${DB_HOST}
  port: 5432
""")

            app = tmp_path / "app.yml"
            app.write_text("""
import: base.yml
database:
  name: myapp
  pool_size: 10
""")

            params = load_params(str(app))

            assert params.database.host == "production.db"
            assert params.database.port == 5432
            assert params.database.name == "myapp"
            assert params.database.pool_size == 10
