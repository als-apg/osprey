"""
Unit tests for validate_database tool.

Tests validate_json_structure for ~15 distinct code paths,
validate_database_loading for hierarchical/in_context,
and detect_pipeline_config for priority/fallback logic.
"""

import json
from pathlib import Path

import pytest

from osprey.services.channel_finder.tools.validate_database import (
    detect_pipeline_config,
    validate_database_loading,
    validate_json_structure,
)

EXAMPLES_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "osprey"
    / "templates"
    / "apps"
    / "control_assistant"
    / "data"
    / "channel_databases"
)


class TestValidateJsonStructure:
    """Test JSON structure validation across all code paths."""

    def test_file_not_found(self, tmp_path):
        """Non-existent file returns error."""
        is_valid, errors, warnings = validate_json_structure(tmp_path / "missing.json")
        assert not is_valid
        assert any("not found" in e for e in errors)

    def test_invalid_json(self, tmp_path):
        """Malformed JSON returns error."""
        f = tmp_path / "bad.json"
        f.write_text("{not valid json")
        is_valid, errors, warnings = validate_json_structure(f)
        assert not is_valid
        assert any("Invalid JSON" in e for e in errors)

    def test_legacy_array_format_valid_with_warning(self, tmp_path):
        """Legacy array format is valid but produces a warning."""
        f = tmp_path / "legacy.json"
        f.write_text(json.dumps([{"channel": "CH1", "address": "PV:CH1", "description": "Test"}]))
        is_valid, errors, warnings = validate_json_structure(f)
        assert is_valid
        assert any("legacy" in w.lower() for w in warnings)

    def test_dict_without_channels_key(self, tmp_path):
        """Dict without 'channels' key returns error."""
        f = tmp_path / "no_channels.json"
        f.write_text(json.dumps({"metadata": {}}))
        is_valid, errors, warnings = validate_json_structure(f)
        assert not is_valid
        assert any("channels" in e for e in errors)

    def test_empty_channels_list(self, tmp_path):
        """Empty channels list returns error."""
        f = tmp_path / "empty.json"
        f.write_text(json.dumps({"channels": []}))
        is_valid, errors, warnings = validate_json_structure(f)
        assert not is_valid
        assert any("no channels" in e.lower() for e in errors)

    def test_non_dict_entry_in_channels(self, tmp_path):
        """Non-dict entry in channels produces error."""
        f = tmp_path / "bad_entry.json"
        f.write_text(json.dumps({"channels": ["not a dict"]}))
        is_valid, errors, warnings = validate_json_structure(f)
        assert not is_valid
        assert any("must be a dict" in e for e in errors)

    def test_valid_standalone_channel(self, tmp_path):
        """Valid standalone channel passes."""
        f = tmp_path / "valid.json"
        f.write_text(
            json.dumps(
                {
                    "channels": [
                        {
                            "template": False,
                            "channel": "CH1",
                            "address": "PV:CH1",
                            "description": "Test channel",
                        }
                    ]
                }
            )
        )
        is_valid, errors, warnings = validate_json_structure(f)
        assert is_valid
        assert errors == []

    def test_standalone_missing_required_field(self, tmp_path):
        """Standalone channel missing required field produces error."""
        f = tmp_path / "missing_field.json"
        f.write_text(json.dumps({"channels": [{"template": False, "channel": "CH1"}]}))
        is_valid, errors, warnings = validate_json_structure(f)
        assert not is_valid
        assert any("address" in e for e in errors)
        assert any("description" in e for e in errors)

    def test_standalone_with_empty_field_values(self, tmp_path):
        """Standalone channel with empty field values produces warning."""
        f = tmp_path / "empty_fields.json"
        f.write_text(
            json.dumps(
                {
                    "channels": [
                        {
                            "template": False,
                            "channel": "",
                            "address": "PV:CH1",
                            "description": "Test",
                        }
                    ]
                }
            )
        )
        is_valid, errors, warnings = validate_json_structure(f)
        assert is_valid
        assert any("empty" in w.lower() for w in warnings)

    def test_valid_template_entry(self, tmp_path):
        """Valid template entry passes."""
        f = tmp_path / "template.json"
        f.write_text(
            json.dumps(
                {
                    "channels": [
                        {
                            "template": True,
                            "base_name": "BPM",
                            "instances": [1, 5],
                            "description": "BPM devices",
                            "sub_channels": ["X", "Y"],
                            "address_pattern": "BPM{instance:02d}{suffix}",
                            "channel_descriptions": {"X": "horizontal", "Y": "vertical"},
                        }
                    ]
                }
            )
        )
        is_valid, errors, warnings = validate_json_structure(f)
        assert is_valid
        assert errors == []

    def test_template_missing_required_fields(self, tmp_path):
        """Template missing required fields produces errors."""
        f = tmp_path / "bad_template.json"
        f.write_text(json.dumps({"channels": [{"template": True}]}))
        is_valid, errors, warnings = validate_json_structure(f)
        assert not is_valid
        assert any("base_name" in e for e in errors)
        assert any("instances" in e for e in errors)
        assert any("description" in e for e in errors)

    def test_template_invalid_instances_not_list(self, tmp_path):
        """Template with non-list instances produces error."""
        f = tmp_path / "bad_instances.json"
        f.write_text(
            json.dumps(
                {
                    "channels": [
                        {"template": True, "base_name": "BPM", "instances": 5, "description": "BPM"}
                    ]
                }
            )
        )
        is_valid, errors, warnings = validate_json_structure(f)
        assert not is_valid
        assert any("instances" in e for e in errors)

    def test_template_instances_start_greater_than_end(self, tmp_path):
        """Template with start > end in instances produces error."""
        f = tmp_path / "bad_range.json"
        f.write_text(
            json.dumps(
                {
                    "channels": [
                        {
                            "template": True,
                            "base_name": "BPM",
                            "instances": [10, 1],
                            "description": "BPM",
                        }
                    ]
                }
            )
        )
        is_valid, errors, warnings = validate_json_structure(f)
        assert not is_valid
        assert any("start" in e and "end" in e for e in errors)

    def test_template_missing_address_pattern_warning(self, tmp_path):
        """Template missing address_pattern produces warning."""
        f = tmp_path / "no_pattern.json"
        f.write_text(
            json.dumps(
                {
                    "channels": [
                        {
                            "template": True,
                            "base_name": "BPM",
                            "instances": [1, 3],
                            "description": "BPM",
                            "sub_channels": ["X"],
                        }
                    ]
                }
            )
        )
        is_valid, errors, warnings = validate_json_structure(f)
        assert is_valid
        assert any("address_pattern" in w for w in warnings)

    def test_template_missing_channel_descriptions_warning(self, tmp_path):
        """Template missing channel_descriptions produces warning."""
        f = tmp_path / "no_descs.json"
        f.write_text(
            json.dumps(
                {
                    "channels": [
                        {
                            "template": True,
                            "base_name": "BPM",
                            "instances": [1, 3],
                            "description": "BPM",
                            "sub_channels": ["X"],
                            "address_pattern": "BPM{instance:02d}{suffix}",
                        }
                    ]
                }
            )
        )
        is_valid, errors, warnings = validate_json_structure(f)
        assert is_valid
        assert any("channel_descriptions" in w for w in warnings)

    def test_unknown_presentation_mode_warning(self, tmp_path):
        """Unknown presentation_mode produces warning."""
        f = tmp_path / "bad_mode.json"
        f.write_text(
            json.dumps(
                {
                    "presentation_mode": "unknown_mode",
                    "channels": [
                        {
                            "template": False,
                            "channel": "CH1",
                            "address": "PV:CH1",
                            "description": "Test",
                        }
                    ],
                }
            )
        )
        is_valid, errors, warnings = validate_json_structure(f)
        assert is_valid
        assert any("presentation_mode" in w for w in warnings)

    def test_invalid_top_level_type(self, tmp_path):
        """Non-list/non-dict top-level type returns error."""
        f = tmp_path / "string.json"
        f.write_text('"just a string"')
        is_valid, errors, warnings = validate_json_structure(f)
        assert not is_valid
        assert any("Invalid top-level type" in e for e in errors)


class TestValidateDatabaseLoading:
    """Test database loading through actual database classes."""

    def test_loads_hierarchical_database(self):
        """Loads hierarchical database and returns stats."""
        db_path = EXAMPLES_DIR / "examples" / "consecutive_instances.json"
        if not db_path.exists():
            pytest.skip("Hierarchical example database not found")

        success, errors, stats = validate_database_loading(db_path, "hierarchical")
        assert success
        assert stats.get("total_channels", 0) > 0

    def test_loads_in_context_database(self):
        """Loads in_context database and returns stats."""
        db_path = EXAMPLES_DIR / "in_context.json"
        if not db_path.exists():
            pytest.skip("In-context database not found")

        success, errors, stats = validate_database_loading(db_path, "in_context")
        assert success
        assert stats.get("total_channels", 0) > 0

    def test_handles_load_failure(self, tmp_path):
        """Handles load failure gracefully."""
        f = tmp_path / "corrupt.json"
        f.write_text("not json at all")
        success, errors, stats = validate_database_loading(f, "in_context")
        assert not success
        assert len(errors) > 0
        assert stats == {}


class TestDetectPipelineConfig:
    """Test pipeline detection with priority and fallback logic."""

    def test_explicit_hierarchical_with_path(self):
        """Explicit pipeline_mode: hierarchical with path returns hierarchical."""
        config = {
            "channel_finder": {
                "pipeline_mode": "hierarchical",
                "pipelines": {
                    "hierarchical": {"database": {"path": "/some/path.json"}},
                    "in_context": {"database": {"path": "/other/path.json"}},
                },
            }
        }
        ptype, db_config = detect_pipeline_config(config)
        assert ptype == "hierarchical"
        assert db_config["path"] == "/some/path.json"

    def test_explicit_in_context_with_path(self):
        """Explicit pipeline_mode: in_context with path returns in_context."""
        config = {
            "channel_finder": {
                "pipeline_mode": "in_context",
                "pipelines": {
                    "hierarchical": {"database": {"path": "/some/path.json"}},
                    "in_context": {"database": {"path": "/ctx/path.json"}},
                },
            }
        }
        ptype, db_config = detect_pipeline_config(config)
        assert ptype == "in_context"
        assert db_config["path"] == "/ctx/path.json"

    def test_no_mode_hierarchical_path_present(self):
        """No pipeline_mode, hierarchical path present returns hierarchical."""
        config = {
            "channel_finder": {
                "pipelines": {
                    "hierarchical": {"database": {"path": "/h/path.json"}},
                }
            }
        }
        ptype, db_config = detect_pipeline_config(config)
        assert ptype == "hierarchical"

    def test_no_mode_in_context_path_present(self):
        """No pipeline_mode, only in_context path present returns in_context."""
        config = {
            "channel_finder": {
                "pipelines": {
                    "in_context": {"database": {"path": "/ic/path.json"}},
                }
            }
        }
        ptype, db_config = detect_pipeline_config(config)
        assert ptype == "in_context"

    def test_no_database_configured(self):
        """No database configured returns (None, None)."""
        config = {"channel_finder": {"pipelines": {}}}
        ptype, db_config = detect_pipeline_config(config)
        assert ptype is None
        assert db_config is None
