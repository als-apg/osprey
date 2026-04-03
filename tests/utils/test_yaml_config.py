"""Tests for comment-preserving YAML config utilities.

Covers every mutation type exposed by the web settings panel and CLI:
- Boolean toggles
- Numeric fields
- String fields
- Enum/select fields
- Array (list) fields
- Nested object updates
- List add/remove operations
- Comment and formatting preservation across all of the above
"""

from __future__ import annotations

import pytest

from osprey.utils.config_writer import (
    config_add_to_list,
    config_read,
    config_remove_from_list,
    config_update_fields,
)

# ---------------------------------------------------------------------------
# Fixture: A realistic config.yml with comments and formatting
# ---------------------------------------------------------------------------

SAMPLE_CONFIG = """\
# ============================================================
# Test Project Configuration
# ============================================================
# This file has comments that must survive round-trips.

# Project identity
project_name: "my-test-project"

# ============================================================
# SAFETY CONTROLS
# ============================================================

# Approval workflow for sensitive operations
approval:
  global_mode: "selective"   # Options: disabled | selective | all_capabilities

# ============================================================
# CONTROL SYSTEM
# ============================================================

control_system:
  type: "mock"  # Options: mock | epics
  writes_enabled: false  # Master safety switch
  limits_checking:
    enabled: false
    database_path: null
    allow_unlisted_channels: true
    on_violation: "skip"
  write_verification:
    enabled: false
    default_level: "callback"
    default_tolerance_percent: 0.1
    timeout: 5.0
    fail_on_mismatch: false

# ============================================================
# ARTIFACT SERVER
# ============================================================

artifact_server:
  host: "127.0.0.1"
  port: 8086
  auto_launch: true

# Array example
scheduling:
  limits:
    max_concurrent: 5
    timeout: 300
    max_retries: 3
"""


@pytest.fixture
def config_file(tmp_path):
    """Write sample config and return its path."""
    p = tmp_path / "config.yml"
    p.write_text(SAMPLE_CONFIG, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Comment Preservation — the core invariant
# ---------------------------------------------------------------------------


class TestCommentPreservation:
    """Every mutation must keep YAML comments intact."""

    def _comment_lines(self, text: str) -> list[str]:
        return [line for line in text.splitlines() if line.strip().startswith("#")]

    def test_update_preserves_all_comments(self, config_file):
        original_comments = self._comment_lines(config_file.read_text())
        config_update_fields(config_file, {"control_system.writes_enabled": True})
        after_comments = self._comment_lines(config_file.read_text())
        assert original_comments == after_comments

    def test_add_to_list_preserves_comments(self, config_file):
        original_comments = self._comment_lines(config_file.read_text())
        config_add_to_list(config_file, ["prompts", "user_owned"], "rules/facility")
        after_comments = self._comment_lines(config_file.read_text())
        # New section may not have comments, but existing ones must survive
        for c in original_comments:
            assert c in after_comments

    def test_remove_from_list_preserves_comments(self, config_file):
        config_add_to_list(config_file, ["prompts", "user_owned"], "rules/facility")
        original_comments = self._comment_lines(config_file.read_text())
        config_remove_from_list(config_file, ["prompts", "user_owned"], "rules/facility")
        after_comments = self._comment_lines(config_file.read_text())
        for c in original_comments:
            assert c in after_comments

    def test_section_headers_preserved(self, config_file):
        """The ===== section banners must survive."""
        config_update_fields(
            config_file,
            {
                "control_system.writes_enabled": True,
                "approval.global_mode": "all_capabilities",
                "artifact_server.port": 9999,
            },
        )
        text = config_file.read_text()
        assert "# ============================================================" in text
        assert "# SAFETY CONTROLS" in text
        assert "# CONTROL SYSTEM" in text
        assert "# ARTIFACT SERVER" in text

    def test_inline_comments_preserved(self, config_file):
        """Inline comments like '# Options: mock | epics' must survive."""
        config_update_fields(config_file, {"control_system.type": "epics"})
        text = config_file.read_text()
        assert "# Options: mock | epics" in text


# ---------------------------------------------------------------------------
# Boolean Updates (toggle switches in the web UI)
# ---------------------------------------------------------------------------


class TestBooleanUpdates:
    def test_toggle_false_to_true(self, config_file):
        config_update_fields(config_file, {"control_system.writes_enabled": True})
        data = config_read(config_file)
        assert data["control_system"]["writes_enabled"] is True

    def test_toggle_true_to_false(self, config_file):
        config_update_fields(config_file, {"artifact_server.auto_launch": False})
        data = config_read(config_file)
        assert data["artifact_server"]["auto_launch"] is False

    def test_nested_boolean(self, config_file):
        config_update_fields(
            config_file,
            {
                "control_system.limits_checking.enabled": True,
                "control_system.write_verification.fail_on_mismatch": True,
            },
        )
        data = config_read(config_file)
        assert data["control_system"]["limits_checking"]["enabled"] is True
        assert data["control_system"]["write_verification"]["fail_on_mismatch"] is True


# ---------------------------------------------------------------------------
# Numeric Updates (number inputs in the web UI)
# ---------------------------------------------------------------------------


class TestNumericUpdates:
    def test_integer_update(self, config_file):
        config_update_fields(config_file, {"artifact_server.port": 9999})
        data = config_read(config_file)
        assert data["artifact_server"]["port"] == 9999

    def test_float_update(self, config_file):
        config_update_fields(
            config_file,
            {
                "control_system.write_verification.timeout": 10.0,
            },
        )
        data = config_read(config_file)
        assert data["control_system"]["write_verification"]["timeout"] == 10.0

    def test_float_tolerance(self, config_file):
        config_update_fields(
            config_file,
            {
                "control_system.write_verification.default_tolerance_percent": 0.5,
            },
        )
        data = config_read(config_file)
        assert data["control_system"]["write_verification"]["default_tolerance_percent"] == 0.5

    def test_deeply_nested_integer(self, config_file):
        config_update_fields(
            config_file,
            {
                "scheduling.limits.timeout": 600,
            },
        )
        data = config_read(config_file)
        assert data["scheduling"]["limits"]["timeout"] == 600


# ---------------------------------------------------------------------------
# String / Enum Updates (text inputs and select dropdowns)
# ---------------------------------------------------------------------------


class TestStringUpdates:
    def test_enum_string(self, config_file):
        config_update_fields(config_file, {"approval.global_mode": "all_capabilities"})
        data = config_read(config_file)
        assert data["approval"]["global_mode"] == "all_capabilities"

    def test_string_with_special_chars(self, config_file):
        config_update_fields(config_file, {"artifact_server.host": "0.0.0.0"})
        data = config_read(config_file)
        assert data["artifact_server"]["host"] == "0.0.0.0"

    def test_null_to_string(self, config_file):
        config_update_fields(
            config_file,
            {
                "control_system.limits_checking.database_path": "data/limits.json",
            },
        )
        data = config_read(config_file)
        assert data["control_system"]["limits_checking"]["database_path"] == "data/limits.json"

    def test_string_to_null(self, config_file):
        config_update_fields(
            config_file,
            {
                "control_system.limits_checking.database_path": None,
            },
        )
        data = config_read(config_file)
        assert data["control_system"]["limits_checking"]["database_path"] is None


# ---------------------------------------------------------------------------
# Array Updates (comma-separated text inputs in the web UI)
# ---------------------------------------------------------------------------


class TestArrayUpdates:
    def test_create_new_list(self, config_file):
        config_update_fields(
            config_file,
            {
                "control_system.limits_checking.exclude_channels": ["CH:01", "CH:02"],
            },
        )
        data = config_read(config_file)
        assert data["control_system"]["limits_checking"]["exclude_channels"] == ["CH:01", "CH:02"]

    def test_replace_list(self, config_file):
        config_update_fields(
            config_file,
            {
                "control_system.limits_checking.exclude_channels": ["A", "B", "C"],
            },
        )
        config_update_fields(
            config_file,
            {
                "control_system.limits_checking.exclude_channels": ["X"],
            },
        )
        data = config_read(config_file)
        assert data["control_system"]["limits_checking"]["exclude_channels"] == ["X"]


# ---------------------------------------------------------------------------
# Multiple Simultaneous Updates (form submit sends all changed fields)
# ---------------------------------------------------------------------------


class TestBatchUpdates:
    def test_multiple_types_at_once(self, config_file):
        config_update_fields(
            config_file,
            {
                "control_system.writes_enabled": True,
                "control_system.type": "epics",
                "artifact_server.port": 7777,
                "approval.global_mode": "disabled",
                "control_system.write_verification.timeout": 15.0,
            },
        )
        data = config_read(config_file)
        assert data["control_system"]["writes_enabled"] is True
        assert data["control_system"]["type"] == "epics"
        assert data["artifact_server"]["port"] == 7777
        assert data["approval"]["global_mode"] == "disabled"
        assert data["control_system"]["write_verification"]["timeout"] == 15.0

    def test_batch_preserves_comments(self, config_file):
        text_before = config_file.read_text()
        comment_count_before = text_before.count("#")

        config_update_fields(
            config_file,
            {
                "control_system.writes_enabled": True,
                "artifact_server.port": 9999,
                "approval.global_mode": "disabled",
            },
        )

        text_after = config_file.read_text()
        comment_count_after = text_after.count("#")
        assert comment_count_after >= comment_count_before


# ---------------------------------------------------------------------------
# New Section Creation (when a key path doesn't exist yet)
# ---------------------------------------------------------------------------


class TestNewSections:
    def test_create_new_top_level_key(self, config_file):
        config_update_fields(config_file, {"new_feature.enabled": True})
        data = config_read(config_file)
        assert data["new_feature"]["enabled"] is True

    def test_create_deeply_nested_key(self, config_file):
        config_update_fields(config_file, {"a.b.c.d": "deep"})
        data = config_read(config_file)
        assert data["a"]["b"]["c"]["d"] == "deep"


# ---------------------------------------------------------------------------
# List Operations (prompts.user_owned management)
# ---------------------------------------------------------------------------


class TestListOperations:
    def test_add_to_nonexistent_list(self, config_file):
        added = config_add_to_list(config_file, ["prompts", "user_owned"], "rules/facility")
        assert added is True
        data = config_read(config_file)
        assert data["prompts"]["user_owned"] == ["rules/facility"]

    def test_add_duplicate_is_noop(self, config_file):
        config_add_to_list(config_file, ["prompts", "user_owned"], "rules/facility")
        added = config_add_to_list(config_file, ["prompts", "user_owned"], "rules/facility")
        assert added is False
        data = config_read(config_file)
        assert data["prompts"]["user_owned"] == ["rules/facility"]

    def test_add_multiple_items(self, config_file):
        config_add_to_list(config_file, ["prompts", "user_owned"], "rules/facility")
        config_add_to_list(config_file, ["prompts", "user_owned"], "rules/safety")
        data = config_read(config_file)
        assert data["prompts"]["user_owned"] == ["rules/facility", "rules/safety"]

    def test_remove_existing_item(self, config_file):
        config_add_to_list(config_file, ["prompts", "user_owned"], "rules/facility")
        config_add_to_list(config_file, ["prompts", "user_owned"], "rules/safety")
        removed = config_remove_from_list(config_file, ["prompts", "user_owned"], "rules/facility")
        assert removed is True
        data = config_read(config_file)
        assert data["prompts"]["user_owned"] == ["rules/safety"]

    def test_remove_nonexistent_item(self, config_file):
        config_add_to_list(config_file, ["prompts", "user_owned"], "rules/facility")
        removed = config_remove_from_list(
            config_file, ["prompts", "user_owned"], "rules/nonexistent"
        )
        assert removed is False

    def test_remove_last_item_prunes_parents(self, config_file):
        config_add_to_list(config_file, ["prompts", "user_owned"], "rules/facility")
        config_remove_from_list(config_file, ["prompts", "user_owned"], "rules/facility")
        data = config_read(config_file)
        assert "prompts" not in data

    def test_remove_last_item_no_prune(self, config_file):
        config_add_to_list(config_file, ["prompts", "user_owned"], "rules/facility")
        config_remove_from_list(
            config_file,
            ["prompts", "user_owned"],
            "rules/facility",
            prune_empty=False,
        )
        data = config_read(config_file)
        assert data["prompts"]["user_owned"] == []

    def test_remove_from_nonexistent_path(self, config_file):
        removed = config_remove_from_list(config_file, ["nonexistent", "path"], "value")
        assert removed is False


# ---------------------------------------------------------------------------
# Key Ordering Preservation
# ---------------------------------------------------------------------------


class TestOrderPreservation:
    def test_key_order_unchanged(self, config_file):
        """Updating a value must not reorder top-level keys."""
        import yaml

        original = yaml.safe_load(config_file.read_text())
        original_keys = list(original.keys())

        config_update_fields(config_file, {"control_system.writes_enabled": True})

        updated = yaml.safe_load(config_file.read_text())
        updated_keys = list(updated.keys())
        assert original_keys == updated_keys


# ---------------------------------------------------------------------------
# config_read (JSON-serializable output)
# ---------------------------------------------------------------------------


class TestConfigRead:
    def test_returns_plain_dict(self, config_file):
        data = config_read(config_file)
        assert isinstance(data, dict)
        assert data["project_name"] == "my-test-project"

    def test_json_serializable(self, config_file):
        import json

        data = config_read(config_file)
        serialized = json.dumps(data)
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.yml"
        p.write_text("", encoding="utf-8")
        config_update_fields(p, {"key": "value"})
        data = config_read(p)
        assert data["key"] == "value"

    def test_minimal_file(self, tmp_path):
        p = tmp_path / "minimal.yml"
        p.write_text("key: value\n", encoding="utf-8")
        config_update_fields(p, {"key": "new_value"})
        data = config_read(p)
        assert data["key"] == "new_value"

    def test_idempotent_update(self, config_file):
        """Updating to the same value should not change the file."""
        config_update_fields(config_file, {"control_system.type": "mock"})
        text1 = config_file.read_text()
        config_update_fields(config_file, {"control_system.type": "mock"})
        text2 = config_file.read_text()
        assert text1 == text2
