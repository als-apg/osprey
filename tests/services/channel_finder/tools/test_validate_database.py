"""Tests for the database validation tool.

Covers JSON structure validation, database-loading validation, the rich
result renderer, and the top-level ``run_validation`` orchestration. Config
loading is patched so no real config.yml is required.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from rich.console import Console

from osprey.cli.styles import osprey_theme
from osprey.services.channel_finder.tools import validate_database as mod
from osprey.services.channel_finder.tools.validate_database import (
    print_validation_results,
    run_validation,
    validate_database_loading,
    validate_json_structure,
)


def _write(path: Path, data) -> Path:
    path.write_text(json.dumps(data, indent=2))
    return path


def _capture_console() -> Console:
    """A rich Console that writes to an in-memory buffer for assertions."""
    return Console(file=io.StringIO(), width=200, force_terminal=False, theme=osprey_theme)


def _text(console: Console) -> str:
    return console.file.getvalue()


# ---------------------------------------------------------------------------
# validate_json_structure
# ---------------------------------------------------------------------------


class TestValidateJsonStructure:
    def test_missing_file(self, tmp_path: Path):
        ok, errors, warnings = validate_json_structure(tmp_path / "nope.json")
        assert ok is False
        assert any("not found" in e for e in errors)

    def test_invalid_json(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("{ this is not json")
        ok, errors, _ = validate_json_structure(p)
        assert ok is False
        assert any("Invalid JSON format" in e for e in errors)

    def test_legacy_list_format_warns_but_valid(self, tmp_path: Path):
        p = _write(
            tmp_path / "list.json",
            [{"channel": "A:B", "address": "A:B", "description": "desc"}],
        )
        ok, errors, warnings = validate_json_structure(p)
        assert ok is True
        assert errors == []
        assert any("legacy array format" in w for w in warnings)

    def test_dict_missing_channels_key(self, tmp_path: Path):
        p = _write(tmp_path / "d.json", {"metadata": {}})
        ok, errors, _ = validate_json_structure(p)
        assert ok is False
        assert any("Missing 'channels' key" in e for e in errors)

    def test_unknown_presentation_mode_warns(self, tmp_path: Path):
        p = _write(
            tmp_path / "d.json",
            {
                "presentation_mode": "weird",
                "channels": [{"channel": "A", "address": "A", "description": "d"}],
            },
        )
        ok, errors, warnings = validate_json_structure(p)
        assert ok is True
        assert any("Unknown presentation_mode" in w for w in warnings)

    def test_invalid_top_level_type(self, tmp_path: Path):
        p = _write(tmp_path / "d.json", 42)
        ok, errors, _ = validate_json_structure(p)
        assert ok is False
        assert any("Invalid top-level type" in e for e in errors)

    def test_channels_not_a_list(self, tmp_path: Path):
        p = _write(tmp_path / "d.json", {"channels": {"not": "a list"}})
        ok, errors, _ = validate_json_structure(p)
        assert ok is False
        assert any("'channels' must be a list" in e for e in errors)

    def test_empty_channels(self, tmp_path: Path):
        p = _write(tmp_path / "d.json", {"channels": []})
        ok, errors, _ = validate_json_structure(p)
        assert ok is False
        assert any("no channels" in e for e in errors)

    def test_channel_entry_not_dict(self, tmp_path: Path):
        p = _write(tmp_path / "d.json", {"channels": ["not-a-dict"]})
        ok, errors, _ = validate_json_structure(p)
        assert ok is False
        assert any("must be a dict" in e for e in errors)

    def test_standalone_missing_required_fields(self, tmp_path: Path):
        p = _write(tmp_path / "d.json", {"channels": [{"channel": "A"}]})
        ok, errors, _ = validate_json_structure(p)
        assert ok is False
        assert any("missing required field 'address'" in e for e in errors)
        assert any("missing required field 'description'" in e for e in errors)

    def test_standalone_empty_field_warns(self, tmp_path: Path):
        p = _write(
            tmp_path / "d.json",
            {"channels": [{"channel": "A", "address": "A", "description": ""}]},
        )
        ok, errors, warnings = validate_json_structure(p)
        assert ok is True
        assert any("field 'description' is empty" in w for w in warnings)

    def test_valid_standalone_no_issues(self, tmp_path: Path):
        p = _write(
            tmp_path / "d.json",
            {"channels": [{"channel": "A", "address": "A:ADDR", "description": "ok"}]},
        )
        ok, errors, warnings = validate_json_structure(p)
        assert ok is True
        assert errors == []
        assert warnings == []


class TestValidateJsonStructureTemplates:
    def test_template_missing_required_fields(self, tmp_path: Path):
        p = _write(tmp_path / "t.json", {"channels": [{"template": True}]})
        ok, errors, _ = validate_json_structure(p)
        assert ok is False
        assert any("missing required field 'base_name'" in e for e in errors)
        assert any("missing required field 'instances'" in e for e in errors)

    def test_template_instances_wrong_shape(self, tmp_path: Path):
        p = _write(
            tmp_path / "t.json",
            {
                "channels": [
                    {
                        "template": True,
                        "base_name": "B",
                        "instances": [1, 2, 3],
                        "description": "d",
                    }
                ]
            },
        )
        ok, errors, _ = validate_json_structure(p)
        assert ok is False
        assert any("'instances' must be [start, end]" in e for e in errors)

    def test_template_instances_start_after_end(self, tmp_path: Path):
        p = _write(
            tmp_path / "t.json",
            {
                "channels": [
                    {
                        "template": True,
                        "base_name": "B",
                        "instances": [5, 2],
                        "description": "d",
                    }
                ]
            },
        )
        ok, errors, _ = validate_json_structure(p)
        assert ok is False
        assert any("start (5) > end (2)" in e for e in errors)

    def test_template_sub_channels_not_list(self, tmp_path: Path):
        p = _write(
            tmp_path / "t.json",
            {
                "channels": [
                    {
                        "template": True,
                        "base_name": "B",
                        "instances": [1, 2],
                        "description": "d",
                        "sub_channels": "SP",
                    }
                ]
            },
        )
        ok, errors, _ = validate_json_structure(p)
        assert ok is False
        assert any("'sub_channels' must be a list" in e for e in errors)

    def test_template_sub_channels_empty_warns(self, tmp_path: Path):
        p = _write(
            tmp_path / "t.json",
            {
                "channels": [
                    {
                        "template": True,
                        "base_name": "B",
                        "instances": [1, 2],
                        "description": "d",
                        "sub_channels": [],
                        "address_pattern": "X",
                        "channel_descriptions": {},
                    }
                ]
            },
        )
        ok, errors, warnings = validate_json_structure(p)
        assert ok is True
        assert any("'sub_channels' is empty" in w for w in warnings)

    def test_template_axes_not_list(self, tmp_path: Path):
        p = _write(
            tmp_path / "t.json",
            {
                "channels": [
                    {
                        "template": True,
                        "base_name": "B",
                        "instances": [1, 2],
                        "description": "d",
                        "axes": "X",
                    }
                ]
            },
        )
        ok, errors, _ = validate_json_structure(p)
        assert ok is False
        assert any("'axes' must be a list" in e for e in errors)

    def test_template_missing_optional_fields_warn(self, tmp_path: Path):
        p = _write(
            tmp_path / "t.json",
            {
                "channels": [
                    {
                        "template": True,
                        "base_name": "B",
                        "instances": [1, 2],
                        "description": "d",
                    }
                ]
            },
        )
        ok, errors, warnings = validate_json_structure(p)
        assert ok is True
        assert any("missing 'address_pattern'" in w for w in warnings)
        assert any("missing 'channel_descriptions'" in w for w in warnings)


# ---------------------------------------------------------------------------
# validate_database_loading
# ---------------------------------------------------------------------------


class TestValidateDatabaseLoading:
    def test_in_context_loads_and_reports_stats(self, tmp_path: Path):
        p = _write(
            tmp_path / "db.json",
            {"channels": [{"channel": "A:B", "address": "A:B", "description": "d"}]},
        )
        ok, errors, stats = validate_database_loading(p, "in_context")
        assert ok is True
        assert errors == []
        assert stats.get("total_channels") == 1

    def test_hierarchical_loads(self, tmp_path: Path):
        p = _write(
            tmp_path / "hier.json",
            {
                "hierarchy": {
                    "levels": [
                        {"name": "system", "type": "tree"},
                        {"name": "signal", "type": "tree"},
                    ],
                    "naming_pattern": "{system}:{signal}",
                },
                "tree": {"SR": {"X": {}, "Y": {}}},
            },
        )
        ok, errors, stats = validate_database_loading(p, "hierarchical")
        assert ok is True
        assert errors == []
        assert stats.get("total_channels", 0) >= 1

    def test_load_failure_returns_traceback(self, tmp_path: Path):
        ok, errors, stats = validate_database_loading(tmp_path / "missing.json", "in_context")
        assert ok is False
        assert any("Failed to load database" in e for e in errors)
        assert stats == {}


# ---------------------------------------------------------------------------
# print_validation_results
# ---------------------------------------------------------------------------


class TestPrintValidationResults:
    def test_valid_no_issues(self):
        console = _capture_console()
        print_validation_results(True, [], [], console=console)
        out = _text(console)
        assert "VALID" in out
        assert "No issues found" in out

    def test_errors_render(self):
        console = _capture_console()
        print_validation_results(False, ["boom happened"], [], console=console)
        out = _text(console)
        assert "INVALID" in out
        assert "ERRORS" in out
        assert "boom happened" in out

    def test_warnings_render(self):
        console = _capture_console()
        print_validation_results(True, [], ["careful now"], console=console)
        out = _text(console)
        assert "WARNINGS" in out
        assert "careful now" in out

    def test_stats_render_with_pipeline(self):
        console = _capture_console()
        stats = {"format": "flat", "total_channels": 12, "systems": ["SR", "BR"]}
        print_validation_results(
            True, [], [], stats=stats, pipeline_type="in_context", console=console
        )
        out = _text(console)
        assert "DATABASE STATISTICS" in out
        assert "Total Channels" in out
        assert "12" in out
        assert "In Context" in out  # pipeline_type title-cased

    def test_verbose_shows_detailed_stats(self):
        console = _capture_console()
        stats = {
            "format": "flat",
            "total_channels": 3,
            "extra_metric": "surprising",
        }
        print_validation_results(True, [], [], stats=stats, verbose=True, console=console)
        out = _text(console)
        assert "DETAILED STATISTICS" in out
        assert "extra_metric" in out
        assert "surprising" in out


# ---------------------------------------------------------------------------
# run_validation
# ---------------------------------------------------------------------------


class TestRunValidation:
    def test_valid_in_context_db_returns_zero(self, tmp_path: Path):
        p = _write(
            tmp_path / "db.json",
            {"channels": [{"channel": "A:B", "address": "A:B", "description": "d"}]},
        )
        console = _capture_console()
        rc = run_validation(database=str(p), pipeline="in_context", console=console)
        assert rc == 0
        assert "VALID" in _text(console)

    def test_invalid_structure_short_circuits_to_one(self, tmp_path: Path):
        p = _write(tmp_path / "db.json", {"channels": []})
        console = _capture_console()
        rc = run_validation(database=str(p), pipeline="in_context", console=console)
        assert rc == 1
        assert "INVALID" in _text(console)

    def test_hierarchical_pipeline_skips_json_structure(self, tmp_path: Path):
        p = _write(
            tmp_path / "hier.json",
            {
                "hierarchy": {
                    "levels": [
                        {"name": "system", "type": "tree"},
                        {"name": "signal", "type": "tree"},
                    ],
                    "naming_pattern": "{system}:{signal}",
                },
                "tree": {"SR": {"X": {}}},
            },
        )
        console = _capture_console()
        rc = run_validation(database=str(p), pipeline="hierarchical", console=console)
        assert rc == 0

    def test_no_database_and_no_config_returns_one(self, monkeypatch):
        import osprey.utils.config as config_mod

        # Config with no pipelines configured -> detect returns (None, None).
        monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: {})
        console = _capture_console()
        rc = run_validation(console=console)
        assert rc == 1
        assert "No database configured" in _text(console)

    def test_no_database_config_raises_returns_one(self, monkeypatch):
        import osprey.utils.config as config_mod

        def boom(*a, **k):
            raise RuntimeError("config broken")

        monkeypatch.setattr(config_mod, "load_config", boom)
        console = _capture_console()
        rc = run_validation(console=console)
        assert rc == 1
        assert "Error reading config" in _text(console)

    def test_no_database_detected_but_missing_path(self, monkeypatch):
        import osprey.utils.config as config_mod

        monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: {})
        # detect returns a type but with no path.
        monkeypatch.setattr(mod, "detect_pipeline_config", lambda config: ("in_context", {}))
        console = _capture_console()
        rc = run_validation(console=console)
        assert rc == 1
        assert "No database path" in _text(console)

    def test_database_given_without_pipeline_uses_detected_type(self, tmp_path, monkeypatch):
        import osprey.utils.config as config_mod

        p = _write(
            tmp_path / "db.json",
            {"channels": [{"channel": "A:B", "address": "A:B", "description": "d"}]},
        )
        monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: {})
        monkeypatch.setattr(
            mod, "detect_pipeline_config", lambda config: ("in_context", {"path": "x"})
        )
        console = _capture_console()
        rc = run_validation(database=str(p), console=console)
        assert rc == 0
