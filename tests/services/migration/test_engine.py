"""Unit tests for :mod:`osprey.services.migration.engine`.

The migration engine is pure business logic that decides, file by file, what
happens to a facility's customizations when OSPREY is upgraded: which files are
safe to auto-copy from the new template, which facility edits must be preserved,
and which require a human/AI merge. Getting the three-way classification wrong
would silently clobber operator customizations, so the classification contract
(:func:`classify_file`) and the analysis that feeds it are asserted branch by
branch. The remaining functions are exercised through real ``tmp_path``
projects to pin their file-system side effects and their error handling (which
deliberately swallows unreadable files rather than raising).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from osprey.services.migration.engine import (
    MANIFEST_FILENAME,
    FileCategory,
    calculate_file_hash,
    classify_file,
    detect_project_settings,
    generate_merge_prompt,
    generate_migration_directory,
    load_manifest,
    migrate_claude_code_config,
    perform_migration_analysis,
    read_file_content,
)


def _write(path: Path, content: str) -> Path:
    """Create parents and write ``content`` to ``path``; return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestLoadManifest:
    def test_returns_none_when_missing(self, tmp_path):
        assert load_manifest(tmp_path) is None

    def test_loads_valid_manifest(self, tmp_path):
        payload = {"osprey_version": "2.0.0", "template": "control_assistant"}
        _write(tmp_path / MANIFEST_FILENAME, json.dumps(payload))
        assert load_manifest(tmp_path) == payload

    def test_returns_none_on_invalid_json(self, tmp_path):
        _write(tmp_path / MANIFEST_FILENAME, "{ not: valid json ]")
        assert load_manifest(tmp_path) is None

    def test_returns_none_when_manifest_is_directory(self, tmp_path):
        # A directory at the manifest path exists() but can't be opened as a
        # file -> OSError -> None (caller handles messaging).
        (tmp_path / MANIFEST_FILENAME).mkdir()
        assert load_manifest(tmp_path) is None


class TestCalculateFileHash:
    def test_matches_hashlib(self, tmp_path):
        f = _write(tmp_path / "a.txt", "hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert calculate_file_hash(f) == expected

    def test_identical_content_same_hash(self, tmp_path):
        a = _write(tmp_path / "a.txt", "same")
        b = _write(tmp_path / "b.txt", "same")
        assert calculate_file_hash(a) == calculate_file_hash(b)

    def test_different_content_different_hash(self, tmp_path):
        a = _write(tmp_path / "a.txt", "one")
        b = _write(tmp_path / "b.txt", "two")
        assert calculate_file_hash(a) != calculate_file_hash(b)

    def test_large_file_streamed_in_chunks(self, tmp_path):
        # Exceeds the 8192-byte chunk size to exercise the streaming loop.
        content = "x" * 20000
        f = _write(tmp_path / "big.txt", content)
        assert calculate_file_hash(f) == hashlib.sha256(content.encode()).hexdigest()

    def test_returns_none_for_missing_file(self, tmp_path):
        assert calculate_file_hash(tmp_path / "nope.txt") is None

    def test_returns_none_for_directory(self, tmp_path):
        assert calculate_file_hash(tmp_path) is None


class TestReadFileContent:
    def test_reads_text(self, tmp_path):
        f = _write(tmp_path / "a.txt", "content here")
        assert read_file_content(f) == "content here"

    def test_returns_none_for_missing_file(self, tmp_path):
        assert read_file_content(tmp_path / "missing.txt") is None

    def test_returns_none_for_non_utf8(self, tmp_path):
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\xff\xfe\x00\x01")
        assert read_file_content(f) is None


class TestClassifyFile:
    @pytest.mark.parametrize("prefix", ["data/", "_agent_data/"])
    def test_data_directories_always_preserved(self, prefix):
        # DATA wins even when the file only exists in the new template.
        result = classify_file(f"{prefix}logs.db", None, None, "newhash")
        assert result is FileCategory.DATA

    def test_new_file_only_in_new_template(self):
        assert classify_file("x.py", None, None, "newhash") is FileCategory.NEW

    def test_removed_file_only_in_old_template(self):
        assert classify_file("x.py", None, "oldhash", None) is FileCategory.REMOVED

    def test_removed_takes_precedence_when_facility_still_has_it(self):
        # old present, new absent -> REMOVED regardless of facility presence.
        assert classify_file("x.py", "fac", "oldhash", None) is FileCategory.REMOVED

    def test_facility_only_preserved(self):
        assert classify_file("x.py", "fac", None, None) is FileCategory.PRESERVE

    def test_three_way_template_changed_facility_unchanged_auto_copy(self):
        result = classify_file("x.py", "same", "same", "different")
        assert result is FileCategory.AUTO_COPY

    def test_three_way_facility_changed_template_unchanged_preserve(self):
        result = classify_file("x.py", "faciledit", "same", "same")
        assert result is FileCategory.PRESERVE

    def test_three_way_both_changed_needs_merge(self):
        result = classify_file("x.py", "faciledit", "orig", "tmpledit")
        assert result is FileCategory.MERGE

    def test_three_way_nothing_changed_preserve(self):
        result = classify_file("x.py", "same", "same", "same")
        assert result is FileCategory.PRESERVE

    def test_facility_and_new_but_no_old_defaults_to_preserve(self):
        # Not new-only (facility present), not removed (new present), not
        # facility-only (new present), not three-way (old missing) -> the
        # safety default of PRESERVE.
        result = classify_file("x.py", "fac", None, "new")
        assert result is FileCategory.PRESERVE


class TestDetectProjectSettings:
    def test_empty_project_returns_base_shape(self, tmp_path):
        settings = detect_project_settings(tmp_path)
        assert settings["detected"] is True
        assert settings["confidence"] == {}
        assert settings["warnings"] == []

    def test_detects_llm_provider_and_model(self, tmp_path):
        _write(
            tmp_path / "config.yml",
            "llm:\n  default_provider: anthropic\n  default_model: claude-x\n",
        )
        settings = detect_project_settings(tmp_path)
        assert settings["provider"] == "anthropic"
        assert settings["model"] == "claude-x"
        assert settings["confidence"]["provider"] == "high"
        assert settings["confidence"]["model"] == "high"

    def test_channel_finder_implies_control_assistant_template(self, tmp_path):
        _write(
            tmp_path / "config.yml",
            "channel_finder:\n  default_pipeline: in_context\n",
        )
        settings = detect_project_settings(tmp_path)
        assert settings["channel_finder_mode"] == "in_context"
        assert settings["confidence"]["channel_finder_mode"] == "medium"
        assert settings["template"] == "control_assistant"
        assert settings["confidence"]["template"] == "high"

    def test_capabilities_without_channel_finder_medium_template(self, tmp_path):
        _write(tmp_path / "config.yml", "capabilities:\n  foo: bar\n")
        settings = detect_project_settings(tmp_path)
        assert settings["template"] == "control_assistant"
        assert settings["confidence"]["template"] == "medium"

    def test_invalid_config_yaml_records_warning(self, tmp_path):
        # A YAML mapping whose value is an unclosed flow sequence fails to parse.
        _write(tmp_path / "config.yml", "llm: [unterminated\n")
        settings = detect_project_settings(tmp_path)
        assert any("config.yml" in w for w in settings["warnings"])

    def test_empty_config_yaml_is_safe(self, tmp_path):
        # safe_load of an empty file returns None -> guarded by `if config`.
        _write(tmp_path / "config.yml", "")
        settings = detect_project_settings(tmp_path)
        assert settings["warnings"] == []
        assert "provider" not in settings

    def test_pyproject_version_ge_constraint_medium(self, tmp_path):
        _write(
            tmp_path / "pyproject.toml",
            '[project]\ndependencies = ["osprey-framework>=2.0.0,<3.0.0"]\n',
        )
        settings = detect_project_settings(tmp_path)
        assert settings["estimated_osprey_version"] == "2.0.0"
        assert settings["confidence"]["osprey_version"] == "medium"

    def test_pyproject_version_eq_constraint_high(self, tmp_path):
        _write(
            tmp_path / "pyproject.toml",
            '[project]\ndependencies = ["osprey-framework==2.1.3"]\n',
        )
        settings = detect_project_settings(tmp_path)
        assert settings["estimated_osprey_version"] == "2.1.3"
        assert settings["confidence"]["osprey_version"] == "high"

    def test_pyproject_without_osprey_dep_no_version(self, tmp_path):
        _write(
            tmp_path / "pyproject.toml",
            '[project]\ndependencies = ["requests>=2.0"]\n',
        )
        settings = detect_project_settings(tmp_path)
        assert "estimated_osprey_version" not in settings

    def test_invalid_pyproject_records_warning(self, tmp_path):
        _write(tmp_path / "pyproject.toml", "this is = not valid toml [[[\n")
        settings = detect_project_settings(tmp_path)
        assert any("pyproject.toml" in w for w in settings["warnings"])

    def test_registry_extend_style_high_confidence(self, tmp_path):
        pkg = tmp_path / "src" / "myfacility"
        _write(
            pkg / "registry.py",
            "class R(OspreyFrameworkRegistry):\n    # we extend the framework\n    pass\n",
        )
        settings = detect_project_settings(tmp_path)
        assert settings["registry_style"] == "extend"
        assert settings["confidence"]["registry_style"] == "high"
        assert settings["package_name"] == "myfacility"

    def test_registry_standalone_style_medium_confidence(self, tmp_path):
        pkg = tmp_path / "src" / "standalonepkg"
        _write(
            pkg / "registry.py",
            "# explicit registration of every capability\nregistry = object()\n",
        )
        settings = detect_project_settings(tmp_path)
        assert settings["registry_style"] == "standalone"
        assert settings["confidence"]["registry_style"] == "medium"
        assert settings["package_name"] == "standalonepkg"

    def test_registry_many_capability_registrations_standalone(self, tmp_path):
        pkg = tmp_path / "src" / "bigpkg"
        body = "\n".join(f"CapabilityRegistration({i})" for i in range(6))
        _write(pkg / "registry.py", body + "\n")
        settings = detect_project_settings(tmp_path)
        assert settings["registry_style"] == "standalone"
        assert settings["confidence"]["registry_style"] == "medium"

    def test_registry_unknown_style_low_confidence(self, tmp_path):
        pkg = tmp_path / "src" / "mystery"
        _write(pkg / "registry.py", "x = 1\n")
        settings = detect_project_settings(tmp_path)
        assert settings["registry_style"] == "extend"
        assert settings["confidence"]["registry_style"] == "low"
        assert settings["package_name"] == "mystery"

    def test_underscore_package_dirs_skipped(self, tmp_path):
        # Only underscore-prefixed dirs present -> no registry detected.
        _write(tmp_path / "src" / "__pycache__" / "registry.py", "x = 1\n")
        settings = detect_project_settings(tmp_path)
        assert "registry_style" not in settings

    def test_combined_detection_merges_all_sources(self, tmp_path):
        _write(
            tmp_path / "config.yml",
            "llm:\n  default_provider: cborg\n  default_model: m\n"
            "channel_finder:\n  default_pipeline: tools\n",
        )
        _write(
            tmp_path / "pyproject.toml",
            '[project]\ndependencies = ["osprey-framework==9.9.9"]\n',
        )
        pkg = tmp_path / "src" / "combo"
        _write(pkg / "registry.py", "class R(OspreyFrameworkRegistry): pass  # extend\n")
        settings = detect_project_settings(tmp_path)
        assert settings["provider"] == "cborg"
        assert settings["channel_finder_mode"] == "tools"
        assert settings["estimated_osprey_version"] == "9.9.9"
        assert settings["package_name"] == "combo"


class TestPerformMigrationAnalysis:
    def _make_dirs(self, tmp_path):
        facility = tmp_path / "facility"
        old = tmp_path / "old"
        new = tmp_path / "new"
        for d in (facility, old, new):
            d.mkdir()
        return facility, old, new

    def test_auto_copy_when_template_changed_facility_untouched(self, tmp_path):
        facility, old, new = self._make_dirs(tmp_path)
        _write(facility / "config.yml", "v1")
        _write(old / "config.yml", "v1")
        _write(new / "config.yml", "v2")
        result = perform_migration_analysis(facility, old, new)
        assert [f["path"] for f in result["auto_copy"]] == ["config.yml"]
        assert result["merge"] == []

    def test_preserve_when_facility_changed_template_same(self, tmp_path):
        facility, old, new = self._make_dirs(tmp_path)
        _write(facility / "config.yml", "custom")
        _write(old / "config.yml", "orig")
        _write(new / "config.yml", "orig")
        result = perform_migration_analysis(facility, old, new)
        assert [f["path"] for f in result["preserve"]] == ["config.yml"]

    def test_merge_when_both_changed(self, tmp_path):
        facility, old, new = self._make_dirs(tmp_path)
        _write(facility / "config.yml", "facility-edit")
        _write(old / "config.yml", "orig")
        _write(new / "config.yml", "template-edit")
        result = perform_migration_analysis(facility, old, new)
        assert [f["path"] for f in result["merge"]] == ["config.yml"]
        info = result["merge"][0]
        assert info["facility_exists"] is True
        assert info["old_vanilla_exists"] is True
        assert info["new_vanilla_exists"] is True

    def test_new_file_only_in_new(self, tmp_path):
        facility, old, new = self._make_dirs(tmp_path)
        _write(new / "added.py", "brand new")
        result = perform_migration_analysis(facility, old, new)
        assert [f["path"] for f in result["new"]] == ["added.py"]

    def test_data_directory_files_categorized_as_data(self, tmp_path):
        facility, old, new = self._make_dirs(tmp_path)
        _write(facility / "data" / "state.db", "rows")
        result = perform_migration_analysis(facility, old, new)
        assert [f["path"] for f in result["data"]] == [str(Path("data") / "state.db")]

    def test_manifest_and_git_files_skipped(self, tmp_path):
        facility, old, new = self._make_dirs(tmp_path)
        _write(new / MANIFEST_FILENAME, "{}")
        _write(new / ".git" / "config", "gitstuff")
        _write(new / "real.py", "code")
        result = perform_migration_analysis(facility, old, new)
        all_paths = [f["path"] for cat in result.values() for f in cat]
        assert all_paths == ["real.py"]

    def test_none_old_vanilla_dir_handled(self, tmp_path):
        facility, _old, new = self._make_dirs(tmp_path)
        _write(facility / "keep.py", "facility only")
        _write(new / "keep.py", "template")
        # With no old baseline, an existing-in-both file falls to PRESERVE.
        result = perform_migration_analysis(facility, None, new)
        info = result["preserve"][0]
        assert info["path"] == "keep.py"
        assert info["old_vanilla_exists"] is False

    def test_nonexistent_directories_are_tolerated(self, tmp_path):
        facility = tmp_path / "facility"
        new = tmp_path / "new"
        new.mkdir()
        _write(new / "x.py", "code")
        # facility dir does not exist; analysis still runs over what's present.
        result = perform_migration_analysis(facility, None, new)
        assert [f["path"] for f in result["new"]] == ["x.py"]

    def test_all_categories_present_in_result(self, tmp_path):
        facility, old, new = self._make_dirs(tmp_path)
        _write(new / "x.py", "y")
        result = perform_migration_analysis(facility, old, new)
        assert set(result) == {
            "auto_copy",
            "preserve",
            "merge",
            "new",
            "data",
            "removed",
        }


class TestGenerateMergePrompt:
    def test_includes_core_sections_and_versions(self):
        prompt = generate_merge_prompt("config.yml", "FAC", "OLD", "NEW", "1.0.0", "2.0.0")
        assert "config.yml" in prompt
        assert "1.0.0 -> 2.0.0" in prompt
        assert "FAC" in prompt
        assert "OLD" in prompt
        assert "NEW" in prompt
        assert "Original Template (1.0.0)" in prompt
        assert "New Template (2.0.0)" in prompt

    def test_omits_original_template_when_none(self):
        prompt = generate_merge_prompt("config.yml", "FAC", None, "NEW", "1.0.0", "2.0.0")
        assert "Original Template" not in prompt
        assert "New Template (2.0.0)" in prompt

    def test_task_and_output_guidance_present(self):
        prompt = generate_merge_prompt("f", "a", "b", "c", "1", "2")
        assert "## Your Task" in prompt
        assert "## Output" in prompt


class TestGenerateMigrationDirectory:
    def _analysis(self):
        return {
            "auto_copy": [{"path": "auto.py"}],
            "preserve": [{"path": "kept.py"}],
            "merge": [{"path": "sub/dir/merged.yml"}],
            "new": [{"path": "new.py"}],
            "data": [{"path": "data/x.db"}],
            "removed": [{"path": "gone.py"}],
        }

    def _dirs(self, tmp_path):
        facility, old, new = (tmp_path / n for n in ("fac", "old", "new"))
        for d in (facility, old, new):
            d.mkdir()
        _write(facility / "sub" / "dir" / "merged.yml", "facility body")
        _write(old / "sub" / "dir" / "merged.yml", "old body")
        _write(new / "sub" / "dir" / "merged.yml", "new body")
        return facility, old, new

    def test_creates_directory_structure(self, tmp_path):
        facility, old, new = self._dirs(tmp_path)
        project = tmp_path / "project"
        project.mkdir()
        migration_dir = generate_migration_directory(
            project, self._analysis(), facility, old, new, "1.0", "2.0"
        )
        assert migration_dir == project / "_migration"
        assert (migration_dir / "merge_required").is_dir()
        assert (migration_dir / "auto_applied").is_dir()
        assert (migration_dir / "preserved").is_dir()
        assert (migration_dir / "README.md").exists()

    def test_readme_summarizes_counts_and_versions(self, tmp_path):
        facility, old, new = self._dirs(tmp_path)
        project = tmp_path / "project"
        project.mkdir()
        migration_dir = generate_migration_directory(
            project, self._analysis(), facility, old, new, "1.0", "2.0"
        )
        readme = (migration_dir / "README.md").read_text()
        assert "1.0 -> 2.0" in readme
        assert "`sub/dir/merged.yml`" in readme

    def test_merge_prompt_written_with_safe_filename(self, tmp_path):
        facility, old, new = self._dirs(tmp_path)
        project = tmp_path / "project"
        project.mkdir()
        migration_dir = generate_migration_directory(
            project, self._analysis(), facility, old, new, "1.0", "2.0"
        )
        prompt_file = migration_dir / "merge_required" / "sub_dir_merged.yml.md"
        assert prompt_file.exists()
        body = prompt_file.read_text()
        assert "facility body" in body
        assert "old body" in body
        assert "new body" in body

    def test_auto_and_preserved_summaries_list_files(self, tmp_path):
        facility, old, new = self._dirs(tmp_path)
        project = tmp_path / "project"
        project.mkdir()
        migration_dir = generate_migration_directory(
            project, self._analysis(), facility, old, new, "1.0", "2.0"
        )
        auto = (migration_dir / "auto_applied" / "summary.md").read_text()
        preserved = (migration_dir / "preserved" / "summary.md").read_text()
        assert "`auto.py`" in auto
        assert "`kept.py`" in preserved

    def test_unreadable_merge_source_uses_placeholder(self, tmp_path):
        facility, old, new = self._dirs(tmp_path)
        # Remove the new-side file so read_file_content returns None.
        (new / "sub" / "dir" / "merged.yml").unlink()
        project = tmp_path / "project"
        project.mkdir()
        migration_dir = generate_migration_directory(
            project, self._analysis(), facility, old, new, "1.0", "2.0"
        )
        body = (migration_dir / "merge_required" / "sub_dir_merged.yml.md").read_text()
        assert "[File not readable]" in body

    def test_none_old_vanilla_dir_supported(self, tmp_path):
        facility, _old, new = self._dirs(tmp_path)
        project = tmp_path / "project"
        project.mkdir()
        migration_dir = generate_migration_directory(
            project, self._analysis(), facility, None, new, "1.0", "2.0"
        )
        body = (migration_dir / "merge_required" / "sub_dir_merged.yml.md").read_text()
        # Without an old template the "Original Template" section is omitted.
        assert "Original Template" not in body

    def test_idempotent_when_directory_exists(self, tmp_path):
        facility, old, new = self._dirs(tmp_path)
        project = tmp_path / "project"
        project.mkdir()
        # Pre-create the _migration tree; exist_ok=True must not raise.
        (project / "_migration" / "merge_required").mkdir(parents=True)
        migration_dir = generate_migration_directory(
            project, self._analysis(), facility, old, new, "1.0", "2.0"
        )
        assert (migration_dir / "README.md").exists()


class TestMigrateClaudeCodeConfig:
    def test_empty_config_produces_no_changes(self):
        servers, agents, changes = migrate_claude_code_config({})
        assert servers == {}
        assert agents == {}
        assert changes == []

    def test_disable_servers_becomes_enabled_false(self):
        servers, _agents, changes = migrate_claude_code_config({"disable_servers": ["scan"]})
        assert servers == {"scan": {"enabled": False}}
        assert any("disable_servers: scan" in c for c in changes)

    def test_extra_servers_merged_and_copied(self):
        spec = {"command": "run"}
        servers, _agents, changes = migrate_claude_code_config({"extra_servers": {"custom": spec}})
        assert servers == {"custom": {"command": "run"}}
        # The spec is copied, not aliased.
        assert servers["custom"] is not spec
        assert any("extra_servers: custom" in c for c in changes)

    def test_extra_servers_non_mapping_passed_through(self):
        servers, _agents, _changes = migrate_claude_code_config(
            {"extra_servers": {"weird": "not-a-dict"}}
        )
        assert servers == {"weird": "not-a-dict"}

    def test_disable_agents_becomes_enabled_false(self):
        _servers, agents, changes = migrate_claude_code_config({"disable_agents": ["planner"]})
        assert agents == {"planner": {"enabled": False}}
        assert any("disable_agents: planner" in c for c in changes)

    def test_existing_servers_and_agents_preserved(self):
        servers, agents, _changes = migrate_claude_code_config(
            {
                "servers": {"keep": {"enabled": True}},
                "agents": {"keepagent": {"enabled": True}},
                "disable_servers": ["off"],
            }
        )
        assert servers["keep"] == {"enabled": True}
        assert servers["off"] == {"enabled": False}
        assert agents["keepagent"] == {"enabled": True}

    def test_all_transforms_combined(self):
        servers, agents, changes = migrate_claude_code_config(
            {
                "disable_servers": ["a"],
                "extra_servers": {"b": {"cmd": "x"}},
                "disable_agents": ["c"],
            }
        )
        assert servers == {"a": {"enabled": False}, "b": {"cmd": "x"}}
        assert agents == {"c": {"enabled": False}}
        assert len(changes) == 3
