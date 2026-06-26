"""Unit tests for osprey.cli.templates.artifact_library."""

from __future__ import annotations

import pytest

from osprey.cli.templates.artifact_library import (
    ARTIFACT_TYPES,
    list_artifacts,
    resolve_artifact,
    validate_artifacts,
)

# ---------------------------------------------------------------------------
# list_artifacts
# ---------------------------------------------------------------------------


class TestListArtifacts:
    def test_hooks_returns_list(self):
        names = list_artifacts("hooks")
        assert isinstance(names, list)
        assert len(names) > 0

    def test_hooks_are_sorted(self):
        names = list_artifacts("hooks")
        assert names == sorted(names)

    def test_hooks_excludes_pycache(self):
        names = list_artifacts("hooks")
        assert "__pycache__" not in names

    def test_hooks_includes_hook_config(self):
        # hook_config.json.j2 is included as "hook-config" short name
        names = list_artifacts("hooks")
        assert "hook-config" in names
        assert "hook_config" not in names  # Raw stem should not appear

    def test_hooks_contains_known_hooks(self):
        names = list_artifacts("hooks")
        # Hooks use user-facing short names: osprey_ prefix stripped, underscores → hyphens
        assert "approval" in names
        assert "writes-check" in names
        assert "limits" in names
        assert "hook-log" in names
        assert "hook-config" in names

    def test_rules_returns_list(self):
        names = list_artifacts("rules")
        assert isinstance(names, list)
        assert len(names) > 0

    def test_rules_are_sorted(self):
        names = list_artifacts("rules")
        assert names == sorted(names)

    def test_rules_contains_known_rules(self):
        names = list_artifacts("rules")
        assert "safety" in names
        assert "error-handling" in names
        assert "artifacts" in names
        assert "python-execution" in names

    def test_rules_short_names_have_no_j2_suffix(self):
        names = list_artifacts("rules")
        for name in names:
            assert not name.endswith(".j2"), f"Rule name should not end with .j2: {name}"
            assert not name.endswith(".md"), f"Rule name should not end with .md: {name}"

    def test_skills_returns_list(self):
        names = list_artifacts("skills")
        assert isinstance(names, list)
        assert len(names) > 0

    def test_skills_are_sorted(self):
        names = list_artifacts("skills")
        assert names == sorted(names)

    def test_skills_contains_known_skills(self):
        names = list_artifacts("skills")
        assert "diagnose" in names
        assert "setup-mode" in names
        assert "session-report" in names

    def test_skills_excludes_underscore_dirs(self):
        # e.g. _terminology should be excluded
        names = list_artifacts("skills")
        for name in names:
            assert not name.startswith("_"), f"Skill name should not start with _: {name}"

    def test_agents_returns_list(self):
        names = list_artifacts("agents")
        assert isinstance(names, list)
        assert len(names) > 0

    def test_agents_are_sorted(self):
        names = list_artifacts("agents")
        assert names == sorted(names)

    def test_agents_contains_known_agents(self):
        names = list_artifacts("agents")
        assert "channel-finder" in names
        assert "data-visualizer" in names

    def test_agents_short_names_have_no_j2_suffix(self):
        names = list_artifacts("agents")
        for name in names:
            assert not name.endswith(".j2")
            assert not name.endswith(".md")

    def test_output_styles_returns_list(self):
        names = list_artifacts("output_styles")
        assert isinstance(names, list)
        assert len(names) > 0

    def test_output_styles_contains_control_operator(self):
        names = list_artifacts("output_styles")
        assert "control-operator" in names

    def test_output_styles_short_names_have_no_suffix(self):
        names = list_artifacts("output_styles")
        for name in names:
            assert not name.endswith(".j2")
            assert not name.endswith(".md")

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown artifact type"):
            list_artifacts("web_panels")

    def test_all_types_covered(self):
        for artifact_type in ARTIFACT_TYPES:
            names = list_artifacts(artifact_type)
            assert isinstance(names, list)


# ---------------------------------------------------------------------------
# validate_artifacts
# ---------------------------------------------------------------------------


class TestValidateArtifacts:
    def test_valid_hooks_passes(self):
        # Hooks use user-facing short names (osprey_ prefix stripped, _ → -)
        validate_artifacts({"hooks": ["approval", "writes-check"]})

    def test_valid_rules_passes(self):
        validate_artifacts({"rules": ["safety", "error-handling"]})

    def test_valid_skills_passes(self):
        validate_artifacts({"skills": ["diagnose", "setup-mode"]})

    def test_valid_agents_passes(self):
        validate_artifacts({"agents": ["channel-finder"]})

    def test_valid_output_styles_passes(self):
        validate_artifacts({"output_styles": ["control-operator"]})

    def test_empty_dict_passes(self):
        validate_artifacts({})

    def test_empty_list_passes(self):
        validate_artifacts({"hooks": [], "rules": []})

    def test_unknown_hook_raises(self):
        with pytest.raises(ValueError) as exc_info:
            validate_artifacts({"hooks": ["nonexistent_hook"]})
        msg = str(exc_info.value)
        assert "nonexistent_hook" in msg
        assert "hooks" in msg

    def test_error_lists_alternatives(self):
        with pytest.raises(ValueError) as exc_info:
            validate_artifacts({"rules": ["nonexistent_rule"]})
        msg = str(exc_info.value)
        # Should list available rules
        assert "safety" in msg or "Available rules:" in msg

    def test_multiple_invalid_names_all_reported(self):
        with pytest.raises(ValueError) as exc_info:
            validate_artifacts({"hooks": ["bad_hook_1", "bad_hook_2"]})
        msg = str(exc_info.value)
        assert "bad_hook_1" in msg
        assert "bad_hook_2" in msg

    def test_mixed_valid_invalid_raises_for_invalid(self):
        with pytest.raises(ValueError) as exc_info:
            validate_artifacts({"rules": ["safety", "nonexistent_rule"]})
        msg = str(exc_info.value)
        assert "nonexistent_rule" in msg
        # "safety" should not be in the error portion as an invalid entry
        assert "safety" not in msg.split("nonexistent_rule")[0].replace("Available", "")

    def test_did_you_mean_suggestion_for_typo(self):
        with pytest.raises(ValueError) as exc_info:
            validate_artifacts({"rules": ["safty"]})  # typo for "safety"
        msg = str(exc_info.value)
        assert "safety" in msg  # should suggest "safety"

    def test_multiple_types_validated_together(self):
        with pytest.raises(ValueError) as exc_info:
            validate_artifacts(
                {
                    "hooks": ["approval"],  # valid
                    "rules": ["nonexistent"],  # invalid
                    "skills": ["diagnose"],  # valid
                }
            )
        msg = str(exc_info.value)
        assert "nonexistent" in msg
        # The valid entries should not appear as errors
        assert "approval" not in msg.split("nonexistent")[0]
        assert "diagnose" not in msg


# ---------------------------------------------------------------------------
# resolve_artifact
# ---------------------------------------------------------------------------


class TestResolveArtifact:
    def test_resolve_hook_returns_path(self):
        # Hooks use user-facing short names (osprey_ prefix stripped, _ → -)
        path = resolve_artifact("hooks", "approval")
        assert path.exists()
        assert path.is_file()
        assert path.suffix == ".py"

    def test_resolve_hook_correct_name(self):
        path = resolve_artifact("hooks", "writes-check")
        assert "osprey_writes_check" in path.name

    def test_resolve_rule_returns_path(self):
        path = resolve_artifact("rules", "safety")
        assert path.exists()
        assert path.is_file()
        # File may be .md or .md.j2
        assert "safety" in path.stem or "safety" in path.name

    def test_resolve_rule_j2_template(self):
        # control-system-safety is a .md.j2 template
        path = resolve_artifact("rules", "control-system-safety")
        assert path.exists()
        assert path.is_file()

    def test_resolve_skill_returns_directory(self):
        path = resolve_artifact("skills", "diagnose")
        assert path.exists()
        assert path.is_dir()
        assert path.name == "diagnose"

    def test_resolve_skill_directory_contains_skill_md(self):
        path = resolve_artifact("skills", "session-report")
        assert (path / "SKILL.md").exists()

    def test_resolve_agent_returns_path(self):
        path = resolve_artifact("agents", "channel-finder")
        assert path.exists()
        assert path.is_file()

    def test_resolve_output_style_returns_path(self):
        path = resolve_artifact("output_styles", "control-operator")
        assert path.exists()
        assert path.is_file()

    def test_resolve_unknown_hook_raises(self):
        with pytest.raises(ValueError) as exc_info:
            resolve_artifact("hooks", "nonexistent_hook")
        msg = str(exc_info.value)
        assert "nonexistent_hook" in msg
        assert "hooks" in msg

    def test_resolve_unknown_rule_raises_with_alternatives(self):
        with pytest.raises(ValueError) as exc_info:
            resolve_artifact("rules", "nonexistent_rule")
        msg = str(exc_info.value)
        assert "nonexistent_rule" in msg
        assert "Available rules:" in msg

    def test_resolve_unknown_skill_raises(self):
        with pytest.raises(ValueError) as exc_info:
            resolve_artifact("skills", "unknown_skill")
        msg = str(exc_info.value)
        assert "unknown_skill" in msg

    def test_resolve_bad_type_raises(self):
        with pytest.raises(ValueError, match="Unknown artifact type"):
            resolve_artifact("web_panels", "ariel")

    def test_resolved_paths_are_absolute(self):
        for artifact_type in ARTIFACT_TYPES:
            names = list_artifacts(artifact_type)
            if names:
                path = resolve_artifact(artifact_type, names[0])
                assert path.is_absolute()

    def test_resolve_returns_path_inside_package(self):
        path = resolve_artifact("hooks", "approval")
        # Should be inside the osprey package templates
        assert "osprey" in str(path)
        assert "templates" in str(path)
