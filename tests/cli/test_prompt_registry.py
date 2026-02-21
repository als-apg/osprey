"""Tests for PromptRegistry — the declarative catalog of prompt artifacts."""

from pathlib import Path

import pytest

from osprey.cli.prompt_registry import PromptArtifact, PromptRegistry


class TestPromptArtifact:
    """Basic PromptArtifact dataclass tests."""

    def test_frozen(self):
        art = PromptArtifact("a", "b", "c", "d")
        with pytest.raises(AttributeError):
            art.canonical_name = "x"

    def test_fields(self):
        art = PromptArtifact("a", "b", "c", "d")
        assert art.canonical_name == "a"
        assert art.template_path == "b"
        assert art.output_path == "c"
        assert art.description == "d"


class TestPromptRegistryDefault:
    """Tests for the default registry contents."""

    @pytest.fixture()
    def registry(self):
        return PromptRegistry.default()

    def test_has_artifacts(self, registry):
        assert len(registry.all_artifacts()) > 0

    def test_get_known_artifact(self, registry):
        art = registry.get("claude-md")
        assert art is not None
        assert art.output_path == "CLAUDE.md"
        assert art.template_path == "CLAUDE.md.j2"

    def test_get_unknown_returns_none(self, registry):
        assert registry.get("nonexistent/artifact") is None

    def test_all_names_sorted(self, registry):
        names = registry.all_names()
        assert names == sorted(names)

    def test_all_names_nonempty(self, registry):
        assert len(registry.all_names()) > 0

    def test_get_by_output(self, registry):
        art = registry.get_by_output("CLAUDE.md")
        assert art is not None
        assert art.canonical_name == "claude-md"

    def test_get_by_output_unknown(self, registry):
        assert registry.get_by_output("nonexistent.md") is None

    # ── Known artifacts ────────────────────────────────────────────
    @pytest.mark.parametrize(
        "name",
        [
            "claude-md",
            "mcp-json",
            "settings-json",
            "agents/channel-finder",
            "agents/data-visualizer",
            "agents/graph-analyst",
            "agents/literature-search",
            "agents/logbook-search",
            "agents/logbook-deep-research",
            "agents/matlab-search",
            "agents/wiki-search",
            "rules/safety",
            "rules/error-handling",
            "rules/artifacts",
            "rules/facility",
            "hooks/approval",
            "hooks/writes-check",
            "hooks/error-guidance",
            "hooks/limits",
            "hooks/notebook-update",
            "skills/session-report",
            "skills/session-report/reference",
        ],
    )
    def test_known_artifact_exists(self, registry, name):
        art = registry.get(name)
        assert art is not None, f"Missing artifact: {name}"
        assert art.canonical_name == name

    def test_facility_exists(self, registry):
        art = registry.get("rules/facility")
        assert art is not None
        assert art.description == "Facility identity & context"


class TestRegistryMatchesTemplateDirectory:
    """Verify that the registry matches the actual template files on disk."""

    def test_all_template_paths_exist(self):
        """Every artifact's template_path must exist in the template directory."""
        registry = PromptRegistry.default()
        template_root = Path(__file__).parent.parent.parent / "src" / "osprey" / "templates" / "claude_code"

        for art in registry.all_artifacts():
            template_file = template_root / art.template_path
            assert template_file.exists(), (
                f"Template {art.template_path} for artifact "
                f"'{art.canonical_name}' not found at {template_file}"
            )

    def test_no_unregistered_templates(self):
        """All files in the template directory should be registered.

        Exemptions: __pycache__, .pyc, directories-only, __init__.py
        """
        registry = PromptRegistry.default()
        template_root = Path(__file__).parent.parent.parent / "src" / "osprey" / "templates" / "claude_code"

        registered_templates = {a.template_path for a in registry.all_artifacts()}

        for template_file in template_root.rglob("*"):
            if not template_file.is_file():
                continue
            if "__pycache__" in str(template_file):
                continue
            if template_file.name == "__init__.py":
                continue

            rel = str(template_file.relative_to(template_root))
            assert rel in registered_templates, (
                f"Template file {rel} is not registered in the PromptRegistry"
            )


class TestCustomRegistry:
    """Tests for constructing a custom registry."""

    def test_empty_registry(self):
        reg = PromptRegistry([])
        assert reg.all_artifacts() == []
        assert reg.all_names() == []
        assert reg.get("anything") is None

    def test_custom_artifacts(self):
        art = PromptArtifact("my/thing", "my.j2", "my.md", "My thing")
        reg = PromptRegistry([art])
        assert reg.get("my/thing") is art
        assert reg.all_names() == ["my/thing"]
