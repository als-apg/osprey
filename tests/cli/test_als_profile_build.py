"""Smoke tests: build ALS production and client profiles end-to-end.

These tests verify that the updated ALS profiles (als-prod.yml, als-client.yml)
build correctly through the new profile-driven build system, producing projects
with the expected artifacts, config, and manifest.

The tests use TemplateManager.create_project() directly, skipping:
  - venv creation (--skip-deps equivalent)
  - overlay file copying (overlay sources live in als-profiles repo, not in osprey)
  - lifecycle phases (pre_build / post_build / validate)

What IS verified:
  - Both profiles parse and pass artifact validation
  - data_bundle is correctly resolved to "control_assistant"
  - Key structural files are generated (CLAUDE.md, .mcp.json, config.yml)
  - .osprey-manifest.json persists artifacts and data_bundle
  - CLAUDE.md renders the control-system variant (not lattice-physics)
  - channel-finder is in agents list (both ALS profiles use it)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

ALS_PROFILES_DIR = Path("/Users/thellert/LBL/ML/als-profiles")
ALS_PROD_PROFILE = ALS_PROFILES_DIR / "als-prod.yml"
ALS_CLIENT_PROFILE = ALS_PROFILES_DIR / "als-client.yml"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_from_profile(profile_path: Path, project_name: str, tmp_path: Path) -> Path:
    """Build a project from an ALS profile using TemplateManager directly.

    Equivalent to `osprey build <project_name> <profile_path> --skip-deps`
    without overlay file copying (overlay sources are external to osprey).
    """
    from osprey.cli.build_profile import load_profile
    from osprey.cli.templates.artifact_library import validate_artifacts
    from osprey.cli.templates.manager import TemplateManager

    build_profile = load_profile(profile_path)

    # Collect and validate artifact selections (mirrors build_cmd.py step 1b)
    artifacts: dict[str, list[str]] = {}
    for artifact_type in ("hooks", "rules", "skills", "agents", "output_styles"):
        names = getattr(build_profile, artifact_type, [])
        if names:
            artifacts[artifact_type] = list(names)

    if artifacts:
        validate_artifacts(artifacts)  # Raises ValueError on unknown names

    # Build context from profile fields (mirrors build_cmd.py step 6)
    context: dict[str, str] = {}
    if build_profile.provider:
        context["default_provider"] = build_profile.provider
    if build_profile.model:
        context["default_model"] = build_profile.model
    if build_profile.channel_finder_mode:
        context["channel_finder_mode"] = build_profile.channel_finder_mode

    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name=project_name,
        output_dir=tmp_path,
        data_bundle=build_profile.data_bundle,
        registry_style="extend",
        context=context,
        force=False,
        artifacts=artifacts or None,
    )

    # Apply config overrides (mirrors build_cmd.py step 8)
    if build_profile.config:
        from osprey.cli.build_cmd import _apply_config_overrides

        _apply_config_overrides(project_dir, build_profile.config)

    # Generate manifest with artifact selections (mirrors build_cmd.py step 10)
    manager.generate_manifest(
        project_dir=project_dir,
        project_name=project_name,
        data_bundle=build_profile.data_bundle,
        registry_style="extend",
        artifacts=artifacts or None,
    )

    return project_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def als_prod_project(tmp_path: Path) -> Path:
    return _build_from_profile(ALS_PROD_PROFILE, "als-assistant", tmp_path)


@pytest.fixture
def als_client_project(tmp_path: Path) -> Path:
    return _build_from_profile(ALS_CLIENT_PROFILE, "als-client", tmp_path)


# ---------------------------------------------------------------------------
# Profile Parsing Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not ALS_PROD_PROFILE.exists(),
    reason="als-profiles repo not present at /Users/thellert/LBL/ML/als-profiles",
)
class TestAlsProdProfileParsing:
    """Verify als-prod.yml parses and validates correctly."""

    def test_profile_loads(self):
        from osprey.cli.build_profile import load_profile

        profile = load_profile(ALS_PROD_PROFILE)
        assert profile.name == "ALS Assistant (Production)"
        assert profile.data_bundle == "control_assistant"

    def test_profile_artifact_validation_passes(self):
        from osprey.cli.build_profile import load_profile
        from osprey.cli.templates.artifact_library import validate_artifacts

        profile = load_profile(ALS_PROD_PROFILE)
        artifacts: dict[str, list[str]] = {}
        for artifact_type in ("hooks", "rules", "skills", "agents", "output_styles"):
            names = getattr(profile, artifact_type, [])
            if names:
                artifacts[artifact_type] = list(names)

        # Should not raise
        validate_artifacts(artifacts)

    def test_profile_has_expected_artifact_types(self):
        from osprey.cli.build_profile import load_profile

        profile = load_profile(ALS_PROD_PROFILE)
        assert len(profile.hooks) > 0
        assert len(profile.rules) > 0
        assert len(profile.skills) > 0
        assert len(profile.agents) > 0
        assert len(profile.output_styles) > 0

    def test_profile_has_channel_finder_agent(self):
        from osprey.cli.build_profile import load_profile

        profile = load_profile(ALS_PROD_PROFILE)
        assert "channel-finder" in profile.agents

    def test_profile_has_no_lattice_artifacts(self):
        from osprey.cli.build_profile import load_profile

        profile = load_profile(ALS_PROD_PROFILE)
        assert "lattice-physics" not in profile.rules


@pytest.mark.skipif(
    not ALS_CLIENT_PROFILE.exists(),
    reason="als-profiles repo not present at /Users/thellert/LBL/ML/als-profiles",
)
class TestAlsClientProfileParsing:
    """Verify als-client.yml parses and validates correctly."""

    def test_profile_loads(self):
        from osprey.cli.build_profile import load_profile

        profile = load_profile(ALS_CLIENT_PROFILE)
        assert profile.name == "ALS Client"
        assert profile.data_bundle == "control_assistant"

    def test_profile_artifact_validation_passes(self):
        from osprey.cli.build_profile import load_profile
        from osprey.cli.templates.artifact_library import validate_artifacts

        profile = load_profile(ALS_CLIENT_PROFILE)
        artifacts: dict[str, list[str]] = {}
        for artifact_type in ("hooks", "rules", "skills", "agents", "output_styles"):
            names = getattr(profile, artifact_type, [])
            if names:
                artifacts[artifact_type] = list(names)

        validate_artifacts(artifacts)

    def test_profile_has_channel_finder_agent(self):
        from osprey.cli.build_profile import load_profile

        profile = load_profile(ALS_CLIENT_PROFILE)
        assert "channel-finder" in profile.agents

    def test_prod_and_client_share_same_artifact_set(self):
        """ALS prod and client should declare the same core artifacts."""
        from osprey.cli.build_profile import load_profile

        prod = load_profile(ALS_PROD_PROFILE)
        client = load_profile(ALS_CLIENT_PROFILE)

        for artifact_type in ("hooks", "rules", "skills", "agents", "output_styles"):
            prod_names = set(getattr(prod, artifact_type, []))
            client_names = set(getattr(client, artifact_type, []))
            assert prod_names == client_names, (
                f"{artifact_type} mismatch between als-prod.yml and als-client.yml: "
                f"prod={sorted(prod_names)}, client={sorted(client_names)}"
            )


# ---------------------------------------------------------------------------
# Build Output Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not ALS_PROD_PROFILE.exists(),
    reason="als-profiles repo not present at /Users/thellert/LBL/ML/als-profiles",
)
class TestAlsProdBuildOutput:
    """Verify als-prod.yml produces a correct project structure."""

    def test_project_directory_created(self, als_prod_project: Path):
        assert als_prod_project.exists()
        assert als_prod_project.is_dir()

    def test_config_yml_exists(self, als_prod_project: Path):
        assert (als_prod_project / "config.yml").exists()

    def test_claude_md_exists(self, als_prod_project: Path):
        assert (als_prod_project / "CLAUDE.md").exists()

    def test_mcp_json_exists(self, als_prod_project: Path):
        assert (als_prod_project / ".mcp.json").exists()

    def test_osprey_manifest_exists(self, als_prod_project: Path):
        from osprey.cli.templates.manifest import MANIFEST_FILENAME

        assert (als_prod_project / MANIFEST_FILENAME).exists()

    def test_manifest_persists_data_bundle(self, als_prod_project: Path):
        from osprey.cli.templates.manifest import MANIFEST_FILENAME

        manifest_data = json.loads((als_prod_project / MANIFEST_FILENAME).read_text())
        assert manifest_data["creation"]["data_bundle"] == "control_assistant"

    def test_manifest_persists_artifacts(self, als_prod_project: Path):
        from osprey.cli.templates.manifest import MANIFEST_FILENAME

        manifest_data = json.loads((als_prod_project / MANIFEST_FILENAME).read_text())
        artifacts = manifest_data.get("artifacts", {})
        assert "hooks" in artifacts
        assert "rules" in artifacts
        assert "skills" in artifacts
        assert "agents" in artifacts
        assert "output_styles" in artifacts
        assert "channel-finder" in artifacts["agents"]

    def test_config_yml_has_epics_control_system(self, als_prod_project: Path):
        """Config overrides from als-prod.yml should be applied."""
        config = yaml.safe_load((als_prod_project / "config.yml").read_text())
        assert config["control_system"]["type"] == "epics"

    def test_config_yml_has_als_timezone(self, als_prod_project: Path):
        config = yaml.safe_load((als_prod_project / "config.yml").read_text())
        assert config["system"]["timezone"] == "America/Los_Angeles"

    def test_claude_md_is_control_system_variant(self, als_prod_project: Path):
        """ALS prod should use control-system CLAUDE.md, not lattice-physics."""
        claude_md = (als_prod_project / "CLAUDE.md").read_text()
        # Control assistant variant
        assert "Control System Assistant" in claude_md
        # Lattice variant should NOT be present
        assert "Lattice Physics Design" not in claude_md

    def test_hooks_directory_has_selected_hooks(self, als_prod_project: Path):
        hooks_dir = als_prod_project / ".claude" / "hooks"
        assert hooks_dir.exists()
        # approval hook should be present (listed in als-prod.yml)
        assert (hooks_dir / "osprey_approval.py").exists()

    def test_rules_directory_has_safety_rule(self, als_prod_project: Path):
        rule = als_prod_project / ".claude" / "rules" / "safety.md"
        assert rule.exists()

    def test_rules_directory_has_no_lattice_rule(self, als_prod_project: Path):
        """lattice-physics.md should NOT exist in control-assistant builds."""
        rule = als_prod_project / ".claude" / "rules" / "lattice-physics.md"
        # Either the file doesn't exist, or it exists but is empty/whitespace
        if rule.exists():
            assert not rule.read_text().strip(), "lattice-physics.md should be empty for ALS prod"


@pytest.mark.skipif(
    not ALS_CLIENT_PROFILE.exists(),
    reason="als-profiles repo not present at /Users/thellert/LBL/ML/als-profiles",
)
class TestAlsClientBuildOutput:
    """Verify als-client.yml produces a correct project structure."""

    def test_project_directory_created(self, als_client_project: Path):
        assert als_client_project.exists()

    def test_config_yml_has_epics_control_system(self, als_client_project: Path):
        config = yaml.safe_load((als_client_project / "config.yml").read_text())
        assert config["control_system"]["type"] == "epics"

    def test_config_yml_has_als_timezone(self, als_client_project: Path):
        config = yaml.safe_load((als_client_project / "config.yml").read_text())
        assert config["system"]["timezone"] == "America/Los_Angeles"

    def test_claude_md_is_control_system_variant(self, als_client_project: Path):
        claude_md = (als_client_project / "CLAUDE.md").read_text()
        assert "Control System Assistant" in claude_md
        assert "Lattice Physics Design" not in claude_md

    def test_manifest_persists_artifacts(self, als_client_project: Path):
        from osprey.cli.templates.manifest import MANIFEST_FILENAME

        manifest_data = json.loads((als_client_project / MANIFEST_FILENAME).read_text())
        artifacts = manifest_data.get("artifacts", {})
        assert "channel-finder" in artifacts.get("agents", [])

    def test_osprey_manifest_data_bundle_is_control_assistant(self, als_client_project: Path):
        from osprey.cli.templates.manifest import MANIFEST_FILENAME

        manifest_data = json.loads((als_client_project / MANIFEST_FILENAME).read_text())
        assert manifest_data["creation"]["data_bundle"] == "control_assistant"
