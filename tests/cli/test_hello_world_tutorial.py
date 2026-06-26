"""Smoke tests: build hello-world tutorial profile end-to-end.

These tests verify that the hello-world example profile builds correctly
through the profile-driven build system, producing a project with the
expected config, channel limits file, hooks, and mock connector integration.

The tests use TemplateManager.create_project() directly, skipping:
  - venv creation (--skip-deps equivalent)
  - overlay file copying
  - lifecycle phases (pre_build / post_build / validate)

What IS verified:
  - Profile parses and passes artifact validation
  - data_bundle is correctly resolved to "hello_world"
  - Key structural files are generated (CLAUDE.md, .mcp.json, config.yml)
  - config.yml has mock control system with limits checking enabled
  - channel_limits.json is copied with correct channel entries
  - Hook files are present in .claude/hooks/
  - MockConnector reads tutorial channels successfully
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _resolve_preset_profile(name: str) -> Path:
    """Resolve a bundled preset profile name to its filesystem path."""
    from importlib.resources import files

    presets_dir = Path(str(files("osprey").joinpath("profiles/presets")))
    profile_path = presets_dir / f"{name}.yml"
    if not profile_path.exists():
        raise FileNotFoundError(f"Preset profile '{name}' not found at {profile_path}")
    return profile_path


def _build_from_profile(profile_name: str, project_name: str, tmp_path: Path) -> Path:
    """Build a project from a bundled example profile using TemplateManager directly.

    Equivalent to ``osprey build <project_name> <profile> --skip-deps``
    without overlay file copying.
    """
    from osprey.cli.build_profile import load_profile
    from osprey.cli.templates.artifact_library import validate_artifacts
    from osprey.cli.templates.manager import TemplateManager

    profile_path = _resolve_preset_profile(profile_name)
    build_profile = load_profile(profile_path)

    # Collect and validate artifact selections (mirrors build_cmd.py step 1b)
    artifacts: dict[str, list[str]] = {}
    for artifact_type in ("hooks", "rules", "skills", "agents", "output_styles"):
        names = getattr(build_profile, artifact_type, [])
        if names:
            artifacts[artifact_type] = list(names)

    if artifacts:
        validate_artifacts(artifacts)

    # Build context from profile fields (mirrors build_cmd.py step 6)
    context: dict[str, str] = {}
    if build_profile.provider:
        context["default_provider"] = build_profile.provider
    if build_profile.model:
        context["default_model"] = build_profile.model

    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name=project_name,
        output_dir=tmp_path,
        data_bundle=build_profile.data_bundle,
        context=context,
        force=False,
        artifacts=artifacts or None,
    )

    # Apply config overrides (mirrors build_cmd.py step 8)
    if build_profile.config:
        from osprey.cli.build_cmd import _apply_config_overrides

        _apply_config_overrides(project_dir, build_profile.config)

    return project_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hello_world_project(tmp_path: Path) -> Path:
    return _build_from_profile("hello-world", "hello-tutorial", tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHelloWorldProfileLoads:
    """Verify hello-world.yml parses and validates correctly."""

    def test_hello_world_profile_loads(self):
        """Load hello-world profile and verify data_bundle and artifact validation."""
        from osprey.cli.build_profile import load_profile
        from osprey.cli.templates.artifact_library import validate_artifacts

        profile_path = _resolve_preset_profile("hello-world")
        profile = load_profile(profile_path)

        assert profile.data_bundle == "hello_world"

        # Collect and validate all artifacts
        artifacts: dict[str, list[str]] = {}
        for artifact_type in ("hooks", "rules", "skills", "agents", "output_styles"):
            names = getattr(profile, artifact_type, [])
            if names:
                artifacts[artifact_type] = list(names)

        # Should not raise
        validate_artifacts(artifacts)


@pytest.mark.unit
class TestHelloWorldBuildOutput:
    """Verify hello-world profile produces a correct project structure."""

    def test_hello_world_build_creates_valid_project(self, hello_world_project: Path):
        """Build project and verify structural files and config values."""
        # Key files exist
        assert (hello_world_project / "config.yml").exists()
        assert (hello_world_project / ".mcp.json").exists()
        assert (hello_world_project / "CLAUDE.md").exists()

        # Parse and verify config.yml
        config = yaml.safe_load((hello_world_project / "config.yml").read_text())
        assert config["control_system"]["type"] == "mock"
        assert config["control_system"]["limits_checking"]["enabled"] is True
        assert (
            config["control_system"]["limits_checking"]["database_path"]
            == "data/channel_limits.json"
        )

    def test_hello_world_limits_file_copied(self, hello_world_project: Path):
        """Verify channel_limits.json is copied with correct channel entries."""
        limits_path = hello_world_project / "data" / "channel_limits.json"
        assert limits_path.exists()

        limits = json.loads(limits_path.read_text())

        # Filter to actual channel entries (exclude metadata keys)
        channels = {k: v for k, v in limits.items() if not k.startswith("_") and k != "defaults"}
        assert len(channels) == 3

        # Verify specific channel properties
        assert limits["SR:MAG:QF:01:CURRENT:SP"]["max_value"] == 300.0
        assert limits["SR:BEAM:CURRENT"]["writable"] is False

    def test_hello_world_hooks_present(self, hello_world_project: Path):
        """Check .claude/hooks/ directory exists with expected hook files."""
        hooks_dir = hello_world_project / ".claude" / "hooks"
        assert hooks_dir.exists()

        # Hello-world profile declares: writes-check, approval, limits
        # These map to osprey_writes_check.py, osprey_approval.py, osprey_limits.py
        assert (hooks_dir / "osprey_writes_check.py").exists()
        assert (hooks_dir / "osprey_approval.py").exists()
        assert (hooks_dir / "osprey_limits.py").exists()


@pytest.mark.unit
@pytest.mark.asyncio
class TestMockConnectorTutorialChannels:
    """Verify MockConnector can read tutorial channel names."""

    async def test_mock_connector_reads_tutorial_channels(self):
        """Instantiate MockConnector and read tutorial channels."""
        from osprey.connectors.control_system.mock_connector import MockConnector

        connector = MockConnector()
        await connector.connect({})

        channel_names = ["SR:BEAM:CURRENT", "SR:MAG:QF:01:CURRENT:RB"]
        for name in channel_names:
            result = await connector.read_channel(name)
            assert result is not None
            assert isinstance(result.value, (int, float))
