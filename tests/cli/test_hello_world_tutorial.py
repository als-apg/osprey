"""Smoke tests: build the hello-world tutorial preset end-to-end.

These tests run ``osprey init --preset hello-world`` and ``osprey build
--skip-deps`` once, and read the render the tutorial walks through.

What IS verified:
  - Preset parses and passes artifact validation
  - data_bundle is correctly resolved to "hello_world"
  - Key structural files are generated (CLAUDE.md, .mcp.json, config.yml)
  - config.yml has mock control system with limits checking enabled, in
    ``optional`` mode, reading the limits database the build writes
  - channel_limits.json holds the three records of data/facility/limits.yaml
  - Hook files are present in .claude/hooks/
  - MockConnector reads tutorial channels successfully
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from osprey.cli.build_cmd import _profile_data_bundle
from tests._builds import BuiltProject, init_project, run_build

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hello_world_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The render of a freshly initialised and built hello-world repo."""
    repo = init_project(tmp_path_factory.mktemp("hello-tutorial"), "hello-world", "hello-tutorial")
    result = run_build(repo)
    assert result.exit_code == 0, result.output
    return BuiltProject(repo).build_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHelloWorldProfileLoads:
    """Verify hello-world.yml parses and validates correctly."""

    def test_hello_world_profile_loads(self):
        """Load the hello-world preset and validate the artifacts it selects.

        The bundle is a preset-side fact rather than a profile key, so it is
        read back from the preset the way the build reads it.
        """
        from osprey.cli.build_profile import resolve_build_profile
        from osprey.cli.templates.artifact_library import validate_artifacts

        profile, _profile_dir = resolve_build_profile(None, "hello-world")

        assert _profile_data_bundle(profile) == "hello_world"

        # Collect and validate all artifacts
        artifacts: dict[str, list[str]] = {}
        for artifact_type in ("hooks", "rules", "skills", "agents", "output_styles"):
            names = getattr(profile, artifact_type, [])
            if names:
                artifacts[artifact_type] = list(names)

        # Should not raise
        validate_artifacts(artifacts)


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
        assert config["control_system"]["limits_checking"] == {
            "enabled": True,
            "mode": "optional",
            "database_path": "data/channel_limits.json",
        }

    def test_hello_world_limits_file_holds_the_three_records(self, hello_world_project: Path):
        """Verify channel_limits.json is the three records of limits.yaml, resolved."""
        limits = json.loads((hello_world_project / "data" / "channel_limits.json").read_text())

        assert limits == {
            "_version": "4.0",
            "SR:BEAM:CURRENT": {"writable": False, "confirm": True},
            "SR:MAG:QD:01:CURRENT:SP": {
                "min_value": 0.0,
                "max_value": 250.0,
                "writable": True,
                "confirm": True,
            },
            "SR:MAG:QF:01:CURRENT:SP": {
                "min_value": 0.0,
                "max_value": 300.0,
                "writable": True,
                "confirm": True,
            },
        }

    def test_hello_world_hooks_present(self, hello_world_project: Path):
        """Check .claude/hooks/ directory exists with expected hook files."""
        hooks_dir = hello_world_project / ".claude" / "hooks"
        assert hooks_dir.exists()

        # writes-check, approval, and limits are the write-safety chain
        # among the preset's eleven hooks; they map to osprey_writes_check.py,
        # osprey_approval.py, osprey_limits.py
        assert (hooks_dir / "osprey_writes_check.py").exists()
        assert (hooks_dir / "osprey_approval.py").exists()
        assert (hooks_dir / "osprey_limits.py").exists()
        # The config-drift SessionStart guard ships in every preset so a
        # hand-edited config.yml never silently runs stale settings (#244).
        assert (hooks_dir / "osprey_config_drift.py").exists()
        # memory-guard gates Write to Claude memory files and
        # NotebookEdit to the agent-data artifacts and notebooks trees.
        assert (hooks_dir / "osprey_memory_guard.py").exists()


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
