"""Tests for the osprey eject command.

Tests the list, capability, and service subcommands.
"""

import pytest
from click.testing import CliRunner

from osprey.cli.eject_cmd import eject


@pytest.fixture
def runner():
    return CliRunner()


class TestEjectList:
    """Test osprey eject list."""

    def test_list_shows_capabilities_and_services(self, runner):
        """Test that eject list shows available components."""
        result = runner.invoke(eject, ["list"])
        assert result.exit_code == 0
        assert "Capabilities:" in result.output
        assert "Services:" in result.output

    def test_list_shows_channel_finding(self, runner):
        """Test that channel_finding appears in ejectable list."""
        result = runner.invoke(eject, ["list"])
        assert result.exit_code == 0
        assert "channel_finding" in result.output

    def test_list_shows_channel_finder_service(self, runner):
        """Test that channel_finder service appears in ejectable list."""
        result = runner.invoke(eject, ["list"])
        assert result.exit_code == 0
        assert "channel_finder" in result.output


class TestEjectCapability:
    """Test osprey eject capability."""

    def test_unknown_capability_fails(self, runner):
        """Test that unknown capability name shows error."""
        result = runner.invoke(eject, ["capability", "nonexistent_capability"])
        assert result.exit_code != 0
        assert "Unknown capability" in result.output

    def test_eject_capability_to_output_path(self, runner, tmp_path):
        """Test ejecting a capability to a specific output path."""
        output_file = tmp_path / "channel_finding.py"
        result = runner.invoke(
            eject, ["capability", "channel_finding", "--output", str(output_file)]
        )
        assert result.exit_code == 0
        assert "Ejected capability" in result.output
        assert output_file.exists()
        # Verify it's actual Python source
        content = output_file.read_text()
        assert "ChannelFindingCapability" in content

    def test_eject_capability_with_include_tests(self, runner, tmp_path, monkeypatch):
        """Test ejecting a capability with --include-tests copies test files."""
        output_file = tmp_path / "channel_finding.py"
        # Create tests/capabilities/ in tmp_path so the command can copy tests there
        test_dest = tmp_path / "tests" / "capabilities"
        test_dest.mkdir(parents=True)
        # Change CWD to tmp_path so tests copy to our temp directory
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(
            eject,
            [
                "capability",
                "channel_finding",
                "--output",
                str(output_file),
                "--include-tests",
            ],
        )
        assert result.exit_code == 0
        assert "Ejected capability" in result.output
        # The --include-tests flag should trigger test copying logic
        assert "Copied test:" in result.output
        # Verify test file was actually copied
        copied_tests = list(test_dest.glob("test_channel_finding*.py"))
        assert len(copied_tests) > 0


class TestEjectService:
    """Test osprey eject service."""

    def test_unknown_service_fails(self, runner):
        """Test that unknown service name shows error."""
        result = runner.invoke(eject, ["service", "nonexistent_service"])
        assert result.exit_code != 0
        assert "Unknown service" in result.output

    def test_eject_service_to_output_path(self, runner, tmp_path):
        """Test ejecting a service to a specific output directory."""
        output_dir = tmp_path / "channel_finder"
        result = runner.invoke(eject, ["service", "channel_finder", "--output", str(output_dir)])
        assert result.exit_code == 0
        assert "Ejected service" in result.output
        assert output_dir.exists()
        # Verify it copied Python files
        py_files = list(output_dir.rglob("*.py"))
        assert len(py_files) > 0
