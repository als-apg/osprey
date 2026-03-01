"""Tests for the osprey eject command.

Tests the list and service subcommands.
"""

import pytest
from click.testing import CliRunner

from osprey.cli.eject_cmd import eject


@pytest.fixture
def runner():
    return CliRunner()


class TestEjectList:
    """Test osprey eject list."""

    def test_list_shows_services(self, runner):
        """Test that eject list shows available components."""
        result = runner.invoke(eject, ["list"])
        assert result.exit_code == 0
        assert "Services:" in result.output

    def test_list_shows_channel_finder_service(self, runner):
        """Test that channel_finder service appears in ejectable list."""
        result = runner.invoke(eject, ["list"])
        assert result.exit_code == 0
        assert "channel_finder" in result.output


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
