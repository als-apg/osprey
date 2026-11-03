"""Tests for the osprey eject command.

Tests the list and service subcommands.
"""

import pytest

from osprey.cli.eject_cmd import eject


class TestEjectList:
    """Test osprey eject list."""

    @pytest.mark.parametrize(
        "expected",
        ["Ejectable services", "channel_finder"],
        ids=["heading", "channel-finder-service"],
    )
    def test_list_shows_the_available_components(self, runner, expected):
        """Test that eject list shows available components."""
        result = runner.invoke(eject, ["list"])
        assert result.exit_code == 0
        assert expected in result.output


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
