"""Tests for registry reset functionality in interactive menu.

This test module verifies that the registry is properly reset when switching
between projects in the interactive menu, preventing capability contamination
between different project types.

Related to GitHub Issue #29: Demo agents swap capabilities

The fix adds reset_registry() calls in handle_chat_action() to ensure
the global registry singleton is cleared when switching between projects.
"""

import os
from unittest.mock import patch

import pytest


class TestRegistryResetInInteractiveMenu:
    """Test that handle_chat_action properly resets registry before starting chat."""

    @patch("osprey.cli.interactive_menu.asyncio.run")
    @patch("osprey.registry.reset_registry")
    def test_handle_chat_action_resets_registry_with_project_path(
        self, mock_reset, mock_asyncio_run, tmp_path
    ):
        """
        Test that handle_chat_action calls reset_registry when given a project path.

        This is the critical fix for GitHub Issue #29 - without this reset,
        capabilities from the first project leak into subsequent projects.
        """
        from osprey.cli.interactive_menu import handle_chat_action

        # Create a minimal test directory with config.yml
        test_project = tmp_path / "test-project"
        test_project.mkdir()
        (test_project / "config.yml").write_text("# minimal config\n")

        # Mock asyncio.run to prevent actual chat from running
        mock_asyncio_run.return_value = None

        # Call handle_chat_action with project path
        original_cwd = os.getcwd()
        try:
            handle_chat_action(project_path=test_project)

            # Verify reset_registry was called - this is the fix!
            mock_reset.assert_called_once()
        finally:
            # Restore original directory
            if os.getcwd() != str(original_cwd):
                os.chdir(original_cwd)

    @patch("osprey.cli.interactive_menu.asyncio.run")
    @patch("osprey.registry.reset_registry")
    def test_handle_chat_action_resets_registry_default_path(
        self, mock_reset, mock_asyncio_run, tmp_path
    ):
        """
        Test that handle_chat_action resets registry even when using default path.

        This ensures the fix works in both code paths:
        - When called with explicit project_path
        - When called without arguments (uses current directory)
        """
        from osprey.cli.interactive_menu import handle_chat_action

        # Create a minimal test directory with config.yml
        test_project = tmp_path / "test-project"
        test_project.mkdir()
        (test_project / "config.yml").write_text("# minimal config\n")

        # Mock asyncio.run to prevent actual chat from running
        mock_asyncio_run.return_value = None

        # Change to project directory and call without explicit path
        original_cwd = os.getcwd()
        try:
            os.chdir(test_project)
            handle_chat_action()  # No project_path argument

            # Verify reset_registry was called in this code path too
            mock_reset.assert_called_once()
        finally:
            # Restore original directory
            os.chdir(original_cwd)

    @patch("osprey.registry.reset_registry")
    def test_reset_called_before_chat_not_after(self, mock_reset, tmp_path):
        """
        Verify that reset_registry is called BEFORE initializing the chat,
        not after. This timing is critical to prevent capability leakage.
        """
        from osprey.cli.interactive_menu import handle_chat_action

        # Create test project
        test_project = tmp_path / "test-project"
        test_project.mkdir()
        (test_project / "config.yml").write_text("# minimal config\n")

        # Track call order
        call_order = []

        def track_reset():
            call_order.append("reset")

        def track_run_cli(*args, **kwargs):
            call_order.append("run_cli")

        mock_reset.side_effect = track_reset

        original_cwd = os.getcwd()
        try:
            with patch("osprey.cli.interactive_menu.asyncio.run", side_effect=track_run_cli):
                handle_chat_action(project_path=test_project)

            # Verify reset was called before run_cli
            assert call_order == [
                "reset",
                "run_cli",
            ], "reset_registry must be called BEFORE run_cli"
        finally:
            if os.getcwd() != str(original_cwd):
                os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
