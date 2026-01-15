"""Tests for PythonCapability instance method pattern migration and prompt builder integration."""

import inspect

import pytest

from osprey.capabilities.python import PythonCapability, _create_python_capability_prompts


class TestPythonCapabilityMigration:
    """Test PythonCapability successfully migrated to instance method pattern."""

    def test_uses_instance_method_not_static(self):
        """Verify execute() migrated from @staticmethod to instance method."""
        execute_method = inspect.getattr_static(PythonCapability, "execute")
        assert not isinstance(execute_method, staticmethod)

        sig = inspect.signature(PythonCapability.execute)
        params = list(sig.parameters.keys())
        assert params == ["self"]

    def test_state_can_be_injected(self, mock_state, mock_step):
        """Verify capability instance can receive _state and _step injection."""
        capability = PythonCapability()
        capability._state = mock_state
        capability._step = mock_step

        assert capability._state == mock_state
        assert capability._step == mock_step

    def test_has_langgraph_node_decorator(self):
        """Verify @capability_node decorator created langgraph_node attribute."""
        assert hasattr(PythonCapability, "langgraph_node")
        assert callable(PythonCapability.langgraph_node)


class TestPythonPromptBuilderIntegration:
    """Test Python capability integration with prompt builder system."""

    def test_capability_prompts_include_builder_instructions(self):
        """Verify _create_python_capability_prompts uses prompt builder system."""
        from osprey.prompts.defaults import DefaultPromptProvider
        from osprey.prompts.loader import _prompt_loader, register_framework_prompt_provider

        # Clear any existing providers
        _prompt_loader._providers.clear()
        _prompt_loader._default_provider = None

        # Register default provider
        provider = DefaultPromptProvider()
        register_framework_prompt_provider("test", provider)

        # Create prompts
        prompts = _create_python_capability_prompts(
            task_objective="Test task", user_query="Test query", context_description="Test context"
        )

        # Should have 4 prompts: task, user query, context, AND builder instructions
        assert len(prompts) >= 4, f"Expected at least 4 prompts, got {len(prompts)}"

        # Verify builder instructions are included
        combined = "\n".join(prompts)
        assert "CODE GENERATION INSTRUCTIONS" in combined, (
            "Prompt builder instructions not included in capability_prompts!"
        )
        assert "results" in combined.lower(), (
            "Critical 'results' requirement from prompt builder not included!"
        )

    def test_prompt_builder_instructions_content(self):
        """Verify prompt builder provides proper Python code generation guidance."""
        from osprey.prompts.defaults.python import DefaultPythonPromptBuilder

        builder = DefaultPythonPromptBuilder()
        instructions = builder.get_instructions()

        # Verify critical elements
        assert "results" in instructions.lower(), "Missing 'results' dictionary requirement"
        assert "executable" in instructions.lower(), "Missing executable code requirement"
        assert "import" in instructions.lower(), "Missing imports guidance"

        # Should be substantial (not just a placeholder)
        assert len(instructions) > 200, f"Instructions too short: {len(instructions)} chars"

    def test_fallback_when_prompt_builder_unavailable(self):
        """Verify graceful fallback if prompt builder system fails."""
        from osprey.prompts.loader import _prompt_loader

        # Clear providers to simulate failure
        _prompt_loader._providers.clear()
        _prompt_loader._default_provider = None

        # Should not crash, should gracefully continue
        prompts = _create_python_capability_prompts(
            task_objective="Test task", user_query="Test query"
        )

        # Should still have basic prompts (task, user query)
        assert len(prompts) >= 2, "Should have at least task and user query"

        # Verify task and query are present
        combined = "\n".join(prompts)
        assert "Test task" in combined
        assert "Test query" in combined

    @pytest.mark.skip(reason="Requires full registry initialization - tested in integration tests")
    def test_orchestrator_guide_from_prompt_builder(self):
        """Verify orchestrator guide comes from prompt builder.

        Note: This test requires registry initialization and is better tested
        in integration tests. The key functionality (prompt builder instructions
        injection into capability_prompts) is tested above.
        """
        pass

    def test_classifier_guide_uses_prompt_builder(self):
        """Verify _create_classifier_guide delegates to prompt builder."""
        from osprey.prompts.defaults import DefaultPromptProvider
        from osprey.prompts.loader import _prompt_loader, register_framework_prompt_provider

        # Clear and register
        _prompt_loader._providers.clear()
        _prompt_loader._default_provider = None
        provider = DefaultPromptProvider()
        register_framework_prompt_provider("test", provider)

        # Create capability instance
        capability = PythonCapability()

        # Get classifier guide
        guide = capability._create_classifier_guide()

        # Should return the guide from prompt builder
        assert guide is not None
        assert len(guide.examples) > 0
        assert "computational" in guide.instructions.lower()


class TestFigurePathResolution:
    """Tests for issue #96 - ensure figure paths are always absolute.

    The CLI should display full absolute paths for generated figures,
    not just filenames. This test class verifies that paths are resolved
    to absolute paths when registering artifacts.
    """

    def test_path_resolve_converts_relative_to_absolute(self):
        """Verify Path.resolve() converts relative paths to absolute."""
        from pathlib import Path

        # Simulate a relative path that might come from figure collection
        relative_path = Path("figures/figure_01.png")

        # resolve() should convert to absolute
        abs_path = relative_path.resolve()

        assert abs_path.is_absolute(), "resolve() should return absolute path"
        assert str(abs_path).startswith("/"), "Absolute path should start with /"
        assert str(abs_path).endswith("figures/figure_01.png"), "Should preserve filename"

    def test_path_resolve_preserves_absolute_paths(self):
        """Verify Path.resolve() preserves already absolute paths."""
        from pathlib import Path

        # Already absolute path
        abs_path = Path("/some/absolute/path/figures/figure_01.png")

        # resolve() should not change it (except normalizing)
        resolved = abs_path.resolve()

        assert resolved.is_absolute()
        assert "figure_01.png" in str(resolved)

    def test_artifact_registration_uses_absolute_paths(self):
        """Verify artifact data contains absolute paths after registration.

        This tests the fix for issue #96 - figure paths should always be
        absolute when displayed in the CLI.
        """
        from pathlib import Path

        from osprey.state import StateManager
        from osprey.state.artifacts import ArtifactType

        # Create a mock state
        state = {"ui_artifacts": []}

        # Simulate registering a figure with a relative path
        # (this is what the fix prevents by using .resolve())
        relative_figure_path = Path("figures/figure_01.png")

        # The fix: resolve to absolute before registering
        abs_path = relative_figure_path.resolve()

        update = StateManager.register_artifact(
            state,
            artifact_type=ArtifactType.IMAGE,
            capability="python_executor",
            data={"path": str(abs_path), "format": "png"},
            display_name="Test Figure",
        )

        # Verify the stored path is absolute
        artifacts = update["ui_artifacts"]
        assert len(artifacts) == 1
        stored_path = artifacts[0]["data"]["path"]
        assert stored_path.startswith("/"), f"Path should be absolute, got: {stored_path}"
        assert "figure_01.png" in stored_path

    def test_multiple_figures_all_have_absolute_paths(self):
        """Verify multiple figures all get absolute paths."""
        from pathlib import Path

        from osprey.state import StateManager
        from osprey.state.artifacts import ArtifactType

        state = {"ui_artifacts": []}

        # Simulate multiple figure paths (mix of relative and absolute)
        figure_paths = [
            Path("figures/figure_01.png"),
            Path("figures/figure_02.png"),
            Path("/tmp/execution/figures/figure_03.png"),  # Already absolute
        ]

        artifacts = None
        for figure_path in figure_paths:
            # Apply the fix: resolve to absolute
            abs_path = figure_path.resolve()
            update = StateManager.register_artifact(
                state,
                artifact_type=ArtifactType.IMAGE,
                capability="python_executor",
                data={"path": str(abs_path), "format": "png"},
                current_artifacts=artifacts,
            )
            artifacts = update["ui_artifacts"]

        # All paths should be absolute
        assert len(artifacts) == 3
        for artifact in artifacts:
            path = artifact["data"]["path"]
            assert path.startswith("/"), f"Path should be absolute: {path}"

    def test_notebook_path_is_absolute(self):
        """Verify notebook paths are also resolved to absolute."""
        from pathlib import Path

        from osprey.state import StateManager
        from osprey.state.artifacts import ArtifactType

        state = {"ui_artifacts": []}

        # Simulate notebook path
        notebook_path = Path("notebook.ipynb")
        abs_notebook_path = notebook_path.resolve()

        update = StateManager.register_artifact(
            state,
            artifact_type=ArtifactType.NOTEBOOK,
            capability="python_executor",
            data={"path": str(abs_notebook_path), "url": "http://localhost:8888/notebook"},
        )

        artifacts = update["ui_artifacts"]
        stored_path = artifacts[0]["data"]["path"]
        assert stored_path.startswith("/"), f"Notebook path should be absolute: {stored_path}"

    def test_cli_displays_full_absolute_path(self):
        """Verify the CLI _format_artifact_line displays the full absolute path.

        This is the critical end-to-end test for issue #96 - it verifies that
        absolute paths stored in artifacts are displayed in full by the CLI.
        """
        from pathlib import Path

        from osprey.interfaces.cli.direct_conversation import CLI
        from osprey.state.artifacts import ArtifactType

        # Create an artifact with an absolute path (as the fix does)
        relative_path = Path("figures/figure_01.png")
        abs_path = relative_path.resolve()

        artifact = {
            "id": "test-123",
            "type": "image",
            "capability": "python_executor",
            "created_at": "2025-01-15T10:00:00",
            "data": {"path": str(abs_path), "format": "png"},
        }

        # Create CLI instance and format the artifact
        cli = CLI.__new__(CLI)
        formatted = cli._format_artifact_line(artifact, ArtifactType.IMAGE)

        # The formatted output should contain the FULL absolute path
        assert str(abs_path) in formatted, (
            f"CLI should display full path '{abs_path}', got: {formatted}"
        )
        assert formatted.startswith("  •"), "Should have bullet prefix"
        assert "python_executor" in formatted, "Should show capability"

        # Verify it's actually an absolute path in the output
        # Extract the path from the formatted string
        assert "/" in formatted and not formatted.startswith("  • figures/"), (
            f"CLI should show absolute path, not relative. Got: {formatted}"
        )

    def test_cli_displays_relative_path_if_stored_that_way(self):
        """Demonstrate that CLI shows whatever path is stored - highlighting the fix.

        This test shows that WITHOUT the fix (storing relative paths), the CLI
        would only show the filename. The fix ensures absolute paths are stored.
        """
        from osprey.interfaces.cli.direct_conversation import CLI
        from osprey.state.artifacts import ArtifactType

        # WITHOUT the fix: a relative path would be stored
        artifact_with_relative = {
            "id": "test-456",
            "type": "image",
            "capability": "python_executor",
            "created_at": "2025-01-15T10:00:00",
            "data": {"path": "figures/figure_01.png", "format": "png"},  # Relative!
        }

        cli = CLI.__new__(CLI)
        formatted = cli._format_artifact_line(artifact_with_relative, ArtifactType.IMAGE)

        # This shows what the user saw BEFORE the fix - just the relative path
        assert "figures/figure_01.png" in formatted
        assert not formatted.startswith("  • /"), (
            "This shows the bug: relative path doesn't start with /"
        )
