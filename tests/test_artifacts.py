"""Tests for the unified artifact system.

This module tests the artifact abstraction layer including:
- ArtifactType enum and artifact creation
- StateManager.register_artifact() unified registration
- Legacy method delegation (register_figure/command/notebook)
- Backward compatibility with ui_captured_* fields
"""

import uuid

from osprey.state import AgentState, StateManager
from osprey.state.artifacts import (
    ArtifactType,
    create_artifact,
    get_artifact_type_icon,
    validate_artifact_data,
)


class TestArtifactType:
    """Tests for the ArtifactType enum."""

    def test_artifact_type_values(self):
        """ArtifactType should have expected string values."""
        assert ArtifactType.IMAGE.value == "image"
        assert ArtifactType.NOTEBOOK.value == "notebook"
        assert ArtifactType.COMMAND.value == "command"
        assert ArtifactType.HTML.value == "html"
        assert ArtifactType.FILE.value == "file"

    def test_artifact_type_is_string_enum(self):
        """ArtifactType should be usable as string (StrEnum returns value directly)."""
        assert str(ArtifactType.IMAGE) == "image"
        assert ArtifactType.IMAGE.value == "image"

    def test_artifact_type_from_string(self):
        """ArtifactType should be constructible from string value."""
        assert ArtifactType("image") == ArtifactType.IMAGE
        assert ArtifactType("notebook") == ArtifactType.NOTEBOOK
        assert ArtifactType("command") == ArtifactType.COMMAND


class TestCreateArtifact:
    """Tests for the create_artifact factory function."""

    def test_creates_artifact_with_required_fields(self):
        """create_artifact should populate all required fields."""
        artifact = create_artifact(
            artifact_type=ArtifactType.IMAGE,
            capability="python_executor",
            data={"path": "/path/to/plot.png", "format": "png"},
        )

        assert "id" in artifact
        assert artifact["type"] == "image"
        assert artifact["capability"] == "python_executor"
        assert "created_at" in artifact
        assert artifact["data"] == {"path": "/path/to/plot.png", "format": "png"}

    def test_creates_artifact_with_unique_id(self):
        """create_artifact should generate unique UUIDs."""
        artifact1 = create_artifact(ArtifactType.IMAGE, "test", {"path": "/a.png"})
        artifact2 = create_artifact(ArtifactType.IMAGE, "test", {"path": "/b.png"})

        assert artifact1["id"] != artifact2["id"]
        # Verify they are valid UUIDs
        uuid.UUID(artifact1["id"])
        uuid.UUID(artifact2["id"])

    def test_creates_artifact_with_optional_display_name(self):
        """create_artifact should include display_name when provided."""
        artifact = create_artifact(
            ArtifactType.IMAGE, "test", {"path": "/plot.png"}, display_name="Analysis Plot"
        )

        assert artifact["display_name"] == "Analysis Plot"

    def test_creates_artifact_without_display_name(self):
        """create_artifact should omit display_name when not provided."""
        artifact = create_artifact(ArtifactType.IMAGE, "test", {"path": "/plot.png"})

        assert "display_name" not in artifact

    def test_creates_artifact_with_metadata(self):
        """create_artifact should include metadata when provided."""
        metadata = {"execution_folder": "/tmp/exec", "notebook_link": "http://localhost:8888"}
        artifact = create_artifact(
            ArtifactType.IMAGE, "test", {"path": "/plot.png"}, metadata=metadata
        )

        assert artifact["metadata"] == metadata

    def test_creates_artifact_without_metadata(self):
        """create_artifact should omit metadata when not provided."""
        artifact = create_artifact(ArtifactType.IMAGE, "test", {"path": "/plot.png"})

        assert "metadata" not in artifact

    def test_creates_notebook_artifact(self):
        """create_artifact should work for NOTEBOOK type."""
        artifact = create_artifact(
            ArtifactType.NOTEBOOK,
            "python_executor",
            {"path": "/path/to/notebook.ipynb", "url": "http://jupyter/notebook"},
            display_name="Execution Notebook",
        )

        assert artifact["type"] == "notebook"
        assert artifact["data"]["path"] == "/path/to/notebook.ipynb"
        assert artifact["data"]["url"] == "http://jupyter/notebook"

    def test_creates_command_artifact(self):
        """create_artifact should work for COMMAND type."""
        artifact = create_artifact(
            ArtifactType.COMMAND,
            "dashboard_builder",
            {"uri": "http://localhost:8080/dashboard", "command_type": "web_app"},
            display_name="Interactive Dashboard",
        )

        assert artifact["type"] == "command"
        assert artifact["data"]["uri"] == "http://localhost:8080/dashboard"
        assert artifact["data"]["command_type"] == "web_app"

    def test_creates_html_artifact(self):
        """create_artifact should work for HTML type."""
        artifact = create_artifact(
            ArtifactType.HTML,
            "visualization",
            {"path": "/path/to/dashboard.html", "framework": "bokeh"},
        )

        assert artifact["type"] == "html"
        assert artifact["data"]["framework"] == "bokeh"

    def test_creates_file_artifact(self):
        """create_artifact should work for FILE type."""
        artifact = create_artifact(
            ArtifactType.FILE,
            "data_export",
            {"path": "/path/to/data.csv", "mime_type": "text/csv", "size_bytes": 1024},
        )

        assert artifact["type"] == "file"
        assert artifact["data"]["mime_type"] == "text/csv"


class TestGetArtifactTypeIcon:
    """Tests for the get_artifact_type_icon helper function."""

    def test_returns_correct_icons(self):
        """get_artifact_type_icon should return correct icons for each type."""
        assert get_artifact_type_icon(ArtifactType.IMAGE) == "🖼"
        assert get_artifact_type_icon(ArtifactType.NOTEBOOK) == "📓"
        assert get_artifact_type_icon(ArtifactType.COMMAND) == "🔗"
        assert get_artifact_type_icon(ArtifactType.HTML) == "🌐"
        assert get_artifact_type_icon(ArtifactType.FILE) == "📄"

    def test_accepts_string_type(self):
        """get_artifact_type_icon should accept string type values."""
        assert get_artifact_type_icon("image") == "🖼"
        assert get_artifact_type_icon("notebook") == "📓"
        assert get_artifact_type_icon("command") == "🔗"


class TestValidateArtifactData:
    """Tests for the validate_artifact_data helper function."""

    def test_validates_image_data(self):
        """validate_artifact_data should require path for IMAGE."""
        assert validate_artifact_data(ArtifactType.IMAGE, {"path": "/plot.png"}) is True
        assert validate_artifact_data(ArtifactType.IMAGE, {"url": "http://example.com"}) is False
        assert validate_artifact_data(ArtifactType.IMAGE, {}) is False

    def test_validates_command_data(self):
        """validate_artifact_data should require uri for COMMAND."""
        assert validate_artifact_data(ArtifactType.COMMAND, {"uri": "http://localhost"}) is True
        assert validate_artifact_data(ArtifactType.COMMAND, {"path": "/file"}) is False
        assert validate_artifact_data(ArtifactType.COMMAND, {}) is False

    def test_validates_file_data(self):
        """validate_artifact_data should require path for FILE."""
        assert validate_artifact_data(ArtifactType.FILE, {"path": "/data.csv"}) is True
        assert validate_artifact_data(ArtifactType.FILE, {}) is False

    def test_validates_notebook_data_flexible(self):
        """validate_artifact_data should be flexible for NOTEBOOK."""
        # Notebook can have path OR url
        assert validate_artifact_data(ArtifactType.NOTEBOOK, {"path": "/nb.ipynb"}) is True
        assert validate_artifact_data(ArtifactType.NOTEBOOK, {"url": "http://jupyter"}) is True
        assert validate_artifact_data(ArtifactType.NOTEBOOK, {}) is True  # Flexible

    def test_validates_html_data_flexible(self):
        """validate_artifact_data should be flexible for HTML."""
        # HTML can have path OR url
        assert validate_artifact_data(ArtifactType.HTML, {"path": "/dash.html"}) is True
        assert validate_artifact_data(ArtifactType.HTML, {"url": "http://dash"}) is True
        assert validate_artifact_data(ArtifactType.HTML, {}) is True  # Flexible


class TestAgentStateArtifacts:
    """Tests for ui_artifacts field in AgentState."""

    def test_agent_state_has_ui_artifacts_field(self):
        """AgentState should have ui_artifacts as a valid field."""
        assert "ui_artifacts" in AgentState.__annotations__

    def test_create_fresh_state_initializes_ui_artifacts(self):
        """StateManager.create_fresh_state should initialize ui_artifacts to empty list."""
        state = StateManager.create_fresh_state("Hello")

        assert "ui_artifacts" in state
        assert state["ui_artifacts"] == []

    def test_create_fresh_state_resets_ui_artifacts(self):
        """create_fresh_state should reset ui_artifacts each turn."""
        prev_state = {
            "ui_artifacts": [{"id": "123", "type": "image"}],
            "capability_context_data": {},
        }

        fresh_state = StateManager.create_fresh_state("Hello", current_state=prev_state)

        # ui_artifacts should be reset (not preserved)
        assert fresh_state["ui_artifacts"] == []


class TestStateManagerRegisterArtifact:
    """Tests for StateManager.register_artifact() unified registration."""

    def test_register_artifact_creates_valid_artifact(self):
        """register_artifact should create properly structured artifact."""
        state = StateManager.create_fresh_state("Test")

        update = StateManager.register_artifact(
            state=state,
            artifact_type=ArtifactType.IMAGE,
            capability="python_executor",
            data={"path": "/path/to/plot.png", "format": "png"},
            display_name="Analysis Plot",
        )

        assert "ui_artifacts" in update
        assert len(update["ui_artifacts"]) == 1

        artifact = update["ui_artifacts"][0]
        assert artifact["type"] == "image"
        assert artifact["capability"] == "python_executor"
        assert artifact["data"]["path"] == "/path/to/plot.png"
        assert artifact["display_name"] == "Analysis Plot"
        assert "id" in artifact
        assert "created_at" in artifact

    def test_register_artifact_accumulates(self):
        """register_artifact should accumulate multiple artifacts."""
        state = StateManager.create_fresh_state("Test")

        # Register first artifact
        update1 = StateManager.register_artifact(
            state, ArtifactType.IMAGE, "test", {"path": "/a.png"}
        )

        # Register second using accumulation pattern
        update2 = StateManager.register_artifact(
            state,
            ArtifactType.IMAGE,
            "test",
            {"path": "/b.png"},
            current_artifacts=update1["ui_artifacts"],
        )

        assert len(update2["ui_artifacts"]) == 2
        assert update2["ui_artifacts"][0]["data"]["path"] == "/a.png"
        assert update2["ui_artifacts"][1]["data"]["path"] == "/b.png"

    def test_register_artifact_with_metadata(self):
        """register_artifact should include metadata."""
        state = StateManager.create_fresh_state("Test")
        metadata = {"execution_folder": "/tmp", "step": 1}

        update = StateManager.register_artifact(
            state, ArtifactType.IMAGE, "test", {"path": "/plot.png"}, metadata=metadata
        )

        artifact = update["ui_artifacts"][0]
        assert artifact["metadata"] == metadata


class TestStateManagerRegisterFigureLegacy:
    """Tests for legacy register_figure() method."""

    def test_register_figure_creates_artifact(self):
        """register_figure should create artifact in ui_artifacts."""
        state = StateManager.create_fresh_state("Test")

        update = StateManager.register_figure(
            state=state,
            capability="python_executor",
            figure_path="/path/to/plot.png",
            display_name="Test Plot",
        )

        # Should have both unified and legacy fields
        assert "ui_artifacts" in update
        assert "ui_captured_figures" in update

        # Check unified artifact
        artifact = update["ui_artifacts"][0]
        assert artifact["type"] == "image"
        assert artifact["data"]["path"] == "/path/to/plot.png"
        assert artifact["data"]["format"] == "png"

    def test_register_figure_maintains_legacy_format(self):
        """register_figure should maintain legacy ui_captured_figures format."""
        state = StateManager.create_fresh_state("Test")

        update = StateManager.register_figure(
            state=state,
            capability="python_executor",
            figure_path="/path/to/plot.png",
            display_name="Test Plot",
            metadata={"notebook_link": "http://jupyter"},
        )

        # Check legacy format
        figure = update["ui_captured_figures"][0]
        assert figure["capability"] == "python_executor"
        assert figure["figure_path"] == "/path/to/plot.png"
        assert figure["display_name"] == "Test Plot"
        assert figure["metadata"]["notebook_link"] == "http://jupyter"
        assert "created_at" in figure

    def test_register_figure_detects_format(self):
        """register_figure should detect image format from extension."""
        state = StateManager.create_fresh_state("Test")

        # Test PNG
        update1 = StateManager.register_figure(state, "test", "/plot.png")
        assert update1["ui_artifacts"][0]["data"]["format"] == "png"

        # Test JPG
        update2 = StateManager.register_figure(state, "test", "/photo.jpg")
        assert update2["ui_artifacts"][0]["data"]["format"] == "jpg"

        # Test SVG
        update3 = StateManager.register_figure(state, "test", "/vector.svg")
        assert update3["ui_artifacts"][0]["data"]["format"] == "svg"


class TestStateManagerRegisterCommandLegacy:
    """Tests for legacy register_command() method."""

    def test_register_command_creates_artifact(self):
        """register_command should create artifact in ui_artifacts."""
        state = StateManager.create_fresh_state("Test")

        update = StateManager.register_command(
            state=state,
            capability="dashboard_builder",
            launch_uri="http://localhost:8080",
            display_name="Dashboard",
            command_type="web_app",
        )

        # Should have both unified and legacy fields
        assert "ui_artifacts" in update
        assert "ui_launchable_commands" in update

        # Check unified artifact
        artifact = update["ui_artifacts"][0]
        assert artifact["type"] == "command"
        assert artifact["data"]["uri"] == "http://localhost:8080"
        assert artifact["data"]["command_type"] == "web_app"

    def test_register_command_maintains_legacy_format(self):
        """register_command should maintain legacy ui_launchable_commands format."""
        state = StateManager.create_fresh_state("Test")

        update = StateManager.register_command(
            state=state,
            capability="dashboard_builder",
            launch_uri="http://localhost:8080",
            display_name="Dashboard",
            command_type="web_app",
        )

        # Check legacy format
        command = update["ui_launchable_commands"][0]
        assert command["capability"] == "dashboard_builder"
        assert command["launch_uri"] == "http://localhost:8080"
        assert command["display_name"] == "Dashboard"
        assert command["command_type"] == "web_app"
        assert "created_at" in command


class TestStateManagerRegisterNotebookLegacy:
    """Tests for legacy register_notebook() method."""

    def test_register_notebook_creates_artifact(self):
        """register_notebook should create artifact in ui_artifacts."""
        state = StateManager.create_fresh_state("Test")

        update = StateManager.register_notebook(
            state=state,
            capability="python_executor",
            notebook_path="/path/to/notebook.ipynb",
            notebook_link="http://jupyter/notebook",
            display_name="Execution Notebook",
        )

        # Should have both unified and legacy fields
        assert "ui_artifacts" in update
        assert "ui_captured_notebooks" in update

        # Check unified artifact
        artifact = update["ui_artifacts"][0]
        assert artifact["type"] == "notebook"
        assert artifact["data"]["path"] == "/path/to/notebook.ipynb"
        assert artifact["data"]["url"] == "http://jupyter/notebook"

    def test_register_notebook_maintains_legacy_format(self):
        """register_notebook should maintain legacy ui_captured_notebooks format."""
        state = StateManager.create_fresh_state("Test")

        update = StateManager.register_notebook(
            state=state,
            capability="python_executor",
            notebook_path="/path/to/notebook.ipynb",
            notebook_link="http://jupyter/notebook",
        )

        # Legacy format is just the link string
        assert update["ui_captured_notebooks"][0] == "http://jupyter/notebook"


class TestArtifactAccumulationPatterns:
    """Tests for proper artifact accumulation patterns."""

    def test_multiple_artifacts_in_single_node(self):
        """Should support registering multiple artifacts in a single node."""
        state = StateManager.create_fresh_state("Test")

        # Pattern: accumulate artifacts within single node
        accumulating = None
        figure_paths = ["/a.png", "/b.png", "/c.png"]

        for path in figure_paths:
            update = StateManager.register_artifact(
                state,
                ArtifactType.IMAGE,
                "python_executor",
                {"path": path, "format": "png"},
                current_artifacts=accumulating,
            )
            accumulating = update["ui_artifacts"]

        # Final update should contain all artifacts
        assert len(update["ui_artifacts"]) == 3

    def test_mixed_artifact_types(self):
        """Should support different artifact types in same execution."""
        state = StateManager.create_fresh_state("Test")

        # Register image
        update1 = StateManager.register_artifact(
            state, ArtifactType.IMAGE, "test", {"path": "/plot.png"}
        )

        # Register notebook
        update2 = StateManager.register_artifact(
            state,
            ArtifactType.NOTEBOOK,
            "test",
            {"path": "/nb.ipynb", "url": "http://jupyter"},
            current_artifacts=update1["ui_artifacts"],
        )

        # Register command
        update3 = StateManager.register_artifact(
            state,
            ArtifactType.COMMAND,
            "test",
            {"uri": "http://dashboard"},
            current_artifacts=update2["ui_artifacts"],
        )

        assert len(update3["ui_artifacts"]) == 3
        types = [a["type"] for a in update3["ui_artifacts"]]
        assert types == ["image", "notebook", "command"]


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing code."""

    def test_legacy_figure_registration_still_works(self):
        """Existing code using register_figure should continue to work."""
        state = StateManager.create_fresh_state("Test")

        # This is the existing pattern in capabilities
        update = StateManager.register_figure(
            state,
            capability="python_executor",
            figure_path="/path/to/plot.png",
            display_name="Test",
            metadata={"execution_folder": "/tmp"},
        )

        # Old code expects ui_captured_figures
        assert "ui_captured_figures" in update
        assert len(update["ui_captured_figures"]) == 1

    def test_legacy_command_registration_still_works(self):
        """Existing code using register_command should continue to work."""
        state = StateManager.create_fresh_state("Test")

        update = StateManager.register_command(
            state,
            capability="dashboard",
            launch_uri="http://localhost:8080",
            display_name="Dashboard",
        )

        # Old code expects ui_launchable_commands
        assert "ui_launchable_commands" in update
        assert len(update["ui_launchable_commands"]) == 1

    def test_legacy_notebook_registration_still_works(self):
        """Existing code using register_notebook should continue to work."""
        state = StateManager.create_fresh_state("Test")

        update = StateManager.register_notebook(
            state,
            capability="python_executor",
            notebook_path="/nb.ipynb",
            notebook_link="http://jupyter",
        )

        # Old code expects ui_captured_notebooks
        assert "ui_captured_notebooks" in update
        assert len(update["ui_captured_notebooks"]) == 1


class TestPopulateLegacyFieldsFromArtifacts:
    """Tests for populate_legacy_fields_from_artifacts() finalization helper."""

    def test_populates_figures_from_image_artifacts(self):
        """Should convert IMAGE artifacts to ui_captured_figures format."""
        from osprey.state import populate_legacy_fields_from_artifacts

        artifacts = [
            {
                "id": "1",
                "type": "image",
                "capability": "python_executor",
                "created_at": "2024-01-15T10:00:00",
                "data": {"path": "/path/to/plot.png", "format": "png"},
                "display_name": "Analysis Plot",
                "metadata": {"folder": "/tmp"},
            }
        ]

        legacy = populate_legacy_fields_from_artifacts(artifacts)

        assert len(legacy["ui_captured_figures"]) == 1
        figure = legacy["ui_captured_figures"][0]
        assert figure["capability"] == "python_executor"
        assert figure["figure_path"] == "/path/to/plot.png"
        assert figure["created_at"] == "2024-01-15T10:00:00"
        assert figure["display_name"] == "Analysis Plot"
        assert figure["metadata"] == {"folder": "/tmp"}

    def test_populates_commands_from_command_artifacts(self):
        """Should convert COMMAND artifacts to ui_launchable_commands format."""
        from osprey.state import populate_legacy_fields_from_artifacts

        artifacts = [
            {
                "id": "2",
                "type": "command",
                "capability": "dashboard",
                "created_at": "2024-01-15T10:00:00",
                "data": {"uri": "http://localhost:8080", "command_type": "web_app"},
                "display_name": "Dashboard",
            }
        ]

        legacy = populate_legacy_fields_from_artifacts(artifacts)

        assert len(legacy["ui_launchable_commands"]) == 1
        cmd = legacy["ui_launchable_commands"][0]
        assert cmd["capability"] == "dashboard"
        assert cmd["uri"] == "http://localhost:8080"
        assert cmd["display_name"] == "Dashboard"

    def test_populates_notebooks_from_notebook_artifacts(self):
        """Should convert NOTEBOOK artifacts to ui_captured_notebooks format (URL list)."""
        from osprey.state import populate_legacy_fields_from_artifacts

        artifacts = [
            {
                "id": "3",
                "type": "notebook",
                "capability": "python_executor",
                "created_at": "2024-01-15T10:00:00",
                "data": {"path": "/nb.ipynb", "url": "http://jupyter/notebook"},
            }
        ]

        legacy = populate_legacy_fields_from_artifacts(artifacts)

        # Legacy format is just URL strings
        assert len(legacy["ui_captured_notebooks"]) == 1
        assert legacy["ui_captured_notebooks"][0] == "http://jupyter/notebook"

    def test_handles_mixed_artifact_types(self):
        """Should properly separate mixed artifact types into legacy fields."""
        from osprey.state import populate_legacy_fields_from_artifacts

        artifacts = [
            {
                "id": "1",
                "type": "image",
                "capability": "test",
                "created_at": "",
                "data": {"path": "/a.png"},
            },
            {
                "id": "2",
                "type": "notebook",
                "capability": "test",
                "created_at": "",
                "data": {"url": "http://nb1"},
            },
            {
                "id": "3",
                "type": "image",
                "capability": "test",
                "created_at": "",
                "data": {"path": "/b.png"},
            },
            {
                "id": "4",
                "type": "command",
                "capability": "test",
                "created_at": "",
                "data": {"uri": "http://cmd"},
            },
            {
                "id": "5",
                "type": "notebook",
                "capability": "test",
                "created_at": "",
                "data": {"url": "http://nb2"},
            },
        ]

        legacy = populate_legacy_fields_from_artifacts(artifacts)

        assert len(legacy["ui_captured_figures"]) == 2
        assert len(legacy["ui_captured_notebooks"]) == 2
        assert len(legacy["ui_launchable_commands"]) == 1

    def test_handles_empty_artifacts_list(self):
        """Should return empty legacy fields for empty artifacts list."""
        from osprey.state import populate_legacy_fields_from_artifacts

        legacy = populate_legacy_fields_from_artifacts([])

        assert legacy["ui_captured_figures"] == []
        assert legacy["ui_launchable_commands"] == []
        assert legacy["ui_captured_notebooks"] == []

    def test_handles_artifacts_without_optional_fields(self):
        """Should handle artifacts missing optional display_name and metadata."""
        from osprey.state import populate_legacy_fields_from_artifacts

        artifacts = [
            {
                "id": "1",
                "type": "image",
                "capability": "test",
                "created_at": "2024-01-15T10:00:00",
                "data": {"path": "/plot.png"},
                # No display_name or metadata
            }
        ]

        legacy = populate_legacy_fields_from_artifacts(artifacts)

        figure = legacy["ui_captured_figures"][0]
        assert "display_name" not in figure
        assert "metadata" not in figure
        assert figure["figure_path"] == "/plot.png"
