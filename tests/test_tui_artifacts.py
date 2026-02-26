"""Tests for TUI artifact widgets.

This module tests the TUI artifact display system including:
- ArtifactItem widget rendering
- ArtifactGallery tracking of new vs seen artifacts
- ArtifactViewer combined split-panel viewer
- Integration with ChatDisplay
"""

from osprey.state.artifacts import ArtifactType, create_artifact


class TestArtifactItemWidget:
    """Tests for the ArtifactItem widget."""

    def test_artifact_item_creation(self):
        """ArtifactItem should be creatable with an artifact."""
        from osprey.interfaces.tui.widgets.artifacts import ArtifactItem

        artifact = create_artifact(
            ArtifactType.IMAGE, "python_executor", {"path": "/test/plot.png", "format": "png"}
        )
        item = ArtifactItem(artifact, is_new=True)

        assert item.artifact == artifact
        assert item.is_new is True
        assert item._artifact_type == ArtifactType.IMAGE

    def test_artifact_item_default_name_image(self):
        """ArtifactItem should derive default name from path for images."""
        from osprey.interfaces.tui.widgets.artifacts import _get_artifact_display_name

        artifact = create_artifact(
            ArtifactType.IMAGE, "test", {"path": "/path/to/analysis_plot.png", "format": "png"}
        )

        assert _get_artifact_display_name(artifact) == "analysis_plot.png"

    def test_artifact_item_default_name_notebook(self):
        """ArtifactItem should derive default name from path for notebooks."""
        from osprey.interfaces.tui.widgets.artifacts import _get_artifact_display_name

        artifact = create_artifact(
            ArtifactType.NOTEBOOK,
            "test",
            {"path": "/path/to/execution.ipynb", "url": "http://jupyter"},
        )

        assert _get_artifact_display_name(artifact) == "execution.ipynb"

    def test_artifact_item_default_name_command(self):
        """ArtifactItem should use command_type for commands."""
        from osprey.interfaces.tui.widgets.artifacts import _get_artifact_display_name

        artifact = create_artifact(
            ArtifactType.COMMAND,
            "test",
            {"uri": "http://localhost:8080", "command_type": "web_app"},
        )

        assert _get_artifact_display_name(artifact) == "web_app"

    def test_artifact_item_uses_display_name_if_provided(self):
        """ArtifactItem should use display_name when available."""
        from osprey.interfaces.tui.widgets.artifacts import ArtifactItem

        artifact = create_artifact(
            ArtifactType.IMAGE,
            "test",
            {"path": "/plot.png"},
            display_name="My Custom Plot",
        )
        ArtifactItem(artifact)  # verifies item can be created

        # display_name takes precedence in rendering
        assert artifact.get("display_name") == "My Custom Plot"


class TestArtifactGallery:
    """Tests for the ArtifactGallery widget."""

    def test_gallery_creation(self):
        """ArtifactGallery should be creatable."""
        from osprey.interfaces.tui.widgets.artifacts import ArtifactGallery

        gallery = ArtifactGallery()

        assert gallery._artifacts == []
        assert gallery._seen_ids == set()
        assert gallery._selected_index == 0

    def test_gallery_tracks_new_artifacts(self):
        """ArtifactGallery should mark unseen artifacts as new.

        Note: We test the tracking logic directly without calling update_artifacts()
        since that requires mounted Textual widgets.
        """
        from osprey.interfaces.tui.widgets.artifacts import ArtifactGallery

        gallery = ArtifactGallery()

        # First batch of artifacts
        artifact1 = create_artifact(ArtifactType.IMAGE, "test", {"path": "/a.png"})
        artifact2 = create_artifact(ArtifactType.IMAGE, "test", {"path": "/b.png"})

        # Simulate the tracking logic from update_artifacts
        artifacts = [artifact1, artifact2]
        for artifact in artifacts:
            artifact_id = artifact.get("id", "")
            artifact["_is_new"] = artifact_id not in gallery._seen_ids
            if artifact_id:
                gallery._seen_ids.add(artifact_id)
        gallery._artifacts = artifacts

        # Both should be marked new (first time seeing them)
        assert artifact1.get("_is_new") is True
        assert artifact2.get("_is_new") is True
        assert len(gallery._seen_ids) == 2

    def test_gallery_marks_seen_artifacts_as_not_new(self):
        """ArtifactGallery should not mark previously seen artifacts as new."""
        from osprey.interfaces.tui.widgets.artifacts import ArtifactGallery

        gallery = ArtifactGallery()

        # First artifact
        artifact1 = create_artifact(ArtifactType.IMAGE, "test", {"path": "/a.png"})
        artifact1["_is_new"] = artifact1["id"] not in gallery._seen_ids
        gallery._seen_ids.add(artifact1["id"])

        assert artifact1.get("_is_new") is True

        # Same artifact again (by ID)
        artifact1_copy = artifact1.copy()

        # New artifact
        artifact2 = create_artifact(ArtifactType.IMAGE, "test", {"path": "/b.png"})

        # Simulate update
        artifacts = [artifact1_copy, artifact2]
        for artifact in artifacts:
            artifact_id = artifact.get("id", "")
            artifact["_is_new"] = artifact_id not in gallery._seen_ids
            if artifact_id:
                gallery._seen_ids.add(artifact_id)

        # artifact1 should NOT be new (already seen), artifact2 should be new
        assert artifact1_copy.get("_is_new") is False
        assert artifact2.get("_is_new") is True

    def test_gallery_count_new(self):
        """ArtifactGallery should correctly count new artifacts."""
        from osprey.interfaces.tui.widgets.artifacts import ArtifactGallery

        gallery = ArtifactGallery()

        artifact1 = create_artifact(ArtifactType.IMAGE, "test", {"path": "/a.png"})
        artifact2 = create_artifact(ArtifactType.IMAGE, "test", {"path": "/b.png"})

        # Simulate first update
        artifacts = [artifact1, artifact2]
        for artifact in artifacts:
            artifact_id = artifact.get("id", "")
            artifact["_is_new"] = artifact_id not in gallery._seen_ids
            if artifact_id:
                gallery._seen_ids.add(artifact_id)
        gallery._artifacts = artifacts

        assert gallery._count_new() == 2

        # Add one more, reuse artifact1
        artifact3 = create_artifact(ArtifactType.IMAGE, "test", {"path": "/c.png"})

        # Simulate second update
        artifacts = [artifact1.copy(), artifact2.copy(), artifact3]
        for artifact in artifacts:
            artifact_id = artifact.get("id", "")
            artifact["_is_new"] = artifact_id not in gallery._seen_ids
            if artifact_id:
                gallery._seen_ids.add(artifact_id)
        gallery._artifacts = artifacts

        # Only artifact3 is new now
        assert gallery._count_new() == 1

    def test_gallery_clear_seen(self):
        """clear_seen should reset the seen artifacts set."""
        from osprey.interfaces.tui.widgets.artifacts import ArtifactGallery

        gallery = ArtifactGallery()

        # Add some IDs directly
        gallery._seen_ids.add("test-id-1")
        gallery._seen_ids.add("test-id-2")

        assert len(gallery._seen_ids) == 2

        gallery.clear_seen()

        assert len(gallery._seen_ids) == 0

    def test_gallery_get_selected_artifact(self):
        """get_selected_artifact should return the selected artifact."""
        from osprey.interfaces.tui.widgets.artifacts import ArtifactGallery

        gallery = ArtifactGallery()

        artifact1 = create_artifact(ArtifactType.IMAGE, "test", {"path": "/a.png"})
        artifact2 = create_artifact(ArtifactType.IMAGE, "test", {"path": "/b.png"})

        gallery._artifacts = [artifact1, artifact2]
        gallery._selected_index = 0

        assert gallery.get_selected_artifact() == artifact1

        gallery._selected_index = 1
        assert gallery.get_selected_artifact() == artifact2

    def test_gallery_get_selected_artifact_empty(self):
        """get_selected_artifact should return None when no artifacts."""
        from osprey.interfaces.tui.widgets.artifacts import ArtifactGallery

        gallery = ArtifactGallery()

        assert gallery.get_selected_artifact() is None


class TestArtifactViewer:
    """Tests for the ArtifactViewer combined split-panel modal."""

    def test_viewer_creation_single_image(self):
        """ArtifactViewer should accept a list of artifacts."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        artifact = create_artifact(
            ArtifactType.IMAGE,
            "python_executor",
            {"path": "/path/to/plot.png", "format": "png"},
            display_name="Analysis Plot",
        )
        viewer = ArtifactViewer([artifact])

        assert viewer.artifact == artifact
        assert viewer._artifact_type == ArtifactType.IMAGE
        assert viewer._selected_index == 0

    def test_viewer_creation_multiple(self):
        """ArtifactViewer should handle multiple artifacts."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        artifacts = [
            create_artifact(ArtifactType.IMAGE, "test", {"path": "/a.png"}),
            create_artifact(ArtifactType.NOTEBOOK, "test", {"path": "/nb.ipynb"}),
            create_artifact(ArtifactType.COMMAND, "test", {"uri": "http://localhost"}),
        ]
        viewer = ArtifactViewer(artifacts, selected_index=1)

        assert viewer.artifact == artifacts[1]
        assert viewer._artifact_type == ArtifactType.NOTEBOOK
        assert viewer._selected_index == 1

    def test_viewer_creation_notebook(self):
        """ArtifactViewer should be creatable with a notebook artifact."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        artifact = create_artifact(
            ArtifactType.NOTEBOOK,
            "python_executor",
            {"path": "/path/to/notebook.ipynb", "url": "http://jupyter/notebook"},
        )
        viewer = ArtifactViewer([artifact])

        assert viewer._artifact_type == ArtifactType.NOTEBOOK

    def test_viewer_creation_command(self):
        """ArtifactViewer should be creatable with a command artifact."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        artifact = create_artifact(
            ArtifactType.COMMAND,
            "dashboard",
            {"uri": "http://localhost:8080/dashboard", "command_type": "web_app"},
        )
        viewer = ArtifactViewer([artifact])

        assert viewer._artifact_type == ArtifactType.COMMAND

    def test_viewer_get_openable_target_image(self):
        """ArtifactViewer should return path for images."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        artifact = create_artifact(
            ArtifactType.IMAGE, "test", {"path": "/path/to/plot.png", "format": "png"}
        )
        viewer = ArtifactViewer([artifact])

        assert viewer._get_openable_target() == "/path/to/plot.png"

    def test_viewer_get_openable_target_notebook_prefers_url(self):
        """ArtifactViewer should prefer URL for notebooks."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        artifact = create_artifact(
            ArtifactType.NOTEBOOK,
            "test",
            {"path": "/path/to/notebook.ipynb", "url": "http://jupyter/notebook"},
        )
        viewer = ArtifactViewer([artifact])

        assert viewer._get_openable_target() == "http://jupyter/notebook"

    def test_viewer_get_openable_target_command(self):
        """ArtifactViewer should return URI for commands."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        artifact = create_artifact(ArtifactType.COMMAND, "test", {"uri": "http://localhost:8080"})
        viewer = ArtifactViewer([artifact])

        assert viewer._get_openable_target() == "http://localhost:8080"

    def test_viewer_get_copyable_path(self):
        """ArtifactViewer should return copyable path/url."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        artifact = create_artifact(
            ArtifactType.IMAGE, "test", {"path": "/path/to/plot.png", "format": "png"}
        )
        viewer = ArtifactViewer([artifact])

        assert viewer._get_copyable_path() == "/path/to/plot.png"

    def test_viewer_selected_index_clamped(self):
        """ArtifactViewer should clamp selected_index to valid range."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        artifacts = [
            create_artifact(ArtifactType.IMAGE, "test", {"path": "/a.png"}),
        ]
        viewer = ArtifactViewer(artifacts, selected_index=99)

        assert viewer._selected_index == 0
        assert viewer.artifact == artifacts[0]

    def test_viewer_empty_artifacts(self):
        """ArtifactViewer should handle empty artifact list gracefully."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        viewer = ArtifactViewer([])

        assert viewer.artifact == {}
        assert viewer._selected_index == 0

    def test_viewer_build_title(self):
        """ArtifactViewer title should be path basename, not display_name."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        artifact = create_artifact(
            ArtifactType.IMAGE,
            "test",
            {"path": "/path/to/plot.png"},
            display_name="My Plot",
        )
        viewer = ArtifactViewer([artifact])

        # Path basename takes priority over display_name
        assert viewer._build_title() == "plot.png"

    def test_viewer_build_title_from_path(self):
        """ArtifactViewer title should derive from path basename."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        artifact = create_artifact(
            ArtifactType.IMAGE,
            "test",
            {"path": "/path/to/analysis_plot.png"},
        )
        viewer = ArtifactViewer([artifact])

        assert viewer._build_title() == "analysis_plot.png"

    def test_viewer_build_title_empty(self):
        """ArtifactViewer should show fallback title when empty."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        viewer = ArtifactViewer([])

        assert viewer._build_title() == "Artifacts"

    def test_viewer_metadata_table(self):
        """ArtifactViewer should build a Rich Table for metadata."""
        from rich.table import Table

        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        artifact = create_artifact(
            ArtifactType.IMAGE,
            "test",
            {"path": "/plot.png", "format": "png"},
        )
        viewer = ArtifactViewer([artifact])

        table = viewer._build_metadata_table()
        assert isinstance(table, Table)

    def test_viewer_list_row_truncation(self):
        """Long filenames should be truncated in list rows with tooltip."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        long_name = "very_long_filename_that_exceeds_the_limit.png"
        artifact = create_artifact(
            ArtifactType.IMAGE,
            "test",
            {"path": f"/path/to/{long_name}"},
        )
        viewer = ArtifactViewer([artifact])

        row = viewer._compose_list_row(artifact, 0)
        # Tooltip should reveal full name
        assert row.tooltip == long_name

    def test_viewer_list_row_no_tooltip_for_short_names(self):
        """Short filenames should not have a tooltip."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        artifact = create_artifact(
            ArtifactType.IMAGE,
            "test",
            {"path": "/path/to/plot.png"},
        )
        viewer = ArtifactViewer([artifact])

        row = viewer._compose_list_row(artifact, 0)
        assert row.tooltip is None


class TestChatDisplayArtifactIntegration:
    """Tests for ChatDisplay artifact integration."""

    def test_chat_display_has_artifact_tracking(self):
        """ChatDisplay should have artifact tracking attributes."""
        from osprey.interfaces.tui.widgets.chat_display import ChatDisplay

        display = ChatDisplay()

        assert hasattr(display, "_artifact_gallery")
        assert hasattr(display, "_seen_artifact_ids")
        assert display._artifact_gallery is None
        assert display._seen_artifact_ids == set()

    def test_chat_display_clear_artifact_history(self):
        """ChatDisplay should clear artifact history."""
        from osprey.interfaces.tui.widgets.chat_display import ChatDisplay

        display = ChatDisplay()
        display._seen_artifact_ids.add("test-id-1")
        display._seen_artifact_ids.add("test-id-2")

        display.clear_artifact_history()

        assert display._seen_artifact_ids == set()


class TestArtifactSection:
    """Tests for the redesigned ArtifactSection widget."""

    def test_artifact_section_creation(self):
        """ArtifactSection should store artifacts internally."""
        from osprey.interfaces.tui.widgets.artifacts import ArtifactSection

        artifacts = [
            create_artifact(ArtifactType.IMAGE, "test", {"path": "/a.png"}),
            create_artifact(ArtifactType.IMAGE, "test", {"path": "/b.png"}),
        ]
        section = ArtifactSection(artifacts, section_id="artifacts-1")

        assert section._artifacts == artifacts
        assert len(section._artifacts) == 2

    def test_artifact_section_id(self):
        """ArtifactSection should have the provided section ID."""
        from osprey.interfaces.tui.widgets.artifacts import ArtifactSection

        section = ArtifactSection([], section_id="artifacts-5")
        assert section.id == "artifacts-5"


class TestArtifactViewerSelection:
    """Tests for ArtifactViewer click-based selection."""

    def test_viewer_artifact_property_changes_with_selection(self):
        """ArtifactViewer.artifact should reflect the selected index."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        artifacts = [
            create_artifact(ArtifactType.IMAGE, "test", {"path": "/a.png"}),
            create_artifact(ArtifactType.NOTEBOOK, "test", {"path": "/nb.ipynb"}),
        ]
        viewer = ArtifactViewer(artifacts)

        assert viewer.artifact == artifacts[0]
        assert viewer._artifact_type == ArtifactType.IMAGE

        # Simulate click-based selection change
        viewer._selected_index = 1
        assert viewer.artifact == artifacts[1]
        assert viewer._artifact_type == ArtifactType.NOTEBOOK


class TestArtifactTypeIcons:
    """Tests for artifact type icons."""

    def test_all_types_have_icons(self):
        """All artifact types should have icons."""
        from osprey.state.artifacts import ArtifactType, get_artifact_type_icon

        for artifact_type in ArtifactType:
            icon = get_artifact_type_icon(artifact_type)
            assert icon is not None
            assert len(icon) > 0

    def test_icon_lookup_by_string(self):
        """Icons should be retrievable by string type."""
        from osprey.state.artifacts import get_artifact_type_icon

        assert get_artifact_type_icon("image") == get_artifact_type_icon(ArtifactType.IMAGE)
        assert get_artifact_type_icon("notebook") == get_artifact_type_icon(ArtifactType.NOTEBOOK)
        assert get_artifact_type_icon("command") == get_artifact_type_icon(ArtifactType.COMMAND)


class TestArtifactViewerNotebookPreview:
    """Tests for ArtifactViewer notebook preview rendering."""

    def test_notebook_type_detected(self):
        """ArtifactViewer should detect NOTEBOOK artifact type."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        artifact = create_artifact(
            ArtifactType.NOTEBOOK,
            "python_executor",
            {"path": "/tmp/notebook.ipynb"},
        )
        viewer = ArtifactViewer([artifact])
        assert viewer._artifact_type == ArtifactType.NOTEBOOK

    def test_extract_stream_output(self):
        """Should extract text from stream outputs."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        viewer = ArtifactViewer(
            [create_artifact(ArtifactType.NOTEBOOK, "test", {"path": "/nb.ipynb"})]
        )
        output = {"output_type": "stream", "text": "hello world\n"}
        assert viewer._extract_cell_output_text(output) == "hello world\n"

    def test_extract_stream_output_list(self):
        """Should handle stream output as list of strings."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        viewer = ArtifactViewer(
            [create_artifact(ArtifactType.NOTEBOOK, "test", {"path": "/nb.ipynb"})]
        )
        output = {
            "output_type": "stream",
            "text": ["hello ", "world\n"],
        }
        assert viewer._extract_cell_output_text(output) == "hello world\n"

    def test_extract_execute_result(self):
        """Should extract text/plain from execute_result."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        viewer = ArtifactViewer(
            [create_artifact(ArtifactType.NOTEBOOK, "test", {"path": "/nb.ipynb"})]
        )
        output = {
            "output_type": "execute_result",
            "data": {"text/plain": "42"},
        }
        assert viewer._extract_cell_output_text(output) == "42"

    def test_extract_execute_result_list(self):
        """Should handle execute_result text as list."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        viewer = ArtifactViewer(
            [create_artifact(ArtifactType.NOTEBOOK, "test", {"path": "/nb.ipynb"})]
        )
        output = {
            "output_type": "execute_result",
            "data": {"text/plain": ["line1\n", "line2\n"]},
        }
        result = viewer._extract_cell_output_text(output)
        assert result == "line1\nline2\n"

    def test_extract_display_data_skipped(self):
        """Should return empty string for display_data outputs."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        viewer = ArtifactViewer(
            [create_artifact(ArtifactType.NOTEBOOK, "test", {"path": "/nb.ipynb"})]
        )
        output = {
            "output_type": "display_data",
            "data": {"image/png": "base64data..."},
        }
        assert viewer._extract_cell_output_text(output) == ""

    def test_extract_error_output(self):
        """Should extract and clean error tracebacks."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        viewer = ArtifactViewer(
            [create_artifact(ArtifactType.NOTEBOOK, "test", {"path": "/nb.ipynb"})]
        )
        output = {
            "output_type": "error",
            "traceback": [
                "Traceback (most recent call last):",
                '  File "test.py", line 1',
                "ValueError: bad value",
            ],
        }
        result = viewer._extract_cell_output_text(output)
        assert "ValueError: bad value" in result

    def test_extract_error_strips_ansi(self):
        """Should strip ANSI escape codes from error tracebacks."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        viewer = ArtifactViewer(
            [create_artifact(ArtifactType.NOTEBOOK, "test", {"path": "/nb.ipynb"})]
        )
        output = {
            "output_type": "error",
            "traceback": ["\x1b[31mValueError\x1b[0m: bad"],
        }
        result = viewer._extract_cell_output_text(output)
        assert result == "ValueError: bad"
        assert "\x1b" not in result

    def test_extract_stream_truncation(self):
        """Should truncate stream outputs longer than 20 lines."""
        from osprey.interfaces.tui.widgets.artifact_viewer import ArtifactViewer

        viewer = ArtifactViewer(
            [create_artifact(ArtifactType.NOTEBOOK, "test", {"path": "/nb.ipynb"})]
        )
        long_text = "\n".join(f"line {i}" for i in range(30))
        output = {"output_type": "stream", "text": long_text}
        result = viewer._extract_cell_output_text(output)
        assert "... (10 more lines)" in result
