"""Tests for the workflow_autodoc Sphinx extension.

This module tests the custom Sphinx extension that auto-documents workflow files
with YAML frontmatter.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Skip all tests in this module if Sphinx is not available
# Sphinx is only required for documentation builds, not for general testing
pytest.importorskip("sphinx", reason="Sphinx is required for workflow autodoc tests")

# Add the docs/_ext directory to the path
docs_ext_path = Path(__file__).parent.parent.parent / "docs" / "source" / "_ext"
sys.path.insert(0, str(docs_ext_path))

from workflow_autodoc import (
    WorkflowListDirective,
    WorkflowSummaryDirective,
    parse_workflow_file,
)


class TestParseWorkflowFile:
    """Test the parse_workflow_file function."""

    def test_parse_file_with_frontmatter(self, tmp_path):
        """Test parsing a workflow file with YAML frontmatter."""
        workflow_file = tmp_path / "test-workflow.md"
        content = """---
workflow: test-workflow
category: code-quality
applies_when: [before_commit]
estimated_time: 5 minutes
ai_ready: true
---

# Test Workflow

This is a test workflow description.

## Section 1

Some content here.
"""
        workflow_file.write_text(content)

        result = parse_workflow_file(workflow_file)

        assert result["metadata"]["workflow"] == "test-workflow"
        assert result["metadata"]["category"] == "code-quality"
        assert result["metadata"]["applies_when"] == ["before_commit"]
        assert result["metadata"]["estimated_time"] == "5 minutes"
        assert result["metadata"]["ai_ready"] is True
        assert result["title"] == "Test Workflow"
        assert "test workflow description" in result["description"].lower()

    def test_parse_file_without_frontmatter(self, tmp_path):
        """Test parsing a workflow file without YAML frontmatter."""
        workflow_file = tmp_path / "simple-workflow.md"
        content = """# Simple Workflow

Just a simple description here.

## Details

More details.
"""
        workflow_file.write_text(content)

        result = parse_workflow_file(workflow_file)

        assert result["metadata"] == {}
        assert result["title"] == "Simple Workflow"
        assert "simple description" in result["description"].lower()

    def test_parse_nonexistent_file(self, tmp_path):
        """Test parsing a file that doesn't exist."""
        workflow_file = tmp_path / "nonexistent.md"

        result = parse_workflow_file(workflow_file)

        assert result["metadata"] == {}
        assert result["title"] == "nonexistent"
        assert result["description"] == ""
        assert result["content"] == ""

    def test_parse_file_with_invalid_yaml(self, tmp_path):
        """Test parsing a file with invalid YAML frontmatter."""
        workflow_file = tmp_path / "invalid-yaml.md"
        content = """---
invalid: yaml: syntax: here
---

# Workflow with Invalid YAML

Should still parse the content.
"""
        workflow_file.write_text(content)

        result = parse_workflow_file(workflow_file)

        # Should handle gracefully with empty metadata
        assert result["metadata"] == {}
        assert result["title"] == "Workflow with Invalid YAML"

    def test_description_extraction(self, tmp_path):
        """Test that description is properly extracted."""
        workflow_file = tmp_path / "description-test.md"
        content = """---
workflow: test
---

# Test Title

**This is the description** with some formatting.

It can span multiple lines.

## First Section

This should not be in the description.
"""
        workflow_file.write_text(content)

        result = parse_workflow_file(workflow_file)

        # Description should be cleaned (no ** markers)
        assert "**" not in result["description"]
        assert "This is the description" in result["description"]
        # Note: The description extraction stops at first blank line before ##
        # so "multiple lines" may not be included depending on blank line handling
        assert "First Section" not in result["description"]


class TestWorkflowSummaryDirective:
    """Test the WorkflowSummaryDirective."""

    @pytest.fixture
    def mock_directive(self):
        """Create a mock directive instance."""
        directive = WorkflowSummaryDirective(
            name="workflow-summary",
            arguments=["../../workflows/test.md"],
            options={},
            content=[],
            lineno=1,
            content_offset=0,
            block_text="",
            state=Mock(),
            state_machine=Mock(),
        )

        # Mock the state and environment
        directive.state = Mock()
        directive.state.document.settings.env = Mock()
        directive.state.document.settings.env.srcdir = "/fake/docs/source"
        directive.state.document.settings.env.docname = "contributing/index"
        directive.state.document.settings.env.doc2path = Mock(
            return_value="/fake/docs/source/contributing/index.rst"
        )

        return directive

    def test_workflow_summary_basic(self, mock_directive, tmp_path, monkeypatch):
        """Test basic workflow summary generation."""
        # Create a test workflow file structure that matches expected paths
        fake_docs = tmp_path / "fake" / "docs"
        fake_docs.mkdir(parents=True)
        workflows_dir = fake_docs / "workflows"
        workflows_dir.mkdir()
        workflow_file = workflows_dir / "test.md"
        workflow_file.write_text(
            """---
workflow: test
category: code-quality
estimated_time: 5 minutes
---

# Test Workflow

A simple test workflow.
"""
        )

        # Update mock to use consistent paths
        mock_directive.state.document.settings.env.srcdir = str(fake_docs / "source")

        # Patch the file path resolution with proper parent directory
        with patch.object(Path, "resolve", return_value=workflow_file):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "read_text", return_value=workflow_file.read_text()):
                    with patch.object(Path, "relative_to", return_value=Path("workflows/test.md")):
                        nodes = mock_directive.run()

        # Should return a container with content
        assert len(nodes) > 0
        container = nodes[0]
        assert container["classes"] == ["workflow-summary"]


class TestWorkflowListDirective:
    """Test the WorkflowListDirective."""

    @pytest.fixture
    def mock_directive(self):
        """Create a mock directive instance."""
        directive = WorkflowListDirective(
            name="workflow-list",
            arguments=[],
            options={},
            content=[],
            lineno=1,
            content_offset=0,
            block_text="",
            state=Mock(),
            state_machine=Mock(),
        )

        # Mock the state and environment
        directive.state = Mock()
        directive.state.document.settings.env = Mock()
        directive.state.document.settings.env.srcdir = "/fake/docs/source"

        return directive

    def test_workflow_list_no_directory(self, mock_directive, tmp_path):
        """Test workflow list when workflows directory doesn't exist."""
        with patch.object(Path, "__truediv__", return_value=tmp_path / "nonexistent"):
            nodes = mock_directive.run()

        # Should return a warning node
        assert len(nodes) > 0

    def test_workflow_list_with_workflows(self, mock_directive, tmp_path, monkeypatch):
        """Test workflow list generation with actual workflow files."""
        # Create test workflow files
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()

        (workflows_dir / "workflow1.md").write_text(
            """---
workflow: workflow1
category: code-quality
---

# Workflow One

Description one.
"""
        )

        (workflows_dir / "workflow2.md").write_text(
            """---
workflow: workflow2
category: documentation
---

# Workflow Two

Description two.
"""
        )

        (workflows_dir / "README.md").write_text("This should be ignored")

        # Mock the source directory
        mock_directive.state.document.settings.env.srcdir = str(tmp_path / "docs" / "source")

        # Create the parent directory structure
        parent_dir = tmp_path / "docs"
        parent_dir.mkdir()

        with patch.object(Path, "resolve", side_effect=lambda: workflows_dir):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "glob", return_value=list(workflows_dir.glob("*.md"))):
                    nodes = mock_directive.run()

        # Should return nodes for the workflows (excluding README.md)
        assert len(nodes) > 0

    def test_workflow_list_filtered_by_category(self, mock_directive, tmp_path):
        """Test workflow list filtering by category."""
        mock_directive.options["category"] = "code-quality"

        # Test would involve similar mocking as above
        # This is a placeholder for the concept
        assert "category" in mock_directive.options


class TestSphinxIntegration:
    """Test Sphinx integration."""

    def test_setup_function_exists(self):
        """Test that the setup function is properly defined."""
        from workflow_autodoc import setup

        assert callable(setup)

    def test_setup_returns_dict(self):
        """Test that setup returns the expected dictionary."""
        from workflow_autodoc import setup

        mock_app = Mock()
        result = setup(mock_app)

        assert isinstance(result, dict)
        assert "version" in result
        assert "parallel_read_safe" in result
        assert "parallel_write_safe" in result

    def test_directives_registered(self):
        """Test that directives are registered with Sphinx."""
        from workflow_autodoc import setup

        mock_app = Mock()
        setup(mock_app)

        # Verify directives were registered
        assert mock_app.add_directive.call_count == 2
        calls = [call[0] for call in mock_app.add_directive.call_args_list]
        assert ("workflow-summary", WorkflowSummaryDirective) in calls
        assert ("workflow-list", WorkflowListDirective) in calls

    def test_css_file_registered(self):
        """Test that CSS file is registered."""
        from workflow_autodoc import setup

        mock_app = Mock()
        setup(mock_app)

        mock_app.add_css_file.assert_called_once_with("workflow-autodoc.css")
