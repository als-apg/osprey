"""
Tests for ProjectManager

Tests the project manager functionality including project discovery,
loading, enabling/disabling, and capability extraction.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from osprey.interfaces.pyqt.project_manager import (
    CapabilityMetadata,
    ProjectManager,
    ProjectMetadata,
    ProjectNotFoundError,
)


class TestProjectMetadata:
    """Test suite for ProjectMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating project metadata."""
        metadata = ProjectMetadata(
            name="test_project",
            path=Path("/test/path"),
            config_path=Path("/test/path/config.yml"),
            description="Test project",
            version="1.0.0",
        )

        assert metadata.name == "test_project"
        assert metadata.path == Path("/test/path")
        assert metadata.description == "Test project"
        assert metadata.version == "1.0.0"

    def test_metadata_with_optional_fields(self):
        """Test metadata with optional fields."""
        metadata = ProjectMetadata(
            name="test_project",
            path=Path("/test/path"),
            config_path=Path("/test/path/config.yml"),
            description="Test project",
            version="1.0.0",
            author="Test Author",
            tags=["tag1", "tag2"],
        )

        assert metadata.author == "Test Author"
        assert metadata.tags == ["tag1", "tag2"]


class TestCapabilityMetadata:
    """Test suite for CapabilityMetadata dataclass."""

    def test_create_capability_metadata(self):
        """Test creating capability metadata."""
        cap = CapabilityMetadata(
            name="test_capability",
            project="test_project",
            description="Test capability",
            input_schema={},
            output_schema={},
        )

        assert cap.name == "test_capability"
        assert cap.project == "test_project"
        assert cap.description == "Test capability"


class TestProjectManagerInitialization:
    """Test suite for ProjectManager initialization."""

    def test_init_with_default_paths(self):
        """Test initialization with default search paths."""
        manager = ProjectManager()

        assert manager.project_search_paths is not None
        assert len(manager.project_search_paths) > 0
        assert isinstance(manager._enabled_projects, set)

    def test_init_with_custom_paths(self):
        """Test initialization with custom search paths."""
        custom_paths = [Path("/custom/path1"), Path("/custom/path2")]
        manager = ProjectManager(project_search_paths=custom_paths)

        assert manager.project_search_paths == custom_paths


class TestProjectDiscovery:
    """Test suite for project discovery."""

    def test_discover_projects_empty_directory(self, tmp_path):
        """Test discovering projects in empty directory."""
        manager = ProjectManager(project_search_paths=[tmp_path])
        discovered = manager.discover_projects()

        assert discovered == []

    def test_discover_single_project(self, tmp_path):
        """Test discovering a single project."""
        # Create project directory with config
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        config_data = {
            "project": {
                "name": "test_project",
                "description": "Test project",
                "version": "1.0.0",
            }
        }

        config_path = project_dir / "config.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        manager = ProjectManager(project_search_paths=[tmp_path])
        discovered = manager.discover_projects()

        assert len(discovered) == 1
        assert discovered[0].name == "test_project"
        assert discovered[0].description == "Test project"
        assert discovered[0].version == "1.0.0"

    def test_discover_multiple_projects(self, tmp_path):
        """Test discovering multiple projects."""
        # Create multiple projects
        for i in range(3):
            project_dir = tmp_path / f"project_{i}"
            project_dir.mkdir()

            config_data = {
                "project": {
                    "name": f"project_{i}",
                    "description": f"Project {i}",
                    "version": "1.0.0",
                }
            }

            with open(project_dir / "config.yml", "w") as f:
                yaml.dump(config_data, f)

        manager = ProjectManager(project_search_paths=[tmp_path])
        discovered = manager.discover_projects()

        assert len(discovered) == 3
        project_names = {p.name for p in discovered}
        assert project_names == {"project_0", "project_1", "project_2"}

    def test_discover_ignores_hidden_directories(self, tmp_path):
        """Test that discovery ignores hidden directories."""
        # Create hidden directory
        hidden_dir = tmp_path / ".hidden_project"
        hidden_dir.mkdir()

        config_data = {"project": {"name": "hidden", "description": "Hidden", "version": "1.0.0"}}
        with open(hidden_dir / "config.yml", "w") as f:
            yaml.dump(config_data, f)

        manager = ProjectManager(project_search_paths=[tmp_path])
        discovered = manager.discover_projects()

        assert len(discovered) == 0

    def test_discover_ignores_common_directories(self, tmp_path):
        """Test that discovery ignores common non-project directories."""
        # Create directories that should be ignored
        ignore_dirs = ["node_modules", "venv", "__pycache__", ".git"]

        for dir_name in ignore_dirs:
            dir_path = tmp_path / dir_name
            dir_path.mkdir()

            config_data = {"project": {"name": dir_name, "description": "Test", "version": "1.0.0"}}
            with open(dir_path / "config.yml", "w") as f:
                yaml.dump(config_data, f)

        manager = ProjectManager(project_search_paths=[tmp_path])
        discovered = manager.discover_projects()

        assert len(discovered) == 0

    def test_discover_handles_missing_config(self, tmp_path):
        """Test that discovery handles directories without config.yml."""
        # Create directory without config
        project_dir = tmp_path / "no_config"
        project_dir.mkdir()

        manager = ProjectManager(project_search_paths=[tmp_path])
        discovered = manager.discover_projects()

        assert len(discovered) == 0

    def test_discover_handles_invalid_config(self, tmp_path):
        """Test that discovery handles invalid config files."""
        project_dir = tmp_path / "invalid_config"
        project_dir.mkdir()

        # Write invalid YAML
        with open(project_dir / "config.yml", "w") as f:
            f.write("invalid: yaml: content: [")

        manager = ProjectManager(project_search_paths=[tmp_path])
        discovered = manager.discover_projects()

        # Should skip invalid config
        assert len(discovered) == 0


class TestProjectLoading:
    """Test suite for project loading."""

    @patch("osprey.interfaces.pyqt.project_manager.ProjectContextManager")
    def test_load_project_success(self, mock_context_manager_class, tmp_path):
        """Test successful project loading."""
        # Setup project
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        config_data = {
            "project": {
                "name": "test_project",
                "description": "Test",
                "version": "1.0.0",
            }
        }
        with open(project_dir / "config.yml", "w") as f:
            yaml.dump(config_data, f)

        # Mock context manager
        mock_context = Mock()
        mock_registry = Mock()
        mock_registry.get_all_capabilities.return_value = []
        mock_context.get_registry.return_value = mock_registry

        mock_context_manager = Mock()
        mock_context_manager.get_context.return_value = None  # Not loaded yet
        mock_context_manager.create_project_context.return_value = mock_context
        mock_context_manager_class.return_value = mock_context_manager

        # Load project
        manager = ProjectManager(project_search_paths=[tmp_path])
        manager.discover_projects()

        context = manager.load_project("test_project")

        assert context is not None
        assert manager.is_project_enabled("test_project")

    @patch("osprey.interfaces.pyqt.project_manager.ProjectContextManager")
    def test_load_project_already_loaded(self, mock_context_manager_class, tmp_path):
        """Test loading already loaded project returns existing context."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        config_data = {
            "project": {
                "name": "test_project",
                "description": "Test",
                "version": "1.0.0",
            }
        }
        with open(project_dir / "config.yml", "w") as f:
            yaml.dump(config_data, f)

        # Mock existing context
        mock_context = Mock()
        mock_context_manager = Mock()
        mock_context_manager.get_context.return_value = mock_context
        mock_context_manager_class.return_value = mock_context_manager

        manager = ProjectManager(project_search_paths=[tmp_path])
        manager.discover_projects()

        context = manager.load_project("test_project")

        assert context == mock_context
        # Should not call create_project_context
        mock_context_manager.create_project_context.assert_not_called()

    def test_load_project_not_found(self, tmp_path):
        """Test loading non-existent project raises error."""
        manager = ProjectManager(project_search_paths=[tmp_path])

        with pytest.raises(ProjectNotFoundError):
            manager.load_project("nonexistent_project")


class TestProjectEnableDisable:
    """Test suite for project enable/disable functionality."""

    @patch("osprey.interfaces.pyqt.project_manager.ProjectContextManager")
    def test_enable_project(self, mock_context_manager_class):
        """Test enabling a project."""
        mock_context = Mock()
        mock_context_manager = Mock()
        mock_context_manager.get_context.return_value = mock_context
        mock_context_manager_class.return_value = mock_context_manager

        manager = ProjectManager()
        result = manager.enable_project("test_project")

        assert result is True
        assert manager.is_project_enabled("test_project")

    @patch("osprey.interfaces.pyqt.project_manager.ProjectContextManager")
    def test_disable_project(self, mock_context_manager_class):
        """Test disabling a project."""
        mock_context = Mock()
        mock_context_manager = Mock()
        mock_context_manager.get_context.return_value = mock_context
        mock_context_manager_class.return_value = mock_context_manager

        manager = ProjectManager()
        manager.enable_project("test_project")
        assert manager.is_project_enabled("test_project")

        result = manager.disable_project("test_project")

        assert result is True
        assert not manager.is_project_enabled("test_project")

    def test_enable_nonexistent_project(self):
        """Test enabling non-existent project returns False."""
        manager = ProjectManager()
        result = manager.enable_project("nonexistent")

        assert result is False

    def test_disable_nonexistent_project(self):
        """Test disabling non-existent project returns False."""
        manager = ProjectManager()
        result = manager.disable_project("nonexistent")

        assert result is False


class TestProjectQueries:
    """Test suite for project query methods."""

    @patch("osprey.interfaces.pyqt.project_manager.ProjectContextManager")
    def test_list_loaded_projects(self, mock_context_manager_class):
        """Test listing loaded projects."""
        mock_context_manager = Mock()
        mock_context_manager.list_projects.return_value = ["project1", "project2"]
        mock_context_manager_class.return_value = mock_context_manager

        manager = ProjectManager()
        loaded = manager.list_loaded_projects()

        assert loaded == ["project1", "project2"]

    def test_list_available_projects(self, tmp_path):
        """Test listing available projects."""
        # Create projects
        for i in range(2):
            project_dir = tmp_path / f"project_{i}"
            project_dir.mkdir()

            config_data = {
                "project": {
                    "name": f"project_{i}",
                    "description": f"Project {i}",
                    "version": "1.0.0",
                }
            }
            with open(project_dir / "config.yml", "w") as f:
                yaml.dump(config_data, f)

        manager = ProjectManager(project_search_paths=[tmp_path])
        available = manager.list_available_projects()

        assert len(available) == 2
        names = {p.name for p in available}
        assert names == {"project_0", "project_1"}

    @patch("osprey.interfaces.pyqt.project_manager.ProjectContextManager")
    def test_get_enabled_projects(self, mock_context_manager_class):
        """Test getting enabled projects."""
        mock_context1 = Mock()
        mock_context2 = Mock()

        mock_context_manager = Mock()

        def get_context_side_effect(name):
            if name == "project1":
                return mock_context1
            elif name == "project2":
                return mock_context2
            return None

        mock_context_manager.get_context.side_effect = get_context_side_effect
        mock_context_manager_class.return_value = mock_context_manager

        manager = ProjectManager()
        manager.enable_project("project1")
        manager.enable_project("project2")

        enabled = manager.get_enabled_projects()

        assert len(enabled) == 2
        assert mock_context1 in enabled
        assert mock_context2 in enabled

    @patch("osprey.interfaces.pyqt.project_manager.ProjectContextManager")
    def test_get_disabled_projects(self, mock_context_manager_class):
        """Test getting disabled projects."""
        mock_context_manager = Mock()
        mock_context_manager.list_projects.return_value = ["project1", "project2", "project3"]
        mock_context_manager.get_context.return_value = Mock()
        mock_context_manager_class.return_value = mock_context_manager

        manager = ProjectManager()
        manager.enable_project("project1")
        # project2 and project3 are disabled

        disabled = manager.get_disabled_projects()

        assert set(disabled) == {"project2", "project3"}


class TestProjectCapabilities:
    """Test suite for project capability extraction."""

    @patch("osprey.interfaces.pyqt.project_manager.ProjectContextManager")
    def test_get_project_capabilities(self, mock_context_manager_class):
        """Test getting capabilities from a project."""
        # Mock capability
        mock_capability = Mock()
        mock_capability.name = "test_capability"
        mock_capability.description = "Test capability description"

        # Mock registry
        mock_registry = Mock()
        mock_registry.get_all_capabilities.return_value = [mock_capability]

        # Mock context
        mock_context = Mock()
        mock_context.get_registry.return_value = mock_registry

        mock_context_manager = Mock()
        mock_context_manager.get_context.return_value = mock_context
        mock_context_manager_class.return_value = mock_context_manager

        manager = ProjectManager()
        capabilities = manager.get_project_capabilities("test_project")

        assert "test_capability" in capabilities
        assert capabilities["test_capability"].name == "test_capability"
        assert capabilities["test_capability"].description == "Test capability description"
        assert capabilities["test_capability"].project == "test_project"

    def test_get_capabilities_project_not_loaded(self):
        """Test getting capabilities from unloaded project raises error."""
        manager = ProjectManager()

        with pytest.raises(ProjectNotFoundError):
            manager.get_project_capabilities("nonexistent_project")


class TestProjectUnloadReload:
    """Test suite for project unload and reload."""

    @patch("osprey.interfaces.pyqt.project_manager.ProjectContextManager")
    def test_unload_project(self, mock_context_manager_class):
        """Test unloading a project."""
        mock_context_manager = Mock()
        mock_context_manager.remove_context.return_value = True
        mock_context_manager.get_context.return_value = Mock()
        mock_context_manager_class.return_value = mock_context_manager

        manager = ProjectManager()
        manager.enable_project("test_project")

        result = manager.unload_project("test_project")

        assert result is True
        assert not manager.is_project_enabled("test_project")
        mock_context_manager.remove_context.assert_called_once_with("test_project")

    @patch("osprey.interfaces.pyqt.project_manager.ProjectContextManager")
    def test_unload_nonexistent_project(self, mock_context_manager_class):
        """Test unloading non-existent project returns False."""
        mock_context_manager = Mock()
        mock_context_manager.remove_context.return_value = False
        mock_context_manager_class.return_value = mock_context_manager

        manager = ProjectManager()
        result = manager.unload_project("nonexistent")

        assert result is False

    @patch("osprey.interfaces.pyqt.project_manager.ProjectContextManager")
    def test_reload_project(self, mock_context_manager_class, tmp_path):
        """Test reloading a project."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        config_data = {
            "project": {
                "name": "test_project",
                "description": "Test",
                "version": "1.0.0",
            }
        }
        with open(project_dir / "config.yml", "w") as f:
            yaml.dump(config_data, f)

        mock_context = Mock()
        mock_registry = Mock()
        mock_registry.get_all_capabilities.return_value = []
        mock_context.get_registry.return_value = mock_registry

        mock_context_manager = Mock()
        mock_context_manager.remove_context.return_value = True
        mock_context_manager.get_context.return_value = None
        mock_context_manager.create_project_context.return_value = mock_context
        mock_context_manager_class.return_value = mock_context_manager

        manager = ProjectManager(project_search_paths=[tmp_path])
        manager.discover_projects()

        reloaded = manager.reload_project("test_project")

        assert reloaded is not None
        mock_context_manager.remove_context.assert_called_once()
        mock_context_manager.create_project_context.assert_called_once()


class TestProjectManagerEdgeCases:
    """Test suite for edge cases."""

    def test_discover_with_nonexistent_search_path(self):
        """Test discovery with non-existent search path."""
        manager = ProjectManager(project_search_paths=[Path("/nonexistent/path")])
        discovered = manager.discover_projects()

        # Should handle gracefully
        assert discovered == []

    def test_get_project_returns_none_for_unloaded(self):
        """Test get_project returns None for unloaded project."""
        manager = ProjectManager()
        context = manager.get_project("nonexistent")

        assert context is None

    def test_metadata_cache_persists(self, tmp_path):
        """Test that metadata cache persists across discover calls."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        config_data = {
            "project": {
                "name": "test_project",
                "description": "Test",
                "version": "1.0.0",
            }
        }
        with open(project_dir / "config.yml", "w") as f:
            yaml.dump(config_data, f)

        manager = ProjectManager(project_search_paths=[tmp_path])

        # First discovery
        discovered1 = manager.discover_projects()
        assert len(discovered1) == 1

        # Second discovery should use cache
        discovered2 = manager.discover_projects()
        assert len(discovered2) == 1
        assert discovered1[0].name == discovered2[0].name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
