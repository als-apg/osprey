"""
Project Manager for Multi-Project GUI Support

This module provides the ProjectManager class for discovering, loading, and managing
multiple Osprey projects at runtime. Each project maintains its own isolated Gateway,
Registry, and ContextManager.

Key Features:
- Dynamic project discovery in configured directories
- Isolated project contexts (no registry merging)
- Runtime enable/disable of individual projects
- Reuses framework's ConfigBuilder, Gateway, and ContextManager
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

import yaml

from osprey.utils.logger import get_logger
from osprey.interfaces.pyqt.project_context_manager import (
    ProjectContextManager,
    IsolatedProjectContext
)

logger = get_logger("project_manager")


@dataclass
class ProjectMetadata:
    """Metadata about a discovered project."""
    name: str
    path: Path
    config_path: Path
    description: str
    version: str
    author: str | None = None
    tags: List[str] = field(default_factory=list)


@dataclass
class CapabilityMetadata:
    """Metadata about a capability."""
    name: str
    project: str
    description: str
    input_schema: Dict
    output_schema: Dict
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)




class ProjectManager:
    """Manages discovery and loading of multiple projects.

    The ProjectManager is responsible for:
    - Discovering projects in configured directories
    - Loading each project with its own Gateway and Registry
    - Tracking which projects are enabled/disabled for routing
    - Providing access to project capabilities

    All projects remain loaded in memory; disabling only affects routing.
    """

    def __init__(self, project_search_paths: List[Path] = None):
        """Initialize ProjectManager.

        Args:
            project_search_paths: Directories to search for projects.
                                 If None, uses default locations.
        """
        self.logger = logger
        self.project_search_paths = project_search_paths or self._get_default_search_paths()
        self._context_manager = ProjectContextManager()  # Manages isolated contexts
        self._metadata_cache: Dict[str, ProjectMetadata] = {}
        self._enabled_projects: Set[str] = set()  # Track enabled/disabled state

        self.logger.info(f"Initialized ProjectManager with search paths: {self.project_search_paths}")

    def discover_projects(self) -> List[ProjectMetadata]:
        """Discover all available projects.

        Scans configured directories for config.yml files and extracts
        project metadata.

        Returns:
            List of discovered project metadata.

        Raises:
            ProjectDiscoveryError: If discovery fails.
        """
        discovered = []

        # Directories to ignore (same as CLI's discover_nearby_projects)
        ignore_dirs = {
            'node_modules', 'venv', '.venv', 'env', '.env',
            '__pycache__', '.git', '.svn', '.hg',
            'build', 'dist', '.egg-info', 'site-packages',
            '.pytest_cache', '.mypy_cache', '.tox',
            'docs', '_agent_data', '.cache', 'temp_configs'
        }

        for search_path in self.project_search_paths:
            if not search_path.exists():
                self.logger.warning(f"Search path does not exist: {search_path}")
                continue

            self.logger.debug(f"Searching for projects in: {search_path}")

            for project_dir in search_path.iterdir():
                if not project_dir.is_dir():
                    continue

                # Skip hidden directories (start with .)
                if project_dir.name.startswith('.'):
                    continue

                # Skip common non-project directories (same as CLI)
                if project_dir.name in ignore_dirs:
                    self.logger.debug(f"Skipping ignored directory: {project_dir.name}")
                    continue

                config_path = project_dir / "config.yml"

                # Check if config exists, handling permission errors
                try:
                    if not config_path.exists():
                        continue
                except (PermissionError, OSError) as e:
                    self.logger.debug(f"Cannot access {config_path}: {e}")
                    continue

                try:
                    metadata = self._parse_project_metadata(project_dir, config_path)
                    discovered.append(metadata)
                    self._metadata_cache[metadata.name] = metadata
                    self.logger.info(f"Discovered project: {metadata.name} at {project_dir}")
                except Exception as e:
                    self.logger.error(f"Failed to parse project {project_dir}: {e}")
                    continue

        self.logger.info(f"Discovered {len(discovered)} projects: {[p.name for p in discovered]}")
        return discovered

    def load_project(self, project_name: str) -> IsolatedProjectContext:
        """Load a project and create its isolated context.

        IMPORTANT: This loads the project into memory with its own
        isolated Gateway, Registry, and Graph. The project remains
        loaded even if later disabled. Disabling only affects routing,
        not the loaded state.

        This method uses ProjectContextManager to ensure complete isolation
        between projects - no config pollution, no registry merging.

        Args:
            project_name: Name of project to load.

        Returns:
            IsolatedProjectContext with loaded components.

        Raises:
            ProjectNotFoundError: If project not found.
            ProjectLoadError: If loading fails.
        """
        # Check if already loaded
        existing_context = self._context_manager.get_context(project_name)
        if existing_context is not None:
            self.logger.debug(f"Project already loaded: {project_name}")
            return existing_context

        # Get metadata
        if project_name not in self._metadata_cache:
            self.discover_projects()

        if project_name not in self._metadata_cache:
            raise ProjectNotFoundError(f"Project not found: {project_name}")

        metadata = self._metadata_cache[project_name]

        try:
            self.logger.info(f"Loading project with isolated context: {project_name}")

            # Create isolated context using ProjectContextManager
            # This ensures NO global state pollution
            context = self._context_manager.create_project_context(
                project_name=project_name,
                project_path=metadata.path,
                config_path=metadata.config_path
            )

            # Set metadata for GUI compatibility
            context.metadata = metadata

            # Enable by default (all agents start enabled)
            self._enabled_projects.add(project_name)

            self.logger.info(
                f"Successfully loaded project: {project_name} with "
                f"{len(context.get_registry().get_all_capabilities())} capabilities"
            )
            return context

        except Exception as e:
            self.logger.error(f"Failed to load project {project_name}: {e}")
            raise ProjectLoadError(f"Failed to load project {project_name}") from e

    def get_project(self, project_name: str) -> IsolatedProjectContext | None:
        """Get a loaded project context.

        Args:
            project_name: Name of project.

        Returns:
            IsolatedProjectContext if loaded, None otherwise.
        """
        return self._context_manager.get_context(project_name)

    def list_loaded_projects(self) -> List[str]:
        """Get names of all loaded projects."""
        return self._context_manager.list_projects()

    def list_available_projects(self) -> List[ProjectMetadata]:
        """Get metadata for all available projects."""
        if not self._metadata_cache:
            self.discover_projects()
        return list(self._metadata_cache.values())

    def get_project_capabilities(self, project_name: str) -> Dict[str, CapabilityMetadata]:
        """Get capabilities for a project from its isolated registry.

        Args:
            project_name: Name of project.

        Returns:
            Dictionary mapping capability names to metadata.

        Raises:
            ProjectNotFoundError: If project not found.
        """
        context = self.get_project(project_name)
        if context is None:
            raise ProjectNotFoundError(f"Project not loaded: {project_name}")

        capabilities = {}

        # Extract capabilities directly from the project's loaded registry
        try:
            registry = context.get_registry()

            # Get all capabilities from the registry
            for capability in registry.get_all_capabilities():
                cap_name = getattr(capability, 'name', None)
                cap_desc = getattr(capability, 'description', '')

                if cap_name:
                    capabilities[cap_name] = CapabilityMetadata(
                        name=cap_name,
                        project=project_name,
                        description=cap_desc,
                        input_schema={},
                        output_schema={},
                        tags=[],
                        examples=[]
                    )

            self.logger.info(f"Extracted {len(capabilities)} capabilities from {project_name}")

        except Exception as e:
            self.logger.warning(f"Could not extract capabilities from registry for {project_name}: {e}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")

        return capabilities

    def enable_project(self, project_name: str) -> bool:
        """Enable a project for routing (runtime control).

        Args:
            project_name: Name of project to enable.

        Returns:
            True if enabled, False if project not loaded.
        """
        if self._context_manager.get_context(project_name) is not None:
            self._enabled_projects.add(project_name)
            self.logger.info(f"Enabled project: {project_name}")
            return True
        return False

    def disable_project(self, project_name: str) -> bool:
        """Disable a project from routing (runtime control).

        IMPORTANT: This does NOT unload the project. The project's
        isolated context remains in memory. This only removes the
        project from the routing pool.

        Args:
            project_name: Name of project to disable.

        Returns:
            True if disabled, False if project not loaded.
        """
        if self._context_manager.get_context(project_name) is not None:
            self._enabled_projects.discard(project_name)
            self.logger.info(f"Disabled project: {project_name}")
            return True
        return False

    def is_project_enabled(self, project_name: str) -> bool:
        """Check if a project is currently enabled.

        Args:
            project_name: Name of project.

        Returns:
            True if enabled, False otherwise.
        """
        return project_name in self._enabled_projects

    def get_enabled_projects(self) -> List[IsolatedProjectContext]:
        """Get list of currently enabled projects for routing.

        Returns only the subset of loaded projects that are currently
        enabled. All projects remain loaded; this filters to only those
        available for routing.

        Returns:
            List of enabled IsolatedProjectContext objects.
        """
        enabled = []
        for project_name in self._enabled_projects:
            context = self._context_manager.get_context(project_name)
            if context is not None:
                enabled.append(context)
        return enabled

    def get_disabled_projects(self) -> List[str]:
        """Get list of currently disabled project names.

        Returns:
            List of disabled project names.
        """
        all_projects = set(self._context_manager.list_projects())
        return list(all_projects - self._enabled_projects)

    def unload_project(self, project_name: str) -> bool:
        """Unload a project and free resources.

        Args:
            project_name: Name of project to unload.

        Returns:
            True if unloaded, False if not loaded.
        """
        if self._context_manager.remove_context(project_name):
            self._enabled_projects.discard(project_name)
            self.logger.info(f"Unloaded project: {project_name}")
            return True
        return False

    def reload_project(self, project_name: str) -> IsolatedProjectContext:
        """Reload a project (unload and load).

        Args:
            project_name: Name of project to reload.

        Returns:
            Reloaded IsolatedProjectContext.
        """
        self.unload_project(project_name)
        return self.load_project(project_name)

    # Private methods

    def _get_default_search_paths(self) -> List[Path]:
        """Get default project search paths."""
        import os

        paths = []

        # Current directory
        paths.append(Path.cwd())

        # OSPREY_PROJECTS environment variable
        if 'OSPREY_PROJECTS' in os.environ:
            for path_str in os.environ['OSPREY_PROJECTS'].split(':'):
                paths.append(Path(path_str))

        # Parent directory (for monorepo structure)
        paths.append(Path.cwd().parent)

        return [p for p in paths if p.exists()]

    def _parse_project_metadata(self, project_dir: Path, config_path: Path) -> ProjectMetadata:
        """Parse project metadata from config file.

        Args:
            project_dir: Project directory path.
            config_path: Path to config.yml.

        Returns:
            ProjectMetadata instance.
        """
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        project_info = config_data.get('project', {})

        return ProjectMetadata(
            name=project_info.get('name', project_dir.name),
            path=project_dir,
            config_path=config_path,
            description=project_info.get('description', ''),
            version=project_info.get('version', '0.0.0'),
            author=project_info.get('author'),
            tags=project_info.get('tags', [])
        )


# Custom Exceptions

class ProjectDiscoveryError(Exception):
    """Raised when project discovery fails."""
    pass


class ProjectNotFoundError(Exception):
    """Raised when project is not found."""
    pass


class ProjectLoadError(Exception):
    """Raised when project loading fails."""
    pass
