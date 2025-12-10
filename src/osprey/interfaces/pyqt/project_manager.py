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
from typing import Dict, List, Optional, Set
import logging
import yaml

from osprey.utils.logger import get_logger
from osprey.utils.config import ConfigBuilder
from osprey.infrastructure.gateway import Gateway

logger = get_logger("project_manager")


@dataclass
class ProjectMetadata:
    """Metadata about a discovered project."""
    name: str
    path: Path
    config_path: Path
    description: str
    version: str
    author: Optional[str] = None
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


@dataclass
class ProjectContext:
    """Complete context for a loaded project.
    
    Each project maintains its own isolated Gateway, Registry, and Graph.
    This ensures no registry merging or capability conflicts between projects.
    """
    metadata: ProjectMetadata
    gateway: Gateway  # from osprey.infrastructure.gateway
    config: ConfigBuilder  # from osprey.utils.config
    graph: any = None  # Project-specific graph with its own registry/capabilities
    registry: any = None  # Project-specific registry
    
    def is_loaded(self) -> bool:
        """Check if project is fully loaded."""
        return all([
            self.gateway is not None,
            self.config is not None,
            self.graph is not None,
            self.registry is not None
        ])


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
        self._projects: Dict[str, ProjectContext] = {}
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
    
    def load_project(self, project_name: str) -> ProjectContext:
        """Load a project and create its context.
        
        IMPORTANT: This loads the project into memory with its own
        Gateway, Registry, and ContextManager. The project remains
        loaded even if later disabled. Disabling only affects routing,
        not the loaded state.
        
        Args:
            project_name: Name of project to load.
            
        Returns:
            ProjectContext with loaded gateway and managers.
            
        Raises:
            ProjectNotFoundError: If project not found.
            ProjectLoadError: If loading fails.
        """
        # Check if already loaded
        if project_name in self._projects:
            self.logger.debug(f"Project already loaded: {project_name}")
            return self._projects[project_name]
        
        # Get metadata
        if project_name not in self._metadata_cache:
            self.discover_projects()
        
        if project_name not in self._metadata_cache:
            raise ProjectNotFoundError(f"Project not found: {project_name}")
        
        metadata = self._metadata_cache[project_name]
        
        try:
            self.logger.info(f"Loading project: {project_name}")
            
            # Load configuration using ConfigBuilder
            config = ConfigBuilder(str(metadata.config_path))
            
            # CRITICAL: Check if project_root needs correction
            # This ensures the project works regardless of where it's moved on the filesystem
            runtime_project_root = str(metadata.path)
            config_project_root = config.raw_config.get('project_root')
            
            # Only update config file if project_root is missing or incorrect
            if not config_project_root or config_project_root != runtime_project_root:
                if config_project_root:
                    self.logger.info(
                        f"Updating project_root in {project_name}/config.yml:\n"
                        f"  Old: {config_project_root}\n"
                        f"  New: {runtime_project_root}"
                    )
                else:
                    self.logger.info(f"Adding missing project_root to {project_name}/config.yml: {runtime_project_root}")
                
                # Update the config
                config.raw_config['project_root'] = runtime_project_root
                
                # Write back to the config file to fix it permanently
                try:
                    with open(metadata.config_path, 'w') as f:
                        yaml.dump(config.raw_config, f, default_flow_style=False, sort_keys=False)
                    self.logger.debug(f"Updated config file: {metadata.config_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to update config file {metadata.config_path}: {e}")
                    # Continue anyway - the in-memory config is correct
            else:
                self.logger.debug(f"project_root already correct for {project_name}: {runtime_project_root}")
            
            # Initialize the project's own registry and graph
            # Each project gets its own isolated registry with its own capabilities
            from osprey.registry import initialize_registry, get_registry
            from osprey.graph import create_graph
            from langgraph.checkpoint.memory import MemorySaver
            
            # Initialize registry - it will now read the correct project_root from config
            initialize_registry(config_path=str(metadata.config_path))
            project_registry = get_registry(config_path=str(metadata.config_path))
            
            # Create graph with this project's registry and capabilities
            checkpointer = MemorySaver()
            project_graph = create_graph(project_registry, checkpointer=checkpointer)
            
            # Create gateway with project's config
            gateway = Gateway(config=config.raw_config)
            
            # Create project context with its own graph and registry
            context = ProjectContext(
                metadata=metadata,
                gateway=gateway,
                config=config,
                graph=project_graph,
                registry=project_registry
            )
            
            # Cache it
            self._projects[project_name] = context
            
            # Enable by default (all agents start enabled)
            self._enabled_projects.add(project_name)
            
            self.logger.info(f"Successfully loaded project: {project_name} with {len(project_registry.get_all_capabilities())} capabilities")
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to load project {project_name}: {e}")
            raise ProjectLoadError(f"Failed to load project {project_name}") from e
    
    def get_project(self, project_name: str) -> Optional[ProjectContext]:
        """Get a loaded project context.
        
        Args:
            project_name: Name of project.
            
        Returns:
            ProjectContext if loaded, None otherwise.
        """
        return self._projects.get(project_name)
    
    def list_loaded_projects(self) -> List[str]:
        """Get names of all loaded projects."""
        return list(self._projects.keys())
    
    def list_available_projects(self) -> List[ProjectMetadata]:
        """Get metadata for all available projects."""
        if not self._metadata_cache:
            self.discover_projects()
        return list(self._metadata_cache.values())
    
    def get_project_capabilities(self, project_name: str) -> Dict[str, CapabilityMetadata]:
        """Get capabilities for a project by loading its registry configuration.
        
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
        
        # Extract capabilities from the project's registry configuration
        try:
            import importlib.util
            import sys
            from pathlib import Path
            from osprey.registry.base import RegistryConfigProvider
            
            # Get registry_path from config
            registry_path = context.config.raw_config.get('registry_path')
            
            if not registry_path:
                self.logger.warning(f"No registry_path in config for {project_name}")
                return capabilities
            
            # Resolve registry path relative to project directory
            if not Path(registry_path).is_absolute():
                registry_file = context.metadata.path / registry_path
            else:
                registry_file = Path(registry_path)
            
            if not registry_file.exists():
                self.logger.warning(f"Registry file not found for {project_name}: {registry_file}")
                return capabilities
            
            # Load the registry module dynamically
            spec = importlib.util.spec_from_file_location(
                f"_project_registry_{project_name}",
                registry_file
            )
            
            if spec is None or spec.loader is None:
                self.logger.warning(f"Could not create module spec for {registry_file}")
                return capabilities
            
            # Add project's src directory to sys.path if needed
            project_src = context.metadata.path / "src"
            if project_src.exists() and str(project_src) not in sys.path:
                sys.path.insert(0, str(project_src))
                self.logger.debug(f"Added {project_src} to sys.path for {project_name}")
            
            # Load the module
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"_project_registry_{project_name}"] = module
            spec.loader.exec_module(module)
            
            # Find the RegistryConfigProvider class
            provider_class = None
            for name in dir(module):
                obj = getattr(module, name)
                if (isinstance(obj, type) and
                    issubclass(obj, RegistryConfigProvider) and
                    obj != RegistryConfigProvider):
                    provider_class = obj
                    break
            
            if provider_class is None:
                self.logger.warning(f"No RegistryConfigProvider found in {registry_file}")
                return capabilities
            
            # Instantiate and get registry config
            provider = provider_class()
            registry_config = provider.get_registry_config()
            
            # Extract capability metadata from registry config
            for cap_reg in registry_config.capabilities:
                capabilities[cap_reg.name] = CapabilityMetadata(
                    name=cap_reg.name,
                    project=project_name,
                    description=cap_reg.description,
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
        if project_name in self._projects:
            self._enabled_projects.add(project_name)
            self.logger.info(f"Enabled project: {project_name}")
            return True
        return False
    
    def disable_project(self, project_name: str) -> bool:
        """Disable a project from routing (runtime control).
        
        IMPORTANT: This does NOT unload the project. The project's
        Gateway, Registry, and ContextManager remain in memory.
        This only removes the project from the routing pool.
        
        Args:
            project_name: Name of project to disable.
            
        Returns:
            True if disabled, False if project not loaded.
        """
        if project_name in self._projects:
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
    
    def get_enabled_projects(self) -> List[ProjectContext]:
        """Get list of currently enabled projects for routing.
        
        Returns only the subset of loaded projects that are currently
        enabled. All projects remain loaded; this filters to only those
        available for routing.
        
        Returns:
            List of enabled ProjectContext objects.
        """
        return [
            context for name, context in self._projects.items()
            if name in self._enabled_projects
        ]
    
    def get_disabled_projects(self) -> List[str]:
        """Get list of currently disabled project names.
        
        Returns:
            List of disabled project names.
        """
        return [
            name for name in self._projects.keys()
            if name not in self._enabled_projects
        ]
    
    def unload_project(self, project_name: str) -> bool:
        """Unload a project and free resources.
        
        Args:
            project_name: Name of project to unload.
            
        Returns:
            True if unloaded, False if not loaded.
        """
        if project_name not in self._projects:
            return False
        
        context = self._projects.pop(project_name)
        self._enabled_projects.discard(project_name)
        self.logger.info(f"Unloaded project: {project_name}")
        return True
    
    def reload_project(self, project_name: str) -> ProjectContext:
        """Reload a project (unload and load).
        
        Args:
            project_name: Name of project to reload.
            
        Returns:
            Reloaded ProjectContext.
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