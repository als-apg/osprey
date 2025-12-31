"""
Project Context Manager for Multi-Agent GUI

This module provides isolated context management for multiple projects running
simultaneously in the GUI. Each project maintains completely isolated:
- Configuration (no global pollution)
- Registry (separate component registries)
- Gateway (independent message processing)
- Graph (isolated execution)

The key innovation is context isolation - projects never share state.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.utils.logger import get_logger

logger = get_logger("project_context")


@dataclass
class IsolatedProjectContext:
    """Completely isolated context for a single project.

    This context ensures that all framework components (config, registry, gateway)
    are isolated per-project. No global state is shared between projects.
    """

    project_name: str
    project_path: Path
    config_path: Path

    # Isolated components (created fresh for each project)
    config_builder: Any = None  # ConfigBuilder instance (also accessible as .config)
    registry_manager: Any = None  # RegistryManager instance (also accessible as .registry)
    gateway: Any = None  # Gateway instance
    graph: Any = None  # LangGraph instance

    # Project metadata (for GUI display and routing)
    metadata: Any = None  # ProjectMetadata instance

    # Cached configuration (to avoid repeated file reads)
    _config_dict: dict[str, Any] | None = None

    @property
    def config(self):
        """Alias for config_builder for cleaner API.

        Allows both context.config_builder and context.config to work.
        """
        return self.config_builder

    @property
    def registry(self):
        """Alias for registry_manager for cleaner API.

        Allows both context.registry_manager and context.registry to work.
        """
        return self.registry_manager

    def get_config_value(self, path: str, default: Any = None) -> Any:
        """Get configuration value from THIS project's config only.

        This method ensures configuration isolation by reading from the
        project's own ConfigBuilder, never touching global config state.

        Args:
            path: Dot-separated config path (e.g., "control_system.type")
            default: Default value if not found

        Returns:
            Configuration value from this project's config
        """
        if self.config_builder is None:
            raise RuntimeError(f"Config not initialized for project {self.project_name}")

        return self.config_builder.get(path, default)

    def get_full_config(self) -> dict[str, Any]:
        """Get complete configuration dictionary for THIS project only.

        Returns a copy to prevent external modifications from affecting
        the project's configuration.

        Returns:
            Complete configuration dictionary (copy)
        """
        if self.config_builder is None:
            raise RuntimeError(f"Config not initialized for project {self.project_name}")

        # Return configurable (pre-computed nested structures)
        return self.config_builder.configurable.copy()

    def get_registry(self):
        """Get THIS project's registry manager.

        Returns:
            RegistryManager instance for this project
        """
        if self.registry_manager is None:
            raise RuntimeError(f"Registry not initialized for project {self.project_name}")

        return self.registry_manager

    def get_gateway(self):
        """Get THIS project's gateway.

        Returns:
            Gateway instance for this project
        """
        if self.gateway is None:
            raise RuntimeError(f"Gateway not initialized for project {self.project_name}")

        return self.gateway

    def get_graph(self):
        """Get THIS project's graph.

        Returns:
            LangGraph instance for this project
        """
        if self.graph is None:
            raise RuntimeError(f"Graph not initialized for project {self.project_name}")

        return self.graph

    def is_fully_loaded(self) -> bool:
        """Check if all components are loaded."""
        return all(
            [
                self.config_builder is not None,
                self.registry_manager is not None,
                self.gateway is not None,
                self.graph is not None,
            ]
        )

    def initialize_global_registry(self):
        """Initialize global framework singletons with this project's instances.

        This ensures CLI operations work by setting global singletons to point
        to this project's isolated instances. This synchronizes:
        1. Registry singleton - for capability/component access
        2. Config singleton - for configuration access

        Note: Other singletons (approval_manager, data_source_manager, memory_storage_manager)
        are created on-demand from the registry/config, so they don't need explicit syncing.

        This is safe because:
        - Each project has its own isolated instances
        - The global singletons are updated each time a project is used
        - CLI operations will use whichever project was last active
        - The GUI processes one query at a time (no parallel execution)
        """
        if self.registry_manager is None:
            logger.warning(
                f"Cannot initialize global singletons - {self.project_name} registry not loaded"
            )
            return

        try:
            # 1. Sync global registry singleton
            from osprey.registry import manager as registry_module

            registry_module._registry = self.registry_manager
            registry_module._registry_config_path = str(self.config_path)
            logger.debug(f"✓ Synced global registry with {self.project_name}")

            # 2. Sync global config singleton
            import osprey.utils.config as config_module

            config_module._default_config = self.config_builder
            config_module._default_configurable = self.config_builder.configurable.copy()
            logger.debug(f"✓ Synced global config with {self.project_name}")

            # Note: Other singletons are created on-demand and will use the synced config/registry:
            # - approval_manager: Created via get_approval_manager() using current config
            # - data_source_manager: Created via get_data_source_manager() using current registry
            # - memory_storage_manager: Created via get_memory_storage_manager() using current config

            logger.info(f"✅ Initialized global singletons for {self.project_name}")

        except Exception as e:
            logger.error(f"Failed to initialize global singletons for {self.project_name}: {e}")


class ProjectContextManager:
    """Manages isolated contexts for multiple projects.

    This manager ensures complete isolation between projects by:
    1. Creating separate ConfigBuilder instances (no global config pollution)
    2. Creating separate RegistryManager instances (no registry merging)
    3. Creating separate Gateway instances (independent message processing)
    4. Creating separate Graph instances (isolated execution)

    Key Design Principles:
    - NO global state sharing between projects
    - NO modification of global singletons
    - Each project is completely self-contained
    - Projects can be loaded/unloaded independently
    """

    def __init__(self):
        """Initialize the project context manager."""
        self.logger = logger
        self._contexts: dict[str, IsolatedProjectContext] = {}
        self.logger.info("Initialized ProjectContextManager for isolated multi-project support")

    def create_project_context(
        self, project_name: str, project_path: Path, config_path: Path
    ) -> IsolatedProjectContext:
        """Create an isolated context for a project.

        This method creates a completely isolated context with its own:
        - ConfigBuilder (reads from project's config.yml)
        - RegistryManager (loads project's registry.py)
        - Gateway (uses project's config)
        - Graph (uses project's registry and capabilities)

        CRITICAL: This does NOT modify any global state. Each project
        gets its own isolated instances of all framework components.

        Args:
            project_name: Unique name for the project
            project_path: Path to project directory
            config_path: Path to project's config.yml

        Returns:
            IsolatedProjectContext with all components initialized

        Raises:
            ValueError: If project already exists
            RuntimeError: If initialization fails
        """
        if project_name in self._contexts:
            raise ValueError(f"Project context already exists: {project_name}")

        self.logger.info(f"Creating isolated context for project: {project_name}")

        try:
            # Create isolated context
            context = IsolatedProjectContext(
                project_name=project_name, project_path=project_path, config_path=config_path
            )

            # Step 1: Create isolated ConfigBuilder
            import osprey.utils.config as config_module

            # Save current default config to restore later (maintain isolation)
            saved_default_config = config_module._default_config
            saved_default_configurable = config_module._default_configurable

            try:
                # CRITICAL: Temporarily set this project's config as default during initialization
                # This ensures that any calls to get_agent_dir() or other config utilities
                # during registry initialization will use THIS project's config
                context.config_builder = config_module._get_config(
                    str(config_path), set_as_default=True
                )
                self.logger.debug(f"Created isolated ConfigBuilder for {project_name}")

                # Step 2: Validate and fix project_root if needed
                self._ensure_correct_project_root(context)

                # Step 3: Create isolated RegistryManager
                # This creates a NEW registry instance, NOT using the global singleton
                from osprey.registry.manager import RegistryManager

                registry_path = context.config_builder.get("registry_path")
                if registry_path:
                    # Resolve relative to project directory
                    if not Path(registry_path).is_absolute():
                        registry_file = project_path / registry_path
                    else:
                        registry_file = Path(registry_path)

                    context.registry_manager = RegistryManager(registry_path=str(registry_file))
                else:
                    # Framework-only registry
                    context.registry_manager = RegistryManager(registry_path=None)

                # Initialize the registry (loads all capabilities)
                # This may call get_agent_dir() which will use the temporarily set default config
                context.registry_manager.initialize()
                self.logger.debug(
                    f"Created isolated RegistryManager for {project_name} "
                    f"with {len(context.registry_manager.get_all_capabilities())} capabilities"
                )

            finally:
                # Restore previous default config to maintain isolation between projects
                config_module._default_config = saved_default_config
                config_module._default_configurable = saved_default_configurable
                self.logger.debug(f"Restored previous default config after loading {project_name}")

            # Step 4: Create isolated Gateway
            # Gateway uses the project's config dictionary
            from osprey.infrastructure.gateway import Gateway

            context.gateway = Gateway(config=context.config_builder.raw_config)
            self.logger.debug(f"Created isolated Gateway for {project_name}")

            # Step 5: Create isolated Graph
            # Graph uses the project's registry and capabilities
            from langgraph.checkpoint.memory import MemorySaver

            from osprey.graph import create_graph

            checkpointer = MemorySaver()
            context.graph = create_graph(context.registry_manager, checkpointer=checkpointer)
            self.logger.debug(f"Created isolated Graph for {project_name}")

            # Store context
            self._contexts[project_name] = context

            self.logger.info(
                f"Successfully created isolated context for {project_name} "
                f"({len(context.registry_manager.get_all_capabilities())} capabilities)"
            )

            return context

        except Exception as e:
            self.logger.error(f"Failed to create context for {project_name}: {e}")
            raise RuntimeError(f"Failed to create project context: {e}") from e

    def get_context(self, project_name: str) -> IsolatedProjectContext | None:
        """Get the isolated context for a project.

        Args:
            project_name: Name of the project

        Returns:
            IsolatedProjectContext if exists, None otherwise
        """
        return self._contexts.get(project_name)

    def remove_context(self, project_name: str) -> bool:
        """Remove a project context and free resources.

        Args:
            project_name: Name of the project

        Returns:
            True if removed, False if not found
        """
        if project_name in self._contexts:
            del self._contexts[project_name]
            self.logger.info(f"Removed context for project: {project_name}")
            return True
        return False

    def list_projects(self) -> list[str]:
        """Get list of all project names with contexts.

        Returns:
            List of project names
        """
        return list(self._contexts.keys())

    def get_all_contexts(self) -> dict[str, IsolatedProjectContext]:
        """Get all project contexts.

        Returns:
            Dictionary mapping project names to contexts
        """
        return self._contexts.copy()

    def _ensure_correct_project_root(self, context: IsolatedProjectContext) -> None:
        """Ensure project_root in config matches actual project location.

        This fixes issues when projects are moved on the filesystem.

        IMPORTANT: This method NO LONGER writes to config.yml files.
        Instead, it only updates the in-memory configuration to use the
        correct runtime project path.

        Args:
            context: Project context to validate
        """
        runtime_project_root = str(context.project_path.resolve())
        config_project_root = context.config_builder.raw_config.get("project_root")

        # Normalize for comparison
        config_project_root_normalized = (
            str(Path(config_project_root).resolve()) if config_project_root else None
        )

        # Update in-memory config if missing or incorrect
        if not config_project_root or config_project_root_normalized != runtime_project_root:
            if config_project_root:
                self.logger.info(
                    f"Runtime override for {context.project_name} project_root:\n"
                    f"  Config: {config_project_root}\n"
                    f"  Runtime: {runtime_project_root}"
                )
            else:
                self.logger.info(
                    f"Setting runtime project_root for {context.project_name}: "
                    f"{runtime_project_root}"
                )

            # Update ONLY in-memory config (DO NOT write to file)
            context.config_builder.raw_config["project_root"] = runtime_project_root

            # Rebuild configurable with corrected project_root
            context.config_builder.configurable = context.config_builder._build_configurable()

            self.logger.debug(
                f"Updated in-memory project_root for {context.project_name} "
                f"(config.yml NOT modified)"
            )
