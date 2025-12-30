"""
Capability Registry for Multi-Project GUI Support

This module provides the CapabilityRegistry class for tracking and indexing
capabilities across all projects with disambiguation support.

Key Features:
- Global index of all capabilities from all projects
- Tracks which project owns each capability
- Handles capability name collisions
- Provides capability lookup with project context
- Generates capability descriptions for LLM routing
"""

from typing import TYPE_CHECKING, Any, Optional

from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.base.capability import Capability

logger = get_logger("capability_registry")


class CapabilityRegistry:
    """
    Global registry for capabilities across all projects.

    Handles:
    - Capability indexing by name and project
    - Name collision detection and resolution
    - Capability metadata queries
    - Capability description generation for LLM routing

    The registry maintains a two-level structure:
    - First level: capability name
    - Second level: project name -> capability instance

    This allows multiple projects to have capabilities with the same name
    while maintaining clear ownership and disambiguation.
    """

    def __init__(self):
        """Initialize capability registry."""
        self.logger = logger
        # Structure: {capability_name: {project_name: capability}}
        self._capabilities: dict[str, dict[str, Capability]] = {}
        # Structure: {project_name: {capability_name: metadata}}
        self._metadata: dict[str, dict[str, Any]] = {}

        self.logger.info("Initialized CapabilityRegistry")

    def register_project_capabilities(
        self,
        project_name: str,
        capabilities: dict[str, "Capability"],
        metadata: dict[str, Any] = None,
    ) -> None:
        """Register all capabilities from a project.

        Args:
            project_name: Name of project.
            capabilities: Dictionary of capability_name -> Capability.
            metadata: Optional dictionary of capability_name -> CapabilityMetadata.
        """
        if metadata is None:
            metadata = {}

        if project_name not in self._metadata:
            self._metadata[project_name] = {}

        for cap_name, capability in capabilities.items():
            if cap_name not in self._capabilities:
                self._capabilities[cap_name] = {}

            self._capabilities[cap_name][project_name] = capability
            self._metadata[project_name][cap_name] = metadata.get(cap_name)

            # Log if name collision (informational - this is expected for framework capabilities)
            if len(self._capabilities[cap_name]) > 1:
                projects = list(self._capabilities[cap_name].keys())
                self.logger.debug(
                    f"⚠️  Capability_Registry: Capability name collision: '{cap_name}' "
                    f"found in projects: {projects}"
                )

        self.logger.info(
            f"Registered {len(capabilities)} capabilities from project: {project_name}"
        )

    def get_capability(self, name: str, project: str = None) -> Optional["Capability"]:
        """Get a capability by name, optionally filtered by project.

        Args:
            name: Capability name.
            project: Optional project name to disambiguate.

        Returns:
            Capability if found, None otherwise.

        Raises:
            AmbiguousCapabilityError: If name is ambiguous and project not specified.
        """
        if name not in self._capabilities:
            return None

        projects_with_cap = self._capabilities[name]

        if project:
            return projects_with_cap.get(project)

        if len(projects_with_cap) > 1:
            raise AmbiguousCapabilityError(
                f"Capability '{name}' found in multiple projects: "
                f"{list(projects_with_cap.keys())}. "
                f"Please specify project name to disambiguate."
            )

        return list(projects_with_cap.values())[0]

    def get_capability_with_project(
        self, name: str, project: str = None
    ) -> tuple[str, "Capability"] | None:
        """Get capability with its project name.

        Args:
            name: Capability name.
            project: Optional project name.

        Returns:
            Tuple of (project_name, capability) if found, None otherwise.

        Raises:
            AmbiguousCapabilityError: If name is ambiguous and project not specified.
        """
        if name not in self._capabilities:
            return None

        projects_with_cap = self._capabilities[name]

        if project:
            cap = projects_with_cap.get(project)
            return (project, cap) if cap else None

        if len(projects_with_cap) > 1:
            raise AmbiguousCapabilityError(
                f"Capability '{name}' found in multiple projects: "
                f"{list(projects_with_cap.keys())}. "
                f"Please specify project name to disambiguate."
            )

        project_name = list(projects_with_cap.keys())[0]
        return (project_name, projects_with_cap[project_name])

    def find_capabilities_by_tag(self, tag: str) -> list[tuple[str, str, "Capability"]]:
        """Find all capabilities with a specific tag.

        Args:
            tag: Tag to search for.

        Returns:
            List of (project_name, capability_name, capability) tuples.
        """
        results = []

        for project_name, caps_metadata in self._metadata.items():
            for cap_name, metadata in caps_metadata.items():
                if metadata and tag in metadata.tags:
                    capability = self._capabilities[cap_name][project_name]
                    results.append((project_name, cap_name, capability))

        self.logger.debug(f"Found {len(results)} capabilities with tag: {tag}")
        return results

    def get_all_capabilities(self) -> dict[str, list[tuple[str, "Capability"]]]:
        """Get all capabilities organized by name.

        Returns:
            Dictionary mapping capability_name to list of (project_name, capability) tuples.
        """
        result = {}

        for cap_name, projects_dict in self._capabilities.items():
            result[cap_name] = [
                (project_name, capability) for project_name, capability in projects_dict.items()
            ]

        return result

    def get_capabilities_by_project(self, project_name: str) -> dict[str, "Capability"]:
        """Get all capabilities for a specific project.

        Args:
            project_name: Name of project.

        Returns:
            Dictionary mapping capability_name to capability.
        """
        result = {}

        for cap_name, projects_dict in self._capabilities.items():
            if project_name in projects_dict:
                result[cap_name] = projects_dict[project_name]

        return result

    def get_capability_description(self, name: str, project: str = None) -> str:
        """Get human-readable description of a capability.

        Args:
            name: Capability name.
            project: Optional project name.

        Returns:
            Description string.
        """
        if name not in self._capabilities:
            return f"Unknown capability: {name}"

        projects_with_cap = self._capabilities[name]

        if project:
            if project not in projects_with_cap:
                return f"Capability '{name}' not found in project '{project}'"

            metadata = self._metadata.get(project, {}).get(name)
            if metadata:
                return f"{name} ({project}): {metadata.description}"
            return f"{name} ({project})"

        if len(projects_with_cap) == 1:
            project_name = list(projects_with_cap.keys())[0]
            metadata = self._metadata.get(project_name, {}).get(name)
            if metadata:
                return f"{name}: {metadata.description}"
            return name

        # Multiple projects
        descriptions = []
        for proj_name in projects_with_cap.keys():
            metadata = self._metadata.get(proj_name, {}).get(name)
            if metadata:
                descriptions.append(f"  - {name} ({proj_name}): {metadata.description}")
            else:
                descriptions.append(f"  - {name} ({proj_name})")

        return f"{name} (available in multiple projects):\n" + "\n".join(descriptions)

    def get_all_capability_descriptions(self) -> str:
        """Get descriptions of all capabilities for LLM routing.

        Returns:
            Formatted string with all capability descriptions.
        """
        descriptions = []

        for cap_name in sorted(self._capabilities.keys()):
            descriptions.append(self.get_capability_description(cap_name))

        return "\n".join(descriptions)

    def has_capability(self, name: str, project: str = None) -> bool:
        """Check if capability exists.

        Args:
            name: Capability name.
            project: Optional project name.

        Returns:
            True if capability exists.
        """
        if name not in self._capabilities:
            return False

        if project:
            return project in self._capabilities[name]

        return True

    def get_capability_count(self) -> int:
        """Get total number of unique capability names."""
        return len(self._capabilities)

    def get_project_count(self) -> int:
        """Get total number of projects with capabilities."""
        return len(self._metadata)

    def get_capability_projects(self, name: str) -> list[str]:
        """Get list of projects that have a specific capability.

        Args:
            name: Capability name.

        Returns:
            List of project names that have this capability.
        """
        if name not in self._capabilities:
            return []

        return list(self._capabilities[name].keys())

    def clear(self) -> None:
        """Clear all registered capabilities."""
        self._capabilities.clear()
        self._metadata.clear()
        self.logger.info("Cleared all capabilities from registry")

    def unregister_project(self, project_name: str) -> bool:
        """Unregister all capabilities from a specific project.

        Args:
            project_name: Name of project to unregister.

        Returns:
            True if project was registered, False otherwise.
        """
        if project_name not in self._metadata:
            return False

        # Remove project's capabilities from the registry
        for cap_name in list(self._capabilities.keys()):
            if project_name in self._capabilities[cap_name]:
                del self._capabilities[cap_name][project_name]

                # Remove capability entry if no projects have it anymore
                if not self._capabilities[cap_name]:
                    del self._capabilities[cap_name]

        # Remove project's metadata
        del self._metadata[project_name]

        self.logger.info(f"Unregistered all capabilities from project: {project_name}")
        return True


# Custom Exceptions


class AmbiguousCapabilityError(Exception):
    """Raised when capability name is ambiguous across projects."""

    pass
