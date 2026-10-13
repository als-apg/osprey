"""Service for managing environment plugins.

This service handles discovery, loading, and validation of environment plugins
registered via the 'tuning_scripts.environments' entry point group.
"""

import functools
import logging
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from osprey.services.optimizer.models.api import EnvironmentInfo
from osprey.services.optimizer.models.protocols import EnvironmentProtocol

from .template_service import TemplateService, TemplateStatus

logger = logging.getLogger(__name__)

# EnvironmentProtocol members a plugin class must expose. issubclass() cannot
# check a Protocol with properties, so get_environment_class verifies these
# attributes by hand.
_REQUIRED_PROTOCOL_MEMBERS = (
    "environment_type",
    "variables",
    "observables",
    "get_variables",
    "set_variables",
)


@functools.cache
def _discovered_environments() -> dict[str, EntryPoint]:
    """Scan the entry-point group for environment plugins (cached: scanned once per process)."""
    logger.info("Discovering environment plugins...")
    environments: dict[str, EntryPoint] = {}

    eps = entry_points(group="tuning_scripts.environments")
    for ep in eps:
        environments[ep.name] = ep
        logger.info("Discovered environment: %s", ep.name)

    return environments


class EnvironmentRegistry:
    """Registry for discovering and loading environment plugins."""

    @classmethod
    def reset(cls) -> None:
        """Drop the cached entry-point scan.  Intended for test isolation."""
        _discovered_environments.cache_clear()

    def discover_environments(self) -> None:
        """Scan for available environments (a no-op once the scan is cached)."""
        _discovered_environments()

    def list_environments(self) -> list[EnvironmentInfo]:
        """Get list of available environments with metadata (plugins + templates).

        A template failure propagates rather than degrading to a plugins-only
        list: to the operator a swallowed error is indistinguishable from
        "those environments don't exist".  An absent template directory is not
        a failure — ``scan_templates`` reports no templates and the list is
        simply plugins-only.
        """
        plugin_envs = list(_discovered_environments())

        results = [
            EnvironmentInfo(
                name=name,
                display_name=name.replace("_", " ").title(),
                description="Standard registered environment plugin",
                valid=True,
                source="plugin",
            )
            for name in plugin_envs
        ]

        for t in TemplateService().scan_templates():
            if t.name in plugin_envs:
                logger.warning(
                    "Template '%s' conflicts with existing plugin. Skipping template.", t.name
                )
                continue
            results.append(self._template_info(t))

        return sorted(results, key=lambda x: x.name)

    @staticmethod
    def _template_info(t: TemplateStatus) -> EnvironmentInfo:
        """Build an EnvironmentInfo for a discovered template (valid or invalid)."""
        if t.valid and t.content:
            return EnvironmentInfo(
                name=t.name,
                display_name=t.content.display_name or t.content.name,
                description=t.content.description,
                valid=True,
                source="template",
            )
        return EnvironmentInfo(
            name=t.name,
            display_name=f"{t.name} (Invalid)",
            description=None,
            valid=False,
            error=t.error,
            source="template",
        )

    def get_template_defaults(self, name: str) -> dict[str, Any] | None:
        """Return the template's config dict, or None when ``name`` is not a template."""
        try:
            return TemplateService().load_template_config(name)
        except FileNotFoundError:
            return None

    def get_environment_class(self, name: str) -> type[EnvironmentProtocol]:
        """Load the environment class for a given name.

        Args:
            name: Environment name

        Returns:
            Environment class implementing EnvironmentProtocol

        Raises:
            KeyError: If environment not found
            TypeError: If environment does not implement protocol
        """
        environments = _discovered_environments()

        if name not in environments:
            raise KeyError(
                f"Environment '{name}' not found. Available plugins: {list(environments)}"
            )

        try:
            env_cls: type[EnvironmentProtocol] = environments[name].load()
            missing = [m for m in _REQUIRED_PROTOCOL_MEMBERS if not hasattr(env_cls, m)]
            if missing:
                raise TypeError(
                    f"Environment class {env_cls!r} for '{name}' does not implement "
                    f"EnvironmentProtocol: missing {missing}"
                )
            return env_cls
        except Exception:
            logger.exception("Failed to load environment '%s'", name)
            raise

    def load_environment(self, name: str, config: dict[str, Any]) -> EnvironmentProtocol:
        """Instantiate an environment.

        If *name* is not a registered plugin but matches a template file,
        the template is loaded and its ``override`` field selects the
        underlying plugin.  Templates may only inherit from plugins (one
        level of indirection); nested template inheritance is not supported.

        Args:
            name: Environment name (plugin or template)
            config: Configuration dictionary for the environment

        Returns:
            Instantiated environment
        """
        if name not in _discovered_environments():
            try:
                template_dir = config.get("template_dir")
                template_service = TemplateService(template_dir=template_dir)
                template_config = template_service.load_template_config(name)

                # Template is the base; caller-supplied config overrides it
                merged_config = {**template_config, **config}

                plugin_name = template_config["override"]
                logger.info(
                    "Loaded environment '%s' from template using plugin '%s'", name, plugin_name
                )

                # Resolve the plugin directly — no further template indirection
                env_cls = self.get_environment_class(plugin_name)
                return env_cls(merged_config)

            except FileNotFoundError as e:
                logger.debug("No template named '%s': %s", name, e)
                # Not a template — fall through to standard plugin lookup (will
                # raise KeyError). Errors from a template that EXISTS (invalid
                # content, unimportable plugin) propagate instead of being
                # masked as "environment not found".

        env_cls = self.get_environment_class(name)
        return env_cls(config)
