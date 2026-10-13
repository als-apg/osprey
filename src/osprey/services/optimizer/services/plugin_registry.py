"""Service for managing environments.

Environments come from two places:

- environment files (YAML) in the environments directory, each served by the
  core ``ConnectorEnvironment``;
- Python plugins registered under the ``osprey.optimizer.environments`` entry
  point group, for facilities that need more than channels (shot gating,
  composite objectives, machine checks).

A name may come from only one of them; a clash is refused at discovery.
"""

import functools
import logging
import os
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any

from osprey.services.optimizer.environments.connector_env import ConnectorEnvironment
from osprey.services.optimizer.environments.spec import EnvironmentFile, load_environment_file
from osprey.services.optimizer.models.api import EnvironmentInfo
from osprey.services.optimizer.models.protocols import EnvironmentProtocol

from .template_service import TemplateService, TemplateStatus

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "osprey.optimizer.environments"

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

    eps = entry_points(group=ENTRY_POINT_GROUP)
    for ep in eps:
        environments[ep.name] = ep
        logger.info("Discovered environment: %s", ep.name)

    return environments


def environments_dir() -> Path:
    """Where environment files live: ``$OPTIMIZER_ENVIRONMENTS_DIR``, else
    ``/app/environments`` (the service image), else ``./environments``."""
    configured = os.environ.get("OPTIMIZER_ENVIRONMENTS_DIR")
    if configured:
        return Path(configured)
    docker_path = Path("/app/environments")
    return docker_path if docker_path.exists() else Path("environments").resolve()


@functools.cache
def _environment_files() -> dict[str, EnvironmentFile]:
    """Load every environment file (cached: once per process).

    One bad file fails discovery rather than vanishing from the list: an
    operator who wrote an environment must not be told it does not exist.
    """
    directory = environments_dir()
    if not directory.exists():
        logger.info("No environments directory at %s", directory)
        return {}
    files: dict[str, EnvironmentFile] = {}
    for path in sorted([*directory.glob("*.yml"), *directory.glob("*.yaml")]):
        env = load_environment_file(path)
        if env.name in files:
            raise ValueError(f"Environment {env.name!r} is declared twice (second in {path})")
        files[env.name] = env
        logger.info("Loaded environment file %s (%s)", env.name, path)
    clash = sorted(set(files) & set(_discovered_environments()))
    if clash:
        raise ValueError(f"Environment name(s) {clash} come from both a file and a plugin")
    return files


class EnvironmentRegistry:
    """Registry for discovering and loading environment plugins."""

    @classmethod
    def reset(cls) -> None:
        """Drop the cached entry-point scan.  Intended for test isolation."""
        _discovered_environments.cache_clear()
        _environment_files.cache_clear()

    def discover_environments(self) -> None:
        """Scan for available environments (a no-op once the scan is cached)."""
        _discovered_environments()
        _environment_files()

    def list_environments(self) -> list[EnvironmentInfo]:
        """Get list of available environments with metadata (plugins + templates).

        A template failure propagates rather than degrading to a plugins-only
        list: to the operator a swallowed error is indistinguishable from
        "those environments don't exist".  An absent template directory is not
        a failure — ``scan_templates`` reports no templates and the list is
        simply plugins-only.
        """
        plugin_envs = [*_discovered_environments(), *_environment_files()]

        results = [
            EnvironmentInfo(
                name=name,
                display_name=name.replace("_", " ").title(),
                description="Standard registered environment plugin",
                valid=True,
                source="plugin",
            )
            for name in _discovered_environments()
        ]
        results += [
            EnvironmentInfo(
                name=name,
                display_name=env.display_name or name.replace("_", " ").title(),
                description=env.description or None,
                valid=True,
                source="file",
            )
            for name, env in _environment_files().items()
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
        if name in _environment_files():
            return ConnectorEnvironment  # type: ignore[return-value]  # structural conformance
        environments = _discovered_environments()

        if name not in environments:
            raise KeyError(
                f"Environment '{name}' not found. Available: "
                f"{[*environments, *_environment_files()]}"
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

        If *name* is not a registered plugin or environment file but matches a
        template file, the template is loaded and its ``override`` field selects
        the underlying environment.  Templates may only inherit from plugins and
        environment files (one level of indirection); nested template
        inheritance is not supported.

        Args:
            name: Environment name (plugin or template)
            config: Configuration dictionary for the environment

        Returns:
            Instantiated environment
        """
        if name not in _discovered_environments() and name not in _environment_files():
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
                return self._instantiate(plugin_name, merged_config)

            except FileNotFoundError as e:
                logger.debug("No template named '%s': %s", name, e)
                # Not a template — fall through to standard plugin lookup (will
                # raise KeyError). Errors from a template that EXISTS (invalid
                # content, unimportable plugin) propagate instead of being
                # masked as "environment not found".

        return self._instantiate(name, config)

    def _instantiate(self, name: str, config: dict[str, Any]) -> EnvironmentProtocol:
        """Build the environment ``name`` (a file or a plugin) from ``config``."""
        spec = _environment_files().get(name)
        if spec is not None:
            return ConnectorEnvironment({**config, "spec": spec})  # type: ignore[return-value]
        return self.get_environment_class(name)(config)
