"""Service for managing environments.

Environments come from two places:

- environment files (YAML) in the environments directory, each served by the
  core ``ConnectorEnvironment``;
- Python plugins registered under the ``osprey.optimizer.environments`` entry
  point group, for facilities that need more than channels (shot gating,
  composite objectives, machine checks).

Plugins are opt-in: only the entry points named in ``$OPTIMIZER_PLUGINS``
(comma-separated) are loaded, and with it unset the service serves environment
files only. A plugin implements ``set_variables`` itself, so it must reach the
machine through
:class:`~osprey_optimizer.environments.interface.ConnectorInterface`, whose
every write is checked. Nothing yet enforces that for plugin code: enabling a
plugin trusts it.

A name may come from only one source; a clash between a file and an enabled
plugin is refused at discovery.

Either kind may offer presets, named run setups a draft starts from: an
environment file under ``presets:``, a plugin class as ``PRESETS``, a mapping of
preset name to :class:`~osprey_optimizer.models.config.Preset` (or its
dict form).
"""

import functools
import logging
import os
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any

from osprey_optimizer.environments.connector_env import ConnectorEnvironment
from osprey_optimizer.environments.spec import EnvironmentFile, load_environment_file
from osprey_optimizer.models.api import EnvironmentInfo
from osprey_optimizer.models.config import Preset
from osprey_optimizer.models.protocols import EnvironmentProtocol

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
    "get_setpoints",
    "set_variables",
)


def enabled_plugins() -> set[str]:
    """The plugin names ``$OPTIMIZER_PLUGINS`` enables (comma-separated); none when unset."""
    configured = os.environ.get("OPTIMIZER_PLUGINS", "")
    return {name.strip() for name in configured.split(",") if name.strip()}


@functools.cache
def _discovered_environments() -> dict[str, EntryPoint]:
    """Scan the entry-point group for enabled plugins (cached: scanned once per process).

    An enabled name with no installed entry point fails discovery, as a bad
    environment file does.
    """
    logger.info("Discovering environment plugins...")
    enabled = enabled_plugins()
    environments: dict[str, EntryPoint] = {}
    skipped: list[str] = []

    eps = entry_points(group=ENTRY_POINT_GROUP)
    for ep in eps:
        if ep.name not in enabled:
            skipped.append(ep.name)
            continue
        environments[ep.name] = ep
        logger.info("Discovered environment: %s", ep.name)

    if skipped:
        logger.info("Skipped plugin(s) not named in OPTIMIZER_PLUGINS: %s", sorted(skipped))
    missing = sorted(enabled - set(environments))
    if missing:
        raise ValueError(f"OPTIMIZER_PLUGINS names plugin(s) {missing} that are not installed")
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
        """Every environment, plugins and files, sorted by name."""
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
        return sorted(results, key=lambda x: x.name)

    def get_presets(self, name: str) -> dict[str, Preset]:
        """The presets environment ``name`` offers, possibly none.

        Raises:
            KeyError: ``name`` is not an environment.
            ValueError: a plugin's ``PRESETS`` does not validate.
        """
        spec = _environment_files().get(name)
        if spec is not None:
            return dict(spec.presets)
        declared = getattr(self.get_environment_class(name), "PRESETS", None) or {}
        try:
            return {key: Preset.model_validate(value) for key, value in declared.items()}
        except ValueError as e:
            raise ValueError(f"Environment {name!r} declares an invalid preset: {e}") from e

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
        """Instantiate environment ``name`` (a file or a plugin) from ``config``."""
        spec = _environment_files().get(name)
        if spec is not None:
            return ConnectorEnvironment({**config, "spec": spec})  # type: ignore[return-value]
        return self.get_environment_class(name)(config)
