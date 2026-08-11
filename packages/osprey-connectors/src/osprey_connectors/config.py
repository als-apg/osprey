"""Configuration system.

Loads a single YAML config file, resolves environment variables,
and provides dot-path access to values.

**Importing this module does not read ``.env`` and does not touch
``os.environ``.** Building a :class:`ConfigBuilder` does (see its ``load_env``
argument), and that is deliberate — but it must be an *application* that asks
for a config, never the mere act of importing an ``osprey`` module. Loading
``.env`` at import time made a library consumer's environment depend on which
directory the process happened to start in, gave non-credential keys
``override=True`` semantics they were never scoped for, and silently undid
callers' own ``os.environ`` writes depending on import order.

Applications load ``.env`` explicitly at startup: the CLI in
``osprey.cli.main``, MCP servers via :func:`osprey.mcp_env.load_dotenv_from_project`,
and the Claude Code launch paths via
:func:`osprey.build.claude_code_resolver.inject_provider_env`.
"""

import copy
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import yaml

# Safe at module level despite this module's no-osprey-imports rule below:
# ``workspace`` imports nothing from osprey itself, so there is no cycle.
from osprey_connectors.workspace import (
    SIMULATION_STATE_DIR_CONFIG_KEY,
    anchored_path,
    dotted_config_str,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime
    from zoneinfo import ZoneInfo

# Use standard logging (not get_logger) to avoid circular imports with logger.py
# The short name 'CONFIG' enables easy filtering: quiet_logger(['registry', 'CONFIG'])
logger = logging.getLogger("CONFIG")


def resolve_env_vars(data: Any, *, environ: "Mapping[str, str] | None" = None) -> Any:
    """Recursively resolve environment variables in configuration data.

    Supports both simple and bash-style default value syntax:
    - ``${VAR_NAME}`` — simple substitution
    - ``${VAR_NAME:-default_value}`` — with default value
    - ``$VAR_NAME`` — simple substitution without braces

    Placeholders inside ``claude_code.servers.*.env`` blocks are preserved
    verbatim: Claude Code expands ``${VAR}`` in ``.mcp.json`` env at MCP
    server *launch* time, so expanding here would bake either secrets or
    empty strings (when vars are unset, e.g. in CI image builds) into build
    artifacts.

    This is the public, standalone version of ConfigBuilder._resolve_env_vars.
    Use it when you need env-var resolution without a full ConfigBuilder
    instance (e.g., after a raw ``yaml.safe_load``).

    Args:
        data: The config data (dict/list/str/scalar) to resolve.
        environ: Optional mapping to resolve ``${VAR}`` against instead of the
            process environment. Lets callers expand against an
            ``os.environ`` + ``.env`` overlay without mutating global state
            (e.g. the Claude Code provider loader). Defaults to ``os.environ``.
    """
    lookup = os.environ if environ is None else environ
    raw_server_envs: dict[str, dict] = {}
    if isinstance(data, dict):
        cc = data.get("claude_code")
        servers = cc.get("servers") if isinstance(cc, dict) else None
        if isinstance(servers, dict):
            for name, spec in servers.items():
                if isinstance(spec, dict) and isinstance(spec.get("env"), dict):
                    raw_server_envs[name] = copy.deepcopy(spec["env"])

    if isinstance(data, dict):
        resolved: Any = {
            key: resolve_env_vars(value, environ=lookup) for key, value in data.items()
        }
    elif isinstance(data, list):
        return [resolve_env_vars(item, environ=lookup) for item in data]
    elif isinstance(data, str):

        def replace_env_var(match):
            if match.group(1):  # ${VAR_NAME:-default} or ${VAR_NAME}
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) is not None else None
            else:  # $VAR_NAME (simple form)
                var_name = match.group(3)
                default_value = None

            env_value = lookup.get(var_name)
            # Match bash :- semantics: empty string triggers default too
            if env_value is None or (not env_value and default_value is not None):
                if default_value is not None:
                    return default_value
                else:
                    # DEBUG, not INFO: this resolver is generic over the whole
                    # config tree and knows nothing about which of the values it
                    # walks actually matter to the caller. Reported at INFO it
                    # rendered every build as a wall of misses for providers the
                    # project never uses, while saying nothing about the keys
                    # that *did* resolve. User-facing credential reporting lives
                    # in osprey.cli.build_environment.report_provider_credentials,
                    # which knows the selected provider.
                    if not os.environ.get("OSPREY_QUIET"):
                        logger.debug(
                            f"Environment variable '{var_name}' not found, "
                            f"keeping placeholder verbatim"
                        )
                    return match.group(0)
            return env_value

        pattern = r"\$\{([^}:]+)(?::-(.*?))?\}|\$([A-Za-z_][A-Za-z0-9_]*)"
        return re.sub(pattern, replace_env_var, data)
    else:
        return data

    # Restore preserved MCP server env blocks (only meaningful at the top of a
    # config dict; sub-recursions never hit this path because they have no
    # ``claude_code`` key and therefore no snapshot to restore).
    if raw_server_envs:
        cc = resolved.setdefault("claude_code", {})
        servers = cc.setdefault("servers", {}) if isinstance(cc, dict) else None
        if isinstance(servers, dict):
            for name, env in raw_server_envs.items():
                spec = servers.get(name)
                if isinstance(spec, dict):
                    spec["env"] = env
    return resolved


# OSPREY runs agent Python code in exactly one backend: a subprocess on the host.
# ``local`` is the historical name for that same backend; ``container`` named a
# Jupyter kernel gateway OSPREY never shipped.
EXECUTION_METHOD_SUBPROCESS = "subprocess"

# Module-level latch so the ``container`` deprecation is logged once per process
# rather than on every config read (the executor resolves per tool call).
# Tests reset it via ``osprey_connectors.config._container_method_warned = False``.
_container_method_warned = False


#: Directory the build owns end to end: every file under it is rendered from the
#: profile/preset and checksummed into the manifest.
BUILD_OWNED_DATA_DIR = "data"

#: Directory runtime state belongs in — excluded from the manifest checksums and
#: preserved across ``osprey build --force``.
RUNTIME_STATE_DIR = "_agent_data"

#: Config keys whose value names a path something writes to *at run time*.
#: These must stay out of ``data/``: that tree is build-owned and checksummed by
#: :func:`osprey.cli.templates.manifest.calculate_file_checksums`, so a runtime
#: write landing there reads as project drift and is erased by the next
#: ``osprey build --force``.
RUNTIME_WRITE_PATH_KEYS = (
    SIMULATION_STATE_DIR_CONFIG_KEY,
    "services.channel_finder.pipelines.hierarchical.feedback.store_path",
)


def find_runtime_write_paths_under_data(
    config: "Mapping[str, Any]", project_root: Path
) -> list[tuple[str, str]]:
    """Find configured runtime-write paths that resolve inside ``<root>/data/``.

    Args:
        config: Loaded ``config.yml`` mapping.
        project_root: Directory relative paths in the config resolve against.

    Returns:
        ``(config_key, configured_value)`` pairs for every offending key, in
        :data:`RUNTIME_WRITE_PATH_KEYS` order. Empty when the config is clean.
    """
    data_root = (project_root / BUILD_OWNED_DATA_DIR).resolve()
    offenders: list[tuple[str, str]] = []

    for key in RUNTIME_WRITE_PATH_KEYS:
        value = dotted_config_str(config, key)
        if value is None:
            continue

        resolved = anchored_path(value, project_root).resolve()
        if resolved == data_root or data_root in resolved.parents:
            offenders.append((key, value))

    return offenders


def _describe_config_source(source: str | None) -> str:
    """Describe where an execution method came from, for log messages.

    Args:
        source: Caller-supplied description (usually a config file path). When
            ``None``, the active config path is resolved on a best-effort basis.

    Returns:
        str: A human-readable config source, never empty.
    """
    if source:
        return str(source)
    try:
        from osprey_connectors.workspace import resolve_config_path

        return str(resolve_config_path())
    except Exception:  # pragma: no cover - defensive: never fail a config read
        return "config.yml"


def resolve_execution_method(
    config: "Mapping[str, Any] | None", *, source: str | None = None
) -> str:
    """Normalize ``execution.execution_method`` to the backend that actually runs.

    This is the single choke point for the execution vocabulary. Every reader of
    ``execution.execution_method`` goes through it so legacy configs keep loading
    while the config file stops claiming a backend OSPREY does not ship:

    - ``subprocess`` — the honest name; returned as-is.
    - ``local`` — historical name for the same subprocess backend; mapped silently.
    - ``container`` — named a Jupyter kernel gateway that was never shipped, so it
      failed outright at execution time. Mapped to ``subprocess`` with a one-time
      deprecation warning, because that value's behavior genuinely changes.
    - unset (missing, ``None``, or blank) — defaults to ``subprocess``.

    Args:
        config: Full configuration mapping (the one containing the ``execution``
            section), or ``None``. Anything without a usable ``execution`` section
            resolves to the default.
        source: Optional description of where the config came from (typically the
            config file path), used in the deprecation warning. Defaults to the
            active config path.

    Returns:
        str: Always ``"subprocess"`` — the only execution backend OSPREY ships.

    Raises:
        ValueError: If ``execution.execution_method`` is set to any other value.

    Example:
        >>> resolve_execution_method({"execution": {"execution_method": "local"}})
        'subprocess'
    """
    global _container_method_warned

    execution = config.get("execution") if isinstance(config, dict) else None
    raw = execution.get("execution_method") if isinstance(execution, dict) else None

    if raw is None:
        return EXECUTION_METHOD_SUBPROCESS

    if not isinstance(raw, str):
        raise ValueError(
            f"Invalid execution.execution_method: {raw!r} "
            f"(in {_describe_config_source(source)}). Expected 'subprocess'."
        )

    method = raw.strip().lower()

    if method in ("", EXECUTION_METHOD_SUBPROCESS, "local"):
        return EXECUTION_METHOD_SUBPROCESS

    if method == "container":
        if not _container_method_warned:
            _container_method_warned = True
            logger.warning(
                "execution.execution_method: 'container' is deprecated (in %s). "
                "OSPREY ships no containerized Python backend, so this value never "
                "worked; agent Python code now runs via the subprocess backend. "
                "Set execution.execution_method: subprocess.",
                _describe_config_source(source),
            )
        return EXECUTION_METHOD_SUBPROCESS

    raise ValueError(
        f"Invalid execution.execution_method: {raw!r} "
        f"(in {_describe_config_source(source)}). Expected 'subprocess' "
        f"(legacy 'local' and 'container' are also accepted)."
    )


def load_project_dotenv() -> None:
    """Load ``./.env`` into ``os.environ``, overriding existing values.

    This is the ``.env`` → environ passthrough the framework depends on: it
    feeds the ``${VAR}`` references Claude Code expands in ``.mcp.json`` at MCP
    server launch time (``EPICS_CA_ADDR_LIST``, ``PHOEBUS_BRIDGE_URL``,
    ``BLUESKY_*``), and it makes ``.env`` the source of truth for API keys over
    a stale shell export. Every key in the file is passed through, not a
    declared subset — narrowing it would drop Channel Access addressing with no
    error.

    ``override=True`` and the breadth of the copy are exactly why this must be
    called deliberately. Call it from process entry points only; never at
    import time, and never from library code that a host application merely
    imports. Missing file, missing ``python-dotenv``, and an unreadable file
    are all non-fatal.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning("python-dotenv not available, skipping .env file loading")
        return

    dotenv_path = Path.cwd() / ".env"
    try:
        if not dotenv_path.exists():
            logger.debug(f"No .env file found at {dotenv_path}")
            return
        load_dotenv(dotenv_path, override=True)
        logger.debug(f"Loaded .env file from {dotenv_path}")
    except OSError as e:
        # e.g. a 0600 .env owned by another uid mounted into a non-root
        # container (dispatch worker on a uid-mismatched host). Provider
        # env should already be in os.environ by the time config is built,
        # so degrade gracefully instead of crash-looping the process.
        logger.warning(f"Could not read .env file at {dotenv_path}: {e}")


class ConfigBuilder:
    """Loads a YAML config, resolves ``${VAR}`` env-var placeholders, and
    pre-computes a ``configurable`` dict for framework and standalone use.

    Singleton access: use :func:`get_config_builder` or :func:`_get_config`.
    """

    # Sentinel to distinguish "no default given" from "default is None".
    _REQUIRED = object()

    def _require_config(self, path: str, default: Any = _REQUIRED) -> Any:
        """Get config value, raising ValueError if required (no default) and missing.

        When *default* is provided, uses it with a warning. When *default* is
        omitted (_REQUIRED sentinel), raises on missing/None values.
        """
        value = self.get(path)

        if value is None:
            if default is self._REQUIRED:
                # No default provided - this is a required configuration
                raise ValueError(
                    f"Missing required configuration: '{path}'. Set this value in config.yml."
                )
            else:
                # Default provided - use it but warn for visibility
                logger.warning(f"Using default value for '{path}' = {default}. ")
                return default
        return value

    def __init__(self, config_path: str | None = None, *, load_env: bool = True):
        """
        Initialize configuration builder.

        Args:
            config_path: Path to the config.yml file. If None, looks in current directory.
            load_env: When True (the default), load ``.env`` from the current
                working directory into ``os.environ``. This ``.env`` → environ
                passthrough is load-bearing: it feeds the ``${VAR}`` references
                Claude Code expands in ``.mcp.json`` at MCP server launch time.
                Pass False only when the caller has already populated the
                environment (or must not mutate it) and wants a config load with
                no side effects on ``os.environ``.

        Raises:
            FileNotFoundError: If config.yml is not found and no path is provided.
        """
        if load_env:
            load_project_dotenv()

        if config_path is None:
            cwd_config = Path.cwd() / "config.yml"
            if cwd_config.exists():
                config_path = cwd_config
            else:
                raise FileNotFoundError(
                    f"No config.yml found in current directory: {Path.cwd()}\n\n"
                    f"Please run this command from a project directory containing config.yml,\n"
                    f"or set CONFIG_FILE environment variable to point to your config file.\n\n"
                    f"Example: export CONFIG_FILE=/path/to/your/config.yml"
                )

        self.config_path = Path(config_path)

        # Fail fast with an actionable message when the config path is not a
        # regular file. A common cause: a container bind-mount whose host source
        # did not exist at container-create time, which the runtime silently
        # materializes as an empty directory on both ends — so config_path
        # resolves to a directory and the YAML loader would otherwise surface a
        # bare "[Errno 21] Is a directory". Treat a missing explicit path the
        # same way rather than failing later in open().
        if self.config_path.is_dir():
            raise IsADirectoryError(
                f"Config path is a directory, not a file: {self.config_path}\n\n"
                f"config.yml is expected to be a file here. This usually means it "
                f"was meant to be provided at runtime (e.g. a bind-mount whose "
                f"source was missing, so the container runtime created an empty "
                f"directory) or was not baked into the image. Ensure config.yml "
                f"exists as a file at this path."
            )
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n\n"
                f"Set the CONFIG_FILE environment variable to a valid config.yml, "
                f"or run from a project directory that contains one."
            )

        self.raw_config, self._unexpanded_config = self._load_config()

        self._warn_on_runtime_write_paths_under_data()

        # Pre-compute nested structures for efficient runtime access
        self.configurable = self._build_configurable()

    def _warn_on_runtime_write_paths_under_data(self) -> None:
        """Warn when a runtime writer is pointed at the build-owned ``data/`` tree.

        Advisory only: the misconfiguration still works until the next
        ``osprey build --force`` wipes ``data/`` and takes the runtime state
        with it, so this warns rather than raising.
        """
        configured_root = self.raw_config.get("project_root")
        project_root = (
            Path(configured_root) if configured_root else self.config_path.parent
        ).expanduser()

        for key, value in find_runtime_write_paths_under_data(self.raw_config, project_root):
            logger.warning(
                "Runtime-write path '%s' = %r resolves inside the build-owned "
                "'%s/' tree (%s). That directory is re-rendered and checksummed on "
                "every build, so runtime writes there show up as project drift and "
                "are erased by 'osprey build --force'. Point it at '%s/' instead.",
                key,
                value,
                BUILD_OWNED_DATA_DIR,
                self.config_path,
                RUNTIME_STATE_DIR,
            )

    def _load_yaml_file(self, file_path: Path) -> dict[str, Any]:
        """Load and validate a YAML configuration file."""
        try:
            with open(file_path) as f:
                config = yaml.safe_load(f)

            if config is None:
                logger.warning(f"Configuration file is empty: {file_path}")
                return {}

            if not isinstance(config, dict):
                error_msg = f"Configuration file must contain a dictionary/mapping: {file_path}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.debug(f"Loaded configuration from {file_path}")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Error parsing YAML configuration: {e}"
            logger.error(error_msg)
            raise yaml.YAMLError(error_msg) from e

    def _resolve_env_vars(self, data: Any) -> Any:
        """Recursively resolve environment variables in configuration data.

        Delegates to the module-level :func:`resolve_env_vars` function.
        """
        return resolve_env_vars(data)

    def _load_config(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load configuration from single file.

        Returns:
            Tuple of (expanded_config, unexpanded_config) where:
            - expanded_config: Config with ${VAR} placeholders resolved to actual values
            - unexpanded_config: Config with ${VAR} placeholders preserved (for deployment)
        """
        import copy

        config = self._load_yaml_file(self.config_path)

        unexpanded_config = copy.deepcopy(config)

        expanded_config = self._resolve_env_vars(config)

        logger.info(f"Loaded configuration from {self.config_path}")
        return expanded_config, unexpanded_config

    def get_unexpanded_config(self) -> dict[str, Any]:
        """Get configuration with environment variable placeholders preserved.

        Returns the configuration as loaded from YAML, without expanding
        ${VAR_NAME} placeholders. This is useful for deployment scenarios
        where secrets should NOT be written to disk - instead, the placeholders
        are preserved and resolved at container runtime.

        Returns:
            dict: Configuration with ${VAR} placeholders intact
        """
        import copy

        return copy.deepcopy(self._unexpanded_config)

    #: Keys older deployed configs may still carry but nothing honours any more:
    #: ``python_env_path`` (the agent interpreter is resolved at run time) and
    #: ``modes`` (Jupyter-era kernel/gateway descriptions with no reader).
    _RETIRED_EXECUTION_KEYS = ("python_env_path", "modes")

    def _get_execution_config(self) -> dict[str, Any]:
        """Get execution configuration with sensible defaults.

        Retired keys (:attr:`_RETIRED_EXECUTION_KEYS`) are dropped here if an
        older config still carries them, so already-deployed projects keep
        loading unchanged instead of failing on keys nothing honours any more.

        Returns:
            dict: Execution configuration including the execution method.
        """
        # Try to get execution config from file
        execution_config = self.get("execution", None)

        # If execution section exists and has content, use it
        if execution_config:
            if isinstance(execution_config, dict):
                for key in self._RETIRED_EXECUTION_KEYS:
                    if key in execution_config:
                        logger.debug(
                            "Ignoring retired 'execution.%s' (%s) in %s; the key has "
                            "no effect on the subprocess execution backend.",
                            key,
                            execution_config[key],
                            self.config_path,
                        )
                execution_config = {
                    key: value
                    for key, value in execution_config.items()
                    if key not in self._RETIRED_EXECUTION_KEYS
                }
            return execution_config

        logger.debug(
            "'execution' section missing from config.yml; defaulting to subprocess Python execution"
        )

        return {"execution_method": EXECUTION_METHOD_SUBPROCESS}

    def _get_python_executor_config(self) -> dict[str, Any]:
        """Get python executor configuration with sensible defaults.

        Returns python_executor configuration from config.yml if present, otherwise
        provides a reasonable default for the execution timeout.

        Returns:
            dict: Python executor configuration with timeout settings
        """
        # Try to get python_executor config from file
        python_executor_config = self.get("python_executor", None)

        # If python_executor section exists and has content, use it
        if python_executor_config:
            return python_executor_config

        # Otherwise, provide sensible defaults
        return {
            "execution_timeout_seconds": 600,
        }

    def _build_configurable(self) -> dict[str, Any]:
        """Build the configurable dictionary with pre-computed nested structures."""
        configurable = {
            "model_configs": self._build_model_configs(),
            "provider_configs": self._build_provider_configs(),
            "service_configs": self._build_service_configs(),
            "execution": self._get_execution_config(),
            "python_executor": self._get_python_executor_config(),
            "logging": self.get("logging", {}),
            "development": self.get("development", {}),
            "project_root": self.get("project_root"),
            "registry_path": self.get("registry_path"),
            "facility_timezone": self.get("system.timezone", "UTC"),
        }

        return configurable

    def _build_model_configs(self) -> dict[str, Any]:
        """Get model configs from flat structure."""
        return self.get("models", {})

    def _build_provider_configs(self) -> dict[str, Any]:
        """Build provider configs."""
        return self.get("api.providers", {})

    def _build_service_configs(self) -> dict[str, Any]:
        """Get service configs from flat structure."""
        return self.get("services", {})

    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation path."""
        keys = path.split(".")
        value = self.raw_config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default


def load_project_config(config_path: str | Path, *, wrap_errors: bool = False) -> dict[str, Any]:
    """Load a built project's ``config.yml`` the way the deploy path loads it.

    The single entry point every project-facing command and deployment step
    uses to read a project's effective config: :class:`ConfigBuilder`, with the
    loader's own INFO chatter quieted. Going through one function is what keeps
    those consumers agreeing on what the config *says* — a caller that
    hand-rolled ``yaml.safe_load`` would resolve ``${VAR}`` without the ``.env``
    passthrough ConfigBuilder performs (see :func:`load_project_dotenv`), and
    would miss the module flags and derived fields ConfigBuilder computes. It
    could then report on — or render from — a config the deploy would never
    have produced.

    This lives here, next to ``ConfigBuilder``, rather than in a CLI helper
    module: the deployment layer is one of its consumers and imports nothing
    from :mod:`osprey.cli` at module scope, so a CLI home would have made this
    the first such edge.

    Existence checks stay at the call site. ``ConfigBuilder``'s own
    ``FileNotFoundError`` advises setting ``CONFIG_FILE``, which is the wrong
    instruction for a ``--project`` verb, so a caller that needs its own
    "no project here" message must check before calling (see
    :func:`osprey.cli.scaffold_cmd._load_config`).

    Args:
        config_path: Path to the project's ``config.yml``.
        wrap_errors: When True, re-raise any load failure as ``RuntimeError``
            naming the config path. Deployment entry points use this so an
            operator sees which file failed rather than a bare YAML error.

    Returns:
        The env-var-expanded config mapping (``ConfigBuilder.raw_config``).

    Raises:
        RuntimeError: If the load fails and *wrap_errors* is True.
    """
    from osprey_connectors.log_filter import quiet_logger

    try:
        with quiet_logger(["registry", "CONFIG"]):
            config: dict[str, Any] = ConfigBuilder(str(config_path)).raw_config
    except Exception as e:
        if not wrap_errors:
            raise
        raise RuntimeError(f"Could not load config file {config_path}: {e}") from e
    return config


_default_config: ConfigBuilder | None = None
_default_configurable: dict[str, Any] | None = None
_config_cache: dict[str, ConfigBuilder] = {}


def _get_config(config_path: str | None = None, set_as_default: bool = False) -> ConfigBuilder:
    """Get configuration instance (singleton pattern with optional explicit path).

    This function supports two modes:
    1. Default singleton: When no config_path provided, uses CONFIG_FILE env var or cwd/config.yml
    2. Explicit path: When config_path provided, caches and returns config for that specific path

    Args:
        config_path: Optional explicit path to configuration file. If provided,
                    this path is used instead of the default singleton behavior.
        set_as_default: If True and config_path is provided, also set this config as the
                       default singleton so future calls without config_path use it.

    Returns:
        ConfigBuilder instance for the specified or default configuration

    Examples:
        >>> # Default singleton behavior (backward compatible)
        >>> config = _get_config()

        >>> # Explicit config path
        >>> config = _get_config("/path/to/config.yml")

        >>> # Explicit path that becomes the default
        >>> config = _get_config("/path/to/config.yml", set_as_default=True)
    """
    global _default_config, _default_configurable

    if config_path is None:
        if _default_config is None:
            config_file = os.environ.get("CONFIG_FILE")
            if config_file:
                _default_config = ConfigBuilder(config_file)
            else:
                _default_config = ConfigBuilder()

            _default_configurable = _default_config.configurable.copy()

            logger.info("Initialized default configuration system")

        return _default_config

    resolved_path = str(Path(config_path).resolve())

    if resolved_path not in _config_cache:
        # Log honestly: a missing file is about to raise in ConfigBuilder, and
        # many callers swallow that — a cheerful "Loading configuration from
        # explicit path" would then be the only (misleading) trace in the log.
        if Path(resolved_path).is_file():
            logger.info(f"Loading configuration from explicit path: {resolved_path}")
        else:
            logger.warning(f"Config file not found: {resolved_path} — load will fail")
        _config_cache[resolved_path] = ConfigBuilder(resolved_path)

    if set_as_default:
        _default_config = _config_cache[resolved_path]
        _default_configurable = _default_config.configurable.copy()
        logger.debug(f"Set explicit config as default: {resolved_path}")

    return _config_cache[resolved_path]


def _get_configurable(
    config_path: str | None = None, set_as_default: bool = False
) -> dict[str, Any]:
    """Get configurable dict with automatic context detection.

    This function supports both framework execution contexts and standalone execution,
    with optional explicit configuration path support.

    Args:
        config_path: Optional explicit path to configuration file
        set_as_default: If True and config_path is provided, set as default config

    Returns:
        Complete configuration dictionary with all configurable values
    """
    config = _get_config(config_path, set_as_default=set_as_default)

    if config_path is None:
        global _default_configurable
        if _default_configurable is None:
            _default_configurable = config.configurable.copy()
        return _default_configurable

    return config.configurable


def get_config_builder(
    config_path: str | None = None, set_as_default: bool = False
) -> ConfigBuilder:
    """Get configuration builder instance for full config access.

    This is the primary public API for accessing the configuration system.
    Returns a ConfigBuilder instance that provides access to both raw configuration
    data and pre-computed configurable structures.

    Args:
        config_path: Optional explicit path to configuration file. If provided,
                    loads configuration from this path. If None, uses the default
                    singleton (CONFIG_FILE env var or cwd/config.yml).
        set_as_default: If True and config_path is provided, also set this config
                       as the default singleton for future calls without config_path.

    Returns:
        ConfigBuilder instance with access to:
        - .raw_config: The raw YAML configuration dictionary
        - .configurable: Pre-computed configuration for framework
        - .get(path, default): Dot-notation access to config values

    Examples:
        >>> # Default configuration
        >>> config = get_config_builder()
        >>> timeout = config.get("execution.timeout", 30)

        >>> # Load from specific path
        >>> config = get_config_builder("/path/to/config.yml")
        >>> raw = config.raw_config

        >>> # Load and set as default for subsequent calls
        >>> config = get_config_builder("/path/to/config.yml", set_as_default=True)
    """
    return _get_config(config_path, set_as_default)


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load raw configuration dictionary from YAML file.

    Convenience function that returns the raw configuration dictionary
    as loaded from the YAML file, with environment variables resolved.

    Args:
        config_path: Optional path to configuration file. If None, uses the
                    default configuration (CONFIG_FILE env var or cwd/config.yml).

    Returns:
        Raw configuration dictionary with all values from the YAML file.

    Examples:
        >>> # Load default configuration
        >>> config = load_config()
        >>> api_key = config.get("api", {}).get("key")

        >>> # Load from specific path
        >>> config = load_config("/path/to/config.yml")
        >>> channels = config.get("channel_finder", {})
    """
    return _get_config(config_path).raw_config


def get_framework_service_config(
    service_name: str, config_path: str | None = None
) -> dict[str, Any]:
    """Get framework service configuration with automatic context detection.

    Args:
        service_name: Name of the framework service
        config_path: Optional explicit path to configuration file

    Returns:
        Dictionary with service configuration
    """
    configurable = _get_configurable(config_path)
    service_configs = configurable.get("service_configs", {})
    return service_configs.get(service_name, {})


def get_agent_dir(sub_dir: str, host_path: bool = False) -> str:
    """
    Get the target directory path within the agent data directory using absolute paths.

    The root comes from ``agent_data.base_dir`` — the single key naming this
    directory — while ``file_paths`` supplies the subdirectory names. A
    subdirectory key absent from ``file_paths`` falls back to its own name.

    Args:
        sub_dir: Subdirectory name (e.g., 'api_calls_dir', 'registry_exports_dir')
        host_path: If True, force return of host filesystem path even when running in container

    Returns:
        Absolute path to the target directory
    """
    from osprey_connectors.workspace import agent_data_base_dir

    config = _get_config()

    project_root = config.get("project_root")
    main_file_paths = config.get("file_paths", {})
    agent_data_root = agent_data_base_dir(config.raw_config)

    if sub_dir in main_file_paths:
        sub_dir_path = main_file_paths[sub_dir]
        logger.debug(f"Found {sub_dir} in file_paths: {sub_dir_path}")
    else:
        sub_dir_path = sub_dir
        logger.debug(f"Using fallback path for {sub_dir}: {sub_dir_path}")

    if project_root:
        project_root_path = Path(project_root)

        if host_path:
            logger.debug(f"Forcing host path resolution for: {sub_dir}")
            path = project_root_path / agent_data_root / sub_dir_path
        else:
            if not project_root_path.exists():
                container_project_roots = ["/app", "/pipelines", "/jupyter"]
                detected_container_root = None

                for container_root in container_project_roots:
                    container_path = Path(container_root)
                    if container_path.exists() and (container_path / agent_data_root).exists():
                        detected_container_root = container_path
                        break

                if detected_container_root:
                    logger.debug(
                        f"Container environment detected: using {detected_container_root} instead of {project_root}"
                    )
                    path = detected_container_root / agent_data_root / sub_dir_path
                else:
                    logger.warning(f"Configured project root does not exist: {project_root}")
                    logger.warning("Falling back to relative path resolution")
                    path = Path(agent_data_root) / sub_dir_path
                    path = path.resolve()
            else:
                path = project_root_path / agent_data_root / sub_dir_path
    else:
        logger.warning("No project root configured, using relative path for agent data directory")
        path = Path(agent_data_root) / sub_dir_path
        path = path.resolve()

    return str(path)


def get_config_value(path: str, default: Any = None, config_path: str | None = None) -> Any:
    """
    Get a specific configuration value by dot-separated path.

    This function provides context-aware access to configuration values,
    working both inside and outside framework execution contexts. Optionally,
    an explicit configuration file path can be provided.

    Args:
        path: Dot-separated configuration path (e.g., "execution.timeout")
        default: Default value to return if path is not found
        config_path: Optional explicit path to configuration file

    Returns:
        The configuration value at the specified path, or default if not found

    Raises:
        ValueError: If path is empty or None

    Examples:
        >>> timeout = get_config_value("execution.timeout", 30)
        >>> debug_mode = get_config_value("development.debug", False)

        >>> # With explicit config path
        >>> timeout = get_config_value("execution.timeout", 30, "/path/to/config.yml")
    """
    if not path:
        raise ValueError("Configuration path cannot be empty or None")

    configurable = _get_configurable(config_path)

    keys = path.split(".")
    value = configurable

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            config = _get_config(config_path)
            return config.get(path, default)

    return value


_tz_drift_warned = False


def get_facility_timezone() -> "ZoneInfo":
    """Get the facility timezone from config as a ZoneInfo object.

    This is the safe default resolver for every simulation synthesis primitive
    and timestamp render site, so it must never raise: an unloaded config (no
    ``config.yml`` and no ``CONFIG_FILE``) or a misconfigured/typo'd zone name
    degrades to UTC with a logged warning rather than propagating to callers.

    Returns:
        ZoneInfo for the configured facility timezone, defaulting to UTC.
    """
    from zoneinfo import ZoneInfo

    try:
        tz_name = get_config_value("facility_timezone", "UTC")
        zone = ZoneInfo(tz_name)
    except Exception as exc:  # noqa: BLE001 - any failure must degrade to UTC, not raise
        logger.warning(f"Falling back to UTC facility timezone ({type(exc).__name__}: {exc})")
        return ZoneInfo("UTC")

    _warn_on_tz_drift(tz_name)
    return zone


@overload
def localize_facility(dt: "datetime") -> "datetime": ...


@overload
def localize_facility(dt: None) -> None: ...


def localize_facility(dt: "datetime | None") -> "datetime | None":
    """Read a naive datetime as facility-local wall-clock; pass an aware one through.

    A naive datetime is stamped with the facility zone -- never the box ``$TZ``,
    which ``astimezone`` would silently impose. :func:`get_facility_timezone`
    degrades to UTC when the zone is unset or unreadable.

    Args:
        dt: A datetime in either state, or ``None``.

    Returns:
        The same instant, timezone-aware, or ``None`` for ``None`` input.
    """
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=get_facility_timezone())
    return dt


def _warn_on_tz_drift(tz_name: str) -> None:
    """Warn once if the container ``$TZ`` disagrees with an explicit system.timezone.

    Agent-facing timestamps key off ``system.timezone``, not ``$TZ``, so a
    divergence only mis-stamps OS-level container logs — but it signals a
    misconfigured deploy (container clock != agent zone). We surface it as a
    one-time warning rather than enforcing equality: ``system.timezone`` stays the
    single source of truth, and this check never changes the returned value or
    raises. Skips the implicit-UTC default (config absent) so CI/tests with
    ``$TZ`` set but no configured ``system.timezone`` stay quiet.
    """
    global _tz_drift_warned
    if _tz_drift_warned:
        return
    host_tz = os.environ.get("TZ")
    if not host_tz or host_tz == tz_name:
        return
    # Only meaningful when system.timezone was explicitly configured (the alias
    # default would otherwise make every $TZ-set environment look divergent).
    if get_config_value("system.timezone", None) is None:
        return
    _tz_drift_warned = True
    logger.warning(
        f"Container $TZ={host_tz!r} differs from the facility system.timezone={tz_name!r}; "
        f"OS-level log timestamps will differ from agent-reported times. Set both to "
        f"the same zone (system.timezone is authoritative for what the agent reports)."
    )


def to_facility_iso(value: Any) -> "str | None":
    """Render a value as a facility-local ISO-8601 string with explicit offset.

    The single shared timestamp-egress transform for agent- and operator-facing
    output (the ARIEL MCP ``serialize_entry``/``entry_get`` and the ARIEL web
    responses). An aware datetime is converted to the facility zone; a naive
    datetime is treated as facility-local wall-clock via :func:`localize_facility`
    (never the box ``$TZ``); ``None`` passes through; anything else degrades to
    ``str(value)`` so callers never crash on an unexpected shape.
    """
    if value is None:
        return None
    if hasattr(value, "astimezone"):  # a datetime
        # localize_facility stamps the naive case; astimezone converts aware
        # values into the facility zone for display.
        return str(localize_facility(value).astimezone(get_facility_timezone()).isoformat())
    return str(value)


def get_full_configuration(config_path: str | None = None) -> dict[str, Any]:
    """
    Get the complete configuration dictionary.

    This function provides access to the entire configurable dictionary,
    working both inside and outside framework execution contexts. Optionally,
    an explicit configuration file path can be provided.

    When an explicit config_path is provided, it is also set as the default
    configuration so that subsequent config access without explicit path will
    use this configuration.

    Args:
        config_path: Optional explicit path to configuration file. If provided,
                    loads configuration from this path and sets it as the default.

    Returns:
        Complete configuration dictionary with all configurable values

    Examples:
        >>> # Default configuration (backward compatible)
        >>> config = get_full_configuration()
        >>> project_root = config.get("project_root")
        >>> models = config.get("model_configs", {})

        >>> # Explicit configuration path (also becomes default)
        >>> config = get_full_configuration("/path/to/my-config.yml")
        >>> models = config.get("model_configs", {})
        >>> # Subsequent calls without path use this config
        >>> other_value = get_config_value("some.setting")
    """
    set_as_default = config_path is not None
    return _get_configurable(config_path, set_as_default=set_as_default)


# No eager _get_config() here, deliberately. Building the default config at
# import time loaded `.env` into os.environ as a side effect of importing any
# osprey module (osprey/utils/__init__.py imports this one), which is the
# behaviour the module docstring rules out. The first real get_config_value()
# call builds it instead; entry points that need `.env` in the environment
# load it themselves.
