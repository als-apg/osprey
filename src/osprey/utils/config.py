"""Configuration system.

Loads a single YAML config file, resolves environment variables,
and provides dot-path access to values.
"""

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

# Use standard logging (not get_logger) to avoid circular imports with logger.py
# The short name 'CONFIG' enables easy filtering: quiet_logger(['registry', 'CONFIG'])
logger = logging.getLogger("CONFIG")


def resolve_env_vars(data: Any) -> Any:
    """Recursively resolve environment variables in configuration data.

    Supports both simple and bash-style default value syntax:
    - ${VAR_NAME} - simple substitution
    - ${VAR_NAME:-default_value} - with default value
    - $VAR_NAME - simple substitution without braces

    This is the public, standalone version of ConfigBuilder._resolve_env_vars.
    Use it when you need env-var resolution without a full ConfigBuilder instance
    (e.g., after a raw yaml.safe_load).
    """
    if isinstance(data, dict):
        return {key: resolve_env_vars(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [resolve_env_vars(item) for item in data]
    elif isinstance(data, str):

        def replace_env_var(match):
            if match.group(1):  # ${VAR_NAME:-default} or ${VAR_NAME}
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) is not None else None
            else:  # $VAR_NAME (simple form)
                var_name = match.group(3)
                default_value = None

            env_value = os.environ.get(var_name)
            # Match bash :- semantics: empty string triggers default too
            if env_value is None or (not env_value and default_value is not None):
                if default_value is not None:
                    return default_value
                else:
                    if not os.environ.get("OSPREY_QUIET"):
                        logger.info(
                            f"Environment variable '{var_name}' not found, keeping original value"
                        )
                    return match.group(0)
            return env_value

        pattern = r"\$\{([^}:]+)(?::-(.*?))?\}|\$([A-Za-z_][A-Za-z0-9_]*)"
        return re.sub(pattern, replace_env_var, data)
    else:
        return data


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

    def __init__(self, config_path: str | None = None):
        """
        Initialize configuration builder.

        Args:
            config_path: Path to the config.yml file. If None, looks in current directory.

        Raises:
            FileNotFoundError: If config.yml is not found and no path is provided.
        """
        try:
            from dotenv import load_dotenv

            dotenv_path = Path.cwd() / ".env"
            if dotenv_path.exists():
                load_dotenv(dotenv_path, override=True)  # .env file is source of truth for API keys
                logger.debug(f"Loaded .env file from {dotenv_path}")
            else:
                logger.debug(f"No .env file found at {dotenv_path}")
        except ImportError:
            logger.warning("python-dotenv not available, skipping .env file loading")

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
        self.raw_config, self._unexpanded_config = self._load_config()

        # Pre-compute nested structures for efficient runtime access
        self.configurable = self._build_configurable()

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

    def _get_execution_config(self) -> dict[str, Any]:
        """Get execution configuration with sensible defaults.

        Returns execution configuration from config.yml if present, otherwise provides
        defaults suitable for local Python execution in tutorial environments.

        Returns:
            dict: Execution configuration including method, python_env_path, and modes
        """
        # Try to get execution config from file
        execution_config = self.get("execution", None)

        # If execution section exists and has content, use it
        if execution_config:
            return execution_config

        logger.warning(
            "'execution' section missing from config.yml; defaulting to local Python execution"
        )

        return {
            "execution_method": "local",
            "python_env_path": sys.executable,
            "modes": {
                "read_only": {
                    "kernel_name": "python3-readonly",
                    "gateway": "read_only",
                    "allows_writes": False,
                    "environment": {},
                },
                "write_access": {
                    "kernel_name": "python3-write",
                    "gateway": "write_access",
                    "allows_writes": True,
                    "requires_approval": True,
                    "environment": {},
                },
            },
        }

    def _get_writes_enabled_with_fallback(self) -> bool:
        """Get control system writes_enabled setting.

        Returns:
            bool: Whether control system writes are enabled
        """
        return self.get("control_system.writes_enabled", False)

    def _get_python_executor_config(self) -> dict[str, Any]:
        """Get python executor configuration with sensible defaults.

        Returns python_executor configuration from config.yml if present, otherwise
        provides reasonable defaults for retry and timeout settings.

        Returns:
            dict: Python executor configuration with retry and timeout settings
        """
        # Try to get python_executor config from file
        python_executor_config = self.get("python_executor", None)

        # If python_executor section exists and has content, use it
        if python_executor_config:
            return python_executor_config

        # Otherwise, provide sensible defaults
        return {
            "max_generation_retries": 3,
            "max_execution_retries": 3,
            "execution_timeout_seconds": 600,
        }

    def _build_configurable(self) -> dict[str, Any]:
        """Build the configurable dictionary with pre-computed nested structures."""
        configurable = {
            "user_id": None,
            "chat_id": None,
            "session_id": None,
            "thread_id": None,
            "session_url": None,
            "agent_control_defaults": self._build_agent_control_defaults(),
            "model_configs": self._build_model_configs(),
            "provider_configs": self._build_provider_configs(),
            "service_configs": self._build_service_configs(),
            "execution": self._get_execution_config(),
            "python_executor": self._get_python_executor_config(),
            "logging": self.get("logging", {}),
            "development": self.get("development", {}),
            "project_root": self.get("project_root"),
            "applications": self.get("applications", []),
            "current_application": self._get_current_application(),
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

    def _build_agent_control_defaults(self) -> dict[str, Any]:
        """Build agent control defaults with explicit configuration control."""

        return {
            # Control system writes control (with backward compatibility)
            "epics_writes_enabled": self._get_writes_enabled_with_fallback(),
            "control_system_writes_enabled": self._get_writes_enabled_with_fallback(),
        }

    def _get_current_application(self) -> str | None:
        """Get the current/primary application name."""
        applications = self.get("applications", [])
        if isinstance(applications, dict) and applications:
            return list(applications.keys())[0]
        elif isinstance(applications, list) and applications:
            return applications[0]
        return None

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
        logger.info(f"Loading configuration from explicit path: {resolved_path}")
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


def get_current_application() -> str | None:
    """Get current application with automatic context detection."""
    configurable = _get_configurable()
    return configurable.get("current_application")


def get_agent_dir(sub_dir: str, host_path: bool = False) -> str:
    """
    Get the target directory path within the agent data directory using absolute paths.

    Args:
        sub_dir: Subdirectory name (e.g., 'user_memory_dir', 'execution_plans_dir')
        host_path: If True, force return of host filesystem path even when running in container

    Returns:
        Absolute path to the target directory
    """
    config = _get_config()

    project_root = config.get("project_root")
    main_file_paths = config.get("file_paths", {})
    agent_data_dir = main_file_paths.get("agent_data_dir", "_agent_data")

    current_app = get_current_application()
    sub_dir_path = None

    if sub_dir in main_file_paths:
        sub_dir_path = main_file_paths[sub_dir]
        logger.debug(f"Found {sub_dir} in main file_paths: {sub_dir_path}")

    if current_app:
        app_file_paths = config.get(f"applications.{current_app}.file_paths", {})
        if sub_dir in app_file_paths:
            sub_dir_path = app_file_paths[sub_dir]
            logger.debug(f"Found {sub_dir} in {current_app} file_paths: {sub_dir_path}")

    if sub_dir_path is None:
        sub_dir_path = sub_dir
        logger.debug(f"Using fallback path for {sub_dir}: {sub_dir_path}")

    if project_root:
        project_root_path = Path(project_root)

        if host_path:
            logger.debug(f"Forcing host path resolution for: {sub_dir}")
            path = project_root_path / agent_data_dir / sub_dir_path
        else:
            if not project_root_path.exists():
                container_project_roots = ["/app", "/pipelines", "/jupyter"]
                detected_container_root = None

                for container_root in container_project_roots:
                    container_path = Path(container_root)
                    if container_path.exists() and (container_path / agent_data_dir).exists():
                        detected_container_root = container_path
                        break

                if detected_container_root:
                    logger.debug(
                        f"Container environment detected: using {detected_container_root} instead of {project_root}"
                    )
                    path = detected_container_root / agent_data_dir / sub_dir_path
                else:
                    logger.warning(f"Configured project root does not exist: {project_root}")
                    logger.warning("Falling back to relative path resolution")
                    path = Path(agent_data_dir) / sub_dir_path
                    path = path.resolve()
            else:
                path = project_root_path / agent_data_dir / sub_dir_path
    else:
        logger.warning("No project root configured, using relative path for agent data directory")
        path = Path(agent_data_dir) / sub_dir_path
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


def get_facility_timezone():
    """Get the facility timezone from config as a ZoneInfo object.

    Returns:
        ZoneInfo for the configured facility timezone, defaulting to UTC.
    """
    from zoneinfo import ZoneInfo

    tz_name = get_config_value("facility_timezone", "UTC")
    return ZoneInfo(tz_name)


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
        >>> user_id = config.get("user_id")
        >>> models = config.get("model_configs", {})

        >>> # Explicit configuration path (also becomes default)
        >>> config = get_full_configuration("/path/to/my-config.yml")
        >>> models = config.get("model_configs", {})
        >>> # Subsequent calls without path use this config
        >>> other_value = get_config_value("some.setting")
    """
    set_as_default = config_path is not None
    return _get_configurable(config_path, set_as_default=set_as_default)


# Eager-init config on import; deferred init is OK if config.yml is absent.
try:
    if "sphinx" not in sys.modules and not os.environ.get("SPHINX_BUILD"):
        _get_config()
except FileNotFoundError:
    pass
