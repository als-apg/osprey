"""
Configuration System

Professional configuration system that works seamlessly both inside and outside
LangGraph contexts. Features:
- Single-file YAML loading with environment resolution
- LangGraph integration, pre-computed structures, context awareness
- Single source of truth with automatic context detection
- Flat structure: framework and application settings coexist via unique naming

Clean, modern configuration architecture supporting both standalone and graph execution.
"""

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    from langgraph.config import get_config
except (RuntimeError, ImportError):
    get_config = None

# Use standard logging (not get_logger) to avoid circular imports with logger.py
# The short name 'CONFIG' enables easy filtering: quiet_logger(['REGISTRY', 'CONFIG'])
logger = logging.getLogger('CONFIG')


class ConfigBuilder:
    """
    Configuration builder with clean, modern architecture.

    Features:
    - Single-file YAML loading with validation and error handling
    - Environment variable resolution
    - Pre-computed nested dictionaries for performance
    - Explicit fail-fast behavior for required configurations
    - Flat structure supporting framework + application settings via unique naming
    """

    # Sentinel object to distinguish between "no default provided" and "default is None"
    _REQUIRED = object()

    def _require_config(self, path: str, default: Any = _REQUIRED) -> Any:
        """
        Get configuration value with explicit control over required vs. optional settings.

        This helper function provides three levels of configuration handling:
        1. Required settings (no default) - fail fast if missing
        2. Optional settings with visibility (default provided) - warn when default is used
        3. Silent optional settings - use standard self.get() for truly optional configs

        Args:
            path: Dot-separated configuration path (e.g., "execution.limits.max_retries")
            default: Default value to use if config is missing. If not provided,
                    the configuration is considered required and will raise ValueError.
                    If provided, logs a warning when the default is used.

        Returns:
            The configuration value, or default if provided and config is missing

        Raises:
            ValueError: If required configuration (no default) is missing or None

        Examples:
            # Required configuration - will fail if missing
            recursion_limit = self._require_config('execution_control.limits.graph_recursion_limit')

            # Optional configuration with explicit default and visibility
            max_retries = self._require_config('execution_control.limits.max_step_retries', 0)

            # Silent optional configuration - use standard get() for noise-free defaults
            debug_mode = self.get('development.debug', False)
        """
        value = self.get(path)

        if value is None:
            if default is self._REQUIRED:
                # No default provided - this is a required configuration
                raise ValueError(
                    f"Missing required configuration: '{path}' must be explicitly set in config.yml. "
                    f"This setting has no default value and must be configured explicitly."
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
        # Load .env file from current working directory
        # This ensures environment variables are available for config resolution
        try:
            from dotenv import load_dotenv
            dotenv_path = Path.cwd() / ".env"
            if dotenv_path.exists():
                load_dotenv(dotenv_path, override=False)  # Don't override existing env vars
                logger.debug(f"Loaded .env file from {dotenv_path}")
            else:
                logger.debug(f"No .env file found at {dotenv_path}")
        except ImportError:
            logger.warning("python-dotenv not available, skipping .env file loading")

        if config_path is None:
            # Check current working directory (where user ran the command)
            cwd_config = Path.cwd() / "config.yml"
            if cwd_config.exists():
                config_path = cwd_config
            else:
                # NO FALLBACK - Fail fast with clear error
                raise FileNotFoundError(
                    f"No config.yml found in current directory: {Path.cwd()}\n\n"
                    f"Please run this command from a project directory containing config.yml,\n"
                    f"or set CONFIG_FILE environment variable to point to your config file.\n\n"
                    f"Example: export CONFIG_FILE=/path/to/your/config.yml"
                )

        self.config_path = Path(config_path)
        self.raw_config = self._load_config()

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

        Supports both simple and bash-style default value syntax:
        - ${VAR_NAME} - simple substitution
        - ${VAR_NAME:-default_value} - with default value
        - $VAR_NAME - simple substitution without braces
        """
        if isinstance(data, dict):
            return {key: self._resolve_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._resolve_env_vars(item) for item in data]
        elif isinstance(data, str):
            def replace_env_var(match):
                # Pattern matches: ${VAR_NAME:-default} or ${VAR_NAME} or $VAR_NAME
                if match.group(1):  # ${VAR_NAME:-default} or ${VAR_NAME}
                    var_name = match.group(1)
                    default_value = match.group(2) if match.group(2) is not None else None
                else:  # $VAR_NAME (simple form)
                    var_name = match.group(3)
                    default_value = None

                env_value = os.environ.get(var_name)
                if env_value is None:
                    if default_value is not None:
                        # Use default value from ${VAR:-default} syntax
                        return default_value
                    else:
                        # Only log warning if not in quiet mode (e.g., from interactive menu subprocess)
                        if not os.environ.get('OSPREY_QUIET'):
                            logger.info(f"Environment variable '{var_name}' not found, keeping original value")
                        return match.group(0)
                return env_value

            # Pattern matches ${VAR_NAME:-default}, ${VAR_NAME}, or $VAR_NAME
            pattern = r'\$\{([^}:]+)(?::-(.*?))?\}|\$([A-Za-z_][A-Za-z0-9_]*)'
            return re.sub(pattern, replace_env_var, data)
        else:
            return data

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from single file."""
        # Load the config file
        config = self._load_yaml_file(self.config_path)

        # Apply environment variable substitution
        config = self._resolve_env_vars(config)

        logger.info(f"Loaded configuration from {self.config_path}")
        return config

    def _get_approval_config(self) -> dict[str, Any]:
        """Get approval configuration with sensible defaults.

        Returns approval configuration from config.yml if present, otherwise provides
        secure defaults suitable for tutorial and development environments.

        Returns:
            dict: Approval configuration with global_mode and capabilities sections
        """
        # Try to get approval config from file
        approval_config = self.get('approval', None)

        # If approval section exists and has content, use it
        if approval_config:
            return approval_config

        # Otherwise, provide sensible defaults for tutorial/development mode
        logger.warning("⚠️  'approval' section missing from config.yml, using framework defaults")
        logger.warning("⚠️  For production use, please add an 'approval' section to your config.yml")

        # Sensible defaults for tutorial/development environments
        return {
            'global_mode': 'selective',
            'capabilities': {
                'python_execution': {
                    'enabled': True,
                    'mode': 'epics_writes'
                },
                'memory': {
                    'enabled': True
                }
            }
        }

    def _get_execution_config(self) -> dict[str, Any]:
        """Get execution configuration with sensible defaults.

        Returns execution configuration from config.yml if present, otherwise provides
        defaults suitable for local Python execution in tutorial environments.

        Returns:
            dict: Execution configuration including method, python_env_path, and modes
        """
        # Try to get execution config from file
        execution_config = self.get('execution', None)

        # If execution section exists and has content, use it
        if execution_config:
            return execution_config

        # Otherwise, provide sensible defaults for tutorial/development mode
        logger.warning("⚠️  'execution' section missing from config.yml, using framework defaults")
        logger.warning("⚠️  Using local Python execution (no container/Jupyter support)")

        # Import here to avoid circular dependencies
        import sys

        # Sensible defaults for tutorial environments (local execution only)
        return {
            'execution_method': 'local',
            'python_env_path': sys.executable,  # Use current Python interpreter
            'modes': {
                'read_only': {
                    'kernel_name': 'python3-readonly',
                    'gateway': 'read_only',
                    'allows_writes': False,
                    'environment': {}
                },
                'write_access': {
                    'kernel_name': 'python3-write',
                    'gateway': 'write_access',
                    'allows_writes': True,
                    'requires_approval': True,
                    'environment': {}
                }
            }
        }

    def _get_python_executor_config(self) -> dict[str, Any]:
        """Get python executor configuration with sensible defaults.

        Returns python_executor configuration from config.yml if present, otherwise
        provides reasonable defaults for retry and timeout settings.

        Returns:
            dict: Python executor configuration with retry and timeout settings
        """
        # Try to get python_executor config from file
        python_executor_config = self.get('python_executor', None)

        # If python_executor section exists and has content, use it
        if python_executor_config:
            return python_executor_config

        # Otherwise, provide sensible defaults
        return {
            'max_generation_retries': 3,
            'max_execution_retries': 3,
            'execution_timeout_seconds': 600
        }

    def _build_configurable(self) -> dict[str, Any]:
        """Build the configurable dictionary with pre-computed nested structures."""
        configurable = {
            # ===== SESSION INFORMATION =====
            "user_id": None,
            "chat_id": None,
            "session_id": None,
            "thread_id": None,
            "session_url": None,

            # ===== EXECUTION LIMITS =====
            "execution_limits": self._build_execution_limits(),

            # ===== AGENT CONTROL DEFAULTS =====
            "agent_control_defaults": self._build_agent_control_defaults(),

            # ===== COMPLEX NESTED STRUCTURES =====
            "model_configs": self._build_model_configs(),
            "provider_configs": self._build_provider_configs(),
            "service_configs": self._build_service_configs(),

            # ===== FRAMEWORK EXECUTION CONFIGURATION =====
            # Python execution settings and executor service configuration
            "execution": self._get_execution_config(),
            "python_executor": self._get_python_executor_config(),

            # ===== LOGGING CONFIGURATION =====
            "logging": self.get('logging', {}),

            # ===== SIMPLE FLAT CONFIGS =====
            "development": self.get('development', {}),
            "epics_config": self.get('execution.epics', {}),
            "approval_config": self._get_approval_config(),

            # ===== PROJECT CONFIGURATION =====
            # Essential for absolute path resolution across deployment environments
            "project_root": self.get('project_root'),

            # ===== APPLICATION CONTEXT =====
            "applications": self.get('applications', []),
            "current_application": self._get_current_application(),
            "registry_path": self.get('registry_path'),
        }

        return configurable

    def _build_model_configs(self) -> dict[str, Any]:
        """Get model configs from flat structure."""
        return self.get('models', {})

    def _build_provider_configs(self) -> dict[str, Any]:
        """Build provider configs."""
        return self.get('api.providers', {})

    def _build_service_configs(self) -> dict[str, Any]:
        """Get service configs from flat structure."""
        return self.get('services', {})

    def _build_execution_limits(self) -> dict[str, Any]:
        """Build execution limits"""

        return {
            "graph_recursion_limit": self._require_config('execution_control.limits.graph_recursion_limit', 100),
            "max_reclassifications": self._require_config('execution_control.limits.max_reclassifications', 1),
            "max_planning_attempts": self._require_config('execution_control.limits.max_planning_attempts', 2),
            "max_step_retries": self._require_config('execution_control.limits.max_step_retries', 0),
            "max_execution_time_seconds": self._require_config('execution_control.limits.max_execution_time_seconds', 300),
            "max_concurrent_classifications": self._require_config('execution_control.limits.max_concurrent_classifications', 5),
        }

    def _build_agent_control_defaults(self) -> dict[str, Any]:
        """Build agent control defaults with explicit configuration control."""

        return {
            # Planning control
            "planning_mode_enabled": False,

            # EPICS control
            "epics_writes_enabled": self._require_config('execution_control.epics.writes_enabled', False),

            # Approval control
            "approval_global_mode": self._require_config('approval.global_mode', 'selective'),
            "python_execution_approval_enabled": self._require_config('approval.capabilities.python_execution.enabled', True),
            "python_execution_approval_mode": self._require_config('approval.capabilities.python_execution.mode', 'all_code'),
            "memory_approval_enabled": self._require_config('approval.capabilities.memory.enabled', True),

            # Performance bypass configuration (configurable via YAML)
            "task_extraction_bypass_enabled": self._require_config('execution_control.agent_control.task_extraction_bypass_enabled', False),
            "capability_selection_bypass_enabled": self._require_config('execution_control.agent_control.capability_selection_bypass_enabled', False),

            # Note: Execution limits (max_reclassifications, max_planning_attempts, etc.)
            # are now centralized in get_execution_limits() utility function
        }

    def _get_current_application(self) -> str | None:
        """Get the current/primary application name."""
        applications = self.get('applications', [])
        if isinstance(applications, dict) and applications:
            return list(applications.keys())[0]
        elif isinstance(applications, list) and applications:
            return applications[0]
        return None

    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation path."""
        keys = path.split('.')
        value = self.raw_config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Global configuration instances
# Default config (singleton pattern for backward compatibility)
_default_config: ConfigBuilder | None = None
_default_configurable: dict[str, Any] | None = None

# Per-path config cache for explicit config paths
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

    # If no explicit path, use default singleton behavior
    if config_path is None:
        if _default_config is None:
            # Check for environment variable override
            config_file = os.environ.get('CONFIG_FILE')
            if config_file:
                _default_config = ConfigBuilder(config_file)
            else:
                _default_config = ConfigBuilder()

            # Cache configurable for efficient non-LangGraph contexts
            _default_configurable = _default_config.configurable.copy()

            logger.info("Initialized default configuration system")

        return _default_config

    # For explicit path, cache per path to avoid reloading
    resolved_path = str(Path(config_path).resolve())

    if resolved_path not in _config_cache:
        logger.info(f"Loading configuration from explicit path: {resolved_path}")
        _config_cache[resolved_path] = ConfigBuilder(resolved_path)

    # If requested, also set this as the default config
    if set_as_default:
        _default_config = _config_cache[resolved_path]
        _default_configurable = _default_config.configurable.copy()
        logger.debug(f"Set explicit config as default: {resolved_path}")

    return _config_cache[resolved_path]


def _get_configurable(config_path: str | None = None, set_as_default: bool = False) -> dict[str, Any]:
    """Get configurable dict with automatic context detection.

    This function supports both LangGraph execution contexts and standalone execution,
    with optional explicit configuration path support.

    Args:
        config_path: Optional explicit path to configuration file
        set_as_default: If True and config_path is provided, set as default config

    Returns:
        Complete configuration dictionary with all configurable values
    """
    try:
        # Prefer LangGraph context for runtime-injected configuration
        # (only when no explicit config_path is provided)
        if config_path is None and get_config:
            config = get_config()
            return config.get("configurable", {})
        else:
            raise ImportError("LangGraph not available or explicit path provided")
    except (RuntimeError, ImportError):
        # Use cached configurable for standalone execution
        config = _get_config(config_path, set_as_default=set_as_default)

        # For default config, use cached configurable for performance
        if config_path is None:
            global _default_configurable
            if _default_configurable is None:
                _default_configurable = config.configurable.copy()
            return _default_configurable

        # For explicit paths, return configurable directly
        return config.configurable


# =============================================================================
# CONTEXT-AWARE UTILITY FUNCTIONS
# =============================================================================

def get_model_config(model_name: str, config_path: str | None = None) -> dict[str, Any]:
    """
    Get model configuration with automatic context detection.

    Works both inside and outside LangGraph contexts.
    All models are configured at the top level in the 'models' section.

    Args:
        model_name: Name of the model (e.g., 'orchestrator', 'classifier', 'time_parsing',
                   'response', 'approval', 'memory', 'task_extraction', 'python_code_generator')
        config_path: Optional explicit path to configuration file for multi-project workflows

    Returns:
        Dictionary with model configuration containing provider, model_id, and optional settings

    Examples:
        Default config (searches current directory):
            >>> get_model_config("orchestrator")
            {'provider': 'anthropic', 'model_id': 'claude-3-5-sonnet-20241022', ...}

        Multi-project workflow:
            >>> get_model_config("orchestrator", config_path="~/other-project/config.yml")
            {'provider': 'openai', 'model_id': 'gpt-4o', ...}

    Configuration format (config.yml):
        models:
          orchestrator:
            provider: anthropic
            model_id: claude-3-5-sonnet-20241022
          classifier:
            provider: anthropic
            model_id: claude-3-5-haiku-20241022
    """
    configurable = _get_configurable(config_path)
    model_configs = configurable.get("model_configs", {})

    # Direct lookup from flat structure
    return model_configs.get(model_name, {})


def get_provider_config(provider_name: str, config_path: str | None = None) -> dict[str, Any]:
    """Get API provider configuration with automatic context detection.

    Args:
        provider_name: Name of the provider (e.g., 'openai', 'anthropic')
        config_path: Optional explicit path to configuration file

    Returns:
        Dictionary with provider configuration
    """
    configurable = _get_configurable(config_path)
    provider_configs = configurable.get("provider_configs", {})
    return provider_configs.get(provider_name, {})


def get_framework_service_config(service_name: str, config_path: str | None = None) -> dict[str, Any]:
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


def get_application_service_config(app_name: str, service_name: str) -> dict[str, Any]:
    """Get application service configuration with automatic context detection."""
    configurable = _get_configurable()
    service_configs = configurable.get("service_configs", {})
    # Try new flat format first (single-config)
    if service_name in service_configs:
        return service_configs.get(service_name, {})
    # Fall back to legacy nested format
    logger.warning(
        f"DEPRECATED: Using legacy nested config format for applications.{app_name}.services.{service_name}. "
        f"Please migrate to flat config structure with services at top level."
    )
    app_services = service_configs.get("applications", {}).get(app_name, {})
    return app_services.get(service_name, {})


def get_pipeline_config(app_name: str = None) -> dict[str, Any]:
    """Get pipeline configuration with automatic context detection."""
    configurable = _get_configurable()
    config = _get_config()

    # Try new flat format first (single-config)
    pipeline_config = config.get('pipeline', {})
    if pipeline_config:
        return pipeline_config

    # Fall back to legacy nested format
    if app_name is None:
        app_name = configurable.get("current_application")

    if app_name:
        logger.warning(
            f"DEPRECATED: Using legacy nested config format for applications.{app_name}.pipeline. "
            f"Please migrate to flat config structure with pipeline at top level."
        )
        app_path = f"applications.{app_name}.pipeline"
        app_config = config.get(app_path, {})
        if app_config:
            return app_config

    # Fall back to framework pipeline config
    logger.warning(
        "DEPRECATED: Using legacy nested config format for osprey.pipeline. "
        "Please migrate to flat config structure with pipeline at top level."
    )
    framework = configurable.get("osprey", {})
    return framework.get("pipeline", {})


def get_execution_limits() -> dict[str, Any]:
    """Get execution limits with automatic context detection."""
    configurable = _get_configurable()
    execution_limits = configurable.get("execution_limits")

    if execution_limits is None:
        raise RuntimeError(
            "Execution limits configuration not found. Please ensure 'execution_limits' is properly "
            "configured in your config.yml or environment settings with the following required fields: "
            "max_reclassifications, max_planning_attempts, max_step_retries, max_execution_time_seconds, graph_recursion_limit"
        )

    return execution_limits


def get_agent_control_defaults() -> dict[str, Any]:
    """Get agent control defaults with automatic context detection."""
    configurable = _get_configurable()
    return configurable.get("agent_control_defaults", {})


def get_session_info() -> dict[str, Any]:
    """Get session information with automatic context detection."""
    configurable = _get_configurable()
    return {
        "user_id": configurable.get("user_id"),
        "chat_id": configurable.get("chat_id"),
        "session_id": configurable.get("session_id"),
        "thread_id": configurable.get("thread_id"),
        "session_url": configurable.get("session_url"),
    }

def get_interface_context() -> str:
    """
    Get interface context indicating which user interface is being used.

    The interface context determines how responses are formatted and which
    features are available (e.g., figure rendering, notebook links, command buttons).

    Returns:
        str: The interface type, one of:
            - "openwebui": Open WebUI interface with rich rendering capabilities
            - "cli": Command-line interface with text-only output
            - "unknown": Interface type not detected or not set

    Example:
        >>> interface = get_interface_context()
        >>> if interface == "openwebui":
        ...     print("Rich UI features available")

    Note:
        This is set automatically by each interface implementation during initialization.
        The value is used by response generators to provide interface-appropriate
        guidance about figures, notebooks, and executable commands.
    """
    configurable = _get_configurable()
    return configurable.get("interface_context", "unknown")


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

    # Get project root and file paths configuration
    project_root = config.get("project_root")
    main_file_paths = config.get("file_paths", {})
    agent_data_dir = main_file_paths.get("agent_data_dir", "_agent_data")

    # Check both main config and current application config for file paths
    current_app = get_current_application()
    sub_dir_path = None

    # First check main config file_paths
    if sub_dir in main_file_paths:
        sub_dir_path = main_file_paths[sub_dir]
        logger.debug(f"Found {sub_dir} in main file_paths: {sub_dir_path}")

    # Then check current application's file_paths (takes precedence)
    if current_app:
        app_file_paths = config.get(f"applications.{current_app}.file_paths", {})
        if sub_dir in app_file_paths:
            sub_dir_path = app_file_paths[sub_dir]
            logger.debug(f"Found {sub_dir} in {current_app} file_paths: {sub_dir_path}")

    # Fallback to the sub_dir name itself if not found anywhere
    if sub_dir_path is None:
        sub_dir_path = sub_dir
        logger.debug(f"Using fallback path for {sub_dir}: {sub_dir_path}")

    # Construct absolute path with explicit validation

    if project_root:
        project_root_path = Path(project_root)

        # Handle host_path override
        if host_path:
            # Force host path regardless of current environment
            logger.debug(f"Forcing host path resolution for: {sub_dir}")
            path = project_root_path / agent_data_dir / sub_dir_path
        else:
            # Container-aware path resolution
            if not project_root_path.exists():
                # Detect if we're running in a container environment
                container_project_roots = ["/app", "/pipelines", "/jupyter"]
                detected_container_root = None

                for container_root in container_project_roots:
                    container_path = Path(container_root)
                    if container_path.exists() and (container_path / agent_data_dir).exists():
                        detected_container_root = container_path
                        break

                if detected_container_root:
                    # Container environment detected - use container project root
                    logger.debug(f"Container environment detected: using {detected_container_root} instead of {project_root}")
                    path = detected_container_root / agent_data_dir / sub_dir_path
                else:
                    # Not in a known container environment - fall back to relative paths
                    logger.warning(f"Configured project root does not exist: {project_root}")
                    logger.warning("Falling back to relative path resolution")
                    path = Path(agent_data_dir) / sub_dir_path
                    path = path.resolve()
            else:
                # Host environment - use configured project root
                path = project_root_path / agent_data_dir / sub_dir_path
    else:
        # Support development environments without explicit project root configuration
        logger.warning("No project root configured, using relative path for agent data directory")
        path = Path(agent_data_dir) / sub_dir_path
        path = path.resolve()  # Ensure absolute path for consistent behavior

    return str(path)


# =============================================================================
# LANGGRAPH NATIVE ACCESS
# =============================================================================

def get_config_value(path: str, default: Any = None, config_path: str | None = None) -> Any:
    """
    Get a specific configuration value by dot-separated path.

    This function provides context-aware access to configuration values,
    working both inside and outside LangGraph execution contexts. Optionally,
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

    # Navigate through dot-separated path in configurable dict
    keys = path.split('.')
    value = configurable

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            # Not found in configurable, try raw config as fallback
            config = _get_config(config_path)
            return config.get(path, default)

    return value


def get_classification_config() -> dict[str, Any]:
    """
    Get classification configuration with sensible defaults.

    Controls parallel LLM-based capability classification to prevent API flooding
    while maintaining reasonable performance during task analysis.

    Returns:
        Dictionary with classification configuration including concurrency limits

    Examples:
        >>> config = get_classification_config()
        >>> max_concurrent = config.get('max_concurrent_classifications', 5)
    """
    configurable = _get_configurable()

    # Get classification concurrency limit from execution_control.limits (consistent with other limits)
    max_concurrent = configurable.get("execution_limits", {}).get("max_concurrent_classifications", 5)

    return {
        "max_concurrent_classifications": max_concurrent
    }


def get_full_configuration(config_path: str | None = None) -> dict[str, Any]:
    """
    Get the complete configuration dictionary.

    This function provides access to the entire configurable dictionary,
    working both inside and outside LangGraph execution contexts. Optionally,
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
    # If explicit path provided, set as default for future access
    set_as_default = config_path is not None
    return _get_configurable(config_path, set_as_default=set_as_default)

# Initialize the global configuration on import (skip for documentation/tests)
# This provides convenience for module-level logger initialization, but is not
# strictly required since logger.py has graceful fallbacks for missing config.
try:
    if 'sphinx' not in sys.modules and not os.environ.get('SPHINX_BUILD'):
        _get_config()
except FileNotFoundError:
    # Allow deferred initialization if config not available at import time
    # Config will be initialized on first access via the singleton pattern
    pass
