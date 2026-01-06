"""Approval Configuration Models for Type-Safe Settings Management.

This module provides strongly typed configuration models for capability-specific
approval settings throughout the system. The models implement comprehensive
validation, secure defaults, and clear data structures without any business logic.

The configuration system follows a hierarchical approach with global settings
that can override capability-specific configurations. All models include
factory methods for creating instances from configuration dictionaries with
extensive validation and helpful error messages.

Key Features:
    - Strongly typed dataclass models with validation
    - Security-first defaults for all approval settings
    - Comprehensive error handling with clear messages
    - Support for flexible configuration input formats
    - Extensive logging for configuration audit trails
    - Immutable configuration objects (frozen dataclasses)

Configuration Hierarchy:
    1. GlobalApprovalConfig: Top-level configuration with global mode
    2. Capability-specific configs: PythonExecutionApprovalConfig, MemoryApprovalConfig
    3. ApprovalMode enum: Defines available approval modes for fine-grained control

Design Philosophy:
    - Security by default: Unknown or missing settings default to secure values
    - Clear validation: Configuration errors provide specific guidance for fixes
    - Type safety: All configuration access is type-safe with IDE support
    - Separation of concerns: Pure data models with no business logic

Examples:
    Create Python execution config from dictionary::\n
        >>> config_dict = {'enabled': True, 'mode': 'epics_writes'}
        >>> config = PythonExecutionApprovalConfig.from_dict(config_dict)
        >>> print(f"Mode: {config.mode.value}")

    Create global configuration::\n
        >>> global_config = GlobalApprovalConfig.from_dict({
        ...     'global_mode': 'selective',
        ...     'capabilities': {
        ...         'python_execution': {'enabled': True, 'mode': 'all_code'},
        ...         'memory': {'enabled': False}
        ...     }
        ... })

    Handle validation errors::\n
        >>> try:
        ...     config = PythonExecutionApprovalConfig.from_dict({'mode': 'invalid'})
        ... except ValueError as e:
        ...     print(f"Configuration error: {e}")

.. note::
   All configuration models are immutable (frozen dataclasses) to prevent
   accidental modification after creation.

.. warning::
   Configuration validation uses security-first defaults. Missing or invalid
   settings will default to the most secure option (approval required).
"""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ApprovalMode(Enum):
    """Enumeration of approval modes for capability-specific approval control.

    Defines the available approval modes that control when human approval
    is required for various operations within the system. Each mode represents
    a different level of approval granularity, from completely disabled to
    requiring approval for all operations.

    The modes are designed to provide flexible control over approval requirements
    while maintaining clear semantic meaning for each level of restriction.

    Available Modes:
        DISABLED: No approval required for any operations
        CONTROL_WRITES: Approval required only for operations that write to control systems
        EPICS_WRITES: (Deprecated) Alias for CONTROL_WRITES - kept for backward compatibility
        ALL_CODE: Approval required for all code execution operations

    Examples:
        Use in configuration::\n
            >>> mode = ApprovalMode.CONTROL_WRITES
            >>> print(f"Mode value: {mode.value}")
            >>> print(f"Mode name: {mode.name}")

        Validate mode from string::\n
            >>> try:
            ...     mode = ApprovalMode("control_writes")
            ...     print(f"Valid mode: {mode}")
            ... except ValueError:
            ...     print("Invalid mode string")

    .. note::
       These modes are primarily used for Python execution approval but the
       pattern can be extended to other capabilities as needed.

    .. deprecated:: 0.9.5
       EPICS_WRITES is deprecated. Use CONTROL_WRITES instead for control-system-agnostic configuration.

    .. seealso::
       :class:`PythonExecutionApprovalConfig` : Configuration class that uses this enum
       :class:`PythonExecutionApprovalEvaluator` : Evaluator that processes these modes
       :class:`GlobalApprovalConfig` : Global configuration that can override mode settings
    """

    DISABLED = "disabled"
    CONTROL_WRITES = "control_writes"
    EPICS_WRITES = "epics_writes"  # Deprecated - kept for backward compatibility
    ALL_CODE = "all_code"


@dataclass(frozen=True)
class PythonExecutionApprovalConfig:
    """Configuration model for Python code execution approval settings.

    Immutable dataclass that encapsulates all settings related to Python code
    execution approval. The configuration includes both a high-level enabled flag
    and a granular mode setting that controls when approval is required based on
    code characteristics.

    The configuration supports multiple approval modes to balance security needs
    with operational efficiency. The most common mode is EPICS_WRITES, which
    requires approval only for code that can modify accelerator systems.

    :param enabled: Whether Python execution approval is enabled globally
    :type enabled: bool
    :param mode: Granular approval mode controlling when approval is required
    :type mode: ApprovalMode

    Examples:
        Create configuration for EPICS writes approval::\n
            >>> config = PythonExecutionApprovalConfig(
            ...     enabled=True,
            ...     mode=ApprovalMode.EPICS_WRITES
            ... )
            >>> print(f"Approval enabled: {config.enabled}")
            >>> print(f"Mode: {config.mode.value}")

        Create configuration with all code approval::\n
            >>> config = PythonExecutionApprovalConfig(
            ...     enabled=True,
            ...     mode=ApprovalMode.ALL_CODE
            ... )

    .. note::
       This is a frozen dataclass - instances cannot be modified after creation.
       Create new instances for different configurations.

    .. seealso::
       :class:`ApprovalMode` : Enum values used by this configuration
       :class:`PythonExecutionApprovalEvaluator` : Evaluator that uses this configuration
       :class:`ApprovalManager` : Manager that provides instances of this configuration
       :meth:`from_dict` : Factory method for creating instances from dictionaries
    """

    enabled: bool
    mode: ApprovalMode

    @classmethod
    def from_dict(cls, data: dict) -> "PythonExecutionApprovalConfig":
        """Create configuration instance from dictionary with comprehensive validation.

        Factory method that creates a PythonExecutionApprovalConfig instance from
        a configuration dictionary, applying security-first defaults and comprehensive
        validation. The method provides helpful error messages for configuration issues.

        Default Behavior:
            - Missing 'enabled' field defaults to True (secure default)
            - Missing 'mode' field defaults to 'all_code' (most secure mode)
            - Invalid mode values raise ValueError with valid options

        :param data: Configuration dictionary containing approval settings
        :type data: dict
        :return: Validated configuration instance
        :rtype: PythonExecutionApprovalConfig
        :raises ValueError: If data is not a dict or contains invalid mode values

        Examples:
            Create from complete configuration::\n
                >>> config_dict = {'enabled': True, 'mode': 'epics_writes'}
                >>> config = PythonExecutionApprovalConfig.from_dict(config_dict)
                >>> print(f"Enabled: {config.enabled}, Mode: {config.mode.value}")

            Create with defaults (secure fallbacks)::\n
                >>> config = PythonExecutionApprovalConfig.from_dict({})
                >>> print(f"Default enabled: {config.enabled}")  # True
                >>> print(f"Default mode: {config.mode.value}")    # 'all_code'

            Handle validation errors::\n
                >>> try:
                ...     config = PythonExecutionApprovalConfig.from_dict({'mode': 'invalid'})
                ... except ValueError as e:
                ...     print(f"Validation error: {e}")

        .. warning::
           This method applies security-first defaults. Missing configuration
           will default to the most secure settings (approval required).

        .. seealso::
           :class:`PythonExecutionApprovalConfig` : Configuration class created by this method
           :class:`ApprovalMode` : Enum values validated by this method
           :class:`GlobalApprovalConfig.from_dict` : Similar factory method for global config
        """
        if not isinstance(data, dict):
            raise ValueError(f"Python execution approval config must be dict, got {type(data)}")

        # Default to secure mode (approval enabled) for safety
        if "enabled" not in data:
            logger.warning(
                "⚠️  Python execution approval 'enabled' not specified in config, defaulting to True"
            )
        enabled = data.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(
                f"Python execution approval 'enabled' must be bool, got {type(enabled)}"
            )

        # Default to most restrictive mode for security compliance
        if "mode" not in data:
            logger.warning(
                "⚠️  Python execution approval 'mode' not specified in config, defaulting to 'all_code'. Consider setting to 'control_writes' for better performance."
            )
        mode_str = data.get("mode", "all_code")

        # Backward compatibility: map old mode names to new ones
        if mode_str == "epics_writes":
            logger.warning(
                "⚠️  DEPRECATED: approval mode 'epics_writes' is deprecated. Please use 'control_writes' instead for control-system-agnostic configuration."
            )
            logger.info("   Automatically mapping 'epics_writes' → 'control_writes'")
            mode_str = "control_writes"

        try:
            mode = ApprovalMode(mode_str)
        except ValueError as e:
            valid_modes = [
                m.value for m in ApprovalMode if m != ApprovalMode.EPICS_WRITES
            ]  # Hide deprecated mode
            raise ValueError(
                f"Invalid approval mode '{mode_str}'. Valid modes: {valid_modes}"
            ) from e

        return cls(enabled=enabled, mode=mode)


@dataclass(frozen=True)
class MemoryApprovalConfig:
    """Configuration model for memory operation approval settings.

    Immutable dataclass that encapsulates settings related to memory operation
    approval. Currently supports simple enabled/disabled logic but is designed
    for extension with more sophisticated approval rules in the future.

    The configuration controls whether memory operations (create, update, delete)
    require human approval before execution. This provides protection for
    sensitive user data and system memory state.

    :param enabled: Whether memory operation approval is required
    :type enabled: bool

    Examples:
        Create memory approval configuration::\n
            >>> config = MemoryApprovalConfig(enabled=True)
            >>> print(f"Memory approval enabled: {config.enabled}")

        Create disabled configuration::\n
            >>> config = MemoryApprovalConfig(enabled=False)
            >>> # Memory operations will not require approval

    .. note::
       This is a frozen dataclass - instances cannot be modified after creation.
       The design allows for future extensions with additional fields.
    """

    enabled: bool

    @classmethod
    def from_dict(cls, data: bool | dict) -> "MemoryApprovalConfig":
        """Create configuration instance from flexible input format with validation.

        Factory method that creates a MemoryApprovalConfig instance from either
        a boolean value (for simple enabled/disabled) or a dictionary (for future
        extensibility). Applies security-first defaults when configuration is ambiguous.

        Supported Input Formats:
            - bool: Directly sets the enabled flag
            - dict: Extracts 'enabled' field with secure default

        :param data: Configuration data as boolean or dictionary
        :type data: Union[bool, dict]
        :return: Validated configuration instance
        :rtype: MemoryApprovalConfig
        :raises ValueError: If data is neither bool nor dict, or bool value is invalid

        Examples:
            Create from boolean::\n
                >>> config = MemoryApprovalConfig.from_dict(True)
                >>> print(f"Enabled: {config.enabled}")

            Create from dictionary::\n
                >>> config_dict = {'enabled': False}
                >>> config = MemoryApprovalConfig.from_dict(config_dict)
                >>> print(f"Enabled: {config.enabled}")

            Create with secure default::\n
                >>> config = MemoryApprovalConfig.from_dict({})
                >>> print(f"Default enabled: {config.enabled}")  # True

            Handle invalid input::\n
                >>> try:
                ...     config = MemoryApprovalConfig.from_dict(\"invalid\")
                ... except ValueError as e:
                ...     print(f\"Invalid input: {e}\")

        .. note::
           When using dictionary format, missing 'enabled' field defaults to True
           for security. This ensures approval is required unless explicitly disabled.
        """
        if isinstance(data, bool):
            return cls(enabled=data)
        elif isinstance(data, dict):
            # Security-first default: enable approval when configuration is ambiguous
            if "enabled" not in data:
                logger.warning(
                    "⚠️  Memory approval 'enabled' not specified in config, defaulting to True"
                )
            enabled = data.get("enabled", True)
            if not isinstance(enabled, bool):
                raise ValueError(f"Memory approval 'enabled' must be bool, got {type(enabled)}")
            return cls(enabled=enabled)
        else:
            raise ValueError(f"Memory approval config must be bool or dict, got {type(data)}")


@dataclass(frozen=True)
class GlobalApprovalConfig:
    """Global approval configuration integrating all capability-specific settings.

    Top-level configuration model that combines global approval mode settings
    with capability-specific configurations. This class implements the hierarchical
    configuration system where global modes can override individual capability
    settings for consistent system-wide approval behavior.

    The global configuration supports three main modes:
    - disabled: All approvals disabled system-wide
    - selective: Use capability-specific settings
    - all_capabilities: All approvals enabled system-wide

    :param global_mode: System-wide approval mode controlling all capabilities
    :type global_mode: str
    :param python_execution: Python code execution approval configuration
    :type python_execution: PythonExecutionApprovalConfig
    :param memory: Memory operation approval configuration
    :type memory: MemoryApprovalConfig

    Examples:
        Create global configuration::\n
            >>> python_config = PythonExecutionApprovalConfig(
            ...     enabled=True, mode=ApprovalMode.EPICS_WRITES
            ... )
            >>> memory_config = MemoryApprovalConfig(enabled=False)
            >>> global_config = GlobalApprovalConfig(
            ...     global_mode="selective",
            ...     python_execution=python_config,
            ...     memory=memory_config
            ... )

        Access capability configurations::\n
            >>> print(f"Global mode: {global_config.global_mode}")
            >>> print(f"Python enabled: {global_config.python_execution.enabled}")
            >>> print(f"Memory enabled: {global_config.memory.enabled}")

    .. note::
       This is a frozen dataclass representing immutable configuration state.
       The configuration hierarchy allows global modes to override capability
       settings when applied by the ApprovalManager.
    """

    global_mode: str  # "disabled" | "selective" | "all_capabilities"
    python_execution: PythonExecutionApprovalConfig
    memory: MemoryApprovalConfig

    @classmethod
    def from_dict(cls, data: dict) -> "GlobalApprovalConfig":
        """Create global configuration instance from dictionary with comprehensive validation.

        Factory method that creates a GlobalApprovalConfig instance from a complete
        configuration dictionary. Performs extensive validation of the global structure
        and delegates capability-specific validation to appropriate config classes.

        The method applies security-first defaults for missing configuration sections
        and provides detailed error messages for configuration issues. All capability
        configurations are validated and instantiated as strongly-typed objects.

        Required Structure:
            - global_mode: One of 'disabled', 'selective', 'all_capabilities'
            - capabilities: Dictionary containing capability-specific settings

        :param data: Complete approval configuration dictionary from config.yml
        :type data: dict
        :return: Validated global configuration instance
        :rtype: GlobalApprovalConfig
        :raises ValueError: If configuration structure is invalid or contains invalid values

        Examples:
            Create from complete configuration::\n
                >>> config_dict = {
                ...     'global_mode': 'selective',
                ...     'capabilities': {
                ...         'python_execution': {'enabled': True, 'mode': 'epics_writes'},
                ...         'memory': {'enabled': False}
                ...     }
                ... }
                >>> config = GlobalApprovalConfig.from_dict(config_dict)
                >>> print(f\"Global mode: {config.global_mode}\")

            Create with missing sections (secure defaults)::\n
                >>> minimal_config = {'global_mode': 'selective', 'capabilities': {}}
                >>> config = GlobalApprovalConfig.from_dict(minimal_config)
                >>> # Missing capabilities will use secure defaults

            Handle validation errors::\n
                >>> try:
                ...     config = GlobalApprovalConfig.from_dict({'global_mode': 'invalid'})
                ... except ValueError as e:
                ...     print(f\"Configuration error: {e}\")

        .. warning::
           Missing capability sections will be created with security-first defaults.
           This ensures the system remains secure even with incomplete configuration.
        """
        if not isinstance(data, dict):
            raise ValueError(f"Approval config must be dict, got {type(data)}")

        # Validate global mode
        global_mode = data.get("global_mode", "selective")
        valid_modes = ["disabled", "selective", "all_capabilities"]
        if global_mode not in valid_modes:
            raise ValueError(f"Invalid global_mode '{global_mode}'. Valid modes: {valid_modes}")

        # Parse capabilities
        capabilities = data.get("capabilities", {})
        if not isinstance(capabilities, dict):
            raise ValueError(f"Approval capabilities must be dict, got {type(capabilities)}")

        # Parse Python execution config with explicit warning
        if "python_execution" not in capabilities:
            logger.warning(
                "⚠️  'python_execution' section missing from approval config, using defaults. This may cause unexpected approval behavior!"
            )
            logger.warning(
                "⚠️  Consider adding: approval.capabilities.python_execution = {enabled: true, mode: 'epics_writes'}"
            )
        python_data = capabilities.get("python_execution", {"enabled": True, "mode": "all_code"})
        python_config = PythonExecutionApprovalConfig.from_dict(python_data)

        # Memory approval config with secure defaults for data protection
        if "memory" not in capabilities:
            logger.warning(
                "⚠️  'memory' section missing from approval config, defaulting to enabled=True"
            )
        memory_data = capabilities.get("memory", {"enabled": True})
        memory_config = MemoryApprovalConfig.from_dict(memory_data)

        return cls(global_mode=global_mode, python_execution=python_config, memory=memory_config)
