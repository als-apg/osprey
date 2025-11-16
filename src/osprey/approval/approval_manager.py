"""Approval Manager - Policy Configuration Layer for System-Wide Approval Control.

This module provides centralized approval policy management with clean separation
of concerns: configuration loading and validation only, with no business logic.
The ApprovalManager serves as the single source of truth for approval settings
across all capabilities in the system.

The manager implements a layered configuration approach:
1. Global approval modes that can override capability-specific settings
2. Capability-specific configuration with typed validation
3. Factory methods for creating configured evaluators

Key Features:
    - Type-safe configuration loading with comprehensive validation
    - Global approval modes (disabled, selective, all_capabilities)
    - Capability-specific configuration resolution with overrides
    - Factory pattern for evaluator creation
    - Extensive logging for security audit trails
    - Singleton pattern for consistent system-wide configuration

Architecture:
    The manager separates policy (configuration) from enforcement (evaluators).
    This design allows for dynamic configuration changes without affecting
    business logic, and enables consistent approval behavior across the system.

Examples:
    Initialize approval manager::\n
        >>> manager = get_approval_manager()  # Loads from global config
        >>> summary = manager.get_config_summary()
        >>> print(f"Global mode: {summary['global_mode']}")

    Get capability-specific configuration::\n
        >>> python_config = manager.get_python_execution_config()
        >>> print(f"Python approval enabled: {python_config.enabled}")
        >>> print(f"Python approval mode: {python_config.mode.value}")

    Create configured evaluators::\n
        >>> evaluator = manager.get_python_execution_evaluator()
        >>> decision = evaluator.evaluate(has_epics_writes=True, has_epics_reads=False)

    .. warning::
       This module handles security-critical configuration. Invalid or missing
       approval configuration will cause immediate startup failures to maintain
       system security integrity.

    .. note::
       The manager uses singleton pattern - configuration is loaded once at startup
       and cached for the lifetime of the application.

    .. seealso::
       :class:`GlobalApprovalConfig` : Configuration model used by the manager
       :class:`PythonExecutionApprovalEvaluator` : Evaluator created by this manager
       :class:`MemoryApprovalEvaluator` : Evaluator created by this manager
       :func:`get_approval_manager` : Singleton access function
"""

import logging

from .config_models import GlobalApprovalConfig, MemoryApprovalConfig, PythonExecutionApprovalConfig
from .evaluators import MemoryApprovalEvaluator, PythonExecutionApprovalEvaluator

logger = logging.getLogger(__name__)


class ApprovalManager:
    """Pure configuration service providing strongly typed approval models.

    Serves as the centralized configuration management system for all approval
    settings across the framework. This class implements a clean separation of
    concerns by handling only configuration loading, validation, and provision
    of typed configuration objects to capabilities.

    The manager implements a hierarchical configuration system where global
    approval modes can override capability-specific settings, ensuring consistent
    security posture across the entire system while allowing for granular control
    when needed.

    Responsibilities:
        - Load and validate approval configuration from global config system
        - Apply global mode overrides to capability-specific settings
        - Provide strongly typed configuration objects with validation
        - Create configured evaluator instances for capabilities
        - Maintain audit trail through comprehensive logging

    Explicitly NOT responsible for:
        - Business logic implementation (delegated to evaluators)
        - Approval decision making (capability-specific in evaluators)
        - State management (stateless configuration service)

    Configuration Hierarchy:
        1. Global mode settings (disabled, selective, all_capabilities)
        2. Capability-specific settings (python_execution, memory, etc.)
        3. Resolved effective configuration (global overrides applied)

    :param approval_config: Raw approval configuration dictionary from config.yml
    :type approval_config: dict

    Examples:
        Initialize with configuration::\n
            >>> config_dict = {
            ...     'global_mode': 'selective',
            ...     'capabilities': {
            ...         'python_execution': {'enabled': True, 'mode': 'epics_writes'},
            ...         'memory': {'enabled': False}
            ...     }
            ... }
            >>> manager = ApprovalManager(config_dict)

        Access resolved configuration::\n
            >>> python_config = manager.get_python_execution_config()
            >>> print(f"Effective setting: {python_config.enabled}")

        Create evaluators::\n
            >>> evaluator = manager.get_python_execution_evaluator()
            >>> # Evaluator is configured with resolved settings

    .. note::
       The manager is designed to be instantiated once at application startup
       and reused throughout the application lifecycle.

    .. warning::
       Configuration validation failures will raise ValueError to prevent
       insecure default behavior in production environments.
    """

    def __init__(self, approval_config: dict):
        """Initialize with approval configuration.

        :param approval_config: Raw approval configuration from config.yml
        :type approval_config: dict
        :raises ValueError: If configuration is invalid or missing required fields
        """
        try:
            logger.info(f"🔍 Loading approval configuration from raw config: {approval_config}")
            self.config = GlobalApprovalConfig.from_dict(approval_config)

            # Log configuration summary for security audit trail
            logger.info("✅ Loaded approval configuration successfully!")
            logger.info(f"   - Global mode: {self.config.global_mode}")
            logger.info(f"   - Python execution enabled: {self.config.python_execution.enabled}")
            logger.info(f"   - Python execution mode: {self.config.python_execution.mode.value}")
            logger.info(f"   - Memory enabled: {self.config.memory.enabled}")

        except ValueError as e:
            logger.error(f"Invalid approval configuration: {e}")
            raise

    def get_python_execution_config(self) -> PythonExecutionApprovalConfig:
        """Get strongly typed Python execution approval configuration.

        Applies global mode overrides to capability-specific settings,
        ensuring consistent behavior across the approval system.

        :return: Configuration object with resolved approval settings
        :rtype: PythonExecutionApprovalConfig
        """
        if self.config.global_mode == "disabled":
            # Override: global disable means everything is disabled
            return PythonExecutionApprovalConfig(enabled=False, mode=self.config.python_execution.mode)
        elif self.config.global_mode == "all_capabilities":
            # Override: global enable means everything is enabled
            return PythonExecutionApprovalConfig(enabled=True, mode=self.config.python_execution.mode)
        else:
            # Selective mode: use capability-specific setting
            return self.config.python_execution

    def get_memory_config(self) -> MemoryApprovalConfig:
        """Get strongly typed memory approval configuration.

        Applies global mode overrides to capability-specific settings,
        ensuring consistent behavior across the approval system.

        :return: Configuration object with resolved approval settings
        :rtype: MemoryApprovalConfig
        """
        if self.config.global_mode == "disabled":
            return MemoryApprovalConfig(enabled=False)
        elif self.config.global_mode == "all_capabilities":
            return MemoryApprovalConfig(enabled=True)
        else:
            return self.config.memory

    def get_python_execution_evaluator(self) -> PythonExecutionApprovalEvaluator:
        """Get configured Python execution approval evaluator.

        Creates a new evaluator instance with current configuration settings.
        The evaluator contains the business logic for making approval decisions.

        :return: Evaluator instance configured with current settings
        :rtype: PythonExecutionApprovalEvaluator
        """
        config = self.get_python_execution_config()
        return PythonExecutionApprovalEvaluator(config)

    def get_memory_evaluator(self) -> MemoryApprovalEvaluator:
        """Get configured memory approval evaluator.

        Creates a new evaluator instance with current configuration settings.
        The evaluator contains the business logic for making approval decisions.

        :return: Evaluator instance configured with current settings
        :rtype: MemoryApprovalEvaluator
        """
        config = self.get_memory_config()
        return MemoryApprovalEvaluator(config)

    def get_config_summary(self) -> dict:
        """Get configuration summary for debugging and monitoring.

        Provides a structured view of current approval configuration
        settings for logging, debugging, and administrative review.

        :return: Dictionary containing configuration summary with keys:
            - 'global_mode': Current global approval mode
            - 'python_execution': Python execution approval settings
            - 'memory': Memory operation approval settings
        :rtype: dict
        """
        return {
            'global_mode': self.config.global_mode,
            'python_execution': {
                'enabled': self.get_python_execution_config().enabled,
                'mode': self.get_python_execution_config().mode.value
            },
            'memory': {
                'enabled': self.get_memory_config().enabled
            }
        }


# Global instance
_approval_manager: ApprovalManager | None = None


def get_approval_manager() -> ApprovalManager:
    """Get global ApprovalManager instance using singleton pattern.

    Provides access to the system-wide approval manager instance, initializing
    it from the configuration system on first access. The function
    performs extensive validation to ensure security-critical approval
    configuration is present and valid before creating the manager instance.

    The singleton pattern ensures consistent configuration across all
    capabilities and prevents multiple initialization of security-critical
    settings. Configuration is loaded once at startup and cached for the
    application lifetime.

    Validation Process:

        1. Verify approval configuration exists in global config
        2. Validate configuration structure and required fields
        3. Ensure global_mode and capabilities sections are present
        4. Create and cache ApprovalManager instance
        5. Log configuration summary for audit trail

    :return: Configured approval manager instance ready for use
    :rtype: ApprovalManager
    :raises ValueError: If approval configuration is missing, has invalid
                       structure, or fails validation checks
    :raises KeyError: If required configuration sections are missing

    Examples:
        Get manager instance::

            >>> manager = get_approval_manager()
            >>> print(f"Manager ready with mode: {manager.config.global_mode}")

        Handle configuration errors::

            >>> try:
            ...     manager = get_approval_manager()
            ... except ValueError as e:
            ...     print(f"Configuration error: {e}")
            ...     # Fix config.yml and restart application

    .. seealso::
       :class:`ApprovalManager` : Manager class created by this function
       :class:`GlobalApprovalConfig` : Configuration model used for initialization
       :func:`configs.config.get_config_value` : Configuration source
       :meth:`ApprovalManager.get_config_summary` : Configuration validation method

    .. warning::
       This function will cause immediate application failure if approval
       configuration is missing or invalid. This is intentional to prevent
       insecure operation in production environments.

    .. note::
       The manager instance is cached globally. Subsequent calls return
       the same instance without re-reading configuration files.
    """
    global _approval_manager

    if _approval_manager is None:
        from osprey.utils.config import get_config_value

        # Get approval configuration from config system
        # The config system will provide sensible defaults if approval section is missing
        approval_config = get_config_value('approval_config')

        # Basic validation - config system should always provide a dict
        if not isinstance(approval_config, dict):
            raise ValueError(
                f"❌ CRITICAL: Approval configuration must be a dictionary, got {type(approval_config).__name__}"
            )

        try:
            _approval_manager = ApprovalManager(approval_config)
            logger.info("✅ Approval manager initialized successfully")
            logger.info(f"   - Global mode: {approval_config['global_mode']}")
            logger.info(f"   - Capabilities configured: {list(approval_config['capabilities'].keys())}")
        except Exception as e:
            # Security-critical component failure requires immediate attention
            logger.error(f"Failed to initialize ApprovalManager with valid config: {e}")
            raise ValueError(f"Invalid approval configuration structure: {e}") from e

    return _approval_manager


def get_python_execution_evaluator() -> PythonExecutionApprovalEvaluator:
    """Get Python execution approval evaluator with current system configuration.

    Convenience function that combines approval manager access and evaluator
    creation in a single call. This provides a streamlined interface for
    capabilities that need to evaluate Python code approval requirements
    without directly managing configuration details.

    The returned evaluator is configured with the current system settings,
    including any global mode overrides and capability-specific configuration.
    The evaluator is ready to use immediately for approval decisions.

    :return: Fully configured evaluator instance ready for approval decisions
    :rtype: PythonExecutionApprovalEvaluator
    :raises ValueError: If approval configuration is missing or invalid

    Examples:
        Evaluate code approval requirement::\n
            >>> evaluator = get_python_execution_evaluator()
            >>> decision = evaluator.evaluate(
            ...     has_epics_writes=True,
            ...     has_epics_reads=False
            ... )
            >>> if decision.needs_approval:
            ...     print(f"Approval required: {decision.reasoning}")

        Use in capability implementation::\n
            >>> def execute_python_code(code: str):
            ...     evaluator = get_python_execution_evaluator()
            ...     # Analyze code for EPICS operations (implementation specific)
            ...     has_writes = analyze_code_for_epics_writes(code)
            ...     decision = evaluator.evaluate(has_epics_writes=has_writes, has_epics_reads=False)
            ...     return decision

    .. note::
       This function is a convenience wrapper around get_approval_manager()
       and manager.get_python_execution_evaluator().
    """
    manager = get_approval_manager()
    return manager.get_python_execution_evaluator()


def get_memory_evaluator() -> MemoryApprovalEvaluator:
    """Get memory approval evaluator with current system configuration.

    Convenience function that combines approval manager access and evaluator
    creation in a single call. This provides a streamlined interface for
    capabilities that need to evaluate memory operation approval requirements
    without directly managing configuration details.

    The returned evaluator is configured with the current system settings,
    including any global mode overrides and memory-specific configuration.
    The evaluator is ready to use immediately for memory operation approval decisions.

    :return: Fully configured evaluator instance ready for approval decisions
    :rtype: MemoryApprovalEvaluator
    :raises ValueError: If approval configuration is missing or invalid

    Examples:
        Evaluate memory operation approval::\n
            >>> evaluator = get_memory_evaluator()
            >>> decision = evaluator.evaluate(operation_type="create")
            >>> if decision.needs_approval:
            ...     print(f"Memory approval required: {decision.reasoning}")

        Use in memory service::\n
            >>> def save_memory(content: str, user_id: str):
            ...     evaluator = get_memory_evaluator()
            ...     decision = evaluator.evaluate(operation_type="save")
            ...     if decision.needs_approval:
            ...         # Create approval interrupt
            ...         return create_memory_approval_interrupt(content, "save", user_id)
            ...     else:
            ...         # Proceed with save operation
            ...         return save_to_memory(content, user_id)

    .. note::
       This function is a convenience wrapper around get_approval_manager()
       and manager.get_memory_evaluator().
    """
    manager = get_approval_manager()
    return manager.get_memory_evaluator()
