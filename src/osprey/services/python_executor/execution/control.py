"""Execution Control Configuration for Python Executor Service.

This module provides specialized execution control configuration for the Python
executor service, with particular focus on EPICS (Experimental Physics and
Industrial Control System) integration and security policies. It defines execution
modes that determine the level of system access and control permissions available
to executed Python code.

The module implements a security-conscious approach to execution control, providing
clear separation between read-only operations (safe for automated execution) and
write operations (requiring additional approval and oversight). This is particularly
important in scientific and industrial control environments where code execution
can have real-world physical consequences.

Key Components:
    - **ExecutionMode**: Enumeration of available execution environments with
      different security and access profiles
    - **ExecutionControlConfig**: Configuration class that determines execution
      mode selection based on code analysis and security policies
    - **Configuration Utilities**: Helper functions for creating and validating
      execution control configurations

The execution control system integrates with the static code analysis pipeline
to automatically determine appropriate execution environments based on the
operations detected in generated code, while providing override capabilities
for manual control when needed.

.. note::
   This module is specifically designed for EPICS integration but can be extended
   for other control systems or security-sensitive environments.

.. warning::
   Write-enabled execution modes can perform system operations with real-world
   consequences. Ensure proper approval workflows are configured before enabling
   write access in production environments.

.. seealso::
   :class:`osprey.services.python_executor.execution_policy_analyzer` : Code analysis integration
   :class:`osprey.services.python_executor.models.ExecutionModeConfig` : Container configuration
   :class:`osprey.services.python_executor.analysis.node` : Analysis node using these modes

Examples:
    Basic execution control configuration::

        >>> config = ExecutionControlConfig(epics_writes_enabled=False)
        >>> mode = config.get_execution_mode(
        ...     has_epics_writes=True,
        ...     has_epics_reads=True
        ... )
        >>> print(f"Selected mode: {mode}")
        Selected mode: ExecutionMode.READ_ONLY

    Enabling write operations with proper safeguards::

        >>> write_config = ExecutionControlConfig(epics_writes_enabled=True)
        >>> mode = write_config.get_execution_mode(
        ...     has_epics_writes=True,
        ...     has_epics_reads=False
        ... )
        >>> print(f"Write mode: {mode}")
        Write mode: ExecutionMode.WRITE_ACCESS

    Configuration validation::

        >>> warnings = config.validate()
        >>> if warnings:
        ...     print(f"Configuration warnings: {warnings}")
"""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Enumeration of Python execution environment modes with different security profiles.

    This enum defines the available execution environments for Python code execution,
    each with different levels of system access and security constraints. The modes
    are designed to provide appropriate isolation and control for different types
    of operations, particularly in scientific and industrial control environments.

    The execution modes form a security hierarchy from most restrictive (READ_ONLY)
    to least restrictive (WRITE_ACCESS), allowing fine-grained control over the
    capabilities available to executed code.

    :cvar READ_ONLY: Safe, isolated environment for read-only operations and analysis
    :cvar WRITE_ACCESS: Full-access environment enabling system writes and control operations

    .. note::
       Each execution mode corresponds to a specific Jupyter container configuration
       with appropriate kernel settings, environment variables, and access controls.

    .. warning::
       WRITE_ACCESS mode can perform operations with real-world consequences in
       control system environments. Use with appropriate approval workflows.

    .. seealso::
       :class:`ExecutionControlConfig` : Configuration logic for mode selection
       :class:`osprey.services.python_executor.models.ExecutionModeConfig` : Container settings

    Examples:
        Mode selection based on operation requirements::

            >>> # Safe analysis operations
            >>> mode = ExecutionMode.READ_ONLY
            >>> print(f"Safe mode: {mode.value}")
            Safe mode: read_only

            >>> # Control operations requiring write access
            >>> mode = ExecutionMode.WRITE_ACCESS
            >>> print(f"Control mode: {mode.value}")
            Control mode: write_access
    """

    READ_ONLY = "read_only"  # Safe read-only operations only
    WRITE_ACCESS = "write_access"  # Live EPICS write access (dangerous!)


@dataclass
class ExecutionControlConfig:
    """Configuration class for control system execution control and security policy management.

    This configuration class encapsulates the security policies and settings that
    determine how Python code execution is controlled within the system. It provides
    the logic for automatically selecting appropriate execution environments based
    on code analysis results and configured security policies.

    The configuration implements a conservative security approach where write
    operations are only permitted when explicitly enabled and detected in the
    code. This ensures that potentially dangerous operations require both
    configuration permission and explicit code intent.

    :param epics_writes_enabled: (Deprecated) Whether EPICS write operations are permitted.
                                 Use control_system_writes_enabled instead.
    :type epics_writes_enabled: bool
    :param control_system_writes_enabled: Whether control system write operations are permitted in this deployment
    :type control_system_writes_enabled: bool
    :param control_system_type: Type of control system (epics, mock, tango, etc.)
    :type control_system_type: str

    .. note::
       This configuration should be set based on the deployment environment and
       security requirements. Production control systems should carefully consider
       the implications of enabling write access.

    .. warning::
       Enabling control system writes allows executed code to potentially affect physical
       systems. Ensure appropriate approval workflows and monitoring are in place.

    .. seealso::
       :class:`ExecutionMode` : Available execution environment modes
       :func:`get_execution_control_config` : Factory function for creating configurations

    Examples:
        Creating a read-only configuration for safe analysis::

            >>> config = ExecutionControlConfig(control_system_writes_enabled=False)
            >>> mode = config.get_execution_mode(has_epics_writes=True, has_epics_reads=True)
            >>> print(f"Mode: {mode}")  # Always READ_ONLY when writes disabled
            Mode: ExecutionMode.READ_ONLY

        Enabling controlled write access::

            >>> write_config = ExecutionControlConfig(control_system_writes_enabled=True)
            >>> # Only grants write access when code actually contains write operations
            >>> read_mode = write_config.get_execution_mode(has_epics_writes=False, has_epics_reads=True)
            >>> write_mode = write_config.get_execution_mode(has_epics_writes=True, has_epics_reads=True)
            >>> print(f"Read mode: {read_mode}, Write mode: {write_mode}")
    """

    # Control system settings
    epics_writes_enabled: bool = False  # Deprecated - kept for backward compatibility
    control_system_writes_enabled: bool | None = None
    control_system_type: str = "epics"  # Default for backward compatibility

    def __post_init__(self):
        """Handle backward compatibility for epics_writes_enabled."""
        # If control_system_writes_enabled not explicitly set, use epics_writes_enabled
        if self.control_system_writes_enabled is None:
            self.control_system_writes_enabled = self.epics_writes_enabled

    def get_execution_mode(self, has_epics_writes: bool, has_epics_reads: bool) -> ExecutionMode:
        """Determine appropriate execution mode based on code analysis and security policy.

        Analyzes the detected operations in the code (from static analysis) and
        applies the configured security policy to determine the most appropriate
        execution environment. The method implements a conservative approach where
        write access is only granted when both the code requires it and the
        configuration permits it.

        The decision logic prioritizes security by defaulting to read-only access
        unless write operations are both detected in the code and explicitly
        enabled in the configuration.

        :param has_epics_writes: Whether static analysis detected EPICS write operations in the code
        :type has_epics_writes: bool
        :param has_epics_reads: Whether static analysis detected EPICS read operations in the code
        :type has_epics_reads: bool
        :return: Execution mode appropriate for the detected operations and security policy
        :rtype: ExecutionMode

        .. note::
           The has_epics_reads parameter is provided for future extensibility but
           currently does not affect mode selection since read operations are
           permitted in all execution modes.

        Examples:
            Mode selection with different code patterns::

                >>> config = ExecutionControlConfig(epics_writes_enabled=True)
                >>>
                >>> # Code with only read operations
                >>> mode = config.get_execution_mode(has_epics_writes=False, has_epics_reads=True)
                >>> print(f"Read-only code: {mode}")
                Read-only code: ExecutionMode.READ_ONLY
                >>>
                >>> # Code with write operations (and writes enabled)
                >>> mode = config.get_execution_mode(has_epics_writes=True, has_epics_reads=True)
                >>> print(f"Write code: {mode}")
                Write code: ExecutionMode.WRITE_ACCESS

            Security policy enforcement::

                >>> secure_config = ExecutionControlConfig(epics_writes_enabled=False)
                >>> # Write operations detected but not permitted by policy
                >>> mode = secure_config.get_execution_mode(has_epics_writes=True, has_epics_reads=True)
                >>> print(f"Secured mode: {mode}")  # Always READ_ONLY when writes disabled
                Secured mode: ExecutionMode.READ_ONLY
        """
        if has_epics_writes and self.epics_writes_enabled:
            return ExecutionMode.WRITE_ACCESS
        else:
            return ExecutionMode.READ_ONLY

    def validate(self) -> list[str]:
        """
        Validate configuration for logical consistency.

        Returns:
            List of validation warnings/errors
        """
        warnings = []

        # Live writes are potentially dangerous - log warning
        if self.epics_writes_enabled:
            warnings.append("WARNING: epics.writes_enabled=true (live EPICS writes enabled!)")

        return warnings


def get_execution_control_config() -> ExecutionControlConfig:
    """
    Get execution control configuration from global config.

    This is the single entry point for getting execution control configuration.
    Supports both the new location (control_system.writes_enabled) and the
    deprecated location (execution_control.epics.writes_enabled) for backward compatibility.

    Returns:
        ExecutionControlConfig instance with type-safe configuration
    """
    try:
        # Import here to avoid circular imports
        from osprey.utils.config import get_config_value

        # Try new location first (control_system.writes_enabled)
        control_system_config = get_config_value("control_system", {})
        writes_enabled_new = control_system_config.get("writes_enabled")

        # Try deprecated location (execution_control.epics.writes_enabled)
        exec_config = get_config_value("execution_control", {})
        writes_enabled_old = exec_config.get("epics", {}).get("writes_enabled")

        # Determine which value to use (new location takes precedence)
        if writes_enabled_new is not None:
            writes_enabled = writes_enabled_new
            if writes_enabled_old is not None and writes_enabled_old != writes_enabled_new:
                logger.warning(
                    "⚠️  Both 'control_system.writes_enabled' and 'execution_control.epics.writes_enabled' are set with different values!"
                )
                logger.warning(
                    f"   Using NEW location value: control_system.writes_enabled = {writes_enabled_new}"
                )
        elif writes_enabled_old is not None:
            writes_enabled = writes_enabled_old
            logger.warning("⚠️  DEPRECATED: 'execution_control.epics.writes_enabled' is deprecated.")
            logger.warning(
                "   Please move this setting to 'control_system.writes_enabled' in your config.yml"
            )
        else:
            writes_enabled = False  # Safe default

        # Get control system type for proper configuration
        control_system_type = control_system_config.get("type", "epics")

        # Build typed config with defaults
        execution_control = ExecutionControlConfig(
            epics_writes_enabled=writes_enabled,  # Kept for backward compat
            control_system_writes_enabled=writes_enabled,
            control_system_type=control_system_type,
        )

        # Validate configuration and log warnings
        warnings = execution_control.validate()
        if warnings:
            for warning in warnings:
                logger.warning(f"Execution control config: {warning}")

        logger.debug(
            f"Loaded execution control config: writes_enabled={execution_control.control_system_writes_enabled}, type={control_system_type}"
        )

        return execution_control

    except Exception as e:
        logger.warning(f"Failed to load execution control config: {e}, using safe defaults")

        # Return safe defaults
        return ExecutionControlConfig(
            epics_writes_enabled=False, control_system_writes_enabled=False
        )
