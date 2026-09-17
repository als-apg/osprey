"""Execution Control Configuration for Python Executor Service.

This module provides specialized execution control configuration for the Python
executor service, with particular focus on control-system integration and
security policies. It defines execution modes that determine the level of system
access and control permissions available to executed Python code.

The module implements a security-conscious approach to execution control, providing
clear separation between read-only operations (safe for automated execution) and
write operations (requiring additional approval and oversight). This is particularly
important in scientific and industrial control environments where code execution
can have real-world physical consequences.

Key Components:
    - **ExecutionMode**: Enumeration of available execution environments with
      different security and access profiles
    - **ExecutionControlConfig**: Configuration class carrying the write
      posture for the control target it was built for, and the config key that
      governs that posture
    - **Configuration Utilities**: Helper functions for creating and validating
      execution control configurations

The write gate reads ``control_system_writes_enabled`` off this config and
combines it with what static analysis found in the code: write access needs
both the configured permission and explicit write intent in the code itself.

.. note::
   The posture is about control-system access in general; it reads no
   protocol-specific fact and holds for every connector the deployment can
   select.

.. warning::
   Write-enabled execution modes can perform system operations with real-world
   consequences. Ensure proper approval workflows are configured before enabling
   write access in production environments.

Examples:
    Configuration validation::

        >>> config = ExecutionControlConfig(control_system_writes_enabled=True)
        >>> warnings = config.validate()
        >>> if warnings:
        ...     print(f"Configuration warnings: {warnings}")
"""

from dataclasses import dataclass
from enum import Enum

from osprey.connectors.types import (
    MOCK,
    WRITES_ENABLED_KEY,
    baseline_target,
    target_writes_enabled,
    target_writes_enabled_key,
)
from osprey.utils.logger import get_logger

logger = get_logger("execution_control")


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
       The execution mode gates what the generated code is permitted to do;
       enforcement happens before execution via approval workflows.

    .. warning::
       WRITE_ACCESS mode can perform operations with real-world consequences in
       control system environments. Use with appropriate approval workflows.

    .. seealso::
       :class:`ExecutionControlConfig` : Configuration logic for mode selection

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
    WRITE_ACCESS = "write_access"  # Live control-system write access (dangerous!)


@dataclass
class ExecutionControlConfig:
    """Configuration class for control system execution control and security policy management.

    This configuration class encapsulates the security policies and settings that
    determine how Python code execution is controlled within the system. It
    carries the answer; the write gate that acts on it lives with the executor
    tools.

    The configuration implements a conservative security approach where write
    operations are only permitted when explicitly enabled and detected in the
    code. This ensures that potentially dangerous operations require both
    configuration permission and explicit code intent.

    :param control_system_writes_enabled: Whether control system write operations are permitted for :attr:`active_target`
    :type control_system_writes_enabled: bool
    :param control_system_type: Type of control system (epics, mock, tango, etc.).
        Defaults to mock so an under-specified config never claims a live system.
    :type control_system_type: str
    :param active_target: Control target whose posture ``control_system_writes_enabled``
        answers, or ``None`` when the config was built without reading one.
    :type active_target: str | None
    :param writes_enabled_key: Dotted config key that governs that posture — the
        per-type block when the target resolves to a connector type, the
        deployment-wide key when it does not. Carried on the config rather than
        re-derived by a refusal, so an operator is never sent to a key that had
        no say in the answer.
    :type writes_enabled_key: str

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
    """

    # Control system settings
    control_system_writes_enabled: bool = False
    control_system_type: str = MOCK  # Fail-closed: never assume a live system
    active_target: str | None = None
    writes_enabled_key: str = WRITES_ENABLED_KEY

    def validate(self) -> list[str]:
        """
        Validate configuration for logical consistency.

        Returns:
            List of validation warnings/errors
        """
        warnings = []

        # Live writes are potentially dangerous - log warning
        if self.control_system_writes_enabled:
            warnings.append(
                f"WARNING: {self.writes_enabled_key}=true (live control-system writes enabled!)"
            )

        return warnings


def get_execution_control_config(target: str | None = None) -> ExecutionControlConfig:
    """
    Get execution control configuration from global config.

    This is the single entry point for getting execution control configuration.
    Write posture is per control target, so the answer is for one target:
    ``control_system.connector.<type>.writes_enabled`` for the type that target
    resolves to, inheriting ``control_system.writes_enabled`` where the block
    says nothing.

    Args:
        target: The control target. ``None`` — the caller has no
            target to name — answers the deployment's *baseline* target. Naming
            a target nobody selected is otherwise not something this codebase
            does; it is sound here because an unstamped run provably builds the
            baseline connector, off the same section this reads.

    Returns:
        ExecutionControlConfig instance with type-safe configuration
    """
    try:
        # Import here to avoid circular imports
        from osprey.utils.config import get_config_value

        control_system_config = get_config_value("control_system", {})
        active_target = target if target is not None else baseline_target(control_system_config)
        writes_enabled = target_writes_enabled(control_system_config, active_target)

        # Get control system type for proper configuration. Fail closed: an
        # unset (or blank) key must not be read as a live control system.
        control_system_type = control_system_config.get("type")
        if not control_system_type:
            logger.warning(
                f"control_system.type is not set; defaulting to '{MOCK}'. "
                f"Set control_system.type explicitly to select a connector."
            )
            control_system_type = MOCK

        # Build typed config with defaults
        execution_control = ExecutionControlConfig(
            control_system_writes_enabled=writes_enabled,
            control_system_type=control_system_type,
            active_target=active_target,
            writes_enabled_key=target_writes_enabled_key(control_system_config, active_target),
        )

        # Validate configuration and log warnings
        warnings = execution_control.validate()
        if warnings:
            for warning in warnings:
                logger.warning(f"Execution control config: {warning}")

        logger.debug(
            f"Loaded execution control config: writes_enabled={execution_control.control_system_writes_enabled}, "
            f"type={control_system_type}, target={active_target}, "
            f"key={execution_control.writes_enabled_key}"
        )

        return execution_control

    except Exception as e:
        logger.warning(f"Failed to load execution control config: {e}, using safe defaults")

        # Return safe defaults
        return ExecutionControlConfig(control_system_writes_enabled=False)
