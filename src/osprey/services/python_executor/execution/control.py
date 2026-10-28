"""Execution modes for the Python executor service.

This module defines the execution modes that determine the level of system
access available to executed Python code: a clear separation between read-only
operations (safe for automated execution) and write operations (requiring
approval and oversight). This matters in scientific and industrial control
environments, where code execution can have real-world physical consequences.

Whether a run may write is not decided here. It is the write verdict's to decide
(:func:`osprey_connectors.posture_store.write_verdict`): the executor's entry
gate, its launch pin and the connector's reference monitor all ask it about the
control target the run is on, and refuse in its words.

.. warning::
   Write-enabled execution modes can perform system operations with real-world
   consequences. Ensure proper approval workflows are configured before enabling
   write access in production environments.
"""

from enum import Enum


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
