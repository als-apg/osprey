"""Code Execution Subsystem.

This module provides the execution infrastructure for running generated Python code
in isolated environments, including container-based and local execution engines.

Components:
    - ContainerExecutor: Container-based execution engine
    - ExecutionWrapper: Execution wrapper utilities
    - ExecutionControl: Execution control and monitoring logic

The execution subsystem handles secure code execution with support for
multiple execution environments (container, local) and comprehensive
result collection.

Examples:
    Direct container execution::

        >>> from osprey.services.python_executor.execution.container_engine import ContainerExecutor
        >>> executor = ContainerExecutor(config)
        >>> result = await executor.execute_code(code, context)
"""

__all__: list[str] = []
