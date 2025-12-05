"""Code Execution Subsystem.

This module provides the execution infrastructure for running generated Python code
in isolated environments, including container-based and local execution engines.

Components:
    - create_executor_node: LangGraph node for code execution
    - ContainerExecutor: Container-based execution engine
    - ExecutionWrapper: Execution wrapper utilities
    - ExecutionControl: Execution control and monitoring logic

The execution subsystem handles secure code execution with support for
multiple execution environments (container, local) and comprehensive
result collection.

Examples:
    Creating executor node::

        >>> from osprey.services.python_executor.execution import create_executor_node
        >>> node = create_executor_node()

    Direct container execution::

        >>> from osprey.services.python_executor.execution.container_engine import ContainerExecutor
        >>> executor = ContainerExecutor(config)
        >>> result = await executor.execute_code(code, context)
"""

from .node import create_executor_node

# Container engine and other components imported directly when needed
__all__ = [
    "create_executor_node",
]
