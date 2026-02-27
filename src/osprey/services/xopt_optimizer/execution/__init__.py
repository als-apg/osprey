"""Execution Subsystem for XOpt Optimizer.

This subsystem executes XOpt optimization runs using the generated
YAML configuration.

PLACEHOLDER: Current implementation is a no-op placeholder.
Actual XOpt execution will be implemented when XOpt prototype
integration is ready.
"""

from .node import create_executor_node

__all__ = ["create_executor_node"]
