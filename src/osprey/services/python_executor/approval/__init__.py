"""Human-in-the-Loop Approval Subsystem.

This module provides the approval workflow infrastructure for human oversight
of code execution, integrating with LangGraph's interrupt system.

Components:
    - create_approval_node: LangGraph node for human approval workflows

The approval subsystem enables production-ready human-in-the-loop patterns
for high-stakes environments requiring manual approval before code execution.

Examples:
    Creating approval node::

        >>> from osprey.services.python_executor.approval import create_approval_node
        >>> node = create_approval_node()
"""

from .node import create_approval_node

__all__ = [
    "create_approval_node",
]
