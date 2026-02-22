"""Unified Approval System for Production-Ready Workflow Management.

This module provides a comprehensive approval system for clean approval handling
across all capabilities. The system enables secure, configurable approval workflows
for operations requiring human oversight.

The approval system consists of several key components:

1. **Configuration Management**: Type-safe configuration loading and validation
2. **Business Logic Evaluators**: Capability-specific approval decision logic
3. **Policy Management**: Centralized approval policy configuration

Key Features:
    - Configurable approval modes (disabled, selective, all_capabilities)
    - Type-safe configuration management with validation
    - Clean separation between configuration and business logic
    - Production-ready error handling and logging

Examples:
    Configuration management::

        >>> from osprey.approval import get_approval_manager
        >>> manager = get_approval_manager()
        >>> config = manager.get_python_execution_config()
        >>> print(f"Python approval enabled: {config.enabled}")

.. note::
   This module automatically initializes approval configuration from the global
   config system. Ensure your config.yml contains proper approval settings.

.. warning::
   Approval configuration is security-critical. Missing or invalid configuration
   will cause immediate startup failures to maintain system security.
"""

from .approval_manager import ApprovalManager, get_approval_manager
from .config_models import (
    ApprovalMode,
    GlobalApprovalConfig,
    MemoryApprovalConfig,
    PythonExecutionApprovalConfig,
)
from .evaluators import ApprovalDecision, MemoryApprovalEvaluator, PythonExecutionApprovalEvaluator

__all__ = [
    # Configuration and policy management
    "ApprovalManager",
    "get_approval_manager",
    # Configuration models
    "ApprovalMode",
    "PythonExecutionApprovalConfig",
    "MemoryApprovalConfig",
    "GlobalApprovalConfig",
    # Business logic evaluators
    "ApprovalDecision",
    "PythonExecutionApprovalEvaluator",
    "MemoryApprovalEvaluator",
]
