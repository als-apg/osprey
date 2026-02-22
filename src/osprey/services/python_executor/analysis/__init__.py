"""Code Analysis Subsystem.

This module provides static analysis infrastructure for Python code,
including security pattern detection and execution policy analysis.

Components:
    - detect_control_system_operations: Detect control system operations in code
    - get_framework_standard_patterns: Get framework-standard control-system-agnostic patterns
    - get_default_patterns: DEPRECATED - Get old nested pattern format (backward compat)
    - ExecutionPolicyAnalyzer: Execution mode and approval decision logic

The analysis subsystem validates generated code before execution,
detecting security risks and determining appropriate execution policies.

Examples:
    Using pattern detection::

        >>> from osprey.services.python_executor.analysis import detect_control_system_operations
        >>> result = detect_control_system_operations(code)
        >>> if result['has_writes']:
        ...     print("Code contains write operations")
"""

from .pattern_detection import (
    detect_control_system_operations,
    get_default_patterns,
    get_framework_standard_patterns,
)

__all__ = [
    "detect_control_system_operations",
    "get_default_patterns",
    "get_framework_standard_patterns",
]
