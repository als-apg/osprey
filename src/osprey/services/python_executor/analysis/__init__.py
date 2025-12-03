"""Code Analysis Subsystem.

This module provides static analysis infrastructure for Python code,
including security pattern detection and execution policy analysis.

Components:
    - create_analyzer_node: LangGraph node for code analysis
    - detect_control_system_operations: Detect control system operations in code
    - get_framework_standard_patterns: Get framework-standard control-system-agnostic patterns
    - get_default_patterns: DEPRECATED - Get old nested pattern format (backward compat)
    - ExecutionPolicyAnalyzer: Execution mode and approval decision logic

The analysis subsystem validates generated code before execution,
detecting security risks and determining appropriate execution policies.

Examples:
    Creating analyzer node::

        >>> from osprey.services.python_executor.analysis import create_analyzer_node
        >>> node = create_analyzer_node()

    Using pattern detection::

        >>> from osprey.services.python_executor.analysis import detect_control_system_operations
        >>> result = detect_control_system_operations(code)
        >>> if result['has_writes']:
        ...     print("Code contains write operations")
"""

from .node import create_analyzer_node
from .pattern_detection import (
    detect_control_system_operations,
    get_default_patterns,
    get_framework_standard_patterns,
)

__all__ = [
    "create_analyzer_node",
    "detect_control_system_operations",
    "get_default_patterns",
    "get_framework_standard_patterns",
]

