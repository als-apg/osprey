"""Code Analysis Subsystem — pattern detection for control system operations.

Provides functions to detect control system read/write operations in code,
used by the approval hook to determine whether human review is needed.

Examples:
    Using pattern detection::

        >>> from osprey.services.python_executor.analysis import detect_control_system_operations
        >>> result = detect_control_system_operations(code)
        >>> if result['has_writes']:
        ...     print("Code contains write operations")
"""

from .pattern_detection import (
    detect_control_system_operations,
    get_framework_standard_patterns,
)

__all__ = [
    "detect_control_system_operations",
    "get_framework_standard_patterns",
]
