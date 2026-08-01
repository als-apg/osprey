"""Python Executor Service — code execution and safety framework.

Provides safety analysis and result serialization for agent Python
execution: pattern detection for control-system operations, execution-mode
gating, channel write limits validation, and robust JSON serialization of
scientific results.
"""

from osprey.connectors.control_system.limits_validator import LimitsValidator

from .analysis import (
    detect_control_system_operations,
    get_framework_standard_patterns,
)
from .execution.control import ExecutionControlConfig, ExecutionMode
from .services import (
    make_json_serializable,
    serialize_results_to_file,
)

__all__ = [
    # Analysis utilities
    "detect_control_system_operations",
    "get_framework_standard_patterns",
    # Limits validation
    "LimitsValidator",
    # Configuration
    "ExecutionMode",
    "ExecutionControlConfig",
    # Serialization utilities
    "make_json_serializable",
    "serialize_results_to_file",
]
