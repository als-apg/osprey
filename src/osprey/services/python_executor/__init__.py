"""Python Executor Service — code execution and safety framework.

Provides secure Python code execution with safety checks, pattern detection,
container isolation, and channel write limits validation.

Core Components:
    - :class:`PythonExecutionRequest`: Type-safe execution request with context data
    - :class:`FileManager`: File operations and execution folder management
    - :class:`NotebookManager`: Jupyter notebook creation and management
    - :class:`LimitsValidator`: Channel write safety limits enforcement
"""

from .analysis import (
    detect_control_system_operations,
    get_framework_standard_patterns,
)
from .exceptions import (
    ChannelLimitsViolationError,
    CodeRuntimeError,
    ContainerConfigurationError,
    ContainerConnectivityError,
    ErrorCategory,
    ExecutionTimeoutError,
    PythonExecutorException,
)
from .execution.control import ExecutionControlConfig, ExecutionMode
from osprey.connectors.control_system.limits_validator import LimitsValidator
from .models import (
    ContainerEndpointConfig,
    ExecutionModeConfig,
    NotebookAttempt,
    NotebookType,
    PythonExecutionContext,
    PythonExecutionRequest,
    PythonExecutionSuccess,
    PythonServiceResult,
)
from .services import (
    FileManager,
    NotebookManager,
    make_json_serializable,
    serialize_results_to_file,
)

__all__ = [
    # Core types
    "PythonExecutionRequest",
    "PythonExecutionSuccess",
    "PythonServiceResult",
    # Analysis utilities
    "detect_control_system_operations",
    "get_framework_standard_patterns",
    # Execution context and notebook management
    "NotebookAttempt",
    "NotebookType",
    "PythonExecutionContext",
    "FileManager",
    "NotebookManager",
    # Limits validation
    "LimitsValidator",
    # Configuration
    "ExecutionModeConfig",
    "ContainerEndpointConfig",
    "ExecutionMode",
    "ExecutionControlConfig",
    # Exception hierarchy
    "PythonExecutorException",
    "ErrorCategory",
    "ContainerConnectivityError",
    "ContainerConfigurationError",
    "CodeRuntimeError",
    "ChannelLimitsViolationError",
    "ExecutionTimeoutError",
    # Serialization utilities
    "make_json_serializable",
    "serialize_results_to_file",
]
