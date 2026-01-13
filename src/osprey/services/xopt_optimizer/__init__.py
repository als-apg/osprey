"""XOpt Optimizer Service - Autonomous Machine Optimization.

This service provides a LangGraph-based subsystem for autonomous machine optimization
in accelerator control environments. It follows the same architectural patterns as
the Python Executor Service, providing intelligent optimization workflows with
human approval, machine state awareness, and result analysis.

Key Components:
    - XOptOptimizerService: Main LangGraph orchestrator for optimization workflows
    - XOptExecutionRequest: Request model for optimization execution
    - XOptServiceResult: Structured result from optimization execution
    - XOptExecutionState: Internal LangGraph state for workflow tracking

Design Principles:
    - Framework-level service adaptable to any facility through configuration
    - Prompt builder system for facility-specific customization
    - Configuration-driven machine state definitions
    - Pluggable tools that leverage existing Osprey capabilities

.. seealso::
   :mod:`osprey.services.python_executor` : Similar service for Python execution
   :mod:`osprey.capabilities.optimization` : Capability that uses this service
"""

from .exceptions import (
    ConfigurationError,
    ErrorCategory,
    MachineStateAssessmentError,
    MaxIterationsExceededError,
    XOptExecutionError,
    XOptExecutorException,
    YamlGenerationError,
)
from .models import (
    MachineState,
    XOptError,
    XOptExecutionRequest,
    XOptExecutionState,
    XOptServiceResult,
    XOptStrategy,
)
from .service import XOptOptimizerService

__all__ = [
    # Service
    "XOptOptimizerService",
    # Models
    "XOptExecutionRequest",
    "XOptExecutionState",
    "XOptServiceResult",
    "XOptError",
    "MachineState",
    "XOptStrategy",
    # Exceptions
    "XOptExecutorException",
    "ErrorCategory",
    "MachineStateAssessmentError",
    "YamlGenerationError",
    "XOptExecutionError",
    "MaxIterationsExceededError",
    "ConfigurationError",
]
