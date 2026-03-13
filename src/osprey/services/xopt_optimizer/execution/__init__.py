"""Execution Subsystem for XOpt Optimizer.

This subsystem executes XOpt optimization runs by submitting optimization
configs to the tuning_scripts API and polling for results.
Falls back to placeholder execution when the API is unavailable.
"""

from .api_client import TuningScriptsAPIError, TuningScriptsClient
from .node import create_executor_node

__all__ = ["TuningScriptsAPIError", "TuningScriptsClient", "create_executor_node"]
