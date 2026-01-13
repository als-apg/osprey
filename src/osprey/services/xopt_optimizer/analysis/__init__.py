"""Analysis Subsystem for XOpt Optimizer.

This subsystem analyzes XOpt results and decides whether to continue
with additional iterations or complete the optimization.
"""

from .node import create_analysis_node

__all__ = ["create_analysis_node"]
