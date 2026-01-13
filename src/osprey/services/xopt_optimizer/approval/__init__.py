"""Approval Subsystem for XOpt Optimizer.

This subsystem handles human approval for XOpt configurations using
the standard Osprey LangGraph interrupt pattern.
"""

from .node import create_approval_node

__all__ = ["create_approval_node"]
