"""State Identification Subsystem for XOpt Optimizer.

This subsystem assesses machine readiness for optimization using a ReAct agent
with tools for reading reference files and channel values.

Supports two modes:
- "react": ReAct agent with tools for reading reference files and channels (default)
- "mock": Fast placeholder that always returns READY

Configuration:
    xopt_optimizer:
      state_identification:
        mode: "react"  # or "mock"
        mock_files: true  # Use mock file data (for testing without real files)
        reference_path: "path/to/docs"  # Optional path to reference files
        model_config_name: "xopt_state_identification"
"""

from .agent import StateIdentificationAgent, create_state_identification_agent
from .node import create_state_identification_node

__all__ = [
    "create_state_identification_node",
    "create_state_identification_agent",
    "StateIdentificationAgent",
]
