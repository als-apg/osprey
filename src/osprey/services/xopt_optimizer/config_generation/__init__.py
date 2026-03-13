"""Config Generation Subsystem for XOpt Optimizer.

This subsystem generates OptimizationConfig dicts for the tuning_scripts API using:
1. Structured mode: LLM with structured output fills in config fields
2. Mock mode: Placeholder config for quick testing (use for fast iteration)

The mode is controlled via configuration:
    osprey.xopt_optimizer.config_generation.mode: "structured" | "mock"
"""

from .agent import ConfigGenerationAgent, create_config_generation_agent
from .node import create_config_generation_node

__all__ = [
    "create_config_generation_node",
    "ConfigGenerationAgent",
    "create_config_generation_agent",
]
