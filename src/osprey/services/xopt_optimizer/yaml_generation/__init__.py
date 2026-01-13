"""YAML Generation Subsystem for XOpt Optimizer.

This subsystem generates XOpt YAML configurations using either:
1. ReAct mode (default): Agent-based generation that dynamically adapts:
   - If example files exist: Agent reads them and learns patterns
   - If no examples: Agent generates from built-in XOpt knowledge
2. Mock mode: Placeholder YAML for quick testing (use for fast iteration)

The mode is controlled via configuration:
    osprey.xopt_optimizer.yaml_generation.mode: "react" | "mock"

When using ReAct mode with examples, place YAML files in:
    osprey.xopt_optimizer.yaml_generation.examples_path: "path/to/yamls"

Example files are optional - the agent adapts its behavior based on availability.
"""

from .agent import YamlGenerationAgent, create_yaml_generation_agent
from .node import create_yaml_generation_node

__all__ = [
    "create_yaml_generation_node",
    "YamlGenerationAgent",
    "create_yaml_generation_agent",
]
