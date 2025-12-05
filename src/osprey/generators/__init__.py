"""Code generation utilities for Osprey Framework.

This package provides generators for creating Osprey components from
various sources (MCP servers, OpenAPI specs, natural language prompts, etc.).

Architecture:
- base_generator: BaseCapabilityGenerator with shared functionality
- models: Shared Pydantic models for LLM analysis
- generate_from_mcp: Generate from MCP servers
- generate_from_prompt: Generate from natural language prompts
- registry_updater: Auto-register generated capabilities
- config_updater: Auto-update config files
"""

from . import config_updater, registry_updater
from .base_generator import BaseCapabilityGenerator
from .generate_from_mcp import MCPCapabilityGenerator
from .generate_from_prompt import PromptCapabilityGenerator
from .models import (
    CapabilityMetadata,
    ClassifierAnalysis,
    ClassifierExampleRaw,
    ExampleStepRaw,
    OrchestratorAnalysis,
    ToolPattern,
)

__all__ = [
    # Generators
    "BaseCapabilityGenerator",
    "MCPCapabilityGenerator",
    "PromptCapabilityGenerator",
    # Models
    "CapabilityMetadata",
    "ClassifierAnalysis",
    "ClassifierExampleRaw",
    "ExampleStepRaw",
    "OrchestratorAnalysis",
    "ToolPattern",
    # Utilities
    "registry_updater",
    "config_updater",
]
