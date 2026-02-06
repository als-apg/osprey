"""Framework Base Module - Core Components and Architecture

This module provides the foundational components for the Osprey Framework's
LangGraph-native architecture. It implements a convention-based system where
capabilities and infrastructure components define standard interfaces for
automatic discovery and integration.

The module exports the complete set of base classes, decorators, and utilities
needed to build agents with standardized error handling, execution tracking,
and state management. All components are designed for seamless LangGraph
integration with proper streaming, configuration, and checkpoint support.

Key Components:
    - BaseCapability: Convention-based capability development with configuration-driven registration
    - BaseInfrastructureNode: Infrastructure components for orchestration and routing
    - Decorators: @capability_node and @infrastructure_node for LangGraph integration
    - Result Types: Comprehensive result and execution tracking hierarchy
    - Planning Framework: TypedDict-based execution planning for serialization
    - Error Classification: Sophisticated error handling with recovery strategies
    - Example System: Few-shot learning and orchestration guidance framework

The architecture emphasizes:
    - Convention over configuration for rapid development
    - Comprehensive error classification and recovery
    - Pure LangGraph integration with native features
    - Type safety through TypedDict and Pydantic models
    - Unified state management across all components

.. note::
   All base classes use reflection-based validation to ensure required
   components are properly implemented. Classes fail fast at decoration
   time with clear error messages if requirements are not met.

.. seealso::
   :mod:`osprey.state` : State management and agent state definitions
   :mod:`osprey.registry` : Component discovery and registration system
"""

from .capability import BaseCapability, slash_command
from .decorators import capability_node, infrastructure_node
from .errors import ErrorSeverity, ExecutionError
from .examples import (
    BaseExample,
    ClassifierActions,
    ClassifierExample,
    OrchestratorExample,
    OrchestratorGuide,
    TaskClassifierGuide,
)
from .nodes import BaseInfrastructureNode
from .planning import ExecutionPlan, PlannedStep
from .results import CapabilityMatch, ExecutionRecord, ExecutionResult

__all__ = [
    "BaseCapability",
    "BaseInfrastructureNode",
    "capability_node",
    "infrastructure_node",
    "ExecutionResult",
    "ExecutionRecord",
    "CapabilityMatch",
    "PlannedStep",
    "ExecutionPlan",
    "ErrorSeverity",
    "ExecutionError",
    "BaseExample",
    "OrchestratorExample",
    "OrchestratorGuide",
    "ClassifierExample",
    "TaskClassifierGuide",
    "ClassifierActions",
    "slash_command",
]
