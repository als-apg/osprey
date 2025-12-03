"""Code Generation Subsystem.

This module provides the code generation infrastructure for the Python executor,
including the protocol interface, factory pattern, and multiple generator implementations.

Components:
    - CodeGenerator: Protocol interface all generators must implement
    - create_code_generator: Factory function for creating generators
    - BasicLLMCodeGenerator: Simple single-pass LLM-based generator
    - ClaudeCodeGenerator: Claude Code SDK-based generator (optional dependency)
    - MockCodeGenerator: Mock generator for testing (no external dependencies)
    - create_generator_node: LangGraph node for code generation

The subsystem integrates with Osprey's registry system for generator discovery
and supports both framework and application-registered custom generators.

Examples:
    Basic usage::

        >>> from osprey.services.python_executor.generation import create_code_generator
        >>> generator = create_code_generator()
        >>> code = await generator.generate_code(request, [])

    Registry-based custom generator::

        >>> # In your application's registry.py
        >>> CodeGeneratorRegistration(
        ...     name="my_generator",
        ...     module_path="myapp.generators",
        ...     class_name="MyGenerator",
        ...     description="Custom generator"
        ... )

    Testing with mock generator::

        >>> from osprey.services.python_executor.generation import MockCodeGenerator
        >>> generator = MockCodeGenerator(behavior="success")
        >>> code = await generator.generate_code(request, [])
"""

from .factory import create_code_generator
from .interface import CodeGenerator
from .basic_generator import BasicLLMCodeGenerator
from .mock_generator import MockCodeGenerator
from .node import create_generator_node

# Try to import Claude Code generator (optional dependency)
try:
    from .claude_code_generator import CLAUDE_SDK_AVAILABLE, ClaudeCodeGenerator
except ImportError:
    ClaudeCodeGenerator = None  # type: ignore
    CLAUDE_SDK_AVAILABLE = False

__all__ = [
    # Protocol
    "CodeGenerator",
    # Factory
    "create_code_generator",
    # Generators
    "BasicLLMCodeGenerator",
    "ClaudeCodeGenerator",  # Included as core dependency (v0.9.6+)
    "MockCodeGenerator",  # For testing
    "CLAUDE_SDK_AVAILABLE",
    # Node
    "create_generator_node",
]

