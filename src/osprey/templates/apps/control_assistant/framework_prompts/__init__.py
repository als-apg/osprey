"""Control Assistant framework prompts package.

This package provides custom prompt builders for control system operations.
The framework automatically creates a provider that uses these custom builders
while falling back to framework defaults for everything else.
"""

from .python import ControlSystemPythonPromptBuilder

__all__ = [
    "ControlSystemPythonPromptBuilder",
]

