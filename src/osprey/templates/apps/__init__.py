"""Application code templates for scaffolding.

This package contains templates for different application types:

- control_assistant/ : Control system integration with channel finder (production-grade)

Each template contains:
- registry.py.j2 : Component registration
- context_classes.py.j2 : Data structures
- capabilities/ : Capability implementations
- Other application-specific files

Template files use Jinja2 syntax ({{ variable }}) for customization during
project generation. Static files (without .j2 extension) are copied as-is.
"""

__all__ = []
