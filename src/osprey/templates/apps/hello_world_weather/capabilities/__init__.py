"""
Hello World Weather Capabilities Module.

Provides weather-related capabilities for the Hello World Weather tutorial
application within the Osprey Agent Framework. This module serves as
the central export point for all weather capabilities, enabling clean imports
and maintaining clear module organization.

The capabilities module demonstrates essential patterns for organizing and
exporting framework capabilities, including proper module structure, import
management, and __all__ declarations for explicit public API definition.

Architecture Integration:
    This module integrates with the framework's capability system by:

    1. **Capability Export**: Provides clean import paths for capability classes
    2. **Module Organization**: Maintains clear separation between different capabilities
    3. **Public API**: Defines explicit public interface through __all__ declaration
    4. **Framework Integration**: Enables automatic discovery and registration
    5. **Development Support**: Facilitates testing and development workflows

Capability Overview:
    The module exports the following weather capabilities:

    - **CurrentWeatherCapability**: Retrieves current weather conditions for
      supported locations using mock weather service integration

.. note::
   This module follows Python packaging best practices with explicit __all__
   declarations and clean import structures. The pattern can be extended for
   applications with multiple capabilities.

.. warning::
   All exported capabilities must be properly registered in the application's
   registry configuration to be available for framework execution.
"""

from .current_weather import CurrentWeatherCapability

__all__ = ["CurrentWeatherCapability"]
"""Explicit public API declaration for Hello World Weather capabilities.

Defines the complete public interface for the weather capabilities module,
ensuring clean imports and explicit API boundaries. This list should include
all capability classes that are intended for external use by the framework
and other application components.

Exported Capabilities:
    - **CurrentWeatherCapability**: Weather data retrieval capability for current conditions

.. note::
   The __all__ declaration ensures that 'from capabilities import *' imports
   only the intended public classes, preventing accidental exposure of internal
   implementation details or utility functions.

.. warning::
   All capabilities listed in __all__ must be properly imported above and
   should be registered in the application's registry configuration for
   framework availability.

Examples:
    Importing specific capabilities::

        >>> from hello_world_weather.capabilities import CurrentWeatherCapability
        >>> capability = CurrentWeatherCapability()
        >>> print(capability.name)
        current_weather

    Importing all public capabilities::

        >>> from hello_world_weather.capabilities import *
        >>> # Only CurrentWeatherCapability is imported due to __all__ declaration

    Framework registry integration::

        >>> # In registry.py
        >>> from hello_world_weather.capabilities import CurrentWeatherCapability
        >>> # Capability is available for registration and framework integration

.. seealso::
   :class:`CurrentWeatherCapability` : Primary weather capability exported by this module
   :mod:`hello_world_weather.registry` : Registry configuration using these capabilities
   :doc:`/developer-guides/capability-organization` : Capability module organization guide
"""
