"""The framework's own exception classes.

Failures of the framework's configuration, build profiles and registry, which
the connectors package never raises. The connectors package keeps its own
vocabulary in :mod:`osprey_connectors.errors`; its
:class:`~osprey.errors.ConfigurationError` is what the package's config
readers raise, a ``ValueError`` naming the dotted key, and is a different class
from :class:`ConfigurationError` here, which is no ``ValueError``: an
``except ValueError`` around framework code does not swallow a framework
configuration refusal.

This module imports nothing, so every layer of the framework can raise from it.
"""

__all__ = ["BuildProfileError", "ConfigurationError", "RegistryError"]


class ConfigurationError(Exception):
    """Exception for configuration-related errors.

    Raised when configuration files are invalid, missing required settings,
    or contain incompatible values that prevent proper system operation.
    """


class BuildProfileError(ConfigurationError):
    """Raised when a build profile is invalid or missing required files."""


class RegistryError(Exception):
    """Exception for registry-related errors.

    Raised when issues occur with component registration, lookup, or
    management within the framework's registry system.
    """
