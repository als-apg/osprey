"""Osprey Agent Framework.

Core framework package providing infrastructure for building
intelligent agents with specialized capabilities.

This package contains:
- Base classes and interfaces
- Service integrations
- Configuration management

Call ``osprey.configure_logging()`` once at startup to see Osprey's log output;
importing the package configures nothing.
"""

from typing import TYPE_CHECKING, Any

from osprey.version import get_running_version

if TYPE_CHECKING:
    from osprey.utils.logger import configure_logging

# Version information. Derived from the git tag at build time and resolved at import
# by osprey.version — see that module for the resolution chain. Bound eagerly rather
# than through __getattr__ below so that `from osprey import __version__` and
# `patch("osprey.__version__", ...)` keep behaving like the plain attribute it was.
__version__ = get_running_version()

__all__ = ["__version__", "configure_logging"]

# Framework is designed for on-demand imports to avoid circular dependencies


def __getattr__(name: str) -> Any:
    """Resolve package-root exports lazily, keeping ``import osprey`` cheap."""
    if name == "configure_logging":
        from osprey.utils.logger import configure_logging

        return configure_logging
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily resolved exports, which ``globals()`` alone misses."""
    return sorted({*globals(), *__all__})
