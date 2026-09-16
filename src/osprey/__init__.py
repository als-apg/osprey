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
# than through __getattr__ below so that `__version__` is a plain module attribute:
# `from osprey import __version__` and `patch("osprey.__version__", ...)` behave as
# they do for any other global.
__version__ = get_running_version()

__all__ = ["__version__", "configure_logging"]

# Framework is designed for on-demand imports to avoid circular dependencies

#: Public name -> the submodule of this package that defines it. Entries are
#: resolved on first attribute access, never at import.
_LAZY_EXPORTS: dict[str, str] = {"configure_logging": ".utils.logger"}


def __getattr__(name: str) -> Any:
    """Resolve a public name from its defining module on first access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
