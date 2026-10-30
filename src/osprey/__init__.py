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

if TYPE_CHECKING:
    from osprey_connectors.logger import configure_logging

    __version__: str

__all__ = ["__version__", "configure_logging"]

# Framework is designed for on-demand imports to avoid circular dependencies

#: Public name -> the module that defines it. Entries are resolved on first
#: attribute access, never at import.
_LAZY_EXPORTS: dict[str, str] = {"configure_logging": "osprey_connectors.logger"}


def __getattr__(name: str) -> Any:
    """Resolve a public name from its defining module on first access."""
    if name == "__version__":
        # Derived from the git tag by osprey.version — see that module for the
        # resolution chain. Resolved on first read, not at import, because in a
        # source checkout that chain runs `git describe`, a subprocess every
        # short-lived process importing osprey (each Claude Code hook) would
        # otherwise pay without reading the version. Bound into the module
        # globals once read, so `from osprey import __version__` and
        # `patch("osprey.__version__", ...)` behave as for any other global.
        from osprey.version import get_running_version

        globals()[name] = version = get_running_version()
        return version
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS, "__version__"})
