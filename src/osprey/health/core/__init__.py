"""Built-in ("core") health-check categories.

This package holds the built-in check categories. Each canonical
category is exposed as a *factory* with the same shape a plugin uses, so the
runner executes core, YAML and plugin categories identically.

Factory contract
----------------
Every core category module exposes a factory named after its canonical
category. A factory has the signature::

    factory(config, context=None) -> CategoryCallable

where

* ``config`` is the pre-loaded config state produced by the CLI's single
  ``config.yml`` load. For most categories this is the parsed config mapping
  (``dict[str, Any] | None``, ``None`` when config is unavailable); the
  ``configuration`` category instead receives a :class:`ConfigState` describing
  the load outcome, since it is the category that *reports* on config loading.
* ``context`` is the opaque health runtime (see ``osprey.health.runtime``),
  providing lazy access to a control-system connector via
  ``await context.get_connector()``. Categories that need no connector ignore
  it. It defaults to ``None`` so config-only categories can be called without
  a runtime.

The factory returns a ``CategoryCallable``: a **no-argument** callable that
returns ``list[CheckResult]`` (sync) or an awaitable of it (async). Anything
the callable needs (config, runtime, resolved timeouts) is captured by closure
at registration time — this keeps the runner's execution path uniform across
core, YAML and plugin categories.

Lazy resolution
----------------
``CORE_CATEGORIES`` is a lazy mapping: iterating it yields the eleven canonical
names without importing anything, and indexing a name imports only that one
sibling module on demand. Sibling category modules are authored independently
and must never edit this file; a not-yet-written or import-failing module
therefore affects only the single name that resolves it, and importing this
package never fails.
"""

from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from osprey.health.models import CheckResult

# A category callable takes no arguments and returns results, sync or async.
CategoryCallable = Callable[[], "list[CheckResult] | Awaitable[list[CheckResult]]"]
# A factory binds config/runtime and returns the category callable.
CategoryFactory = Callable[..., CategoryCallable]

# Canonical category name -> (sibling module name, factory attribute) within
# this package. Static so the eleven valid names are known without importing any
# sibling module; resolution imports the module lazily on first access.
_CORE_CATEGORY_SPECS: dict[str, tuple[str, str]] = {
    "configuration": ("configuration", "configuration"),
    "file_system": ("file_system", "file_system"),
    "python_environment": ("python_environment", "python_environment"),
    "containers": ("containers", "containers"),
    "openobserve": ("openobserve", "openobserve"),
    "providers": ("providers", "providers"),
    "claude_cli": ("claude_cli", "claude_cli"),
    "claude_cli_pinned": ("claude_cli", "claude_cli_pinned"),
    "model_chat": ("model_chat", "model_chat"),
    "ariel": ("ariel", "ariel"),
    "channel_finder": ("channel_finder", "channel_finder"),
}


class _LazyCoreCategoryRegistry(Mapping[str, CategoryFactory]):
    """Read-only mapping of canonical category name to factory, resolved lazily.

    ``__getitem__`` imports the owning sibling module via importlib on demand
    and returns its factory attribute, so package import stays cheap and a
    missing sibling module only breaks the name that resolves to it.
    """

    def __getitem__(self, name: str) -> CategoryFactory:
        try:
            module_name, attr = _CORE_CATEGORY_SPECS[name]
        except KeyError:
            raise KeyError(f"Unknown core health category: {name!r}") from None
        module = importlib.import_module(f"{__name__}.{module_name}")
        return cast(CategoryFactory, getattr(module, attr))

    def __iter__(self) -> Iterator[str]:
        return iter(_CORE_CATEGORY_SPECS)

    def __len__(self) -> int:
        return len(_CORE_CATEGORY_SPECS)

    def __contains__(self, name: object) -> bool:
        return name in _CORE_CATEGORY_SPECS


CORE_CATEGORIES: Mapping[str, CategoryFactory] = _LazyCoreCategoryRegistry()

# The eleven canonical core category names, without importing any sibling module.
CORE_CATEGORY_NAMES: tuple[str, ...] = tuple(_CORE_CATEGORY_SPECS)


def get_core_category_factory(name: str) -> CategoryFactory:
    """Return the factory for a canonical core category, importing it lazily.

    Args:
        name: Canonical core category name (one of :data:`CORE_CATEGORY_NAMES`).

    Returns:
        The category's factory callable.

    Raises:
        KeyError: If ``name`` is not a known core category.
    """
    return CORE_CATEGORIES[name]


__all__ = [
    "CORE_CATEGORIES",
    "CORE_CATEGORY_NAMES",
    "CategoryCallable",
    "CategoryFactory",
    "get_core_category_factory",
]
