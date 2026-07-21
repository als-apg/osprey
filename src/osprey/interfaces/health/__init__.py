"""Health web view — the browser-facing surface of the health framework.

Exposes the same configurable health suite the ``osprey health`` CLI renders,
served from a long-lived process. :class:`HealthConfigLoader` is the synchronous
config-load phase of the view's refresh cycle.
"""

from __future__ import annotations

from .loader import HealthConfigLoader, LoadedHealthConfig

__all__ = ["HealthConfigLoader", "LoadedHealthConfig"]
