"""Health web view — the browser-facing surface of the health framework.

Exposes the same configurable health suite the ``osprey health`` CLI renders,
served from a long-lived process. The surface-agnostic refresh building blocks
(:class:`~osprey.health.loader.HealthConfigLoader`,
:class:`~osprey.health.lifecycle.HealthRuntimeLifecycle`) live in
:mod:`osprey.health`; this package holds the web-only pieces — the FastAPI app
factory, the caching refresh engine, and the dashboard bundle.
"""

from __future__ import annotations
