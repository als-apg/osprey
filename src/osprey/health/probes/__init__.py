"""Declarative health probes and their lazy registry.

A *probe* is the async worker behind one declarative check type (``http``,
``mcp``, ``container``, ``channel_read``, ``provider_canary``). Every probe
conforms to a single interface::

    async def run(spec: Mapping[str, Any], ctx: ProbeContext) -> CheckResult

where ``spec`` is the parsed check parameters (a mapping) and ``ctx`` carries
shared per-run handles — chiefly the :class:`~osprey.health.runtime.HealthRuntime`
that owns the control-system connector.

Probes live in sibling modules that are resolved **lazily** by
:func:`get_probe`: importing this package never imports any probe module, and a
probe module is imported only when its type is first requested. This keeps the
registry authoritative in one place (the :data:`_PROBE_MODULES` table) while
letting each probe module be authored independently — no probe author edits this
file, and a not-yet-written sibling never breaks ``import``.
"""

from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from osprey.health.models import CheckResult

if TYPE_CHECKING:
    from osprey.health.runtime import HealthRuntime


@dataclass
class ProbeContext:
    """Shared per-run handles passed to every probe.

    Attributes:
        runtime: The suite's :class:`~osprey.health.runtime.HealthRuntime`,
            the single lazy owner of the control-system connector. Probes that
            need a connection acquire it via ``await ctx.runtime.get_connector()``;
            probes that do not (e.g. ``http``) simply ignore it.
    """

    runtime: HealthRuntime


#: An async probe worker: ``(spec, ctx) -> CheckResult``.
ProbeCallable = Callable[[Mapping[str, Any], ProbeContext], Awaitable[CheckResult]]


#: Static registry mapping each v1 probe type to ``(module, attribute)``. The
#: module name is relative to this package; the attribute is the probe callable.
#: Sibling modules are imported lazily via :func:`get_probe`, so entries may name
#: modules that are still being written without breaking package import.
_PROBE_MODULES: dict[str, tuple[str, str]] = {
    "http": ("http", "run"),
    "mcp": ("mcp", "run"),
    "container": ("container", "run"),
    "channel_read": ("channel_read", "run"),
    "provider_canary": ("provider_canary", "run"),
}

#: The set of known probe type names, for config validation without importing
#: any probe module.
PROBE_TYPES: frozenset[str] = frozenset(_PROBE_MODULES)


def get_probe(type_name: str) -> ProbeCallable:
    """Resolve a probe type name to its async callable, importing it on demand.

    Args:
        type_name: A v1 probe type, e.g. ``"http"``. Must be a key of
            :data:`PROBE_TYPES`.

    Returns:
        The probe's ``run`` callable conforming to :data:`ProbeCallable`.

    Raises:
        KeyError: If ``type_name`` is not a known probe type.
    """
    try:
        module_name, attr = _PROBE_MODULES[type_name]
    except KeyError:
        raise KeyError(
            f"unknown probe type {type_name!r}; known types: {sorted(PROBE_TYPES)}"
        ) from None
    module = importlib.import_module(f".{module_name}", __package__)
    return getattr(module, attr)  # type: ignore[no-any-return]
