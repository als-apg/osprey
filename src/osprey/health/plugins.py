"""Load facility health categories from ``health.plugins`` modules.

A plugin is a dotted Python module path listed under ``health.plugins`` that
exposes ``get_health_categories() -> dict[str, CategoryCallable]`` — a mapping of
category name to a no-argument callable returning ``list[CheckResult]`` (sync or
async), the same callable shape core and YAML categories use. Plugin categories
run alongside core and YAML categories through the identical runner path.

Loading is defensive: no plugin failure ever crashes the suite. An import error,
a missing/non-callable ``get_health_categories``, a bad return type, or a name
that collides with a core/YAML/earlier-plugin category each yields a single
``error`` :class:`~osprey.health.models.CheckResult` in the diagnostic
``plugins`` category — never an exception. Successfully loaded plugin categories
default to ``cost: poll`` with the suite timeout as their budget; a metadata-only
``health.categories.<name>`` override adjusts cost/timeout by name (mirroring the
core-category override channel).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from osprey.health.config import (
    CORE_CATEGORY_NAMES,
    CategoryRecord,
    Cost,
    resolve_callable_timeout_s,
)
from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from osprey.health.config import HealthSettings

#: Diagnostic category under which plugin-loading failures are reported.
PLUGINS_DIAGNOSTIC_CATEGORY = "plugins"

_ENTRYPOINT = "get_health_categories"


@dataclass
class PluginLoadResult:
    """Outcome of loading all configured health plugins.

    Attributes:
        categories: Successfully loaded plugin categories, keyed by name.
        errors: One ``error`` result row per plugin/category that failed to load
            (import failure, bad entrypoint, bad return, or name collision).
    """

    categories: dict[str, CategoryRecord] = field(default_factory=dict)
    errors: list[CheckResult] = field(default_factory=list)


def load_plugin_categories(settings: HealthSettings) -> PluginLoadResult:
    """Load every ``health.plugins`` module into runtime category records.

    Categories are accepted in plugin-list order; a name already claimed by a
    core category, a YAML category, or an earlier plugin collides and is rejected
    with an ``error`` row rather than overwriting the incumbent.

    Args:
        settings: Parsed health settings carrying ``plugins``, YAML
            ``categories`` (for collision detection), ``overrides`` (metadata by
            name), and ``suite_timeout_s`` (the default poll budget).

    Returns:
        A :class:`PluginLoadResult` with the loaded categories and any error rows.
    """
    result = PluginLoadResult()
    taken: set[str] = set(CORE_CATEGORY_NAMES) | set(settings.categories)

    for path in settings.plugins:
        raw = _load_entrypoint(path, result.errors)
        if raw is None:
            continue
        for cat_name, cat_callable in raw.items():
            if not isinstance(cat_name, str) or not callable(cat_callable):
                result.errors.append(
                    _error_row(
                        path,
                        f"health plugin '{path}' returned an invalid category entry "
                        f"{cat_name!r}: expected str name -> callable",
                    )
                )
                continue
            if cat_name in taken:
                result.errors.append(
                    _error_row(
                        f"{path}:{cat_name}",
                        f"health plugin '{path}' category {cat_name!r} collides with an "
                        f"existing core, YAML, or plugin category",
                    )
                )
                continue
            result.categories[cat_name] = _build_record(cat_name, cat_callable, settings)
            taken.add(cat_name)

    return result


def _load_entrypoint(path: str, errors: list[CheckResult]) -> Any:
    """Import ``path`` and return its ``get_health_categories()`` mapping.

    Any failure appends one ``error`` row to ``errors`` and returns ``None``.
    """
    try:
        module = importlib.import_module(path)
    except Exception as exc:  # noqa: BLE001 - any import failure is a reported error row
        errors.append(_error_row(path, f"failed to import health plugin '{path}': {exc}"))
        return None

    entrypoint = getattr(module, _ENTRYPOINT, None)
    if not callable(entrypoint):
        errors.append(_error_row(path, f"health plugin '{path}' does not define {_ENTRYPOINT}()"))
        return None

    try:
        raw = entrypoint()
    except Exception as exc:  # noqa: BLE001 - a raising entrypoint is a reported error row
        errors.append(_error_row(path, f"health plugin '{path}' {_ENTRYPOINT}() raised: {exc}"))
        return None

    if not isinstance(raw, dict):
        errors.append(
            _error_row(
                path,
                f"health plugin '{path}' {_ENTRYPOINT}() must return a dict of "
                f"name -> callable, got {type(raw).__name__}",
            )
        )
        return None

    return raw


def _build_record(name: str, func: Any, settings: HealthSettings) -> CategoryRecord:
    """Build a poll-default category record, applying any metadata override."""
    override = settings.overrides.get(name)
    cost = override.cost if override and override.cost is not None else Cost.POLL
    override_timeout = override.timeout_s if override else None
    timeout_s = resolve_callable_timeout_s(cost, override_timeout, settings.suite_timeout_s)
    return CategoryRecord(name=name, cost=cost, timeout_s=timeout_s, func=func)


def _error_row(name: str, message: str) -> CheckResult:
    """A single plugin-loading ``error`` row in the diagnostic category."""
    return CheckResult(name, PLUGINS_DIAGNOSTIC_CATEGORY, Status.ERROR, message)
