"""Load facility health categories from ``health.plugins`` entries.

A plugin exposes ``get_health_categories() -> dict[str, CategoryCallable]`` — a
mapping of category name to a callable returning ``list[CheckResult]`` (sync or
async), the same callable shape core and YAML categories use. The callable
normally takes no arguments; an ``async def`` one may declare a ``runtime``
parameter and be handed the suite's shared
:class:`~osprey.health.runtime.HealthRuntime` by keyword, while a sync one that
declares it is refused with an ``error`` row. The full contract is on the
``contributing/extending-osprey`` page. Plugin categories run alongside core and
YAML categories through the identical runner path.

An entry under ``health.plugins`` names that plugin in one of two ways:

* a **dotted module path** (``my_package.health_checks``), imported from the
  process's ``sys.path`` — the plugin has to be installed or otherwise
  importable; or
* a **file path** ending in ``.py`` (``./health/facility_checks.py``), loaded
  straight off disk. A relative path is resolved against the **project root**,
  the same anchor ``data/`` and ``plans/`` paths use, so a deployment can keep
  its checks next to its profile without packaging them or arranging
  ``PYTHONPATH`` in every surface that runs the suite.

A file-path plugin is loaded under a deterministic synthetic module name derived
from its resolved path (``osprey_health_plugin_<hash>``) and registered in
``sys.modules`` before it executes, so dataclasses and postponed annotations
inside it resolve normally. Loading the same file again — which the long-lived
surfaces do on every config-change refresh — **re-executes** the file and
replaces that ``sys.modules`` entry. Module-level state in a plugin therefore
does not survive a refresh, and a plugin must not rely on it doing so.

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

import hashlib
import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
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

#: Name prefix of the synthetic module a file-path plugin is loaded under. The
#: suffix is a digest of the resolved path, so the name is stable across refresh
#: cycles and two different files never collide on it.
PLUGIN_MODULE_PREFIX = "osprey_health_plugin_"

_ENTRYPOINT = "get_health_categories"

#: The one thing that tells a file path from a dotted module path. Case-sensitive:
#: a module cannot be named ``.PY`` on an import path anyway, and matching loosely
#: would make the meaning of an entry depend on the host filesystem.
_PATH_SUFFIX = ".py"


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


def load_plugin_categories(
    settings: HealthSettings, *, project_root: Path | None = None
) -> PluginLoadResult:
    """Load every ``health.plugins`` entry into runtime category records.

    Categories are accepted in plugin-list order; a name already claimed by a
    core category, a YAML category, or an earlier plugin collides and is rejected
    with an ``error`` row rather than overwriting the incumbent.

    Args:
        settings: Parsed health settings carrying ``plugins``, YAML
            ``categories`` (for collision detection), ``overrides`` (metadata by
            name), and ``suite_timeout_s`` (the default poll budget).
        project_root: Anchor for relative ``.py`` entries — the deployment repo
            root, which ``build_records`` already holds and passes. Omitted only
            by callers with no config in hand (the dotted-path form never
            consults it); the anchor is then derived the same way every other
            config-relative path derives it, through
            :func:`~osprey.utils.config_paths.resolve_config_relative_path`.

    Returns:
        A :class:`PluginLoadResult` with the loaded categories and any error rows.
    """
    result = PluginLoadResult()
    taken: set[str] = set(CORE_CATEGORY_NAMES) | set(settings.categories)

    for path in settings.plugins:
        raw = _load_entrypoint(path, result.errors, project_root=project_root)
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


def _load_entrypoint(
    path: str, errors: list[CheckResult], *, project_root: Path | None = None
) -> Any:
    """Import ``path`` and return its ``get_health_categories()`` mapping.

    Any failure appends one ``error`` row to ``errors`` and returns ``None``.
    """
    module = _import_plugin(path, errors, project_root=project_root)
    if module is None:
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


def _import_plugin(
    entry: str, errors: list[CheckResult], *, project_root: Path | None
) -> ModuleType | None:
    """Import one ``health.plugins`` entry, by file path or by dotted name.

    Any failure appends one ``error`` row to ``errors`` and returns ``None`` —
    a bad plugin degrades the suite by one diagnostic row, it never raises.
    """
    if entry.endswith(_PATH_SUFFIX):
        return _import_plugin_file(entry, errors, project_root=project_root)

    try:
        return importlib.import_module(entry)
    except Exception as exc:  # noqa: BLE001 - any import failure is a reported error row
        errors.append(_error_row(entry, f"failed to import health plugin '{entry}': {exc}"))
        return None


def _import_plugin_file(
    entry: str, errors: list[CheckResult], *, project_root: Path | None
) -> ModuleType | None:
    """Load a ``.py`` plugin entry off disk under its synthetic module name."""
    resolved = _resolve_plugin_file(entry, project_root)
    if not resolved.is_file():
        errors.append(_error_row(entry, f"health plugin file '{entry}' not found: {resolved}"))
        return None

    # A digest only to keep the name short and filesystem-agnostic — nothing here
    # is a security boundary, hence usedforsecurity=False (a FIPS-strict host
    # otherwise refuses sha1 outright).
    digest = hashlib.sha1(str(resolved).encode(), usedforsecurity=False).hexdigest()[:12]
    module_name = f"{PLUGIN_MODULE_PREFIX}{digest}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, resolved)
        if spec is None or spec.loader is None:
            raise ImportError(f"no Python import machinery handles {resolved}")
        module = importlib.util.module_from_spec(spec)
        # Registered BEFORE exec_module: dataclasses and postponed annotations
        # resolve a class's own module through sys.modules while the file is
        # still executing. A re-load replaces the entry, which is why a plugin
        # cannot carry module-level state across a refresh cycle.
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - any load failure is a reported error row
        # A half-executed module must not be left behind for the next cycle to
        # find and treat as loaded.
        sys.modules.pop(module_name, None)
        errors.append(
            _error_row(entry, f"failed to import health plugin '{entry}' ({resolved}): {exc}")
        )
        return None

    return module


def _resolve_plugin_file(entry: str, project_root: Path | None) -> Path:
    """Absolute path of a ``.py`` plugin entry.

    ``~`` is expanded and an absolute entry is used as-is; a relative one is
    joined onto *project_root*. With no anchor in hand the shared config-path
    rule answers instead — the same rule, reached through its own fallback,
    rather than a second one written here.
    """
    path = Path(entry).expanduser()
    if path.is_absolute():
        return path
    if project_root is None:
        from osprey.utils.config_paths import resolve_config_relative_path

        return resolve_config_relative_path(path)
    return (Path(project_root) / path).resolve()


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
