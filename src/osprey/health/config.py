"""Parsing and validation for the ``health:`` configuration section.

Compiles the declarative ``health.categories.<name>`` YAML into the runtime
data structures the runner executes, and centralizes the framework's health
policy: cost classes, per-check and per-category timeout defaults, metadata
overrides for built-in and plugin categories, collision rules, ``requires:``
validation, and the scalar suite settings (``suite_timeout_s``,
``on_demand_timeout_s``, ``interval_s``).

The configuration is read through the standard loader (``ConfigBuilder.get``),
which has already resolved ``${VAR}`` placeholders — this module performs no
second interpolation pass. Library validation failures raise
:class:`~osprey.errors.ConfigurationError`; the CLI converts those into report
rows at its single load-failure boundary.

Three category sources are executed identically by the runner:

* **Core** — built-in callables registered in code (not parsed here). Their
  names are :data:`CORE_CATEGORY_NAMES`; a ``health.categories.<name>`` block
  for a core name may carry metadata-only overrides (``cost``/``timeout_s``)
  but must not define ``checks:``.
* **YAML** — ``health.categories.<name>.checks[]`` declarative probe checks,
  compiled here into :class:`CategoryRecord`\\ s of :class:`CheckSpec`\\ s.
* **Plugins** — dotted module paths in ``health.plugins`` exposing callables
  (loaded elsewhere); metadata overrides apply to plugin names too.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from osprey.errors import ConfigurationError

from .core import CORE_CATEGORY_NAMES as _CORE_CATEGORY_NAMES_ORDERED
from .models import Status

# --- Policy constants -------------------------------------------------------

#: Default display title for the health report when ``health.title`` is unset.
DEFAULT_HEALTH_TITLE: str = "System Health"

#: Default suite (poll-class) deadline in seconds when not configured.
DEFAULT_SUITE_TIMEOUT_S: float = 30.0

#: Default per-category budget for an on_demand callable category (per unique
#: item, for item-looping callables such as ``model_chat``).
DEFAULT_ON_DEMAND_CALLABLE_TIMEOUT_S: float = 60.0

#: Default per-check ``timeout_s`` by probe ``type``. Also the authoritative v1
#: probe vocabulary — an unknown ``type`` is a load-time error.
DEFAULT_PROBE_TIMEOUTS: dict[str, float] = {
    "http": 5.0,
    "mcp": 10.0,
    "container": 10.0,
    "channel_read": 5.0,
    "provider_canary": 5.0,
    "archiver_freshness": 10.0,
}

#: Canonical names of the built-in (core) categories, single-sourced from the
#: lazy core registry (:mod:`osprey.health.core`, which never imports sibling
#: category modules). A YAML ``checks:`` list under any of these names is
#: rejected ("cannot redefine built-in category"); a metadata-only block for
#: one of these names is a valid override.
CORE_CATEGORY_NAMES: frozenset[str] = frozenset(_CORE_CATEGORY_NAMES_ORDERED)

# Reserved keys within a single check mapping; everything else becomes params.
_RESERVED_CHECK_KEYS = frozenset({"name", "type", "timeout_s", "timeout_status", "requires"})

#: Valid values for ``health.auto.mcp.url_key`` — which server connection URL the
#: auto-derived ``mcp_servers`` probes target.
_AUTO_MCP_URL_KEYS: frozenset[str] = frozenset({"host_url", "docker_url"})


class Cost(StrEnum):
    """Cost class of a category, controlling which deadline budget bounds it."""

    POLL = "poll"
    ON_DEMAND = "on_demand"


# --- Runtime data structures ------------------------------------------------


@dataclass(frozen=True)
class CheckSpec:
    """A single declarative probe check compiled from YAML.

    Attributes:
        name: Unique (within its category) machine-readable check name.
        type: Probe type — one of :data:`DEFAULT_PROBE_TIMEOUTS`.
        params: Probe-specific parameters (every key that is not reserved).
        timeout_s: Per-check timeout in seconds.
        timeout_status: Status emitted when the per-check timeout fires.
        requires: Names of earlier checks in the same category this check
            depends on; it is skipped if any dependency does not pass.
    """

    name: str
    type: str
    params: Mapping[str, Any]
    timeout_s: float
    timeout_status: Status
    requires: tuple[str, ...]


@dataclass
class CategoryRecord:
    """A runtime category executed by the runner.

    Exactly one of ``checks`` (declarative YAML/plugin-declared categories) or
    ``func`` (core/plugin callable categories) is populated.

    Attributes:
        name: Category name (also a valid ``--category`` value).
        cost: Cost class (``poll`` or ``on_demand``).
        timeout_s: Category-level budget in seconds.
        checks: Declarative checks, or ``None`` for callable categories.
        func: The category callable returning ``list[CheckResult]`` (sync or
            async), or ``None`` for declarative categories.
    """

    name: str
    cost: Cost
    timeout_s: float
    checks: list[CheckSpec] | None = None
    func: Any = None


@dataclass
class CategoryOverride:
    """Metadata-only overrides for a core or plugin category name."""

    cost: Cost | None = None
    timeout_s: float | None = None


@dataclass(frozen=True)
class AutoMcpSettings:
    """Settings for the auto-derived ``mcp_servers`` health category.

    Parsed from ``health.auto.mcp``. Controls whether the framework synthesizes
    an ``mcp_servers`` health category from the configured MCP servers, and
    which connection URL the generated probes target.

    Attributes:
        enabled: Whether the auto-derived ``mcp_servers`` category is emitted.
        url_key: Which server URL the derived probes use — ``"host_url"`` or
            ``"docker_url"``.
        url_key_explicit: ``True`` only when ``url_key`` was set in the config;
            ``False`` when it holds its default. Consumers (``derive.py``) rely
            on this to distinguish an explicit choice from the default,
            resolving the probe URL as ``url_key`` when explicit, else
            ``docker_url`` when containerized, else ``host_url``.
    """

    enabled: bool = True
    url_key: str = "host_url"
    url_key_explicit: bool = False


@dataclass
class HealthSettings:
    """Parsed ``health:`` configuration.

    Attributes:
        suite_timeout_s: Poll-class suite deadline.
        interval_s: Minimum server-side re-run interval (validated, not
            enforced in P1).
        on_demand_timeout_s: Explicit on_demand budget, or ``None`` to derive it
            at merge time as the sum of resolved on_demand category budgets.
        title: Human-readable display title for the health report.
        categories: Declarative (YAML) categories keyed by name.
        overrides: Metadata-only overrides for core/plugin category names.
        plugins: Dotted plugin module paths exposing category callables.
        auto: Settings for the auto-derived ``mcp_servers`` category
            (``health.auto.mcp``); the default instance when the section is
            absent.
    """

    suite_timeout_s: float
    interval_s: float
    on_demand_timeout_s: float | None
    title: str = DEFAULT_HEALTH_TITLE
    categories: dict[str, CategoryRecord] = field(default_factory=dict)
    overrides: dict[str, CategoryOverride] = field(default_factory=dict)
    plugins: list[str] = field(default_factory=list)
    auto: AutoMcpSettings = field(default_factory=AutoMcpSettings)


# --- Timeout resolution helpers ---------------------------------------------


def resolve_callable_timeout_s(
    cost: Cost,
    override_timeout_s: float | None,
    suite_timeout_s: float,
) -> float:
    """Resolve the category budget for a non-item-looping callable category.

    Poll callables default to the suite timeout; on_demand callables default to
    :data:`DEFAULT_ON_DEMAND_CALLABLE_TIMEOUT_S`. An explicit override wins.
    """
    if override_timeout_s is not None:
        return override_timeout_s
    return suite_timeout_s if cost is Cost.POLL else DEFAULT_ON_DEMAND_CALLABLE_TIMEOUT_S


def resolve_item_looping_on_demand_timeout(
    n_items: int,
    override_timeout_s: float | None,
) -> tuple[float, float]:
    """Resolve (category_budget, per_item_budget) for an on_demand item-looping
    callable such as ``model_chat``.

    Without an override the category budget is the per-item default times
    ``max(n_items, 1)``; with an override the override becomes the category
    budget and the per-item budget becomes ``override / max(n_items, 1)``. Using
    ``max(n_items, 1)`` keeps a zero-item category (which short-circuits to a
    single ``skip`` row before the deadline machinery) from dividing by zero.
    """
    n = max(n_items, 1)
    if override_timeout_s is not None:
        return override_timeout_s, override_timeout_s / n
    default = DEFAULT_ON_DEMAND_CALLABLE_TIMEOUT_S
    return default * n, default


# --- Parsing ----------------------------------------------------------------


def _as_float(value: Any, path: str) -> float:
    """Coerce a config scalar to float, raising ConfigurationError otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"health.{path} must be a number, got {value!r}")
    return float(value)


def _positive_float(value: Any, path: str) -> float:
    result = _as_float(value, path)
    if result <= 0:
        raise ConfigurationError(f"health.{path} must be greater than 0, got {result}")
    return result


def _parse_cost(value: Any, path: str) -> Cost:
    try:
        return Cost(value)
    except ValueError:
        raise ConfigurationError(
            f"health.{path} must be 'poll' or 'on_demand', got {value!r}"
        ) from None


def _parse_timeout_status(value: Any, path: str) -> Status:
    if value not in (Status.ERROR.value, Status.WARNING.value):
        raise ConfigurationError(f"health.{path} must be 'error' or 'warning', got {value!r}")
    return Status(value)


def _parse_requires(value: Any, path: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigurationError(f"health.{path} must be a list of check names")
    return tuple(value)


def _parse_check(raw: Any, category: str, index: int) -> CheckSpec:
    where = f"categories.{category}.checks[{index}]"
    if not isinstance(raw, Mapping):
        raise ConfigurationError(f"health.{where} must be a mapping")

    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigurationError(f"health.{where}.name is required and must be a non-empty string")

    probe_type = raw.get("type")
    if not isinstance(probe_type, str) or not probe_type:
        raise ConfigurationError(f"health.{where}.type is required and must be a non-empty string")
    if probe_type not in DEFAULT_PROBE_TIMEOUTS:
        allowed = ", ".join(sorted(DEFAULT_PROBE_TIMEOUTS))
        raise ConfigurationError(
            f"health.{where}.type '{probe_type}' is not a known probe type ({allowed})"
        )

    if "timeout_s" in raw:
        timeout_s = _positive_float(raw["timeout_s"], f"{where}.timeout_s")
    else:
        timeout_s = DEFAULT_PROBE_TIMEOUTS[probe_type]

    timeout_status = (
        _parse_timeout_status(raw["timeout_status"], f"{where}.timeout_status")
        if "timeout_status" in raw
        else Status.ERROR
    )

    requires = _parse_requires(raw.get("requires"), f"{where}.requires")
    params = {k: v for k, v in raw.items() if k not in _RESERVED_CHECK_KEYS}

    return CheckSpec(
        name=name,
        type=probe_type,
        params=params,
        timeout_s=timeout_s,
        timeout_status=timeout_status,
        requires=requires,
    )


def _validate_requires(checks: list[CheckSpec], category: str) -> None:
    """Reject duplicate check names, unknown ``requires`` targets, self-refs and
    forward references (a dependency must appear earlier in the list)."""
    seen: set[str] = set()
    for check in checks:
        if check.name in seen:
            raise ConfigurationError(
                f"health.categories.{category}: duplicate check name '{check.name}'"
            )
        for dep in check.requires:
            if dep == check.name:
                raise ConfigurationError(
                    f"health.categories.{category}.{check.name}: check cannot require itself"
                )
            if dep not in seen:
                raise ConfigurationError(
                    f"health.categories.{category}.{check.name}: requires '{dep}', which is "
                    f"not an earlier check in this category (unknown or forward reference)"
                )
        seen.add(check.name)


def _parse_category(name: str, raw: Any) -> tuple[CategoryRecord | None, CategoryOverride | None]:
    """Parse one ``health.categories.<name>`` block.

    Returns a ``(record, override)`` pair with exactly one element populated: a
    declarative :class:`CategoryRecord` when ``checks:`` is present, otherwise a
    metadata-only :class:`CategoryOverride`.
    """
    if not isinstance(raw, Mapping):
        raise ConfigurationError(f"health.categories.{name} must be a mapping")

    override_timeout_s = (
        _positive_float(raw["timeout_s"], f"categories.{name}.timeout_s")
        if "timeout_s" in raw
        else None
    )
    override_cost = _parse_cost(raw["cost"], f"categories.{name}.cost") if "cost" in raw else None

    if "checks" not in raw:
        return None, CategoryOverride(cost=override_cost, timeout_s=override_timeout_s)

    if name in CORE_CATEGORY_NAMES:
        raise ConfigurationError(
            f"health.categories.{name}: cannot redefine built-in category with a 'checks:' list; "
            f"use metadata-only keys (cost/timeout_s) to override it instead"
        )

    checks_raw = raw["checks"]
    if not isinstance(checks_raw, list):
        raise ConfigurationError(f"health.categories.{name}.checks must be a list")

    checks = [_parse_check(item, name, i) for i, item in enumerate(checks_raw)]
    _validate_requires(checks, name)

    cost = override_cost if override_cost is not None else Cost.POLL
    timeout_s = (
        override_timeout_s
        if override_timeout_s is not None
        else (
            DEFAULT_SUITE_TIMEOUT_S if cost is Cost.POLL else DEFAULT_ON_DEMAND_CALLABLE_TIMEOUT_S
        )
    )
    return (
        CategoryRecord(name=name, cost=cost, timeout_s=timeout_s, checks=checks),
        None,
    )


def parse_health_config(health: Mapping[str, Any] | None) -> HealthSettings:
    """Parse the ``health:`` section into :class:`HealthSettings`.

    Args:
        health: The ``health`` mapping from the loaded config (already
            ``${VAR}``-resolved), or ``None``/empty when unconfigured.

    Returns:
        The parsed settings with declarative categories, metadata overrides,
        plugins, and validated scalar suite settings.

    Raises:
        ConfigurationError: On any invalid value — bad types, unknown probe
            type, ``checks:`` under a core name, duplicate/unknown/forward
            ``requires`` targets, or ``interval_s <= suite_timeout_s``.
    """
    if health is None:
        health = {}
    if not isinstance(health, Mapping):
        raise ConfigurationError("health: section must be a mapping")

    suite_timeout_s = (
        _positive_float(health["suite_timeout_s"], "suite_timeout_s")
        if "suite_timeout_s" in health
        else DEFAULT_SUITE_TIMEOUT_S
    )

    on_demand_timeout_s = (
        _positive_float(health["on_demand_timeout_s"], "on_demand_timeout_s")
        if "on_demand_timeout_s" in health
        else None
    )

    if "interval_s" in health:
        interval_s = _positive_float(health["interval_s"], "interval_s")
        if interval_s <= suite_timeout_s:
            raise ConfigurationError(
                f"health.interval_s ({interval_s}) must be greater than "
                f"suite_timeout_s ({suite_timeout_s})"
            )
    else:
        interval_s = max(60.0, 2 * suite_timeout_s)

    title = _parse_title(health.get("title"))

    plugins = _parse_plugins(health.get("plugins"))

    auto = _parse_auto(health.get("auto"))

    categories: dict[str, CategoryRecord] = {}
    overrides: dict[str, CategoryOverride] = {}
    categories_raw = health.get("categories") or {}
    if not isinstance(categories_raw, Mapping):
        raise ConfigurationError("health.categories must be a mapping")
    for name, raw in categories_raw.items():
        record, override = _parse_category(str(name), raw)
        if record is not None:
            categories[str(name)] = record
        if override is not None:
            overrides[str(name)] = override

    return HealthSettings(
        suite_timeout_s=suite_timeout_s,
        interval_s=interval_s,
        on_demand_timeout_s=on_demand_timeout_s,
        title=title,
        categories=categories,
        overrides=overrides,
        plugins=plugins,
        auto=auto,
    )


def _parse_title(value: Any) -> str:
    if value is None:
        return DEFAULT_HEALTH_TITLE
    if not isinstance(value, str):
        raise ConfigurationError(f"health.title must be a string, got {value!r}")
    return value


def _parse_plugins(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigurationError("health.plugins must be a list of dotted module paths")
    return list(value)


def _parse_auto(value: Any) -> AutoMcpSettings:
    """Parse the ``health.auto`` section into :class:`AutoMcpSettings`.

    Only ``health.auto.mcp.{enabled, url_key}`` are read; any other keys under
    ``auto`` or ``auto.mcp`` are ignored, consistent with the additive posture
    of the rest of the parser. An absent section (or absent ``mcp`` subsection)
    yields the default instance with ``url_key_explicit`` false.
    """
    if value is None:
        return AutoMcpSettings()
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"health.auto must be a mapping, got {value!r}")

    mcp_raw = value.get("mcp")
    if mcp_raw is None:
        return AutoMcpSettings()
    if not isinstance(mcp_raw, Mapping):
        raise ConfigurationError(f"health.auto.mcp must be a mapping, got {mcp_raw!r}")

    if "enabled" in mcp_raw:
        enabled = mcp_raw["enabled"]
        if not isinstance(enabled, bool):
            raise ConfigurationError(f"health.auto.mcp.enabled must be a boolean, got {enabled!r}")
    else:
        enabled = True

    url_key_explicit = "url_key" in mcp_raw
    if url_key_explicit:
        url_key = mcp_raw["url_key"]
        if url_key not in _AUTO_MCP_URL_KEYS:
            allowed = ", ".join(sorted(_AUTO_MCP_URL_KEYS))
            raise ConfigurationError(
                f"health.auto.mcp.url_key {url_key!r} is not valid; must be one of ({allowed})"
            )
    else:
        url_key = "host_url"

    return AutoMcpSettings(enabled=enabled, url_key=url_key, url_key_explicit=url_key_explicit)
