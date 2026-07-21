"""Config loading and category-record assembly for a health run.

This module holds the surface-agnostic core the ``osprey health`` CLI used to
carry inline: the single ``config.yml`` load (:func:`load_config`) and the
assembly of the merged category records — built-in "core" categories,
declarative YAML categories, and facility plugins (:func:`build_records`, with
its :func:`core_record` / :func:`skip_record` helpers).

Extracting these off the CLI lets non-CLI surfaces (e.g. the web health view)
reuse the exact same record-assembly and config-degradation behavior instead of
reimplementing it. The CLI (:mod:`osprey.cli.health_cmd`) re-imports these names
and wires them together unchanged.

Design contracts honored here:

* **Single config load.** :func:`load_config` loads ``config.yml`` exactly once
  and reports the outcome through a
  :class:`~osprey.health.core.configuration.ConfigState`. A load failure never
  raises — it degrades into a ``config_ok=False`` result while the rest of the
  report still renders.
* **Config-dependent core categories degrade to skips.** Core categories that
  read the loaded config (:data:`CONFIG_DEPENDENT`) collapse to a single
  "config unavailable" skip row when the config could not be loaded/parsed.
* **``--full`` is the sole on_demand gate.** :data:`ON_DEMAND_CORE` names the
  core categories that default to the on_demand cost class; ``build_records``
  only sets the default cost, never the gating.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osprey.health.config import CategoryRecord, HealthSettings
    from osprey.health.core.configuration import ConfigState

#: Produces ``(expanded, unexpanded)`` from a freshly constructed config builder,
#: or raises (``FileNotFoundError`` for a racy disappearance; any other exception
#: for bad YAML / non-mapping / is-a-directory). Callers own builder construction
#: — the CLI its shared singleton, the web loader a private
#: ``ConfigBuilder(load_env=False)`` — while :func:`_load_config_result` owns the
#: single degradation-and-parse contract both must honor.
_ConfigLoader = Callable[[], "tuple[dict[str, Any] | None, dict[str, Any] | None]"]

# Core categories that read the loaded config and therefore degrade to a single
# "config unavailable" skip row when the config could not be loaded/parsed.
CONFIG_DEPENDENT = frozenset(
    {"openobserve", "providers", "claude_cli_pinned", "model_chat", "ariel", "channel_finder"}
)

# Core categories in the on_demand cost class (gated behind ``--full``).
ON_DEMAND_CORE = frozenset({"claude_cli_pinned", "model_chat"})


def load_config(
    config_path: Path, project_path: Path
) -> tuple[ConfigState, dict[str, Any] | None, HealthSettings | None, bool]:
    """Perform the single ``config.yml`` load and parse its ``health:`` section.

    Returns a four-tuple ``(config_state, expanded, settings, config_ok)`` where
    ``config_state`` describes the load outcome for the ``configuration``
    category, ``expanded`` is the ``${VAR}``-resolved config mapping (or ``None``
    when no usable mapping was produced), ``settings`` is the parsed health
    settings (or ``None`` on any failure), and ``config_ok`` is ``True`` only
    when a usable config mapping loaded *and* its ``health:`` section parsed.

    A missing file, empty/non-mapping file, bad YAML, or an invalid ``health:``
    section all yield ``config_ok=False`` — never an exception.
    """

    def _load() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        from osprey.utils.config import get_config_builder

        builder = get_config_builder(str(config_path), set_as_default=True)
        return builder.raw_config, builder.get_unexpanded_config()

    return _load_config_result(config_path, project_path, _load)


def _load_config_result(
    config_path: Path, project_path: Path, load_builder: _ConfigLoader
) -> tuple[ConfigState, dict[str, Any] | None, HealthSettings | None, bool]:
    """Apply the config-load degradation-and-parse contract around a builder load.

    The single source of the ``config.yml`` degradation contract shared by the
    CLI one-shot load (:func:`load_config`) and the long-lived web loader
    (:func:`osprey.interfaces.health.loader._load_config`): only builder
    construction differs between them, so each supplies its own ``load_builder``
    while this owns the exists-check, the raise-to-degrade branches, and the
    ``health:`` parse — none of which ever raise.
    """
    from osprey.errors import ConfigurationError
    from osprey.health.config import parse_health_config
    from osprey.health.core.configuration import ConfigState

    if not config_path.exists():
        return (
            ConfigState(config_path, exists=False, cwd=project_path, config=None),
            None,
            None,
            False,
        )

    try:
        expanded, unexpanded = load_builder()
    except FileNotFoundError:
        # Racy disappearance between the exists() check and the load.
        return (
            ConfigState(config_path, exists=False, cwd=project_path, config=None),
            None,
            None,
            False,
        )
    except Exception as exc:  # noqa: BLE001 - bad YAML / non-mapping / is-a-directory
        state = ConfigState(
            config_path, exists=True, cwd=project_path, config=None, yaml_error=str(exc)
        )
        return state, None, None, False

    if not expanded:
        # Empty file (the loader normalizes an empty document to ``{}``). Report
        # it as an empty config via the ``yaml_valid`` row and degrade.
        state = ConfigState(config_path, exists=True, cwd=project_path, config=None)
        return state, None, None, False

    try:
        settings = parse_health_config(expanded.get("health"))
    except ConfigurationError as exc:
        state = ConfigState(
            config_path,
            exists=True,
            cwd=project_path,
            config=unexpanded,
            health_error=str(exc),
        )
        return state, expanded, None, False

    state = ConfigState(config_path, exists=True, cwd=project_path, config=unexpanded)
    return state, expanded, settings, True


def core_record(
    name: str,
    func: Any,
    default_cost: Any,
    settings: HealthSettings | None,
    expanded: dict[str, Any] | None,
    suite_timeout_s: float,
) -> CategoryRecord:
    """Wrap a core category callable in a :class:`CategoryRecord`.

    Applies any metadata-only ``health.categories.<name>`` override (cost and/or
    timeout) and resolves the category budget from the framework's timeout
    policy — item-looping resolution for an on_demand ``model_chat``, the flat
    callable resolution otherwise.
    """
    from osprey.health.config import (
        CategoryRecord,
        Cost,
        resolve_callable_timeout_s,
        resolve_item_looping_on_demand_timeout,
    )

    override = settings.overrides.get(name) if settings else None
    cost = override.cost if override and override.cost is not None else default_cost
    override_timeout = override.timeout_s if override else None

    if name == "model_chat" and cost is Cost.ON_DEMAND:
        from osprey.health.core.model_chat import unique_model_pairs

        # Size the budget with the category's own pairing rule so the record's
        # on_demand budget matches the per-item budget it computes internally.
        n_items = len(unique_model_pairs(expanded or {}))
        budget, _ = resolve_item_looping_on_demand_timeout(n_items, override_timeout)
        return CategoryRecord(name=name, cost=cost, timeout_s=budget, func=func)

    budget = resolve_callable_timeout_s(cost, override_timeout, suite_timeout_s)
    return CategoryRecord(name=name, cost=cost, timeout_s=budget, func=func)


def skip_record(name: str, message: str, suite_timeout_s: float) -> CategoryRecord:
    """Build a poll-class record emitting a single ``skip`` row.

    Used for config-dependent categories when the config could not be loaded:
    the row always renders (poll cost, so ``--full`` gating never replaces it
    with an on_demand hint) and carries the degraded reason.
    """
    from osprey.health.config import CategoryRecord, Cost
    from osprey.health.models import CheckResult, Status

    def _run() -> list[CheckResult]:
        return [CheckResult(name, name, Status.SKIP, message)]

    return CategoryRecord(name=name, cost=Cost.POLL, timeout_s=suite_timeout_s, func=_run)


def build_records(
    config_state: ConfigState,
    expanded: dict[str, Any] | None,
    settings: HealthSettings | None,
    config_ok: bool,
    project_path: Path,
    suite_timeout_s: float,
) -> tuple[list[CategoryRecord], list[Any]]:
    """Assemble the merged category records and any plugin-load error rows.

    Core categories are always present. When the config loaded and its
    ``health:`` section parsed (``config_ok``), YAML and plugin categories are
    merged too and plugin-load failures are returned as diagnostic error rows;
    otherwise config-dependent core categories collapse to "config unavailable"
    skip rows and no YAML/plugin categories are loaded.
    """
    from osprey.health.config import Cost
    from osprey.health.core import CORE_CATEGORY_NAMES, get_core_category_factory

    records: list[CategoryRecord] = []

    for name in CORE_CATEGORY_NAMES:
        if name == "configuration":
            factory = get_core_category_factory(name)
            func = factory(config_state, context=None)
            records.append(core_record(name, func, Cost.POLL, settings, expanded, suite_timeout_s))
            continue

        if name in CONFIG_DEPENDENT and not config_ok:
            records.append(skip_record(name, "config unavailable", suite_timeout_s))
            continue

        factory = get_core_category_factory(name)
        if name in ("file_system", "channel_finder"):
            # Both resolve on-disk paths relative to the project root, so they
            # take the same ``cwd`` thread-through the uniform call omits.
            func = factory(expanded, context=None, cwd=project_path)
        else:
            func = factory(expanded, context=None)
        default_cost = Cost.ON_DEMAND if name in ON_DEMAND_CORE else Cost.POLL
        records.append(core_record(name, func, default_cost, settings, expanded, suite_timeout_s))

    extra_rows: list[Any] = []
    if config_ok and settings is not None:
        from osprey.health.plugins import load_plugin_categories

        records.extend(settings.categories.values())
        plugin_result = load_plugin_categories(settings)
        records.extend(plugin_result.categories.values())
        extra_rows = plugin_result.errors

    return records, extra_rows
