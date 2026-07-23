"""Tests for ``health.plugins`` loading (task 4.9).

Fake plugin modules are injected into ``sys.modules`` so ``importlib`` resolves
them without touching the filesystem. Covers the happy path (sync + async
callables), every failure mode (import error, missing/bad entrypoint, bad return,
invalid entry), collisions against core/YAML/earlier-plugin names, and metadata
cost/timeout overrides.
"""

from __future__ import annotations

import sys
import types

from osprey.health.config import (
    DEFAULT_ON_DEMAND_CALLABLE_TIMEOUT_S,
    CategoryOverride,
    CategoryRecord,
    Cost,
    HealthSettings,
)
from osprey.health.models import CheckResult, Status
from osprey.health.plugins import PLUGINS_DIAGNOSTIC_CATEGORY, load_plugin_categories

SUITE_TIMEOUT = 30.0


def _settings(*, plugins=None, categories=None, overrides=None) -> HealthSettings:
    return HealthSettings(
        suite_timeout_s=SUITE_TIMEOUT,
        interval_s=60.0,
        on_demand_timeout_s=None,
        categories=categories or {},
        overrides=overrides or {},
        plugins=plugins or [],
    )


def _install(monkeypatch, name: str, entrypoint) -> None:
    """Register a fake plugin module exposing ``get_health_categories``."""
    module = types.ModuleType(name)
    if entrypoint is not _MISSING:
        module.get_health_categories = entrypoint
    monkeypatch.setitem(sys.modules, name, module)


_MISSING = object()


def _sync_cat() -> list[CheckResult]:
    return [CheckResult("row", "alpha", Status.OK, "ok")]


async def _async_cat() -> list[CheckResult]:
    return [CheckResult("row", "beta", Status.OK, "ok")]


def test_happy_path_sync_and_async(monkeypatch) -> None:
    _install(monkeypatch, "plug_ok", lambda: {"alpha": _sync_cat, "beta": _async_cat})
    result = load_plugin_categories(_settings(plugins=["plug_ok"]))

    assert result.errors == []
    assert set(result.categories) == {"alpha", "beta"}
    alpha = result.categories["alpha"]
    assert isinstance(alpha, CategoryRecord)
    assert alpha.func is _sync_cat  # callable captured, not invoked
    assert alpha.cost is Cost.POLL
    assert alpha.timeout_s == SUITE_TIMEOUT
    assert result.categories["beta"].func is _async_cat


def test_import_failure_is_error_row() -> None:
    result = load_plugin_categories(_settings(plugins=["osprey.does.not.exist"]))

    assert result.categories == {}
    assert len(result.errors) == 1
    row = result.errors[0]
    assert row.status is Status.ERROR
    assert row.category == PLUGINS_DIAGNOSTIC_CATEGORY
    assert "import" in row.message.lower()


def test_missing_entrypoint_is_error_row(monkeypatch) -> None:
    _install(monkeypatch, "plug_no_entry", _MISSING)
    result = load_plugin_categories(_settings(plugins=["plug_no_entry"]))

    assert result.categories == {}
    assert len(result.errors) == 1
    assert "does not define" in result.errors[0].message


def test_non_callable_entrypoint_is_error_row(monkeypatch) -> None:
    _install(monkeypatch, "plug_bad_entry", "not-callable")
    result = load_plugin_categories(_settings(plugins=["plug_bad_entry"]))

    assert result.categories == {}
    assert "does not define" in result.errors[0].message


def test_entrypoint_raises_is_error_row(monkeypatch) -> None:
    def boom():
        raise RuntimeError("kaboom")

    _install(monkeypatch, "plug_raise", boom)
    result = load_plugin_categories(_settings(plugins=["plug_raise"]))

    assert result.categories == {}
    assert "raised" in result.errors[0].message
    assert "kaboom" in result.errors[0].message


def test_bad_return_type_is_error_row(monkeypatch) -> None:
    _install(monkeypatch, "plug_badret", lambda: ["not", "a", "dict"])
    result = load_plugin_categories(_settings(plugins=["plug_badret"]))

    assert result.categories == {}
    assert "must return a dict" in result.errors[0].message


def test_invalid_entry_value_is_error_row(monkeypatch) -> None:
    _install(monkeypatch, "plug_badval", lambda: {"alpha": "not-callable"})
    result = load_plugin_categories(_settings(plugins=["plug_badval"]))

    assert result.categories == {}
    assert "invalid category entry" in result.errors[0].message


def test_collision_with_core_name_is_error_row(monkeypatch) -> None:
    _install(monkeypatch, "plug_core_clash", lambda: {"providers": _sync_cat})
    result = load_plugin_categories(_settings(plugins=["plug_core_clash"]))

    assert result.categories == {}
    assert len(result.errors) == 1
    assert "collides" in result.errors[0].message


def test_collision_with_yaml_name_is_error_row(monkeypatch) -> None:
    _install(monkeypatch, "plug_yaml_clash", lambda: {"myyaml": _sync_cat})
    yaml_cat = CategoryRecord(name="myyaml", cost=Cost.POLL, timeout_s=SUITE_TIMEOUT, checks=[])
    result = load_plugin_categories(
        _settings(plugins=["plug_yaml_clash"], categories={"myyaml": yaml_cat})
    )

    assert result.categories == {}
    assert "collides" in result.errors[0].message


def test_collision_with_earlier_plugin_first_wins(monkeypatch) -> None:
    _install(monkeypatch, "plug_a", lambda: {"dup": _sync_cat})
    _install(monkeypatch, "plug_b", lambda: {"dup": _async_cat})
    result = load_plugin_categories(_settings(plugins=["plug_a", "plug_b"]))

    assert set(result.categories) == {"dup"}
    assert result.categories["dup"].func is _sync_cat  # first loaded wins
    assert len(result.errors) == 1
    assert "collides" in result.errors[0].message


def test_override_applies_cost_and_timeout(monkeypatch) -> None:
    _install(monkeypatch, "plug_over", lambda: {"alpha": _sync_cat})
    overrides = {"alpha": CategoryOverride(cost=Cost.ON_DEMAND, timeout_s=12.0)}
    result = load_plugin_categories(_settings(plugins=["plug_over"], overrides=overrides))

    record = result.categories["alpha"]
    assert record.cost is Cost.ON_DEMAND
    assert record.timeout_s == 12.0


def test_override_cost_only_uses_on_demand_default(monkeypatch) -> None:
    _install(monkeypatch, "plug_over2", lambda: {"alpha": _sync_cat})
    overrides = {"alpha": CategoryOverride(cost=Cost.ON_DEMAND)}
    result = load_plugin_categories(_settings(plugins=["plug_over2"], overrides=overrides))

    record = result.categories["alpha"]
    assert record.cost is Cost.ON_DEMAND
    assert record.timeout_s == DEFAULT_ON_DEMAND_CALLABLE_TIMEOUT_S


def test_override_timeout_only_keeps_poll(monkeypatch) -> None:
    _install(monkeypatch, "plug_over3", lambda: {"alpha": _sync_cat})
    overrides = {"alpha": CategoryOverride(timeout_s=7.5)}
    result = load_plugin_categories(_settings(plugins=["plug_over3"], overrides=overrides))

    record = result.categories["alpha"]
    assert record.cost is Cost.POLL
    assert record.timeout_s == 7.5
