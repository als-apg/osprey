"""Tests for ``health.plugins`` loading (task 4.9).

Fake plugin modules are injected into ``sys.modules`` so ``importlib`` resolves
them without touching the filesystem. Covers the happy path (sync + async
callables), every failure mode (import error, missing/bad entrypoint, bad return,
invalid entry), collisions against core/YAML/earlier-plugin names, and metadata
cost/timeout overrides.

The second half of the file covers the file-path entry form — an entry ending
in ``.py`` names a file, resolved against the project root — with real files
under ``tmp_path`` rather than ``sys.modules`` fakes.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from osprey.health.config import (
    DEFAULT_ON_DEMAND_CALLABLE_TIMEOUT_S,
    CategoryOverride,
    CategoryRecord,
    Cost,
    HealthSettings,
)
from osprey.health.models import CheckResult, Status
from osprey.health.plugins import (
    PLUGIN_MODULE_PREFIX,
    PLUGINS_DIAGNOSTIC_CATEGORY,
    load_plugin_categories,
)

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


# --- file-path entries ------------------------------------------------------
#
# An entry ending in ``.py`` names a FILE, resolved against the project root
# (absolute entries are used as-is). Everything below writes a real file under
# ``tmp_path`` — the point of the form is that the file needs no packaging and
# no ``PYTHONPATH``, so faking it through ``sys.modules`` would test nothing.


_PLUGIN_SOURCE = """
from osprey.health.models import CheckResult, Status


def _facility() -> list:
    return [CheckResult("row", "facility", Status.OK, "ok")]


def get_health_categories():
    return {"%s": _facility}
"""


@pytest.fixture(autouse=True)
def _drop_synthetic_modules():
    """Leave no synthetic plugin module behind for the next test to inherit."""
    yield
    for name in [n for n in sys.modules if n.startswith(PLUGIN_MODULE_PREFIX)]:
        del sys.modules[name]


def _write_plugin(root: Path, relative: str, category: str = "facility") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_PLUGIN_SOURCE % category, encoding="utf-8")
    return path


def test_relative_path_entry_resolves_against_project_root(tmp_path: Path) -> None:
    _write_plugin(tmp_path, "health/facility_checks.py")

    result = load_plugin_categories(
        _settings(plugins=["./health/facility_checks.py"]), project_root=tmp_path
    )

    assert result.errors == []
    assert set(result.categories) == {"facility"}
    record = result.categories["facility"]
    assert record.cost is Cost.POLL
    assert record.timeout_s == SUITE_TIMEOUT


def test_relative_path_without_dot_slash_also_resolves(tmp_path: Path) -> None:
    _write_plugin(tmp_path, "health/facility_checks.py")

    result = load_plugin_categories(
        _settings(plugins=["health/facility_checks.py"]), project_root=tmp_path
    )

    assert result.errors == []
    assert set(result.categories) == {"facility"}


def test_relative_path_is_not_resolved_against_the_cwd(tmp_path: Path, monkeypatch) -> None:
    """The anchor is the project root the caller passed, never the process CWD."""
    _write_plugin(tmp_path, "health/facility_checks.py")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = load_plugin_categories(
        _settings(plugins=["health/facility_checks.py"]), project_root=tmp_path
    )

    assert result.errors == []
    assert set(result.categories) == {"facility"}


def test_absolute_path_entry_is_used_as_is(tmp_path: Path) -> None:
    path = _write_plugin(tmp_path / "outside", "checks.py")
    other_root = tmp_path / "project"
    other_root.mkdir()

    result = load_plugin_categories(_settings(plugins=[str(path)]), project_root=other_root)

    assert result.errors == []
    assert set(result.categories) == {"facility"}


def test_missing_file_is_error_row(tmp_path: Path) -> None:
    result = load_plugin_categories(_settings(plugins=["./health/nope.py"]), project_root=tmp_path)

    assert result.categories == {}
    assert len(result.errors) == 1
    row = result.errors[0]
    assert row.status is Status.ERROR
    assert row.category == PLUGINS_DIAGNOSTIC_CATEGORY
    assert "not found" in row.message
    assert str(tmp_path / "health" / "nope.py") in row.message


def test_bad_syntax_file_is_error_row(tmp_path: Path) -> None:
    path = tmp_path / "broken.py"
    path.write_text("def get_health_categories(:\n", encoding="utf-8")

    result = load_plugin_categories(_settings(plugins=["./broken.py"]), project_root=tmp_path)

    assert result.categories == {}
    assert len(result.errors) == 1
    assert "import" in result.errors[0].message.lower()


def test_file_raising_on_import_is_error_row(tmp_path: Path) -> None:
    path = tmp_path / "explodes.py"
    path.write_text("raise RuntimeError('kaboom')\n", encoding="utf-8")

    result = load_plugin_categories(_settings(plugins=["./explodes.py"]), project_root=tmp_path)

    assert result.categories == {}
    assert len(result.errors) == 1
    assert "kaboom" in result.errors[0].message


def test_file_without_entrypoint_is_error_row(tmp_path: Path) -> None:
    (tmp_path / "empty.py").write_text("VALUE = 1\n", encoding="utf-8")

    result = load_plugin_categories(_settings(plugins=["./empty.py"]), project_root=tmp_path)

    assert result.categories == {}
    assert "does not define" in result.errors[0].message


def test_loaded_file_is_registered_in_sys_modules(tmp_path: Path) -> None:
    """Registered before exec: dataclasses and PEP 563 annotations look it up there."""
    path = _write_plugin(tmp_path, "checks.py")

    result = load_plugin_categories(_settings(plugins=["./checks.py"]), project_root=tmp_path)

    assert result.errors == []
    registered = [n for n in sys.modules if n.startswith(PLUGIN_MODULE_PREFIX)]
    assert len(registered) == 1
    module = sys.modules[registered[0]]
    assert module.__file__ == str(path)


def test_dataclass_in_a_plugin_file_loads(tmp_path: Path) -> None:
    """The concrete reason the module is in ``sys.modules`` before ``exec_module``."""
    (tmp_path / "dc.py").write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "from osprey.health.models import CheckResult, Status\n"
        "\n"
        "@dataclass\n"
        "class Thing:\n"
        "    name: str\n"
        "\n"
        "def get_health_categories():\n"
        "    return {'dc': lambda: [CheckResult(Thing('row').name, 'dc', Status.OK, 'ok')]}\n",
        encoding="utf-8",
    )

    result = load_plugin_categories(_settings(plugins=["./dc.py"]), project_root=tmp_path)

    assert result.errors == []
    assert set(result.categories) == {"dc"}


def test_module_name_is_deterministic_and_reload_replaces_it(tmp_path: Path) -> None:
    """A refresh cycle re-executes the file under the same synthetic name."""
    _write_plugin(tmp_path, "checks.py", category="first")
    first = load_plugin_categories(_settings(plugins=["./checks.py"]), project_root=tmp_path)
    first_names = [n for n in sys.modules if n.startswith(PLUGIN_MODULE_PREFIX)]
    first_module = sys.modules[first_names[0]]

    _write_plugin(tmp_path, "checks.py", category="second")
    second = load_plugin_categories(_settings(plugins=["./checks.py"]), project_root=tmp_path)
    second_names = [n for n in sys.modules if n.startswith(PLUGIN_MODULE_PREFIX)]

    assert set(first.categories) == {"first"}
    assert set(second.categories) == {"second"}  # re-executed, not cached
    assert first_names == second_names  # deterministic name, derived from the path
    assert sys.modules[second_names[0]] is not first_module  # entry replaced


def test_path_and_dotted_entries_mix(tmp_path: Path, monkeypatch) -> None:
    _install(monkeypatch, "plug_dotted", lambda: {"alpha": _sync_cat})
    _write_plugin(tmp_path, "checks.py")

    result = load_plugin_categories(
        _settings(plugins=["plug_dotted", "./checks.py"]), project_root=tmp_path
    )

    assert result.errors == []
    assert set(result.categories) == {"alpha", "facility"}


def test_dotted_entry_needs_no_anchor(monkeypatch) -> None:
    """The anchor is optional; a dotted entry never consults it."""
    _install(monkeypatch, "plug_anchorless", lambda: {"alpha": _sync_cat})

    result = load_plugin_categories(_settings(plugins=["plug_anchorless"]))

    assert result.errors == []
    assert set(result.categories) == {"alpha"}


def test_path_entry_collision_is_error_row(tmp_path: Path) -> None:
    _write_plugin(tmp_path, "a.py", category="dup")
    _write_plugin(tmp_path, "b.py", category="dup")

    result = load_plugin_categories(_settings(plugins=["./a.py", "./b.py"]), project_root=tmp_path)

    assert set(result.categories) == {"dup"}
    assert len(result.errors) == 1
    assert "collides" in result.errors[0].message
    assert "./b.py" in result.errors[0].message
