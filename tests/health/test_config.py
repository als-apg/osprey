"""Unit tests for ``health:`` configuration parsing and validation."""

from __future__ import annotations

import pytest

from osprey.errors import ConfigurationError
from osprey.health.config import (
    CORE_CATEGORY_NAMES,
    DEFAULT_HEALTH_TITLE,
    DEFAULT_ON_DEMAND_CALLABLE_TIMEOUT_S,
    DEFAULT_PROBE_TIMEOUTS,
    DEFAULT_SUITE_TIMEOUT_S,
    CategoryOverride,
    CategoryRecord,
    CheckSpec,
    Cost,
    HealthSettings,
    parse_health_config,
    resolve_callable_timeout_s,
    resolve_item_looping_on_demand_timeout,
)
from osprey.health.models import Status

# --- Empty / defaults -------------------------------------------------------


def test_parse_none_yields_defaults() -> None:
    s = parse_health_config(None)
    assert isinstance(s, HealthSettings)
    assert s.suite_timeout_s == DEFAULT_SUITE_TIMEOUT_S
    assert s.on_demand_timeout_s is None
    assert s.interval_s == 60.0  # max(60, 2*30)
    assert s.title == DEFAULT_HEALTH_TITLE
    assert s.categories == {}
    assert s.overrides == {}
    assert s.plugins == []


def test_parse_empty_dict_yields_defaults() -> None:
    assert parse_health_config({}).suite_timeout_s == DEFAULT_SUITE_TIMEOUT_S


def test_non_mapping_health_section_raises() -> None:
    with pytest.raises(ConfigurationError):
        parse_health_config(["not", "a", "mapping"])  # type: ignore[arg-type]


# --- suite_timeout_s / interval_s -------------------------------------------


def test_suite_timeout_override_derives_interval() -> None:
    s = parse_health_config({"suite_timeout_s": 40})
    assert s.suite_timeout_s == 40.0
    assert s.interval_s == 80.0  # max(60, 2*40)


def test_interval_default_floor_is_60() -> None:
    # 2 * 10 = 20, floored to 60.
    assert parse_health_config({"suite_timeout_s": 10}).interval_s == 60.0


def test_explicit_interval_greater_than_suite_ok() -> None:
    assert parse_health_config({"suite_timeout_s": 30, "interval_s": 31}).interval_s == 31.0


def test_explicit_interval_not_greater_than_suite_raises() -> None:
    with pytest.raises(ConfigurationError):
        parse_health_config({"suite_timeout_s": 30, "interval_s": 30})
    with pytest.raises(ConfigurationError):
        parse_health_config({"suite_timeout_s": 30, "interval_s": 20})


def test_suite_timeout_non_numeric_raises() -> None:
    with pytest.raises(ConfigurationError):
        parse_health_config({"suite_timeout_s": "soon"})


def test_suite_timeout_bool_rejected() -> None:
    # bool is an int subclass — must not be accepted as a numeric timeout.
    with pytest.raises(ConfigurationError):
        parse_health_config({"suite_timeout_s": True})


def test_suite_timeout_non_positive_raises() -> None:
    with pytest.raises(ConfigurationError):
        parse_health_config({"suite_timeout_s": 0})


def test_on_demand_timeout_explicit_value() -> None:
    assert parse_health_config({"on_demand_timeout_s": 120}).on_demand_timeout_s == 120.0


# --- plugins ----------------------------------------------------------------


def test_plugins_parsed() -> None:
    s = parse_health_config({"plugins": ["my.mod", "other.mod"]})
    assert s.plugins == ["my.mod", "other.mod"]


def test_plugins_not_list_raises() -> None:
    with pytest.raises(ConfigurationError):
        parse_health_config({"plugins": "my.mod"})


def test_plugins_non_string_items_raise() -> None:
    with pytest.raises(ConfigurationError):
        parse_health_config({"plugins": ["ok", 3]})


# --- title ------------------------------------------------------------------


def test_title_absent_defaults() -> None:
    assert parse_health_config({}).title == DEFAULT_HEALTH_TITLE == "System Health"


def test_title_present_used() -> None:
    assert parse_health_config({"title": "ALS Booster Health"}).title == "ALS Booster Health"


def test_title_explicit_none_defaults() -> None:
    # An explicit null in YAML falls back to the default rather than raising.
    assert parse_health_config({"title": None}).title == DEFAULT_HEALTH_TITLE


def test_title_non_string_raises() -> None:
    with pytest.raises(ConfigurationError):
        parse_health_config({"title": 123})


def test_title_does_not_affect_other_parsing() -> None:
    # Adding the key must not perturb any sibling parsing behaviour.
    s = parse_health_config({"title": "Custom", "suite_timeout_s": 40})
    assert s.title == "Custom"
    assert s.suite_timeout_s == 40.0
    assert s.interval_s == 80.0  # max(60, 2*40) — unchanged by title
    assert s.categories == {}
    assert s.overrides == {}
    assert s.plugins == []


# --- YAML declarative categories --------------------------------------------


def _http_category(**overrides):
    cat = {
        "checks": [
            {"name": "web", "type": "http", "url": "http://x", "expect_status": 200},
        ]
    }
    cat.update(overrides)
    return {"categories": {"my_cat": cat}}


def test_yaml_category_compiles_to_record() -> None:
    s = parse_health_config(_http_category())
    assert set(s.categories) == {"my_cat"}
    rec = s.categories["my_cat"]
    assert isinstance(rec, CategoryRecord)
    assert rec.name == "my_cat"
    assert rec.cost is Cost.POLL
    assert rec.func is None
    assert rec.checks is not None and len(rec.checks) == 1


def test_check_spec_defaults() -> None:
    rec = parse_health_config(_http_category()).categories["my_cat"]
    assert rec.checks is not None
    check = rec.checks[0]
    assert isinstance(check, CheckSpec)
    assert check.name == "web"
    assert check.type == "http"
    assert check.timeout_s == DEFAULT_PROBE_TIMEOUTS["http"]
    assert check.timeout_status is Status.ERROR
    assert check.requires == ()
    # Non-reserved keys become params; reserved keys are excluded.
    assert check.params == {"url": "http://x", "expect_status": 200}


def test_check_spec_explicit_timeout_and_status() -> None:
    cfg = {
        "categories": {
            "c": {
                "checks": [
                    {
                        "name": "web",
                        "type": "http",
                        "timeout_s": 12,
                        "timeout_status": "warning",
                    }
                ]
            }
        }
    }
    check = parse_health_config(cfg).categories["c"].checks[0]  # type: ignore[index]
    assert check.timeout_s == 12.0
    assert check.timeout_status is Status.WARNING


def test_default_timeout_per_probe_type() -> None:
    for probe_type, expected in DEFAULT_PROBE_TIMEOUTS.items():
        cfg = {"categories": {"c": {"checks": [{"name": "n", "type": probe_type}]}}}
        check = parse_health_config(cfg).categories["c"].checks[0]  # type: ignore[index]
        assert check.timeout_s == expected


def test_unknown_probe_type_raises() -> None:
    cfg = {"categories": {"c": {"checks": [{"name": "n", "type": "smtp"}]}}}
    with pytest.raises(ConfigurationError):
        parse_health_config(cfg)


def test_missing_check_name_raises() -> None:
    cfg = {"categories": {"c": {"checks": [{"type": "http"}]}}}
    with pytest.raises(ConfigurationError):
        parse_health_config(cfg)


def test_missing_check_type_raises() -> None:
    cfg = {"categories": {"c": {"checks": [{"name": "n"}]}}}
    with pytest.raises(ConfigurationError):
        parse_health_config(cfg)


def test_invalid_timeout_status_raises() -> None:
    cfg = {
        "categories": {"c": {"checks": [{"name": "n", "type": "http", "timeout_status": "boom"}]}}
    }
    with pytest.raises(ConfigurationError):
        parse_health_config(cfg)


def test_yaml_category_explicit_cost_on_demand_default_timeout() -> None:
    cfg = {"categories": {"c": {"cost": "on_demand", "checks": [{"name": "n", "type": "http"}]}}}
    rec = parse_health_config(cfg).categories["c"]
    assert rec.cost is Cost.ON_DEMAND
    assert rec.timeout_s == DEFAULT_ON_DEMAND_CALLABLE_TIMEOUT_S


def test_yaml_category_poll_default_timeout_is_suite() -> None:
    rec = parse_health_config(_http_category()).categories["my_cat"]
    assert rec.timeout_s == DEFAULT_SUITE_TIMEOUT_S


def test_yaml_category_explicit_category_timeout() -> None:
    rec = parse_health_config(_http_category(timeout_s=9)).categories["my_cat"]
    assert rec.timeout_s == 9.0


def test_invalid_cost_raises() -> None:
    cfg = {"categories": {"c": {"cost": "sometimes", "checks": [{"name": "n", "type": "http"}]}}}
    with pytest.raises(ConfigurationError):
        parse_health_config(cfg)


# --- requires validation ----------------------------------------------------


def test_requires_valid_backward_reference() -> None:
    cfg = {
        "categories": {
            "c": {
                "checks": [
                    {"name": "a", "type": "http"},
                    {"name": "b", "type": "http", "requires": ["a"]},
                ]
            }
        }
    }
    rec = parse_health_config(cfg).categories["c"]
    assert rec.checks[1].requires == ("a",)  # type: ignore[index]


def test_requires_forward_reference_raises() -> None:
    cfg = {
        "categories": {
            "c": {
                "checks": [
                    {"name": "a", "type": "http", "requires": ["b"]},
                    {"name": "b", "type": "http"},
                ]
            }
        }
    }
    with pytest.raises(ConfigurationError):
        parse_health_config(cfg)


def test_requires_unknown_target_raises() -> None:
    cfg = {"categories": {"c": {"checks": [{"name": "a", "type": "http", "requires": ["ghost"]}]}}}
    with pytest.raises(ConfigurationError):
        parse_health_config(cfg)


def test_requires_self_reference_raises() -> None:
    cfg = {"categories": {"c": {"checks": [{"name": "a", "type": "http", "requires": ["a"]}]}}}
    with pytest.raises(ConfigurationError):
        parse_health_config(cfg)


def test_duplicate_check_names_raise() -> None:
    cfg = {
        "categories": {
            "c": {"checks": [{"name": "a", "type": "http"}, {"name": "a", "type": "http"}]}
        }
    }
    with pytest.raises(ConfigurationError):
        parse_health_config(cfg)


def test_requires_not_a_list_raises() -> None:
    cfg = {"categories": {"c": {"checks": [{"name": "a", "type": "http", "requires": "b"}]}}}
    with pytest.raises(ConfigurationError):
        parse_health_config(cfg)


# --- core-name collisions & metadata overrides ------------------------------


def test_checks_under_core_name_raises() -> None:
    for core in ("providers", "model_chat", "file_system"):
        cfg = {"categories": {core: {"checks": [{"name": "n", "type": "http"}]}}}
        with pytest.raises(ConfigurationError):
            parse_health_config(cfg)


def test_metadata_only_override_for_core_name() -> None:
    cfg = {"categories": {"providers": {"timeout_s": 10, "cost": "on_demand"}}}
    s = parse_health_config(cfg)
    assert "providers" not in s.categories  # not a declarative category
    override = s.overrides["providers"]
    assert isinstance(override, CategoryOverride)
    assert override.timeout_s == 10.0
    assert override.cost is Cost.ON_DEMAND


def test_metadata_only_override_for_plugin_name() -> None:
    # A non-core name with no checks is a metadata override (e.g. for a plugin).
    s = parse_health_config({"categories": {"my_plugin_cat": {"timeout_s": 15}}})
    assert s.overrides["my_plugin_cat"].timeout_s == 15.0
    assert "my_plugin_cat" not in s.categories


def test_empty_metadata_override_block() -> None:
    s = parse_health_config({"categories": {"providers": {}}})
    override = s.overrides["providers"]
    assert override.cost is None and override.timeout_s is None


def test_core_category_names_content() -> None:
    assert "providers" in CORE_CATEGORY_NAMES
    assert "claude_cli_pinned" in CORE_CATEGORY_NAMES
    assert len(CORE_CATEGORY_NAMES) == 11


# --- timeout resolution helpers ---------------------------------------------


def test_resolve_callable_timeout_poll_default() -> None:
    assert resolve_callable_timeout_s(Cost.POLL, None, 30.0) == 30.0


def test_resolve_callable_timeout_on_demand_default() -> None:
    assert resolve_callable_timeout_s(Cost.ON_DEMAND, None, 30.0) == 60.0


def test_resolve_callable_timeout_override_wins() -> None:
    assert resolve_callable_timeout_s(Cost.POLL, 8.0, 30.0) == 8.0
    assert resolve_callable_timeout_s(Cost.ON_DEMAND, 8.0, 30.0) == 8.0


def test_item_looping_default_product() -> None:
    cat, per_item = resolve_item_looping_on_demand_timeout(3, None)
    assert cat == 180.0  # 60 * 3
    assert per_item == 60.0


def test_item_looping_zero_items_uses_max_n_1() -> None:
    cat, per_item = resolve_item_looping_on_demand_timeout(0, None)
    assert cat == 60.0  # 60 * max(0, 1)
    assert per_item == 60.0


def test_item_looping_override_divides_per_item() -> None:
    cat, per_item = resolve_item_looping_on_demand_timeout(4, 120.0)
    assert cat == 120.0
    assert per_item == 30.0  # 120 / 4


def test_item_looping_override_zero_items_no_zero_division() -> None:
    cat, per_item = resolve_item_looping_on_demand_timeout(0, 45.0)
    assert cat == 45.0
    assert per_item == 45.0  # 45 / max(0, 1)


# --- multiple categories ----------------------------------------------------


def test_multiple_yaml_and_override_categories() -> None:
    cfg = {
        "suite_timeout_s": 20,
        "categories": {
            "web_a": {"checks": [{"name": "a", "type": "http"}]},
            "web_b": {"checks": [{"name": "b", "type": "mcp", "url": "http://m"}]},
            "providers": {"timeout_s": 7},
        },
    }
    s = parse_health_config(cfg)
    assert set(s.categories) == {"web_a", "web_b"}
    assert set(s.overrides) == {"providers"}
    assert s.categories["web_b"].checks[0].type == "mcp"  # type: ignore[index]
