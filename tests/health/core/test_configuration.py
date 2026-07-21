"""Tests for the core ``configuration`` health category.

Covers the lazy ``CORE_CATEGORIES`` registry surface and the per-row status
semantics of the migrated configuration checks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.health.core import (
    CORE_CATEGORIES,
    CORE_CATEGORY_NAMES,
    get_core_category_factory,
)
from osprey.health.core.configuration import ConfigState, configuration
from osprey.health.models import CheckResult, Status

_CANONICAL_NAMES = (
    "configuration",
    "file_system",
    "python_environment",
    "containers",
    "openobserve",
    "providers",
    "claude_cli",
    "claude_cli_pinned",
    "model_chat",
)


def _run(state: ConfigState) -> list[CheckResult]:
    """Build and invoke the configuration category callable."""
    callable_ = configuration(state)
    return callable_()


def _by_name(results: list[CheckResult]) -> dict[str, CheckResult]:
    return {r.name: r for r in results}


def _loaded_state(config: dict, **kwargs) -> ConfigState:
    """A ConfigState for a config.yml that exists and parsed cleanly."""
    return ConfigState(
        config_path=Path("/proj/config.yml"),
        exists=True,
        cwd=Path("/proj"),
        config=config,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Registry surface
# --------------------------------------------------------------------------- #


class TestCoreRegistry:
    def test_canonical_names_present_without_import(self):
        assert set(CORE_CATEGORY_NAMES) == set(_CANONICAL_NAMES)
        assert set(CORE_CATEGORIES) == set(_CANONICAL_NAMES)
        assert len(CORE_CATEGORIES) == 9

    def test_contains(self):
        assert "configuration" in CORE_CATEGORIES
        assert "not_a_category" not in CORE_CATEGORIES

    def test_lazy_lookup_returns_configuration_factory(self):
        factory = CORE_CATEGORIES["configuration"]
        assert factory is configuration
        assert get_core_category_factory("configuration") is configuration

    def test_unknown_name_raises_keyerror(self):
        with pytest.raises(KeyError):
            CORE_CATEGORIES["nope"]
        with pytest.raises(KeyError):
            get_core_category_factory("nope")

    def test_factory_returns_no_arg_callable(self):
        factory = CORE_CATEGORIES["configuration"]
        cb = factory(_loaded_state({}), None)
        results = cb()
        assert isinstance(results, list)
        assert all(isinstance(r, CheckResult) for r in results)


# --------------------------------------------------------------------------- #
# config_file_exists / yaml_valid / health_config (error-capable rows)
# --------------------------------------------------------------------------- #


class TestLoadOutcomeRows:
    def test_missing_config_file_is_single_error_row(self):
        state = ConfigState(
            config_path=Path("/proj/config.yml"),
            exists=False,
            cwd=Path("/proj"),
        )
        results = _run(state)
        assert len(results) == 1
        row = results[0]
        assert row.name == "config_file_exists"
        assert row.status == Status.ERROR
        assert "not found" in row.message
        assert "/proj" in row.details

    def test_existing_config_emits_ok_row(self):
        results = _by_name(_run(_loaded_state({})))
        assert results["config_file_exists"].status == Status.OK
        assert "config.yml" in results["config_file_exists"].message

    def test_yaml_error_stops_after_yaml_valid_error(self):
        state = ConfigState(
            config_path=Path("/proj/config.yml"),
            exists=True,
            cwd=Path("/proj"),
            config=None,
            yaml_error="YAML parsing error: bad indent",
        )
        results = _run(state)
        names = [r.name for r in results]
        assert names == ["config_file_exists", "yaml_valid"]
        assert results[1].status == Status.ERROR
        assert "bad indent" in results[1].message

    def test_none_config_without_explicit_error_reports_empty(self):
        state = ConfigState(
            config_path=Path("/proj/config.yml"),
            exists=True,
            cwd=Path("/proj"),
            config=None,
        )
        results = _run(state)
        assert [r.name for r in results] == ["config_file_exists", "yaml_valid"]
        assert results[1].status == Status.ERROR
        assert results[1].message == "Config file is empty"

    def test_valid_yaml_emits_ok_row(self):
        results = _by_name(_run(_loaded_state({})))
        assert results["yaml_valid"].status == Status.OK

    def test_health_config_error_row_when_invalid(self):
        state = _loaded_state({}, health_error="interval_s must exceed suite_timeout_s")
        results = _by_name(_run(state))
        assert "health_config" in results
        assert results["health_config"].status == Status.ERROR
        assert "interval_s" in results["health_config"].message
        # Report still renders the rest of the category.
        assert "timezone" in results

    def test_no_health_config_row_when_valid(self):
        results = _by_name(_run(_loaded_state({})))
        assert "health_config" not in results


# --------------------------------------------------------------------------- #
# _check_config_structure
# --------------------------------------------------------------------------- #


class TestStructure:
    def test_no_models_section(self):
        results = _by_name(_run(_loaded_state({})))
        assert results["model_configs"].status == Status.WARNING
        assert "No models section" in results["model_configs"].message
        # model_configs_valid still emitted (unconditional), ok when no models.
        assert results["model_configs_valid"].status == Status.OK
        assert "recommended_models" not in results

    def test_missing_recommended_model(self):
        config = {"models": {"other": {"provider": "p", "model_id": "m"}}}
        results = _by_name(_run(_loaded_state(config)))
        assert results["recommended_models"].status == Status.WARNING
        assert "python_code_generator" in results["recommended_models"].message
        assert "model_configs" not in results

    def test_recommended_model_present(self):
        config = {"models": {"python_code_generator": {"provider": "p", "model_id": "m"}}}
        results = _by_name(_run(_loaded_state(config)))
        assert results["recommended_models"].status == Status.OK

    def test_invalid_model_config_missing_fields(self):
        config = {
            "models": {
                "python_code_generator": {"provider": "p"},  # missing model_id
                "bad": "not-a-dict",
            }
        }
        results = _by_name(_run(_loaded_state(config)))
        assert results["model_configs_valid"].status == Status.WARNING
        msg = results["model_configs_valid"].message
        assert "missing model_id" in msg
        assert "bad" in msg

    def test_valid_model_configs(self):
        config = {"models": {"python_code_generator": {"provider": "p", "model_id": "m"}}}
        results = _by_name(_run(_loaded_state(config)))
        assert results["model_configs_valid"].status == Status.OK

    def test_no_deployed_services(self):
        results = _by_name(_run(_loaded_state({})))
        assert results["deployed_services"].status == Status.WARNING

    def test_deployed_services_present(self):
        config = {"deployed_services": ["svc"], "services": {"svc": {}}}
        results = _by_name(_run(_loaded_state(config)))
        assert results["deployed_services"].status == Status.OK
        assert "svc" in results["deployed_services"].message

    def test_undefined_service_is_error(self):
        config = {"deployed_services": ["missing"], "services": {}}
        results = _by_name(_run(_loaded_state(config)))
        assert results["service_definitions"].status == Status.ERROR
        assert "missing" in results["service_definitions"].message

    def test_defined_services_ok(self):
        config = {"deployed_services": ["svc"], "services": {"svc": {}}}
        results = _by_name(_run(_loaded_state(config)))
        assert results["service_definitions"].status == Status.OK

    def test_no_api_providers(self):
        results = _by_name(_run(_loaded_state({})))
        assert results["api_providers"].status == Status.WARNING

    def test_api_providers_present(self):
        config = {"api": {"providers": {"anthropic": {}}}}
        results = _by_name(_run(_loaded_state(config)))
        assert results["api_providers"].status == Status.OK
        assert "anthropic" in results["api_providers"].message


# --------------------------------------------------------------------------- #
# _check_environment_variables
# --------------------------------------------------------------------------- #


class TestEnvironmentVariables:
    def test_no_var_refs_emits_no_row(self):
        results = _by_name(_run(_loaded_state({"plain": "value"})))
        assert "environment_variables" not in results

    def test_all_vars_set(self, monkeypatch):
        monkeypatch.setenv("HEALTH_TEST_VAR", "x")
        config = {"key": "${HEALTH_TEST_VAR}"}
        results = _by_name(_run(_loaded_state(config)))
        assert results["environment_variables"].status == Status.OK
        assert "1 environment variables set" in results["environment_variables"].message

    def test_missing_vars_warning(self, monkeypatch):
        monkeypatch.delenv("HEALTH_MISSING_VAR", raising=False)
        config = {"key": "${HEALTH_MISSING_VAR}"}
        results = _by_name(_run(_loaded_state(config)))
        assert results["environment_variables"].status == Status.WARNING
        assert "HEALTH_MISSING_VAR" in results["environment_variables"].message


# --------------------------------------------------------------------------- #
# _check_timezone
# --------------------------------------------------------------------------- #


class TestTimezone:
    def test_default_utc_is_warning(self):
        results = _by_name(_run(_loaded_state({})))
        assert results["timezone"].status == Status.WARNING
        assert "UTC" in results["timezone"].message

    def test_explicit_utc_is_warning(self):
        config = {"system": {"timezone": "UTC"}}
        results = _by_name(_run(_loaded_state(config)))
        assert results["timezone"].status == Status.WARNING

    def test_configured_timezone_is_ok(self):
        config = {"system": {"timezone": "Europe/Berlin"}}
        results = _by_name(_run(_loaded_state(config)))
        assert results["timezone"].status == Status.OK
        assert "Europe/Berlin" in results["timezone"].message

    def test_timezone_resolves_env_var(self, monkeypatch):
        monkeypatch.setenv("TZ", "America/New_York")
        config = {"system": {"timezone": "${TZ}"}}
        results = _by_name(_run(_loaded_state(config)))
        assert results["timezone"].status == Status.OK
        assert "America/New_York" in results["timezone"].message


def test_all_rows_carry_configuration_category():
    config = {
        "models": {"python_code_generator": {"provider": "p", "model_id": "m"}},
        "deployed_services": ["svc"],
        "services": {"svc": {}},
        "api": {"providers": {"anthropic": {}}},
        "system": {"timezone": "Europe/Berlin"},
    }
    results = _run(_loaded_state(config))
    assert results  # non-empty
    assert all(r.category == "configuration" for r in results)
