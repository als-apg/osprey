"""Tests for ``resolve_execution_method`` — the execution vocabulary choke point.

OSPREY ships exactly one Python execution backend (a host subprocess). These tests
pin the contract every reader of ``execution.execution_method`` depends on: legacy
values keep loading, the ``container`` behavior change is announced exactly once per
process, and unknown values fail loudly.
"""

import logging

import pytest

from osprey.utils import config as config_module
from osprey.utils.config import EXECUTION_METHOD_SUBPROCESS, resolve_execution_method


@pytest.fixture(autouse=True)
def _reset_container_warning():
    """Clear the module-level warn-once latch around every test."""
    config_module._container_method_warned = False
    yield
    config_module._container_method_warned = False


def _config(method):
    """Build a minimal config mapping carrying an execution method."""
    return {"execution": {"execution_method": method}}


class TestAcceptedValues:
    """Every accepted input resolves to the one backend that actually runs."""

    def test_subprocess_resolves_to_subprocess(self):
        assert resolve_execution_method(_config("subprocess")) == "subprocess"

    def test_local_resolves_to_subprocess(self):
        assert resolve_execution_method(_config("local")) == "subprocess"

    def test_container_resolves_to_subprocess(self):
        assert resolve_execution_method(_config("container")) == "subprocess"

    def test_missing_execution_section_resolves_to_subprocess(self):
        assert resolve_execution_method({}) == "subprocess"

    def test_missing_execution_method_key_resolves_to_subprocess(self):
        assert resolve_execution_method({"execution": {}}) == "subprocess"

    def test_none_config_resolves_to_subprocess(self):
        assert resolve_execution_method(None) == "subprocess"

    def test_null_execution_method_resolves_to_subprocess(self):
        # ``execution_method:`` with no value parses to None in YAML.
        assert resolve_execution_method(_config(None)) == "subprocess"

    def test_blank_execution_method_resolves_to_subprocess(self):
        assert resolve_execution_method(_config("   ")) == "subprocess"

    def test_values_are_case_and_whitespace_insensitive(self):
        assert resolve_execution_method(_config("  Local ")) == "subprocess"
        assert resolve_execution_method(_config("SUBPROCESS")) == "subprocess"

    def test_exported_constant_matches_return_value(self):
        assert EXECUTION_METHOD_SUBPROCESS == "subprocess"
        assert resolve_execution_method(_config("subprocess")) == EXECUTION_METHOD_SUBPROCESS


class TestContainerDeprecationWarning:
    """``container`` changes behavior (broken -> working), so it warns — once."""

    def test_container_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="CONFIG"):
            resolve_execution_method(_config("container"))

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "container" in message
        assert "subprocess" in message

    def test_container_warns_only_once_per_process(self, caplog):
        with caplog.at_level(logging.WARNING, logger="CONFIG"):
            for _ in range(5):
                assert resolve_execution_method(_config("container")) == "subprocess"

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f"expected exactly one warning, got {len(warnings)}"

    def test_warn_once_latch_is_module_level_not_per_config(self, caplog):
        """Distinct config mappings share the one process-wide latch."""
        with caplog.at_level(logging.WARNING, logger="CONFIG"):
            resolve_execution_method({"execution": {"execution_method": "container"}})
            resolve_execution_method(
                {"execution": {"execution_method": "container"}, "other": {"key": "value"}}
            )

        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1

    def test_warning_names_the_config_source(self, caplog):
        with caplog.at_level(logging.WARNING, logger="CONFIG"):
            resolve_execution_method(_config("container"), source="/facility/config.yml")

        message = caplog.records[0].getMessage()
        assert "/facility/config.yml" in message

    def test_warning_falls_back_to_active_config_path(self, caplog, monkeypatch):
        monkeypatch.setenv("OSPREY_CONFIG", "/tmp/derived-source/config.yml")

        with caplog.at_level(logging.WARNING, logger="CONFIG"):
            resolve_execution_method(_config("container"))

        assert "/tmp/derived-source/config.yml" in caplog.records[0].getMessage()

    def test_local_maps_silently(self, caplog):
        """Same behavior, new name — no warning, so nobody is trained to ignore them."""
        with caplog.at_level(logging.WARNING, logger="CONFIG"):
            resolve_execution_method(_config("local"))

        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_subprocess_and_unset_map_silently(self, caplog):
        with caplog.at_level(logging.WARNING, logger="CONFIG"):
            resolve_execution_method(_config("subprocess"))
            resolve_execution_method({})

        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


class TestRejectedValues:
    """Anything else is a config error naming the offending value."""

    @pytest.mark.parametrize(
        "method",
        ["jupyter", "docker", "podman", "remote", "kernel_gateway", "Subprocesses"],
    )
    def test_unknown_value_raises_value_error(self, method):
        with pytest.raises(ValueError) as exc_info:
            resolve_execution_method(_config(method))

        assert method in str(exc_info.value)
        assert "subprocess" in str(exc_info.value)

    def test_error_names_the_config_source(self):
        with pytest.raises(ValueError) as exc_info:
            resolve_execution_method(_config("jupyter"), source="/facility/config.yml")

        assert "/facility/config.yml" in str(exc_info.value)

    @pytest.mark.parametrize("method", [42, True, ["subprocess"], {"name": "subprocess"}])
    def test_non_string_value_raises_value_error(self, method):
        with pytest.raises(ValueError) as exc_info:
            resolve_execution_method(_config(method))

        assert "execution_method" in str(exc_info.value)

    def test_rejection_does_not_consume_the_warn_once_latch(self, caplog):
        with pytest.raises(ValueError):
            resolve_execution_method(_config("jupyter"))

        with caplog.at_level(logging.WARNING, logger="CONFIG"):
            resolve_execution_method(_config("container"))

        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


class TestLegacyConfigFileShim:
    """The end-to-end shim: a config.yml still saying ``container`` keeps working.

    The unit tests above pin the resolver against in-memory mappings. These load
    a real file through the config loader, because that is the path a facility
    that never edits its config.yml actually takes.
    """

    @staticmethod
    def _write_config(tmp_path, method, name="config.yml"):
        config_file = tmp_path / name
        config_file.write_text(
            f"project_root: {tmp_path}\nexecution:\n  execution_method: {method}\n"
        )
        return config_file

    @staticmethod
    def _deprecations(caplog):
        return [r for r in caplog.records if "is deprecated" in r.getMessage()]

    def test_container_config_file_loads_and_resolves_to_subprocess(self, tmp_path, caplog):
        from osprey.utils.config import get_full_configuration

        config_file = self._write_config(tmp_path, "container")

        with caplog.at_level(logging.WARNING, logger="CONFIG"):
            configurable = get_full_configuration(str(config_file))
            resolved = resolve_execution_method(configurable, source=str(config_file))

        # The loader passes the legacy value through untouched — no rewriting of
        # the operator's file — and the resolver is what every reader sees.
        assert configurable["execution"]["execution_method"] == "container"
        assert resolved == "subprocess"

        deprecations = self._deprecations(caplog)
        assert len(deprecations) == 1, (
            f"expected exactly one deprecation warning, got {len(deprecations)}: "
            f"{[r.getMessage() for r in deprecations]}"
        )
        message = deprecations[0].getMessage()
        assert str(config_file) in message
        assert "subprocess" in message

    def test_repeated_reads_of_container_configs_warn_exactly_once(self, tmp_path, caplog):
        """Warn-once is per process, not per read — the executor resolves per tool
        call, so a per-read warning would bury the operator in duplicates.
        """
        from osprey.utils.config import get_full_configuration

        first = self._write_config(tmp_path, "container", name="first.yml")
        second = self._write_config(tmp_path, "CONTAINER", name="second.yml")

        with caplog.at_level(logging.WARNING, logger="CONFIG"):
            for config_file in (first, first, second):
                configurable = get_full_configuration(str(config_file))
                assert (
                    resolve_execution_method(configurable, source=str(config_file)) == "subprocess"
                )

        deprecations = self._deprecations(caplog)
        assert len(deprecations) == 1, (
            f"expected exactly one deprecation warning across 3 reads, got "
            f"{len(deprecations)}: {[r.getMessage() for r in deprecations]}"
        )
        # The one warning names the config that first tripped it.
        assert str(first) in deprecations[0].getMessage()

    def test_subprocess_config_file_loads_without_warning(self, tmp_path, caplog):
        from osprey.utils.config import get_full_configuration

        config_file = self._write_config(tmp_path, "subprocess")

        with caplog.at_level(logging.WARNING, logger="CONFIG"):
            configurable = get_full_configuration(str(config_file))
            resolved = resolve_execution_method(configurable, source=str(config_file))

        assert resolved == "subprocess"
        assert self._deprecations(caplog) == []


class TestExecutionDefaults:
    """The built-in execution defaults advertise the honest backend."""

    def test_default_execution_config_uses_subprocess(self, tmp_path):
        from osprey.utils.config import ConfigBuilder

        config_file = tmp_path / "config.yml"
        config_file.write_text("project_root: .\n")

        builder = ConfigBuilder(str(config_file), load_env=False)
        execution = builder._get_execution_config()

        assert execution["execution_method"] == "subprocess"

    def test_defaults_round_trip_through_the_resolver(self, tmp_path):
        from osprey.utils.config import ConfigBuilder

        config_file = tmp_path / "config.yml"
        config_file.write_text("project_root: .\n")

        builder = ConfigBuilder(str(config_file), load_env=False)
        resolved = resolve_execution_method({"execution": builder._get_execution_config()})

        assert resolved == "subprocess"
