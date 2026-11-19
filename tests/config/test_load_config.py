"""The config loader, the config it loads, and the process default.

:func:`~osprey_connectors.config.load_config` reads one file into a
:class:`~osprey_connectors.config.LoadedConfig`; the process default
(``current_config`` and its kin) is which one an unqualified lookup reads; the
facility zone is read at load and never raises.
"""

import logging
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from osprey_connectors import config as config_module
from osprey_connectors.config import (
    UnresolvedRef,
    active_config,
    current_config,
    facility_timezone,
    install_config,
    load_config,
    localize_facility,
    locate_config,
    using_config,
)
from osprey_connectors.config_sections import ControlSystemSection, FacilityZone, Unreadable
from osprey_connectors.errors import (
    ConfigLoadError,
    ConfigNotFoundError,
    ConfigParseError,
    ConfigurationError,
)
from osprey_connectors.model import evolve
from osprey_connectors.testing import config_of


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


class TestLoadConfig:
    def test_loads_yaml(self, tmp_path):
        config_file = _write(
            tmp_path / "config.yml",
            "project_root: /test/project\n"
            "registry_path: ./my_app/registry.py\n"
            "models:\n  orchestrator:\n    provider: openai\n    model_id: gpt-4\n",
        )

        loaded = load_config(config_file)

        assert loaded.path == config_file
        assert loaded.get("project_root") == "/test/project"
        assert loaded.get("models.orchestrator.provider") == "openai"

    def test_a_directory_is_a_load_error_and_not_a_missing_file(self, tmp_path):
        """A bind-mount whose source was missing is created as a directory.

        It is refused as a load error, never as a ``FileNotFoundError``: the
        readers that degrade on a missing config (no limits validator, writes
        off) must not read a broken mount as "no config here".
        """
        config_dir = tmp_path / "config.yml"
        config_dir.mkdir()

        with pytest.raises(ConfigLoadError, match="is a directory, not a file") as caught:
            load_config(config_dir)
        assert not isinstance(caught.value, FileNotFoundError)
        assert caught.value.path == config_dir

    def test_a_missing_file_is_a_file_not_found_error_naming_where_it_looked(self, tmp_path):
        missing = tmp_path / "does_not_exist" / "config.yml"

        with pytest.raises(ConfigNotFoundError, match="Config file not found") as caught:
            load_config(missing)
        assert isinstance(caught.value, FileNotFoundError)
        assert isinstance(caught.value, ConfigurationError)
        assert caught.value.searched == (missing,)
        assert "OSPREY_CONFIG" in str(caught.value)
        assert "CONFIG_FILE" not in str(caught.value)

    def test_no_config_named_and_none_found_lists_both_places(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ConfigNotFoundError, match="No config.yml found") as caught:
            load_config()
        assert caught.value.searched == (tmp_path / "build" / "config.yml", tmp_path / "config.yml")
        assert "OSPREY_CONFIG" in str(caught.value)

    def test_invalid_yaml_is_a_parse_error_with_its_line(self, tmp_path):
        config_file = _write(
            tmp_path / "config.yml",
            "\ninvalid: yaml: syntax:\n  - this is broken\n    - nested incorrectly\n",
        )

        with pytest.raises(ConfigParseError) as caught:
            load_config(config_file)
        assert caught.value.path == config_file
        assert caught.value.line == 2

    @pytest.mark.parametrize(
        ("body", "message", "key"),
        [
            ("- a\n- b\n", "Configuration file must contain a dictionary/mapping", None),
            ("just a string\n", "Configuration file must contain a dictionary/mapping", None),
            ("42\n", "Configuration file must contain a dictionary/mapping", None),
            (
                "execution: subprocess\n",
                "'execution' in .* must be a mapping, got str",
                "execution",
            ),
            (
                "api:\n  providers: [a]\n",
                "'api.providers' in .* must be a mapping",
                "api.providers",
            ),
        ],
        ids=["list", "string", "number", "execution", "providers"],
    )
    def test_a_file_of_the_wrong_shape_is_refused(self, tmp_path, body, message, key):
        config_file = _write(tmp_path / "config.yml", body)

        with pytest.raises(ConfigParseError, match=message) as caught:
            load_config(config_file)
        assert caught.value.key == key
        assert isinstance(caught.value, ValueError)

    def test_an_empty_file_loads_as_an_empty_tree(self, tmp_path):
        assert load_config(_write(tmp_path / "config.yml", "")).raw_tree() == {}

    def test_an_empty_section_reads_as_no_settings(self, tmp_path):
        loaded = load_config(_write(tmp_path / "config.yml", "models:\nservices:\n"))

        assert loaded.section("models") == {}
        assert loaded.section("services") == {}

    def test_placeholders_resolve_against_the_environment_given(self, tmp_path):
        config_file = _write(
            tmp_path / "config.yml",
            "project_root: ${ROOT}\napi:\n  key: ${KEY}\n  url: https://x\n",
        )

        loaded = load_config(config_file, environ={"ROOT": "/home/p", "KEY": "k-1"})

        assert loaded.get("project_root") == "/home/p"
        assert loaded.get("api.key") == "k-1"
        assert loaded.get("api.url") == "https://x"

    def test_the_tree_as_written_keeps_its_placeholders(self, tmp_path):
        """A deployment writing the config out again must not bake a secret into it."""
        config_file = _write(tmp_path / "config.yml", "api:\n  key: ${SECRET}\n")

        loaded = load_config(config_file, environ={"SECRET": "actual-secret"})

        assert loaded.get("api.key") == "actual-secret"
        assert loaded.unexpanded_tree()["api"]["key"] == "${SECRET}"

    def test_an_unset_variable_stays_verbatim_and_is_recorded(self, tmp_path):
        """A lone ``${VAR}`` is data, never absent; the reference is named for diagnostics."""
        config_file = _write(
            tmp_path / "config.yml",
            "control_system:\n  limits_checking:\n    enabled: ${LIMITS_ON}\n"
            "items:\n  - ${FIRST}\n"
            "claude_code:\n  servers:\n    a:\n      env:\n        TOKEN: ${KEPT_FOR_LAUNCH}\n",
        )

        loaded = load_config(config_file, environ={})

        assert loaded.get("control_system.limits_checking.enabled") == "${LIMITS_ON}"
        assert loaded.unresolved == (
            UnresolvedRef(key="control_system.limits_checking.enabled", var="LIMITS_ON"),
            UnresolvedRef(key="items.0", var="FIRST"),
        )

    def test_the_load_reads_no_env_chain_and_writes_no_environment(self, tmp_path, monkeypatch):
        _write(tmp_path / ".env", "OSPREY_LOAD_PURITY_PROBE=from-the-file\n")
        config_file = _write(tmp_path / "config.yml", "value: ${OSPREY_LOAD_PURITY_PROBE}\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("OSPREY_LOAD_PURITY_PROBE", raising=False)

        loaded = load_config(config_file)

        assert loaded.get("value") == "${OSPREY_LOAD_PURITY_PROBE}"
        assert "OSPREY_LOAD_PURITY_PROBE" not in os.environ

    def test_the_repo_root_of_a_render_is_the_repo(self, tmp_path):
        loaded = load_config(_write(tmp_path / "build" / "config.yml", "a: 1\n"))

        assert loaded.repo_root == tmp_path

    def test_an_existing_project_root_names_the_repo_root(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        loaded = load_config(_write(tmp_path / "flat" / "config.yml", f"project_root: {repo}\n"))

        assert loaded.repo_root == repo

    def test_a_project_root_that_is_not_here_does_not(self, tmp_path):
        """A container reads a config staged on the host, whose root names nothing here."""
        config_file = _write(tmp_path / "config.yml", "project_root: /host/only/path\n")

        assert load_config(config_file).repo_root == tmp_path

    def test_a_renamed_timeout_key_is_refused_at_load(self, tmp_path):
        config_file = _write(
            tmp_path / "config.yml",
            "control_system:\n  connector:\n    epics:\n      timeout: 5\n",
        )

        with pytest.raises(ConfigurationError, match="timeout_s") as caught:
            load_config(config_file)
        assert caught.value.key == "control_system.connector.epics.timeout"


class TestLoadedConfig:
    def test_get_walks_dotted_keys_and_answers_the_default_when_absent(self):
        loaded = config_of({"a": {"b": {"c": 3}}, "n": None, "lst": [1]})

        assert loaded.get("a.b.c") == 3
        assert loaded.get("a.x", "d") == "d"
        assert loaded.get("lst.0", "d") == "d"
        assert loaded.get("n", "d") is None

    def test_get_hands_out_a_copy(self):
        loaded = config_of({"control_system": {"writes": "allowed"}})

        handed = loaded.get("control_system")
        assert isinstance(handed, dict)
        handed["writes"] = "never"
        loaded.raw_tree()["control_system"]["writes"] = "never"
        loaded.section("control_system")["writes"] = "never"

        assert loaded.get("control_system.writes") == "allowed"

    def test_an_empty_key_is_refused(self):
        with pytest.raises(ConfigurationError):
            config_of({}).get("")

    def test_a_section_that_is_not_a_mapping_is_refused_by_key(self):
        loaded = config_of({"web": "on", "empty": None})

        assert loaded.section("empty") == {}
        assert loaded.section("absent") == {}
        with pytest.raises(ConfigurationError, match="'web'") as caught:
            loaded.section("web")
        assert caught.value.key == "web"

    @pytest.mark.parametrize(
        ("value", "expected"), [(True, True), ("false", False), ("on", True), (None, False)]
    )
    def test_a_flag_reads_what_it_spells(self, value, expected):
        assert config_of({"switch": value}).flag("switch", default=False) is expected

    def test_an_absent_flag_is_its_default(self):
        assert config_of({}).flag("switch", default=True) is True

    def test_a_flag_that_spells_neither_is_refused_by_key(self):
        with pytest.raises(ConfigurationError, match="switch") as caught:
            config_of({"switch": "${UNSET}"}).flag("switch", default=False)
        assert caught.value.key == "switch"


class TestFacilityZone:
    @pytest.mark.parametrize(
        ("tree", "name", "source"),
        [
            ({"system": {"timezone": "America/Los_Angeles"}}, "America/Los_Angeles", "configured"),
            ({}, "UTC", "default"),
            ({"system": "not a mapping"}, "UTC", "default"),
            ({"system": {"timezone": "Not/ARealZone"}}, "UTC", "fallback"),
            ({"system": {"timezone": 5}}, "UTC", "fallback"),
            ({"system": {"timezone": None}}, "UTC", "fallback"),
            ({"system": {"timezone": "${UNSET_ZONE}"}}, "UTC", "fallback"),
        ],
        ids=["configured", "absent", "no-system", "typo", "number", "null", "unset-var"],
    )
    def test_the_zone_is_read_at_load_and_never_raises(self, tree, name, source):
        zone = config_of(tree, environ={}).system.timezone

        assert (zone.name, zone.source) == (name, source)
        assert (zone.problem is None) == (source != "fallback")
        assert zone.zone == ZoneInfo(name)

    def test_an_unset_variable_says_so(self):
        zone = config_of({"system": {"timezone": "${UNSET_ZONE}"}}, environ={}).system.timezone

        assert zone == FacilityZone(name="UTC", source="fallback", problem=zone.problem)
        assert "unset environment variable" in (zone.problem or "")


class TestControlSystemSection:
    """The ``control_system`` section is read once, at load, and never refused there."""

    def test_the_section_is_read_at_load(self):
        section = {"type": "epics", "writes": "asked", "connector": {"epics": {}}}
        loaded = config_of({"control_system": section})

        assert loaded.control_system == ControlSystemSection.read(section)
        assert loaded.control_system.writes == "asked"

    @pytest.mark.parametrize(
        "value", [None, "not a mapping", ["a", "list"], 5], ids=["null", "str", "list", "int"]
    )
    def test_a_section_no_reader_can_use_loads_and_arms_nothing(self, value):
        loaded = config_of({"control_system": value})

        assert loaded.control_system == ControlSystemSection()
        assert loaded.control_system.writes == "never"

    def test_a_written_but_unreadable_leaf_is_never_read_as_absent(self):
        loaded = config_of(
            {"control_system": {"limits_checking": {"enabled": "${UNSET_LIMITS}"}}}, environ={}
        )

        limits = loaded.control_system.limits_checking
        assert limits is not None and not isinstance(limits, Unreadable)
        assert isinstance(limits.enabled, Unreadable)

    def test_the_raw_tree_stays_plain_dicts(self):
        loaded = config_of({"control_system": {"connector": {"epics": {"gateways": {}}}}})

        assert type(loaded.get("control_system")) is dict
        assert type(loaded.raw_tree()["control_system"]["connector"]) is dict


class TestEvolveIsRefused:
    def test_evolve_names_why_instead_of_dropping_the_tree(self):
        loaded = config_of({"a": 1})

        with pytest.raises(TypeError, match="built by load_config"):
            evolve(loaded, path=Path("other.yml"))

    def test_model_validate_is_refused_too(self):
        loaded = config_of({"a": 1})

        with pytest.raises(TypeError, match="built by load_config"):
            type(loaded).model_validate(dict(loaded))


class TestProcessDefault:
    def test_the_lazy_default_is_the_located_file(self, tmp_path, monkeypatch):
        config_file = _write(tmp_path / "build" / "config.yml", "marker: build\n")
        _write(tmp_path / "config.yml", "marker: flat\n")
        monkeypatch.chdir(tmp_path)

        assert locate_config() == config_file
        assert current_config().get("marker") == "build"
        assert active_config() is current_config()

    def test_osprey_config_names_the_file_and_config_file_does_not(self, tmp_path, monkeypatch):
        named = _write(tmp_path / "named.yml", "marker: named\n")
        other = _write(tmp_path / "other.yml", "marker: other\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CONFIG_FILE", str(other))
        monkeypatch.setenv("OSPREY_CONFIG", str(named))

        assert current_config().get("marker") == "named"

    def test_a_re_rendered_file_is_read_again_without_a_reset(self, tmp_path, monkeypatch):
        config_file = _write(tmp_path / "config.yml", "marker: first\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(config_file))
        assert current_config().get("marker") == "first"

        config_file.write_text("marker: second\n", encoding="utf-8")
        stat = config_file.stat()
        os.utime(config_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

        assert current_config().get("marker") == "second"

    def test_nothing_to_find_is_a_missing_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        assert active_config() is None
        with pytest.raises(ConfigNotFoundError):
            current_config()
        with pytest.raises(FileNotFoundError):
            current_config().get("anything")

    def test_an_installed_config_answers_ahead_of_the_located_one(self, tmp_path, monkeypatch):
        _write(tmp_path / "config.yml", "marker: located\n")
        monkeypatch.chdir(tmp_path)
        installed = config_of({"marker": "installed"})

        assert install_config(installed) is None
        assert current_config() is installed
        assert install_config(None) is installed
        assert current_config().get("marker") == "located"

    def test_a_scoped_config_answers_inside_its_block_only(self):
        install_config(config_of({"marker": "process"}))
        inner = config_of({"marker": "scoped"})

        with using_config(inner) as scoped:
            assert scoped is inner
            assert current_config().get("marker") == "scoped"
            with using_config(config_of({"marker": "nested"})):
                assert current_config().get("marker") == "nested"
            assert active_config() is inner
        assert current_config().get("marker") == "process"

    def test_a_scope_named_by_path_loads_it(self, tmp_path):
        config_file = _write(tmp_path / "config.yml", "marker: from-path\n")

        with using_config(config_file) as scoped:
            assert scoped.path == config_file
            assert current_config().get("marker") == "from-path"

    def test_an_explicit_path_is_read_without_becoming_the_default(self, tmp_path):
        config_file = _write(tmp_path / "config.yml", "marker: explicit\n")

        assert load_config(config_file).get("marker") == "explicit"
        assert active_config() is None

    def test_the_lazy_default_applies_the_env_chain_before_it_loads(self, tmp_path, monkeypatch):
        _write(tmp_path / ".env", "OSPREY_LAZY_CHAIN_PROBE=from-the-chain\n")
        _write(tmp_path / "config.yml", "value: ${OSPREY_LAZY_CHAIN_PROBE}\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("OSPREY_LAZY_CHAIN_PROBE", raising=False)

        assert current_config().get("value") == "from-the-chain"


class TestFacilityTimezone:
    """``facility_timezone`` never raises: it is the safe default for every synthesis
    primitive and timestamp render site."""

    def test_utc_when_no_config_can_be_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        assert facility_timezone() == ZoneInfo("UTC")

    def test_utc_when_the_configured_zone_is_invalid(self, tmp_path, monkeypatch):
        config_file = _write(tmp_path / "config.yml", "system:\n  timezone: Not/ARealZone\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(config_file))

        assert facility_timezone() == ZoneInfo("UTC")

    def test_the_configured_zone_when_valid(self, tmp_path, monkeypatch):
        config_file = _write(tmp_path / "config.yml", "system:\n  timezone: America/Los_Angeles\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(config_file))

        assert facility_timezone() == ZoneInfo("America/Los_Angeles")

    def test_the_zone_of_the_config_given(self):
        loaded = config_of({"system": {"timezone": "Asia/Tokyo"}})

        assert facility_timezone(loaded) == ZoneInfo("Asia/Tokyo")

    def test_no_config_is_reported_once_not_once_per_call(self, tmp_path, monkeypatch, caplog):
        """Synthesis calls this per channel per chunk; once is not zero, though."""
        monkeypatch.chdir(tmp_path)

        with caplog.at_level(logging.WARNING, logger="CONFIG"):
            for _ in range(5):
                facility_timezone()

        fallback = [r for r in caplog.records if "Falling back to UTC" in r.message]
        assert len(fallback) == 1

    def test_a_bad_zone_is_reported_once_per_load(self, caplog):
        loaded = config_of({"system": {"timezone": "Not/ARealZone"}})

        with caplog.at_level(logging.WARNING, logger="CONFIG"):
            for _ in range(5):
                facility_timezone(loaded)
            facility_timezone(config_of({"system": {"timezone": "Not/ARealZone"}}))

        fallback = [r for r in caplog.records if "Falling back to UTC" in r.message]
        assert len(fallback) == 2


class TestTimezoneDriftWarning:
    """A container ``$TZ`` that disagrees with an explicit ``system.timezone`` is
    reported once per load; the zone read stays ``system.timezone``."""

    def test_warns_once_on_drift(self, monkeypatch, caplog):
        loaded = config_of({"system": {"timezone": "America/Los_Angeles"}})
        monkeypatch.setenv("TZ", "UTC")

        with caplog.at_level(logging.WARNING, logger="CONFIG"):
            facility_timezone(loaded)
            facility_timezone(loaded)

        drift = [r for r in caplog.records if "differs from the facility" in r.message]
        assert len(drift) == 1
        assert facility_timezone(loaded) == ZoneInfo("America/Los_Angeles")

    def test_no_warning_when_tz_matches(self, monkeypatch, caplog):
        loaded = config_of({"system": {"timezone": "America/Los_Angeles"}})
        monkeypatch.setenv("TZ", "America/Los_Angeles")

        with caplog.at_level(logging.WARNING, logger="CONFIG"):
            facility_timezone(loaded)

        assert not [r for r in caplog.records if "differs from the facility" in r.message]

    def test_no_warning_when_the_zone_is_not_configured(self, monkeypatch, caplog):
        """The implicit UTC must stay quiet, or every CI run with $TZ set would warn."""
        monkeypatch.setenv("TZ", "America/New_York")

        with caplog.at_level(logging.WARNING, logger="CONFIG"):
            facility_timezone(config_of({}))

        assert not [r for r in caplog.records if "differs from the facility" in r.message]


class TestLocalizeFacility:
    def test_a_naive_time_is_stamped_with_the_zone_named(self):
        from datetime import datetime

        tokyo = ZoneInfo("Asia/Tokyo")

        assert localize_facility(datetime(2026, 6, 1, 9, 0), tokyo).utcoffset().total_seconds() == (
            9 * 3600
        )

    def test_an_aware_time_and_none_pass_through(self):
        from datetime import UTC, datetime

        aware = datetime(2026, 6, 1, 9, 0, tzinfo=UTC)

        assert localize_facility(aware, ZoneInfo("Asia/Tokyo")) is aware
        assert localize_facility(None, ZoneInfo("Asia/Tokyo")) is None


class TestToFacilityIso:
    """``to_facility_iso`` is the one timestamp-egress transform for ARIEL's MCP and web
    output, so both render the same facility-local string."""

    def test_none_passes_through(self):
        from osprey.utils.facility import to_facility_iso

        assert to_facility_iso(None) is None

    def test_non_datetime_degrades_to_str(self):
        from osprey.utils.facility import to_facility_iso

        assert to_facility_iso("2026-06-01T00:00:00+00:00") == "2026-06-01T00:00:00+00:00"

    def test_naive_datetime_is_stamped_facility_local_not_box_local(self, monkeypatch):
        from datetime import datetime

        from osprey.utils.facility import to_facility_iso

        install_config(config_of({"system": {"timezone": "Asia/Tokyo"}}))
        monkeypatch.setenv("TZ", "America/Los_Angeles")

        assert to_facility_iso(datetime(2026, 6, 1, 9, 0, 0)) == "2026-06-01T09:00:00+09:00"

    def test_aware_datetime_rendered_in_facility_zone(self):
        from datetime import UTC, datetime

        from osprey.utils.facility import to_facility_iso

        install_config(config_of({"system": {"timezone": "Asia/Tokyo"}}))

        out = to_facility_iso(datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC))
        assert out == "2026-06-01T09:00:00+09:00"


def test_the_module_keeps_no_view_of_the_config():
    """The derived ``configurable`` view and its aliases are gone: one tree, one spelling."""
    for name in (
        "ConfigBuilder",
        "get_configurable",
        "get_full_configuration",
        "_build_configurable",
    ):
        assert not hasattr(config_module, name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
