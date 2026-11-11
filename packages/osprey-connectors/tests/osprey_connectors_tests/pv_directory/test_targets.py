"""Which source answers each section on each control target.

A section answers only from the machine the session is on: the simulator's data
on ``va``, the facility's services on ``live`` and on the ``standin`` that
rehearses it, and otherwise nothing. :func:`pv_directory_targets` is that table,
and :func:`has_simulator` the one answer to whether the ``va`` half of it exists.
"""

from __future__ import annotations

import copy

import pytest

from osprey_connectors.archive_targets import APPLIANCE_TARGETS, archive_targets
from osprey_connectors.pv_directory.contract import BackendSpec, Source
from osprey_connectors.pv_directory.targets import has_simulator, pv_directory_targets
from osprey_connectors.types import (
    CONTROL_TARGETS,
    EPICS_ARCHIVER,
    TARGET_LIVE,
    TARGET_STANDIN,
    TARGET_VA,
)

KINDS = ("pv_directory", "alarm_log", "alarm_state", "put_log")
SERVED_VA = {"services": {"virtual_accelerator": {"port": 5064}}}
IN_PROCESS_VA = {"control_system": {"type": "virtual_accelerator"}}
FACILITY_WITH_SIMULATOR = {
    "control_system": {
        "type": "epics",
        "connector": {"epics": {"gateways": {}}, "virtual_accelerator": {"facility": "demo"}},
    }
}
FACILITY_ONLY = {"control_system": {"type": "epics", "connector": {"epics": {"gateways": {}}}}}

FULL_SECTION = {
    "type": "channelfinder",
    "settings": {"url": "https://directory.example/ChannelFinder", "timeout_s": 4},
    "description_property": "recordDesc",
    "display": {"exact_properties": ["iocName", "hostName"], "filter_cap": 500},
    "caveats": ["directory caveat"],
    "alarm_log": {
        "type": "phoebus_alarm_logger",
        "settings": {"url": "https://log.example"},
        "caveats": ["log caveat"],
    },
    "alarm_state": {
        "type": "phoebus_alarm_server",
        "settings": {"bootstrap_servers": "kafka.example:9092", "topic": "Accelerator"},
        "replay_file": "alarms/rehearsal.jsonl",
        "caveats": ["state caveat"],
    },
    "put_log": {
        "type": "caputlog",
        "settings": {"url": "https://puts.example"},
        "caveats": ["put caveat"],
    },
}


def _config(section, *, base=SERVED_VA):
    return {**copy.deepcopy(base), "pv_directory": copy.deepcopy(section)}


def _table(section, *, base=SERVED_VA, ariel_configured=True):
    return pv_directory_targets(_config(section, base=base), ariel_configured=ariel_configured)


class TestHasSimulator:
    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            pytest.param(SERVED_VA, True, id="served"),
            pytest.param(IN_PROCESS_VA, True, id="in-process"),
            pytest.param(FACILITY_WITH_SIMULATOR, True, id="facility-type-with-simulator-block"),
            pytest.param(FACILITY_ONLY, False, id="facility-only"),
            pytest.param({}, True, id="no-control-system-section-defaults-to-the-simulator"),
            pytest.param(
                {**FACILITY_ONLY, "services": {"virtual_accelerator": {"port": True}}},
                False,
                id="port-names-none",
            ),
        ],
    )
    def test_answer(self, config, expected):
        assert has_simulator(config) is expected

    def test_a_malformed_config_has_no_simulator(self):
        assert has_simulator(None) is False  # type: ignore[arg-type]


class TestShape:
    def test_every_row_names_every_target(self):
        table = _table(FULL_SECTION)
        assert set(table) == {*KINDS, "logbook"}
        for row in table.values():
            assert list(row) == list(CONTROL_TARGETS)

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({}, id="absent"),
            pytest.param({"pv_directory": None}, id="null"),
            pytest.param({"pv_directory": "channelfinder"}, id="not-a-mapping"),
        ],
    )
    def test_no_section_serves_nothing(self, config):
        table = pv_directory_targets({**SERVED_VA, **config}, ariel_configured=True)
        for row in table.values():
            assert set(row.values()) == {None}

    def test_the_config_is_not_touched(self):
        config = _config(FULL_SECTION)
        before = copy.deepcopy(config)
        pv_directory_targets(config, ariel_configured=True)
        assert config == before


class TestDirectorySources:
    def test_va_only_section_with_a_simulator(self):
        row = _table({})["pv_directory"]
        assert row[TARGET_LIVE] is None
        assert row[TARGET_STANDIN] is None
        assert row[TARGET_VA] == BackendSpec(
            kind="pv_directory",
            type="simulator_view",
            target=TARGET_VA,
            settings={},
            rehearsal=False,
        )

    def test_section_with_no_type_and_no_simulator_serves_nothing(self):
        table = _table({}, base=FACILITY_ONLY)
        assert set(table["pv_directory"].values()) == {None}

    def test_facility_type_serves_live_and_standin(self):
        row = _table(FULL_SECTION)["pv_directory"]
        for target, rehearsal in ((TARGET_LIVE, False), (TARGET_STANDIN, True)):
            assert row[target].type == "channelfinder"
            assert row[target].target == target
            assert row[target].rehearsal is rehearsal

    def test_facility_type_with_no_simulator_leaves_va_empty(self):
        assert _table(FULL_SECTION, base=FACILITY_ONLY)["pv_directory"][TARGET_VA] is None

    def test_simulator_spec_carries_no_facility_setting(self):
        settings = _table(FULL_SECTION)["pv_directory"][TARGET_VA].settings
        assert "url" not in settings
        assert "timeout_s" not in settings

    @pytest.mark.parametrize("base", [SERVED_VA, IN_PROCESS_VA, FACILITY_WITH_SIMULATOR])
    def test_no_target_is_served_by_both_simulator_and_facility_data(self, base):
        table = _table(FULL_SECTION, base=base)
        for kind in KINDS:
            for target, spec in table[kind].items():
                if spec is None:
                    continue
                simulated = spec.type in {"simulator_view", "alarm_replay"}
                assert simulated == (target == TARGET_VA), (kind, target, spec.type)


class TestAlarmStateSources:
    def test_a_replay_serves_va_only(self):
        row = _table({"alarm_state": {"replay_file": "alarms/run.jsonl"}})["alarm_state"]
        assert row[TARGET_LIVE] is None
        assert row[TARGET_STANDIN] is None
        assert row[TARGET_VA] == BackendSpec(
            kind="alarm_state",
            type="alarm_replay",
            target=TARGET_VA,
            settings={"replay_file": "alarms/run.jsonl"},
            rehearsal=False,
        )

    def test_a_replay_serves_va_with_no_simulator_built(self):
        row = _table({"alarm_state": {"replay_file": "a.jsonl"}}, base=FACILITY_ONLY)["alarm_state"]
        assert row[TARGET_VA].type == "alarm_replay"

    def test_demo_replay_passes_verbatim(self):
        row = _table({"alarm_state": {"replay_file": "demo:example-alarms"}})["alarm_state"]
        assert row[TARGET_VA].settings["replay_file"] == "demo:example-alarms"

    def test_a_facility_alarm_source_never_serves_va(self):
        block = {k: v for k, v in FULL_SECTION["alarm_state"].items() if k != "replay_file"}
        row = _table({"alarm_state": block})["alarm_state"]
        assert row[TARGET_VA] is None
        assert row[TARGET_LIVE].type == "phoebus_alarm_server"
        assert row[TARGET_STANDIN].type == "phoebus_alarm_server"
        assert row[TARGET_STANDIN].rehearsal is True

    def test_type_and_replay_serve_different_targets(self):
        row = _table(FULL_SECTION)["alarm_state"]
        assert row[TARGET_VA].type == "alarm_replay"
        assert "bootstrap_servers" not in row[TARGET_VA].settings
        assert row[TARGET_LIVE].type == "phoebus_alarm_server"
        assert row[TARGET_LIVE].settings["bootstrap_servers"] == "kafka.example:9092"
        assert "replay_file" not in row[TARGET_LIVE].settings

    @pytest.mark.parametrize("replay_file", [None, "", "   ", 3])
    def test_a_replay_file_naming_nothing_serves_nothing(self, replay_file):
        row = _table({"alarm_state": {"replay_file": replay_file}})["alarm_state"]
        assert set(row.values()) == {None}


class TestLogsServeTheFacilityOnly:
    @pytest.mark.parametrize("kind", ["alarm_log", "put_log"])
    def test_live_and_standin(self, kind):
        row = _table(FULL_SECTION)[kind]
        assert row[TARGET_VA] is None
        assert row[TARGET_LIVE].kind == kind
        assert row[TARGET_LIVE].rehearsal is False
        assert row[TARGET_STANDIN].rehearsal is True
        assert row[TARGET_LIVE].settings["url"] == FULL_SECTION[kind]["settings"]["url"]

    @pytest.mark.parametrize("kind", ["alarm_log", "put_log"])
    @pytest.mark.parametrize(
        "block",
        [
            pytest.param(None, id="null"),
            pytest.param({}, id="empty"),
            pytest.param({"settings": {"url": "https://x.example"}}, id="no-type"),
            pytest.param({"type": ""}, id="blank-type"),
            pytest.param({"type": 3}, id="type-not-a-string"),
            pytest.param("caputlog", id="not-a-mapping"),
        ],
    )
    def test_a_block_naming_no_type_serves_nothing(self, kind, block):
        assert set(_table({kind: block})[kind].values()) == {None}

    def test_settings_not_a_mapping_read_as_none(self):
        row = _table({"put_log": {"type": "caputlog", "settings": "nope"}})["put_log"]
        assert row[TARGET_LIVE].settings == {}


class TestLogbook:
    def test_serves_live_and_standin_when_ariel_is_configured(self):
        row = _table({})["logbook"]
        assert row[TARGET_VA] is None
        assert row[TARGET_LIVE] == Source(
            target=TARGET_LIVE, service="ariel", label="ariel", rehearsal=False
        )
        assert row[TARGET_STANDIN] == Source(
            target=TARGET_STANDIN, service="ariel", label="ariel", rehearsal=True
        )

    def test_absent_without_ariel(self):
        assert set(_table(FULL_SECTION, ariel_configured=False)["logbook"].values()) == {None}

    def test_holds_no_spec(self):
        for source in _table(FULL_SECTION)["logbook"].values():
            assert not isinstance(source, BackendSpec)


class TestFold:
    def test_directory_spec_folds_description_exact_properties_and_caveats(self):
        settings = _table(FULL_SECTION)["pv_directory"][TARGET_LIVE].settings
        assert settings == {
            "url": "https://directory.example/ChannelFinder",
            "timeout_s": 4,
            "description_property": "recordDesc",
            "exact_properties": ["iocName", "hostName"],
            "caveats": ["directory caveat"],
        }

    def test_simulator_spec_folds_the_same_keys(self):
        settings = _table(FULL_SECTION)["pv_directory"][TARGET_VA].settings
        assert settings == {
            "description_property": "recordDesc",
            "exact_properties": ["iocName", "hostName"],
            "caveats": ["directory caveat"],
        }

    @pytest.mark.parametrize("kind", ["alarm_log", "alarm_state", "put_log"])
    def test_each_sub_block_folds_its_own_caveats_only(self, kind):
        settings = _table(FULL_SECTION)[kind][TARGET_LIVE].settings
        assert settings["caveats"] == FULL_SECTION[kind]["caveats"]
        assert "description_property" not in settings
        assert "exact_properties" not in settings

    def test_replay_spec_folds_the_sub_blocks_caveats(self):
        settings = _table(FULL_SECTION)["alarm_state"][TARGET_VA].settings
        assert settings == {
            "replay_file": "alarms/rehearsal.jsonl",
            "caveats": ["state caveat"],
        }

    def test_a_key_the_section_does_not_set_is_not_folded(self):
        settings = _table({"type": "channelfinder"})["pv_directory"][TARGET_LIVE].settings
        assert settings == {}

    @pytest.mark.parametrize("value", ["", "  ", 7])
    def test_a_blank_description_property_is_not_folded(self, value):
        table = _table({"type": "channelfinder", "description_property": value})
        assert "description_property" not in table["pv_directory"][TARGET_LIVE].settings
        assert "description_property" not in table["pv_directory"][TARGET_VA].settings

    def test_display_not_a_mapping_folds_nothing(self):
        section = {"type": "channelfinder", "display": "wide"}
        assert _table(section)["pv_directory"][TARGET_LIVE].settings == {}

    def test_folded_values_are_copies_of_the_config(self):
        config = _config(FULL_SECTION)
        table = pv_directory_targets(config, ariel_configured=False)
        table["pv_directory"][TARGET_LIVE].settings["exact_properties"].append("mutated")
        table["put_log"][TARGET_LIVE].settings["caveats"].append("mutated")
        assert config["pv_directory"]["display"]["exact_properties"] == ["iocName", "hostName"]
        assert config["pv_directory"]["put_log"]["caveats"] == ["put caveat"]

    def test_specs_of_two_targets_share_no_settings(self):
        row = _table(FULL_SECTION)["put_log"]
        assert row[TARGET_LIVE].settings is not row[TARGET_STANDIN].settings


def test_facility_targets_agree_with_archive_targets():
    config = _config(FULL_SECTION)
    config["archiver"] = {"type": EPICS_ARCHIVER, "epics_archiver": {"url": "https://a.example"}}
    assert archive_targets(config) == APPLIANCE_TARGETS

    table = pv_directory_targets(config, ariel_configured=True)
    for kind in KINDS:
        facility = {
            target
            for target, spec in table[kind].items()
            if spec is not None and spec.type not in {"simulator_view", "alarm_replay"}
        }
        assert facility == APPLIANCE_TARGETS, kind
    served = {target for target, source in table["logbook"].items() if source is not None}
    assert served == APPLIANCE_TARGETS
