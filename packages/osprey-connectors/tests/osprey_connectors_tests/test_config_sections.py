"""The sections of a rendered config, each read once: what the parse keeps, and what it refuses where.

Reading never raises; what used to raise at the factory, the derivation or
``connect()`` still raises there, from what the parse kept. The settings a
connector type reads are refused when the factory reads the block, naming the
key.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from osprey_connectors.config_sections import (
    AgentDataSection,
    ArchiverSection,
    ConnectorBlock,
    ControlSystemSection,
    GatewayRow,
    LimitsBlock,
    LoggingSection,
    SimulationSection,
    Unreadable,
)
from osprey_connectors.connector_types import BUILTIN_CONNECTORS, MOCK_ARCHIVER
from osprey_connectors.control_system import ConnectorContext, ControlSystemSettings
from osprey_connectors.control_system._epics_connector import EPICSConnector, EpicsSettings
from osprey_connectors.errors import ConfigurationError
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.gateways import derive_endpoints
from osprey_connectors.simulation import resolve_fast_tick_s, resolve_tick_s
from osprey_connectors.workspace import DEFAULT_AGENT_DATA_BASE_DIR, agent_data_base_dir

GATEWAY = {"address": "cagw.example", "port": 5064}


def _epics(block: dict[str, Any]) -> ControlSystemSection:
    return ControlSystemSection.read({"type": "epics", "connector": {"epics": block}})


# -- the control_system section ------------------------------------------------


class TestTheSectionIsReadOnce:
    def test_a_non_mapping_reads_as_an_empty_section(self):
        for value in (None, [], "epics", 5, ("type",)):
            assert ControlSystemSection.read(value) == ControlSystemSection()

    def test_a_section_already_read_is_returned_as_it_is(self):
        section = ControlSystemSection.read({"type": "epics"})
        assert ControlSystemSection.read(section) is section

    def test_the_section_is_a_copy_of_what_it_was_read_from(self):
        raw = {"type": "epics", "connector": {"epics": {"gateways": {"read_only": dict(GATEWAY)}}}}
        section = ControlSystemSection.read(raw)
        raw["connector"]["epics"]["gateways"]["read_only"]["address"] = "elsewhere"
        raw["writes"] = "allowed"
        assert section.writes == "never"
        assert (
            section.raw["connector"]["epics"]["gateways"]["read_only"]["address"] == "cagw.example"
        )

    def test_a_mapping_that_cannot_be_read_arms_nothing(self):
        class Raising(dict):
            def get(self, *_args: object, **_kwargs: object) -> object:
                raise RuntimeError("boom")

        section = ControlSystemSection.read(Raising(writes="allowed"))
        assert section.writes == "never"
        assert isinstance(section.login_tiers, Unreadable)
        assert isinstance(section.limits_checking, Unreadable)
        assert isinstance(section.connectors, Unreadable)

    def test_only_the_exact_tier_words_arm(self):
        for written, tier in (("asked", "asked"), ("allowed", "allowed"), ("ALLOWED", "never")):
            assert ControlSystemSection.read({"writes": written}).writes == tier
        assert ControlSystemSection.read({"writes": True}).writes == "never"

    def test_one_unreadable_login_tier_makes_the_whole_map_unreadable(self):
        section = ControlSystemSection.read({"login_tiers": {"alice": "allowed", "bob": "on"}})
        assert isinstance(section.login_tiers, Unreadable)
        assert section.login_tiers.raw == {"alice": "allowed", "bob": "on"}

    def test_a_login_tier_key_is_any_non_empty_string_as_the_rung_reads_it(self):
        section = ControlSystemSection.read({"login_tiers": {" alice": "asked"}})
        assert section.login_tiers == {" alice": "asked"}


class TestLimitsLeaves:
    def test_absent_written_and_unreadable_are_three_answers(self):
        block = LimitsBlock.read({"enabled": True}, "control_system.limits_checking")
        assert block.enabled is True
        assert block.mode is None
        unreadable = LimitsBlock.read({"enabled": "true", "mode": None}, "k")
        assert unreadable.enabled == Unreadable(key="k.enabled", raw="true")
        # A YAML null is written: unreadable, never absent.
        assert unreadable.mode == Unreadable(key="k.mode", raw=None)

    def test_a_lone_placeholder_stays_data(self):
        block = LimitsBlock.read({"enabled": "${LIMITS_ON}"}, "k")
        assert block.enabled == Unreadable(key="k.enabled", raw="${LIMITS_ON}")

    def test_a_limits_block_that_is_not_a_mapping_is_unreadable(self):
        section = ControlSystemSection.read({"limits_checking": "true"})
        assert section.limits_checking == Unreadable(
            key="control_system.limits_checking", raw="true"
        )


class TestConnectorBlocks:
    def test_a_block_that_is_not_a_mapping_is_unreadable_and_keeps_its_key(self):
        section = ControlSystemSection.read({"connector": {"epics": "x", 3: {}}})
        assert section.connectors == {
            "epics": Unreadable(key="control_system.connector.epics", raw="x")
        }

    def test_a_block_names_the_keys_that_are_not_strings(self):
        block = _epics({1: "one", "timeout_s": 2}).block("epics")
        assert isinstance(block, ConnectorBlock)
        assert block.settings == {"timeout_s": 2}
        assert block.unreadable_keys == ("1",)
        assert block.written

    @pytest.mark.parametrize(
        ("table", "declared"),
        [
            ([], "none_declared"),
            (None, "missing"),
            ({}, "missing"),
            ("x", "missing"),
            ({"read_only": GATEWAY}, "rows"),
        ],
    )
    def test_how_a_block_spells_its_gateways(self, table, declared):
        block = _epics({"gateways": table}).block("epics")
        assert isinstance(block, ConnectorBlock)
        assert block.declared_gateways == declared


class TestGatewayRows:
    def test_a_numeric_string_port_is_its_number(self):
        assert GatewayRow.read({"address": "gw", "port": "5064"}, "k") == GatewayRow(
            address="gw", port=5064
        )

    def test_the_address_is_validated_never_stripped(self):
        assert GatewayRow.read({"address": " gw "}, "k") == GatewayRow(address=" gw ")

    @pytest.mark.parametrize(
        "entry",
        [
            {"port": 5064},
            {"address": "", "port": 5064},
            {"address": "  "},
            {"address": 7},
            {"address": "gw", "port": True},
            {"address": "gw", "port": "five"},
            {"address": "gw", "use_name_server": "maybe"},
            "gw:5064",
        ],
    )
    def test_an_entry_no_connection_is_made_through_is_unreadable(self, entry):
        assert isinstance(GatewayRow.read(entry, "k"), Unreadable)

    @pytest.mark.parametrize("entry", [None, {}, "", 0])
    def test_an_empty_entry_configures_nothing(self, entry):
        assert GatewayRow.read(entry, "k") is None


# -- what is refused, and where ------------------------------------------------


class TestRefusedAtDerivation:
    """Behaviour change: a gateway no connection is made through is refused when derived.

    Before, it derived a row (``host`` and ``port`` as written) that
    verification refused later.
    """

    @pytest.mark.parametrize(
        ("gateway", "key"),
        [
            ({"address": "", "port": 5064}, "control_system.connector.epics.gateways.read_only"),
            ({"address": "gw", "port": "x"}, "control_system.connector.epics.gateways.read_only"),
            (
                {"address": "gw", "use_name_server": "maybe"},
                "control_system.connector.epics.gateways.read_only",
            ),
        ],
    )
    def test_a_bad_gateway_row_names_its_key(self, gateway, key):
        config = {
            "control_system": {
                "type": "epics",
                "connector": {"epics": {"gateways": {"read_only": gateway}}},
            }
        }
        with pytest.raises(ConfigurationError) as raised:
            derive_endpoints(config, "live", writes_enabled=False, readonly_run=False)
        assert raised.value.key == key

    def test_only_the_selected_roles_row_is_refused_and_it_names_that_role(self):
        """Writes armed select ``write_access``: its bad row is refused, naming it."""
        config = {
            "control_system": {
                "type": "epics",
                "connector": {
                    "epics": {
                        "gateways": {
                            "read_only": {"address": "", "port": 5064},
                            "write_access": {"address": "gw", "port": "x"},
                        }
                    }
                },
            }
        }
        with pytest.raises(ConfigurationError) as raised:
            derive_endpoints(config, "live", writes_enabled=True, readonly_run=False)
        assert raised.value.key == "control_system.connector.epics.gateways.write_access"

    def test_a_bad_row_of_the_role_not_selected_refuses_nothing(self):
        """Behaviour change: a malformed unselected row derives no endpoint and refuses nothing."""
        config = {
            "control_system": {
                "type": "epics",
                "connector": {
                    "epics": {
                        "gateways": {
                            "read_only": {"address": "gw", "port": 5064},
                            "write_access": {"address": "", "port": 5065},
                        }
                    }
                },
            }
        }
        derivation = derive_endpoints(config, "live", writes_enabled=False, readonly_run=False)
        assert derivation.selected_role == "read_only"
        assert set(derivation.endpoints) == {"read_only"}

    def test_a_name_server_spelled_neither_true_nor_false_keeps_the_flag_reader_words(self):
        config = {
            "control_system": {
                "type": "epics",
                "connector": {
                    "epics": {
                        "gateways": {"read_only": {"address": "gw", "use_name_server": "maybe"}}
                    }
                },
            }
        }
        with pytest.raises(ConfigurationError, match="use_name_server must be true or false"):
            derive_endpoints(config, "live", writes_enabled=False, readonly_run=False)

    def test_a_numeric_string_port_derives_an_integer(self):
        config = {
            "control_system": {
                "type": "epics",
                "connector": {
                    "epics": {"gateways": {"read_only": {"address": "gw", "port": "6064"}}}
                },
            }
        }
        row = derive_endpoints(config, "live", writes_enabled=False, readonly_run=False).endpoints[
            "read_only"
        ]
        assert (row.host, row.port, row.mode) == ("gw", 6064, "addr_list")

    def test_routing_globs_that_are_not_a_list_are_refused(self):
        config = {"control_system": {"type": "epics", "connector": {"epics": {"pva_channels": 5}}}}
        with pytest.raises(ConfigurationError) as raised:
            derive_endpoints(config, "live", writes_enabled=False, readonly_run=False)
        assert raised.value.key == "control_system.connector.epics.pva_channels"


class TestRefusedWhenTheFactoryReadsTheBlock:
    """Behaviour change: the type's settings model refuses its block at build, naming the key."""

    @pytest.mark.parametrize(
        ("block", "key"),
        [
            ({"pva_channels": 5}, "control_system.connector.epics.pva_channels"),
            (
                {"gateways": {"read_only": {"address": "gw", "port": True}}},
                "control_system.connector.epics.gateways.read_only",
            ),
            ({"gateways": ["gw:5064"]}, "control_system.connector.epics.gateways"),
            ({"fresh_reads": "sometimes"}, "control_system.connector.epics.fresh_reads"),
            ({"timeout_s": 0}, "control_system.connector.epics.timeout_s"),
            ({"timeout": 5}, "control_system.connector.epics.timeout"),
        ],
    )
    def test_a_block_the_model_cannot_read_fails_the_build(self, block, key):
        with pytest.raises(ConfigurationError) as raised:
            ConnectorFactory.build_control_system_connector(_epics(block))
        assert raised.value.key == key

    def test_a_block_with_a_key_that_is_not_a_string_fails_the_build(self):
        with pytest.raises(ConfigurationError) as raised:
            ConnectorFactory.build_control_system_connector(_epics({1: "one"}))
        assert raised.value.key == "control_system.connector.epics"

    def test_a_pva_gateway_is_read_only_beside_routing_globs(self):
        _, context = ConnectorFactory.build_control_system_connector(
            _epics({"pva_gateway": "junk"})
        )
        assert isinstance(context.settings, EpicsSettings)
        assert context.settings.pva_gateway is None

    def test_the_built_context_holds_the_types_model(self):
        _, context = ConnectorFactory.build_control_system_connector(
            _epics({"gateways": {"read_only": GATEWAY}, "timeout_s": "2.5", "probe_channel": "X"})
        )
        settings = context.settings
        assert isinstance(settings, EpicsSettings)
        assert settings.timeout_s == 2.5
        assert settings.gateways == {"read_only": GatewayRow(address="cagw.example", port=5064)}
        # A key the model does not name is kept as written.
        assert settings.model_extra == {"probe_channel": "X"}


class TestHandBuiltContexts:
    def test_a_mapping_is_read_for_the_contexts_own_type(self):
        with pytest.raises(ValueError, match="control_system.connector.epics.timeout_s"):
            ConnectorContext(connector_type="epics", settings={"timeout_s": -1})

    def test_an_out_of_tree_connector_reads_its_keys_from_the_extras(self):
        context = ConnectorContext(settings={"site_key": [1, 2]})
        assert type(context.settings) is ControlSystemSettings
        assert context.settings.model_extra == {"site_key": [1, 2]}

    def test_connect_reads_the_settings_into_the_connectors_model(self):
        settings = ControlSystemSettings.from_block(
            {"gateways": {"read_only": GATEWAY}}, key="control_system.connector.epics"
        )
        assert EPICSConnector.settings_model.from_block(
            settings.model_dump(), key="control_system.connector.epics"
        ).gateways == {"read_only": GatewayRow(address="cagw.example", port=5064)}


# -- the other sections ----------------------------------------------------------


ARCHIVER_SECTIONS: list[object] = [
    None,
    {},
    {"type": "epics_archiver", "settings": {"url": "https://a"}},
    {"type": "epics_archiver", "epics_archiver": {"url": "https://a"}},
    {"type": "epics_archiver", "settings": {"url": "a"}, "epics_archiver": {"url": "b"}},
    {"type": "mock_archiver", "settings": "x"},
    {"type": "site.Archiver", "settings": None, "stray": {"a": 1}},
    {"auto_bin_points": 500},
    {"auto_bin_points": "500"},
]


# -- the build lint over ``archiver:`` before the section held it, as the oracle --

_OLD_ARCHIVER_TYPES = [row.name for row in BUILTIN_CONNECTORS if row.kind == "archiver"]
_OLD_SETTINGS_LEAF = "settings"
_OLD_SECTION_LEAVES = ("type", _OLD_SETTINGS_LEAF, "auto_bin_points")


def _old_resolve_type(section: object) -> str:
    value = section.get("type") if isinstance(section, dict) else None
    return str(value) if value else MOCK_ARCHIVER


def _old_own_name_block(connector_type: str) -> str | None:
    return None if "." in connector_type else connector_type


def _old_homes(section: object) -> list[tuple[str, object]]:
    if not isinstance(section, dict):
        return []
    homes: list[tuple[str, object]] = []
    if _OLD_SETTINGS_LEAF in section:
        homes.append((_OLD_SETTINGS_LEAF, section[_OLD_SETTINGS_LEAF]))
    own = _old_own_name_block(_old_resolve_type(section))
    if own is not None and own != _OLD_SETTINGS_LEAF and own in section:
        homes.append((own, section[own]))
    return homes


def _old_archiver_settings_errors(section: object) -> list[str]:
    """The build lint as ``archiver_selection.archiver_settings_errors`` spelled it at H2."""
    if not isinstance(section, dict):
        return []
    selected = _old_resolve_type(section)
    own = _old_own_name_block(selected)
    known = {*_OLD_SECTION_LEAVES, *_OLD_ARCHIVER_TYPES, *([own] if own else [])}
    errors = [
        f"`{key}:` under `archiver:` is a block no archiver reads; the selected "
        f"archiver, {selected}, takes its settings from `archiver.settings`"
        for key, value in section.items()
        if isinstance(value, dict) and key not in known
    ]
    homes = _old_homes(section)
    if len(homes) > 1:
        errors.append(
            f"`archiver.settings` and `archiver.{homes[1][0]}` both configure {homes[1][0]}; "
            f"keep one — `archiver.settings` is the spelling for every archiver"
        )
    errors.extend(
        f"`archiver.{key}` is {value!r}, not a block; it holds the selected "
        f"archiver's settings as a mapping"
        for key, value in homes
        if value is not None and not isinstance(value, dict)
    )
    return errors


@pytest.mark.parametrize("section", ARCHIVER_SECTIONS)
def test_the_archiver_section_names_what_the_build_lint_names(section):
    assert list(
        ArchiverSection.read(copy.deepcopy(section)).problems
    ) == _old_archiver_settings_errors(section)


def test_the_archiver_section_keeps_where_its_settings_are_written():
    read = ArchiverSection.read({"type": "epics_archiver", "epics_archiver": {"url": "u"}})
    assert (read.type, read.settings_home, read.settings) == (
        "epics_archiver",
        "own_name",
        {"url": "u"},
    )
    assert isinstance(ArchiverSection.read({"settings": "x"}).settings, Unreadable)
    assert isinstance(ArchiverSection.read({"auto_bin_points": True}).auto_bin_points, Unreadable)


@pytest.mark.parametrize(
    "section",
    [
        None,
        {},
        {"tick_s": 0.5},
        {"tick_s": 2, "fast_tick_s": 0.05},
        {"tick_s": 0},
        {"fast_tick_s": "1"},
    ],
)
def test_the_simulation_section_reads_the_periods_the_resolvers_read(section):
    read = SimulationSection.read(section)
    config = {} if section is None else {"simulation": section}
    for value, resolve in ((read.tick_s, resolve_tick_s), (read.fast_tick_s, resolve_fast_tick_s)):
        if isinstance(value, Unreadable):
            with pytest.raises(ValueError):
                resolve(config)
        elif value is not None:
            assert resolve(config) == value


def test_a_simulation_section_that_is_not_a_mapping_is_unreadable():
    assert SimulationSection.read([1]).unreadable == Unreadable(key="simulation", raw=[1])


@pytest.mark.parametrize("section", [None, {}, {"base_dir": "data/agent"}, {"base_dir": ""}, "x"])
def test_the_agent_data_section_reads_the_base_dir_the_workspace_reads(section):
    read = AgentDataSection.read(section)
    assert (read.base_dir or DEFAULT_AGENT_DATA_BASE_DIR) == agent_data_base_dir(
        {"agent_data": section}
    )


def test_the_logging_switches_read_a_quoted_false_as_off():
    read = LoggingSection.read(
        {"rich_tracebacks": "false", "show_full_paths": "yes", "show_traceback_locals": "?"}
    )
    assert read.rich_tracebacks is False
    assert read.show_full_paths is True
    assert isinstance(read.show_traceback_locals, Unreadable)
    assert LoggingSection.read(None) == LoggingSection()
