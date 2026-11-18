"""Tests for the ``channel_directory.fleet`` settings: defaults, targets, the period rule, classes."""

from __future__ import annotations

from typing import Any

import pytest

from osprey.services.channel_directory.fleet.settings import (
    DEFAULT_CLASS,
    DEFAULT_LISTING_DEADLINE_S,
    DEFAULT_PERIOD_S,
    DEFAULT_RETENTION_DAYS,
    MIN_PERIOD_S,
    NOT_REFRESHED_IN_PROCESS,
    ChannelSelector,
    FleetSettings,
    HealthChannel,
    resolve_fleet,
)
from osprey.services.channel_directory.resolve import SectionError

#: A live machine on EPICS with a served virtual accelerator and a stand-in beside it.
_LIVE = {
    "type": "epics",
    "connector": {
        "epics": {"timeout_s": 4},
        "virtual_accelerator": {"port": 5064, "timeout_s": 2},
        "live_standin": {"port": 5070},
    },
}

#: The simulator in process: no type stated.
_IN_PROCESS: dict[str, Any] = {}

#: The simulator served: its own type, no serving leaf.
_SERVED_VA = {"type": "virtual_accelerator", "connector": {"virtual_accelerator": {"port": 5064}}}


def _resolve(
    fleet: Any,
    control_system: dict[str, Any] | None = None,
    display: dict[str, Any] | None = None,
) -> FleetSettings:
    section: dict[str, Any] = {"fleet": fleet}
    if display is not None:
        section["display"] = display
    config: dict[str, Any] = {"channel_directory": section}
    if control_system is not None:
        config["control_system"] = control_system
    return resolve_fleet(config, fleet)


def _refused(fleet: Any, control_system: dict[str, Any] | None = None) -> str:
    with pytest.raises(SectionError) as raised:
        _resolve(fleet, control_system if control_system is not None else _LIVE)
    return raised.value.key


class TestDefaults:
    def test_empty_block_fills_every_default(self) -> None:
        settings = _resolve({}, _LIVE)

        assert [target.name for target in settings.targets] == ["live"]
        assert settings.period_s == DEFAULT_PERIOD_S
        assert settings.retention_days == DEFAULT_RETENTION_DAYS
        assert settings.listing_deadline_s == DEFAULT_LISTING_DEADLINE_S
        assert settings.group_by == "host"
        assert settings.classes == {}
        assert settings.roster_sources == ()

    def test_default_class_is_the_heartbeat_counter(self) -> None:
        default = _resolve({}, _LIVE).default_class

        assert default.name == DEFAULT_CLASS
        assert default.liveness is not None
        assert default.liveness.kind == "counter"
        assert default.liveness.selector == ChannelSelector(suffix="HEARTBEAT", role="heartbeat")
        assert default.health == ()

    def test_default_heartbeat_follows_the_display_suffix(self) -> None:
        display = {"server_health_suffixes": {"heartbeat": ":hb", "uptime": ":up"}}

        default = _resolve({}, _LIVE, display).default_class

        assert default.liveness is not None
        assert default.liveness.selector.suffix == ":hb"

    def test_display_suffixes_without_heartbeat_leave_no_liveness_channel(self) -> None:
        """As ``Facts`` reads them: a mapping naming a role replaces the defaults."""
        default = _resolve({}, _LIVE, {"server_health_suffixes": {"uptime": ":up"}}).default_class

        assert default.liveness is None

    def test_default_heartbeat_suffix_is_the_simulator_signal(self) -> None:
        pytest.importorskip("osprey_connectors.simulation.view")
        from osprey_connectors.simulation.view import HEARTBEAT

        default = _resolve({}, _LIVE).default_class
        assert default.liveness is not None
        assert default.liveness.selector.suffix == HEARTBEAT


class TestTargets:
    def test_default_is_the_baseline_target(self) -> None:
        assert [t.name for t in _resolve({}, _SERVED_VA).targets] == ["va"]

    def test_named_targets_keep_their_order_and_read_bounds(self) -> None:
        settings = _resolve({"targets": ["va", "live"], "period_s": 10}, _LIVE)

        assert [(t.name, t.refreshable, t.read_bound_s) for t in settings.targets] == [
            ("va", True, 2.0),
            ("live", True, 4.0),
        ]
        assert settings.read_bound_s == 4.0

    def test_unset_timeout_is_the_connector_default(self) -> None:
        settings = _resolve({"targets": ["standin"]}, _LIVE)

        assert settings.targets[0].read_bound_s == 5.0

    def test_in_process_simulator_is_not_refreshable(self) -> None:
        """D12: each login reads its own instance."""
        settings = _resolve({}, _IN_PROCESS)

        (target,) = settings.targets
        assert (target.name, target.refreshable, target.reason) == (
            "va",
            False,
            NOT_REFRESHED_IN_PROCESS,
        )
        assert target.read_bound_s is None
        assert settings.refreshed_targets() == ()

    def test_lookup_by_name(self) -> None:
        settings = _resolve({"targets": ["live"]}, _LIVE)

        assert settings.target("live") is settings.targets[0]
        assert settings.target("va") is None

    @pytest.mark.parametrize(
        "targets",
        [[], "live", ["moon"], ["live", "live"], [None]],
    )
    def test_malformed_targets(self, targets: Any) -> None:
        assert _refused({"targets": targets}) == "channel_directory.fleet.targets"

    def test_unconfigured_target_is_refused(self) -> None:
        assert _refused({"targets": ["standin"]}, _SERVED_VA) == "channel_directory.fleet.targets"

    def test_bad_connector_timeout_names_its_own_key(self) -> None:
        control_system = {"type": "epics", "connector": {"epics": {"timeout_s": -1}}}

        assert _refused({}, control_system) == "control_system.connector.epics.timeout_s"


class TestPeriodRule:
    """D6: ``period_s`` at least twice the read bound and at least 5 s."""

    def test_twice_the_read_bound_passes(self) -> None:
        assert _resolve({"period_s": 8}, _LIVE).period_s == 8.0

    def test_under_twice_the_read_bound_fails(self) -> None:
        with pytest.raises(SectionError) as raised:
            _resolve({"period_s": 7.9}, _LIVE)

        assert raised.value.key == "channel_directory.fleet.period_s"
        assert "8s" in raised.value.problem

    def test_the_largest_bound_of_the_named_targets_counts(self) -> None:
        assert _refused({"targets": ["va", "standin"], "period_s": 9}) == (
            "channel_directory.fleet.period_s"
        )

    def test_minimum_holds_with_a_short_read_bound(self) -> None:
        fast = {"type": "epics", "connector": {"epics": {"timeout_s": 1}}}

        assert _resolve({"period_s": MIN_PERIOD_S}, fast).period_s == MIN_PERIOD_S
        assert _refused({"period_s": 4.9}, fast) == "channel_directory.fleet.period_s"

    def test_minimum_holds_with_no_refreshed_target(self) -> None:
        assert _refused({"period_s": 4}, _IN_PROCESS) == "channel_directory.fleet.period_s"

    def test_default_period_holds_against_a_long_read_bound(self) -> None:
        slow = {"type": "epics", "connector": {"epics": {"timeout_s": 20}}}

        assert _refused({}, slow) == "channel_directory.fleet.period_s"


class TestScalars:
    @pytest.mark.parametrize(
        ("leaf", "value"),
        [
            ("period_s", 0),
            ("period_s", "soon"),
            ("period_s", True),
            ("retention_days", -1),
            ("listing_deadline_s", 0),
            ("group_by", "rack"),
            ("roster_sources", "site.roster:SOURCE"),
            ("roster_sources", ["site.roster"]),
            ("roster_sources", ["site.roster:SOURCE", "site.roster:SOURCE"]),
        ],
    )
    def test_invalid_value_names_its_key(self, leaf: str, value: Any) -> None:
        assert _refused({leaf: value}) == f"channel_directory.fleet.{leaf}"

    def test_unknown_key_is_refused(self) -> None:
        assert _refused({"period": 30}) == "channel_directory.fleet.period"

    def test_block_that_is_not_a_mapping(self) -> None:
        assert _refused(["live"]) == "channel_directory.fleet"

    @pytest.mark.parametrize("group_by", ["host", "class_name", "group"])
    def test_group_by(self, group_by: str) -> None:
        assert _resolve({"group_by": group_by}, _LIVE).group_by == group_by

    def test_values_read_through(self) -> None:
        settings = _resolve(
            {
                "retention_days": 2,
                "listing_deadline_s": "45",
                "roster_sources": ["site.roster:SOURCE", "other.mod:make"],
            },
            _LIVE,
        )

        assert settings.retention_days == 2.0
        assert settings.listing_deadline_s == 45.0
        assert settings.roster_sources == ("site.roster:SOURCE", "other.mod:make")


_VXWORKS = {
    "liveness": {"suffix": ":IOC_STS", "kind": "status"},
    "health": [
        {"suffix": ":CPU_LOAD", "good": [0, 80]},
        {"pattern": "{server}:MEM_FREE", "good": [1000, None]},
    ],
}


class TestClasses:
    def test_named_class(self) -> None:
        settings = _resolve({"classes": {"vxworks": _VXWORKS}}, _LIVE)

        vxworks = settings.classes["vxworks"]
        assert vxworks.liveness is not None
        assert vxworks.liveness.kind == "status"
        assert vxworks.liveness.selector == ChannelSelector(suffix=":IOC_STS")
        assert vxworks.health == (
            HealthChannel(selector=ChannelSelector(suffix=":CPU_LOAD"), low=0.0, high=80.0),
            HealthChannel(
                selector=ChannelSelector(pattern="{server}:MEM_FREE"), low=1000.0, high=None
            ),
        )

    def test_pattern_addresses_a_server(self) -> None:
        selector = ChannelSelector(pattern="{server}:hb")

        assert selector.address_for("IOC01") == "IOC01:hb"
        assert ChannelSelector(suffix=":hb").address_for("IOC01") is None

    def test_good_range(self) -> None:
        bounded = HealthChannel(selector=ChannelSelector(suffix=":x"), low=0.0, high=80.0)
        open_high = HealthChannel(selector=ChannelSelector(suffix=":x"), low=10.0)

        assert bounded.in_range(0.0) and bounded.in_range(80.0)
        assert not bounded.in_range(-0.1) and not bounded.in_range(80.1)
        assert open_high.in_range(1e9) and not open_high.in_range(9.9)

    def test_class_without_liveness_has_none(self) -> None:
        settings = _resolve({"classes": {"soft": {"health": []}}}, _LIVE)

        assert settings.classes["soft"].liveness is None

    def test_classes_apply_on_live_only(self) -> None:
        settings = _resolve({"targets": ["live", "va"], "classes": {"vxworks": _VXWORKS}}, _LIVE)

        assert settings.class_for("vxworks", target="live").name == "vxworks"
        assert settings.class_for("vxworks", target="va") is settings.default_class
        assert settings.class_for("unknown", target="live") is settings.default_class
        assert settings.class_for(None, target="live") is settings.default_class

    @pytest.mark.parametrize(
        ("classes", "key"),
        [
            ("vxworks", "channel_directory.fleet.classes"),
            ({"vxworks": "status"}, "channel_directory.fleet.classes.vxworks"),
            ({"default": {}}, "channel_directory.fleet.classes.default"),
            ({"x": {"heartbeat": {}}}, "channel_directory.fleet.classes.x.heartbeat"),
            (
                {"x": {"liveness": {"suffix": ":hb"}}},
                "channel_directory.fleet.classes.x.liveness.kind",
            ),
            (
                {"x": {"liveness": {"suffix": ":hb", "kind": "gauge"}}},
                "channel_directory.fleet.classes.x.liveness.kind",
            ),
            (
                {"x": {"liveness": {"kind": "counter"}}},
                "channel_directory.fleet.classes.x.liveness",
            ),
            (
                {"x": {"liveness": {"suffix": ":hb", "pattern": "{server}:hb", "kind": "counter"}}},
                "channel_directory.fleet.classes.x.liveness",
            ),
            (
                {"x": {"liveness": {"suffix": "", "kind": "counter"}}},
                "channel_directory.fleet.classes.x.liveness.suffix",
            ),
            (
                {"x": {"liveness": {"pattern": "IOC:hb", "kind": "counter"}}},
                "channel_directory.fleet.classes.x.liveness.pattern",
            ),
            (
                {"x": {"liveness": {"pattern": "{server}:{n}", "kind": "counter"}}},
                "channel_directory.fleet.classes.x.liveness.pattern",
            ),
            ({"x": {"health": {"suffix": ":cpu"}}}, "channel_directory.fleet.classes.x.health"),
            (
                {"x": {"health": [{"suffix": ":cpu"}]}},
                "channel_directory.fleet.classes.x.health[0].good",
            ),
            (
                {"x": {"health": [{"suffix": ":cpu", "good": [None, None]}]}},
                "channel_directory.fleet.classes.x.health[0].good",
            ),
            (
                {"x": {"health": [{"suffix": ":cpu", "good": [80, 0]}]}},
                "channel_directory.fleet.classes.x.health[0].good",
            ),
            (
                {"x": {"health": [{"suffix": ":cpu", "good": [0, "high"]}]}},
                "channel_directory.fleet.classes.x.health[0].good",
            ),
            (
                {"x": {"health": [{"suffix": ":cpu", "good": [0, float("nan")]}]}},
                "channel_directory.fleet.classes.x.health[0].good",
            ),
            (
                {"x": {"health": [{"suffix": ":cpu", "good": [0, 1], "kind": "counter"}]}},
                "channel_directory.fleet.classes.x.health[0].kind",
            ),
        ],
    )
    def test_malformed_class_names_its_key(self, classes: Any, key: str) -> None:
        assert _refused({"classes": classes}) == key
