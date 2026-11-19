"""What ``connect()`` takes, and what a caller cannot hand it.

A control-system connector is connected with one frozen
:class:`~osprey_connectors.control_system.ConnectorContext`. ``connect()`` is
final: it stores the context, builds the limits validator from the
deployment's *loaded* config for the context's type, then calls the
connector's own ``_open(settings)``. The reference monitor judges every write
by that loaded config too, so nothing a caller puts in the context, its
settings included, can hand a connector an empty validator or arm its writes.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from osprey_connectors.connector_types import EPICS, TARGET_VA, TRANSPORT_CA
from osprey_connectors.control_system import ConnectorContext, ControlSystemConnector
from osprey_connectors.errors import ChannelLimitsViolationError
from osprey_connectors.testing import config_of
from osprey_connectors_tests._config_trees import config_tree, serve_config
from osprey_connectors_tests._fake_connector import FakeConnector
from osprey_connectors_tests._write_fakes import arm_switch

#: The deployment's loaded section: limits checked, exclusive, writes allowed.
_LOADED = {"writes": "allowed", "limits_checking": {"enabled": True, "mode": "exclusive"}}

#: What a caller might try to slip in through the connector's own block: the
#: keys the reference monitor and the limits posture read from a section.
_MONITOR_KEYS = {"writes": "allowed", "limits_checking": {"enabled": False}}


def _serve(tmp_path: Path, section: dict[str, Any]) -> None:
    """Serve *section* as the loaded ``control_system`` config, with a one-channel limits file."""
    database = tmp_path / "limits.json"
    database.write_text(
        json.dumps({"FOO": {"writable": True, "min_value": 0.0, "max_value": 10.0}})
    )
    tree = config_tree(
        {"control_system.limits_checking.database_path": str(database)},
        base={"control_system": section},
    )
    serve_config(config_of(tree, environ={}))


class TestTheContextCarriesNoJudgement:
    """The context holds what the connector was built for, and nothing the monitor reads."""

    def test_its_fields_are_exactly_these(self):
        assert set(ConnectorContext.model_fields) == {
            "connector_type",
            "control_target",
            "transport",
            "settings",
            "readonly_wording",
        }

    @pytest.mark.parametrize(
        "field",
        ["limits_validator", "validator", "limits", "control_system", "section", "writes"],
    )
    def test_a_validator_or_a_section_is_refused(self, field):
        with pytest.raises(ValidationError):
            ConnectorContext(**{field: object()})

    def test_connect_takes_the_context_alone(self):
        parameters = list(inspect.signature(ControlSystemConnector.connect).parameters)

        assert parameters == ["self", "context"]

    def test_the_context_is_frozen(self):
        context = ConnectorContext(control_target=TARGET_VA)

        with pytest.raises(ValidationError):
            context.control_target = "live"  # type: ignore[misc]

    def test_a_readonly_wording_outside_the_two_is_refused(self):
        with pytest.raises(ValidationError):
            ConnectorContext(readonly_wording="operator")  # type: ignore[arg-type]


class TestConnectBuildsTheValidator:
    """The base builds the validator from the loaded config, for every connector."""

    async def test_a_connector_that_builds_none_is_checked(self, tmp_path):
        """The fake's ``_open`` loads no validator; ``connect()`` builds one anyway."""
        _serve(tmp_path, _LOADED)
        connector = FakeConnector()

        await connector.connect(ConnectorContext(connector_type=EPICS))

        assert connector._limits_validator is not None
        with pytest.raises(ChannelLimitsViolationError):
            connector._limits_validator.validate("FOO", 99.0)

    async def test_settings_cannot_turn_the_limits_off(self, tmp_path):
        _serve(tmp_path, _LOADED)
        connector = FakeConnector()

        await connector.connect(ConnectorContext(connector_type=EPICS, settings=_MONITOR_KEYS))

        assert connector._limits_validator is not None
        assert connector._limits_validator.policy["mode"] == "exclusive"
        with pytest.raises(ChannelLimitsViolationError):
            connector._limits_validator.validate("FOO", 99.0)

    async def test_the_validator_follows_the_loaded_config_only(self, tmp_path):
        """Limits off in the loaded config: no validator, whatever the settings say."""
        _serve(tmp_path, {"limits_checking": {"enabled": False}})
        connector = FakeConnector()

        await connector.connect(
            ConnectorContext(
                connector_type=EPICS,
                settings={"limits_checking": {"enabled": True, "mode": "exclusive"}},
            )
        )

        assert connector._limits_validator is None


class TestSettingsCannotArmWrites:
    """The write verdict reads the loaded section, not the block ``connect()`` was handed."""

    async def test_a_tier_that_never_allows_still_refuses(self, monkeypatch, tmp_path):
        arm_switch(monkeypatch, tmp_path / "data", TARGET_VA, stamp=False)
        _serve(tmp_path, {"writes": "never"})
        connector = FakeConnector()

        await connector.connect(ConnectorContext(control_target=TARGET_VA, settings=_MONITOR_KEYS))

        assert connector._writes_enabled is False
        assert connector._last_verdict is not None
        assert connector._last_verdict.reason == "tier_never"

    async def test_the_same_connector_writes_when_the_loaded_tier_allows(
        self, monkeypatch, tmp_path
    ):
        """The control: the refusal above is the loaded tier's, not the test's setup."""
        arm_switch(monkeypatch, tmp_path / "data", TARGET_VA, stamp=False)
        _serve(tmp_path, {"writes": "allowed"})
        connector = FakeConnector()

        await connector.connect(ConnectorContext(control_target=TARGET_VA))

        assert connector._writes_enabled is True


class TestTheContextIsWhatTheConnectorReads:
    async def test_connect_stores_the_context(self, tmp_path):
        _serve(tmp_path, _LOADED)
        context = ConnectorContext(
            connector_type=EPICS, control_target=TARGET_VA, transport=TRANSPORT_CA
        )
        connector = FakeConnector()

        await connector.connect(context)

        assert connector.context is context
        assert connector._connector_type == EPICS
        assert connector._control_target == TARGET_VA
        assert connector.transport == TRANSPORT_CA

    def test_an_unconnected_connector_names_nothing(self):
        connector = FakeConnector()

        assert connector.context == ConnectorContext()
        assert (connector._connector_type, connector._control_target, connector.transport) == (
            None,
            None,
            None,
        )

    @pytest.mark.parametrize("stamp", ["_connector_type", "_control_target", "transport"])
    def test_no_stamp_can_be_rewritten_after_connect(self, stamp):
        with pytest.raises(AttributeError):
            setattr(FakeConnector(), stamp, "live")

    def test_a_subclass_that_defines_connect_is_refused(self):
        with pytest.raises(TypeError, match="connect"):

            class _Overrides(FakeConnector):
                async def connect(self, context: ConnectorContext) -> None:  # type: ignore[misc]  # noqa: ARG002 - the override is the point
                    return None
