"""Tests for ControlSystemConnector._writes_enabled: the write verdict at the connector.

The property is the reference monitor's decision, and it is
:func:`osprey_connectors.write_switch.write_verdict` for the connector's own
machine: the login's write tier, the machine the factory stamped, and the
login's switch for that machine. The tier is a ceiling and target-blind; the
switch picks the machine.
"""

from collections.abc import Callable
from typing import Any

import pytest

from osprey_connectors.connector_types import TARGET_LIVE, TARGET_VA
from osprey_connectors.control_system import ConnectorContext
from osprey_connectors.control_system._base import (
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
)
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors_tests._config_trees import control_system_config, installed
from osprey_connectors_tests._write_fakes import (
    ARMED_SECTION,
    RecordingConnector,
    arm_switch,
    armed_context,
    serve_writes_enabled,
)
from tests._vocabulary import REFUSAL_BAN_LIST
from tests.facility.served_tree import in_process_config, served_tree


@pytest.fixture
def switch_on(monkeypatch, tmp_path):
    """The login's switch on for the simulator, connectors stamped with it."""
    return arm_switch(monkeypatch, tmp_path / "armed-data")


@pytest.fixture
def armed(switch_on):
    """A write tier that allows, and the switch on: a write reaches the transport."""
    serve_writes_enabled()
    return switch_on


class _StubConnector(ControlSystemConnector):
    """Minimal concrete subclass for testing base-class properties."""

    async def _open(self, config: dict[str, Any]) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def read_channel(self, address: str, *, timeout_s: float | None = None) -> ChannelValue:
        raise NotImplementedError

    async def _send(self, address: str, value: Any, *, timeout_s: float, wait: bool):
        raise NotImplementedError

    async def subscribe(
        self,
        address: str,
        callback: Callable[[ChannelValue], None],
        *,
        on_connection: Callable[[bool], None] | None = None,
    ) -> str:
        raise NotImplementedError

    async def unsubscribe(self, subscription_id: str) -> None:
        pass


class TestInitSubclassWrapping:
    """Tests for the write guard of the final write methods."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("switch_on")
    async def test_write_blocked_when_disabled(self):
        """A tier that does not allow writes refuses, the write never attempted."""
        connector = _StubConnector()
        with installed(control_system_config({})):
            result = await connector.write_channel("TEST:PV", 1.0)
        assert isinstance(result, ChannelWriteResult)
        assert result.outcome is WriteOutcome.REFUSED
        assert result.refusal_reason == "WRITES_DISABLED"
        assert "TEST:PV" in result.error_message
        assert refusal_phrase("tier_never") in result.error_message
        assert REFUSAL_BAN_LIST.search(result.error_message) is None

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("armed")
    async def test_write_passes_through_when_enabled(self):
        """Tier allows and switch on: the original write_channel is called."""
        connector = RecordingConnector()
        result = await connector.write_channel("TEST:PV", 42.0)
        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.value_written == 42.0

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("switch_on")
    async def test_write_multiple_blocked_when_disabled(self):
        """A tier that does not allow writes refuses every operation of a batch."""
        connector = _StubConnector()
        ops = [("PV:A", 1.0), ("PV:B", 2.0)]
        with installed(control_system_config({})):
            results = await connector.write_multiple_channels(ops)
        assert len(results) == 2
        assert all(r.outcome is WriteOutcome.REFUSED for r in results)
        assert "PV:A" in results[0].error_message
        assert "PV:B" in results[1].error_message
        assert refusal_phrase("tier_never") in results[0].error_message

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("armed")
    async def test_write_multiple_passes_through_when_enabled(self):
        """Tier allows and switch on: the original write_multiple_channels is called."""
        connector = RecordingConnector()
        ops = [("PV:A", 1.0), ("PV:B", 2.0)]
        results = await connector.write_multiple_channels(ops)
        assert len(results) == 2
        assert all(r.outcome is WriteOutcome.CONFIRMED for r in results)


class TestWritesEnabledProperty:
    """Tests for the _writes_enabled base-class property."""

    @pytest.mark.usefixtures("switch_on")
    def test_a_tier_that_does_not_allow_refuses(self):
        connector = _StubConnector()
        with installed(control_system_config({})):
            assert connector._writes_enabled is False

    @pytest.mark.usefixtures("armed")
    def test_a_tier_that_allows_with_the_switch_on_allows(self):
        assert _StubConnector()._writes_enabled is True

    def test_a_tier_that_allows_with_the_switch_off_refuses(self, monkeypatch, tmp_path):
        """No record at all is every switch off."""
        arm_switch(monkeypatch, tmp_path / "data", None)
        monkeypatch.setattr(
            ControlSystemConnector, "_context", ConnectorContext(control_target=TARGET_VA)
        )
        serve_writes_enabled()
        assert _StubConnector()._writes_enabled is False

    @pytest.mark.usefixtures("switch_on")
    def test_a_config_that_cannot_be_found_fails_safe(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "missing.yml"))
        assert _StubConnector()._writes_enabled is False

    @pytest.mark.usefixtures("switch_on")
    def test_a_config_read_that_raises_fails_safe(self, monkeypatch):
        def broken():
            raise RuntimeError("config broken")

        monkeypatch.setattr("osprey_connectors.config.current_config", broken)
        assert _StubConnector()._writes_enabled is False


class TestMockWritesDisabledViaBaseClass:
    """VAInProcessConnector write blocking comes from the base class: full write path."""

    def test_mock_has_no_enable_writes_attr(self):
        """VAInProcessConnector must not carry an _enable_writes attribute."""
        from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector

        connector = VAInProcessConnector()
        assert not hasattr(connector, "_enable_writes")

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("switch_on")
    async def test_write_blocked_full_path(self, tmp_path):
        """Full path: VAInProcessConnector under a tier that does not allow blocks writes."""
        from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector

        view = served_tree(tmp_path, ["BEAM:CURRENT"])
        connector = VAInProcessConnector()
        with installed(control_system_config({})):
            await connector.connect(
                armed_context(settings=in_process_config(view, response_delay_ms=0))
            )
            result = await connector.write_channel("BEAM:CURRENT", 500.0)

        assert isinstance(result, ChannelWriteResult)
        assert result.outcome is WriteOutcome.REFUSED
        assert "BEAM:CURRENT" in result.error_message
        assert refusal_phrase("tier_never", "simulator") in result.error_message

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("switch_on")
    async def test_write_allowed_full_path(self, tmp_path):
        """Full path: an armed VAInProcessConnector writes and confirms by readback.

        No limits database and a noise-free confirming read, so the fleet
        default resolves to a confirmed write.
        """
        from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector

        view = served_tree(tmp_path, ["BEAM:CURRENT"])
        connector = VAInProcessConnector()
        with installed(control_system_config(ARMED_SECTION)):
            await connector.connect(
                armed_context(settings=in_process_config(view, response_delay_ms=0))
            )
            result = await connector.write_channel("BEAM:CURRENT", 500.0)

        assert isinstance(result, ChannelWriteResult)
        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.address == "BEAM:CURRENT"
        assert result.value_written == 500.0


#: A deployment whose tier allows writes. The tier names no machine: the
#: simulator's connector block is configured, the ``epics`` one too, and which
#: of them a write may reach is the login's switch.
_TIER_ALLOWED_SECTION = {
    "type": "epics",
    "writes": "allowed",
    "connector": {
        "virtual_accelerator": {"response_delay_ms": 0},
        "epics": {"port": 5064},
    },
}

#: The same allowed tier with a retired per-type leaf disarming ``epics``. The
#: leaf is read by nothing at runtime (ruling D2), so it holds back nothing.
_LIVE_DISARMED_SECTION = {
    "type": "epics",
    "writes": "allowed",
    "connector": {"epics": {"writes_enabled": False}},
}


class _TypeSeenConnector(RecordingConnector):
    """Records the type stamp visible to ``connect()``."""

    def __init__(self):
        super().__init__()
        self.type_seen_in_connect: Any = "not connected"

    async def _open(self, config):  # noqa: ARG002 - the control-system connector interface fixes this signature
        self.type_seen_in_connect = self._connector_type


class TestFactoryTypeStamp:
    """The factory stamps the connector type it built, before connect() runs."""

    def test_unstamped_by_default(self):
        """An instance nobody built through the factory carries no type."""
        assert _StubConnector()._connector_type is None

    @pytest.mark.asyncio
    async def test_stamp_is_visible_inside_connect(self):
        from osprey_connectors import connector_types
        from osprey_connectors.config_sections import ControlSystemSection
        from osprey_connectors.factory import ConnectorFactory
        from osprey_connectors_tests._harness import isolated_connector_registries

        with isolated_connector_registries(clear=True):
            ConnectorFactory.register_control_system(
                connector_types.VIRTUAL_ACCELERATOR, _TypeSeenConnector
            )
            connector = await ConnectorFactory.create_control_system_connector(
                ControlSystemSection.read({"type": connector_types.VIRTUAL_ACCELERATOR})
            )

        assert connector.type_seen_in_connect == connector_types.VIRTUAL_ACCELERATOR
        assert connector._connector_type == connector_types.VIRTUAL_ACCELERATOR


class TestTheTierIsTargetBlind:
    """The tier is one ``control_system.writes`` key, and it arms every machine.

    Which machine a write may reach is the login's switch, not the tier (ruling
    PR-3). The retired ``writes_enabled`` leaves are not read at runtime, at any
    level (ruling D2; rows ``tier-absent-old-armed-flat``,
    ``tier-absent-old-armed-connector``).
    """

    def _stub(self, connector_type: str, target: str | None) -> _StubConnector:
        connector = _StubConnector()
        connector._context = ConnectorContext(connector_type=connector_type, control_target=target)
        return connector

    def test_an_allowed_tier_allows_the_switched_machine(self, monkeypatch, tmp_path):
        """The tier allows; the live machine's switch is on."""
        arm_switch(monkeypatch, tmp_path / "data", TARGET_LIVE, stamp=False)
        connector = self._stub("epics", TARGET_LIVE)
        with installed(control_system_config(_TIER_ALLOWED_SECTION)):
            assert connector._writes_enabled is True

    @pytest.mark.asyncio
    async def test_a_machine_whose_switch_is_off_is_refused(self, monkeypatch, tmp_path):
        """The same tier, the simulator switched on: the live machine stays off."""
        arm_switch(monkeypatch, tmp_path / "data", TARGET_VA, stamp=False)
        connector = self._stub("epics", TARGET_LIVE)
        with installed(control_system_config(_TIER_ALLOWED_SECTION)):
            result = await connector.write_channel("TEST:PV", 1.0)

        assert result.outcome is WriteOutcome.REFUSED
        assert refusal_phrase("switch_off", "real") in result.error_message
        assert REFUSAL_BAN_LIST.search(result.error_message) is None

    def test_a_retired_type_leaf_does_not_hold_back_an_allowed_tier(self, monkeypatch, tmp_path):
        arm_switch(monkeypatch, tmp_path / "data", TARGET_LIVE, stamp=False)
        connector = self._stub("epics", TARGET_LIVE)
        with installed(control_system_config(_LIVE_DISARMED_SECTION)):
            assert connector._writes_enabled is True

    @pytest.mark.parametrize("stand_in", [True, "true", 1, "yes"])
    def test_the_retired_key_arms_nothing(self, stand_in, monkeypatch, tmp_path):
        """Not even a literal ``true``: no tier key reads never (row ``tier-absent-old-armed-flat``)."""
        arm_switch(monkeypatch, tmp_path / "data", TARGET_LIVE, stamp=False)
        connector = self._stub("epics", TARGET_LIVE)
        section = {"type": "epics", "writes_enabled": stand_in}
        with installed(control_system_config(section)):
            assert connector._writes_enabled is False

    @pytest.mark.asyncio
    async def test_an_unstamped_connector_is_refused_as_an_unknown_machine(
        self, monkeypatch, tmp_path
    ):
        """No stamp names no machine, whatever the tier and every switch say."""
        arm_switch(monkeypatch, tmp_path / "data", TARGET_VA, stamp=False)
        connector = _StubConnector()
        with installed(control_system_config(_LIVE_DISARMED_SECTION)):
            result = await connector.write_channel("TEST:PV", 1.0)

        assert result.outcome is WriteOutcome.REFUSED
        assert refusal_phrase("target_unknown") in result.error_message
        assert REFUSAL_BAN_LIST.search(result.error_message) is None
