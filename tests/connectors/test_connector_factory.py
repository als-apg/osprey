"""Tests for connector factory."""

from unittest.mock import patch

import pytest

from osprey.connectors import types
from osprey.connectors.archiver.base import ArchiverConnector
from osprey.connectors.archiver.mock_archiver_connector import MockArchiverConnector
from osprey.connectors.control_system.base import ControlSystemConnector
from osprey.connectors.control_system.mock_connector import MockConnector
from osprey.connectors.factory import (
    _BUILTIN_ARCHIVERS,
    _BUILTIN_CONTROL_SYSTEMS,
    ConnectorFactory,
    isolated_connector_registries,
    register_builtin_connectors,
)


@pytest.fixture(autouse=True)
def setup_test_connectors():
    """Register mock connectors as the only registrations each test sees.

    Snapshot/restore brackets the clear so registrations made elsewhere in the
    process survive this module's teardown.
    """
    with isolated_connector_registries(clear=True):
        # Register mock connectors (simulates what registry does)
        ConnectorFactory.register_control_system("mock", MockConnector)
        ConnectorFactory.register_archiver("mock_archiver", MockArchiverConnector)

        yield


class TestConnectorFactory:
    """Test ConnectorFactory functionality."""

    @pytest.mark.asyncio
    async def test_create_mock_control_system_connector(self):
        """Test creating a mock control system connector."""
        config = {
            "type": "mock",
            "connector": {"mock": {"response_delay_ms": 0, "noise_level": 0.01}},
        }

        connector = await ConnectorFactory.create_control_system_connector(config)

        assert isinstance(connector, ControlSystemConnector)
        assert isinstance(connector, MockConnector)
        assert connector._connected is True

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_create_mock_archiver_connector(self):
        """Test creating a mock archiver connector."""
        config = {
            "type": "mock_archiver",
            "mock_archiver": {"sample_rate_hz": 1.0, "noise_level": 0.01},
        }

        connector = await ConnectorFactory.create_archiver_connector(config)

        assert isinstance(connector, ArchiverConnector)
        assert isinstance(connector, MockArchiverConnector)
        assert connector._connected is True

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_factory_creates_independent_instances(self):
        """Test that factory creates independent connector instances."""
        config = {"type": "mock", "connector": {"mock": {"response_delay_ms": 0}}}

        connector1 = await ConnectorFactory.create_control_system_connector(config)
        connector2 = await ConnectorFactory.create_control_system_connector(config)

        # Should be different instances
        assert connector1 is not connector2

        # Disconnecting one should not affect the other
        await connector1.disconnect()
        assert connector1._connected is False
        assert connector2._connected is True

        await connector2.disconnect()

    def test_register_custom_connector(self):
        """Test registering a custom connector."""

        # Create a dummy connector class
        class CustomConnector(ControlSystemConnector):
            async def connect(self, config):
                pass

            async def disconnect(self):
                pass

            async def read_channel(self, channel_address, timeout=None):
                pass

            async def write_channel(
                self,
                channel_address,
                value,
                timeout=None,
                confirm=None,
            ):
                pass

            async def read_multiple_channels(self, channel_addresses, timeout=None):
                pass

            async def subscribe(self, channel_address, callback):
                pass

            async def unsubscribe(self, subscription_id):
                pass

            async def get_metadata(self, channel_address):
                pass

            async def validate_channel(self, channel_address):
                pass

        # Register it
        ConnectorFactory.register_control_system("custom_test", CustomConnector)

        # Check it's in the list
        assert "custom_test" in ConnectorFactory.list_control_systems()


# Written out here rather than read from ``_BUILTIN_CONTROL_SYSTEMS`` /
# ``_BUILTIN_ARCHIVERS``: a name dropped from a tuple must still produce a case.
_EVERY_BUILTIN_CONTROL_SYSTEM = (
    types.MOCK,
    types.EPICS,
    types.VIRTUAL_ACCELERATOR,
    types.DOOCS,
    types.TANGO,
    types.LIVE_STANDIN,
)
_EVERY_BUILTIN_ARCHIVER = (
    types.MOCK_ARCHIVER,
    types.EPICS_ARCHIVER,
    types.MONGODB_ARCHIVER,
    types.DOOCS_ARCHIVER,
    types.MYA_ARCHIVER,
)


def _registry_for(name: str) -> dict:
    if name in _EVERY_BUILTIN_CONTROL_SYSTEM:
        return ConnectorFactory._control_system_connectors
    return ConnectorFactory._archiver_connectors


class TestBuiltinRegistrationConverges:
    """``register_builtin_connectors()`` heals a partially-populated registry.

    The function short-circuits when every name in ``_BUILTIN_CONTROL_SYSTEMS``
    and ``_BUILTIN_ARCHIVERS`` is already registered. A built-in left out of
    those tuples is invisible to the check: a registry holding every other
    built-in satisfies the early return and the missing entry is never added —
    leaving a project configured for that type unable to build its connector.
    These pin the tuples as the complete list.
    """

    def test_the_tuples_list_every_builtin_that_is_registered(self):
        """A first call on an empty registry registers exactly the tuples' names."""
        ConnectorFactory._control_system_connectors.clear()
        ConnectorFactory._archiver_connectors.clear()

        register_builtin_connectors()

        registered = set(ConnectorFactory._control_system_connectors) | set(
            ConnectorFactory._archiver_connectors
        )
        assert registered == set(_BUILTIN_CONTROL_SYSTEMS) | set(_BUILTIN_ARCHIVERS)
        assert registered == set(_EVERY_BUILTIN_CONTROL_SYSTEM) | set(_EVERY_BUILTIN_ARCHIVER)

    @pytest.mark.parametrize(
        "name",
        _EVERY_BUILTIN_CONTROL_SYSTEM + _EVERY_BUILTIN_ARCHIVER,
        ids=str,
    )
    def test_a_registry_missing_only_this_builtin_regains_it(self, name):
        """Every other built-in present, this one missing: the next call adds it."""
        ConnectorFactory._control_system_connectors.clear()
        ConnectorFactory._archiver_connectors.clear()
        register_builtin_connectors()
        registry = _registry_for(name)
        expected = registry.pop(name)

        register_builtin_connectors()

        assert registry.get(name) is expected

    def test_registering_twice_replaces_nothing(self):
        """An existing registration under a built-in name survives a second call.

        Re-registering the same class would look identical, so the entries are
        swapped for sentinels first: only a call that skips present names
        leaves them in place.
        """
        register_builtin_connectors()

        class SentinelControlSystem:
            pass

        class SentinelArchiver:
            pass

        ConnectorFactory._control_system_connectors[types.LIVE_STANDIN] = SentinelControlSystem
        ConnectorFactory._archiver_connectors[types.MONGODB_ARCHIVER] = SentinelArchiver
        # Force the call past its early return so the per-name guard is what runs.
        del ConnectorFactory._control_system_connectors[types.EPICS]

        register_builtin_connectors()

        assert (
            ConnectorFactory._control_system_connectors[types.LIVE_STANDIN] is SentinelControlSystem
        )
        assert ConnectorFactory._archiver_connectors[types.MONGODB_ARCHIVER] is SentinelArchiver
        assert types.EPICS in ConnectorFactory._control_system_connectors


class TestBuiltinArchiverRegistration:
    """The factory's archiver list agrees with the framework registry's."""

    def test_factory_builtins_and_framework_registry_agree_on_archivers(self):
        """The factory's built-in archiver names match the framework registry's.

        Two independent lists name the shipped archivers; a name in one and not
        the other means a connector the registry advertises but the factory
        cannot construct, or the reverse.
        """
        from osprey.connectors.factory import _BUILTIN_ARCHIVERS
        from osprey.registry.builtins import FrameworkRegistryProvider

        registry_archivers = {
            reg.name
            for reg in FrameworkRegistryProvider().get_registry_config().connectors
            if reg.connector_type == "archiver"
        }
        assert set(_BUILTIN_ARCHIVERS) == registry_archivers


class TestLiveStandinRegistration:
    """The live stand-in is a control target of its own, not a mode of ``epics``.

    It is served by ``EPICSConnector`` — a stand-in is a soft IOC, so Channel
    Access is what reaches it — but it registers under its own key. That key is
    what the factory stamps onto the instance as ``_connector_type``, and the
    stamp is what selects both the connector block the instance is configured
    from and the write posture read out of it. Sharing the ``epics`` key would
    hand the stand-in the facility's authored block and the facility's arming.
    """

    @pytest.mark.asyncio
    async def test_live_standin_builds_an_epics_connector_stamped_with_its_own_type(self):
        """Building ``live_standin`` yields an EPICS connector stamped ``live_standin``.

        The stamp is the assertion that matters: ``_connector_type`` comes from
        the config's type key and never from the class, so per-type posture
        resolves to ``control_system.connector.live_standin.writes_enabled``
        rather than to the ``epics`` block's.
        """
        from osprey.connectors import types
        from osprey.connectors.control_system.epics_connector import EPICSConnector
        from osprey.connectors.factory import register_builtin_connectors

        register_builtin_connectors()

        # No gateways: connect() then touches no EPICS_* environment variable
        # and opens no CA context, so nothing here reaches a network.
        config = {
            "type": types.LIVE_STANDIN,
            "connector": {types.LIVE_STANDIN: {"timeout": 1.0}},
        }

        connector = await ConnectorFactory.create_control_system_connector(config)

        assert isinstance(connector, EPICSConnector)
        assert connector._connector_type == types.LIVE_STANDIN
        assert connector._timeout == 1.0

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_epics_target_is_still_stamped_epics(self):
        """The shared class does not blur the two keys together.

        Both types build the same class, so a stamp derived from the class
        instead of the key would look right on one of them and be wrong on the
        other. Pinning the sibling keeps that failure visible.
        """
        from osprey.connectors import types
        from osprey.connectors.factory import register_builtin_connectors

        register_builtin_connectors()

        connector = await ConnectorFactory.create_control_system_connector(
            {"type": types.EPICS, "connector": {types.EPICS: {"timeout": 1.0}}}
        )

        assert connector._connector_type == types.EPICS

        await connector.disconnect()


class TestArchiverTypeResolution:
    """How ``create_archiver_connector`` turns ``archiver.type`` into a class."""

    @pytest.mark.parametrize(
        ("connector_type", "message"),
        [
            pytest.param(
                "osprey_no_such_package.archivers.Thing",
                "Could not import connector module 'osprey_no_such_package.archivers': "
                "No module named 'osprey_no_such_package'",
                id="module-not-importable",
            ),
            pytest.param(
                "osprey.connectors.archiver.mock_archiver_connector.NoSuchArchiver",
                "Module 'osprey.connectors.archiver.mock_archiver_connector' has no class "
                "'NoSuchArchiver'",
                id="class-missing",
            ),
            pytest.param(
                "hdf5_archiver",
                "Unknown archiver type: 'hdf5_archiver'. Available types: ['mock_archiver']. "
                "Use a dotted module path for custom connectors.",
                id="unknown-name-lists-available",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_an_unusable_archiver_type_is_refused(self, connector_type, message):
        with pytest.raises(ValueError) as caught:
            await ConnectorFactory.create_archiver_connector({"type": connector_type})

        assert message in str(caught.value)

    @pytest.mark.asyncio
    async def test_a_dotted_archiver_path_is_imported_and_remembered(self):
        dotted = "osprey.connectors.archiver.mock_archiver_connector.MockArchiverConnector"

        connector = await ConnectorFactory.create_archiver_connector({"type": dotted})

        assert isinstance(connector, MockArchiverConnector)
        assert ConnectorFactory._archiver_connectors[dotted] is MockArchiverConnector
        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_an_unloadable_config_falls_back_to_the_mock_archiver_with_a_warning(
        self, caplog
    ):
        def unreadable(*args, **kwargs):
            raise RuntimeError("config.yml is unreadable")

        with patch("osprey_connectors.config.get_config_value", unreadable):
            connector = await ConnectorFactory.create_archiver_connector(None)

        assert isinstance(connector, MockArchiverConnector)
        assert "Could not load config: config.yml is unreadable, using defaults" in caplog.text
        await connector.disconnect()
