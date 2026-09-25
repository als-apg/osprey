"""Mock connector for testing dynamic import in ConnectorFactory."""

from osprey.connectors.archiver.base import ArchiverConnector
from osprey.connectors.control_system.base import ControlSystemConnector


class MockDynamicConnector(ControlSystemConnector):
    """Minimal connector used by test_dynamic_connector.py."""

    async def connect(self, config=None):
        pass

    async def disconnect(self):
        pass

    async def read_channel(self, channel_address, timeout=None):  # noqa: ARG002 - the control-system connector interface fixes this signature
        from datetime import datetime

        from osprey.connectors.control_system.base import ChannelValue

        return ChannelValue(value=42, timestamp=datetime.now())

    async def write_channel(self, channel_address, value, timeout=None, confirm=False):  # noqa: ARG002 - the control-system connector interface fixes this signature
        """Declares its own ``confirm`` default, so a forwarded ``None`` would show."""
        from osprey.connectors.control_system.base import ChannelWriteResult, WriteOutcome

        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=WriteOutcome.CONFIRMED if confirm else WriteOutcome.UNREQUESTED,
            observed_value=value if confirm else None,
        )

    async def read_multiple_channels(self, channel_addresses, timeout=None):  # noqa: ARG002 - the control-system connector interface fixes this signature
        return {}

    async def subscribe(self, channel_address, callback):  # noqa: ARG002 - the control-system connector interface fixes this signature
        return "sub-1"

    async def unsubscribe(self, subscription_id):
        pass

    async def get_metadata(self, channel_address):  # noqa: ARG002 - the control-system connector interface fixes this signature
        from osprey.connectors.control_system.base import ChannelMetadata

        return ChannelMetadata()

    async def validate_channel(self, channel_address):  # noqa: ARG002 - the control-system connector interface fixes this signature
        return True


class RecordingArchiver(ArchiverConnector):
    """Archiver that keeps the settings block ``connect()`` was handed."""

    settings: dict | None = None

    async def connect(self, config):
        type(self).settings = config

    async def disconnect(self):
        raise NotImplementedError

    async def get_data(
        self,
        channels,
        start_date,
        end_date,
        precision_ms=1000,
        timeout=None,
        processing="raw",
    ):
        raise NotImplementedError

    async def get_metadata(self, channel):
        raise NotImplementedError

    async def check_availability(self, channels):
        raise NotImplementedError
