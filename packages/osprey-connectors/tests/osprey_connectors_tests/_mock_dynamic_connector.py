"""Mock connector for testing dynamic import in ConnectorFactory."""

from osprey_connectors.archiver._base import ArchiverConnector
from osprey_connectors.control_system._base import ControlSystemConnector


class MockDynamicConnector(ControlSystemConnector):
    """Minimal connector used by test_dynamic_connector.py."""

    async def connect(self, config=None):
        pass

    async def disconnect(self):
        pass

    async def read_channel(self, address, *, timeout_s=None):  # noqa: ARG002 - the control-system connector interface fixes this signature
        from datetime import UTC, datetime

        from osprey_connectors.control_system._base import ChannelValue

        return ChannelValue(value=42, sampled_at=datetime.now(UTC))

    async def write_channel(self, address, value, *, timeout_s=None, confirm=False):  # noqa: ARG002 - the control-system connector interface fixes this signature
        """Declares its own ``confirm`` default, so a forwarded ``None`` would show."""
        from osprey_connectors.control_system._base import ChannelWriteResult, WriteOutcome

        return ChannelWriteResult(
            address=address,
            value_written=value,
            outcome=WriteOutcome.CONFIRMED if confirm else WriteOutcome.UNREQUESTED,
            observed_value=value if confirm else None,
        )

    async def subscribe(self, address, callback):  # noqa: ARG002 - the control-system connector interface fixes this signature
        return "sub-1"

    async def unsubscribe(self, subscription_id):
        pass

    async def get_metadata(self, address):  # noqa: ARG002 - the control-system connector interface fixes this signature
        from osprey_connectors.control_system._base import ChannelMetadata

        return ChannelMetadata()

    async def validate_channel(self, address):  # noqa: ARG002 - the control-system connector interface fixes this signature
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
        *,
        start_at,
        end_at,
        precision_ms=1000,
        timeout_s=None,
        processing="raw",
    ):
        raise NotImplementedError

    async def get_metadata(self, channel):
        raise NotImplementedError

    async def check_availability(self, channels):
        raise NotImplementedError
