"""Mock connector for testing dynamic import in ConnectorFactory."""

from osprey.connectors.control_system.base import ControlSystemConnector


class MockDynamicConnector(ControlSystemConnector):
    """Minimal connector used by test_dynamic_connector.py."""

    async def connect(self, config=None):
        pass

    async def disconnect(self):
        pass

    async def read_channel(self, channel_address, timeout=None):
        from datetime import datetime

        from osprey.connectors.control_system.base import ChannelValue

        return ChannelValue(value=42, timestamp=datetime.now())

    async def write_channel(
        self, channel_address, value, timeout=None, verification_level="callback", tolerance=None
    ):
        from osprey.connectors.control_system.base import ChannelWriteResult, WriteVerification

        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            success=True,
            verification=WriteVerification(level=verification_level, verified=True),
        )

    async def read_multiple_channels(self, channel_addresses, timeout=None):
        return {}

    async def subscribe(self, channel_address, callback):
        return "sub-1"

    async def unsubscribe(self, subscription_id):
        pass

    async def get_metadata(self, channel_address):
        from osprey.connectors.control_system.base import ChannelMetadata

        return ChannelMetadata()

    async def validate_channel(self, channel_address):
        return True
