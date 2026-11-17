"""Mock connector for testing dynamic import in ConnectorFactory."""

from osprey_connectors.archiver._base import ArchiverConnector
from osprey_connectors.control_system._base import ControlSystemConnector


class MockDynamicConnector(ControlSystemConnector):
    """Minimal connector used by test_dynamic_connector.py.

    Every send is recorded with whether it was asked to wait for a
    confirmation, and the channel then holds the value sent.
    """

    def __init__(self) -> None:
        self.sends: list[tuple[str, object, bool]] = []
        self._held: dict[str, object] = {}

    async def connect(self, config=None):
        pass

    async def disconnect(self):
        pass

    async def read_channel(self, address, *, timeout_s=None):  # noqa: ARG002 - the control-system connector interface fixes this signature
        from datetime import UTC, datetime

        from osprey_connectors.control_system._base import ChannelValue

        return ChannelValue(value=self._held.get(address, 42), sampled_at=datetime.now(UTC))

    async def _send(self, address, value, *, timeout_s, wait):  # noqa: ARG002 - the send hook fixes this signature
        from osprey_connectors.control_system._base import SendReport

        self.sends.append((address, value, wait))
        self._held[address] = value
        return SendReport()

    async def read_multiple_channels(self, addresses, *, timeout_s=None):  # noqa: ARG002 - the control-system connector interface fixes this signature
        return {}

    async def subscribe(self, address, callback, *, on_connection=None):  # noqa: ARG002 - the control-system connector interface fixes this signature
        return "sub-1"

    async def unsubscribe(self, subscription_id):
        pass


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
