"""Mock connector for testing dynamic import in ConnectorFactory."""

from osprey_connectors.archiver import ChannelArchive
from osprey_connectors.control_system._base import ControlSystemConnector


class MockDynamicConnector(ControlSystemConnector):
    """Minimal connector used by test_dynamic_connector.py.

    Every send is recorded with whether it was asked to wait for a
    confirmation, and the channel then holds the value sent.
    """

    def __init__(self) -> None:
        self.sends: list[tuple[str, object, bool]] = []
        self._held: dict[str, object] = {}

    async def _open(self, config=None):
        pass

    async def disconnect(self):
        pass

    async def read_channel(self, address, *, timeout_s=None):  # noqa: ARG002 - the control-system connector interface fixes this signature
        from datetime import UTC, datetime

        from osprey_connectors.control_system._base import ChannelValue

        return ChannelValue(
            address=address, value=self._held.get(address, 42), sampled_at=datetime.now(UTC)
        )

    async def _send(self, address, value, *, timeout_s, wait):  # noqa: ARG002 - the send hook fixes this signature
        from osprey_connectors.control_system._base import SendReport

        self.sends.append((address, value, wait))
        self._held[address] = value
        return SendReport()

    async def subscribe(self, address, callback, *, on_connection=None):  # noqa: ARG002 - the control-system connector interface fixes this signature
        return "sub-1"

    async def unsubscribe(self, subscription_id):
        pass


class RecordingArchiver(ChannelArchive):
    """Archive that keeps the settings block its spec was built with."""

    settings: dict | None = None

    def __init__(self, spec, *, project_dir=None, **seams):
        super().__init__(spec, project_dir=project_dir, **seams)
        type(self).settings = dict(spec.settings)

    async def history(
        self,
        addresses,
        *,
        start_at,
        end_at,
        precision_ms=1000,
        processing="raw",
        timeout_s=None,
    ):
        raise NotImplementedError

    async def coverage(self, addresses, *, timeout_s=None):
        raise NotImplementedError
