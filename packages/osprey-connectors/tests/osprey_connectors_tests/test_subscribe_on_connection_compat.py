"""``subscribe(..., on_connection=...)`` reaches connectors written before the keyword.

A facility's own connector may define ``subscribe(self, channel_address,
callback)``. The base class wraps it, so every framework caller can pass
``on_connection`` to any connector: the wrapper reports ``True`` before the
first value and passes the keyword through untouched to a ``subscribe`` that
takes it.
"""

import inspect
import warnings
from datetime import datetime

import pytest

from osprey_connectors.control_system._base import ChannelValue, ControlSystemConnector
from osprey_connectors.control_system._doocs_connector import DOOCSConnector
from osprey_connectors.control_system._epics_connector import EPICSConnector
from osprey_connectors.control_system._tango_connector import TangoConnector
from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors_tests._mock_dynamic_connector import MockDynamicConnector


def _value(v: float) -> ChannelValue:
    return ChannelValue(value=v, timestamp=datetime.now())


class _Stub(ControlSystemConnector):
    """The abstract surface, with nothing behind it."""

    async def connect(self, config=None):
        pass

    async def disconnect(self):
        pass

    async def read_channel(self, channel_address, timeout=None):  # noqa: ARG002 - interface signature
        return _value(0.0)

    async def write_channel(self, channel_address, value, timeout=None, confirm=None):
        raise NotImplementedError

    async def read_multiple_channels(self, channel_addresses, timeout=None):  # noqa: ARG002 - interface signature
        return {}

    async def unsubscribe(self, subscription_id):
        pass

    async def get_metadata(self, channel_address):
        raise NotImplementedError

    async def validate_channel(self, channel_address):  # noqa: ARG002 - interface signature
        return True


class _Legacy(_Stub):
    """A ``subscribe`` from before ``on_connection``: delivers two values inline."""

    async def subscribe(self, channel_address, callback):  # noqa: ARG002 - legacy signature
        callback(_value(1.0))
        callback(_value(2.0))
        return "legacy-1"


class _Current(_Stub):
    """A ``subscribe`` that reports connection state itself."""

    def __init__(self) -> None:
        self.received: list = []

    async def subscribe(self, channel_address, callback, *, on_connection=None):  # noqa: ARG002 - interface signature
        self.received.append(on_connection)
        if on_connection is not None:
            on_connection(True)
        callback(_value(1.0))
        return "current-1"


class _LegacyWithExtra(_Stub):
    """A legacy ``subscribe`` with an extra positional parameter of its own."""

    async def subscribe(self, channel_address, callback, period=1.0):
        callback(_value(period))
        return f"extra-{channel_address}"


class TestLegacySignature:
    async def test_on_connection_fires_once_before_the_first_value(self):
        events: list = []
        sub = await _Legacy().subscribe(
            "SR:CH",
            lambda v: events.append(("value", v.value)),
            on_connection=lambda up: events.append(("up", up)),
        )
        assert sub == "legacy-1"
        assert events == [("up", True), ("value", 1.0), ("value", 2.0)]

    async def test_without_on_connection_the_original_runs_untouched(self):
        values: list = []
        assert await _Legacy().subscribe("SR:CH", lambda v: values.append(v.value)) == "legacy-1"
        assert values == [1.0, 2.0]

    async def test_a_channel_that_never_delivers_reports_nothing(self):
        reports: list = []
        sub = await MockDynamicConnector().subscribe(
            "SR:CH", lambda v: None, on_connection=reports.append
        )
        assert sub == "sub-1"
        assert reports == []

    async def test_extra_arguments_still_reach_the_original(self):
        events: list = []
        sub = await _LegacyWithExtra().subscribe(
            "SR:CH", lambda v: events.append(v.value), 3.0, on_connection=events.append
        )
        assert sub == "extra-SR:CH"
        assert events == [True, 3.0]

    def test_the_wrapped_signature_offers_on_connection(self):
        params = inspect.signature(_Legacy.subscribe).parameters
        assert list(params)[:3] == ["self", "channel_address", "callback"]
        assert params["on_connection"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["on_connection"].default is None


class TestCurrentSignature:
    async def test_on_connection_is_passed_through_not_double_called(self):
        connector = _Current()
        reports: list = []
        await connector.subscribe("SR:CH", lambda v: None, on_connection=reports.append)
        assert connector.received == [reports.append]
        assert reports == [True]

    def test_a_subscribe_that_takes_the_keyword_is_not_wrapped(self):
        assert _Current.subscribe is _Current.__dict__["subscribe"]
        assert not hasattr(_Current.subscribe, "__wrapped__")


class TestDeprecation:
    def test_defining_a_legacy_subscribe_warns_once_naming_the_class(self):
        with pytest.warns(DeprecationWarning, match="_LateLegacy.subscribe") as caught:

            class _LateLegacy(_Stub):
                async def subscribe(self, channel_address, callback):  # noqa: ARG002 - legacy signature
                    return "late-1"

        assert len(caught) == 1

    def test_a_current_subscribe_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            class _LateCurrent(_Stub):
                async def subscribe(self, channel_address, callback, *, on_connection=None):  # noqa: ARG002 - interface signature
                    return "late-2"


def test_built_in_connectors_keep_their_own_subscribe():
    for cls in (EPICSConnector, VAInProcessConnector, DOOCSConnector, TangoConnector):
        assert not hasattr(cls.subscribe, "__wrapped__"), cls.__name__
