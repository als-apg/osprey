"""One channel's key, as each connector spells it.

A session admits one write per channel at a time and compares these keys. What
counts as one channel is protocol knowledge, so each connector answers: EPICS
folds ``ca://X``, ``X.VAL`` and ``X`` together, and every other connector only
strips whitespace.
"""

from __future__ import annotations

import pytest

from osprey_connectors.control_system._epics_connector import EPICSConnector
from osprey_connectors.control_system._va_connector import VirtualAcceleratorConnector
from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors_tests._fake_connector import FakeConnector


@pytest.mark.parametrize("cls", [EPICSConnector, VirtualAcceleratorConnector])
class TestEpics:
    def test_the_scheme_and_the_value_field_are_one_channel(self, cls) -> None:
        connector = cls()
        keys = {connector.channel_key(a) for a in ("ca://X", "X", " X.VAL ", "ca://X.VAL")}
        assert keys == {"X"}

    def test_another_field_is_another_channel(self, cls) -> None:
        assert cls().channel_key("X.DESC") == "X.DESC"


@pytest.mark.parametrize("cls", [VAInProcessConnector, FakeConnector])
class TestEveryOtherConnector:
    def test_whitespace_is_stripped(self, cls) -> None:
        assert cls().channel_key("  SR:QF:01:SP ") == "SR:QF:01:SP"

    def test_the_address_keeps_its_suffix_and_scheme(self, cls) -> None:
        connector = cls()
        assert connector.channel_key("SR:QF:01:SP.VAL") == "SR:QF:01:SP.VAL"
        assert connector.channel_key("ca://SR:QF:01:SP") == "ca://SR:QF:01:SP"
