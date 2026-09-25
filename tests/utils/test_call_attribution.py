"""The two call-attribution headers: accepted shapes, refused shapes, absence."""

from __future__ import annotations

import logging

import pytest

from osprey.utils.call_attribution import (
    CONVERSATION_HEADER,
    TOOL_USE_HEADER,
    attribution_from_headers,
)


def test_the_header_names() -> None:
    assert CONVERSATION_HEADER == "X-Osprey-Conversation"
    assert TOOL_USE_HEADER == "X-Osprey-Tool-Use-Id"


@pytest.mark.parametrize(
    "value",
    ["toolu_01ABCdef", "3f2a1c9e-0b1d-4c2e-9a7f-1234567890ab", "kernel:abc.def", "x" * 128],
)
def test_accepted_shapes_are_kept_verbatim(value: str) -> None:
    assert attribution_from_headers(value, value) == (value, value)


@pytest.mark.parametrize(
    ("value", "shape"),
    [
        ("", "empty"),
        ("x" * 129, "longer than"),
        ("a/b", "outside the accepted charset"),
        ("<script>", "outside the accepted charset"),
        ("has space", "outside the accepted charset"),
    ],
)
def test_refused_shapes_return_none_and_log_the_shape_only(
    value: str, shape: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="osprey.utils.call_attribution"):
        assert attribution_from_headers(value, None) == (None, None)
    (message,) = [record.getMessage() for record in caplog.records]
    assert shape in message
    assert CONVERSATION_HEADER in message
    if value:
        assert value not in message


def test_each_header_is_judged_on_its_own(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="osprey.utils.call_attribution"):
        assert attribution_from_headers("conv-1", "bad/id") == ("conv-1", None)
    (message,) = [record.getMessage() for record in caplog.records]
    assert TOOL_USE_HEADER in message


def test_absent_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="osprey.utils.call_attribution"):
        assert attribution_from_headers(None, None) == (None, None)
    assert caplog.records == []
