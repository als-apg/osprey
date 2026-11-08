"""``osprey.apps.sdk.caller_kind``: who the terminal proxy says is calling."""

from __future__ import annotations

import pytest

from osprey.apps.sdk import CALLER_HEADER, caller_from_header, caller_kind


@pytest.mark.parametrize("kind", ["agent", "panel"])
def test_the_proxy_stamped_kind_is_read(kind):
    assert caller_kind({CALLER_HEADER: kind}) == kind


def test_the_header_name_is_matched_case_insensitively():
    assert caller_kind({"x-osprey-caller": "agent"}) == "agent"


def test_an_absent_header_raises_naming_it():
    with pytest.raises(ValueError, match=CALLER_HEADER):
        caller_kind({})


@pytest.mark.parametrize("value", ["operator", "Agent", "", "agent "])
def test_an_unknown_value_raises_naming_the_header(value):
    with pytest.raises(ValueError, match=CALLER_HEADER):
        caller_kind({CALLER_HEADER: value})


@pytest.mark.parametrize(("value", "kind"), [("agent", "agent"), ("panel", "panel")])
def test_caller_from_header_reads_the_two_values(value, kind):
    assert caller_from_header(value) == kind


@pytest.mark.parametrize("value", [None, "", "browser", "AGENT"])
def test_caller_from_header_refuses_anything_else(value):
    assert caller_from_header(value) is None
