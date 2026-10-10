"""Every env-var placeholder left in a config value, wherever it sits.

:func:`~osprey_connectors.config.unresolved_placeholders` is the one check both
supervisors of connector-host children run on a target's connector block
before anything is spawned, so a placeholder is never mistaken for an endpoint.
"""

import pytest

from osprey_connectors.config import is_unresolved_placeholder, unresolved_placeholders


@pytest.mark.parametrize(
    ("value", "found"),
    [
        pytest.param("${EPICS_TESTING_PORT}", ["${EPICS_TESTING_PORT}"], id="braced"),
        pytest.param("$EPICS_TESTING_PORT", ["$EPICS_TESTING_PORT"], id="bare"),
        pytest.param("gw-${SITE}.example.org:$PORT", ["${SITE}", "$PORT"], id="embedded"),
        pytest.param("gw.example.org", [], id="resolved-text"),
        pytest.param(5064, [], id="not-a-string"),
        pytest.param(None, [], id="none"),
    ],
)
def test_a_scalar_names_every_placeholder_it_carries(value, found):
    assert unresolved_placeholders(value) == found


def test_placeholders_nested_in_mappings_lists_and_tuples_are_found_depth_first():
    block = {
        "gateways": {"read_only": {"address": "127.0.0.1", "port": "${EPICS_TESTING_PORT}"}},
        "channels": ["SR:A", ("SR:B", "$CHANNEL_SUFFIX")],
        "timeout_s": 5.0,
    }

    assert unresolved_placeholders(block) == ["${EPICS_TESTING_PORT}", "$CHANNEL_SUFFIX"]


def test_a_fully_resolved_block_carries_none():
    block = {
        "gateways": {"read_only": {"address": "127.0.0.1", "port": 5064}},
        "channels": ["SR:A", ("SR:B",)],
    }

    assert unresolved_placeholders(block) == []


def test_the_lone_placeholder_check_keeps_its_anchored_meaning():
    # The walk finds a reference anywhere in a string; the lone check answers
    # only for a value that is exactly one reference.
    assert unresolved_placeholders("gw-${SITE}") == ["${SITE}"]
    assert is_unresolved_placeholder("gw-${SITE}") is False
    assert is_unresolved_placeholder("${SITE}") is True
