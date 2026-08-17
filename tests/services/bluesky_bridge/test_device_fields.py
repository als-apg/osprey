"""Direct tests for the read-only device-field predicate.

The predicate's two consumers (`plan_validation._collect_device_names` and
`document_plane._swept_device_lists`) are covered by their own suites, but only
against the field names those suites happen to use. This module pins the rule
itself, spelling by spelling, so a clause cannot be dropped without a red test.

That matters most for ``readback``: it is the spelling OSPREY's own shipped
plans move to, and until they do, no consumer test exercises that clause at all.
"""

from __future__ import annotations

import pytest

from osprey.services.bluesky_bridge.device_fields import is_read_only_device_field


@pytest.mark.parametrize(
    "key",
    [
        # OSPREY's own vocabulary — the spelling the shipped plans use.
        "readbacks",
        "readback",
        # Bluesky-conventional spellings a facility-authored plan may use. The
        # catalog is open, so all of these must keep working.
        "detectors",
        "detector",
        "dets",
        "readables",
        "readable",
        # Case is not part of the rule.
        "Readbacks",
        "DETECTORS",
        "Dets",
    ],
)
def test_read_only_field_names(key):
    assert is_read_only_device_field(key) is True


@pytest.mark.parametrize(
    "key",
    [
        # Driven fields: the devices a plan steps through.
        "correctors",
        "setpoints",
        "setpoint",
        "axes",
        "motors",
        # Non-device parameters that share the mapping with device fields.
        "num_points",
        "num",
        "span_a",
        "snake_axes",
        # `det` alone is not the read-only plural; only exact `dets` is special
        # cased, because a bare three-letter prefix is too eager to claim.
        "det",
        "detached",
        # No enclosing field name at all, and the degenerate empty one.
        None,
        "",
    ],
)
def test_fields_that_are_not_read_only(key):
    assert is_read_only_device_field(key) is False


def test_every_clause_of_the_rule_is_load_bearing():
    """Each alternative in the predicate claims a name no other one claims.

    A clause silently dropped in a refactor would still leave the other three
    passing their own cases; this asserts the four are disjoint on these
    witnesses, so losing any one turns a witness red.
    """
    witnesses = {
        "dets": "exact-match clause",
        "detectors": "'detect' substring clause",
        "readbacks": "'readback' substring clause",
        "readables": "'readable' substring clause",
    }
    for name, clause in witnesses.items():
        assert is_read_only_device_field(name), clause

    # The witnesses really are distinct: no witness is claimed by more than one
    # clause, so none of them can stand in for another.
    assert "detect" not in "dets"
    assert "readback" not in "readables"
    assert "readable" not in "readbacks"
