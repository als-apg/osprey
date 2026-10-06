"""The active-scenarios state: the set a reader serves."""

from __future__ import annotations

import pytest

from osprey_connectors.simulation.state import Overlap, composed_set

VIEW = {
    "nominal": set(),
    "burst": {"SR:VAC:PRESSURE"},
    "leak": {"SR:VAC:PRESSURE"},
    "thermal": {"SR:RF:TEMP"},
}


def test_a_composing_set_is_served_as_it_is() -> None:
    assert composed_set(VIEW, ["nominal", "burst", "thermal"]) == (
        ["nominal", "burst", "thermal"],
        [],
    )


def test_a_set_writing_one_target_twice_is_served_as_nominal_alone() -> None:
    assert composed_set(VIEW, ["nominal", "burst", "leak"]) == (
        ["nominal"],
        [Overlap(target="SR:VAC:PRESSURE", first="burst", second="leak")],
    )


def test_an_unknown_name_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown scenarios \\['nope'\\]"):
        composed_set(VIEW, ["nominal", "nope"])
