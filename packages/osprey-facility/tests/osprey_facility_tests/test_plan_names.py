"""The plan-name predicate: which names a run can record."""

from __future__ import annotations

import pytest

from osprey_facility.plan_names import is_plan_name


@pytest.mark.parametrize(
    ("name", "recordable"),
    [
        ("A:B", True),
        ("A_B-1", True),
        ("A:B.RBV", False),
        ("ES1/MCA01", False),
        ("A.B/C", False),
    ],
)
def test_a_name_holding_a_dot_or_slash_is_no_plan_name(name: str, recordable: bool) -> None:
    assert is_plan_name(name) is recordable
