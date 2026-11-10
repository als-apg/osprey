"""The refresh vocabulary: ``pass``, ``periodic``, ``fast`` and ``event``.

The wiring check accepts an engine description carrying any refresh word the
facility schema states, and still refuses a word it does not.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.facility.errors import FacilityBuildError
from osprey.facility.schema import core
from osprey.facility.wiring import _check_description
from osprey_connectors.simulation.view import REFRESH

MODEL: dict[str, Any] = {"name": "ring", "engine": "test"}
RECORD: dict[str, Any] = {"address": "SR:DCCT", "direction": "read", "element": "dcct"}


def _description(refresh: Any) -> dict[str, Any]:
    return {"engine_role": "monitor", "plane": None, "refresh": refresh}


def test_the_vocabulary_is_the_four_words() -> None:
    assert set(REFRESH) == {"pass", "periodic", "fast", "event"}
    assert {member.value for member in core.RefreshEnum} == set(REFRESH)


@pytest.mark.parametrize("refresh", ["pass", "periodic", "fast", "event"])
def test_a_description_with_each_refresh_is_accepted(refresh: str) -> None:
    _check_description(MODEL, RECORD, _description(refresh), frozenset())


@pytest.mark.parametrize("refresh", ["slow", "Fast", "", 1])
def test_an_unknown_refresh_is_refused(refresh: Any) -> None:
    with pytest.raises(FacilityBuildError) as caught:
        _check_description(MODEL, RECORD, _description(refresh), frozenset())
    assert "refresh" in str(caught.value)
