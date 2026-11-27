"""The plan-name pattern is event-model's own grammar for a descriptor's keys."""

from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from osprey_facility.plan_names import PLAN_NAME_PATTERN, is_plan_name


def _descriptor_schema() -> dict[str, Any]:
    spec = importlib.util.find_spec("event_model")
    assert spec is not None and spec.submodule_search_locations
    path = Path(spec.submodule_search_locations[0]) / "schemas" / "event_descriptor.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_pattern_is_event_models_own() -> None:
    """The descriptor schema states exactly one key pattern, and it is this one."""
    assert list(_descriptor_schema()["patternProperties"]) == [PLAN_NAME_PATTERN]


@pytest.mark.parametrize("name", ["A:B", "A:B.RBV", "ES1/MCA01", "A_B-1", "A.B/C"])
def test_the_predicate_agrees_with_the_descriptor_schema(name: str) -> None:
    descriptor = {
        "uid": str(uuid.uuid4()),
        "run_start": str(uuid.uuid4()),
        "time": 0.0,
        "data_keys": {name: {"source": "x", "dtype": "number", "shape": []}},
        "object_keys": {name: [name]},
    }
    try:
        jsonschema.validate(descriptor, _descriptor_schema())
    except jsonschema.ValidationError:
        recorded = False
    else:
        recorded = True

    assert is_plan_name(name) is recorded
