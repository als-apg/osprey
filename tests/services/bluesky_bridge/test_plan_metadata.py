"""Coverage for the plan-metadata model, its parser, and `PlanSpec`'s new
`metadata`/`provenance` fields (task 1.1).

Pure pydantic — no bluesky import, so this runs in the fast import-clean lane
alongside the rest of `plan_types.py`'s own tests.
"""

from __future__ import annotations

import types

import pytest
from pydantic import BaseModel

from osprey.services.bluesky_bridge.plan_metadata import (
    PlanMetadata,
    PlanMetadataError,
    parse_plan_metadata,
    parse_plan_metadata_dict,
)
from osprey.services.bluesky_bridge.plan_types import PlanSpec

_WELL_FORMED = {
    "name": "orm",
    "description": "Sweep each corrector, reading all BPMs at every point.",
    "category": "accelerator",
    "required_devices": ["SR:MAG:HCM:01:CURRENT:SP", "SR:DIAG:BPM:01:X"],
    "writes": True,
}


class _Params(BaseModel):
    pass


def test_well_formed_dict_parses_into_plan_metadata_with_all_fields() -> None:
    metadata = parse_plan_metadata_dict(_WELL_FORMED, source="test")

    assert metadata.name == "orm"
    assert metadata.description == _WELL_FORMED["description"]
    assert metadata.category == "accelerator"
    assert metadata.required_devices == _WELL_FORMED["required_devices"]
    assert metadata.writes is True


def test_well_formed_module_parses_via_parse_plan_metadata() -> None:
    module = types.SimpleNamespace(PLAN_METADATA=dict(_WELL_FORMED), __name__="a_plan_module")

    metadata = parse_plan_metadata(module)

    assert metadata == PlanMetadata(**_WELL_FORMED)


def test_module_without_plan_metadata_attribute_raises_typed_error() -> None:
    module = types.SimpleNamespace(__name__="no_metadata_module")

    with pytest.raises(PlanMetadataError, match="no_metadata_module"):
        parse_plan_metadata(module)


def test_plan_metadata_not_a_dict_raises_typed_error() -> None:
    module = types.SimpleNamespace(PLAN_METADATA=["not", "a", "dict"], __name__="bad_module")

    with pytest.raises(PlanMetadataError, match="bad_module"):
        parse_plan_metadata(module)


def test_source_label_appears_in_error_message() -> None:
    with pytest.raises(PlanMetadataError, match="my/facility/plan.py"):
        parse_plan_metadata_dict({}, source="my/facility/plan.py")


@pytest.mark.parametrize("missing_field", sorted(_WELL_FORMED))
def test_missing_required_field_raises_typed_error_naming_the_field(missing_field: str) -> None:
    raw = {k: v for k, v in _WELL_FORMED.items() if k != missing_field}

    with pytest.raises(PlanMetadataError, match=missing_field) as excinfo:
        parse_plan_metadata_dict(raw, source="test")

    assert isinstance(excinfo.value, PlanMetadataError)


def test_wrong_type_required_devices_raises_typed_error() -> None:
    raw = {**_WELL_FORMED, "required_devices": "not-a-list"}

    with pytest.raises(PlanMetadataError, match="required_devices"):
        parse_plan_metadata_dict(raw, source="test")


def test_wrong_type_writes_raises_typed_error() -> None:
    raw = {**_WELL_FORMED, "writes": "not-a-bool"}

    with pytest.raises(PlanMetadataError, match="writes"):
        parse_plan_metadata_dict(raw, source="test")


def test_plan_spec_to_dict_includes_metadata_when_set() -> None:
    metadata = PlanMetadata(**_WELL_FORMED)
    spec = PlanSpec(
        name="orm",
        plan=lambda devices, params: None,
        schema=_Params,
        description="A plan with metadata.",
        metadata=metadata,
        provenance="facility",
    )

    payload = spec.to_dict()

    assert payload["metadata"] == metadata.model_dump()
    assert payload["provenance"] == "facility"


def test_plan_spec_to_dict_metadata_is_none_when_unset() -> None:
    spec = PlanSpec(
        name="bare",
        plan=lambda devices, params: None,
        schema=_Params,
        description="No metadata attached.",
    )

    payload = spec.to_dict()

    assert payload["metadata"] is None


def test_plan_spec_old_style_construction_still_works_and_defaults_provenance() -> None:
    """Regression: `PlanSpec`'s old-style construction (name/plan/schema/
    description only, no metadata/provenance kwargs) must keep working
    unchanged."""
    spec = PlanSpec(
        name="count",
        plan=lambda devices, params: None,
        schema=_Params,
        description="Read detectors N times with no motor motion.",
    )

    assert spec.metadata is None
    assert spec.provenance == "shipped"
    assert spec.to_dict()["provenance"] == "shipped"
