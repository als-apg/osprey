"""Coverage for the plan-metadata model, its parser, and `PlanSpec`'s
`metadata`/`provenance` fields.

The model is a closed contract — three required fields (`name`, `description`,
`writes`) and two optional presentation fields — and nothing else: which
channels a plan touches is read off its role-typed parameter fields, not
re-declared here.

Pure pydantic — no bluesky import, so this runs in the fast import-clean lane
alongside the rest of `plan_types.py`'s own tests.
"""

from __future__ import annotations

import logging
import types
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

import osprey
from osprey.services.bluesky_bridge import plan_metadata
from osprey.services.bluesky_bridge.plan_metadata import (
    PlanMetadata,
    PlanMetadataError,
    parse_plan_metadata,
    parse_plan_metadata_dict,
    summary_parameters,
)
from osprey.services.bluesky_bridge.plan_types import PlanSpec

_WELL_FORMED = {
    "name": "orm",
    "description": "Sweep each corrector, reading all BPMs at every point.",
    "writes": True,
}


class _Params(BaseModel):
    pass


def test_well_formed_dict_parses_into_plan_metadata_with_all_fields() -> None:
    metadata = parse_plan_metadata_dict(_WELL_FORMED, source="test")

    assert metadata.name == "orm"
    assert metadata.description == _WELL_FORMED["description"]
    assert metadata.writes is True


def test_model_declares_three_required_and_two_optional_fields() -> None:
    required = {name for name, field in PlanMetadata.model_fields.items() if field.is_required()}
    optional = set(PlanMetadata.model_fields) - required

    assert required == {"name", "description", "writes"}
    assert optional == {"layout", "summary"}


def test_layout_rows_parse() -> None:
    rows = [["correctors", "readbacks"], ["span", "num"], ["sweep"]]

    metadata = parse_plan_metadata_dict({**_WELL_FORMED, "layout": rows}, source="test")

    assert metadata.layout == rows


def test_layout_defaults_to_none() -> None:
    assert parse_plan_metadata_dict(_WELL_FORMED, source="test").layout is None


def test_layout_refuses_a_repeated_name() -> None:
    raw = {**_WELL_FORMED, "layout": [["span", "num"], ["num"]]}

    with pytest.raises(PlanMetadataError, match="layout") as excinfo:
        parse_plan_metadata_dict(raw, source="test")

    assert "num" in str(excinfo.value)


def test_layout_refuses_an_empty_row() -> None:
    with pytest.raises(PlanMetadataError, match="layout"):
        parse_plan_metadata_dict({**_WELL_FORMED, "layout": [["span"], []]}, source="test")


def test_layout_refuses_an_empty_name() -> None:
    with pytest.raises(PlanMetadataError, match="layout"):
        parse_plan_metadata_dict({**_WELL_FORMED, "layout": [["span", ""]]}, source="test")


def test_summary_parts_parse() -> None:
    parts = ["gap {gap}", "{start} to {stop}", "{num} points", "{{fixed}} {monitors}"]

    metadata = parse_plan_metadata_dict({**_WELL_FORMED, "summary": parts}, source="test")

    assert metadata.summary == parts
    assert summary_parameters(metadata) == {"gap", "start", "stop", "num", "monitors"}


def test_summary_defaults_to_none() -> None:
    metadata = parse_plan_metadata_dict(_WELL_FORMED, source="test")

    assert metadata.summary is None
    assert summary_parameters(metadata) == set()


@pytest.mark.parametrize(
    "part",
    ["", "{a!r}", "{a!}", "{a:>3}", "{a:}", "{a[0]}", "{a.b}", "{}", "{0}", "{a", "a}"],
    ids=[
        "empty-part",
        "conversion",
        "empty-conversion",
        "format-spec",
        "empty-format-spec",
        "index",
        "attribute",
        "empty-placeholder",
        "positional",
        "unclosed",
        "stray-close",
    ],
)
def test_summary_refuses_a_format_spec(part: str) -> None:
    with pytest.raises(PlanMetadataError, match="summary"):
        parse_plan_metadata_dict({**_WELL_FORMED, "summary": ["{num} points", part]}, source="t")


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

    assert "unknown key(s)" not in str(excinfo.value)


def test_wrong_type_writes_raises_typed_error() -> None:
    raw = {**_WELL_FORMED, "writes": "not-a-bool"}

    with pytest.raises(PlanMetadataError, match="writes"):
        parse_plan_metadata_dict(raw, source="test")


@pytest.mark.parametrize(
    "unknown_key, value",
    [
        ("required_devices", ["SR:MAG:HCM:01:CURRENT:SP"]),
        ("category", "accelerator"),
        ("some_invented_key", 17),
    ],
)
def test_unknown_key_is_rejected_and_named_in_the_error(unknown_key: str, value: object) -> None:
    """A plan declaring anything outside the declared metadata fields — including a
    key from an earlier metadata shape — quarantines loudly, naming the key."""
    raw = {**_WELL_FORMED, unknown_key: value}

    with pytest.raises(PlanMetadataError) as excinfo:
        parse_plan_metadata_dict(raw, source="legacy/plan.py")

    message = str(excinfo.value)
    assert f"unknown key(s): {unknown_key}" in message
    assert message.startswith("legacy/plan.py: PLAN_METADATA is invalid")


def test_unknown_key_rejected_when_loaded_from_a_module() -> None:
    module = types.SimpleNamespace(
        PLAN_METADATA={**_WELL_FORMED, "category": "accelerator"},
        __name__="legacy_plan_module",
    )

    with pytest.raises(PlanMetadataError, match="unknown key"):
        parse_plan_metadata(module)


def test_error_names_every_unknown_key_at_once() -> None:
    raw = {**_WELL_FORMED, "category": "accelerator", "required_devices": []}

    with pytest.raises(PlanMetadataError) as excinfo:
        parse_plan_metadata_dict(raw, source="test")

    message = str(excinfo.value)
    assert "category" in message
    assert "required_devices" in message


def test_error_separates_bad_fields_from_unknown_keys() -> None:
    raw = {"name": "orm", "description": "Sweep.", "category": "accelerator"}

    with pytest.raises(PlanMetadataError) as excinfo:
        parse_plan_metadata_dict(raw, source="test")

    assert str(excinfo.value) == (
        "test: PLAN_METADATA is invalid for field(s): writes; unknown key(s): category"
    )


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


# ---------------------------------------------------------------------------
# The served schema states every default the plan has
# ---------------------------------------------------------------------------


class _FactoryDefaultParams(BaseModel):
    target: str
    monitors: list[str] = Field(default_factory=lambda: ["BPM:1"])


class _Inner(BaseModel):
    tags: list[str] = Field(default_factory=lambda: ["a", "b"])


class _NestedFactoryParams(BaseModel):
    inner: _Inner


def _raise() -> list[str]:
    raise RuntimeError("factory is broken")


class _RaisingFactoryParams(BaseModel):
    target: str
    monitors: list[str] = Field(default_factory=_raise)
    span: float = 1.0


def _spec_over(schema: type[BaseModel]) -> PlanSpec:
    return PlanSpec(name="probe", plan=lambda devices, params: None, schema=schema)


def test_to_dict_schema_states_a_factory_default() -> None:
    schema = _spec_over(_FactoryDefaultParams).to_dict()["schema"]

    assert schema["properties"]["monitors"]["default"] == ["BPM:1"]


def test_to_dict_schema_states_a_nested_factory_default() -> None:
    schema = _spec_over(_NestedFactoryParams).to_dict()["schema"]

    assert schema["$defs"]["_Inner"]["properties"]["tags"]["default"] == ["a", "b"]


def test_to_dict_survives_a_factory_that_raises(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="osprey.services.bluesky_bridge.plan_types"):
        schema = _spec_over(_RaisingFactoryParams).to_dict()["schema"]

    assert "default" not in schema["properties"]["monitors"]
    assert schema["properties"]["span"]["default"] == 1.0
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "probe" in warnings[0]
    assert "monitors" in warnings[0]


def test_a_required_field_beside_a_factory_default_stays_required() -> None:
    schema = _spec_over(_FactoryDefaultParams).to_dict()["schema"]

    assert schema["required"] == ["target"]


def test_summary_token_grammar_is_the_panels() -> None:
    """The loader accepts exactly the placeholders the panel fills."""
    panel = (
        Path(osprey.__file__).parent / "interfaces/bluesky_web/panels/bluesky/plan-presentation.js"
    ).read_text()
    assert f"const SUMMARY_TOKEN = /{plan_metadata._SUMMARY_TOKEN.pattern}/g;" in panel
