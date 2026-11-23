"""The published JSON Schema of facility.json is pinned, honest and agreed.

- **Pin**: the committed current schema
  (``schema/_generated/facility.schema.json``) is what the generator writes from
  the model today, so a shape change cannot land without regenerating.
- **Honesty**: current is an additive extension of the floor for its header
  version N (``facility-<N>.schema.json``), as judged by
  ``osprey_facility.schema.compat.classify``; a breaking change must bump N.
- **Agreement**: current's ``schema`` const, ``FACILITY_HEADER`` and the floor
  on disk all name the same thing.

The panels reader's agreement with the header and a built demo document's
validation against current need the panels and the framework, so they stay in
the root ``tests/facility/test_facility_json_schema.py``.
"""

from __future__ import annotations

import copy
import json
from types import ModuleType
from typing import Any

import pytest

from osprey_facility.schema import _generate
from osprey_facility.schema.compat import classify
from osprey_facility.validate import FACILITY_HEADER
from osprey_facility_tests._harness import PACKAGE_SRC

GENERATED_DIR = PACKAGE_SRC / "osprey_facility" / "schema" / "_generated"
COMMAND = "uv run python -m osprey_facility.schema._generate"


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    return _generate


@pytest.fixture(scope="module")
def current() -> dict[str, Any]:
    return json.loads((GENERATED_DIR / "facility.schema.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def floor(generator: ModuleType) -> dict[str, Any]:
    path = GENERATED_DIR / generator.floor_name(FACILITY_HEADER)
    return json.loads(path.read_text(encoding="utf-8"))


# --- (a) pin -----------------------------------------------------------------


def test_committed_current_is_what_the_generator_writes(generator: ModuleType) -> None:
    committed = (GENERATED_DIR / generator.CURRENT).read_text(encoding="utf-8")
    generated = generator.render(generator.build())
    assert generated == committed, (
        f"_generated/{generator.CURRENT} is stale: the facility model's shape changed. "
        f"Run `{COMMAND}`, then bump N or mark additive by regenerating "
        "(an additive change only needs the regenerated current; a breaking one "
        "needs a new N in FACILITY_HEADER and a new floor)."
    )


# --- (b) honesty ---------------------------------------------------------------


def test_current_is_an_additive_extension_of_its_floor(
    floor: dict[str, Any], current: dict[str, Any]
) -> None:
    result = classify(floor, current)
    assert result.additive, (
        f"facility.schema.json breaks the floor for {FACILITY_HEADER}; bump N "
        "(FACILITY_HEADER) and regenerate with "
        f"`{COMMAND}`. Breaking changes:\n" + "\n".join(result.breaking)
    )


def _add_optional_property(schema: dict[str, Any]) -> str:
    schema["$defs"]["Channel"]["properties"]["fixture_extra"] = {
        "anyOf": [{"type": "string"}, {"type": "null"}],
        "default": None,
    }
    return "#/$defs/Channel/properties/fixture_extra"


def _add_def(schema: dict[str, Any]) -> str:
    schema["$defs"]["FixtureRecord"] = {"type": "object", "properties": {}}
    return "#/$defs/FixtureRecord"


def _change_description(schema: dict[str, Any]) -> str:
    schema["$defs"]["Channel"]["description"] = "A changed description."
    return ""


def _remove_property(schema: dict[str, Any]) -> str:
    props = schema["$defs"]["Channel"]["properties"]
    name = next(n for n in sorted(props) if n not in {"id", "title", "description"})
    del props[name]
    return f"#/$defs/Channel/properties/{name}"


def _retype_property(schema: dict[str, Any]) -> str:
    schema["$defs"]["Channel"]["properties"]["id"]["type"] = "integer"
    return "#/$defs/Channel/properties/id/type"


def _remove_enum_member(schema: dict[str, Any]) -> str:
    schema["$defs"]["PlaneEnum"]["enum"].remove("y")
    return "#/$defs/PlaneEnum/enum"


def _newly_require(schema: dict[str, Any]) -> str:
    optional = next(p for p in sorted(schema["properties"]) if p not in schema["required"])
    schema["required"].append(optional)
    return "#/required"


def _require_new_property(schema: dict[str, Any]) -> str:
    schema["properties"]["fixture_required"] = {"type": "string"}
    schema["required"].append("fixture_required")
    return "#/properties/fixture_required"


def _remove_property_named_title(schema: dict[str, Any]) -> str:
    del schema["$defs"]["Document"]["properties"]["title"]
    return "#/$defs/Document/properties/title"


def _remove_property_named_description(schema: dict[str, Any]) -> str:
    del schema["$defs"]["Channel"]["properties"]["description"]
    return "#/$defs/Channel/properties/description"


@pytest.mark.parametrize(
    "mutate",
    [_add_optional_property, _add_def, _change_description],
    ids=["new-optional-property", "new-defs-entry", "changed-description"],
)
def test_classifier_passes_an_additive_change(floor: dict[str, Any], mutate: Any) -> None:
    newer = copy.deepcopy(floor)
    pointer = mutate(newer)
    result = classify(floor, newer)
    assert result.additive, result.breaking
    if pointer:
        assert any(line.startswith(pointer) for line in result.additions), result.additions


@pytest.mark.parametrize(
    "mutate",
    [
        _remove_property,
        _retype_property,
        _remove_enum_member,
        _newly_require,
        _require_new_property,
        _remove_property_named_title,
        _remove_property_named_description,
    ],
    ids=[
        "removed-property",
        "retyped-property",
        "removed-enum-member",
        "newly-required-field",
        "new-required-property",
        "removed-property-named-title",
        "removed-property-named-description",
    ],
)
def test_classifier_fails_a_breaking_change_with_bump_n(floor: dict[str, Any], mutate: Any) -> None:
    newer = copy.deepcopy(floor)
    pointer = mutate(newer)
    result = classify(floor, newer)
    assert not result.additive
    assert any(line.startswith(pointer) for line in result.breaking), result.breaking
    with pytest.raises(AssertionError, match="bump N"):
        test_current_is_an_additive_extension_of_its_floor(floor, newer)


def test_the_real_breaking_message_names_the_generator(floor: dict[str, Any]) -> None:
    newer = copy.deepcopy(floor)
    _retype_property(newer)
    with pytest.raises(AssertionError) as caught:
        test_current_is_an_additive_extension_of_its_floor(floor, newer)
    assert COMMAND in str(caught.value)


def test_open_passes_instance_values_through_untouched() -> None:
    instance = {"additionalProperties": False, "linkml_meta": {"x": 1}}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "linkml_meta": {"domain_of": ["Model"]},
        "properties": {
            "settings": {
                "type": "object",
                "additionalProperties": False,
                "default": copy.deepcopy(instance),
                "const": copy.deepcopy(instance),
                "enum": [copy.deepcopy(instance)],
                "examples": [copy.deepcopy(instance)],
            }
        },
    }
    opened = _generate._open(schema)
    assert opened == {
        "type": "object",
        "properties": {
            "settings": {
                "type": "object",
                "default": instance,
                "const": instance,
                "enum": [instance],
                "examples": [instance],
            }
        },
    }


# --- (c) agreement -------------------------------------------------------------


def test_current_schema_const_is_the_facility_header(current: dict[str, Any]) -> None:
    assert current["properties"]["schema"]["const"] == FACILITY_HEADER


def test_a_floor_exists_for_the_current_header(generator: ModuleType) -> None:
    floor_path = GENERATED_DIR / generator.floor_name(FACILITY_HEADER)
    assert floor_path.is_file(), f"{floor_path.name} is missing; run `{COMMAND}`"
