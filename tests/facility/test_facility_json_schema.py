"""The published JSON Schema of facility.json agrees with the panels and stays open.

The pin, the honesty against the floor and the schema's own agreement with
``FACILITY_HEADER`` are the description's, and live in the package suite
(``packages/osprey-facility/tests/osprey_facility_tests/test_facility_json_schema.py``).
What stays here needs the panels or the framework:

- **Agreement**: the panels reader's declared headers and file name name what
  the build writes, and it reads a further address as the description does.
- **Openness**: a built demo document validates against current, and still does
  with a key a later additive change could introduce.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jsonschema
import pytest

import osprey_facility
from osprey_facility.validate import FACILITY_HEADER
from osprey_panels import document

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATED_DIR = (
    REPO_ROOT / "packages" / "osprey-facility" / "src" / "osprey_facility" / "schema" / "_generated"
)


@pytest.fixture(scope="module")
def current() -> dict[str, Any]:
    return json.loads((GENERATED_DIR / "facility.schema.json").read_text(encoding="utf-8"))


# --- agreement -----------------------------------------------------------------


def test_the_panels_reader_reads_the_current_header() -> None:
    assert FACILITY_HEADER in document.READS


def test_the_panels_reader_names_the_file_the_build_writes() -> None:
    assert document.FACILITY_FILE == osprey_facility.FACILITY_FILE


def test_the_panels_reader_mirrors_the_record_facts() -> None:
    assert document.RECORD_FACTS == frozenset(osprey_facility.RECORD_FACTS)


def test_the_panels_reader_reads_a_further_address_as_the_description_does(
    tmp_path: Path,
) -> None:
    from osprey_facility.records import FacilityRecords
    from tests.facility._pyaml_trees import built_document
    from tests.facility.test_wiring_also import SETPOINT_ALSO, _tree

    built = built_document(tmp_path, _tree())[0]
    panels, records = document.FacilityDocument(built), FacilityRecords(built)

    ids = [str(channel["id"]) for channel in built["channels"]]
    assert SETPOINT_ALSO in ids
    for address in ids:
        assert panels.channel(address) == records.channel(address), address


# --- openness --------------------------------------------------------------


@pytest.fixture(scope="module")
def validator(current: dict[str, Any]) -> jsonschema.protocols.Validator:
    cls = jsonschema.validators.validator_for(current)
    cls.check_schema(current)
    return cls(current)


def test_a_built_demo_document_validates(
    built_demo: BuiltProject, validator: jsonschema.protocols.Validator
) -> None:
    validator.validate(built_demo.facility)


def test_an_extra_top_level_key_validates(
    built_demo: BuiltProject, validator: jsonschema.protocols.Validator
) -> None:
    doc = copy.deepcopy(built_demo.facility)
    doc["fixture_later_addition"] = {"any": "shape"}
    validator.validate(doc)


def test_an_extra_key_inside_a_record_validates(
    built_demo: BuiltProject, validator: jsonschema.protocols.Validator
) -> None:
    doc = copy.deepcopy(built_demo.facility)
    assert doc["channels"], "the demo build has no channels to extend"
    doc["channels"][0]["fixture_later_addition"] = 1
    validator.validate(doc)


def test_the_schema_is_not_vacuous(
    built_demo: BuiltProject, validator: jsonschema.protocols.Validator
) -> None:
    doc = copy.deepcopy(built_demo.facility)
    doc["schema"] = "osprey.facility.facility/0"
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(doc)
