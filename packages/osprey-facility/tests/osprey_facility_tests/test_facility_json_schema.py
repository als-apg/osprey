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
from pathlib import Path
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


# --- identity -----------------------------------------------------------------

DIALECT = "https://json-schema.org/draft/2020-12/schema"
IDENTITY_BASE = "https://als-apg.github.io/osprey/latest/schemas/facility"


def _with_identity(schema: dict[str, Any], n: int) -> dict[str, Any]:
    return {"$schema": DIALECT, "$id": f"{IDENTITY_BASE}/{n}.schema.json", **schema}


def test_top_level_identity_is_neither_breaking_nor_additive(floor: dict[str, Any]) -> None:
    bare = {k: v for k, v in floor.items() if k not in {"$schema", "$id"}}
    newer = _with_identity(copy.deepcopy(bare), 4)
    result = classify(bare, newer)
    assert result.breaking == ()
    assert result.additions == ()
    removed = classify(newer, bare)
    assert removed.breaking == ()
    assert removed.additions == ()


def test_a_changed_top_level_id_is_not_breaking(floor: dict[str, Any]) -> None:
    older = _with_identity(copy.deepcopy(floor), 4)
    newer = _with_identity(copy.deepcopy(floor), 5)
    result = classify(older, newer)
    assert result.breaking == ()
    assert result.additions == ()


@pytest.mark.parametrize("keyword", ["$id", "$schema"])
def test_identity_below_the_root_still_compares(floor: dict[str, Any], keyword: str) -> None:
    newer = copy.deepcopy(floor)
    newer["$defs"]["Channel"][keyword] = "https://example.invalid/channel.schema.json"
    result = classify(floor, newer)
    assert not result.additive
    assert any(line.startswith(f"#/$defs/Channel/{keyword}") for line in result.breaking), (
        result.breaking
    )


def test_a_changed_id_below_the_root_is_breaking(floor: dict[str, Any]) -> None:
    older = copy.deepcopy(floor)
    older["$defs"]["Channel"]["$id"] = "https://example.invalid/a.schema.json"
    newer = copy.deepcopy(floor)
    newer["$defs"]["Channel"]["$id"] = "https://example.invalid/b.schema.json"
    result = classify(older, newer)
    assert any(line.startswith("#/$defs/Channel/$id") for line in result.breaking), result.breaking


# --- published identity ---------------------------------------------------------


def test_generator_check_round_trips(generator: ModuleType, tmp_path: Path) -> None:
    assert generator.main(["--output-dir", str(tmp_path)]) == 0
    assert generator.main(["--check", "--output-dir", str(tmp_path)]) == 0
    assert generator.main(["--check"]) == 0, f"_generated/ is stale; run `{COMMAND}`"


def test_generator_check_fails_on_a_stale_current(generator: ModuleType, tmp_path: Path) -> None:
    generator.main(["--output-dir", str(tmp_path)])
    (tmp_path / generator.CURRENT).write_text("{}\n", encoding="utf-8")
    assert generator.main(["--check", "--output-dir", str(tmp_path)]) == 1


def test_current_carries_the_dialect_and_the_id_of_its_header(
    generator: ModuleType, current: dict[str, Any]
) -> None:
    n = generator.header_version(FACILITY_HEADER)
    assert generator.DIALECT == DIALECT
    assert generator.IDENTITY_BASE == IDENTITY_BASE
    assert current["$schema"] == DIALECT
    assert current["$id"] == f"{IDENTITY_BASE}/{n}.schema.json"
    assert current["$id"].endswith(f"/{n}.schema.json")


def test_the_id_names_the_newest_additive_form_of_n(
    floor: dict[str, Any], current: dict[str, Any]
) -> None:
    # Same N as the floor, and nothing in current breaks the floor: the $id
    # names the newest additive form of N, not a different version.
    result = classify(floor, current)
    assert result.breaking == ()
    assert result.additive


def test_floors_are_unchanged_and_carry_no_identity(generator: ModuleType) -> None:
    floors = sorted(GENERATED_DIR.glob("facility-[0-9]*.schema.json"))
    floors = [path for path in floors if ".final." not in path.name]
    assert floors, "no floor is committed"
    for path in floors:
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert not set(schema) & set(generator.IDENTITY_KEYS), path.name
    # Regenerating never rewrites a committed floor.
    before = {path: path.read_bytes() for path in floors}
    assert generator.main(["--check"]) == 0
    assert {path: path.read_bytes() for path in floors} == before


def test_a_written_floor_is_current_without_identity(generator: ModuleType, tmp_path: Path) -> None:
    generator.main(["--output-dir", str(tmp_path)])
    current = json.loads((tmp_path / generator.CURRENT).read_text(encoding="utf-8"))
    floor_text = (tmp_path / generator.floor_name(FACILITY_HEADER)).read_text(encoding="utf-8")
    assert floor_text == generator.render(generator.without_identity(current))
    assert not set(json.loads(floor_text)) & set(generator.IDENTITY_KEYS)


def _bump(header: str) -> str:
    prefix, version = header.rsplit("/", 1)
    return f"{prefix}/{int(version) + 1}"


def test_an_n_bump_freezes_the_old_current_as_its_final(
    generator: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import osprey_facility.validate as validate

    n = generator.header_version(FACILITY_HEADER)
    generator.main(["--output-dir", str(tmp_path)])
    old_current = (tmp_path / generator.CURRENT).read_text(encoding="utf-8")
    old_floor = (tmp_path / generator.floor_name(FACILITY_HEADER)).read_bytes()

    monkeypatch.setattr(validate, "FACILITY_HEADER", _bump(FACILITY_HEADER))
    assert generator.main(["--output-dir", str(tmp_path)]) == 0

    final_path = tmp_path / generator.final_name(n)
    assert final_path.read_text(encoding="utf-8") == old_current
    assert json.loads(final_path.read_text(encoding="utf-8"))["$id"] == (
        f"{IDENTITY_BASE}/{n}.schema.json"
    )
    new_current = json.loads((tmp_path / generator.CURRENT).read_text(encoding="utf-8"))
    assert new_current["$id"] == f"{IDENTITY_BASE}/{n + 1}.schema.json"
    new_floor = json.loads(
        (tmp_path / generator.floor_name(_bump(FACILITY_HEADER))).read_text(encoding="utf-8")
    )
    assert not set(new_floor) & set(generator.IDENTITY_KEYS)
    assert (tmp_path / generator.floor_name(FACILITY_HEADER)).read_bytes() == old_floor
    assert generator.main(["--check", "--output-dir", str(tmp_path)]) == 0

    # A second run at the new N leaves the frozen final alone.
    assert generator.main(["--output-dir", str(tmp_path)]) == 0
    assert final_path.read_text(encoding="utf-8") == old_current


def test_a_current_without_id_freezes_no_final(generator: ModuleType, tmp_path: Path) -> None:
    generator.main(["--output-dir", str(tmp_path)])
    current = tmp_path / generator.CURRENT
    bare = generator.without_identity(json.loads(current.read_text(encoding="utf-8")))
    current.write_text(generator.render(bare), encoding="utf-8")
    generator.main(["--output-dir", str(tmp_path)])
    assert not list(tmp_path.glob("*.final.schema.json"))
