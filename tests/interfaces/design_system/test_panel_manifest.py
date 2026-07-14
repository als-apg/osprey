"""Tests for osprey.interfaces.design_system.panels.manifest.

Exercises the panel manifest descriptor and its fail-closed schema
validator: a valid manifest parses to a :class:`PanelManifest`; unknown
keys are preserved (forward compatibility) rather than rejected; every
required-field / type / slug rule fires with the right
:class:`ManifestRule`; :func:`validate_manifest` collects *all* errors
rather than short-circuiting; the fail-closed doors raise
:class:`PanelManifestError` carrying every failure line; and
:func:`load_manifest_file` round-trips a JSON file on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.interfaces.design_system.panels.manifest import (
    CURRENT_SCHEMA_VERSION,
    ManifestError,
    ManifestRule,
    PanelManifest,
    PanelManifestError,
    assert_valid,
    load_manifest,
    load_manifest_file,
    parse_manifest,
    validate_manifest,
)


def _valid_manifest() -> dict:
    """A minimal, fully valid manifest object."""
    return {"id": "reference-panel", "label": "Reference Panel", "entry": "index.html"}


# --- Happy path ---------------------------------------------------------------


def test_valid_manifest_parses_to_dataclass() -> None:
    manifest = parse_manifest(_valid_manifest())
    assert isinstance(manifest, PanelManifest)
    assert manifest.id == "reference-panel"
    assert manifest.label == "Reference Panel"
    assert manifest.entry == "index.html"
    assert manifest.version == CURRENT_SCHEMA_VERSION
    assert manifest.extras == {}


def test_valid_manifest_has_no_errors() -> None:
    assert validate_manifest(_valid_manifest()) == []


def test_explicit_version_is_preserved() -> None:
    data = _valid_manifest() | {"version": 2}
    manifest = parse_manifest(data)
    assert manifest.version == 2


def test_slug_with_digits_and_hyphens_is_valid() -> None:
    data = _valid_manifest() | {"id": "panel-3d-view"}
    assert validate_manifest(data) == []
    assert parse_manifest(data).id == "panel-3d-view"


# --- Forward compatibility ----------------------------------------------------


def test_unknown_keys_are_preserved_not_rejected() -> None:
    data = _valid_manifest() | {
        "source": "runtime-discovery",
        "approval_state": "approved",
    }
    assert validate_manifest(data) == []
    manifest = parse_manifest(data)
    assert manifest.extras == {
        "source": "runtime-discovery",
        "approval_state": "approved",
    }
    # Known fields never leak into extras.
    assert "id" not in manifest.extras


# --- Required fields ----------------------------------------------------------


@pytest.mark.parametrize("missing", ["id", "label", "entry"])
def test_missing_required_field(missing: str) -> None:
    data = _valid_manifest()
    del data[missing]
    errors = validate_manifest(data)
    assert [e.rule for e in errors] == [ManifestRule.MISSING_FIELD]
    assert missing in str(errors[0])


@pytest.mark.parametrize("field_name", ["id", "label", "entry"])
def test_empty_required_field(field_name: str) -> None:
    data = _valid_manifest() | {field_name: "   "}
    errors = validate_manifest(data)
    assert [e.rule for e in errors] == [ManifestRule.EMPTY_FIELD]


@pytest.mark.parametrize("field_name", ["id", "label", "entry"])
def test_wrong_type_required_field(field_name: str) -> None:
    data = _valid_manifest() | {field_name: 123}
    errors = validate_manifest(data)
    assert [e.rule for e in errors] == [ManifestRule.WRONG_TYPE]


# --- id slug ------------------------------------------------------------------


@pytest.mark.parametrize("bad_id", ["Reference", "has space", "-leading", "under_score"])
def test_bad_id_slug(bad_id: str) -> None:
    data = _valid_manifest() | {"id": bad_id}
    errors = validate_manifest(data)
    assert [e.rule for e in errors] == [ManifestRule.BAD_ID]


# --- version ------------------------------------------------------------------


def test_wrong_type_version() -> None:
    data = _valid_manifest() | {"version": "1"}
    errors = validate_manifest(data)
    assert [e.rule for e in errors] == [ManifestRule.WRONG_TYPE]
    assert "version" in str(errors[0])


def test_bool_version_rejected() -> None:
    # JSON booleans are Python ints; the validator must reject them.
    data = _valid_manifest() | {"version": True}
    errors = validate_manifest(data)
    assert [e.rule for e in errors] == [ManifestRule.WRONG_TYPE]


# --- Non-object input ---------------------------------------------------------


@pytest.mark.parametrize("bad", [None, [], "index.html", 42])
def test_non_object_input(bad: object) -> None:
    errors = validate_manifest(bad)
    assert [e.rule for e in errors] == [ManifestRule.NOT_AN_OBJECT]


# --- Collect-all (never short-circuit) ----------------------------------------


def test_validate_collects_all_errors() -> None:
    # Bad id (BAD_ID), missing label (MISSING_FIELD), non-string entry
    # (WRONG_TYPE), bad version (WRONG_TYPE) — four independent failures.
    data = {"id": "Bad Id", "entry": 5, "version": "x"}
    errors = validate_manifest(data)
    rules = {e.rule for e in errors}
    assert ManifestRule.BAD_ID in rules
    assert ManifestRule.MISSING_FIELD in rules
    assert ManifestRule.WRONG_TYPE in rules
    assert len(errors) >= 4


# --- Fail-closed doors --------------------------------------------------------


def test_assert_valid_raises_with_every_line() -> None:
    data = {"id": "Bad Id", "entry": 5}
    with pytest.raises(PanelManifestError) as excinfo:
        assert_valid(data)
    errors = excinfo.value.errors
    text = str(excinfo.value)
    # Every collected error renders on its own line.
    assert len(errors) == len(text.splitlines())
    for error in errors:
        assert str(error) in text


def test_parse_manifest_raises_on_invalid() -> None:
    with pytest.raises(PanelManifestError):
        parse_manifest({"label": "x", "entry": "index.html"})


def test_valid_manifest_passes_assert_valid() -> None:
    assert_valid(_valid_manifest())  # does not raise


# --- Error rendering ----------------------------------------------------------


def test_manifest_error_str_uses_source_prefix() -> None:
    error = ManifestError(rule=ManifestRule.MISSING_FIELD, message="boom", source="panel.json")
    assert str(error) == "panel.json: boom"


def test_source_flows_into_error_messages() -> None:
    errors = validate_manifest({}, source="my-panel/manifest.json")
    assert errors
    assert all(str(e).startswith("my-panel/manifest.json:") for e in errors)


# --- load_manifest / load_manifest_file ---------------------------------------


def test_load_manifest_parses_json_string() -> None:
    manifest = load_manifest(json.dumps(_valid_manifest()))
    assert manifest.id == "reference-panel"


def test_load_manifest_rejects_malformed_json() -> None:
    with pytest.raises(PanelManifestError):
        load_manifest("{ not json")


def test_load_manifest_file_round_trips(tmp_path: Path) -> None:
    data = _valid_manifest() | {"version": 1, "extra": "keep-me"}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    manifest = load_manifest_file(path)
    assert manifest.id == data["id"]
    assert manifest.label == data["label"]
    assert manifest.entry == data["entry"]
    assert manifest.version == 1
    assert manifest.extras == {"extra": "keep-me"}


def test_load_manifest_file_reports_path_as_source(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text(json.dumps({"label": "x", "entry": "index.html"}), encoding="utf-8")
    with pytest.raises(PanelManifestError) as excinfo:
        load_manifest_file(path)
    assert str(path) in str(excinfo.value)
