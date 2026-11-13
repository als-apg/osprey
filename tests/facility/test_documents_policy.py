"""``documents.yaml`` and ``policy.yaml``: loaded at S1, documents emitted, policy checked, never emitted."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.combine import combine
from osprey.facility.errors import FacilityBuildError
from osprey.facility.sources import load_sources
from osprey.facility.validate import relation_types, run_stages


def _write(root: Path, files: dict[str, Any]) -> Path:
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        text = data if isinstance(data, str) else yaml.safe_dump(data, sort_keys=False)
        path.write_text(text, encoding="utf-8")
    return root


def _load_errors(tmp_path: Path, files: dict[str, Any]) -> list[FacilityBuildError]:
    return load_sources(_write(tmp_path, files)).errors


def _document(tmp_path: Path, files: dict[str, Any]) -> dict[str, Any]:
    loaded = load_sources(_write(tmp_path, files))
    assert [e.format_message() for e in loaded.errors] == []
    result = combine(loaded.sources)
    assert [e.format_message() for e in result.errors] == []
    return result.document


def _one(errors: list[FacilityBuildError]) -> FacilityBuildError:
    assert len(errors) == 1, [e.format_message() for e in errors]
    return errors[0]


_DOCS = [
    {"id": "spec-2", "title": "Magnet list", "kind": "spec", "system": "windchill"},
    {"id": "dwg-1", "title": "Rack layout", "kind": "drawing"},
]


class TestDocuments:
    def test_documents_are_emitted_sorted_by_id(self, tmp_path: Path) -> None:
        document = _document(tmp_path, {"documents.yaml": _DOCS})
        assert [d["id"] for d in document["documents"]] == ["dwg-1", "spec-2"]
        assert document["documents"][1] == _DOCS[0]

    def test_documents_are_loaded_onto_sources(self, tmp_path: Path) -> None:
        loaded = load_sources(_write(tmp_path, {"documents.yaml": _DOCS}))
        assert loaded.errors == []
        assert loaded.sources.documents == _DOCS

    @pytest.mark.parametrize("text", ["", "# no documents yet\n", "[]\n"])
    def test_empty_documents_emit_no_key(self, tmp_path: Path, text: str) -> None:
        assert "documents" not in _document(tmp_path, {"documents.yaml": text})

    def test_absent_documents_emit_no_key(self, tmp_path: Path) -> None:
        assert "documents" not in _document(tmp_path, {})

    def test_duplicate_document_id_is_refused(self, tmp_path: Path) -> None:
        error = _one(_load_errors(tmp_path, {"documents.yaml": [_DOCS[0], {"id": "spec-2"}]}))
        assert error.kind == "source-invalid"
        assert error.record_kind == "document"
        assert error.record_id == "spec-2"
        assert error.sources == ("documents.yaml",)
        assert "twice" in error.detail

    def test_unknown_document_key_is_refused(self, tmp_path: Path) -> None:
        error = _one(_load_errors(tmp_path, {"documents.yaml": [{"id": "d", "owner": "x"}]}))
        assert error.kind == "source-invalid"
        assert error.record_id == "d"
        assert error.sources == ("documents.yaml",)
        assert "unknown key `owner`" in error.detail

    def test_document_without_id_is_refused(self, tmp_path: Path) -> None:
        error = _one(_load_errors(tmp_path, {"documents.yaml": [{"title": "t"}]}))
        assert error.sources == ("documents.yaml",)
        assert "`id`" in error.detail

    def test_documents_not_a_list_is_refused(self, tmp_path: Path) -> None:
        error = _one(_load_errors(tmp_path, {"documents.yaml": {"id": "d"}}))
        assert error.sources == ("documents.yaml",)
        assert "is not a list" in error.detail

    def test_schema_errors_name_documents_yaml(self, tmp_path: Path) -> None:
        _write(tmp_path, {"documents.yaml": [{"id": "d", "title": ["not", "a", "string"]}]})
        report = run_stages(tmp_path, project_name="demo")
        assert report.failed == "schema"
        assert report.errors
        for error in report.errors:
            assert error.record_id.startswith("documents.0")
            assert error.sources == ("documents.yaml",)


_POLICY = {
    "schema": "osprey.facility.policy/1",
    "relations": {
        "precedence": {"default": ["spec", "drawing"], "powered_by": ["drawing"]},
        "stated_over_inferred": True,
    },
}


class TestPolicy:
    def test_policy_is_loaded_without_its_header(self, tmp_path: Path) -> None:
        loaded = load_sources(_write(tmp_path, {"documents.yaml": _DOCS, "policy.yaml": _POLICY}))
        assert loaded.errors == []
        assert loaded.sources.policy == {"relations": _POLICY["relations"]}

    def test_policy_is_never_emitted(self, tmp_path: Path) -> None:
        document = _document(tmp_path, {"documents.yaml": _DOCS, "policy.yaml": _POLICY})
        assert "policy" not in document
        assert run_stages(tmp_path, project_name="demo").ok

    @pytest.mark.parametrize("text", ["", "# nothing yet\n", "schema: osprey.facility.policy/1\n"])
    def test_empty_policy_emits_no_key(self, tmp_path: Path, text: str) -> None:
        loaded = load_sources(_write(tmp_path, {"policy.yaml": text}))
        assert loaded.errors == []
        assert loaded.sources.policy is None
        assert "policy" not in combine(loaded.sources).document

    def test_every_relation_type_is_a_precedence_key(self, tmp_path: Path) -> None:
        precedence = {name: ["spec"] for name in relation_types()}
        policy = {"relations": {"precedence": {"default": ["spec"], **precedence}}}
        assert _load_errors(tmp_path, {"documents.yaml": _DOCS, "policy.yaml": policy}) == []

    @pytest.mark.parametrize(
        ("policy", "key", "where"),
        [
            ({"relation": {}}, "relation", "policy.yaml"),
            ({"relations": {"order": []}}, "order", "policy.yaml.relations"),
            (
                {"relations": {"precedence": {"fed_by": ["spec"]}}},
                "fed_by",
                "policy.yaml.relations.precedence",
            ),
        ],
    )
    def test_unknown_key_at_each_level_is_refused(
        self, tmp_path: Path, policy: dict[str, Any], key: str, where: str
    ) -> None:
        error = _one(_load_errors(tmp_path, {"policy.yaml": policy}))
        assert error.kind == "source-invalid"
        assert error.sources == ("policy.yaml",)
        assert error.record_id == where
        assert f"unknown key `{key}`" in error.detail
        assert "policy.yaml" in error.format_message()

    def test_bad_header_is_refused(self, tmp_path: Path) -> None:
        error = _one(_load_errors(tmp_path, {"policy.yaml": {"schema": "policy-1"}}))
        assert error.sources == ("policy.yaml",)
        assert "header" in error.detail

    def test_policy_not_a_mapping_is_refused(self, tmp_path: Path) -> None:
        error = _one(_load_errors(tmp_path, {"policy.yaml": ["default"]}))
        assert error.sources == ("policy.yaml",)
        assert "is not a mapping" in error.detail

    def test_relations_not_a_mapping_is_refused(self, tmp_path: Path) -> None:
        error = _one(_load_errors(tmp_path, {"policy.yaml": {"relations": ["x"]}}))
        assert error.record_id == "policy.yaml.relations"
        assert "is not a mapping" in error.detail


def _policy_errors(
    tmp_path: Path, relations: dict[str, Any], docs: Any = _DOCS
) -> list[FacilityBuildError]:
    files: dict[str, Any] = {"policy.yaml": {"relations": relations}}
    if docs is not None:
        files["documents.yaml"] = docs
    return _load_errors(tmp_path, files)


_PRECEDENCE = "policy.yaml.relations.precedence"


class TestPolicyValues:
    def test_list_of_declared_kinds_is_accepted(self, tmp_path: Path) -> None:
        relations = {
            "precedence": {"default": ["spec", "drawing"], "powered_by": ["drawing"]},
            "stated_over_inferred": False,
        }
        assert _policy_errors(tmp_path, relations) == []

    def test_default_undecided_is_accepted(self, tmp_path: Path) -> None:
        relations = {"precedence": {"default": "undecided", "powered_by": ["spec"]}}
        assert _policy_errors(tmp_path, relations) == []

    def test_default_undecided_needs_no_documents(self, tmp_path: Path) -> None:
        assert _policy_errors(tmp_path, {"precedence": {"default": "undecided"}}, None) == []

    def test_absent_stated_over_inferred_is_accepted(self, tmp_path: Path) -> None:
        assert _policy_errors(tmp_path, {"precedence": {"default": ["spec"]}}) == []

    def test_build_with_valid_policy_passes(self, tmp_path: Path) -> None:
        _write(tmp_path, {"documents.yaml": _DOCS, "policy.yaml": _POLICY})
        assert run_stages(tmp_path, project_name="demo").ok

    @pytest.mark.parametrize(
        ("precedence", "where", "detail"),
        [
            ({"default": ["spec", "spec"]}, "default", "lists `spec` twice"),
            ({"powered_by": ["spec", "drawing", "spec"]}, "powered_by", "lists `spec` twice"),
            ({"default": [""]}, "default", "entry 0 is '', not a document kind"),
            ({"default": ["spec", "  "]}, "default", "entry 1 is '  ', not a document kind"),
            ({"default": [3]}, "default", "entry 0 is 3, not a document kind"),
            ({"default": [None]}, "default", "entry 0 is None, not a document kind"),
            ({"default": [["spec"]]}, "default", "entry 0 is ['spec'], not a document kind"),
            (
                {"default": ["memo"]},
                "default",
                "lists `memo`, which no document in documents.yaml has as its kind",
            ),
            (
                {"powered_by": ["spec", "memo"]},
                "powered_by",
                "lists `memo`, which no document in documents.yaml has as its kind",
            ),
            ({"powered_by": "undecided"}, "powered_by", "is not a list of document kinds"),
            ({"default": "spec"}, "default", "is not a list of document kinds or `undecided`"),
            ({"default": {"spec": 1}}, "default", "is not a list of document kinds or `undecided`"),
        ],
    )
    def test_bad_precedence_value_stops_at_its_path(
        self, tmp_path: Path, precedence: dict[str, Any], where: str, detail: str
    ) -> None:
        error = _one(_policy_errors(tmp_path, {"precedence": precedence}))
        assert error.kind == "source-invalid"
        assert error.record_kind == "path"
        assert error.record_id == f"{_PRECEDENCE}.{where}"
        assert error.sources == ("policy.yaml",)
        assert error.detail == detail

    @pytest.mark.parametrize("docs", [None, "", "[]\n"])
    def test_kind_with_no_documents_stops(self, tmp_path: Path, docs: Any) -> None:
        error = _one(_policy_errors(tmp_path, {"precedence": {"default": ["spec"]}}, docs))
        assert error.kind == "source-invalid"
        assert error.record_id == f"{_PRECEDENCE}.default"
        assert error.sources == ("policy.yaml",)
        assert error.detail == "lists `spec`, but no documents.yaml declares documents"

    def test_document_without_kind_declares_no_kind(self, tmp_path: Path) -> None:
        docs = [{"id": "d1", "title": "untyped"}]
        error = _one(_policy_errors(tmp_path, {"precedence": {"default": ["spec"]}}, docs))
        assert error.record_id == f"{_PRECEDENCE}.default"
        assert "no document in documents.yaml has as its kind" in error.detail

    def test_every_bad_entry_is_reported(self, tmp_path: Path) -> None:
        errors = _policy_errors(tmp_path, {"precedence": {"default": ["memo", "spec", "spec", 1]}})
        assert [e.detail for e in errors] == [
            "lists `memo`, which no document in documents.yaml has as its kind",
            "lists `spec` twice",
            "entry 3 is 1, not a document kind",
        ]

    def test_unknown_type_key_is_not_value_checked(self, tmp_path: Path) -> None:
        error = _one(_policy_errors(tmp_path, {"precedence": {"fed_by": ["memo"]}}))
        assert "unknown key `fed_by`" in error.detail

    @pytest.mark.parametrize("value", ["yes", "true", 1, 0, None, ["true"]])
    def test_non_bool_stated_over_inferred_stops(self, tmp_path: Path, value: Any) -> None:
        error = _one(_policy_errors(tmp_path, {"stated_over_inferred": value}))
        assert error.kind == "source-invalid"
        assert error.record_kind == "path"
        assert error.record_id == "policy.yaml.relations.stated_over_inferred"
        assert error.sources == ("policy.yaml",)
        assert error.detail == f"is {value!r}, not true or false"
        assert "policy.yaml.relations.stated_over_inferred" in error.format_message()

    @pytest.mark.parametrize("value", [True, False])
    def test_bool_stated_over_inferred_is_accepted(self, tmp_path: Path, value: bool) -> None:
        assert _policy_errors(tmp_path, {"stated_over_inferred": value}) == []

    def test_bad_value_stops_the_build_at_load(self, tmp_path: Path) -> None:
        _write(tmp_path, {"policy.yaml": {"relations": {"precedence": {"default": ["spec"]}}}})
        report = run_stages(tmp_path, project_name="demo")
        assert report.failed == "load"
        assert [e.kind for e in report.errors] == ["source-invalid"]
