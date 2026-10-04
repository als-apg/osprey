"""The Middle Layer's model files, and the helper check A reads them through.

Every export at a model-file version carries its ``<stem>.model.json`` with
every section of the contract, beside siblings that name one exporter. The
helper loads a committed file, fails naming the path of a missing one, skips a
refused section naming the Middle Layer's message and finds every refusal by
its dotted path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.facility._model_reference import (
    FIXTURES,
    MATLAB_LINES,
    MATLAB_RINGS,
    OWNER_STEP,
    model_path,
    model_reference,
    refused_entries,
    section,
)
from tests.templates.mml_export_contract import MODEL_FILE_EXPORTERS, MODEL_SECTION_KEYS


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _exports() -> list[Path]:
    """Every committed ``*.ao.json`` under the fixture root, one per export run."""
    return sorted(FIXTURES.glob("*/*.ao.json"))


def _stem(ao: Path) -> str:
    return ao.name[: -len(".ao.json")]


def _token(path: Path) -> str:
    return str(_read_json(path)["_export"]["exporter"])


def _write_model(root: Path, tree: str, stem: str, document: dict[str, Any]) -> Path:
    path = model_path(tree, stem, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_the_three_facility_model_files_are_committed() -> None:
    missing = [
        f"{tree}/{model_path(tree, stem).name}"
        for tree, stem in (*MATLAB_RINGS, *MATLAB_LINES)
        if not model_path(tree, stem).is_file()
    ]
    assert len(MATLAB_RINGS) + len(MATLAB_LINES) == 3
    assert missing == [], f"not committed: {missing}; {OWNER_STEP}"


def test_every_model_file_export_has_its_model_file_with_every_section() -> None:
    found = 0
    for ao in _exports():
        if _token(ao) not in MODEL_FILE_EXPORTERS:
            continue
        found += 1
        model = ao.with_name(f"{_stem(ao)}.model.json")
        assert model.is_file(), f"{ao.parent.name}/{model.name} is missing"
        document = _read_json(model)
        missing = [key for key in MODEL_SECTION_KEYS if key not in document]
        assert missing == [], f"{ao.parent.name}/{model.name} lacks {missing}"
    assert found >= 1


def test_every_sibling_of_one_stem_carries_one_exporter_token() -> None:
    for ao in _exports():
        stem = _stem(ao)
        tokens = {path.name: _token(path) for path in sorted(ao.parent.glob(f"{stem}.*.json"))}
        assert len(set(tokens.values())) == 1, f"{ao.parent.name}/{stem}: {tokens}"


def test_a_model_file_sits_only_beside_a_model_file_export() -> None:
    for model in sorted(FIXTURES.glob("*/*.model.json")):
        ao = model.with_name(model.name[: -len(".model.json")] + ".ao.json")
        assert ao.is_file(), f"{model.parent.name}/{model.name} has no ao.json"
        assert _token(ao) in MODEL_FILE_EXPORTERS, model.name


def test_a_missing_model_file_fails_naming_its_path(tmp_path: Path) -> None:
    with pytest.raises(pytest.fail.Exception) as info:
        model_reference("machine", "quokka.sr", root=tmp_path)

    message = str(info.value)
    assert str(tmp_path / "machine" / "quokka.sr.model.json") in message
    assert OWNER_STEP in message


def test_a_committed_model_file_loads(tmp_path: Path) -> None:
    document = {"_export": {}, "tune": {"method": "findm66"}}
    _write_model(tmp_path, "machine", "quokka.sr", document)

    assert model_reference("machine", "quokka.sr", root=tmp_path) == document


def test_a_refused_section_skips_naming_the_middle_layer_message(tmp_path: Path) -> None:
    _write_model(
        tmp_path,
        "machine",
        "quokka.sr",
        {"tune_response": {"refused": "MemberOf 'Tune Corrector' was not found"}},
    )
    reference = model_reference("machine", "quokka.sr", root=tmp_path)

    with pytest.raises(pytest.skip.Exception) as info:
        section(reference, "tune_response")

    assert "tune_response" in str(info.value)
    assert "MemberOf 'Tune Corrector' was not found" in str(info.value)


def test_an_answered_section_is_returned_whole() -> None:
    answer = {"method": "findm66", "cavity_on": [0.1, 0.2]}

    assert section({"tune": answer}, "tune") is answer


def test_a_section_that_merely_carries_a_refused_entry_is_answered() -> None:
    """A refusal inside a section is that entry's, not the section's."""
    answer = {"physics": [1.0, 2.0], "hardware": {"refused": "no cavity"}}

    assert section({"chromaticity": answer}, "chromaticity") is answer


def test_refusals_are_found_by_dotted_path() -> None:
    reference = {
        "_export": {"refused": "not a section"},
        "state": {"mcf": {"refused": "transport line"}, "energy_gev": 3},
        "tune": {"refused": "transport line"},
        "chromaticity": {"hardware": {"refused": "no cavity"}, "physics": [1, 2]},
    }

    assert refused_entries(reference) == {
        "state.mcf": "transport line",
        "tune": "transport line",
        "chromaticity.hardware": "no cavity",
    }
