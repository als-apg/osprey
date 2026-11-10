"""The facts document's ``kinds`` namespace: what each listed kind's facts writer wrote.

A registered kind may write facts of its own from its sub-block; they land in
``facts["kinds"][<kind>]``, apart from the per-model kind block. The namespace
is absent from a document no kind wrote into, is no key of the zero-source
facts, and a reader of the document takes its absence for ``{}``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.errors import FacilityBuildError
from osprey.facility.render import render_facility_outputs
from osprey.facility.views.facts import FACTS_FILE, FACTS_KEYS, read_facts, zero_source_facts
from tests.facility._pyaml_trees import built_document, measured_tree, with_correctors, with_rf
from tests.facility.fixtures.registry_kind import REGISTRY_KIND, install

pytest.importorskip("pyaml")

BLOCK = {
    "trigger_sample": 12,
    "settings": [
        {
            "id": "gain",
            "label": "Gain",
            "description": "Monitor gain",
            "scope": "global",
            "address": "SR:TUNE:Y",
            "timeout_s": 2.0,
        }
    ],
    "steps": [{"id": "arm", "label": "Arm", "description": "Arm the chain", "default": True}],
}


@pytest.fixture
def registry_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch)


def _tree(block: dict[str, Any] | None = None) -> dict[str, Any]:
    tree = with_correctors(with_rf(measured_tree()))
    tree["measurement/SR.yaml"] |= {
        "kinds": [REGISTRY_KIND],
        "corrector_delta": 1.0e-5,
        "kind_blocks": {REGISTRY_KIND: dict(block or BLOCK)},
    }
    return tree


def _render(tmp_path: Path, tree: dict[str, Any]) -> Path:
    document, facility = built_document(tmp_path / "tree", tree)
    render = tmp_path / "render"
    render.mkdir()
    render_facility_outputs(render, document, {}, facility)
    (render / "config.yml").write_text("project_name: demo\n", encoding="utf-8")
    return render


def _facts(render: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((render / "data" / FACTS_FILE).read_text("utf-8"))
    return loaded


def test_no_kind_writing_facts_leaves_the_document_without_the_namespace(tmp_path):
    facts = _facts(_render(tmp_path, measured_tree()))

    assert "kinds" not in facts
    assert "kinds" not in FACTS_KEYS
    assert "kinds" not in zero_source_facts({"code": "x", "name": "x", "description": None})


@pytest.mark.usefixtures("registry_kind")
def test_a_kinds_facts_writer_writes_its_own_namespace(tmp_path):
    from osprey.apps import sdk

    render = _render(tmp_path, _tree())

    expected = {"chain": ["arm"], "trigger_sample": 12, "settings": ["gain"]}
    assert _facts(render)["kinds"] == {REGISTRY_KIND: expected}
    assert sdk.read_facts(render)["kinds"][REGISTRY_KIND] == expected
    assert "kinds" not in _facts(render)["measurement_models"]["SR"]["kinds"][REGISTRY_KIND]


def test_a_document_without_the_namespace_is_still_read_as_facts(tmp_path):
    render = _render(tmp_path, measured_tree())

    facts = read_facts(render, "demo")

    assert facts["measurement_models"]
    assert facts.get("kinds", {}) == {}


@pytest.mark.usefixtures("registry_kind")
def test_two_models_writing_one_kinds_facts_two_ways_stop_the_build(tmp_path):
    tree = _tree()
    tree["measurement/LINE.yaml"] |= {
        "kinds": ["orm", REGISTRY_KIND],
        "instruments": {"tune": "SR:TUNE:X"},
        "kind_blocks": {REGISTRY_KIND: {**BLOCK, "trigger_sample": 3}},
    }

    with pytest.raises(FacilityBuildError) as stopped:
        built_document(tmp_path / "tree", tree)

    assert stopped.value.format_message() == (
        f"facility: value-invalid: measurement SR — kind {REGISTRY_KIND}'s facts are one per "
        "deployment; models LINE and SR give it blocks it writes two ways; fix: give "
        f"`kind_blocks.{REGISTRY_KIND}` of SR what LINE gives it"
    )

    tree["measurement/LINE.yaml"]["kind_blocks"][REGISTRY_KIND] = dict(BLOCK)
    built_document(tmp_path / "same", tree)
