"""A kind's own facts: what its facts writer wrote, in each model's kind block.

A registered kind may write facts of its own from a model's sub-block; they
land beside the block in ``measurement_models[<model>].kinds[<kind>].facts``,
so everything about a kind on a model is in that model's kind block. Two
models may give one kind different blocks, and each carries what its own block
writes. The facts document has no namespace of the kinds' facts apart from the
models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.render import render_facility_outputs
from osprey.facility.views.facts import FACTS_FILE, FACTS_KEYS, zero_source_facts
from tests.facility._pyaml_trees import built_document, measured_tree, with_correctors, with_rf
from tests.facility.fixtures.registry_kind import REGISTRY_KIND, install

pytestmark = pytest.mark.usefixtures("entry_points_read_afresh")

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


def test_no_kind_writing_facts_leaves_every_kind_block_without_facts(tmp_path):
    facts = _facts(_render(tmp_path, measured_tree()))

    assert "kinds" not in facts
    assert "kinds" not in FACTS_KEYS
    assert "kinds" not in zero_source_facts({"code": "x", "name": "x", "description": None})
    for record in facts["measurement_models"].values():
        assert all("facts" not in entry for entry in record["kinds"].values())


@pytest.mark.usefixtures("registry_kind")
def test_a_kinds_facts_writer_writes_into_the_models_kind_block(tmp_path):
    from osprey.apps import sdk

    render = _render(tmp_path, _tree())

    expected = {"chain": ["arm"], "trigger_sample": 12, "settings": ["gain"]}
    entry = _facts(render)["measurement_models"]["SR"]["kinds"][REGISTRY_KIND]
    assert entry["facts"] == expected
    assert entry["block"] == BLOCK
    assert sdk.read_facts(render)["measurement_models"]["SR"]["kinds"][REGISTRY_KIND] == entry
    assert "kinds" not in _facts(render)


@pytest.mark.usefixtures("registry_kind")
def test_two_models_giving_one_kind_two_blocks_each_carry_their_own_facts(tmp_path):
    tree = _tree()
    tree["measurement/LINE.yaml"] |= {
        "kinds": ["orm", REGISTRY_KIND],
        "instruments": {"tune": "SR:TUNE:X"},
        "kind_blocks": {REGISTRY_KIND: {**BLOCK, "trigger_sample": 3}},
    }

    models = _facts(_render(tmp_path, tree))["measurement_models"]

    assert models["SR"]["kinds"][REGISTRY_KIND]["facts"]["trigger_sample"] == 12
    assert models["LINE"]["kinds"][REGISTRY_KIND]["facts"]["trigger_sample"] == 3
    assert "facts" not in models["LINE"]["kinds"]["orm"]
