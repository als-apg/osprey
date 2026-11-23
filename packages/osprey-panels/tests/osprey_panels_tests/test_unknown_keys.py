"""A key the panels reader does not know changes no screen.

``facility.json`` may grow keys the panels never read: at the top level and inside a
record. The reader ignores both, so the screens a document with such keys emits equal
the screens of the document without them, once ``content_hash`` (which hashes the
file's bytes, and so differs) is masked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osprey_panels import emit, load_graph
from osprey_panels.model import PanelGraph
from osprey_panels_tests._fixture import (
    FIXTURE_DOCUMENT,
    PANELS_DIR,
    facility_bytes,
    load_fixture_document,
)

#: The tables whose first row gets one unknown key: the record kinds, and the
#: ``signals`` and ``relation_types`` vocabularies.
RECORD_KINDS = ("signals", "relation_types", "places", "devices", "channels", "groups")


def _files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _masked(files: dict[str, bytes], graph: PanelGraph) -> dict[str, bytes]:
    """The files with this graph's ``content_hash`` masked.

    Masks the full digest (the description stamp and ``index.json``'s
    ``facility_hash``) and its short form (the ``model <hex>`` header label), and
    nothing else.
    """
    digest = graph.content_hash.encode()
    short = b"model " + digest[:8]
    return {
        name: data.replace(digest, b"<content_hash>").replace(short, b"model <short>")
        for name, data in files.items()
    }


def _emit(tmp_path: Path, name: str, document: dict[str, Any]) -> tuple[PanelGraph, Path]:
    facility = tmp_path / name / "facility.json"
    facility.parent.mkdir(parents=True)
    facility.write_bytes(facility_bytes(document))
    graph = load_graph(facility, PANELS_DIR)
    out = tmp_path / name / "screens"
    emit.write(graph, out)
    return graph, out


def _with_unknown_keys(document: dict[str, Any]) -> dict[str, Any]:
    document["a_later_top_level_key"] = {"rows": [1, 2, 3], "note": "not read by the panels"}
    for kind in RECORD_KINDS:
        document[kind][0]["a_later_record_key"] = {"note": f"not read on {kind}"}
    return document


@pytest.fixture(scope="module")
def emitted(tmp_path_factory: pytest.TempPathFactory):
    tmp_path = tmp_path_factory.mktemp("unknown-keys")
    plain = _emit(tmp_path, "plain", load_fixture_document())
    widened = _emit(tmp_path, "widened", _with_unknown_keys(load_fixture_document()))
    return plain, widened


def test_the_unknown_keys_reach_the_written_document() -> None:
    document = json.loads(facility_bytes(_with_unknown_keys(load_fixture_document())))
    assert "a_later_top_level_key" in document
    for kind in RECORD_KINDS:
        assert "a_later_record_key" in document[kind][0], kind
    assert "a_later_top_level_key" not in json.loads(FIXTURE_DOCUMENT.read_text())


def test_the_unknown_keys_change_the_content_hash(emitted) -> None:
    (plain, _), (widened, _) = emitted
    assert plain.content_hash != widened.content_hash


def test_a_document_with_unknown_keys_emits_identical_screens(emitted) -> None:
    (plain, plain_out), (widened, widened_out) = emitted
    plain_files, widened_files = _files(plain_out), _files(widened_out)

    assert plain_files.keys() == widened_files.keys()
    assert any(name.endswith(emit.SCREEN_SUFFIX) for name in plain_files)
    assert _masked(plain_files, plain) == _masked(widened_files, widened)


def test_the_mask_hides_only_the_content_hash(emitted) -> None:
    (plain, plain_out), (widened, widened_out) = emitted
    index = json.loads(_masked(_files(plain_out), plain)[emit.INDEX])
    assert index["facility_hash"] == "<content_hash>"
    assert plain.content_hash not in json.dumps(index)
    raw_differing = {
        name for name, data in _files(plain_out).items() if data != _files(widened_out)[name]
    }
    assert emit.INDEX in raw_differing  # the hash really differs before the mask

    # Negative control: a difference that is not the hash survives the mask.
    files = _files(plain_out)
    name = next(n for n in sorted(files) if n.endswith(emit.SCREEN_SUFFIX))
    changed = {**files, name: files[name] + b"\n<!-- not the hash -->"}
    assert _masked(changed, plain) != _masked(files, plain)
    assert _masked(changed, plain)[name].endswith(b"<!-- not the hash -->")


def test_a_widened_document_checks_clean_against_its_own_screens(emitted) -> None:
    (_, _), (widened, widened_out) = emitted
    assert [p for p in emit.check(widened, widened_out) if p.failing] == []
