"""The shared YAML writer starts every dump fresh, whatever the dump before it did.

``config_writer._yaml`` is one module-level instance every comment-preserving
write goes through. A dump that raises part-way must not leave its emitter,
serializer or representer behind for the next write to trip over.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from ruamel.yaml import CommentedMap
from ruamel.yaml.representer import RepresenterError

from osprey.utils import config_writer
from osprey.utils.config_writer import load_config_document, save_config_document


class _Unrepresentable:
    """An object no representer knows, so representing it raises mid-document."""


def _interrupted(document: CommentedMap) -> CommentedMap:
    shared = CommentedMap({"x": 1})
    document["first"] = shared
    document["again"] = shared
    document["broken"] = _Unrepresentable()
    return document


def test_a_write_after_a_failed_dump_succeeds(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text("kept: 1\n", encoding="utf-8")

    with pytest.raises(RepresenterError):
        save_config_document(path, _interrupted(CommentedMap()))

    document = CommentedMap({"name": "demo", "items": ["a", "b"]})
    save_config_document(path, document)

    assert path.read_text(encoding="utf-8") == "name: demo\nitems:\n  - a\n  - b\n"
    assert load_config_document(path) == {"name": "demo", "items": ["a", "b"]}


def test_a_dump_after_a_failed_dump_carries_no_state_of_it() -> None:
    with pytest.raises(RepresenterError):
        config_writer._yaml.dump(_interrupted(CommentedMap()), io.StringIO())

    shared = CommentedMap({"x": 1})
    rendered = io.StringIO()
    config_writer._yaml.dump(CommentedMap({"one": shared, "two": shared}), rendered)

    assert rendered.getvalue() == "one: &id001\n  x: 1\ntwo: *id001\n"
