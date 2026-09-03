"""The shared YAML loader parses on libyaml when it is there and on PyYAML otherwise."""

from __future__ import annotations

import io

import pytest
import yaml

from osprey_connectors import yaml_loader

_DOCUMENT = """\
# a comment
name: demo
count: 3
ratio: 0.5
flags: [yes, no, ~]
nested:
  when: 2026-09-02
  text: "quoted: colon"
  multi: |
    line one
    line two
"""


@pytest.mark.skipif(not yaml.__with_libyaml__, reason="libyaml bindings not installed")
def test_picks_the_c_loader_when_libyaml_is_present() -> None:
    assert yaml_loader.safe_loader() is yaml.CSafeLoader


def test_falls_back_to_the_pure_python_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(yaml, "CSafeLoader", raising=False)
    assert yaml_loader.safe_loader() is yaml.SafeLoader
    assert yaml_loader.safe_load(_DOCUMENT) == yaml.safe_load(_DOCUMENT)


def test_parses_like_safe_load_from_str_bytes_and_file() -> None:
    expected = yaml.safe_load(_DOCUMENT)
    assert yaml_loader.safe_load(_DOCUMENT) == expected
    assert yaml_loader.safe_load(_DOCUMENT.encode("utf-8")) == expected
    assert yaml_loader.safe_load(io.StringIO(_DOCUMENT)) == expected
    assert yaml_loader.safe_load("") is None


def test_rejects_python_object_tags_like_safe_load() -> None:
    with pytest.raises(yaml.YAMLError):
        yaml_loader.safe_load("!!python/object/apply:os.system ['true']")


def test_invalid_yaml_raises_yaml_error() -> None:
    with pytest.raises(yaml.YAMLError):
        yaml_loader.safe_load("key: [unclosed")
