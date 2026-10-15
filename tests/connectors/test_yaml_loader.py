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


#: The shapes the build emits, including the scalars an emitter must quote.
_EMITTED = {
    "nested": {"list": [1, 2, {"inner": "value"}], "empty_list": [], "empty_map": {}},
    "text": "Grüße, ångström, µrad",
    "long": "a sentence " * 12,
    "multi": "line one\nline two\n",
    "quoted": ["on", "no", "null", "1e3", "a: b", "#x", " leading", "yes", "~", ""],
    "ints": [0, -7, 2**40],
    "floats": [0.5, -0.0, 1e-12, float("inf"), float("-inf"), 3.0],
    "none": None,
    "bools": [True, False],
}

#: Every set of keyword arguments the build dumps with.
_DUMP_KWARGS = [
    {"sort_keys": False, "default_flow_style": False, "allow_unicode": True},
    {"sort_keys": False, "default_flow_style": False, "allow_unicode": True, "width": 100},
    {"sort_keys": False, "default_flow_style": False, "allow_unicode": True, "width": 4096},
    {"sort_keys": False, "default_flow_style": False},
]


@pytest.mark.skipif(not yaml.__with_libyaml__, reason="libyaml bindings not installed")
def test_picks_the_c_dumper_when_libyaml_is_present() -> None:
    assert yaml_loader.safe_dumper() is yaml.CSafeDumper


def test_dumper_falls_back_to_the_pure_python_dumper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(yaml, "CSafeDumper", raising=False)
    assert yaml_loader.safe_dumper() is yaml.SafeDumper
    assert yaml_loader.safe_dump(_EMITTED) == yaml.safe_dump(_EMITTED)


@pytest.mark.parametrize("kwargs", _DUMP_KWARGS, ids=lambda kw: f"width={kw.get('width')}")
def test_dumps_the_same_bytes_as_safe_dump(kwargs: dict) -> None:
    assert yaml_loader.safe_dump(_EMITTED, **kwargs) == yaml.safe_dump(_EMITTED, **kwargs)


def test_dumps_to_a_stream_like_safe_dump() -> None:
    written = io.StringIO()
    assert yaml_loader.safe_dump(_EMITTED, written, sort_keys=False) is None
    assert written.getvalue() == yaml.safe_dump(_EMITTED, sort_keys=False)


def test_refuses_python_objects_like_safe_dump() -> None:
    with pytest.raises(yaml.representer.RepresenterError):
        yaml_loader.safe_dump({"path": object()})
