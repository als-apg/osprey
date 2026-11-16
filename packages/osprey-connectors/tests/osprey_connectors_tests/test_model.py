"""The frozen collections of ``osprey_connectors.model``: what a frozen record's fields hold."""

from __future__ import annotations

import copy
import json
import pickle

import pytest
from pydantic import Field

from osprey_connectors.model import (
    FrozenModel,
    FrozenTuple,
    ReadOnlyMapping,
    empty_mapping,
    evolve,
)


class Record(FrozenModel):
    names: FrozenTuple[str] = ()
    counts: ReadOnlyMapping[str, int] = Field(default_factory=empty_mapping)
    nested: ReadOnlyMapping[str, ReadOnlyMapping[str, int] | None] | None = None


def test_a_list_and_a_dict_given_are_held_frozen():
    record = Record(names=["a", "b"], counts={"x": 1}, nested={"n": {"y": 2}, "m": None})

    assert record.names == ("a", "b")
    assert record.counts == {"x": 1}
    for mapping in (record.counts, record.nested, record.nested["n"]):
        with pytest.raises(TypeError):
            mapping["z"] = 3
        for change in ("update", "pop", "setdefault", "popitem", "clear"):
            with pytest.raises(TypeError):
                getattr(mapping, change)()
        with pytest.raises(TypeError):
            mapping |= {"z": 3}


def test_a_default_mapping_is_frozen_too():
    with pytest.raises(TypeError):
        Record().counts["z"] = 1


def test_a_read_only_mapping_is_still_a_dict_to_every_reader():
    record = Record(counts={"x": 1})

    assert isinstance(record.counts, dict)
    assert json.dumps(record.counts) == '{"x": 1}'
    assert record.counts | {"y": 2} == {"x": 1, "y": 2}


def test_the_json_form_is_plain_lists_and_dicts():
    record = Record(names=["a"], counts={"x": 1}, nested={"n": {"y": 2}})
    dumped = record.model_dump(mode="json")

    assert dumped == {"names": ["a"], "counts": {"x": 1}, "nested": {"n": {"y": 2}}}
    assert type(dumped["counts"]) is dict and type(dumped["nested"]["n"]) is dict
    assert Record.model_validate_json(record.model_dump_json()) == record


def test_a_copy_and_a_pickle_are_equal_and_still_frozen():
    record = Record(names=["a"], counts={"x": 1})

    for again in (copy.deepcopy(record), pickle.loads(pickle.dumps(record)), evolve(record)):
        assert again == record
        with pytest.raises(TypeError):
            again.counts["z"] = 1
