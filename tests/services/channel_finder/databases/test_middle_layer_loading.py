"""Shape checks on loading a middle-layer database.

A middle-layer file is a dict of systems -> families -> fields. The parser
used to navigate any nesting it was given and read a wrong-shaped file as an
empty tree, which every consumer then reported as a facility with zero
channels. These pin that a wrong shape raises, and that metadata keys (the
``_``-prefixed ones) stay tolerated at every level.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.channel_finder.databases.middle_layer import MiddleLayerDatabase


def _write(tmp_path: Path, body: object) -> str:
    path = tmp_path / "ml.json"
    path.write_text(json.dumps(body))
    return str(path)


_FIELDS = {"Monitor": {"ChannelNames": ["SR01:BPM:X"], "DataType": "double"}}


def test_a_root_that_is_not_a_dict_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid database format.*root is a list"):
        MiddleLayerDatabase(_write(tmp_path, [{"SR": {"BPM": _FIELDS}}]))


def test_a_system_that_is_not_a_dict_raises_and_names_it(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="system 'SR'.*got a list"):
        MiddleLayerDatabase(_write(tmp_path, {"SR": ["BPM"]}))


def test_a_family_that_is_not_a_dict_raises_and_names_it(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="family 'SR:BPM'.*got a str"):
        MiddleLayerDatabase(_write(tmp_path, {"SR": {"BPM": "SR01:BPM:X"}}))


def test_the_flat_paradigms_list_of_channels_is_refused_rather_than_read_as_empty(
    tmp_path: Path,
) -> None:
    """The shape a half-migrated file most often has: a flat channel list."""
    body = {"channels": [{"address": "FACILITY:TIER:SRC", "description": "profile"}]}
    with pytest.raises(ValueError, match="system 'channels'.*got a list"):
        MiddleLayerDatabase(_write(tmp_path, body))


def test_metadata_keys_are_tolerated_at_every_level(tmp_path: Path) -> None:
    body = {
        "_meta": {"tier": 3, "generated": "today"},
        "_description": "a facility",
        "SR": {"_description": "Storage Ring", "BPM": {"_description": "BPMs", **_FIELDS}},
    }
    db = MiddleLayerDatabase(_write(tmp_path, body))
    assert set(db.channel_map) == {"SR01:BPM:X"}
