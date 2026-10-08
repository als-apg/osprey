"""One device listed under several mml families resolves to one id, or the import stops.

The synthetic export gains a ``BPM`` family over three of ``BPMx``'s four
cells, bound to the same ``Monitor`` addresses ``BPMx`` is wired through.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.layers.mml.importer import LAYER_DIR, import_mml
from osprey.facility.layers.mml.mapping import MAPPING_FILE, ImportStop

SYNTHETIC = Path(__file__).resolve().parents[1] / "fixtures" / "mml" / "synthetic"
AO = "quokka.sr.ao.json"


def _shared(tmp_path: Path) -> tuple[Path, Path]:
    """The synthetic export with ``BPM`` added, and its ``data/facility`` beside it.

    Returns:
        The copied AO file and the facility directory holding the copied
        mapping, which gives ``BPM`` a family entry and a ``read`` ``X``.
    """
    export = tmp_path / "export"
    export.mkdir()
    for source in SYNTHETIC.glob("quokka.sr.*"):
        shutil.copyfile(source, export / source.name)
    ao = json.loads((export / AO).read_text(encoding="utf-8"))
    monitors = ao["BPMx"]["Monitor"]["ChannelNames"][:3]
    ao["BPM"] = {
        "FamilyName": "BPM",
        "MemberOf": ["BPM", "Diagnostics"],
        "DeviceList": [[1, 1], [2, 1], [3, 1]],
        "CommonNames": ["BPM(1,1)", "BPM(2,1)", "BPM(3,1)"],
        "X": {"MemberOf": ["BPM", "Monitor"], "DataType": "Scalar", "ChannelNames": monitors},
    }
    (export / AO).write_text(json.dumps(ao), encoding="utf-8")

    facility = tmp_path / "data" / "facility"
    target = facility / MAPPING_FILE
    target.parent.mkdir(parents=True)
    document = yaml.safe_load((SYNTHETIC / MAPPING_FILE).read_text(encoding="utf-8"))
    document["families"]["BPM"] = {
        "class": "BeamPositionMonitor",
        "aliases": ["BPM"],
        "description": "Beam position, both planes, over three cells.",
        "provenance": "stated",
        "channels": 3,
        "fields": {"X": {"description": "Horizontal beam position.", "provenance": "stated"}},
    }
    document["directions"]["BPM.X"] = {
        "direction": "read",
        "provenance": "stated",
        "override": False,
    }
    _write(facility, document)
    return export / AO, facility


def _mapping(facility: Path) -> dict[str, Any]:
    return yaml.safe_load((facility / MAPPING_FILE).read_text(encoding="utf-8"))


def _write(facility: Path, document: dict[str, Any]) -> None:
    (facility / MAPPING_FILE).write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )


def test_two_identities_for_one_wired_address_stop_the_import_before_anything_is_written(
    tmp_path: Path,
) -> None:
    ao, facility = _shared(tmp_path)
    with pytest.raises(ImportStop) as stop:
        import_mml([ao], facility)
    assert stop.value.format_message().splitlines() == [
        "import mml: mapping-invalid: families.BPMx.devices: QK:BPMx:1:CUR:RB is SR/qk_bpmx_1 "
        "in BPMx and SR/BPM_1_1 in BPM (and 2 more addresses); give both one identity: "
        "{coordinates: <stem>} on one family and {same_as: <it>} on the other"
    ]
    assert sorted(path.name for path in (facility / LAYER_DIR).iterdir()) == ["mapping.yaml"]
    assert sorted(path.name for path in facility.iterdir()) == ["imported"]


def test_a_shared_address_no_wiring_record_states_is_not_refused(tmp_path: Path) -> None:
    ao, facility = _shared(tmp_path)
    body = json.loads(ao.read_text(encoding="utf-8"))
    body["BPM"]["X"]["ChannelNames"] = ["QK:SEPTUM:1:CUR:RB", "QK:BPM:2:X", "QK:BPM:3:X"]
    ao.write_text(json.dumps(body), encoding="utf-8")
    import_mml([ao], facility)
    channels = {
        row["id"]: row
        for row in yaml.safe_load((facility / LAYER_DIR / "channels.yaml").read_text("utf-8"))
    }
    assert sorted(channels["QK:SEPTUM:1:CUR:RB"]["endpoint_of"]) == [
        "SR/BPM_1_1",
        "SR/qk_septum_1",
    ]
