"""The served scope, and the read model a build's renders share to ask for it.

Every render of one build asks for the scope of the models it serves over the
build's one document. :meth:`FacilityRecords.of` gives them one read model, and
:func:`served_scope` scans the channels once for each set of served models.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey_facility import TEXTURE
from osprey_facility.records import FacilityRecords
from osprey_facility.served import served_scope


def _document() -> dict[str, Any]:
    return {
        "devices": [
            {"id": "RING/Q1", "model": "ring"},
            {"id": "LINE/Q1", "model": "line"},
            {"id": "HALL/T1"},
        ],
        "channels": [
            {"id": "RING:Q1:SP", "on": {"device": "RING/Q1"}},
            {"id": "LINE:Q1:SP", "on": {"device": "LINE/Q1"}},
            {"id": "LINE:WIRED"},
            {"id": "HALL:T1", "on": {"device": "HALL/T1"}},
            {"id": "LOOSE"},
        ],
        "models": [
            {"name": "ring", "wiring": []},
            {"name": "line", "wiring": [{"address": "LINE:WIRED"}]},
            {"name": TEXTURE},
        ],
    }


def _ids(channels: list[Any]) -> list[str]:
    return [str(channel["id"]) for channel in channels]


def test_the_scope_is_each_served_models_channels_and_every_unowned_one() -> None:
    records = FacilityRecords(_document())

    assert _ids(served_scope(records, ["ring", TEXTURE])) == ["RING:Q1:SP", "HALL:T1", "LOOSE"]
    assert _ids(served_scope(records, ["line", TEXTURE])) == [
        "LINE:Q1:SP",
        "LINE:WIRED",
        "HALL:T1",
        "LOOSE",
    ]
    assert _ids(served_scope(records, ["line", "ring", TEXTURE])) == _ids(
        list(records.of_kind("channel"))
    )
    assert _ids(served_scope(records, [TEXTURE])) == _ids(list(records.of_kind("channel")))


def test_a_scope_asked_for_again_scans_no_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    records = FacilityRecords(_document())
    first = served_scope(records, ["ring", TEXTURE])
    other = served_scope(records, ["line", TEXTURE])

    def refuse(_records: FacilityRecords, selection: Any) -> Any:
        raise AssertionError(f"selected {selection} again")

    monkeypatch.setattr(FacilityRecords, "select_channels", refuse)
    again = served_scope(records, ["ring", TEXTURE])

    assert again == first
    assert again is not first
    assert served_scope(records, ["line", TEXTURE]) == other


def test_a_callers_change_to_a_scope_reaches_no_other_caller() -> None:
    records = FacilityRecords(_document())
    served_scope(records, ["ring", TEXTURE]).clear()

    assert _ids(served_scope(records, ["ring", TEXTURE])) == ["RING:Q1:SP", "HALL:T1", "LOOSE"]


def test_another_read_model_has_its_own_scopes() -> None:
    document = _document()
    served_scope(FacilityRecords(document), ["ring", TEXTURE])
    document["channels"].append({"id": "LATER"})

    assert "LATER" in _ids(served_scope(FacilityRecords(document), ["ring", TEXTURE]))


def test_one_document_is_indexed_once() -> None:
    document, other = _document(), _document()

    records = FacilityRecords.of(document)

    assert FacilityRecords.of(document) is records
    assert FacilityRecords.of(other) is not records
    assert FacilityRecords.of(document) is records
    assert FacilityRecords.of(document).document == document


def test_an_equal_document_that_is_another_object_is_indexed_itself() -> None:
    document = _document()
    records = FacilityRecords.of(document)
    changed = {**document, "channels": [{"id": "ONLY"}]}

    assert FacilityRecords.of(changed) is not records
    assert FacilityRecords.of(changed).ids("channel") == {"ONLY"}


def test_only_the_last_few_documents_stay_indexed() -> None:
    document = _document()
    records = FacilityRecords.of(document)
    for _ in range(8):
        FacilityRecords.of(_document())

    assert FacilityRecords.of(document) is not records
