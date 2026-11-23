"""The facility file an ARIEL entry writer checks entries against, and the shipped demo."""

from __future__ import annotations

import logging

import pytest
import yaml

from osprey.facility.records import FacilityRecords
from osprey.facility.references import find_facility_references
from osprey.services.ariel_search.ingestion import facility_references
from osprey.simulation.apply import active_logbook_entries
from tests._facility_file import write_facility_document, write_facility_v4

_IDENTIFIER_EDGE = "_"


def _config_for(monkeypatch, render_dir) -> None:
    monkeypatch.setattr(
        "osprey_connectors.workspace.load_osprey_config",
        lambda: {"config_dir": str(render_dir)},
    )


def _records_at(caplog) -> FacilityRecords | None:
    with caplog.at_level(logging.INFO):
        return facility_references.load_facility_records()


def test_no_facility_file_means_no_facility_records(monkeypatch, tmp_path, caplog) -> None:
    _config_for(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "osprey.channel_roster.sources.facility_file_path", lambda _c: tmp_path / "facility.json"
    )

    assert _records_at(caplog) is None
    lines = [r for r in caplog.records if "facility file not built" in r.getMessage()]
    assert [r.levelno for r in lines] == [logging.INFO]


def test_a_broken_facility_file_means_no_facility_records_and_a_warning(
    monkeypatch, tmp_path, caplog
) -> None:
    broken = tmp_path / "facility.json"
    broken.write_text("{not json", encoding="utf-8")
    _config_for(monkeypatch, tmp_path)
    monkeypatch.setattr("osprey.channel_roster.sources.facility_file_path", lambda _c: broken)

    assert _records_at(caplog) is None
    lines = [r for r in caplog.records if str(broken) in r.getMessage()]
    assert [r.levelno for r in lines] == [logging.WARNING]


@pytest.mark.foreign_header
def test_a_v3_facility_file_means_no_facility_records_and_a_warning(
    monkeypatch, tmp_path, caplog
) -> None:
    path = write_facility_document(
        tmp_path, {"channels": [{"id": "SR:X"}]}, schema="osprey.facility.facility/3"
    )
    _config_for(monkeypatch, tmp_path)
    monkeypatch.setattr("osprey.channel_roster.sources.facility_file_path", lambda _c: path)

    assert _records_at(caplog) is None
    lines = [r for r in caplog.records if str(path) in r.getMessage()]
    assert [r.levelno for r in lines] == [logging.WARNING]
    assert "this reader reads osprey.facility.facility/4" in lines[0].getMessage()
    assert "rebuild with osprey build" in lines[0].getMessage()


def test_a_v4_facility_file_gives_its_records(monkeypatch, tmp_path, caplog) -> None:
    path = write_facility_v4(tmp_path, {"channels": [{"id": "SR:X"}]})
    _config_for(monkeypatch, tmp_path)
    monkeypatch.setattr("osprey.channel_roster.sources.facility_file_path", lambda _c: path)

    records = _records_at(caplog)

    assert records is not None
    assert records.record("channel", "SR:X") is not None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def _is_identifier(character: str) -> bool:
    return character.isalnum() or character == _IDENTIFIER_EDGE


def _expected_triples(text: str, targets: frozenset[str]) -> list[tuple[int, int, str]]:
    """Every target in ``text`` with identifier boundaries, longest first at a position."""
    starts: dict[int, str] = {}
    for target in sorted(targets, key=len):
        at = text.find(target)
        while at != -1:
            after = at + len(target)
            before_ok = at == 0 or not _is_identifier(text[at - 1])
            after_ok = after == len(text) or not _is_identifier(text[after])
            if before_ok and after_ok:
                starts[at] = target
            at = text.find(target, at + 1)
    triples: list[tuple[int, int, str]] = []
    position = 0
    for at in sorted(starts):
        if at >= position:
            triples.append((at, at + len(starts[at]), starts[at]))
            position = at + len(starts[at])
    return triples


def test_the_shipped_demo_logbook_quotes_become_references(built_demo) -> None:
    built = built_demo
    config = yaml.safe_load((built.build_dir / "config.yml").read_text(encoding="utf-8"))
    records = FacilityRecords(built.facility)
    entries = active_logbook_entries(config, built.build_dir)

    union: set[str] = set()
    for entry in entries:
        text = entry["raw_text"]
        found = find_facility_references(text, records)
        assert [(r.start, r.end, r.target) for r in found] == _expected_triples(
            text, records.targets()
        )
        assert all(text[r.start : r.end] == r.target for r in found)
        union.update(r.target for r in found)

    assert union
