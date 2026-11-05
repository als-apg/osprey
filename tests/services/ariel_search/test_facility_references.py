"""The facility file an ARIEL entry writer checks entries against, and the shipped demo."""

from __future__ import annotations

import logging

import yaml

from osprey.facility.records import FacilityRecords
from osprey.facility.references import find_facility_references
from osprey.services.ariel_search.ingestion import facility_references
from osprey.simulation.apply import active_logbook_entries

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


def test_the_shipped_demo_logbook_quotes_become_references(built_control_assistant) -> None:
    built = built_control_assistant
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
